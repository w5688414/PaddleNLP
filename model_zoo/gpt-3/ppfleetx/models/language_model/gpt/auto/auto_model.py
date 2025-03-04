# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import math

import paddle
import paddle.distributed.auto_parallel as auto
import paddle.incubate as incubate
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.tensor as tensor
from paddle.common_ops_import import convert_dtype
from paddle.base import layers
from paddle.nn.layer.transformer import _convert_param_attr_to_list
from ppfleetx.distributed.apis import auto_env
from ppfleetx.utils.log import logger

from ..dygraph.processor import (
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
)

try:
    from paddle.incubate.nn.layer.fused_dropout_add import FusedDropoutAdd
except:
    FusedDropoutAdd = None
FusedDropoutAdd = None

try:
    from paddle.nn.functional.flash_attention import flash_attention
except:
    flash_attention = None

try:
    from paddle.jit.api import set_dynamic_shape
except:
    from paddle.jit.dy2static.utils_helper import set_dynamic_shape

def shard_op_for_sequence_parallel_linear(tgt, mesh):
    # FIXME Hack to shard op for module (linear)
    # we only shard the second to the last op (matmul) leave the last op (elementwise_add) un-touched
    last_op = tgt.block.ops[-2]
    assert last_op.type in ["matmul", "matmul_v2"]
    from paddle.distributed.auto_parallel.static.dist_context import get_default_distributed_context
    from paddle.distributed.auto_parallel.static.dist_op import DistributedOperator
    default_dist_ctx = get_default_distributed_context()
    original_id = last_op.desc.original_id()
    assert len(last_op.output_arg_names) == 1, "Output is more than one: [{}].".format(str(last_op))
    assert original_id not in default_dist_ctx._dist_ops_for_program, "Op already has dist attribute."
    
    output_var_name = last_op.output_arg_names[0]
    assert output_var_name != tgt.name, "out name: {}, output_var_name: {}".format(output_var_name, tgt.name)
    dist_op = DistributedOperator(last_op)
    output_tensor = dist_op.get_serial_output(output_var_name)
    tensor_dist_attr = dist_op.dist_attr.get_output_dist_attr(output_var_name)
    dims_mapping = [-1] * len(output_tensor.shape)
    # NOTE explicitlly set the dims_mapping for DP since we could not distinguish with "Any" and "Replicated"
    if auto_env.get_mesh().dp_dim is not None:
        dims_mapping[1] = auto.static.utils.convert_to_dims_mapping([auto_env.get_mesh().dp_dim], mesh)[0]
    tensor_dist_attr.dims_mapping = dims_mapping
    tensor_dist_attr.mark_annotated("dims_mapping")


def get_attr(layer, name):
    if getattr(layer, name, None) is not None:
        return getattr(layer, name, None)
    else:
        return get_attr(layer._layer, name)


class MultiHeadAttention(nn.Layer):
    """
    Attention mapps queries and a set of key-value pairs to outputs, and
    Multi-Head Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces.

    """

    Cache = collections.namedtuple("Cache", ["k", "v"])
    StaticCache = collections.namedtuple("StaticCache", ["k", "v"])

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        kdim=None,
        vdim=None,
        need_weights=False,
        weight_attr=None,
        bias_attr=None,
        output_layer_weight_attr=None,
        fuse_attn_qkv=False,
        scale_qk_coeff=1.0,
        use_recompute=False,
        recompute_granularity="full",
        use_flash_attn=False,
        ipp=None,
        sequence_parallel=False,
    ):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights
        self.fuse_attn_qkv = fuse_attn_qkv
        self.scale_qk_coeff = scale_qk_coeff
        self.use_recompute = use_recompute
        self.recompute_granularity = recompute_granularity
        self.ipp = ipp
        self.use_flash_attn = use_flash_attn if flash_attention else None
        self.sequence_parallel = sequence_parallel


        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim[{}] must be divisible by num_heads[{}]".format(self.embed_dim, num_heads)

        if self.fuse_attn_qkv:
            assert self.kdim == embed_dim
            assert self.vdim == embed_dim
            self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, weight_attr, bias_attr=bias_attr)
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)
            self.k_proj = nn.Linear(self.kdim, embed_dim, weight_attr, bias_attr=bias_attr)
            self.v_proj = nn.Linear(self.vdim, embed_dim, weight_attr, bias_attr=bias_attr)

        self.out_proj = nn.Linear(embed_dim, embed_dim, output_layer_weight_attr, bias_attr=bias_attr)

    def _fuse_prepare_qkv(self, query, use_cache=False, cache=None):
        auto.shard_tensor(self.qkv_proj.weight, auto_env.get_mesh()[self.ipp], [None, auto_env.get_mesh().mp_dim])

        mix_layer = self.qkv_proj(query)
        mix_layer = paddle.reshape_(mix_layer, [0, 0, -1, 3 * self.head_dim])
        q, k, v = paddle.split(mix_layer, num_or_sections=3, axis=-1)

        assert not isinstance(cache, self.StaticCache), "cache currently does not support the StaticCache type"

        if isinstance(cache, self.Cache):
            # for decoder self-attention in inference
            k = tensor.concat([cache.k, k], axis=1)
            v = tensor.concat([cache.v, v], axis=1)
        if use_cache is True:
            cache = self.Cache(k, v)

        return (q, k, v, cache) if use_cache else (q, k, v, None)

    def _prepare_qkv(self, query, key, value, use_cache=False, cache=None):
        r"""
        Prapares linear projected queries, keys and values for usage of subsequnt
        multiple parallel attention. If `cache` is not None, using cached results
        to reduce redundant calculations.

        """
        auto.shard_tensor(self.q_proj.weight, auto_env.get_mesh()[self.ipp], [None, auto_env.get_mesh().mp_dim])

        q = self.q_proj(query)
        q = tensor.reshape(x=q, shape=[0, 0, -1, self.head_dim])

        if isinstance(cache, self.StaticCache):
            # for encoder-decoder attention in inference and has cached
            k, v = cache.k, cache.v
        else:
            k, v = self.compute_kv(key, value)

        if isinstance(cache, self.Cache):
            # for decoder self-attention in inference
            k = tensor.concat([cache.k, k], axis=1)
            v = tensor.concat([cache.v, v], axis=1)
        if use_cache is True:
            cache = self.Cache(k, v)

        return (q, k, v, cache) if use_cache else (q, k, v, None)

    def compute_kv(self, key, value):
        r"""
        Applies linear projection on input keys and values, then splits heads
        (reshape and transpose) to get keys and values from different representation
        subspaces. The results are used as key-values pairs for subsequent multiple
        parallel attention.

        It is part of calculations in multi-head attention, and is provided as
        a method to pre-compute and prefetch these results, thus we can use them
        to construct cache for inference.

        """
        auto.shard_tensor(self.k_proj.weight, auto_env.get_mesh()[self.ipp], [None, auto_env.get_mesh().mp_dim])
        auto.shard_tensor(self.v_proj.weight, auto_env.get_mesh()[self.ipp], [None, auto_env.get_mesh().mp_dim])

        k = self.k_proj(key)
        v = self.v_proj(value)
        k = tensor.reshape(x=k, shape=[0, 0, -1, self.head_dim])
        v = tensor.reshape(x=v, shape=[0, 0, -1, self.head_dim])
        return k, v

    def gen_cache(self, key, value=None, type=Cache):
        """
        Generates cache for `forward` usage in inference accroding to arguments.
        The generated cache is an instance of `MultiHeadAttention.Cache` or an
        instance of `MultiHeadAttention.StaticCache`.
        """
        if type == MultiHeadAttention.StaticCache:  # static_kv
            k, v = self.compute_kv(key, value)
            return self.StaticCache(k, v)
        elif value is None:  # incremental_state
            k = layers.fill_constant_batch_size_like(
                input=key, shape=[-1, self.num_heads, 0, self.head_dim], dtype=key.dtype, value=0
            )
            v = layers.fill_constant_batch_size_like(
                input=key, shape=[-1, self.num_heads, 0, self.head_dim], dtype=key.dtype, value=0
            )
            return self.Cache(k, v)
        else:
            # incremental_state with initial value, mainly for usage like UniLM
            return self.Cache(key, value)

    def _flash_attention(self, q, k, v, attn_mask=None):

        # if sequence_parallel, the activation is [s, b, h], but flash attention lib need [b, s, h]
        # we revert it to be [b, s, h] before 
        if self.sequence_parallel:
            perm = [1, 0, 2, 3]
            q = tensor.transpose(x=q, perm=perm)
            k = tensor.transpose(x=k, perm=perm)
            v = tensor.transpose(x=v, perm=perm)

        out, weights = flash_attention(
            q, k, v, self.dropout, causal=True, return_softmax=self.need_weights, training=self.training
        )
        out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # and convert it back to [s, b, h] after
        if self.sequence_parallel:
            perm = [1, 0, 2]
            out = tensor.transpose(x=out, perm=perm)

        return (out, weights)

    def core_attn(self, q, k, v, attn_mask=None):
        # input shape is [b, s, nhead, ndim] or [s, b, nhead, ndim] if sequence_parallel
        # after transpose, [b, nhead, s, ndim] not matter sequence_parallel or not 
        perm = [1, 2, 0, 3] if self.sequence_parallel else [0, 2, 1, 3]
        q = tensor.transpose(x=q, perm=perm)
        k = tensor.transpose(x=k, perm=perm)
        v = tensor.transpose(x=v, perm=perm)

        # scale dot product attention
        scale_qk_coeff = self.scale_qk_coeff * self.head_dim**0.5
        product = paddle.matmul(x=q.scale(1.0 / scale_qk_coeff), y=k, transpose_y=True)

        if self.scale_qk_coeff != 1.0:
            product = product.scale(self.scale_qk_coeff)

        if attn_mask is not None:
            product = product + attn_mask
            weights = F.softmax(product)
        else:
            weights = incubate.softmax_mask_fuse_upper_triangle(product)

        if self.dropout:
            weights = F.dropout(weights, self.dropout, training=self.training, mode="upscale_in_train")

        out = paddle.matmul(weights, v)

        # combine heads
        # before transpose, out is [b, nhead, s, ndim]
        # after transpose, out need to be [b, s, nhead, ndim](not sp)  or  [s, b, nhead, ndim](not sp)
        if self.sequence_parallel:
            out = tensor.transpose(out, perm=[2, 0, 1, 3])
        else:
            out = tensor.transpose(out, perm=[0, 2, 1, 3])

        out = tensor.reshape(x=out, shape=[0, 0, -1])

        return out, weights

    def forward(self, query, key, value, attn_mask=None, use_cache=False, cache=None):
        r"""
        Applies multi-head attention to map queries and a set of key-value pairs
        to outputs.
        """
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        if self.fuse_attn_qkv:
            q, k, v, cache = self._fuse_prepare_qkv(query, use_cache, cache)
        else:
            q, k, v, cache = self._prepare_qkv(query, key, value, use_cache, cache)

        if self.use_flash_attn and attn_mask is None:
            attn_func = self._flash_attention
        else:
            attn_func = self.core_attn

        if self.use_recompute and self.recompute_granularity == "core_attn":
            out, weights = auto.recompute(attn_func)(q, k, v, attn_mask)
        else:
            out, weights = attn_func(q, k, v, attn_mask=attn_mask)

        auto.shard_tensor(self.out_proj.weight, auto_env.get_mesh()[self.ipp], [auto_env.get_mesh().mp_dim, None])

        # project to output
        out = self.out_proj(out)

        if self.sequence_parallel:
            # FIXME Hack to shard op for module (linear)
            shard_op_for_sequence_parallel_linear(out, auto_env.get_mesh()[self.ipp])    

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        if use_cache:
            outs.append(cache)
        return out if len(outs) == 1 else tuple(outs)


class TransformerDecoder(nn.Layer):
    """
    TransformerDecoder is a stack of N decoder layers.
    """

    def __init__(
        self,
        decoder_layers,
        num_layers,
        norm=None,
        hidden_size=None,
        use_recompute=False,
        recompute_granularity="full",
        sequence_parallel=False,
    ):
        super(TransformerDecoder, self).__init__()

        self.num_layers = num_layers
        self.layers = decoder_layers
        self.norm = norm
        self.use_recompute = use_recompute
        self.recompute_granularity = recompute_granularity
        if norm == "LayerNorm":
            self.norm = nn.LayerNorm(hidden_size, epsilon=1e-5)
        elif norm is not None:
            raise ValueError("Only support LayerNorm")
        self.sequence_parallel = sequence_parallel

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, use_cache=False, cache=None):
        r"""
        Applies a stack of N Transformer decoder layers on inputs. If `norm` is
        provided, also applies layer normalization on the output of last decoder
        layer.
        """
        output = tgt
        new_caches = []

        for i, mod in enumerate(self.layers):
            ipp = mod.ipp
            # TODO(zhaoyingli) Annotation of SP and DP are conflict here!
            # DP-PP required the input of Transformer Layer to be annotated as :
            # auto.shard_tensor(output, auto_env.get_mesh()[ipp], [auto_env.get_mesh().dp_dim, None, None])
            # But the above annotation will cutoff the propagation of SP sharding when SP is used along.
            # A better solution is to allow the DP sharding propagate across PP Mesh.  
            if not self.sequence_parallel: 
                auto.shard_tensor(output, auto_env.get_mesh()[ipp], [auto_env.get_mesh().dp_dim, None, None])
            elif auto_env.get_mesh().dp_dim is not None:
                auto.shard_tensor(output, auto_env.get_mesh()[ipp], [auto_env.get_mesh().sp_dim, auto_env.get_mesh().dp_dim, None])
            else:
                pass

            if cache is None:
                if use_cache:
                    output, new_cache = mod(output, memory, tgt_mask=tgt_mask, use_cache=use_cache, cache=cache)
                    new_caches.append(new_cache)
                else:
                    if self.use_recompute and self.recompute_granularity == "full":
                        output = auto.recompute(mod)(output, memory, tgt_mask, use_cache, cache)
                    else:
                        output = mod(output, memory, tgt_mask, use_cache, cache)
            else:
                output, new_cache = mod(output, memory, tgt_mask=tgt_mask, use_cache=use_cache, cache=cache[i])
                new_caches.append(new_cache)

        if self.norm is not None:
            output = self.norm(output)
        return output if use_cache is False else (output, new_caches)

    def gen_cache(self, memory, do_zip=False):
        r"""
        Generates cache for `forward` usage. The generated cache is a list, and
        each element in it is a tuple( :code:`(incremental_cache, static_cache)` )
        produced by `TransformerDecoderLayer.gen_cache`. See `TransformerDecoderLayer.gen_cache`
        for more details. If `do_zip` is True, apply `zip` on these tuples to get
        a list with two elements.
        """
        cache = [layer.gen_cache(memory) for layer in self.layers]
        if do_zip:
            cache = list(zip(*cache))
        return cache


class TransformerDecoderLayer(nn.Layer):
    """
    The transformer decoder layer.

    It contains multiheadattention and some linear layers.
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout=0.1,
        activation="gelu",
        attn_dropout=None,
        act_dropout=None,
        normalize_before=True,
        topk=1,
        weight_attr=None,
        bias_attr=None,
        output_layer_weight_attr=None,
        fuse_attn_qkv=False,
        scale_qk_coeff=1.0,
        use_recompute=False,
        recompute_granularity="full",
        use_flash_attn=False,
        use_fused_dropout_add=True,
        ipp=None,
        sequence_parallel=False,
    ):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(TransformerDecoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before
        self.use_recompute = use_recompute
        self.recompute_granularity = recompute_granularity
        self.ipp = ipp
        self.sequence_parallel = sequence_parallel

        if not FusedDropoutAdd:
            self.use_fused_dropout_add = False
        else:
            self.use_fused_dropout_add = use_fused_dropout_add

        weight_attrs = _convert_param_attr_to_list(weight_attr, 3)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 3)
        output_layer_weight_attrs = _convert_param_attr_to_list(output_layer_weight_attr, 3)

        self.self_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0],
            output_layer_weight_attr=output_layer_weight_attrs[0],
            fuse_attn_qkv=fuse_attn_qkv,
            scale_qk_coeff=scale_qk_coeff,
            use_recompute=use_recompute,
            recompute_granularity=recompute_granularity,
            use_flash_attn=use_flash_attn,
            ipp=ipp,
            sequence_parallel=sequence_parallel,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward, weight_attrs[2], bias_attr=bias_attrs[2])
        self.linear2 = nn.Linear(dim_feedforward, d_model, output_layer_weight_attrs[2], bias_attr=bias_attrs[2])

        self.norm1 = nn.LayerNorm(d_model, epsilon=1e-5)
        self.norm2 = nn.LayerNorm(d_model, epsilon=1e-5)
        if not self.use_fused_dropout_add:
            self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
            self.dropout2 = nn.Dropout(act_dropout, mode="upscale_in_train")
        else:
            self.fused_dropout_add1 = FusedDropoutAdd(dropout, mode="upscale_in_train")
            self.fused_dropout_add2 = FusedDropoutAdd(act_dropout, mode="upscale_in_train")

        if activation == "gelu":
            self.activation = nn.GELU(approximate=True)
        else:
            self.activation = getattr(F, activation)

    def forward(self, tgt, memory, tgt_mask=None, use_cache=False, cache=None):

        auto.shard_tensor(self.linear1.weight, auto_env.get_mesh()[self.ipp], [None, auto_env.get_mesh().mp_dim])
        auto.shard_tensor(self.linear2.weight, auto_env.get_mesh()[self.ipp], [auto_env.get_mesh().mp_dim, None])

        residual = tgt

        if self.normalize_before:
            tgt = self.norm1(tgt)

        if self.sequence_parallel:
            # NOTE since we don't support a semantic to distinguish with "Any" and "Replicated" in annotation 
            # we need to annotate the "DP" expicitly here.
            auto.shard_tensor(tgt, auto_env.get_mesh()[self.ipp], [None, auto_env.get_mesh().dp_dim, None])

        if use_cache is False:
            if self.use_recompute and self.recompute_granularity == "full_attn":
                tgt = auto.recompute(self.self_attn)(tgt, None, None, tgt_mask, use_cache, cache)
            else:
                tgt = self.self_attn(tgt, tgt, tgt, tgt_mask, use_cache, cache)
        else:
            tgt, incremental_cache = self.self_attn(tgt, tgt, tgt, tgt_mask, use_cache, cache)

        if self.sequence_parallel:
            # TODO(JZ-LIANG) make sure unsharded annotation would not be changed
            auto.shard_tensor(tgt, auto_env.get_mesh()[self.ipp], [auto_env.get_mesh().sp_dim, auto_env.get_mesh().dp_dim, None])

        if not self.use_fused_dropout_add:
            tgt = residual + self.dropout1(tgt)
        else:
            tgt = self.fused_dropout_add1(tgt, residual)

        if not self.normalize_before:
            tgt = self.norm1(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)

        if self.sequence_parallel:
            # Enter TP Region
            auto.shard_tensor(tgt, auto_env.get_mesh()[self.ipp], [None, auto_env.get_mesh().dp_dim, None])
            tgt = self.linear2(self.activation(self.linear1(tgt)))
            # NOTE shard_op to cut off the SP sharding propagation backward.
            shard_op_for_sequence_parallel_linear(tgt, auto_env.get_mesh()[self.ipp])

            # Enter SP Region
            auto.shard_tensor(tgt, auto_env.get_mesh()[self.ipp], [auto_env.get_mesh().sp_dim, auto_env.get_mesh().dp_dim, None])
            if not self.use_fused_dropout_add:
                tgt = self.dropout2(tgt)
                tgt = residual + tgt
            else:
                tgt = self.fused_dropout_add2(tgt, residual)
        else:
            # Mixed SP and TP Region
            if not self.use_fused_dropout_add:
                tgt = self.dropout2(self.linear2(self.activation(self.linear1(tgt))))
                tgt = residual + tgt
            else:
                tgt = self.fused_dropout_add2(self.linear2(self.activation(self.linear1(tgt))), residual)

        if not self.normalize_before:
            tgt = self.norm2(tgt)

        return tgt if use_cache is False else (tgt, incremental_cache)

    def gen_cache(self, memory):
        incremental_cache = self.self_attn.gen_cache(memory, type=self.self_attn.Cache)
        return incremental_cache


class GPTEmbeddings(nn.Layer):
    """
    Include embeddings from word and position embeddings.
    """

    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        hidden_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        initializer_range=0.02,
        freeze_embedding=False,
        sequence_parallel=False,
    ):
        super(GPTEmbeddings, self).__init__()
        self.sequence_parallel = sequence_parallel
        self.word_embeddings = nn.Embedding(
            vocab_size,
            hidden_size,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=initializer_range)),
        )

        self.position_embeddings = nn.Embedding(
            max_position_embeddings,
            hidden_size,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=initializer_range)),
        )

        if freeze_embedding:
            self.word_embeddings.weight.learning_rate = 0.0
            self.position_embeddings.weight.learning_rate = 0.0

        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, position_ids=None):
        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)
            position_ids = seq_length - ones

        auto.shard_tensor(self.word_embeddings.weight, auto_env.get_mesh()[0], [auto_env.get_mesh().mp_dim, None])

        input_embedings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = input_embedings + position_embeddings

        # [b, s, h] -> [s, b, h] 
        if self.sequence_parallel:

            auto.shard_tensor(embeddings, auto_env.get_mesh()[0], [auto_env.get_mesh().dp_dim, None, None]) # annotation to prevent unsharded propogation backward
            embeddings = paddle.transpose(embeddings, perm=[1, 0, 2])
            # TODO (JZ-LIANG) only constrain the sharding of seq axis propagate forward from here but not backward.
            auto.shard_tensor(embeddings, auto_env.get_mesh()[0], [auto_env.get_mesh().sp_dim, auto_env.get_mesh().dp_dim, None])

        embeddings = self.dropout(embeddings)
        return embeddings


class GPTModelAuto(nn.Layer):
    def __init__(
        self,
        vocab_size=51200,
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        ffn_hidden_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        use_recompute=False,
        initializer_range=0.02,
        topk=1,
        fuse_attn_qkv=False,
        scale_qk_by_layer_num=True,
        recompute_granularity="full",
        freeze_embedding=False,
        use_flash_attn=False,
        fused_softmax_with_triangular=False,
        use_fused_dropout_add=True,
        sequence_parallel=False,
    ):

        super(GPTModelAuto, self).__init__()

        self.initializer_range = initializer_range
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.fused_softmax_with_triangular = fused_softmax_with_triangular
        self.sequence_parallel = sequence_parallel

        if not auto_env.get_mesh():
            raise RuntimeError(
                "Please call auto_env.init_dist_env(config). AutoPrallel modeling need `mesh` to annotate distributed attribute."
            )

        if use_flash_attn:
            if flash_attention:
                logger.info("Flash-attention enabled.")
            else:
                use_flash_attn = False
                logger.warning("Flash-attention is not support in this Paddle version.")

        self.embeddings = GPTEmbeddings(
            vocab_size,
            hidden_size,
            hidden_dropout_prob,
            max_position_embeddings,
            type_vocab_size,
            self.initializer_range,
            freeze_embedding,
            sequence_parallel=sequence_parallel,
        )

        layer_per_stage = num_layers // auto_env.get_mesh().pp_degree
        layer_to_pipe = [i // layer_per_stage for i in range(num_layers)]
        decoder_layers = nn.LayerList()
        for i in range(num_layers):
            decoder_layers.append(
                TransformerDecoderLayer(
                    d_model=hidden_size,
                    nhead=num_attention_heads,
                    dim_feedforward=ffn_hidden_size,
                    dropout=hidden_dropout_prob,
                    activation="gelu",
                    attn_dropout=attention_probs_dropout_prob,
                    act_dropout=hidden_dropout_prob,
                    weight_attr=paddle.ParamAttr(
                        initializer=nn.initializer.Normal(mean=0.0, std=self.initializer_range)
                    ),
                    output_layer_weight_attr=paddle.ParamAttr(
                        initializer=nn.initializer.Normal(
                            mean=0.0, std=self.initializer_range / math.sqrt(2.0 * num_layers)
                        )
                    ),
                    bias_attr=None,
                    fuse_attn_qkv=fuse_attn_qkv,
                    scale_qk_coeff=num_layers if scale_qk_by_layer_num else 1.0,
                    use_recompute=use_recompute,
                    recompute_granularity=recompute_granularity,
                    use_fused_dropout_add=use_fused_dropout_add,
                    use_flash_attn=use_flash_attn,
                    ipp=layer_to_pipe[i],
                    sequence_parallel=sequence_parallel,
                )
            )

        self.decoder = TransformerDecoder(
            decoder_layers,
            num_layers,
            norm="LayerNorm",
            hidden_size=hidden_size,
            use_recompute=use_recompute,
            recompute_granularity=recompute_granularity,
            sequence_parallel=sequence_parallel,
        )

    def forward(self, input_ids, position_ids=None, attention_mask=None, use_cache=False, cache=None):

        if position_ids is None:
            past_length = 0
            if cache is not None:
                past_length = attention_mask.shape[-1] - 1
            position_ids = paddle.arange(past_length, input_ids.shape[-1] + past_length, dtype=input_ids.dtype)
            position_ids = position_ids.unsqueeze(0)
            position_ids = paddle.expand_as(position_ids, input_ids)

        input_ids.stop_gradient = True
        position_ids.stop_gradient = True
        auto.shard_tensor(
            input_ids, auto_env.get_mesh()[0], [auto_env.get_mesh().dp_dim] + [None] * (len(input_ids.shape) - 1)
        )

        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # fused_softmax_with_triangular is only suppported on GPU/DCU.
        # If on non-GPU devices, we use user defined mask and non-fused softmax.
        if not self.fused_softmax_with_triangular or not paddle.is_compiled_with_cuda():
            # TODO, use registered buffer
            causal_mask = paddle.tensor.triu(
                paddle.ones((input_ids.shape[-1], input_ids.shape[-1])) * -1e4, diagonal=1
            )
            if attention_mask is not None:
                if len(attention_mask.shape) == 2:
                    attention_mask = attention_mask[:, None, None, :]
                attention_mask = attention_mask + causal_mask
            else:
                attention_mask = causal_mask
            # The tensor returned by triu not in static graph.
            attention_mask.stop_gradient = True

        encoder_outputs = self.decoder(
            embedding_output,
            memory=None,
            tgt_mask=None
            if (self.fused_softmax_with_triangular and self.training and paddle.is_compiled_with_cuda())
            else attention_mask,  # use softmax_mask_fuse_upper_triangle
            use_cache=use_cache,
            cache=cache,
        )
        
        return encoder_outputs


class GPTForPretrainingAuto(nn.Layer):
    """
    GPT Model with pretraining tasks on top.

    Args:
        gpt (:class:`GPTModel`):
            An instance of :class:`GPTModel`.

    """

    def __init__(self, gpt):
        super(GPTForPretrainingAuto, self).__init__()
        self.gpt = gpt
        self.sequence_parallel = gpt.sequence_parallel

    def forward(
        self, input_ids, position_ids=None, attention_mask=None, masked_positions=None, use_cache=False, cache=None
    ):

        outputs = self.gpt(
            input_ids, position_ids=position_ids, attention_mask=attention_mask, use_cache=use_cache, cache=cache
        )
        if use_cache:
            encoder_outputs, cached_kvs = outputs[:2]
        else:
            encoder_outputs = outputs

        # FIXME should we force the encoder_outputs mesh is the last stage ? 
        if self.sequence_parallel:
            # NOTE Since Two Hard Rule Hack in Completion(line 464) and Reshard(line 1391) by @Aoyulong, the shard_tensor of op output 
            # might be invalid in some cases. So we need to instead using shard_op for the input of next Op. 
            # TODO Fix the above two hack in future.
            # auto.shard_tensor(encoder_outputs, auto_env.get_mesh()[-1], [None, auto_env.get_mesh().dp_dim, None])
            x_dims_mapping = [None, auto_env.get_mesh().dp_dim] + [None] * (len(encoder_outputs.shape) - 2)
        else:
            x_dims_mapping = [auto_env.get_mesh().dp_dim] + [None] * (len(encoder_outputs.shape) - 1)
        w_dims_mapping = [auto_env.get_mesh().mp_dim, None]
        matmul = auto.shard_op(paddle.matmul, auto_env.get_mesh()[-1], [x_dims_mapping, w_dims_mapping, None])
        logits = matmul(encoder_outputs, get_attr(self.gpt.embeddings.word_embeddings, "weight"), transpose_y=True)

        if use_cache:
            return logits, cached_kvs
        else:
            return logits


class GPTPretrainingCriterionAuto(nn.Layer):
    """
    Criterion for GPT. It calculates the final loss.
    """

    def __init__(self, sequence_parallel=False):
        super(GPTPretrainingCriterionAuto, self).__init__()
        self.loss_func = paddle.nn.CrossEntropyLoss(reduction="none")
        self.sequence_parallel = sequence_parallel

    def forward(self, prediction_scores, masked_lm_labels, loss_mask):
        """
        Args:
            prediction_scores(Tensor):
                The logits of masked token prediction. Its data type should be float32 and
                its shape is [batch_size, sequence_length, vocab_size].
            masked_lm_labels(Tensor):
                The labels of the masked language modeling, the dimensionality of `masked_lm_labels`
                is equal to `prediction_scores`. Its data type should be int64 and
                its shape is [batch_size, sequence_length, 1].
            loss_mask(Tensor):
                Mask used for calculating the loss of the masked language modeling to avoid
                calculating some unwanted tokens.
                Its data type should be float32 and its shape is [batch_size, sequence_length, 1].

        Returns:
            Tensor: The pretraining loss. Its data type should be float32 and its shape is [1].

        """
        masked_lm_labels.stop_gradient = True
        loss_mask.stop_gradient = True
        auto.shard_tensor(
            loss_mask, auto_env.get_mesh()[-1], [auto_env.get_mesh().dp_dim] + [None] * (len(loss_mask.shape) - 1)
        )
        if self.sequence_parallel:
            # [s, b, h] --> [b, s, h]
            prediction_scores = prediction_scores.transpose([1, 0, 2])

        masked_lm_loss = self.loss_func(prediction_scores, masked_lm_labels.unsqueeze(2))

        loss_mask = loss_mask.reshape([-1])
        masked_lm_loss = paddle.sum(masked_lm_loss.reshape([-1]) * loss_mask)

        loss = masked_lm_loss / loss_mask.sum()
        return loss


class GPTForGenerationAuto(nn.Layer):
    """
    GPT Model with pretraining tasks on top.

    Args:
        gpt (:class:`GPTModel`):
            An instance of :class:`GPTModel`.

    """

    def __init__(self, gpt, configs):
        super(GPTForGenerationAuto, self).__init__()
        self.gpt = gpt
        self.configs = configs

        self.max_length = self.configs.get("max_dec_len", 20)
        self.min_length = self.configs.get("min_dec_len", 0)
        self.decode_strategy = self.configs.get("decode_strategy", "sampling")
        self.early_finish = self.configs.get("early_finish", True)
        self.temperature = self.configs.get("temperature", 1.0)
        self.top_k = self.configs.get("top_k", 0)
        self.top_p = self.configs.get("top_p", 1.0)
        self.use_topp_sampling = self.configs.get("use_topp_sampling", False)
        self.inference = self.configs.get("inference", False)
        self.repetition_penalty = self.configs.get("repetition_penalty", 1.0)
        self.num_beams = self.configs.get("num_beams", 1)
        self.num_beam_groups = self.configs.get("num_beam_groups", 1)
        self.length_penalty = self.configs.get("length_penalty", 0.0)
        self.early_stopping = self.configs.get("early_stopping", False)
        self.bos_token_id = self.configs.get("bos_token_id", None)
        self.eos_token_id = self.configs.get("eos_token_id", None)
        self.pad_token_id = self.configs.get("pad_token_id", None)
        self.decoder_start_token_id = self.configs.get("decoder_start_token_id", None)
        self.forced_bos_token_id = self.configs.get("forced_bos_token_id", None)
        self.forced_eos_token_id = self.configs.get("forced_eos_token_id", None)
        self.num_return_sequences = self.configs.get("num_return_sequences", 1)
        self.diversity_rate = self.configs.get("diversity_rate", 0.0)
        self.use_cache = self.configs.get("use_cache", True)

    def prepare_input_ids_for_generation(self, bos_token_id, encoder_output=None):
        batch_size = 1
        if bos_token_id is None:
            raise ValueError("`bos_token_id` should be defined when no " "`input_ids` are provided.")
        if encoder_output is not None:
            batch_size = encoder_output.shape[0]
        return paddle.ones([batch_size, 1], dtype="int64") * bos_token_id

    def prepare_attention_mask_for_generation(self, input_ids, pad_token_id, eos_token_id):
        is_pad_token_in_inputs_ids = (pad_token_id is not None) and paddle.any(
            input_ids == pad_token_id
        ).item()
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id)
        )
        if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
            attention_mask = (input_ids == pad_token_id).astype(paddle.get_default_dtype()) * -1e9
        else:
            attention_mask = paddle.zeros_like(input_ids, dtype=paddle.get_default_dtype())
        return paddle.unsqueeze(attention_mask, axis=[1, 2])

    def update_scores_for_generation(self, scores, next_scores, length, unfinished_flag):
        # update scores

        unfinished_scores = (scores * length + next_scores) / (length + 1)
        scores = paddle.where(unfinished_flag, unfinished_scores, scores)
        return scores

    def get_logits_processor(
        self,
        min_length=None,
        max_length=None,
        eos_token_id=None,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        num_beams=1,
        num_beam_groups=1,
        diversity_rate=0.0,
        repetition_penalty=None,
    ):
        processors = LogitsProcessorList()

        if min_length is not None and eos_token_id is not None and min_length > -1:
            processors.append(MinLengthLogitsProcessor(min_length, eos_token_id))
        if num_beam_groups > 1 and diversity_rate > 0.0:
            processors.append(
                HammingDiversityLogitsProcessor(
                    diversity_rate=diversity_rate, num_beams=num_beams, num_beam_groups=num_beam_groups
                )
            )
        if repetition_penalty is not None and repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
        if forced_bos_token_id is not None:
            processors.append(ForcedBOSTokenLogitsProcessor(forced_bos_token_id))
        if forced_eos_token_id is not None:
            processors.append(ForcedEOSTokenLogitsProcessor(max_length, forced_eos_token_id))
        # TODO
        # Add more pre_processing for distribution

        return processors

    def expand_inputs_for_generation(self, input_ids, expand_size, attention_mask=None, **model_kwargs):

        index = paddle.tile(paddle.arange(input_ids.shape[0]).unsqueeze(-1), [1, expand_size]).reshape([-1])

        input_ids = paddle.gather(input_ids, index)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = paddle.gather(attention_mask, index)

        if "token_type_ids" in model_kwargs and model_kwargs["token_type_ids"] is not None:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = paddle.gather(token_type_ids, index)

        if "position_ids" in model_kwargs and model_kwargs["position_ids"] is not None:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = paddle.gather(position_ids, index)

        if "seq_len" in model_kwargs and model_kwargs["seq_len"] is not None:
            seq_len = model_kwargs["seq_len"]
            model_kwargs["seq_len"] = paddle.gather(seq_len, index)

        if "encoder_output" in model_kwargs and model_kwargs["encoder_output"] is not None:
            encoder_output = model_kwargs["encoder_output"]
            model_kwargs["encoder_output"] = paddle.gather(encoder_output, index)

        if "role_ids" in model_kwargs and model_kwargs["role_ids"] is not None:
            role_ids = model_kwargs["role_ids"]
            model_kwargs["role_ids"] = paddle.gather(role_ids, index)

        return input_ids, model_kwargs

    def prepare_inputs_for_generation(self, input_ids, use_cache=False, cache=None, **kwargs):
        # only last token for inputs_ids if cache is defined in kwargs
        position_ids = kwargs.get("position_ids", None)
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            if len(attention_mask.shape) == 4:
                attention_mask = attention_mask[:, -1, -1, :]
            if "int" in paddle.common_ops_import.convert_dtype(attention_mask.dtype):
                attention_mask = (1.0 - attention_mask) * -1e4
        return {"input_ids": input_ids, "position_ids": position_ids, "attention_mask": attention_mask, "cache": cache}

    def update_model_kwargs_for_generation(self, next_tokens, outputs, model_kwargs, is_encoder_decoder=False):
        # Update the model inputs during generation.
        # Note that If `token_type_ids` and `attention_mask` in `model_kwargs`
        # and they contain pad value, the result vectors updated by this method
        # may be different from expected. In this case, you need to rewrite the
        # method.

        # update cache
        if isinstance(outputs, tuple):
            model_kwargs["cache"] = outputs[1]

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs and model_kwargs["token_type_ids"] is not None:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = paddle.concat([token_type_ids, token_type_ids[:, -1:]], axis=-1)

        # update position_ids
        if "position_ids" in model_kwargs and model_kwargs["position_ids"] is not None:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = position_ids[:, -1:] + 1

        # update attention_mask
        if not is_encoder_decoder and "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            # nn.Pad2D don't support the data type `bool`
            if convert_dtype(attention_mask.dtype) == "bool":
                attention_mask = paddle.cast(attention_mask, "int64")
            if len(attention_mask.shape) == 4:
                attention_mask = nn.Pad2D([0, 0, 0, 1], mode="replicate")(attention_mask)
                attention_mask = nn.Pad2D([0, 1, 0, 0], value=-1e4)(attention_mask)
                dtype = convert_dtype(attention_mask.dtype)
                if "int" in dtype:
                    attention_mask[:, :, -1, -1] = 1
                elif "float" in dtype:
                    attention_mask[:, :, -1, -1] = 0.0
                else:
                    raise ValueError("The data type of input `attention_mask` must " "be bool, int or float")
            else:
                attention_mask = paddle.concat(
                    [attention_mask, paddle.ones([attention_mask.shape[0], 1], dtype="int64")], axis=-1
                )
            model_kwargs["attention_mask"] = attention_mask

        # update role_ids
        if "role_ids" in model_kwargs and model_kwargs["role_ids"] is not None:
            role_ids = model_kwargs["role_ids"]
            model_kwargs["role_ids"] = paddle.concat([role_ids, role_ids[:, -1:]], axis=-1)

        model_kwargs["res"] = paddle.concat([model_kwargs["res"], next_tokens], axis=1)

        return model_kwargs

    def sample(
        self,
        input_ids,
        logits_processors,
        max_length,
        pad_token_id,
        eos_token_id,
        top_k=None,
        top_p=None,
        temperature=None,
        min_tokens_to_keep=1,
        **model_kwargs
    ):
        def TopKProcess(probs, top_k, min_tokens_to_keep):
            top_k = min(max(top_k, min_tokens_to_keep), probs.shape[-1])
            # Remove all tokens with a probability less than the last token of the top-k
            topk_probs, _ = paddle.topk(probs, k=top_k)
            probs = paddle.where(probs >= topk_probs[:, -1:], probs, paddle.full_like(probs, 0.0))
            return probs

        def TopPProcess(probs, top_p, min_tokens_to_keep):
            sorted_probs = paddle.sort(probs, descending=True)
            sorted_indices = paddle.argsort(probs, descending=True)
            cumulative_probs = paddle.cumsum(sorted_probs, axis=-1)

            # Remove tokens with cumulative probs above the top_p, But keep at
            # least min_tokens_to_keep tokens
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Set 'min_tokens_to_keep - 1' because the first token is kept
                sorted_indices_to_remove[:, : min_tokens_to_keep - 1] = 0
            # Keep the first token
            sorted_indices_to_remove = paddle.cast(sorted_indices_to_remove, dtype="int64")
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0

            # Scatter sorted tensors to original indexing
            sorted_indices = sorted_indices + paddle.arange(probs.shape[0]).unsqueeze(-1) * probs.shape[-1]
            condition = paddle.scatter(
                sorted_indices_to_remove.flatten(), sorted_indices.flatten(), sorted_indices_to_remove.flatten()
            )
            condition = paddle.cast(condition, "bool").reshape(probs.shape)
            probs = paddle.where(condition, paddle.full_like(probs, 0.0), probs)
            return probs

        batch_size, cur_len = input_ids.shape
        # used for compute on gpu, avoid memcpy D2H
        cur_len_gpu = paddle.full([1], cur_len, dtype="int64")

        origin_len = input_ids.shape[1]
        # used for compute on gpu, avoid memcpy D2H
        origin_len_gpu = paddle.full([1], origin_len, dtype="int64")

        unfinished_flag = paddle.full([batch_size, 1], True, dtype="bool")
        scores = paddle.full([batch_size, 1], 0.0, dtype=paddle.get_default_dtype())

        res = paddle.assign(input_ids)
        model_kwargs["res"] = res

        # use_cache is immutable, we split it off other mutable kwargs.
        assert "use_cache" in model_kwargs
        immutable = {"use_cache": model_kwargs["use_cache"]}
        del model_kwargs["use_cache"]

        def _forward_(**args):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **args, **immutable)
            return self.gpt(**model_inputs, **immutable)

        def _post_process_(outputs, input_ids, cur_len, origin_len, scores, unfinished_flag, model_kwargs):

            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            x_dims_mapping = [auto_env.get_mesh().dp_dim] + [None] * (len(logits.shape) - 1)
            w_dims_mapping = [auto_env.get_mesh().mp_dim, None]
            matmul = auto.shard_op(paddle.matmul, auto_env.get_mesh()[-1], [x_dims_mapping, w_dims_mapping, None])
            with paddle.base.name_scope("skip_quant"):
                logits = matmul(logits, get_attr(self.gpt.embeddings.word_embeddings, "weight"), transpose_y=True)

            # [batch_size, vocab_size]
            logits = logits[:, -1, :]

            # pre-process distribution
            logits = logits_processors(input_ids, logits)

            # sample
            origin_probs = F.softmax(logits)
            if temperature is None or temperature == 1.0:
                probs = paddle.assign(origin_probs)
                origin_probs = paddle.log(origin_probs)
            else:
                origin_probs = paddle.log(origin_probs)
                logits = logits / temperature
                probs = F.softmax(logits)
            if top_k is not None and top_k != 0:
                probs = TopKProcess(probs, top_k, min_tokens_to_keep)
            if top_p is not None and top_p < 1.0:
                if self.use_topp_sampling:
                    try:
                        from ppfleetx_ops import topp_sampling
                    except ImportError:
                        raise ImportError(
                            "please install ppfleetx_ops by 'cd ppfleetx/ops && python setup_cuda.py install'!"
                        )
                    top_ps_tensor = paddle.full(shape=[probs.shape[0]], fill_value=top_p, dtype=probs.dtype)
                    # TODO fake random seed here
                    # Users should set the random seed dynamically when inference
                    _, next_tokens = topp_sampling(probs, top_ps_tensor, random_seed=100)
                else:
                    probs = TopPProcess(probs, top_p, min_tokens_to_keep)

            if not self.use_topp_sampling:
                next_tokens = paddle.multinomial(probs)

            next_scores = paddle.index_sample(origin_probs, next_tokens)

            if eos_token_id is not None:
                next_tokens = paddle.where(unfinished_flag, next_tokens, paddle.full_like(next_tokens, pad_token_id))

            scores = self.update_scores_for_generation(scores, next_scores, cur_len - origin_len, unfinished_flag)

            input_ids = next_tokens

            if eos_token_id is not None:
                unfinished_flag = paddle.logical_and(unfinished_flag, next_tokens != eos_token_id)

            model_kwargs = self.update_model_kwargs_for_generation(
                next_tokens, outputs, model_kwargs, is_encoder_decoder=self.is_encoder_decoder
            )

            return input_ids, scores, unfinished_flag, model_kwargs

        # Note(GuoxiaWang):Pre-while call for inference, simulate a do while loop statement
        # the value in model_kwargs should be tensor before while loop
        outputs = _forward_(**model_kwargs)

        input_ids, scores, unfinished_flag, model_kwargs = _post_process_(
            outputs, input_ids, cur_len_gpu, origin_len_gpu, scores, unfinished_flag, model_kwargs
        )
        if not self.inference:
            cur_len += 1
        else:
            # Note(ZhenyuLi): Avoid the synchronization caused by scale in dy2static
            paddle.increment(cur_len)
        paddle.increment(cur_len_gpu)

        attn_mask = model_kwargs["attention_mask"]
        # make the shape of attention_mask = (-1, -1, -1, -1) in dy2static.
        set_dynamic_shape(model_kwargs["attention_mask"], [-1, -1, -1, -1])
        model_kwargs["cache"] = outputs[1] if isinstance(outputs, tuple) else None
        max_length = paddle.to_tensor(max_length)
        while cur_len < max_length:
            # Note(GuoxiaWang): Remove outputs = _forward_(**model_kwargs)
            # and change it to pass directly to _post_process_ to avoid
            # closed-loop problem of dynamic-to-static model
            input_ids, scores, unfinished_flag, model_kwargs = _post_process_(
                _forward_(**model_kwargs),
                input_ids,
                cur_len_gpu,
                origin_len_gpu,
                scores,
                unfinished_flag,
                model_kwargs,
            )
            if not self.inference:
                cur_len += 1
            else:
                # Note(ZhenyuLi): Avoid the synchronization caused by scale in dy2static
                paddle.increment(cur_len)
            paddle.increment(cur_len_gpu)

            # early finish should be True in generation scenes,
            # If users want to test the inference speed, you can just set it False.
            if self.early_finish and not paddle.any(unfinished_flag):
                break

        return model_kwargs["res"][:, origin_len:], scores

    def forward(self, input_ids=None, **model_kwargs):

        max_length = self.max_length
        min_length = self.min_length
        decode_strategy = self.decode_strategy
        temperature = self.temperature
        top_k = self.top_k
        top_p = self.top_p
        repetition_penalty = self.repetition_penalty
        num_beams = self.num_beams
        num_beam_groups = self.num_beam_groups
        bos_token_id = self.bos_token_id
        eos_token_id = self.eos_token_id
        pad_token_id = self.pad_token_id
        decoder_start_token_id = self.decoder_start_token_id
        forced_bos_token_id = self.forced_bos_token_id
        forced_eos_token_id = self.forced_eos_token_id
        num_return_sequences = self.num_return_sequences
        diversity_rate = self.diversity_rate
        use_cache = self.use_cache

        assert decode_strategy in [
            "greedy_search",
            "sampling",
            "beam_search",
        ], "`decode_strategy` must be one of 'greedy_search', 'sampling' or 'beam_search' but received {}.".format(
            decode_strategy
        )

        bos_token_id = bos_token_id if bos_token_id is not None else getattr(self.gpt, "bos_token_id", None)
        eos_token_id = eos_token_id if eos_token_id is not None else getattr(self.gpt, "eos_token_id", None)
        pad_token_id = pad_token_id if pad_token_id is not None else getattr(self.gpt, "pad_token_id", None)
        forced_bos_token_id = (
            forced_bos_token_id if forced_bos_token_id is not None else getattr(self.gpt, "forced_bos_token_id", None)
        )
        forced_eos_token_id = (
            forced_eos_token_id if forced_eos_token_id is not None else getattr(self.gpt, "forced_eos_token_id", None)
        )
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else getattr(self.gpt, "decoder_start_token_id", None)
        )

        # params check
        if input_ids is None:
            # Init `input_ids` with bos_token_id
            input_ids = self.prepare_input_ids_for_generation(bos_token_id)

        if model_kwargs.get("attention_mask", None) is None:
            # TODO
            # Init `attention_mask` depending on `pad_token_id`
            model_kwargs["attention_mask"] = self.prepare_attention_mask_for_generation(
                input_ids, pad_token_id, eos_token_id
            )

        if model_kwargs.get("position_ids", None) is None:
            model_kwargs["position_ids"] = paddle.arange(
                0, model_kwargs["attention_mask"].shape[-1], dtype=input_ids.dtype
            ).unsqueeze(0)

        self.is_encoder_decoder = False

        model_kwargs["use_cache"] = use_cache

        if self.inference:
            # Note(ZhenyuLi): Avoid the synchronization caused by scale in dy2static
            if hasattr(paddle.framework, "_no_check_dy2st_diff"):
                # TODO(wanghuancoder): _no_check_dy2st_diff is used to turn off the checking of behavior
                # inconsistency between dynamic graph and static graph. _no_check_dy2st_diff should be
                # removed after static graphs support inplace and stride.
                with paddle.framework._no_check_dy2st_diff():
                    min_len = input_ids.shape[-1]
                    max_len = input_ids.shape[-1]
                    paddle.increment(min_len, min_length)
                    paddle.increment(max_len, max_length)
            else:
                min_len = input_ids.shape[-1]
                max_len = input_ids.shape[-1]
                paddle.increment(min_len, min_length)
                paddle.increment(max_len, max_length)
        else:
            input_len = input_ids.shape[-1]
            max_len = max_length + input_len
            min_len = min_length + input_len

        logits_processors = self.get_logits_processor(
            min_length=min_len,
            max_length=max_len,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_rate=diversity_rate,
            repetition_penalty=repetition_penalty,
        )

        if decode_strategy == "sampling":
            if num_return_sequences > 1:
                input_ids, model_kwargs = self.expand_inputs_for_generation(
                    input_ids, expand_size=num_return_sequences, **model_kwargs
                )
            if hasattr(paddle.framework, "_no_check_dy2st_diff"):
                # TODO(wanghuancoder): _no_check_dy2st_diff is used to turn off the checking of behavior
                # inconsistency between dynamic graph and static graph. _no_check_dy2st_diff should be
                # removed after static graphs support inplace and stride.
                with paddle.framework._no_check_dy2st_diff():
                    ret = self.sample(
                        input_ids,
                        logits_processors,
                        max_len,
                        pad_token_id,
                        eos_token_id,
                        top_k,
                        top_p,
                        temperature,
                        **model_kwargs,
                    )
            else:
                ret = self.sample(
                    input_ids,
                    logits_processors,
                    max_len,
                    pad_token_id,
                    eos_token_id,
                    top_k,
                    top_p,
                    temperature,
                    **model_kwargs,
                )
        else:
            raise ValueError(f"Not support {decode_strategy} strategy yet!")
        return ret


