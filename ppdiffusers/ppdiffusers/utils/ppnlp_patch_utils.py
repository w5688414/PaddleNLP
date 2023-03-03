# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import builtins
import contextlib
import copy
import functools
import weakref
from collections import OrderedDict
from types import FunctionType, MethodType
from typing import Any, Callable, Dict, List, Optional, Tuple

from .constants import DIFFUSERS_CACHE, PPDIFFUSERS_CACHE
from .hub_utils import HF_HUB_OFFLINE
from .import_utils import (
    is_paddle_available,
    is_paddlenlp_available,
    is_safetensors_available,
)
from .load_utils import smart_load
from .logging import get_logger

logger = get_logger(__name__)

__all__ = []


def copy_func(f):
    "Copy a non-builtin function (NB `copy.copy` does not work for this)"
    if not isinstance(f, FunctionType):
        return copy.copy(f)
    fn = FunctionType(f.__code__, f.__globals__, f.__name__, f.__defaults__, f.__closure__)
    fn.__kwdefaults__ = f.__kwdefaults__
    fn.__dict__.update(f.__dict__)
    fn.__annotations__.update(f.__annotations__)
    fn.__qualname__ = f.__qualname__
    return fn


# copied from https://github.com/fastai/fastcore/blob/c9b4c088d3706569c076e7c197c724730be190ab/fastcore/basics.py#L938-L954
def patch_to(cls, as_prop=False, cls_method=False):
    "Decorator: add `f` to `cls`"
    if not isinstance(cls, (tuple, list)):
        cls = (cls,)

    def _inner(f):
        for c_ in cls:
            nf = copy_func(f)
            nm = f.__name__
            # `functools.update_wrapper` when passing patched function to `Pipeline`, so we do it manually
            for o in functools.WRAPPER_ASSIGNMENTS:
                setattr(nf, o, getattr(f, o))
            nf.__qualname__ = f"{c_.__name__}.{nm}"
            if cls_method:
                setattr(c_, nm, MethodType(nf, c_))
            else:
                setattr(c_, nm, property(nf) if as_prop else nf)
        # Avoid clobbering existing functions
        return globals().get(nm, builtins.__dict__.get(nm, None))

    return _inner


if is_paddle_available():
    import paddle
    import paddle.nn as nn

    paddle.long = paddle.int64
    paddle.int = paddle.int32
    paddle.double = paddle.float64
    paddle.half = paddle.float16
    paddle.from_numpy = paddle.to_tensor
    paddle.Tensor.half = lambda x: paddle.cast(x, paddle.float16)
    paddle.Tensor.float = lambda x: paddle.cast(x, paddle.float32)
    paddle.Tensor.double = lambda x: paddle.cast(x, paddle.float64)
    paddle.Tensor.int = lambda x: paddle.cast(x, paddle.int32)
    paddle.Tensor.long = lambda x: paddle.cast(x, paddle.int64)
    paddle.Tensor.bool = lambda x: paddle.cast(x, paddle.bool)
    paddle.Tensor.bfloat16 = lambda x: paddle.cast(x, paddle.bfloat16)
    paddle.Tensor.clamp = paddle.clip
    paddle.clamp = paddle.clip

    def view_pt(x, *shape: builtins.int, name=None):
        return paddle.reshape(x, shape=shape, name=name)

    paddle.view = view_pt
    paddle.Tensor.view = view_pt
    setattr(paddle.Tensor, "data", property(lambda x: x))
    paddle.Tensor.data_ptr = lambda x: x.value().get_tensor()._ptr()

    def permute_pt(x, *perm: builtins.int, name=None):
        return paddle.transpose(x, perm=perm, name=name)

    paddle.permute = permute_pt
    paddle.Tensor.permute = permute_pt
    paddle.cat = paddle.concat
    paddle.Tensor.softmax = nn.functional.softmax

    def size_pt(self, i=None):
        if i is None:
            return self.shape
        return self.shape[i]

    paddle.Tensor.size = size_pt
    paddle.Tensor.contiguous = lambda x: x

    # must return self!
    @patch_to(nn.Layer)
    def eval(self):
        # Layer-level setting
        self.training = False
        for layer in self.sublayers():
            layer.training = False
        return self

    @patch_to(nn)
    def Parameter(data: paddle.Tensor, requires_grad=True):
        tensor = paddle.create_parameter(data.shape, dtype=data.dtype, default_initializer=nn.initializer.Assign(data))
        if not requires_grad:
            tensor.stop_gradient = True
        return tensor

    @contextlib.contextmanager
    def device_scope(device="cpu"):
        new_device = device.replace("cuda", "gpu")
        old_device = paddle.get_device()
        try:
            paddle.set_device(new_device)
            yield
        finally:
            paddle.set_device(old_device)

    paddle.device_scope = device_scope

    @patch_to(nn.Layer)
    def get_sublayer(self, target: str):
        if target == "":
            return self

        atoms: List[str] = target.split(".")
        mod: nn.Layer = self

        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(mod.__class__.__name__ + " has no " "attribute `" + item + "`")

            mod = getattr(mod, item)

            if not isinstance(mod, nn.Layer):
                raise AttributeError("`" + item + "` is not " "an nn.Layer")
        return mod

    class _WrappedHook:
        def __init__(self, hook: Callable, module: Optional["nn.Layer"] = None):
            self.hook: Callable = hook
            functools.update_wrapper(self, hook)

            self.with_module: bool = False

            if module is not None:
                self.module: weakref.ReferenceType["nn.Layer"] = weakref.ref(module)
                self.with_module = True

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            if self.with_module:
                module = self.module()
                if module is None:
                    raise RuntimeError("You are trying to call the hook of a dead Module!")
                return self.hook(module, *args, **kwargs)
            return self.hook(*args, **kwargs)

        def __getstate__(self) -> Dict:
            result = {"hook": self.hook, "with_module": self.with_module}
            if self.with_module:
                result["module"] = self.module()

            return result

        def __setstate__(self, state: Dict):
            self.hook = state["hook"]
            self.with_module = state["with_module"]

            if self.with_module:
                if state["module"] is None:
                    raise RuntimeError("You are trying to revive the hook of a dead Module!")
                self.module = weakref.ref(state["module"])

    from paddle.fluid.dygraph.layers import HookRemoveHelper

    @patch_to(nn.Layer)
    def register_load_state_dict_pre_hook(self, hook, with_module=False):
        handle = HookRemoveHelper(self.load_state_dict_pre_hooks)
        self.load_state_dict_pre_hooks[handle._hook_id] = _WrappedHook(hook, self if with_module else None)
        return handle

    raw_set_state_dict = nn.Layer.set_state_dict

    @patch_to(nn.Layer)
    def set_state_dict(self, state_dict, use_structured_name: bool = True):
        for hook in self.load_state_dict_pre_hooks.values():
            hook(state_dict)
        return raw_set_state_dict(self, state_dict, use_structured_name=use_structured_name)

    nn.Layer.load_dict = nn.Layer.set_state_dict
    nn.Layer.set_dict = nn.Layer.set_state_dict

    raw_init = nn.Layer.__init__

    @patch_to(nn.Layer)
    def __init__(self, name_scope=None, dtype="float32"):
        raw_init(self, name_scope=name_scope, dtype=dtype)
        self.load_state_dict_pre_hooks = OrderedDict()


if is_paddle_available() and is_paddlenlp_available():
    import paddle

    import paddlenlp.transformers
    from paddlenlp import __version__
    from paddlenlp.transformers import PretrainedConfig, PretrainedModel

    @patch_to(PretrainedModel, as_prop=True)
    def dtype(parameter: nn.Layer) -> paddle.dtype:
        last_dtype = None
        for _, t in parameter.named_parameters():
            last_dtype = t.dtype
            if hasattr(t, "is_floating_point"):
                if t.is_floating_point():
                    return t.dtype
            else:
                if t.dtype in [paddle.float16, paddle.float32, paddle.float64, paddle.bfloat16]:
                    return t.dtype
        return last_dtype

    @patch_to(PretrainedModel, as_prop=True)
    def device(self):
        try:
            return next(self.named_parameters())[1].place
        except StopIteration:
            return paddle.get_device()

    try:
        from paddlenlp.transformers import XLMRobertaTokenizer
    except ImportError:
        # patch xlm-roberta tokenizer
        """Tokenization classes for XLM-RoBERTa model."""
        import os
        from shutil import copyfile

        import sentencepiece as spm

        from paddlenlp.transformers.tokenizer_utils import (
            AddedToken,
            PretrainedTokenizer,
        )

        SPIECE_UNDERLINE = "▁"

        class XLMRobertaTokenizer(PretrainedTokenizer):

            resource_files_names = {"vocab_file": "sentencepiece.bpe.model"}
            pretrained_resource_files_map = {}
            pretrained_init_configuration = {}
            max_model_input_sizes = {
                "xlm-roberta-base": 512,
                "xlm-roberta-large": 512,
                "xlm-roberta-large-finetuned-conll02-dutch": 512,
                "xlm-roberta-large-finetuned-conll02-spanish": 512,
                "xlm-roberta-large-finetuned-conll03-english": 512,
                "xlm-roberta-large-finetuned-conll03-german": 512,
            }
            model_input_names = ["input_ids", "attention_mask"]

            def __init__(
                self,
                vocab_file,
                bos_token="<s>",
                eos_token="</s>",
                sep_token="</s>",
                cls_token="<s>",
                unk_token="<unk>",
                pad_token="<pad>",
                mask_token="<mask>",
                sp_model_kwargs: Optional[Dict[str, Any]] = None,
                **kwargs
            ) -> None:
                # Mask token behave like a normal word, i.e. include the space before it
                mask_token = (
                    AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token
                )

                self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

                super().__init__(
                    bos_token=bos_token,
                    eos_token=eos_token,
                    unk_token=unk_token,
                    sep_token=sep_token,
                    cls_token=cls_token,
                    pad_token=pad_token,
                    mask_token=mask_token,
                    sp_model_kwargs=self.sp_model_kwargs,
                    **kwargs,
                )

                self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
                self.sp_model.Load(str(vocab_file))
                self.vocab_file = vocab_file

                # Original fairseq vocab and spm vocab must be "aligned":
                # Vocab    |    0    |    1    |   2    |    3    |  4  |  5  |  6  |   7   |   8   |  9
                # -------- | ------- | ------- | ------ | ------- | --- | --- | --- | ----- | ----- | ----
                # fairseq  | '<s>'   | '<pad>' | '</s>' | '<unk>' | ',' | '.' | '▁' | 's'   | '▁de' | '-'
                # spm      | '<unk>' | '<s>'   | '</s>' | ','     | '.' | '▁' | 's' | '▁de' | '-'   | '▁a'

                # Mimic fairseq token-to-id alignment for the first 4 token
                self.fairseq_tokens_to_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}

                # The first "real" token "," has position 4 in the original fairseq vocab and position 3 in the spm vocab
                self.fairseq_offset = 1

                self.fairseq_tokens_to_ids["<mask>"] = len(self.sp_model) + self.fairseq_offset
                self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}

            def __getstate__(self):
                state = self.__dict__.copy()
                state["sp_model"] = None
                state["sp_model_proto"] = self.sp_model.serialized_model_proto()
                return state

            def __setstate__(self, d):
                self.__dict__ = d

                # for backward compatibility
                if not hasattr(self, "sp_model_kwargs"):
                    self.sp_model_kwargs = {}

                self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
                self.sp_model.LoadFromSerializedProto(self.sp_model_proto)

            def build_inputs_with_special_tokens(
                self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
            ) -> List[int]:
                """
                Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
                adding special tokens. An XLM-RoBERTa sequence has the following format:
                - single sequence: `<s> X </s>`
                - pair of sequences: `<s> A </s></s> B </s>`
                Args:
                    token_ids_0 (`List[int]`):
                        List of IDs to which the special tokens will be added.
                    token_ids_1 (`List[int]`, *optional*):
                        Optional second list of IDs for sequence pairs.
                Returns:
                    `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
                """

                if token_ids_1 is None:
                    return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
                cls = [self.cls_token_id]
                sep = [self.sep_token_id]
                return cls + token_ids_0 + sep + sep + token_ids_1 + sep

            def get_special_tokens_mask(
                self,
                token_ids_0: List[int],
                token_ids_1: Optional[List[int]] = None,
                already_has_special_tokens: bool = False,
            ) -> List[int]:
                """
                Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
                special tokens using the tokenizer `prepare_for_model` method.
                Args:
                    token_ids_0 (`List[int]`):
                        List of IDs.
                    token_ids_1 (`List[int]`, *optional*):
                        Optional second list of IDs for sequence pairs.
                    already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                        Whether or not the token list is already formatted with special tokens for the model.
                Returns:
                    `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
                """

                if already_has_special_tokens:
                    return super().get_special_tokens_mask(
                        token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
                    )

                if token_ids_1 is None:
                    return [1] + ([0] * len(token_ids_0)) + [1]
                return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

            def create_token_type_ids_from_sequences(
                self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
            ) -> List[int]:
                """
                Create a mask from the two sequences passed to be used in a sequence-pair classification task. XLM-RoBERTa does
                not make use of token type ids, therefore a list of zeros is returned.
                Args:
                    token_ids_0 (`List[int]`):
                        List of IDs.
                    token_ids_1 (`List[int]`, *optional*):
                        Optional second list of IDs for sequence pairs.
                Returns:
                    `List[int]`: List of zeros.
                """

                sep = [self.sep_token_id]
                cls = [self.cls_token_id]

                if token_ids_1 is None:
                    return len(cls + token_ids_0 + sep) * [0]
                return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

            @property
            def vocab_size(self):
                return len(self.sp_model) + self.fairseq_offset + 1  # Add the <mask> token

            def get_vocab(self):
                vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
                vocab.update(self.added_tokens_encoder)
                return vocab

            def _tokenize(self, text: str) -> List[str]:
                return self.sp_model.encode(text, out_type=str)

            def _convert_token_to_id(self, token):
                """Converts a token (str) in an id using the vocab."""
                if token in self.fairseq_tokens_to_ids:
                    return self.fairseq_tokens_to_ids[token]
                spm_id = self.sp_model.PieceToId(token)

                # Need to return unknown token if the SP model returned 0
                return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

            def _convert_id_to_token(self, index):
                """Converts an index (integer) in a token (str) using the vocab."""
                if index in self.fairseq_ids_to_tokens:
                    return self.fairseq_ids_to_tokens[index]
                return self.sp_model.IdToPiece(index - self.fairseq_offset)

            def convert_tokens_to_string(self, tokens):
                """Converts a sequence of tokens (strings for sub-words) in a single string."""
                out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
                return out_string

            def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
                if not os.path.isdir(save_directory):
                    logger.error(f"Vocabulary path ({save_directory}) should be a directory")
                    return
                out_vocab_file = os.path.join(
                    save_directory,
                    (filename_prefix + "-" if filename_prefix else "") + self.resource_files_names["vocab_file"],
                )

                if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(
                    self.vocab_file
                ):
                    copyfile(self.vocab_file, out_vocab_file)
                elif not os.path.isfile(self.vocab_file):
                    with open(out_vocab_file, "wb") as fi:
                        content_spiece_model = self.sp_model.serialized_model_proto()
                        fi.write(content_spiece_model)

                return (out_vocab_file,)

        paddlenlp.transformers.XLMRobertaTokenizer = XLMRobertaTokenizer

    # patch BertModel forward
    from paddlenlp.transformers import BertModel

    raw_forward = BertModel.forward

    @patch_to(BertModel)
    def forward(
        self,
        input_ids: paddle.Tensor,
        token_type_ids: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if attention_mask is None:
            attention_mask = paddle.ones_like(input_ids)
        return raw_forward(
            self,
            input_ids,
            token_type_ids,
            position_ids,
            attention_mask,
            past_key_values,
            use_cache,
            output_hidden_states,
            output_attentions,
            return_dict,
        )

    raw_from_pretrained = PretrainedModel.from_pretrained

    TRANSFORMERS_SAFE_WEIGHTS_NAME = "model.safetensors"
    TRANSFORMERS_WEIGHTS_NAME = "pytorch_model.bin"
    from .download_utils import _add_variant, _get_model_file

    @classmethod
    def from_pretrained_model(cls, pretrained_model_name_or_path, *args, from_hf_hub=False, **kwargs):
        if cls.constructed_from_pretrained_config() and hasattr(cls, "smart_convert"):
            return cls.from_pretrained_v3(pretrained_model_name_or_path, from_hf_hub=from_hf_hub, *args, **kwargs)
        return raw_from_pretrained(pretrained_model_name_or_path, *args, from_hf_hub=False, **kwargs)

    @classmethod
    def from_pretrained_v3_model(cls, pretrained_model_name_or_path, from_hf_hub: bool = False, *args, **kwargs):
        cache_dir = (
            kwargs.pop("cache_dir", DIFFUSERS_CACHE) if from_hf_hub else kwargs.pop("cache_dir", PPDIFFUSERS_CACHE)
        )
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        from_diffusers = kwargs.pop("from_diffusers", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        paddle_dtype = kwargs.pop("paddle_dtype", None)
        subfolder = kwargs.pop("subfolder", None)
        variant = kwargs.pop("variant", None)

        user_agent = {
            "ppdiffusers": __version__,
            "file_type": "model",
            "framework": "paddle",
        }

        config = None
        # 1. get the PretrainedConfig to init model
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            # must from hf hub
            if from_hf_hub:
                if subfolder is not None:
                    kwargs["subfolder"] = subfolder
            else:
                if subfolder is not None:
                    config_path = os.path.join(config_path, subfolder)

            config = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=False,
                force_download=force_download,
                from_hf_hub=from_hf_hub,
                **kwargs,
            )
        assert config is not None

        model = cls(config)
        # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
        # Load model
        model_file = None
        if from_diffusers:
            if is_safetensors_available():
                try:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path,
                        weights_name=_add_variant(TRANSFORMERS_SAFE_WEIGHTS_NAME, variant),
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                        from_hf_hub=from_hf_hub,
                    )
                except Exception:  # noqa: E722
                    pass
            if model_file is None:
                model_file = _get_model_file(
                    pretrained_model_name_or_path,
                    weights_name=_add_variant(TRANSFORMERS_WEIGHTS_NAME, variant),
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                    from_hf_hub=from_hf_hub,
                )
        else:
            model_file = _get_model_file(
                pretrained_model_name_or_path,
                weights_name=_add_variant("model_state.pdparams", variant),
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
                from_hf_hub=from_hf_hub,
            )
        assert model_file is not None

        # try load model_file with paddle / torch / safetensor
        state_dict = smart_load(model_file)

        # convert weights
        if from_diffusers and hasattr(cls, "smart_convert"):
            state_dict = cls.smart_convert(state_dict, model)

        loaded_state_dict_keys = list(state_dict.keys())

        if paddle_dtype is not None:
            model.to(dtype=paddle_dtype)

        model, missing_keys, unexpected_keys, mismatched_keys = cls._load_pretrained_model(
            model=model,
            state_dict=state_dict,
            loaded_keys=loaded_state_dict_keys,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            dtype=paddle_dtype,
        )
        loading_info = {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "mismatched_keys": mismatched_keys,
            "error_msgs": "",
        }
        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
                " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
                " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
            )
        else:
            logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")

        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint"
                f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
                " training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
                " to use it for predictions and inference."
            )

        if output_loading_info:
            return model, loading_info

        return model

    PretrainedModel.from_pretrained = from_pretrained_model
    PretrainedModel.from_pretrained_v3 = from_pretrained_v3_model
