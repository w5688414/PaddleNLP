# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from .utils.import_utils import is_paddle_available, is_paddlenlp_available

if is_paddle_available() and is_paddlenlp_available():
    from paddlenlp.transformers import (
        BertModel,
        BitBackbone,
        ChineseCLIPConfig,
        ChineseCLIPVisionConfig,
        CLIPTextConfig,
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPVisionConfig,
        CLIPVisionModel,
        CLIPVisionModelWithProjection,
        DPTForDepthEstimation,
    )
    from paddlenlp.utils.log import logger

    from .models.modeling_pytorch_paddle_utils import (
        convert_pytorch_state_dict_to_paddle,
    )
    from .pipelines.alt_diffusion.modeling_roberta_series import (
        RobertaSeriesModelWithTransformation,
    )
    from .pipelines.latent_diffusion.pipeline_latent_diffusion import LDMBertModel
    from .pipelines.paint_by_example.image_encoder import PaintByExampleImageEncoder
    from .pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
    from .pipelines.stable_diffusion_safe.safety_checker import (
        SafeStableDiffusionSafetyChecker,
    )

    # Hack this, https://github.com/PaddlePaddle/PaddleNLP/pull/5074 !
    def create_from_pretrained(model_type="clip", config_type="text_config"):
        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, from_hf_hub: bool = False, cache_dir=None, **kwargs):
            kwargs.update({"from_hf_hub": from_hf_hub, "cache_dir": cache_dir})
            config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

            if config_dict.get("model_type") == model_type:
                projection_dim = config_dict.get("projection_dim", None)
                config_dict = config_dict[config_type]
                if projection_dim is not None:
                    config_dict["projection_dim"] = projection_dim

            if (
                "model_type" in config_dict
                and hasattr(cls, "model_type")
                and config_dict["model_type"] != cls.model_type
            ):
                logger.warning(
                    f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                    f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
                )

            return cls.from_dict(config_dict, **kwargs)

        return from_pretrained

    CLIPTextConfig.from_pretrained = create_from_pretrained("clip", "text_config")
    CLIPVisionConfig.from_pretrained = create_from_pretrained("clip", "vision_config")
    ChineseCLIPConfig.from_pretrained = create_from_pretrained("chinese_clip", "text_config")
    ChineseCLIPVisionConfig.from_pretrained = create_from_pretrained("chinese_clip", "vision_config")

    @classmethod
    def clip_smart_convert(cls, state_dict, pd_model):
        new_model_state = {}
        name_mapping_dict = {
            ".encoder.": ".transformer.",
            ".layer_norm": ".norm",
            ".mlp.": ".",
            ".fc1.": ".linear1.",
            ".fc2.": ".linear2.",
            ".final_layer_norm.": ".ln_final.",
            ".embeddings.": ".",
            ".position_embedding.": ".positional_embedding.",
            ".patch_embedding.": ".conv1.",
            "visual_projection.weight": "vision_projection",
            "text_projection.weight": "text_projection",
            ".pre_layrnorm.": ".ln_pre.",
            ".post_layernorm.": ".ln_post.",
        }
        ignore_value = [
            "position_ids",
        ]
        if cls in [PaintByExampleImageEncoder]:
            # ignore mapper. prefix, we will use convert_pytorch_state_dict_to_paddle to convert mapper.xxxx state_dict
            ignore_value.append("mapper.")
        else:
            name_mapping_dict.update({".vision_model.": "."})

        donot_transpose = ["embeddings", "norm", "concept_embeds", "special_care_embeds"]
        for name, value in state_dict.items():
            # step1: ignore position_ids
            if any(i in name for i in ignore_value):
                continue
            # step2: transpose nn.Linear weight
            if value.ndim == 2 and not any(i in name for i in donot_transpose):
                value = value.T
            # step3: hf_name -> ppnlp_name mapping
            for hf_name, ppnlp_name in name_mapping_dict.items():
                name = name.replace(hf_name, ppnlp_name)
            # step4: 0d tensor -> 1d tensor
            if name == "logit_scale" and value.ndim == 1:
                value = value.reshape((1,))
            # step5: safety_checker need prefix "clip."
            if "vision_model" in name and cls in [StableDiffusionSafetyChecker, SafeStableDiffusionSafetyChecker]:
                name = "clip." + name
            new_model_state[name] = value

        if cls in [PaintByExampleImageEncoder]:
            # convert mapper
            mappersd = convert_pytorch_state_dict_to_paddle(state_dict, pd_model, sub_layer="mapper.")
            new_model_state.update(mappersd)

        return new_model_state

    @classmethod
    def bert_smart_convert(cls, state_dict, pd_model):
        new_model_state = {}
        name_mapping_dict = {
            # about embeddings
            "embeddings.LayerNorm.weight": "embeddings.layer_norm.weight",
            "embeddings.LayerNorm.bias": "embeddings.layer_norm.bias",
            # about encoder layer
            "encoder.layer": "encoder.layers",
            "attention.self.query": "self_attn.q_proj",
            "attention.self.key": "self_attn.k_proj",
            "attention.self.value": "self_attn.v_proj",
            "attention.output.dense": "self_attn.out_proj",
            "attention.output.LayerNorm.weight": "norm1.weight",
            "attention.output.LayerNorm.bias": "norm1.bias",
            "intermediate.dense": "linear1",
            "output.dense": "linear2",
            "output.LayerNorm.weight": "norm2.weight",
            "output.LayerNorm.bias": "norm2.bias",
            # about cls predictions ignore
            "cls.predictions.transform.dense": "cls.predictions.transform",
            "cls.predictions.decoder.weight": "cls.predictions.decoder_weight",
            "cls.predictions.transform.LayerNorm.weight": "cls.predictions.layer_norm.weight",
            "cls.predictions.transform.LayerNorm.bias": "cls.predictions.layer_norm.bias",
            "cls.predictions.bias": "cls.predictions.decoder_bias",
        }
        ignore_value = ["position_ids"]
        donot_transpose = ["embeddings", "norm"]
        for name, value in state_dict.items():
            # step1: ignore position_ids
            if any(i in name for i in ignore_value):
                continue
            # step2: transpose nn.Linear weight
            if value.ndim == 2 and not any(i in name for i in donot_transpose):
                value = value.T
            # step3: hf_name -> ppnlp_name mapping
            for hf_name, ppnlp_name in name_mapping_dict.items():
                name = name.replace(hf_name, ppnlp_name)
            new_model_state[name] = value

        return new_model_state

    @classmethod
    def ldmbert_smart_convert(cls, state_dict, pd_model):
        transformers2ppnlp = {
            "model.embed_tokens.weight": "embeddings.word_embeddings.weight",
            "model.embed_positions.weight": "embeddings.position_embeddings.weight",
            "model.layer_norm.": "final_layer_norm.",
            "model.layers": "encoder.layers",
            ".self_attn_layer_norm.": ".norm1.",
            ".final_layer_norm.": ".norm2.",
            ".fc1.": ".linear1.",
            ".fc2.": ".linear2.",
        }
        ignore_value = ["to_logits"]
        donot_transpose = ["embed_tokens", "embed_positions", "norm"]
        new_model_state = {}
        for name, value in state_dict.items():
            # step1: ignore to_logits
            if any(i in name for i in ignore_value):
                continue
            # step2: transpose nn.Linear weight
            if value.ndim == 2 and not any(i in name for i in donot_transpose):
                value = value.T
            # step3: hf_name -> ppnlp_name mapping
            for hf_name, ppnlp_name in transformers2ppnlp.items():
                name = name.replace(hf_name, ppnlp_name)
            new_model_state[name] = value
        return new_model_state

    # TODO implement LDMBertModel with PretrainedConfig
    LDMBertModel.smart_convert = ldmbert_smart_convert
    for cls_ in [
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPVisionModel,
        CLIPVisionModelWithProjection,
        StableDiffusionSafetyChecker,
        SafeStableDiffusionSafetyChecker,
        PaintByExampleImageEncoder,
    ]:
        setattr(cls_, "smart_convert", clip_smart_convert)

    for cls_ in [BertModel, RobertaSeriesModelWithTransformation]:
        setattr(cls_, "smart_convert", bert_smart_convert)

    for cls_ in [DPTForDepthEstimation, BitBackbone]:
        setattr(cls_, "smart_convert", convert_pytorch_state_dict_to_paddle)

    # patch get_image_processor_dict support subfolder.
    import json

    from .utils import (
        DIFFUSERS_CACHE,
        FROM_HF_HUB,
        HF_HUB_OFFLINE,
        PPDIFFUSERS_CACHE,
        _get_model_file,
    )

    IMAGE_PROCESSOR_NAME = "preprocessor_config.json"
    from paddlenlp.transformers.image_processing_utils import ImageProcessingMixin

    @classmethod
    def get_image_processor_dict(cls, pretrained_model_name_or_path, **kwargs):
        from_hf_hub = kwargs.pop("from_hf_hub", FROM_HF_HUB)
        cache_dir = (
            kwargs.pop("cache_dir", DIFFUSERS_CACHE) if from_hf_hub else kwargs.pop("cache_dir", PPDIFFUSERS_CACHE)
        )
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        user_agent = kwargs.pop("user_agent", None)

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)

        resolved_image_processor_file = _get_model_file(
            pretrained_model_name_or_path,
            weights_name=IMAGE_PROCESSOR_NAME,
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
        try:
            # Load image_processor dict
            with open(resolved_image_processor_file, "r", encoding="utf-8") as reader:
                text = reader.read()
            image_processor_dict = json.loads(text)

        except json.JSONDecodeError:
            raise EnvironmentError(
                f"It looks like the config file at '{resolved_image_processor_file}' is not a valid JSON file."
            )

        logger.info(
            f"loading configuration file {resolved_image_processor_file} from cache at {resolved_image_processor_file}"
        )

        return image_processor_dict, kwargs

    ImageProcessingMixin.get_image_processor_dict = get_image_processor_dict
