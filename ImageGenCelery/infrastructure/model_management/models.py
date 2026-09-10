import gc
import os
import threading

import torch
from diffusers import Flux2KleinPipeline

from common_lib.base_services import BaseServiceSingleton
from config.config import Config

# Before the first comfy import below -- comfy's attention backend is chosen at module-import
# time. No-op when celery_app already called it.
from infrastructure.model_management.comfy_runtime import apply_comfy_runtime_args

apply_comfy_runtime_args()

try:
    from nunchaku import NunchakuQwenEncoderModel
    from nunchaku.models.transformers.transformer_flux2 import NunchakuFlux2Transformer2DModel
except Exception as e:
    print(f"Can not load model nunchaku: {e}")
    NunchakuQwenEncoderModel = None
    NunchakuFlux2Transformer2DModel = None


class ModelManager(BaseServiceSingleton):
    """Loads/caches the flux2_klein_9b_nunchaku pipeline only -- every other model
    family (Flux, SD15, Z-Image, Qwen, Wan, Krea 2, FluxOneReward, support models)
    was trimmed along with the pipelines that used them.
    """

    def __init__(self, config: Config):
        super(ModelManager, self).__init__(config)
        self.config = config
        self._models_info = {
            "flux2_klein_9b_nunchaku": self.config.model_paths.flux2_klein.flux2_klein_9b_nunchaku.split(","),
        }
        self.list_model_names = self._models_info.keys()
        self.dict_device = {key: self._models_info[key][0].strip() for key in self._models_info}
        self.dict_model_path = {key: self._models_info[key][1].strip() for key in self._models_info}
        self._init_models()

    def _init_models(self):
        self.flux2_klein_9b_nunchaku = None

        self._model_instances = {
            "flux2_klein_9b_nunchaku": [self.flux2_klein_9b_nunchaku, self.load_flux2_klein_9b_nunchaku],
        }
        self._models = {}
        self._lock = threading.Lock()

    def unload_models(self):
        del self._model_instances
        del self._models
        del self._lock

        gc.collect()
        torch.cuda.empty_cache()
        self._init_models()

    def unload_models_by_name(self, model_names: str):
        self.logger.info(f"---Start unload model {model_names}---")
        if isinstance(model_names, str):
            model_names = [model_names]

        with self._lock:
            for model_name in model_names:
                if model_name not in self._model_instances:
                    raise ValueError(f"Model {model_name} not found")

                model_ref, _ = self._model_instances[model_name]
                if model_ref is not None:
                    del model_ref
                    self._model_instances[model_name][0] = None

        gc.collect()
        torch.cuda.empty_cache()
        self.logger.info(f"---Done unload model {model_names}---")

    def load_flux2_klein_9b_nunchaku(self):
        self.logger.info("---Start load flux2_klein_9b_nunchaku---")
        transformer = NunchakuFlux2Transformer2DModel.from_pretrained(
            os.path.join(self.config.model_path, self.dict_model_path["flux2_klein_9b_nunchaku"], "svdq-fp4_r32-FLUX.2-klein-9B-Nunchaku.safetensors"),
            torch_dtype=torch.bfloat16,
        )
        text_encoder = NunchakuQwenEncoderModel.from_pretrained(
            os.path.join(self.config.model_path, self.dict_model_path["flux2_klein_9b_nunchaku"], "text_encoder_int4/svdq-int4-Qwen3-text-Nunchaku.safetensors"),
            device=self.dict_device["flux2_klein_9b_nunchaku"],
            torch_dtype=torch.bfloat16,
        )
        pipe = Flux2KleinPipeline.from_pretrained(
            os.path.join(self.config.model_path, self.dict_model_path["flux2_klein_9b_nunchaku"]),
            text_encoder=text_encoder,
            transformer=transformer,
            torch_dtype=torch.bfloat16
        )
        pipe.to(self.dict_device["flux2_klein_9b_nunchaku"])
        self.logger.info("---Done load flux2_klein_9b_nunchaku---")
        return pipe

    def get_model(self, model_name: str):
        try:
            return self._get_model(model_name)
        except:
            raise ValueError(f"Model {model_name} not found")

    def _get_model(self, model_name):
        if self._model_instances[model_name][0] is None:
            with self._lock:
                if self._model_instances[model_name][0] is None:
                    load_model_func = self._model_instances[model_name][1]
                    model = load_model_func()
                    self._model_instances[model_name][0] = model
                    # Clean VRAM
                    gc.collect()
                    torch.cuda.empty_cache()

        # Chỉ dọn VRAM sau khi thực sự load model (ở nhánh trên). Gọi
        # empty_cache() trên mọi cache hit rất đắt: cudaFree đồng bộ cả device
        # nên phải chờ kernel của các worker khác đang share GPU chạy xong.
        return self._model_instances[model_name][0]
