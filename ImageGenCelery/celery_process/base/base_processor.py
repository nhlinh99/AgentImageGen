import base64
from typing import Any, Dict

import numpy as np
from common_lib.base_services import BaseServiceSingleton
from PIL import Image

from celery_process.base.celery_canvas_utils import flatten_celery_canvas_task_names, mark_is_last_pipeline_task_kwargs
from config.config import Config
from domain.image.io import convert_byte_to_image


class BaseProcessor(BaseServiceSingleton):
    def __init__(self, config: Config):
        super(BaseProcessor, self).__init__(config)

    def process(self, request: Any, **kwargs):
        """Template method shared by every processor: transform the request,
        build the subclass's pipeline chain, wrap the result.
        """
        user_id = kwargs.get("user_id", "")
        request = self.transform(request)
        return self.chain_process_result(self._build_pipeline_chain(request, user_id))

    def _build_pipeline_chain(self, request: Any, user_id: str):
        raise NotImplementedError("_build_pipeline_chain method not implemented")

    @staticmethod
    def transform(data: Any) -> Any:
        """Default: validate/normalize a single `.image` field to a PIL Image.
        Override for requests with a different shape (multiple images, no image, etc).
        """
        image = data.image
        if image is None:
            raise ValueError("Input Invalid!!!")

        if isinstance(image, Image.Image):
            print("Input Image Valid")
        elif isinstance(image, np.ndarray):
            data.image = Image.fromarray(image)
        elif isinstance(image, bytes):
            data.image = convert_byte_to_image(base64.b64decode(image))
        else:
            raise ValueError("Input Invalid!!!")

        return data

    def chain_process_result(self, chain_task: Any) -> Dict[str, Any]:
        mark_is_last_pipeline_task_kwargs(chain_task)
        task_names = flatten_celery_canvas_task_names(chain_task)
        return {"chain_task": chain_task, "task_names": task_names}
