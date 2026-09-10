"""Inference processor — builds the Celery signature for the inference.process demo task."""

from typing import Any

from celery import signature

from celery_app import celery_app
from celery_app.inference.constants import INFERENCE_PROCESS
from celery_process.base.base_processor import BaseProcessor
from config.config import Config


class CeleryInferenceProcessor(BaseProcessor):
    def __init__(self, config: Config):
        super(CeleryInferenceProcessor, self).__init__(config)

    @staticmethod
    def transform(data: Any) -> Any:
        return data

    def _build_pipeline_chain(self, request: Any, user_id: str):
        return signature(
            INFERENCE_PROCESS,
            kwargs={
                "job_id": request.job_id,
                "user_id": user_id,
                "image": request.image,
            },
            app=celery_app,
            immutable=True,
        )
