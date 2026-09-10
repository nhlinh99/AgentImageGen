import logging

# Must run before anything pulls in comfy: comfy's attention module picks its backend at
# import time, so the sage-attention flag is only honoured if it is set first. This is the
# earliest hook every worker shares -- each entrypoint and every task module imports
# celery_app before touching comfy.
from infrastructure.model_management.comfy_runtime import apply_comfy_runtime_args

apply_comfy_runtime_args()

from celery import Celery
from kombu import Queue

from celery_app.inference.constants import *
from config.config import Config

logging.getLogger("celery.app.trace").setLevel(logging.WARNING)
_config = Config()

class CeleryConfig1:
    task_track_started = True
    task_publish_retry = True
    task_publish_retry_policy = {
        "max_retries": 3,
        "interval_start": 0,
        "interval_step": 0.2,
        "interval_max": 0.5
    }

    worker_prefetch_multiplier = 1
    worker_max_tasks_per_child = 1_000

    # Note: logging
    worker_log_format = "[%(asctime)s][%(processName)s][%(levelname)s] %(message)s"
    worker_task_log_format = "[%(asctime)s][%(processName)s][%(levelname)s] Task %(task_name)s[%(task_id)s] %(message)s"
    task_routes = {
        INFERENCE_PROCESS: {'queue': INFERENCE_PROCESS},
    }
celery_app = Celery(
    app_name="stable_diffusion_service",
    broker=f"amqp://{_config.rabbitmq.user}:{_config.rabbitmq.password}@{_config.rabbitmq.host}:{_config.rabbitmq.port}/{_config.rabbitmq.vhost}",
    backend=f"redis://default:{_config.redis.password}@{_config.redis.host}:{_config.redis.port}/{_config.redis.db}",
    task_serializer='pickle',
    accept_content=['pickle'],
    result_serializer='pickle',
    task_compression='zstd',    # Requires pip install celery[zstd]
    result_compression='zstd',
    include=[
        "celery_app.signal_task.tasks",
        "celery_app.inference.tasks",
    ]
)

celery_app.config_from_object(CeleryConfig1)
celery_app.conf.task_queues = (
    Queue(_config.celery_general_queue),
    Queue(INFERENCE_PROCESS),
)
# Default nếu chưa set queue
celery_app.conf.task_default_queue = _config.celery_general_queue
celery_app.conf.result_backend = f"redis://default:{_config.redis.password}@{_config.redis.host}:{_config.redis.port}/{_config.redis.db}"
celery_app.conf.result_backend_always_retry = True
celery_app.conf.track_started = True
celery_app.conf.result_persistent = True
celery_app.conf.result_expires = 600
