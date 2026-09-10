"""Generic Celery worker entrypoint for the Helm chart (image-gen-celery-v2).

Unlike celery_worker.py/celery_support.py/celery_ksampler.py, this script owns no
queue list of its own -- every worker Deployment in the chart passes its queues
explicitly via --queue-name (see values.yaml `workers[].processes`), so importing
`celery_app` (which registers every task module via its own `include=[...]`) is
enough; no per-domain `celery_app.<module>.constants` imports are needed here.
"""
from pathlib import Path

from workers._multiprocess import run_worker_cli
from celery_app import celery_app
from celery_app.inference.constants import INFERENCE_PROCESS
from config.config import Config

_config = Config()

def get_queue_names():
    """Get queue names for worker."""
    list_queue = [
        _config.celery_general_queue,
        INFERENCE_PROCESS,
    ]

    return ",".join(list_queue)


if __name__ == "__main__":
    run_worker_cli(
        celery_app=celery_app,
        get_queue_names=get_queue_names,
        base_dir=Path(__file__).parent.parent.absolute(),
        default_worker_name="worker",
        default_concurrency=1,
    )
