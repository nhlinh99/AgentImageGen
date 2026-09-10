import time
import traceback
from typing import Any, Dict, Optional

from celery.signals import task_failure, task_prerun, task_success
from pydantic import BaseModel

from application.services.job_media_helpers import check_job_cancelled
from celery_app.base_task.base_task import BaseTask
from config.config import Config
from infrastructure.persistence.job.job_service import JobService
from domain.schema.job_models import Status

_config = Config()

time_dict = {}


# ------------------ Signal: task bắt đầu ------------------
@task_prerun.connect
def task_started_handler(sender: Optional[BaseTask] = None, args=None, kwargs={}, **_):
    if not isinstance(sender, BaseTask):
        return

    job_id = kwargs.get('job_id')
    job_service: JobService = sender.job_service
    task_name = sender.name

    # Update job task status if job_id is provided
    if job_id:
        check_job_cancelled(job_service, job_id)
        time_key = f"time_{task_name}_{job_id}"
        time_dict[time_key] = time.perf_counter()

# ------------------ Signal: task thành công ------------------
@task_success.connect
def task_success_handler(sender: Optional[BaseTask] = None, result=None, **kwargs):
    if not isinstance(sender, BaseTask):
        return

    if isinstance(result, BaseModel):
        result = result.model_dump(mode="json")

    task_name: str = sender.name
    job_id = kwargs.get('job_id') or getattr(sender.request, "kwargs", {}).get("job_id")
    job_service: JobService = sender.job_service

    time_key = f"time_{task_name}_{job_id}"

    # Update job task status if job_id is provided
    if job_id:
        try:
            req_kw = getattr(sender.request, "kwargs", None) or {}
            is_last_pipeline_task = kwargs.get("is_last_pipeline_task")
            if is_last_pipeline_task is None:
                is_last_pipeline_task = req_kw.get("is_last_pipeline_task")
            # From mark_is_last_pipeline_task_kwargs: only explicit True finalizes the job here.
            should_run_completion_check = bool(is_last_pipeline_task)
            job_task_order = kwargs.get("job_task_order")
            if job_task_order is None:
                job_task_order = req_kw.get("job_task_order")
            success = job_service.update_task_status(
                job_id=job_id,
                task_name=task_name,
                status=Status.COMPLETED.value,
                result=None,
                time_process=time.perf_counter() - time_dict[time_key],
                job_task_order=job_task_order,
            )

            time_dict.pop(time_key)

            if success:
                print(f"[Signal] Job Task SUCCESS - {task_name} | job_id={job_id}")

                # Finalize job only on terminal chain steps (see mark_is_last_pipeline_task_kwargs)
                if should_run_completion_check:
                    job_already_completed = False
                    try:
                        job_doc = job_service.get_job_status(job_id)
                        job_already_completed = job_doc.get("status") == Status.COMPLETED.value
                    except ValueError:
                        pass

                    if not job_already_completed:
                        job_service.update_job_status(
                            job_id=job_id,
                            status=Status.COMPLETED,
                            result_data=result
                        )
                        print(f"[Signal] All tasks completed - Job {job_id} status updated to COMPLETED")
                        job_service.clear_pipeline_redis_for_job(job_id)
                    else:
                        print(
                            f"[Signal] Job {job_id} already COMPLETED; "
                            f"skipping duplicate finalize for {task_name}"
                        )
            else:
                print(
                    f"[Signal] Failed to update task status for {sender.name} | job_id={job_id} "
                    f"(task not queued or not found; job left unchanged)"
                )
        except Exception as e:
            print(f"[Signal] Error updating task status: {str(e)}")
            job_service.update_job_status(
                job_id=job_id,
                status=Status.FAILED,
                error_message=(
                    f"Error updating task status after task success for '{task_name}' | job_id={job_id}: {str(e)}"
                ),
            )
            job_service.clear_pipeline_redis_for_job(job_id)


def _format_task_failure_detail(task_name: str, exception: Optional[BaseException]) -> str:
    """Exception message, innermost source line, and full traceback for logging/storage."""
    if exception is None:
        return f"Task '{task_name}' failed: Unknown error"
    header = f"Task '{task_name}' failed: {exception!s}"
    tb = exception.__traceback__
    loc = ""
    if tb:
        frames = traceback.extract_tb(tb)
        if frames:
            f = frames[-1]
            loc = f'\nAt: File "{f.filename}", line {f.lineno}, in {f.name}'
    tb_text = "".join(traceback.format_exception(type(exception), exception, tb))
    return header + loc + "\n\n" + tb_text


# ------------------ Signal: task thất bại ------------------
@task_failure.connect
def task_failure_handler(sender: Optional[BaseTask] = None, exception=None, args=None, kwargs=None, **_):
    if not isinstance(sender, BaseTask):
        return

    kwargs = kwargs or {}
    req_kw = getattr(sender.request, "kwargs", None) or {}
    task_name: str = sender.name
    job_id = kwargs.get("job_id") or req_kw.get("job_id")

    job_service: JobService = sender.job_service
    time_key = f"time_{task_name}_{job_id}"

    # Update job task status if job_id is provided
    if job_id:
        check_job_cancelled(job_service, job_id)
        try:
            detailed_error = _format_task_failure_detail(task_name, exception)

            job_task_order = kwargs.get("job_task_order")
            if job_task_order is None:
                job_task_order = req_kw.get("job_task_order")
            elapsed = 0.0
            if time_key in time_dict:
                elapsed = time.perf_counter() - time_dict[time_key]
            task_success = job_service.update_task_status(
                job_id=job_id,
                task_name=task_name,
                status=Status.FAILED.value,
                error=detailed_error,
                time_process=elapsed,
                job_task_order=job_task_order,
            )

            time_dict.pop(time_key, None)

            job_success = job_service.update_job_status(
                job_id=job_id,
                status=Status.FAILED,
                error_message=detailed_error
            )

            if task_success:
                print(f"[Signal] Job Task FAILURE - {task_name} | job_id={job_id}\n{detailed_error}")
            else:
                print(f"[Signal] Failed to update task status for {task_name} | job_id={job_id}")

            if job_success:
                print(f"[Signal] Job document status updated to FAILED for job_id={job_id}")
            else:
                print(f"[Signal] Failed to update job status for job_id={job_id}")

        except Exception as e:
            print(f"[Signal] Error updating task/job status: {str(e)}")
        finally:
            job_service.clear_pipeline_redis_for_job(job_id)
