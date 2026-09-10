"""
Job Models Module
Defines enums and data models for job management system
"""

from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from pydantic import BaseModel, Field

INFERENCE_SERVICE = "Inference"


class Status(str, Enum):
    """Unified status enumeration for jobs and tasks"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(str, Enum):
    """Job priority enumeration"""
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class ServiceType(str, Enum):
    """AI service types"""
    INFERENCE = INFERENCE_SERVICE


class TaskDocument(BaseModel):
    """Task document model for individual tasks within a job"""
    task_name: str = Field(..., description="Name of the task")
    task_order: int = Field(..., description="Order of task in pipeline")
    status: Status = Field(..., description="Current task status")

    # Timestamps
    created_at: str = Field(..., description="Task creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    started_at: Optional[str] = Field(None, description="Task start timestamp")
    completed_at: Optional[str] = Field(None, description="Task completion timestamp")

    # Celery integration
    celery_task_id: Optional[str] = Field(None, description="Celery task identifier")

    # Results and errors
    result: Optional[Dict[str, Any]] = Field(None, description="Task result data")
    error: Optional[str] = Field(None, description="Error message if task failed")

    # Retry information
    retry_count: int = Field(default=0, description="Number of retry attempts for this task")
    max_retries: int = Field(default=3, description="Maximum retry attempts for this task")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class JobDocument(BaseModel):
    """Job document model for MongoDB"""
    job_id: str = Field(..., description="Unique job identifier")
    user_id: str = Field(..., description="User identifier")
    service_type: str = Field(..., description="Type of AI service")
    queue_lane: Optional[str] = Field(
        default=None,
        description="Redis queue lane: fast, medium, or slow (multi-lane admission control)",
    )
    status: Status = Field(..., description="Current job status")
    priority: JobPriority = Field(default=JobPriority.NORMAL, description="Job priority")

    queue_position: Optional[int] = Field(default=None, description="Position in queue")
    first_queue_position: Optional[int] = Field(
        default=None,
        description="Queue position (jobs ahead in queue + processing) snapshot at job creation",
    )

    # Timestamps
    created_at: str = Field(..., description="Job creation timestamp")
    updated_at: Optional[str] = Field(default=None, description="Last update timestamp")
    started_at: Optional[str] = Field(default=None, description="Processing start timestamp")
    completed_at: Optional[str] = Field(default=None, description="Completion timestamp")

    # Data
    request_data: Dict[str, Any] = Field(..., description="Original request data")
    result_data: Optional[Dict[str, Any]] = Field(default=None, description="Processing results")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")

    # Celery integration
    celery_task_id: Optional[str] = Field(default=None, description="Celery task identifier")

    # Retry information
    retry_count: int = Field(default=0, description="Number of retry attempts")
    max_retries: int = Field(default=3, description="Maximum retry attempts")

    # Task tracking
    tasks: Optional[List[TaskDocument]] = Field(default=None, description="List of tasks in the job pipeline")
    total_tasks: Optional[int] = Field(default=None, description="Total number of tasks")
    completed_tasks: Optional[int] = Field(default=0, description="Number of completed tasks")
    failed_tasks: Optional[int] = Field(default=0, description="Number of failed tasks")

    # Lock information
    locked: Optional[bool] = Field(default=False, description="Whether the job is currently locked")
    locked_at: Optional[str] = Field(default=None, description="Timestamp when job was locked")
    locked_by: Optional[str] = Field(default=None, description="User ID who locked the job")
    lock_expires_at: Optional[str] = Field(default=None, description="Timestamp when lock expires")
    unlocked_at: Optional[str] = Field(default=None, description="Timestamp when job was unlocked")

    # Favorite flag
    favorite: Optional[bool] = Field(default=False, description="Whether the job is marked as favorite")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class JobCreateRequest(BaseModel):
    """Request model for creating a job"""
    user_id: str = Field(..., description="User identifier")
    service_type: ServiceType = Field(..., description="Type of AI service")
    request_data: Dict[str, Any] = Field(..., description="Request data for processing")
    priority: JobPriority = Field(default=JobPriority.NORMAL, description="Job priority")


class JobStatusResponse(BaseModel):
    """Response model for job status"""
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Current job status")
    service_type: str = Field(..., description="Type of AI service")
    queue_position: Optional[int] = Field(None, description="Position in queue")
    estimated_wait_time: Optional[int] = Field(None, description="Estimated wait time in seconds")

    # Timestamps
    created_at: str = Field(..., description="Job creation timestamp")
    started_at: Optional[str] = Field(None, description="Processing start timestamp")
    completed_at: Optional[str] = Field(None, description="Completion timestamp")

    # Processing information
    retry_count: int = Field(default=0, description="Number of retry attempts")
    result_data: Optional[Dict[str, Any]] = Field(None, description="Processing results")
    error_message: Optional[str] = Field(None, description="Error message if failed")

    # Task information
    tasks: Optional[List[TaskDocument]] = Field(None, description="List of tasks in the job")
    total_tasks: Optional[int] = Field(None, description="Total number of tasks")
    completed_tasks: Optional[int] = Field(None, description="Number of completed tasks")
    failed_tasks: Optional[int] = Field(None, description="Number of failed tasks")

    # Lock information
    is_locked: Optional[bool] = Field(default=False, description="Whether the job is currently locked")
    locked: Optional[bool] = Field(default=False, description="Whether the job is locked (MongoDB field)")
    locked_at: Optional[str] = Field(None, description="Timestamp when job was locked")
    locked_by: Optional[str] = Field(None, description="User ID who locked the job")
    lock_expires_at: Optional[str] = Field(None, description="Timestamp when lock expires")
    unlocked_at: Optional[str] = Field(None, description="Timestamp when job was unlocked")

    # Favorite flag
    favorite: Optional[bool] = Field(default=False, description="Whether the job is marked as favorite")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class JobCreateResponse(BaseModel):
    """Response model for job creation"""
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Initial job status")
    queue_position: int = Field(..., description="Position in queue")
    first_queue_position: Optional[int] = Field(
        None,
        description="Initial queue position at creation (for wait-time / analytics)",
    )
    created_at: str = Field(..., description="Job creation timestamp")
    estimated_wait_time: Optional[int] = Field(None, description="Estimated wait time in seconds")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class JobListResponse(BaseModel):
    """Response model for job list"""
    jobs: list = Field(..., description="List of job information")
    total: int = Field(..., description="Total number of jobs")
    limit: int = Field(..., description="Requested limit")
    offset: int = Field(..., description="Requested offset")


class QueueStatistics(BaseModel):
    """Model for queue statistics"""
    total_jobs: int = Field(..., description="Total jobs in queue")
    service_queues: Dict[str, Dict[str, Any]] = Field(..., description="Statistics by service")
    priority_distribution: Dict[str, int] = Field(..., description="Jobs by priority")
    average_wait_time: float = Field(..., description="Average wait time in seconds")


class JobStatistics(BaseModel):
    """Model for job statistics"""
    status_counts: Dict[str, int] = Field(..., description="Jobs by status")
    service_counts: Dict[str, int] = Field(..., description="Jobs by service type")
    queue_stats: QueueStatistics = Field(..., description="Queue statistics")
    total_jobs: int = Field(..., description="Total jobs in system")


class JobUpdateRequest(BaseModel):
    """Request model for updating job status"""
    status: Status = Field(..., description="New job status")
    result_data: Optional[Dict[str, Any]] = Field(None, description="Processing results")
    error_message: Optional[str] = Field(None, description="Error message")
    celery_task_id: Optional[str] = Field(None, description="Celery task identifier")


class JobCancelRequest(BaseModel):
    """Request model for cancelling a job"""
    user_id: str = Field(..., description="User identifier for authorization")


class JobLockRequest(BaseModel):
    """Request model for locking a job"""
    lock_duration: int = Field(default=3600, description="Lock duration in seconds (default 1 hour)")


class JobUnlockRequest(BaseModel):
    """Request model for unlocking a job"""
    # No additional fields needed - job_id comes from URL path
    pass


class ServiceTypeLockRequest(BaseModel):
    """Request model for locking jobs by service type"""
    service_type: str = Field(..., description="Service type to lock jobs for (tryon, faceswap, humangen, background_swap)")
    lock_duration: int = Field(default=3600, description="Lock duration in seconds (default 1 hour)")


class TaskCreateRequest(BaseModel):
    """Request model for creating tasks for a job"""
    job_id: str = Field(..., description="Job identifier")
    task_names: List[str] = Field(..., description="List of task names in the pipeline")


class TaskUpdateRequest(BaseModel):
    """Request model for updating task status"""
    job_id: str = Field(..., description="Job identifier")
    task_name: str = Field(..., description="Name of the task to update")
    status: Status = Field(..., description="New task status")
    celery_task_id: Optional[str] = Field(None, description="Celery task identifier")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result data")
    error: Optional[str] = Field(None, description="Error message if task failed")


class TaskResponse(BaseModel):
    """Response model for task information"""
    task_name: str = Field(..., description="Name of the task")
    task_order: int = Field(..., description="Order of task in pipeline")
    status: str = Field(..., description="Current task status")

    # Timestamps
    created_at: str = Field(..., description="Task creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    started_at: Optional[str] = Field(None, description="Task start timestamp")
    completed_at: Optional[str] = Field(None, description="Task completion timestamp")

    # Celery integration
    celery_task_id: Optional[str] = Field(None, description="Celery task identifier")

    # Results and errors
    result: Optional[Dict[str, Any]] = Field(None, description="Task result data")
    error: Optional[str] = Field(None, description="Error message if task failed")

    # Retry information
    retry_count: int = Field(default=0, description="Number of retry attempts")
    max_retries: int = Field(default=3, description="Maximum retry attempts")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class JobTasksResponse(BaseModel):
    """Response model for job tasks list"""
    job_id: str = Field(..., description="Job identifier")
    tasks: List[TaskResponse] = Field(..., description="List of tasks")
    total_tasks: int = Field(..., description="Total number of tasks")
    completed_tasks: int = Field(..., description="Number of completed tasks")
    failed_tasks: int = Field(..., description="Number of failed tasks")
    progress_percentage: float = Field(..., description="Job progress percentage")


# Utility functions for job management
def create_job_id() -> str:
    """Generate a unique job identifier"""
    import uuid
    return str(uuid.uuid4())


def is_retryable_status(status: Status) -> bool:
    """Check if a job status allows retry"""
    return status in [Status.FAILED]


def is_cancellable_status(status: Status) -> bool:
    """Check if a job status allows cancellation"""
    return status in [Status.QUEUED]


def is_final_status(status: Status) -> bool:
    """Check if a job status is final (no further changes)"""
    return status in [Status.COMPLETED, Status.FAILED, Status.CANCELLED]


def get_priority_score(priority: JobPriority) -> int:
    """Get numeric score for priority (lower = higher priority)"""
    priority_scores = {
        JobPriority.HIGH: 1,
        JobPriority.NORMAL: 2,
        JobPriority.LOW: 3
    }
    return priority_scores.get(priority, 2)


def get_service_queue_key(service_type: ServiceType) -> str:
    """Get Redis queue key for service type"""
    return f"queue:jobs:{service_type.value}"


# Utility functions for task management
def is_task_retryable_status(status: Status) -> bool:
    """Check if a task status allows retry"""
    return status in [Status.FAILED]


def is_task_final_status(status: Status) -> bool:
    """Check if a task status is final (no further changes)"""
    return status in [Status.COMPLETED, Status.FAILED]


def is_task_running_status(status: Status) -> bool:
    """Check if a task status indicates it's currently running"""
    return status in [Status.PROCESSING]


def calculate_job_progress(completed_tasks: int, total_tasks: int) -> float:
    """Calculate job progress percentage"""
    if total_tasks == 0:
        return 0.0
    return round((completed_tasks / total_tasks) * 100, 2)


def get_task_status_priority(status: Status) -> int:
    """Get numeric priority for task status (lower = higher priority for display)"""
    status_priorities = {
        Status.FAILED: 1,
        Status.PROCESSING: 2,
        Status.QUEUED: 3,
        Status.COMPLETED: 4,
        Status.CANCELLED: 5
    }
    return status_priorities.get(status, 3)


# DateTime conversion utilities
def convert_datetime_to_string(obj: Union[BaseModel, Dict[str, Any]]) -> Union[BaseModel, Dict[str, Any]]:
    """
    Convert datetime fields to ISO format strings in JobDocument models

    Args:
        obj: JobDocument model instance or dictionary

    Returns:
        Object with datetime fields converted to strings
    """
    if isinstance(obj, dict):
        # Handle dictionary
        result = {}
        for key, value in obj.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, dict):
                result[key] = convert_datetime_to_string(value)
            elif isinstance(value, list):
                result[key] = [convert_datetime_to_string(item) for item in value]
            else:
                result[key] = value
        return result

    elif hasattr(obj, '__dict__'):
        # Handle Pydantic models
        result = {}
        for field_name, field_value in obj.__dict__.items():
            if isinstance(field_value, datetime):
                result[field_name] = field_value.isoformat()
            elif isinstance(field_value, dict):
                result[field_name] = convert_datetime_to_string(field_value)
            elif isinstance(field_value, list):
                result[field_name] = [convert_datetime_to_string(item) for item in field_value]
            else:
                result[field_name] = field_value
        return result

    return obj


def convert_job_document_datetime_fields(job_doc: Union[JobDocument, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert datetime fields in JobDocument to strings for JSON serialization

    Args:
        job_doc: JobDocument instance or dictionary

    Returns:
        Dictionary with datetime fields converted to ISO format strings
    """
    if isinstance(job_doc, JobDocument):
        # Convert Pydantic model to dict first
        job_dict = job_doc.dict()
    else:
        job_dict = job_doc

    # Convert datetime fields to strings
    datetime_fields = ['created_at', 'updated_at', 'started_at', 'completed_at', 'locked_at', 'lock_expires_at', 'unlocked_at']

    for field in datetime_fields:
        if field in job_dict and job_dict[field] is not None:
            if isinstance(job_dict[field], datetime):
                job_dict[field] = job_dict[field].isoformat()

    # Handle nested TaskDocument objects
    if 'tasks' in job_dict and job_dict['tasks'] is not None:
        for task in job_dict['tasks']:
            if isinstance(task, dict):
                for field in datetime_fields:
                    if field in task and task[field] is not None:
                        if isinstance(task[field], datetime):
                            task[field] = task[field].isoformat()
            elif hasattr(task, '__dict__'):
                # Handle TaskDocument objects
                task_dict = task.dict()
                for field in datetime_fields:
                    if field in task_dict and task_dict[field] is not None:
                        if isinstance(task_dict[field], datetime):
                            task_dict[field] = task_dict[field].isoformat()
                task = task_dict

    return job_dict
