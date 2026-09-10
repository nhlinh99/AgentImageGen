# Job Management System

A comprehensive job management system for AI processing pipelines, built with Redis queues and MongoDB persistence.

## Overview

The Job Management System provides a robust foundation for managing AI processing jobs with the following features:

- **Job Lifecycle Management**: Complete job lifecycle from creation to completion
- **Priority-based Queuing**: Support for high, normal, and low priority jobs
- **Service-specific Queues**: Separate queues for different AI services (tryon, faceswap, humangen, background_swap)
- **Real-time Status Tracking**: Live job status updates with queue position and ETA
- **Error Handling**: Comprehensive error handling and status tracking
- **User Authorization**: Job ownership verification for security
- **Comprehensive Monitoring**: Statistics and metrics for system health

## Architecture

### System Components

1. **JobService**: Manages job lifecycle and operations
2. **QueueService**: Handles Redis-based priority queues
3. **Job Models**: Data models and enums for job management

### Data Flow

```
Client Request → JobService → MongoDB (Job Storage) → QueueService → Redis (Queue)
                                                                    ↓
Worker Process ← Celery Task ← QueueService ← Redis (Next Job)
```

## Installation

The job management system is part of the DiffusionService and uses existing dependencies:

- Redis (for queue management)
- MongoDB (for job persistence)
- Pydantic (for data validation)
- Celery (for task processing)

## Usage

### Basic Job Creation

```python
from services.job.job_service import JobService
from services.job.job_models import JobPriority, ServiceType

# Initialize service
job_service = JobService(config)

# Create a job
job_result = job_service.create_job(
    user_id="user123",
    service_type=ServiceType.TRYON.value,
    request_data={"image": "base64_data", "options": {...}},
    priority=JobPriority.HIGH
)

print(f"Job created: {job_result['job_id']}")
print(f"Queue position: {job_result['queue_position']}")
```

### Job Status Monitoring

```python
# Get job status
status = job_service.get_job_status("job_id_here")
print(f"Status: {status['status']}")
print(f"Queue position: {status['queue_position']}")
print(f"Estimated wait time: {status['estimated_wait_time']} seconds")
```

### Queue Management

```python
from services.job.queue_service import QueueService

queue_service = QueueService(config)

# Get next job from queue
next_job = queue_service.get_next_job(ServiceType.TRYON.value)
if next_job:
    job_id, position = next_job
    print(f"Processing job: {job_id}")

# Get queue statistics
stats = queue_service.get_queue_statistics()
print(f"Total jobs in queue: {stats['total_jobs']}")
```

### Job Lifecycle Management

```python
# Update job status during processing
job_service.update_job_status(
    job_id="job_id",
    status=JobStatus.PROCESSING,
    celery_task_id="celery_task_123"
)

# Complete job with results
job_service.update_job_status(
    job_id="job_id",
    status=JobStatus.COMPLETED,
    result_data={"output_image": "base64_result", "confidence": 0.95}
)

# Handle job failure
job_service.update_job_status(
    job_id="job_id",
    status=JobStatus.FAILED,
    error_message="Processing failed due to invalid input"
)
```

## API Reference

### JobService

#### Methods

- `create_job(user_id, service_type, request_data, priority)`: Create a new job
- `get_job_status(job_id)`: Get job status and queue information
- `update_job_status(job_id, status, result_data, error_message, celery_task_id)`: Update job status
- `cancel_job(job_id, user_id)`: Cancel a job (authorization required)
- `get_user_jobs(user_id, limit, offset)`: Get jobs for a specific user
- `get_job_statistics()`: Get system-wide job statistics

### QueueService

#### Methods

- `add_job_to_queue(job_id, priority, service_type)`: Add job to queue
- `get_next_job(service_type)`: Get next job from queue
- `get_queue_position(job_id, service_type)`: Get job position in queue
- `remove_job_from_queue(job_id, service_type)`: Remove job from queue
- `get_queue_length(service_type)`: Get queue length
- `get_queue_statistics()`: Get comprehensive queue statistics
- `get_jobs_in_queue(service_type, limit)`: Get list of jobs in queue

## Data Models

### JobStatus Enum

- `QUEUED`: Job in queue, waiting for processing
- `PROCESSING`: Job being processed by worker
- `COMPLETED`: Job completed successfully
- `FAILED`: Job failed with error
- `CANCELLED`: Job cancelled by user

### JobPriority Enum

- `HIGH`: High priority (processed first)
- `NORMAL`: Normal priority (default)
- `LOW`: Low priority (processed last)

### ServiceType Enum

- `TRYON`: Virtual try-on service
- `FACESWAP`: Face swap service
- `HUMANGEN`: Human generation service
- `BACKGROUND_SWAP`: Background swap service

## Redis Data Structure

### Queue Keys

- `queue:jobs:{service_type}`: Sorted set of job IDs with priority scores
- `queue:count:{service_type}`: Integer counter for queue position
- `queue:position:{service_type}`: Hash mapping job_id to position
- `queue:processing_times`: List of recent processing times for ETA calculation

### Priority Scoring

Jobs are scored using: `timestamp + priority_offset`

- High Priority: `timestamp + 0`
- Normal Priority: `timestamp + 1000`
- Low Priority: `timestamp + 2000`

Lower scores = higher priority (Redis sorted sets are ascending)

## MongoDB Schema

### Jobs Collection

```javascript
{
  "_id": ObjectId,
  "job_id": "uuid-string",
  "user_id": "user123",
  "service_type": "tryon",
  "status": "queued",
  "priority": "high",
  "queue_position": 1,
  "created_at": ISODate,
  "updated_at": ISODate,
  "started_at": ISODate,
  "completed_at": ISODate,
  "request_data": {...},
  "result_data": {...},
  "error_message": "string",
  "celery_task_id": "celery-task-123",
  "retry_count": 0,
  "max_retries": 3,
  "pipeline_id": "pipeline-123"
}
```

## Error Handling

The system includes comprehensive error handling:

- **Job Not Found**: Returns appropriate error for non-existent jobs
- **Authorization Errors**: Prevents unauthorized job cancellation
- **Invalid Status Transitions**: Prevents invalid status updates
- **Queue Errors**: Handles Redis connection and operation failures
- **Database Errors**: Handles MongoDB connection and operation failures

## Monitoring and Observability

### Key Metrics

- Job counts by status
- Job counts by service type
- Queue lengths and positions
- Processing times and ETA calculations
- Priority distribution
- Success/failure rates

### Statistics Endpoints

- `get_job_statistics()`: System-wide job statistics
- `get_queue_statistics()`: Queue-specific statistics
- `get_user_jobs()`: User-specific job history

## Integration with Celery

The job management system integrates seamlessly with Celery:

1. **Task Creation**: Jobs are created with Celery task IDs
2. **Status Updates**: Job status is updated as tasks progress
3. **Result Handling**: Task results are stored in job documents
4. **Error Handling**: Task failures are captured in job status

## Example Integration

```python
from celery import current_task
from services.job.job_service import JobService
from services.job.job_models import JobStatus

# In Celery task
@celery_app.task
def process_ai_job(job_id, service_type, request_data):
    job_service = JobService()
    
    try:
        # Update status to processing
        job_service.update_job_status(
            job_id=job_id,
            status=JobStatus.PROCESSING,
            celery_task_id=current_task.request.id
        )
        
        # Process the job
        result = ai_pipeline.process(request_data)
        
        # Update status to completed
        job_service.update_job_status(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            result_data=result
        )
        
        return result
        
    except Exception as e:
        # Update status to failed
        job_service.update_job_status(
            job_id=job_id,
            status=JobStatus.FAILED,
            error_message=str(e)
        )
        raise
```

## Testing

Run the example usage script to test the system:

```bash
cd DiffusionService/services/job
python example_usage.py
```

This will demonstrate:
- Job creation and management
- Queue operations
- Error handling scenarios
- Statistics and monitoring

## Performance Considerations

- **Redis Operations**: All queue operations are O(log N) or better
- **MongoDB Indexing**: Ensure indexes on `job_id`, `user_id`, and `status` fields
- **Connection Pooling**: Services use connection pooling for database operations
- **Batch Operations**: Consider batching for bulk operations
- **Memory Usage**: Processing times are limited to last 100 jobs for ETA calculation

## Security

- **User Authorization**: Jobs can only be cancelled by their owners
- **Input Validation**: All inputs are validated using Pydantic models
- **Error Sanitization**: Error messages are sanitized to prevent information leakage
- **Connection Security**: Redis and MongoDB connections use secure configurations

## Future Enhancements

- **WebSocket Support**: Real-time job status updates
- **Job Scheduling**: Support for delayed job execution
- **Resource Management**: CPU/GPU resource allocation
- **Advanced Analytics**: Detailed performance analytics
- **Multi-tenancy**: Enhanced support for multiple organizations
- **Job Dependencies**: Support for job chains and dependencies
