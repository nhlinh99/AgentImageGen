# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo layout

This is a monorepo of independently-deployable Python services plus one vendored dependency:

- **`ImageGenBackend/`** — FastAPI HTTP API. Accepts inference requests, writes jobs to Mongo/Redis, returns immediately (`202`-style "deferred" response). Does **not** run any GPU inference itself.
- **`ImageGenCelery/`** — Two roles in one codebase:
  - `orchestrator.py` (entrypoint) polls the same job queue the backend writes to, resolves a `celery_process.*Processor` for the job's `service_type`, builds a Celery chain/group, and dispatches it with `apply_async()`.
  - `workers/celery_general.py` (entrypoint) is the actual Celery worker process that executes GPU-bound tasks (via ComfyUI, through `CommonComfy`).
- **`CommonLib/`** — Shared package (`common_lib`) installed into both `ImageGenBackend` and `ImageGenCelery`: Mongo/Redis/RabbitMQ pydantic-settings (`config.py`), `BaseService`/`BaseServiceSingleton`/`Singleton` metaclass (`base_services.py`), job models, logging format, storage.
- **`CommonComfy/`** — Vendored/embedded ComfyUI (`comfyui_common_lib/comfy_node/...` mirrors upstream ComfyUI's own tree, including `custom_nodes/`) plus an `evaluation/` module (CLIP, SSIM, LPIPS scoring). This is the inference engine the Celery GPU workers drive — treat `comfy_node/` as third-party vendored code, not first-party.
- **`deployment/helm/`** — Helm charts, currently just the `mongodb`, `rabbitmq`, `redis` dependency subcharts (no top-level app chart yet).

`ImageGenBackend` and `ImageGenCelery` each also carry a **local** `common/`/`libs`-style duplicate of a few `common_lib` pieces (e.g. `ImageGenBackend/common/base_services.py`) — check which one a file actually imports (`common_lib.X` vs `common.X`) before assuming they're the same.

## Architecture: job lifecycle

1. Client hits an `ImageGenBackend` route (`api/routes/inference.py` etc.) → `application/services/job_submit_service.py: JobSubmitService.run_pipeline()`.
2. `JobSubmitService` converts any image fields to URLs (`application/services/image_url_conversion.py`), creates a job document via `JobService` (`infrastructure/persistence/job/job_service.py`, Mongo + Redis), and returns immediately — **it never calls a pipeline directly**. Response includes `pipeline_execution: "deferred_to_job_processor"`.
3. `ImageGenCelery/orchestrator.py` → `dispatch/dispatcher.py: PipelineOrchestrator._job_processing_loop()` polls `QueueService.get_next_job_for_scheduler()` for the next admissible job across three priority lanes (fast/medium/slow, each independently capped — see `job_limits` in Config/Settings).
4. For the job's `service_type`, the orchestrator looks up a `celery_process.*Processor` (subclass of `celery_process/base/base_processor.py: BaseProcessor`) via `pipeline_map`, and a matching Pydantic request model via `data_model_map` (both in `dispatcher.py` — **register new service types in both dicts**).
5. `BaseProcessor.process()` is a template method: `transform()` (normalize `.image` to PIL) → subclass's `_build_pipeline_chain()` (builds the actual Celery chain/group of tasks) → `chain_process_result()` (flattens task names via `celery_canvas_utils.py`, marks the last task in the chain).
6. Orchestrator calls `.apply_async()` on the resulting chain, records task names/celery task id back onto the job (`JobService`), and continues polling.
7. Celery workers (`workers/celery_general.py`, started per the Helm chart with `--queue-name`) pick up tasks from `celery_app.*` task modules (each task registered with `@celery_app.task(..., name=...)`) and execute them — GPU-bound tasks import `infrastructure/model_management/comfy_runtime.py` to drive ComfyUI (`CommonComfy`) before doing any model work.
8. Job state/results are read back via `ImageGenBackend`'s `api/routes/jobs.py` / `results.py`.

Because the backend and the orchestrator/workers are **separate processes reading/writing the same Mongo/Redis state**, a change to the job document shape, queue lane keys (`infrastructure/persistence/job/queue_lane.py` / `queue_service.py`), or `ServiceType`/status enums (`domain/schema` in Celery, `api/schemas/model.py` in Backend) must be kept consistent across both services — there is no shared schema package enforcing this at import time (only `common_lib.job_models`/`config` are actually shared).

## Adding a new inference pipeline / service type

Touches all of these (grep for an existing one, e.g. `INFERENCE`/`ServiceType.INFERENCE`, as a template):
1. `ImageGenCelery/domain/schema` — add the `ServiceType` enum value + the Celery-side request model.
2. `ImageGenCelery/celery_process/<name>.py` — new `BaseProcessor` subclass implementing `_build_pipeline_chain()`.
3. `ImageGenCelery/dispatch/dispatcher.py` — register the new processor + request model in `pipeline_map` / `data_model_map`.
4. `ImageGenCelery/celery_app/<name>/` — task module(s) with `@celery_app.task(..., name=...)`, plus a `constants.py` for the queue/task name, included in the worker's `get_queue_names()`.
5. `ImageGenBackend/api/schemas` + `api/routes` — request/response models and the route calling `JobSubmitService.run_pipeline()`.

## Settings pattern

Both `ImageGenBackend/config/settings.py` (`Settings`) and `ImageGenCelery/config/config.py` (`Config`) are `pydantic-settings` `BaseSettings`, each composing shared `common_lib.config` groups (`MongoSettings`, `RedisSettings`, `RabbitMQSettings` — env-prefixed `MONGO_`/`REDIS_`/`RABBITMQ_`) plus their own local groups (`JobLimitSettings`/`JobLimitsSettings` env-prefixed `LIMIT_`, backend-only `StorageSettings` env-prefixed `FILE_SERVER_`). Both call `load_dotenv()` at import time so `os.environ` and pydantic-settings agree. `.env.example` in each service documents every var and its default — copy it to `.env` for local dev. `FILE_SERVER_MODE=local` (default) disables the remote file-server storage path entirely (`RemoteStorageDisabledError` on remote-only ops); Celery has no file-server credential fields at all, so it can only ever run in local mode.

`USE_SAGE_ATTENTION` (Celery `Config`) is **process-global**: it changes the attention implementation for every model in the worker process, not just one pipeline.

## Running services locally

No build system, task runner, linter config, or test suite exists yet beyond `ImageGenCelery/.pylintrc` (pylint, 150-char lines, docstring/broad-except/unused-argument checks disabled) — there is currently no test directory under any of the four Python packages.

```bash
# Install shared packages into a service's environment (editable, from repo root)
pip install -e CommonLib
pip install -e CommonComfy

# Per-service deps
pip install -r ImageGenBackend/requirements.txt
pip install -r ImageGenCelery/requirements.txt          # CPU-only, no torch/CUDA
pip install -r ImageGenCelery/requirements-gpu.txt       # add on GPU worker hosts (torch stack via CommonComfy, diffusers, etc.)

# Backend API (from ImageGenBackend/)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Celery orchestrator (from ImageGenCelery/) — polls queue, dispatches chains
python orchestrator.py

# Celery worker (from ImageGenCelery/) — executes GPU tasks
python -m workers.celery_general
```

Each service needs its own `.env` (copy from that service's `.env.example`) pointing at Mongo/Redis/RabbitMQ; `ImageGenBackend` and `ImageGenCelery` are expected to point at the *same* Mongo/Redis instance since they share job state through it.

## Conventions to follow

- Service classes derive from `common_lib.base_services.BaseServiceSingleton` (or the local `common.base_services` duplicate in Backend) — one instance per class process-wide via the `Singleton` metaclass. Don't add a competing DI pattern.
- A `ponytail:` comment marks a deliberate scope cut with a named upgrade trigger (e.g. `config/config.py`'s `Flux2KleinModelPaths` — other loader fields were deleted, restore only if those pipelines return). Preserve these comments when touching nearby code; they're load-bearing context, not stale notes.
- Module docstrings in this codebase frequently cross-reference an external `BrandStudio` project's file layout (e.g. `main.py` mirrors `BrandStudio/backend/app/main.py`) — that project isn't in this repo; treat those references as historical design provenance, not a dependency to resolve.
