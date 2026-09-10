"""
Entrypoint for the orchestrator process: the thread that reads the job queue
(Redis/Mongo) and creates the Celery chain/group task for workers to run.
"""
import signal

from config.config import Config
from dispatch.dispatcher import PipelineOrchestrator

if __name__ == "__main__":
    orchestrator = PipelineOrchestrator(Config())

    def _handle_sigterm(signum, frame):
        orchestrator.stop()

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    orchestrator.run()
