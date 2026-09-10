from celery_app import celery_app
from celery_app.base_task.base_task import BaseTask
from celery_app.inference.constants import INFERENCE_PROCESS


@celery_app.task(bind=True, base=BaseTask, name=INFERENCE_PROCESS)
def inference_process(self: BaseTask, *args, image=None, **kwargs):
    print("Hello World")

    image_info = None
    if image:
        # image is a URL (e.g. ImageGenBackend's GET /images/{folder}/{filename}) -- fetch
        # it over HTTP rather than reading a local path, since this worker and the backend
        # don't share a filesystem.
        result = self.image_storage.load_image_remote(image)
        if result.success:
            image_info = {"size": result.image.size, "mode": result.image.mode}
            self.logger.info("Loaded input image %s: %s", image, image_info)
        else:
            self.logger.warning("Failed to load input image %s: %s", image, result.error)

    return {"message": "Hello World", "image": image, "image_info": image_info}
