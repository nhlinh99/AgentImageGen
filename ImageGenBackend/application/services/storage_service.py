import os
import numpy as np
from PIL import Image
from typing import Any, Optional

from api.schemas.model import *
from config.settings import Settings
from domain.interfaces.image_storage import IImageStorage
from domain.value_objects.storage_result import ImageFormat
from infrastructure.storage import StorageManager
from infrastructure.storage.base import RemoteStorageDisabledError
from infrastructure.persistence.mongo.mongo_service import MongoService

from common.base_services import BaseServiceSingleton


class StorageService(BaseServiceSingleton):
    def __init__(self, config: Settings):
        super(StorageService, self).__init__(config)
        self.config = config
        self.mongo_service = MongoService(config)
        self.image_storage: IImageStorage = StorageManager(config)

        self.database_name = self.config.mongo.database_name
        self.collection_diffusion = self.config.mongo.collection_diffusion
        self.collection_human_gen = self.config.mongo.collection_human_gen
        self.storage_name = "StableDiffusion"

    def _require_remote_mode(self, op: str) -> None:
        """Remote-only operations announce themselves as unavailable in local mode
        instead of silently failing (returning "") or hitting an unconfigured
        remote file server."""
        if self.image_storage.mode != "remote":
            raise RemoteStorageDisabledError(
                f"'{op}' requires FILE_SERVER_MODE=remote (currently "
                f"'{self.image_storage.mode}'). Set FILE_SERVER_MODE=remote and configure "
                f"FILE_SERVER_* to use remote storage."
            )

    def upload_image(self, image: Image.Image, image_name: str) -> str:
        self._require_remote_mode("upload_image")
        image_format = self._infer_image_format(image=image, filename=image_name)
        result = self.image_storage.save_image_remote(
            image,
            image_name,
            self.storage_name,
            format=image_format,
        )
        if result.success:
            return result.file_url or ""
        return ""

    def store_local(self, image: Image.Image, image_name: str, storage_name: str = "", is_url_format: bool = True) -> str:
        if storage_name == "":
            storage_name = self.storage_name
        image_format = self._infer_image_format(image=image, filename=image_name)
        result = self.image_storage.save_image_local(
            image,
            image_name,
            storage_name,
            format=image_format,
        )
        if result.success:
            if is_url_format:
                # Matches GET /storage/{folder}/{filename} in api/routes/storage.py.
                return os.path.join(self.config.base_url, "storage", storage_name, image_name)

            return result.file_path or ""

        return ""

    def store_remote(self, image: Image.Image, image_name: str, storage_name: str = "", is_url_format: bool = False) -> str:
        self._require_remote_mode("store_remote")
        if storage_name == "":
            storage_name = self.storage_name
        image_format = self._infer_image_format(image=image, filename=image_name)
        result = self.image_storage.save_image_remote(
            image,
            image_name,
            storage_name,
            format=image_format,
        )

        if result.success:
            if is_url_format:
                return os.path.join(self.config.base_url, storage_name, image_name)
            return result.file_url or ""

        return ""

    def store_video_remote(
        self,
        file_bytes: bytes,
        video_name: str,
        storage_name: str = "",
        content_type: Optional[str] = None,
        is_url_format: bool = False,
    ) -> str:
        """
        Upload video bytes to remote storage, same contract as ``store_remote`` for images.

        If ``content_type`` is omitted, it is inferred from ``video_name`` extension
        (.mp4/.m4v -> video/mp4, .webm -> video/webm, .mov -> video/quicktime; else video/mp4).
        """
        self._require_remote_mode("store_video_remote")
        if storage_name == "":
            storage_name = self.storage_name
        ct = content_type
        if not ct:
            ext = os.path.splitext(video_name)[1].lower()
            ext_map = {
                ".mp4": "video/mp4",
                ".m4v": "video/mp4",
                ".webm": "video/webm",
                ".mov": "video/quicktime",
            }
            ct = ext_map.get(ext, "video/mp4")
        else:
            ct = ct.split(";")[0].strip().lower()
        result = self.image_storage.save_video_remote(
            file_bytes,
            video_name,
            storage_name,
            ct,
        )
        if result.success:
            if is_url_format:
                return os.path.join(self.config.base_url, storage_name, video_name)
            return result.file_url or ""
        return ""

    @staticmethod
    def _infer_image_format(image: Any, filename: str) -> ImageFormat:
        """
        Pick output format from filename / image mode.

        - If filename ends with .png -> PNG
        - If PIL image has alpha (RGBA/LA/P+transparency) or numpy has 4 channels -> PNG
          (must run *before* JPEG-by-extension, or Pillow errors on RGBA-as-JPEG).
        - If filename is .jpg/.jpeg/.jfif/.jpe and image is opaque -> JPEG
        - Else -> JPEG
        """
        name = (filename or "").lower()
        if name.endswith(".png"):
            return ImageFormat.PNG

        try:
            if isinstance(image, Image.Image):
                if image.mode in ("RGBA", "LA") or ("A" in image.getbands()):
                    return ImageFormat.PNG
                if image.mode == "P" and "transparency" in image.info:
                    return ImageFormat.PNG
            elif isinstance(image, np.ndarray):
                if image.ndim == 3 and image.shape[2] == 4:
                    return ImageFormat.PNG
        except Exception:
            # If anything goes wrong, fall back to extension / JPEG below.
            pass

        _root, ext = os.path.splitext(name)
        if ext in (".jpg", ".jpeg", ".jfif", ".jpe"):
            return ImageFormat.JPEG

        return ImageFormat.JPEG
