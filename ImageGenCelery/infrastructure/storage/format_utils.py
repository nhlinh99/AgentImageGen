"""Upload-format inference for celery_process/celery_app storage call sites.

ImageGenCelery-specific (not part of the shared common_lib.storage surface) --
callers pass the result straight into StorageManager.save_image_remote /
save_video_remote's ``format`` / ``content_type`` kwargs.
"""
import os
from typing import Any, Optional

import numpy as np
from PIL import Image

from infrastructure.storage.base import ImageFormat


def infer_image_format(image: Any, filename: str) -> ImageFormat:
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


def infer_video_content_type(video_name: str, content_type: Optional[str] = None) -> str:
    """
    Normalize an explicit content_type, or infer one from ``video_name``'s extension
    (.mp4/.m4v -> video/mp4, .webm -> video/webm, .mov -> video/quicktime; else video/mp4).
    """
    if content_type:
        return content_type.split(";")[0].strip().lower()
    ext = os.path.splitext(video_name)[1].lower()
    ext_map = {
        ".mp4": "video/mp4",
        ".m4v": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
    }
    return ext_map.get(ext, "video/mp4")
