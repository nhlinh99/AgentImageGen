"""Convert raw image payloads (URL / base64 / numpy / PIL / bytes) into stored URLs.

Moved out of infrastructure/image_processing/utils.py — it persists images via
StorageService, which is an application-layer service, so infrastructure
importing it was backwards (infra must never depend on application). This is
application-layer orchestration, not a pure utility.
"""
from __future__ import annotations

from typing import Any, Dict
import base64
import io
import uuid

import numpy as np
from PIL import Image

from application.services.storage_service import StorageService
from common.logging_format import module_logger
from config.settings import Settings
from domain.services.image_processing import detect_image_format_from_bytes, resize_image_custom

logger = module_logger("log_common")

storage_service = StorageService(Settings())


def convert_image_to_url(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert request_data image (url, numpy, PIL, bytes) into local file URLs

    :param request_data: Request data dictionary
    :return: Updated request data with image URLs
    """
    processed_data = request_data.copy()

    def convert_image_field(value, field_name: str):
        """Convert a single image field to local URL"""
        try:
            # Handle URL - return as is
            if isinstance(value, str) and (value.startswith('http://') or value.startswith('https://')):
                return value

            # Handle base64 data URL
            elif isinstance(value, str) and value.startswith('data:image'):
                # Extract base64 data
                header, encoded = value.split(',', 1)
                image_bytes = base64.b64decode(encoded)

                # Convert to PIL Image
                image = Image.open(io.BytesIO(image_bytes))

                # Resize if larger than 1 megapixel
                image = resize_image_custom(image, logger=logger)

                # Generate unique filename
                file_extension = header.split('/')[-1].split(';')[0]
                if file_extension == 'jpeg':
                    file_extension = 'jpg'

                filename = f"temp_{field_name}_{uuid.uuid4().hex[:8]}.{file_extension}"

                # Save image locally using storage manager
                # is_url_format=True (default): returns a URL served by GET
                # /storage/{folder}/{filename} (api/routes/storage.py), not a raw file_path.
                file_path = storage_service.store_local(
                    image=image,
                    image_name=filename,
                    storage_name="uploaded_images",
                )

                if file_path:
                    logger.info("Saved image %s to local storage: %s", field_name, file_path)
                    return file_path
                logger.error("Failed to save image %s", field_name)
                raise RuntimeError(
                    f"Failed to convert field '{field_name}': could not persist data URL image to storage"
                )

            # Handle numpy array
            elif isinstance(value, np.ndarray):
                # Convert numpy array to PIL Image
                if value.dtype != np.uint8:
                    value = (value * 255).astype(np.uint8)

                if len(value.shape) == 3 and value.shape[2] == 3:
                    image = Image.fromarray(value, 'RGB')
                elif len(value.shape) == 3 and value.shape[2] == 4:
                    image = Image.fromarray(value, 'RGBA')
                elif len(value.shape) == 2:
                    image = Image.fromarray(value, 'L')
                else:
                    raise ValueError(
                        f"Unsupported numpy array shape for field '{field_name}': {value.shape}. "
                        "Expected 2D (H, W) or 3D (H, W, 3) or (H, W, 4)."
                    )

                # Resize if larger than 1 megapixel
                image = resize_image_custom(image, logger=logger)

                # Generate unique filename
                filename = f"temp_{field_name}_{uuid.uuid4().hex[:8]}.jpg"

                # Save image locally using storage manager
                # is_url_format=True (default): returns a URL served by GET
                # /storage/{folder}/{filename} (api/routes/storage.py), not a raw file_path.
                file_path = storage_service.store_local(
                    image=image,
                    image_name=filename,
                    storage_name="uploaded_images",
                )

                if file_path:
                    logger.info("Saved numpy array %s to local storage: %s", field_name, file_path)
                    return file_path
                logger.error("Failed to save numpy array %s", field_name)
                raise RuntimeError(
                    f"Failed to convert field '{field_name}': could not persist numpy image to storage"
                )

            # Handle PIL Image
            elif isinstance(value, Image.Image):
                # Resize if larger than 1 megapixel
                image = resize_image_custom(value)

                # Generate unique filename
                filename = f"temp_{field_name}_{uuid.uuid4().hex[:8]}.jpg"

                # Save image locally using storage manager
                # is_url_format=True (default): returns a URL served by GET
                # /storage/{folder}/{filename} (api/routes/storage.py), not a raw file_path.
                file_path = storage_service.store_local(
                    image=image,
                    image_name=filename,
                    storage_name="uploaded_images",
                )

                if file_path:
                    logger.info("Saved PIL image %s to local storage: %s", field_name, file_path)
                    return file_path
                logger.error("Failed to save PIL image %s", field_name)
                raise RuntimeError(
                    f"Failed to convert field '{field_name}': could not persist PIL image to storage"
                )

            # Handle bytes
            elif isinstance(value, bytes):
                # Detect image format from bytes using magic numbers
                detected_format = detect_image_format_from_bytes(value)

                # Try to open the image
                try:
                    # Create BytesIO object and ensure it's at the beginning
                    image_bytes_io = io.BytesIO(value)
                    image_bytes_io.seek(0)

                    # Open image - PIL should be able to detect format from magic numbers
                    image = Image.open(image_bytes_io)

                    # If we detected a format but PIL didn't, use our detected format
                    # Otherwise, use PIL's detected format
                    if detected_format:
                        final_format = detected_format
                    else:
                        final_format = image.format
                        if not final_format:
                            logger.warning(
                                "Could not determine image format for field %s, defaulting to JPEG", field_name
                            )
                            final_format = 'JPEG'

                except Exception as e:
                    raise ValueError(
                        f"Cannot identify image from bytes for field '{field_name}': {e}"
                    ) from e

                # Map format to file extension
                format_to_ext = {
                    'PNG': 'png',
                    'JPEG': 'jpg',
                    'JPG': 'jpg',
                    'WEBP': 'webp',
                    'GIF': 'gif',
                    'BMP': 'bmp',
                    'TIFF': 'tiff',
                }

                # Get file extension based on format, default to jpg if unknown
                file_extension = format_to_ext.get(final_format, 'jpg')

                # Resize if larger than 1 megapixel
                image = resize_image_custom(image, logger=logger)

                # Generate unique filename with correct extension
                filename = f"temp_{field_name}_{uuid.uuid4().hex[:8]}.{file_extension}"

                # Save image locally using storage manager
                # is_url_format=True (default): returns a URL served by GET
                # /storage/{folder}/{filename} (api/routes/storage.py), not a raw file_path.
                file_path = storage_service.store_local(
                    image=image,
                    image_name=filename,
                    storage_name="uploaded_images",
                )

                if file_path:
                    logger.info(
                        "Saved bytes %s (%s format) to local storage: %s", field_name, final_format, file_path
                    )
                    return file_path
                logger.error("Failed to save bytes %s", field_name)
                raise RuntimeError(
                    f"Failed to convert field '{field_name}': could not persist image bytes to storage"
                )

            return value

        except ValueError:
            raise
        except Exception as e:
            logger.error("Error converting image field %s: %s", field_name, str(e))
            raise

    def process_dict(data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively process dictionary for image fields"""
        processed = {}
        for key, value in data.items():
            if isinstance(value, dict):
                processed[key] = process_dict(value)
            elif isinstance(value, list):
                processed[key] = [
                    process_dict(item) if isinstance(item, dict) else convert_image_field(item, key)
                    for item in value
                ]
            elif value is None:
                processed[key] = value
            else:
                processed[key] = convert_image_field(value, key)
        return processed

    # Process the entire request data recursively
    processed_data = process_dict(processed_data)

    return processed_data
