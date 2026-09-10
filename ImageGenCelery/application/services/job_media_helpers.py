"""Job/media helpers that need infra (JobService, redis, storage) — not pure domain code."""
from __future__ import annotations

import base64
import io
import os
import uuid
from typing import Any, Dict
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image, ImageOps

from common_lib.logging_format import module_logger
from config.config import Config
from domain.image.io import detect_image_format_from_bytes
from domain.image.transform import resize_image_custom, rgba_to_rgb_white
from infrastructure.cache.redis.utils import delete_keys_with_prefix
from infrastructure.persistence.job.job_service import JobService
from infrastructure.storage.format_utils import infer_image_format
from domain.schema.job_models import Status
from infrastructure.storage import StorageManager

logger = module_logger(__name__)

image_storage = StorageManager(Config())


def check_job_cancelled(job_service: JobService, job_id: str) -> None:
    """
    Check if job is cancelled and raise an error if it is
    
    Args:
        job_service: JobService instance
        job_id: Job identifier
        
    Raises:
        Exception: If job is cancelled
    """
    if not job_id:
        return
        
    try:
        job_doc = job_service._get_job_from_mongodb(job_id)
        if job_doc and job_doc.get("status") == Status.CANCELLED.value:
            cancellation_reason = job_doc.get("cancellation_reason", "Job was cancelled")
            delete_keys_with_prefix("*", job_id)
            raise Exception(f"Job {job_id} has been cancelled: {cancellation_reason}")
    except Exception as e:
        if "cancelled" in str(e).lower():
            raise  # Re-raise cancellation errors
        # For other errors (like job not found), log but don't raise
        print(f"[Signal] Warning: Could not check job cancellation status: {str(e)}")

def convert_image_to_url(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert request_data image (url, numpy, PIL, bytes) into local file URLs
    
    :param request_data: Request data dictionary
    :param logger: Logger instance
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
                _upload_result = image_storage.save_image_remote(
                    image, filename, "uploaded_images",
                    format=infer_image_format(image, filename),
                )
                file_path = (_upload_result.file_url or "") if _upload_result.success else ""
                
                if file_path:
                    logger.info(f"Saved image {field_name} to local storage: {file_path}")
                    return file_path
                logger.error(f"Failed to save image {field_name}")
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
                _upload_result = image_storage.save_image_remote(
                    image, filename, "uploaded_images",
                    format=infer_image_format(image, filename),
                )
                file_path = (_upload_result.file_url or "") if _upload_result.success else ""
                
                if file_path:
                    logger.info(f"Saved numpy array {field_name} to local storage: {file_path}")
                    return file_path
                logger.error(f"Failed to save numpy array {field_name}")
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
                _upload_result = image_storage.save_image_remote(
                    image, filename, "uploaded_images",
                    format=infer_image_format(image, filename),
                )
                file_path = (_upload_result.file_url or "") if _upload_result.success else ""
                
                if file_path:
                    logger.info(f"Saved PIL image {field_name} to local storage: {file_path}")
                    return file_path
                logger.error(f"Failed to save PIL image {field_name}")
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
                            logger.warning(f"Could not determine image format for field {field_name}, defaulting to JPEG")
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
                _upload_result = image_storage.save_image_remote(
                    image, filename, "uploaded_images",
                    format=infer_image_format(image, filename),
                )
                file_path = (_upload_result.file_url or "") if _upload_result.success else ""
                
                if file_path:
                    logger.info(f"Saved bytes {field_name} ({final_format} format) to local storage: {file_path}")
                    return file_path
                logger.error(f"Failed to save bytes {field_name}")
                raise RuntimeError(
                    f"Failed to convert field '{field_name}': could not persist image bytes to storage"
                )
            
            return value
                    
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error converting image field {field_name}: {str(e)}")
            raise
    
    def process_dict(data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively process dictionary for image fields"""
        processed = {}
        for key, value in data.items():
            if isinstance(value, dict):
                processed[key] = process_dict(value)
            elif isinstance(value, list):
                processed[key] = [process_dict(item) if isinstance(item, dict) else convert_image_field(item, key) for item in value]
            elif value is None:
                processed[key] = value
            else:
                processed[key] = convert_image_field(value, key)
        return processed
    
    # Process the entire request data recursively
    processed_data = process_dict(processed_data)
    
    return processed_data

_VIDEO_URL_EXTENSIONS = (".mp4", ".m4v", ".webm", ".mov", ".avi", ".mkv")


def _is_video_url_or_path(s: str) -> bool:
    """True if the string looks like a video URL or filesystem path (by extension)."""
    if not isinstance(s, str) or not s.strip():
        return False
    if s.startswith(("http://", "https://")):
        try:
            path = unquote(urlparse(s).path)
        except Exception:
            path = s
    else:
        path = s
    base = os.path.basename(path).lower()
    if "?" in base:
        base = base.split("?", 1)[0]
    return any(base.endswith(ext) for ext in _VIDEO_URL_EXTENSIONS)

def convert_url_to_numpy(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert request_data image URLs into numpy arrays.

    Strings that look like video URLs or paths (known video file extensions) are left unchanged.

    :param request_data: Request data dictionary
    :return: Updated request data with numpy arrays where applicable
    """

    processed_data = request_data.copy()
    
    def convert_url_field(value, field_name: str):
        """Convert a single URL field to numpy array"""
        try:
            if isinstance(value, str) and _is_video_url_or_path(value):
                return value

            # Handle file path (local URL)
            if isinstance(value, str) and (os.path.exists(value) or value.startswith('/')):
                # Load image from local file path
                image = Image.open(value)
                image = ImageOps.exif_transpose(image)  # Handle EXIF orientation

                # RGBA input: flatten transparent areas onto white rather than
                # dropping alpha and keeping whatever RGB values sit underneath.
                image = rgba_to_rgb_white(image)

                # Convert PIL Image to numpy array
                numpy_array = np.array(image)

                logger.info(f"Converted local file {field_name} to numpy array with shape: {numpy_array.shape}")
                return numpy_array
            
            # Handle HTTP/HTTPS URLs
            elif isinstance(value, str) and (value.startswith('http://') or value.startswith('https://')):
                # Load image from remote URL using storage manager
                storage_result = image_storage.load_image_remote(value, method="pil")
                
                if storage_result.success and storage_result.image:
                    # RGBA input: flatten transparent areas onto white rather than
                    # dropping alpha and keeping whatever RGB values sit underneath.
                    image = rgba_to_rgb_white(storage_result.image)

                    # Convert PIL Image to numpy array
                    numpy_array = np.array(image)

                    logger.info(f"Converted remote URL {field_name} to numpy array with shape: {numpy_array.shape}")
                    return numpy_array
                else:
                    logger.error(f"Failed to load image from URL {field_name}: {storage_result.error}")
                    return value  # Return original value if loading fails
            
            # Handle base64 data URL (fallback)
            elif isinstance(value, str) and value.startswith('data:image'):
                header, encoded = value.split(',', 1)
                image_bytes = base64.b64decode(encoded)
                
                # Convert bytes to PIL Image
                image = Image.open(io.BytesIO(image_bytes))
                image = ImageOps.exif_transpose(image)  # Handle EXIF orientation

                # RGBA input: flatten transparent areas onto white rather than
                # dropping alpha and keeping whatever RGB values sit underneath.
                image = rgba_to_rgb_white(image)

                # Convert PIL Image to numpy array
                numpy_array = np.array(image)

                logger.info(f"Converted base64 data URL {field_name} to numpy array with shape: {numpy_array.shape}")
                return numpy_array
            
            # Return original value if not a URL or image
            return value
            
        except Exception as e:
            logger.error(f"Error converting URL field {field_name} to numpy: {str(e)}")
            return value  # Return original value if conversion fails
    
    def process_dict(data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively process dictionary for URL fields"""
        processed = {}
        for key, value in data.items():
            if isinstance(value, dict):
                processed[key] = process_dict(value)
            elif isinstance(value, list):
                processed[key] = [process_dict(item) if isinstance(item, dict) else convert_url_field(item, key) for item in value]
            else:
                processed[key] = convert_url_field(value, key)
        return processed
    
    # Process the entire request data recursively
    processed_data = process_dict(processed_data)
    
    return processed_data
