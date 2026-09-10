"""Image byte/format IO helpers — pure, no infra dependency."""
from __future__ import annotations

import io
import os
from typing import Optional, Tuple

import numpy as np
try:
    import torch
except ImportError:
    torch = None
from PIL import Image, ImageOps

from common_lib.logging_format import module_logger
from domain.image.transform import rgba_to_rgb_white

logger = module_logger(__name__)


def detect_image_format_from_bytes(image_bytes: bytes) -> Optional[str]:
    """
    Detect image format from bytes using magic numbers (file signatures).
    
    :param image_bytes: Image data as bytes
    :return: Image format string (PNG, JPEG, WEBP, GIF, BMP, TIFF) or None if unknown
    """
    if not image_bytes or len(image_bytes) < 4:
        return None
    
    # Check magic numbers (file signatures)
    # PNG: 89 50 4E 47 0D 0A 1A 0A
    if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        return 'PNG'
    
    # JPEG: FF D8 FF
    if image_bytes[:3] == b'\xff\xd8\xff':
        return 'JPEG'
    
    # WEBP: RIFF....WEBP
    if image_bytes[:4] == b'RIFF' and len(image_bytes) > 12 and image_bytes[8:12] == b'WEBP':
        return 'WEBP'
    
    # GIF: GIF87a or GIF89a
    if image_bytes[:6] in (b'GIF87a', b'GIF89a'):
        return 'GIF'
    
    # BMP: BM
    if image_bytes[:2] == b'BM':
        return 'BMP'
    
    # TIFF: II* (little-endian) or MM* (big-endian)
    if image_bytes[:4] in (b'II*\x00', b'MM\x00*'):
        return 'TIFF'
    
    return None

def get_image_size_from_bytes(image_bytes: bytes) -> Optional[Tuple[int, int]]:
    """
    Get image size (width, height) from image bytes.
    
    This function opens the image from bytes and returns its dimensions.
    It handles various image formats including PNG, JPEG, WEBP, GIF, BMP, and TIFF.
    
    :param image_bytes: Image data as bytes
    :return: Tuple of (width, height) in pixels, or None if image cannot be opened
    
    Examples:
        >>> image_bytes = open("image.png", "rb").read()
        >>> size = get_image_size_from_bytes(image_bytes)
        >>> width, height = size
        >>> print(f"Image size: {width}x{height}")
    """
    if not image_bytes:
        logger.warning("Empty image bytes provided")
        return None
    
    try:
        # Create BytesIO object and ensure it's at the beginning
        image_bytes_io = io.BytesIO(image_bytes)
        image_bytes_io.seek(0)
        
        # Open image using PIL
        image = Image.open(image_bytes_io)
        
        # Get image size (width, height)
        width, height = image.size
        
        logger.debug(f"Image size detected: {width}x{height}")
        return (width, height)
        
    except Exception as e:
        logger.error(f"Error getting image size from bytes: {str(e)}")
        return None

def check_image_path_exists(image_path: str) -> bool:
    """
    Check if an image file path exists and is a valid file.
    
    This function verifies that the path exists, is a file (not a directory),
    and optionally validates that it's a readable image file.
    
    :param image_path: Path to the image file (can be absolute or relative)
    :return: True if the image path exists and is a valid file, False otherwise
    
    Examples:
        >>> check_image_path_exists("/path/to/image.jpg")
        True
        
        >>> check_image_path_exists("relative/path/image.png")
        True
        
        >>> check_image_path_exists("/nonexistent/image.jpg")
        False
        
        >>> check_image_path_exists("/path/to/directory")
        False
    """
    if not image_path:
        logger.warning("Empty image path provided")
        return False
    
    try:
        # Check if path exists
        if not os.path.exists(image_path):
            logger.debug(f"Image path does not exist: {image_path}")
            return False
        
        # Check if it's a file (not a directory)
        if not os.path.isfile(image_path):
            logger.warning(f"Path exists but is not a file: {image_path}")
            return False
        
        # Optionally verify it's a readable image file
        try:
            with Image.open(image_path) as img:
                img.verify()  # Verify it's a valid image
            logger.debug(f"Image path exists and is valid: {image_path}")
            return True
        except Exception as e:
            logger.warning(f"Path exists but is not a valid image file: {image_path}, error: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Error checking image path existence: {image_path}, error: {str(e)}")
        return False

def convert_numpy_to_torch(image: np.ndarray, device: str = "cpu"):
    image = image.astype(np.float32) / 255.0
    image_torch = torch.from_numpy(image).float().to(device)[None,]
    return image_torch

def convert_pil_to_torch(image: Image.Image, device: str = "cpu"):
    image_np = np.array(image)
    image_torch = convert_numpy_to_torch(image_np, device=device)
    return image_torch

def convert_byte_to_image(image_byte: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_byte))
    image = ImageOps.exif_transpose(image)  # RGB image
    # RGBA input: flatten transparent areas onto white rather than dropping alpha
    # and keeping whatever RGB values sit underneath (often black/garbage).
    image = rgba_to_rgb_white(image)
    return image

def convert_byte_to_image_L(image_byte: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_byte))
    image = ImageOps.exif_transpose(image)  # RGB image
    image = image.convert("L")
    return image

def convert_image_to_byte(image: Image.Image) -> bytes:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()
