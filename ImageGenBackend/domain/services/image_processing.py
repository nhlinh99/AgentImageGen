"""Pure image-data operations: format sniffing, size, path check, resize rule.

Moved out of infrastructure/image_processing/utils.py — these are stdlib +
PIL/numpy only (no framework/ORM/HTTP/vendor SDK, no swappable backend), so
they belong in domain, not infrastructure. ``convert_image_to_url`` (the one
function in that old file that actually persisted images) stayed out — it
needs StorageService and lives in application/services/image_url_conversion.py.
"""

from __future__ import annotations

from common.constants import MAX_MEGAPIXEL
from common.logging_format import module_logger
from typing import Tuple, Optional
from PIL import Image
import io
import os

logger = module_logger("log_common")


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

        logger.debug("Image size detected: %sx%s", width, height)
        return (width, height)

    except Exception as e:
        logger.error("Error getting image size from bytes: %s", str(e))
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
            logger.debug("Image path does not exist: %s", image_path)
            return False

        # Check if it's a file (not a directory)
        if not os.path.isfile(image_path):
            logger.warning("Path exists but is not a file: %s", image_path)
            return False

        # Optionally verify it's a readable image file
        try:
            with Image.open(image_path) as img:
                img.verify()  # Verify it's a valid image
            logger.debug("Image path exists and is valid: %s", image_path)
            return True
        except Exception as e:
            logger.warning("Path exists but is not a valid image file: %s, error: %s", image_path, str(e))
            return False

    except Exception as e:
        logger.error("Error checking image path existence: %s, error: %s", image_path, str(e))
        return False


def resize_image_custom(image: Image.Image, max_pixels: int = MAX_MEGAPIXEL * 10**6,
                        logger=None) -> Image.Image:
    """
    Resize image with custom logic:
    - If image is >= max_pixels (default 4 megapixels), resize down to max_pixels
    - Small images are left untouched (no upsizing)

    :param image: PIL Image object
    :param max_pixels: Maximum number of pixels (width * height)
    :param logger: Optional logger instance for logging resize events
    :return: Resized PIL Image if needed, otherwise original image
    """
    width, height = image.size
    current_pixels = width * height

    # If image is at/above max_pixels, resize down
    if current_pixels >= max_pixels:
        # Calculate new dimensions maintaining aspect ratio
        scale_factor = (max_pixels / current_pixels) ** 0.5
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Ensure minimum size of 1x1
        new_width = max(1, new_width)
        new_height = max(1, new_height)

        # Log resize event if logger is provided
        if logger:
            logger.info(
                "Downsizing large image from %sx%s (%s pixels) to %sx%s (%s pixels)",
                width, height, current_pixels, new_width, new_height, new_width * new_height,
            )

        # Resize image using LANCZOS resampling for better quality
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_image

    # Image is within acceptable range, return original
    return image
