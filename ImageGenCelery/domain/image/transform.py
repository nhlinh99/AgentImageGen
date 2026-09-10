"""Image transform/composition helpers — pure, no infra dependency."""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from domain.constants import MAX_MEGAPIXEL, STANDARD_MEGAPIXEL
from common_lib.logging_format import module_logger

logger = module_logger(__name__)


def split_image_to_grid(
    image, columns: int, rows: int
) -> List[Image.Image]:
    """
    Split an image into a grid of equal tiles by column and row.

    :param image: Input image (PIL Image, numpy array, or file path string)
    :param columns: Number of columns (horizontal splits)
    :param rows: Number of rows (vertical splits)
    :return: List of PIL Images in row-major order (left-to-right, top-to-bottom).
             Total length is columns * rows.

    Examples:
        >>> img = Image.open("photo.jpg")
        >>> tiles = split_image_to_grid(img, columns=2, rows=2)  # 4 tiles
        >>> len(tiles)
        4
    """
    if columns < 1 or rows < 1:
        raise ValueError("columns and rows must be >= 1")

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        raise ValueError("image must be a PIL Image, numpy array, or file path string")

    if image.mode != "RGB" and image.mode != "RGBA":
        image = image.convert("RGB")

    w, h = image.size
    tile_w = w // columns
    tile_h = h // rows

    if tile_w <= 0 or tile_h <= 0:
        raise ValueError(
            f"Image size {w}x{h} too small for grid {columns}x{rows}. "
            f"Need at least {columns}x{rows} tile dimensions."
        )

    result = []
    for r in range(rows):
        for c in range(columns):
            left = c * tile_w
            top = r * tile_h
            right = left + tile_w
            bottom = top + tile_h
            result.append(image.crop((left, top, right, bottom)))
    return result

def rgba_to_rgb_white(image: "Image.Image | np.ndarray") -> "Image.Image | np.ndarray":
    """
    Convert an RGBA image to RGB by compositing onto a white background.

    - If input is PIL: returns PIL RGB image
    - If input is numpy: returns numpy uint8 RGB array (H, W, 3)
    """
    if isinstance(image, Image.Image):
        if image.mode != "RGBA":
            # If it already has no alpha, just ensure RGB.
            return image.convert("RGB")
        white_bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        composed = Image.alpha_composite(white_bg, image)
        return composed.convert("RGB")

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Unsupported type: {type(image)}")

    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"Expected (H,W,3) or (H,W,4), got {image.shape}")

    if image.shape[2] == 3:
        return image.astype(np.uint8) if image.dtype != np.uint8 else image

    rgba = image
    if np.issubdtype(rgba.dtype, np.floating):
        rgba_f = np.clip(rgba, 0.0, 1.0).astype(np.float32)
        rgb = rgba_f[..., :3]
        a = rgba_f[..., 3:4]
        out = rgb * a + (1.0 - a) * 1.0  # white background
        return (out * 255.0).round().astype(np.uint8)

    rgba_u8 = rgba.astype(np.uint8)
    rgb = rgba_u8[..., :3].astype(np.float32)
    a = (rgba_u8[..., 3:4].astype(np.float32)) / 255.0
    out = rgb * a + (1.0 - a) * 255.0  # white background
    return out.round().astype(np.uint8)

def adjust_contrast(image, contrast_factor: float = 1.0):
    """
    Adjust the contrast of an image.
    
    Supports both PIL Image objects and numpy arrays (RGB format).
    
    :param image: Input image (PIL Image or numpy array in RGB format)
    :param contrast_factor: Contrast adjustment factor.
                          - 1.0 = original image (no change)
                          - < 1.0 = decrease contrast (0.0 = gray)
                          - > 1.0 = increase contrast
                          Typical range: 0.5 to 2.0
    :return: Contrast-adjusted image in the same format as input
    
    Examples:
        >>> # Increase contrast by 50%
        >>> img = Image.open("photo.jpg")
        >>> enhanced = adjust_contrast(img, 1.5)
        
        >>> # Decrease contrast with numpy array
        >>> img_array = np.array(Image.open("photo.jpg"))  # RGB format
        >>> reduced = adjust_contrast(img_array, 0.7)
    """
    is_numpy = isinstance(image, np.ndarray)
    
    # Convert to PIL if numpy array
    if is_numpy:
        # Numpy array is already in RGB format
        if len(image.shape) == 2:
            # Grayscale
            pil_img = Image.fromarray(image)
        elif len(image.shape) == 3 and image.shape[2] in [3, 4]:
            # RGB or RGBA
            pil_img = Image.fromarray(image)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
    else:
        pil_img = image
    
    # Adjust contrast using PIL
    enhancer = ImageEnhance.Contrast(pil_img)
    adjusted = enhancer.enhance(contrast_factor)
    
    # Convert back to numpy if input was numpy
    if is_numpy:
        return np.array(adjusted)
    else:
        return adjusted

def balance_color(image, method: str = "percentile", target_gray: float = 128.0,
                  percentile: float = 1.0):
    """
    Balance the color of an image using simple white balance methods.

    Supports both PIL Image objects and numpy arrays in RGB/RGBA format.

    Args:
        image: Input image (PIL Image or numpy array in RGB/RGBA format)
        method: Algorithm used for balancing. Options:
            - "gray_world": Assumes average color should be neutral gray
            - "max_white": Scales channels so brightest pixel reaches full scale
            - "percentile": Stretch histogram between percentiles per channel
        target_gray: Target mean value for "gray_world" method (default 128.0)
        percentile: Percentile used for "percentile" method (default 1.0)

    Returns:
        Image with balanced colors in the same format as input.
    """

    is_numpy = isinstance(image, np.ndarray)

    if is_numpy:
        img_array = image.copy()
    elif isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        raise ValueError("Unsupported image type. Expected PIL Image or numpy array.")

    if img_array.ndim == 2:
        # Grayscale image, nothing to balance
        return image

    if img_array.ndim != 3 or img_array.shape[2] not in (3, 4):
        raise ValueError(f"Unsupported image shape for color balance: {img_array.shape}")

    has_alpha = img_array.shape[2] == 4
    if has_alpha:
        rgb = img_array[:, :, :3].astype(np.float32)
        alpha = img_array[:, :, 3]
    else:
        rgb = img_array.astype(np.float32)

    eps = 1e-6

    if method == "gray_world":
        channel_means = rgb.reshape(-1, 3).mean(axis=0)
        channel_means = np.maximum(channel_means, eps)
        gray_value = target_gray
        scale = gray_value / channel_means
        balanced = np.clip(rgb * scale, 0, 255)
    elif method == "max_white":
        channel_max = rgb.reshape(-1, 3).max(axis=0)
        channel_max = np.maximum(channel_max, eps)
        scale = 255.0 / channel_max
        balanced = np.clip(rgb * scale, 0, 255)
    elif method == "percentile":
        pct = max(0.0, min(percentile, 50.0))
        lower = np.percentile(rgb, pct, axis=(0, 1))
        upper = np.percentile(rgb, 100.0 - pct, axis=(0, 1))
        scale = 255.0 / np.maximum(upper - lower, eps)
        balanced = np.clip((rgb - lower) * scale, 0, 255)
    else:
        raise ValueError(f"Unsupported color balance method: {method}")

    if has_alpha:
        result = np.concatenate([balanced, alpha[:, :, None]], axis=2).astype(np.uint8)
    else:
        result = balanced.astype(np.uint8)

    if is_numpy:
        return result

    return Image.fromarray(result)

def get_median_color(image, mask):
    """
    Get the median color of an image within the masked region.
    
    This function calculates the median color (RGB) of pixels in the image
    where the mask is non-zero. The median is computed channel-wise.
    
    :param image: Input image (PIL Image, numpy array, or cv2 image)
                  Expected shape: (H, W, 3) for RGB or (H, W) for grayscale
    :param mask: Binary mask (PIL Image or numpy array)
                 Expected shape: (H, W) or (H, W, 1)
                 Non-zero values indicate the region of interest
    :return: Tuple of median color values (R, G, B) as integers
             Returns (0, 0, 0) if no valid pixels are found
    
    Examples:
        >>> from PIL import Image
        >>> import numpy as np
        >>> # Create a simple test image
        >>> img = Image.new('RGB', (100, 100), color=(100, 150, 200))
        >>> mask = np.ones((100, 100), dtype=np.uint8) * 255
        >>> get_median_color(img, mask)
        (100, 150, 200)
        
        >>> # With a partial mask
        >>> mask = np.zeros((100, 100), dtype=np.uint8)
        >>> mask[25:75, 25:75] = 255  # Only center region
        >>> median = get_median_color(img, mask)
    """
    try:
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        
        # Ensure image is numpy array
        if not isinstance(image, np.ndarray):
            logger.error("Image must be a PIL Image or numpy array")
            return (0, 0, 0)
        
        # Ensure mask is numpy array
        if not isinstance(mask, np.ndarray):
            logger.error("Mask must be a PIL Image or numpy array")
            return (0, 0, 0)
        
        # Convert grayscale image to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Ensure mask is 2D
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        # Check dimensions match
        if image.shape[:2] != mask.shape[:2]:
            logger.error(f"Image shape {image.shape[:2]} does not match mask shape {mask.shape[:2]}")
            return (0, 0, 0)
        
        # Create binary mask (non-zero values)
        binary_mask = mask > 0
        
        # Check if mask has any valid pixels
        if not np.any(binary_mask):
            logger.warning("Mask contains no valid pixels")
            return (0, 0, 0)
        
        # Extract pixels where mask is non-zero
        masked_pixels = image[binary_mask]
        
        # Calculate median for each channel
        median_r = int(np.median(masked_pixels[:, 0]))
        median_g = int(np.median(masked_pixels[:, 1]))
        median_b = int(np.median(masked_pixels[:, 2]))
        
        return (median_r, median_g, median_b)
    
    except Exception as e:
        logger.error(f"Error calculating median color: {str(e)}")
        return (0, 0, 0)

def resize_image_custom(image: Image.Image, max_pixels: int = MAX_MEGAPIXEL * 10**6, 
                        standard_pixels = STANDARD_MEGAPIXEL * 10**6, logger=None) -> Image.Image:
    """
    Resize image with custom logic:
    - If image is <= 0.5 megapixels, resize to 0.8 megapixels
    - If image is larger than max_pixels (default 1 megapixel), resize down to max_pixels
    
    :param image: PIL Image object
    :param max_pixels: Maximum number of pixels (width * height)
    :param logger: Optional logger instance for logging resize events
    :return: Resized PIL Image if needed, otherwise original image
    """
    width, height = image.size
    current_pixels = width * height
    
    # If image is <= 0.5 megapixels, resize to 0.8 megapixels
    if current_pixels <= 0.8 * 10**6:  # 0.8 megapixels
        target_pixels = standard_pixels
        # Calculate new dimensions maintaining aspect ratio
        scale_factor = (target_pixels / current_pixels) ** 0.5
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Ensure minimum size of 1x1
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        
        # Log resize event if logger is provided
        if logger:
            logger.info(f"Upsizing small image from {width}x{height} ({current_pixels:,} pixels) to {new_width}x{new_height} ({new_width * new_height:,} pixels)")
        
        # Resize image using LANCZOS resampling for better quality
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_image
    
    # If image is larger than max_pixels, resize down
    if current_pixels > max_pixels:
        # Calculate new dimensions maintaining aspect ratio
        scale_factor = (max_pixels / current_pixels) ** 0.5
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Ensure minimum size of 1x1
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        
        # Log resize event if logger is provided
        if logger:
            logger.info(f"Downsizing large image from {width}x{height} ({current_pixels:,} pixels) to {new_width}x{new_height} ({new_width * new_height:,} pixels)")
        
        # Resize image using LANCZOS resampling for better quality
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_image
    
    # Image is within acceptable range, return original
    return image

def add_border_to_match_base_ratio(
    image: Image.Image,
    width_base: int,
    height_base: int,
    fill_color: Optional[Tuple[int, ...]] = None,
) -> Image.Image:
    """
    Add symmetric border to match a target base aspect ratio (width_base:height_base).

    This function does not stretch or crop the image, it only pads borders.

    :param image: Input PIL Image
    :param width_base: Target ratio width part (must be > 0)
    :param height_base: Target ratio height part (must be > 0)
    :param fill_color: Optional border color. If None, use black (or transparent for RGBA)
    :return: Padded PIL Image with aspect ratio matching width_base:height_base
    """
    if not isinstance(image, Image.Image):
        raise ValueError("image must be a PIL Image")
    if width_base <= 0 or height_base <= 0:
        raise ValueError("width_base and height_base must be > 0")

    src_w, src_h = image.size
    if src_w <= 0 or src_h <= 0:
        raise ValueError("image size must be valid")

    target_ratio = width_base / height_base
    current_ratio = src_w / src_h

    # Already the same ratio -> no border needed
    if current_ratio == target_ratio:
        return image

    if current_ratio > target_ratio:
        # Image is wider than target ratio -> pad top only
        target_h = int(np.ceil(src_w / target_ratio))
        pad_total = max(0, target_h - src_h)
        top = pad_total
        bottom = 0
        left, right = 0, 0
    else:
        # Image is taller than target ratio -> pad left/right
        target_w = int(np.ceil(src_h * target_ratio))
        pad_total = max(0, target_w - src_w)
        left = pad_total // 2
        right = pad_total - left
        top, bottom = 0, 0

    if fill_color is None:
        fill_color = (0, 0, 0, 0) if image.mode == "RGBA" else (0, 0, 0)

    return ImageOps.expand(image, border=(left, top, right, bottom), fill=fill_color)


# Cap for generation size on the image_gen path. Diffusion cost scales with pixel count,
# and the distilled/Lightning checkpoints drift off-distribution well above this.
MAX_GEN_SIZE = 1280
# Latents are packed as `side // vae_scale_factor // 2` (8 * 2) by the Qwen/FLUX
# pipelines, and ComfyUI's empty-latent nodes work in 8px units -- a side that is not a
# multiple of 16 gets floored and the output stops matching the request.
SIZE_GRANULARITY = 16


def fit_within_max_size(
    width: int,
    height: int,
    max_size: int = MAX_GEN_SIZE,
    granularity: int = SIZE_GRANULARITY,
) -> Tuple[int, int]:
    """Shrink (width, height) so neither side exceeds ``max_size``, preserving aspect
    ratio, then snap both sides to a multiple of ``granularity``.

    Only ever shrinks: a size already within the cap is snapped but never enlarged.
    The snap is clamped so rounding cannot push a side back over ``max_size``.
    Callers generate at the returned size and resample to the requested size afterwards.
    """
    longest = max(width, height)
    if longest <= 0:
        raise ValueError(f"Invalid size {width}x{height}")

    scale = min(1.0, max_size / longest)
    cap = (max_size // granularity) * granularity

    def _snap(value: float) -> int:
        snapped = int(round(value * scale / granularity)) * granularity
        return max(granularity, min(snapped, cap))

    return _snap(width), _snap(height)
