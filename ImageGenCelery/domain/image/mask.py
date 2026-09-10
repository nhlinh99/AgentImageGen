"""Mask manipulation helpers — pure, no infra dependency."""
from __future__ import annotations

from typing import List

import cv2
import numpy as np
from PIL import Image

from common_lib.logging_format import module_logger

logger = module_logger(__name__)


def apply_mask_alpha(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    image: (H, W, 3) uint8 or float32 [0,1]
    mask: (H, W) or (H, W, 1) bool/float in [0,1] or 0/255
    returns RGBA uint8 (H, W, 4)
    """
    # Normalize image to uint8 RGB
    if image.ndim == 2:
        # grayscale -> replicate channels
        image = np.stack([image, image, image], axis=-1)
    if image.shape[2] == 4:
        # drop existing alpha (or could keep/integrate)
        image = image[..., :3]

    if np.issubdtype(image.dtype, np.floating):
        # assume in [0,1]
        image = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    # Normalize mask to single-channel float in [0,1]
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[..., 0]
    if mask.ndim == 3 and mask.shape[2] == 3:
        # if somehow 3-channel mask, convert to single channel (any)
        mask = mask[..., 0]

    # If boolean or float
    if mask.dtype == np.bool_:
        alpha = mask.astype(np.uint8) * 255
    else:
        # assume numeric: scale/clamp if necessary
        if np.issubdtype(mask.dtype, np.floating):
            alpha = (np.clip(mask, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        else:
            alpha = mask.astype(np.uint8)
            # if mask likely 0/1, scale up
            if alpha.max() <= 1:
                alpha = alpha * 255

    # Ensure shapes match
    H, W = image.shape[:2]
    if alpha.shape != (H, W):
        # try resizing/broadcasting would be unsafe — raise so caller fixes upstream
        raise ValueError(f"Mask shape {alpha.shape} doesn't match image {(H,W)}")

    rgba = np.concatenate([image, alpha[..., None]], axis=-1)  # (H, W, 4)
    return rgba

def draw_mask_alpha(image: np.ndarray, mask: np.ndarray):
    new_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    new_image = cv2.addWeighted(image, 0.5, new_mask, 0.5, 0)
    return new_image

def remove_outlier_mask(mask: np.ndarray) -> np.ndarray:
    """
    Remove outlier masks and keep only the mask with largest area using contour detection and convex hull.
    
    :param mask: Input mask image as numpy array (grayscale or binary)
    :return: Cleaned mask with only the largest contour filled
    """
    # Ensure mask is binary (0 or 255)
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)
    
    # Apply threshold to ensure binary mask
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Find all contours in the mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours found, return empty mask
    if len(contours) == 0:
        return np.zeros_like(mask)
    
    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate convex hull of the largest contour
    convex_hull = cv2.convexHull(largest_contour)
    
    # Create a new empty mask
    result_mask = np.zeros_like(mask)
    
    # Fill the convex hull on the result mask
    cv2.fillConvexPoly(result_mask, convex_hull, 255)
    
    # Combine with the original mask to preserve original shape while removing outliers
    combined_mask = cv2.bitwise_and(mask, result_mask).astype(np.uint8)
    return combined_mask

def subtract_mask(mask_1: np.ndarray, mask_2: np.ndarray) -> np.ndarray:
    """
    Subtract mask_2 from mask_1, removing overlapping regions.
    Result = mask_1 - mask_2 (regions in mask_1 that are not in mask_2)
    
    :param mask_1: First mask to subtract (numpy array)
    :param mask_2: Second mask to subtract from (numpy array)
    :return: Result mask after subtraction
    """
    # Ensure both masks are binary (0 or 255)
    if mask_1.dtype != np.uint8:
        mask_1 = (mask_1 * 255).astype(np.uint8)
    if mask_2.dtype != np.uint8:
        mask_2 = (mask_2 * 255).astype(np.uint8)
    
    # Apply threshold to ensure binary masks
    _, binary_mask_1 = cv2.threshold(mask_1, 127, 255, cv2.THRESH_BINARY)
    _, binary_mask_2 = cv2.threshold(mask_2, 127, 255, cv2.THRESH_BINARY)
    
    # Subtract mask_2 from mask_1 using bitwise operations
    # Result will have regions that are in mask_1 but not in mask_2
    result_mask = cv2.subtract(binary_mask_1, binary_mask_2)
    
    return result_mask

def add_mask(mask_1: np.ndarray, mask_2: np.ndarray) -> np.ndarray:
    """
    Combine two masks together using bitwise OR operation.
    Result will have regions that are in either mask_1 or mask_2 or both.
    
    :param mask_1: First mask (numpy array)
    :param mask_2: Second mask (numpy array)
    :return: Combined mask
    """
    # Ensure both masks are binary (0 or 255)
    if mask_1.dtype != np.uint8:
        mask_1 = (mask_1 * 255).astype(np.uint8)
    if mask_2.dtype != np.uint8:
        mask_2 = (mask_2 * 255).astype(np.uint8)
    
    # Apply threshold to ensure binary masks
    _, binary_mask_1 = cv2.threshold(mask_1, 127, 255, cv2.THRESH_BINARY)
    _, binary_mask_2 = cv2.threshold(mask_2, 127, 255, cv2.THRESH_BINARY)
    
    # Combine masks using bitwise OR
    # Result will have white pixels where either mask has white pixels
    result_mask = cv2.bitwise_or(binary_mask_1, binary_mask_2)
    
    return result_mask

def fill_mask(mask: np.ndarray, fill_all: bool = True, use_convex_hull: bool = False) -> np.ndarray:
    """
    Fill the full mask area using contour detection.
    This function finds all contours in the mask and fills them completely,
    ensuring no holes or gaps remain in the mask regions.
    
    :param mask: Input mask image as numpy array (grayscale or binary)
    :param fill_all: If True, fills all contours. If False, only fills the largest contour (default: True)
    :param use_convex_hull: If True, uses convex hull to fill contours. If False, fills original contours (default: False)
    :return: Filled mask with all contour areas completely filled
    """
    # Ensure mask is binary (0 or 255)
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)
    
    # Apply threshold to ensure binary mask
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Find all contours in the mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours found, return empty mask
    if len(contours) == 0:
        return np.zeros_like(mask)
    
    # Create a new empty mask
    result_mask = np.zeros_like(mask)
    
    if fill_all:
        # Fill all contours
        for contour in contours:
            if use_convex_hull:
                # Calculate convex hull for each contour
                convex_hull = cv2.convexHull(contour)
                cv2.fillConvexPoly(result_mask, convex_hull, 255)
            else:
                # Fill the contour directly (including holes)
                cv2.drawContours(result_mask, [contour], -1, 255, thickness=cv2.FILLED)
    else:
        # Only fill the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        if use_convex_hull:
            # Calculate convex hull of the largest contour
            convex_hull = cv2.convexHull(largest_contour)
            cv2.fillConvexPoly(result_mask, convex_hull, 255)
        else:
            # Fill the largest contour directly
            cv2.drawContours(result_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    return result_mask

def get_segment_image(image: np.ndarray, mask: np.ndarray, background_color: tuple = None) -> np.ndarray:
    """
    Extract the masked region from an image by applying the mask.
    Pixels outside the mask will be set to transparent (if RGBA) or background color.
    
    :param image: Input image as numpy array (RGB, RGBA, or grayscale)
    :param mask: Input mask as numpy array (grayscale or binary), white regions will be kept
    :param background_color: Background color for non-masked regions (B, G, R) or (B, G, R, A).
                           If None and image is RGB, uses (0, 0, 0). 
                           If None and image has alpha, uses (0, 0, 0, 0).
    :return: Segmented image with mask applied
    """
    # Ensure mask is binary (0 or 255)
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)
    
    # Apply threshold to ensure binary mask
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Handle different image formats
    if len(image.shape) == 2:
        # Grayscale image
        result = cv2.bitwise_and(image, image, mask=binary_mask)
        
    elif len(image.shape) == 3:
        # Color image (RGB or RGBA)
        num_channels = image.shape[2]
        
        if num_channels == 4:
            # RGBA image - use alpha channel
            result = image.copy()
            
            # Normalize mask to 0-1 range for alpha blending
            mask_normalized = binary_mask.astype(np.float32) / 255.0
            
            # Apply mask to alpha channel
            result[:, :, 3] = (result[:, :, 3] * mask_normalized).astype(np.uint8)
            
            # If background color is specified, apply it to non-masked regions
            if background_color is not None:
                inverse_mask = cv2.bitwise_not(binary_mask)
                inverse_mask_normalized = inverse_mask.astype(np.float32) / 255.0
                
                for c in range(min(len(background_color), 4)):
                    background_contribution = (background_color[c] * inverse_mask_normalized).astype(np.uint8)
                    masked_contribution = cv2.bitwise_and(result[:, :, c], result[:, :, c], mask=binary_mask)
                    result[:, :, c] = cv2.add(masked_contribution, background_contribution)
        else:
            # RGB image
            result = cv2.bitwise_and(image, image, mask=binary_mask)
            
            # If background color is specified, add it to non-masked regions
            if background_color is not None:
                inverse_mask = cv2.bitwise_not(binary_mask)
                background = np.zeros_like(image)
                
                for c in range(min(len(background_color), num_channels)):
                    background[:, :, c] = background_color[c]
                
                background_contribution = cv2.bitwise_and(background, background, mask=inverse_mask)
                result = cv2.add(result, background_contribution)
    else:
        # Unsupported image format
        raise ValueError(f"Unsupported image shape: {image.shape}")
    
    return result

def create_bbox_mask(mask_shape: tuple, bounding_box: List[int]) -> np.ndarray:
    """
    Create a binary mask from a bounding box.
    The mask will be white (255) inside the bounding box and black (0) outside.
    
    :param mask_shape: Shape of the output mask (height, width) or (height, width, channels)
    :param bounding_box: Bounding box coordinates [x1, y1, x2, y2]
    :return: Binary mask with bounding box area filled with white
    """
    # Validate bounding box
    if bounding_box is None or len(bounding_box) < 4:
        logger.warning("Invalid bounding box provided, returning empty mask")
        return np.zeros(mask_shape[:2], dtype=np.uint8)
    
    # Extract bounding box coordinates [x1, y1, x2, y2]
    x1, y1, x2, y2 = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]
    
    # Get mask dimensions
    mask_height, mask_width = mask_shape[:2]
    
    # Ensure bounding box is within mask boundaries
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(mask_width, int(x2))
    y2 = min(mask_height, int(y2))
    
    # Create a new empty mask (all black)
    bbox_mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
    
    # Fill the bounding box region with white
    if x2 > x1 and y2 > y1:
        bbox_mask[y1:y2, x1:x2] = 255
    
    return bbox_mask

def crop_mask_by_bbox(mask: np.ndarray, bounding_box: List[int]) -> np.ndarray:
    """
    Crop the mask to only include the region within the bounding box.
    This creates a bounding box mask and combines it with the input mask using AND operation.
    Regions outside the bounding box will be set to 0 (black).
    
    :param mask: Input mask as numpy array (grayscale or binary)
    :param bounding_box: Bounding box coordinates [x1, y1, x2, y2]
    :return: New mask with only the bounding box area preserved, rest is black
    """
    # Validate inputs
    if bounding_box is None or len(bounding_box) < 4:
        logger.warning("Invalid bounding box provided, returning original mask")
        return mask
    
    # Ensure mask is binary (0 or 255)
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)
    
    # Apply threshold to ensure binary mask
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Create bounding box mask
    bbox_mask = create_bbox_mask(binary_mask.shape, bounding_box)
    
    # Combine masks using AND operation (intersection)
    result_mask = cv2.bitwise_and(binary_mask, bbox_mask)
    
    return result_mask

def distort_mask(mask: np.ndarray, 
                 distortion_type: str = "elastic",
                 strength: float = 10.0,
                 kernel_size: int = 5,
                 alpha: float = 50.0,
                 sigma: float = 5.0,
                 seed: int = None) -> np.ndarray:
    """
    Apply various types of distortion to a mask for more natural/organic appearance.
    
    This function supports multiple distortion methods including elastic deformation,
    morphological operations, and random warping.
    
    :param mask: Input mask as numpy array (grayscale or binary)
    :param distortion_type: Type of distortion to apply. Options:
                           - "elastic": Elastic deformation using displacement fields
                           - "erode": Morphological erosion to shrink mask
                           - "dilate": Morphological dilation to expand mask
                           - "open": Morphological opening (erosion followed by dilation)
                           - "close": Morphological closing (dilation followed by erosion)
                           - "blur": Gaussian blur followed by threshold
                           - "random_warp": Random perspective/affine transformation
                           - "wave": Sinusoidal wave distortion
    :param strength: Strength/intensity of the distortion (interpretation varies by type)
                    For elastic: displacement strength in pixels (default: 10.0)
                    For morphological: kernel size multiplier (default: 1.0)
                    For blur: blur kernel size (default: 5.0)
                    For wave: amplitude in pixels (default: 10.0)
    :param kernel_size: Kernel size for morphological operations (default: 5)
    :param alpha: Alpha parameter for elastic deformation (default: 50.0)
    :param sigma: Sigma parameter for elastic deformation Gaussian filter (default: 5.0)
    :param seed: Random seed for reproducible distortions (default: None)
    :return: Distorted mask as numpy array
    
    Examples:
        >>> # Elastic deformation
        >>> mask = np.ones((100, 100), dtype=np.uint8) * 255
        >>> distorted = distort_mask(mask, "elastic", strength=15.0)
        
        >>> # Morphological erosion
        >>> distorted = distort_mask(mask, "erode", kernel_size=3)
        
        >>> # Wave distortion
        >>> distorted = distort_mask(mask, "wave", strength=5.0)
    """
    # Ensure mask is binary (0 or 255)
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)
    
    # Apply threshold to ensure binary mask
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Apply distortion based on type
    if distortion_type == "elastic":
        # Elastic deformation using displacement fields
        height, width = binary_mask.shape[:2]
        
        # Generate random displacement fields
        dx = np.random.randn(height, width).astype(np.float32) * strength
        dy = np.random.randn(height, width).astype(np.float32) * strength
        
        # Apply Gaussian filter to smooth the displacement fields
        dx = cv2.GaussianBlur(dx, (0, 0), sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), sigma)
        
        # Scale by alpha
        dx = dx * alpha / strength
        dy = dy * alpha / strength
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        
        # Apply displacement
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        # Remap the mask
        distorted_mask = cv2.remap(binary_mask, map_x, map_y, 
                                   interpolation=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)
        
        # Re-binarize after interpolation
        _, distorted_mask = cv2.threshold(distorted_mask, 127, 255, cv2.THRESH_BINARY)
        
    elif distortion_type == "erode":
        # Morphological erosion
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        iterations = max(1, int(strength))
        distorted_mask = cv2.erode(binary_mask, kernel, iterations=iterations)
        
    elif distortion_type == "dilate":
        # Morphological dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        iterations = max(1, int(strength))
        distorted_mask = cv2.dilate(binary_mask, kernel, iterations=iterations)
        
    elif distortion_type == "open":
        # Morphological opening (removes small objects/noise)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        iterations = max(1, int(strength))
        distorted_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
        
    elif distortion_type == "close":
        # Morphological closing (fills small holes)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        iterations = max(1, int(strength))
        distorted_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        
    elif distortion_type == "blur":
        # Gaussian blur with re-thresholding
        blur_kernel = max(3, int(strength))
        if blur_kernel % 2 == 0:  # Ensure odd kernel size
            blur_kernel += 1
        
        blurred = cv2.GaussianBlur(binary_mask, (blur_kernel, blur_kernel), 0)
        _, distorted_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        
    elif distortion_type == "random_warp":
        # Random affine/perspective transformation
        height, width = binary_mask.shape[:2]
        
        # Define random displacement for perspective transform
        max_displacement = strength
        
        # Source points (corners of the image)
        src_points = np.float32([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ])
        
        # Destination points with random displacement
        dst_points = np.float32([
            [np.random.uniform(0, max_displacement), 
             np.random.uniform(0, max_displacement)],
            [width - 1 - np.random.uniform(0, max_displacement), 
             np.random.uniform(0, max_displacement)],
            [width - 1 - np.random.uniform(0, max_displacement), 
             height - 1 - np.random.uniform(0, max_displacement)],
            [np.random.uniform(0, max_displacement), 
             height - 1 - np.random.uniform(0, max_displacement)]
        ])
        
        # Get perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply transformation
        distorted_mask = cv2.warpPerspective(binary_mask, matrix, (width, height),
                                             flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_REFLECT)
        
        # Re-binarize
        _, distorted_mask = cv2.threshold(distorted_mask, 127, 255, cv2.THRESH_BINARY)
        
    elif distortion_type == "wave":
        # Sinusoidal wave distortion
        height, width = binary_mask.shape[:2]
        
        # Wave parameters
        amplitude = strength
        frequency = 0.05  # Wave frequency
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        
        # Apply sinusoidal displacement
        # Horizontal wave
        map_x = (x + amplitude * np.sin(2 * np.pi * frequency * y)).astype(np.float32)
        # Vertical wave
        map_y = (y + amplitude * np.sin(2 * np.pi * frequency * x)).astype(np.float32)
        
        # Remap the mask
        distorted_mask = cv2.remap(binary_mask, map_x, map_y,
                                   interpolation=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)
        
        # Re-binarize
        _, distorted_mask = cv2.threshold(distorted_mask, 127, 255, cv2.THRESH_BINARY)
        
    else:
        logger.warning(f"Unknown distortion type: {distortion_type}, returning original mask")
        distorted_mask = binary_mask
    
    return (distorted_mask * 255).astype(np.uint8)

def fill_color_by_mask(image, mask, color):
    """
    Fill a specific color into an image where the mask is non-zero.
    
    This function creates a copy of the input image and fills the specified color
    in regions where the mask is non-zero (typically 255 for white mask pixels).
    The original image is not modified.
    
    :param image: Input image (PIL Image or numpy array)
                  Expected shape: (H, W, 3) for RGB or (H, W) for grayscale
    :param mask: Binary mask (PIL Image or numpy array)
                 Expected shape: (H, W) or (H, W, 1)
                 Non-zero values indicate regions to fill
    :param color: Color to fill (tuple of RGB values or single grayscale value)
                  Examples: (255, 0, 0) for red, (100, 150, 200) for custom RGB
    :return: Modified image as numpy array (H, W, 3)
             Returns original image if error occurs
    
    Examples:
        >>> from PIL import Image
        >>> import numpy as np
        >>> # Create a test image
        >>> img = np.ones((100, 100, 3), dtype=np.uint8) * 100
        >>> # Create a circular mask
        >>> mask = np.zeros((100, 100), dtype=np.uint8)
        >>> cv2.circle(mask, (50, 50), 30, 255, -1)
        >>> # Fill red color in the circular region
        >>> result = fill_color_by_mask(img, mask, (255, 0, 0))
        >>> result.shape
        (100, 100, 3)
        
        >>> # Using PIL Images
        >>> img = Image.new('RGB', (100, 100), color=(100, 100, 100))
        >>> mask = Image.new('L', (100, 100), color=0)
        >>> result = fill_color_by_mask(img, mask, (0, 255, 0))
    """
    try:
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image.copy() if isinstance(image, np.ndarray) else np.array(image)
        
        if isinstance(mask, Image.Image):
            mask_array = np.array(mask)
        else:
            mask_array = mask.copy() if isinstance(mask, np.ndarray) else np.array(mask)
        
        # Ensure image is numpy array
        if not isinstance(image_array, np.ndarray):
            logger.error("Image must be a PIL Image or numpy array")
            return image_array
        
        # Ensure mask is numpy array
        if not isinstance(mask_array, np.ndarray):
            logger.error("Mask must be a PIL Image or numpy array")
            return image_array
        
        # Make a copy to avoid modifying the original
        result = image_array.copy()
        
        # Convert grayscale image to RGB if needed
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        elif result.shape[2] == 4:  # RGBA
            result = cv2.cvtColor(result, cv2.COLOR_RGBA2RGB)
        
        # Ensure mask is 2D
        if len(mask_array.shape) == 3:
            mask_array = mask_array[:, :, 0]
        
        # Check dimensions match
        if result.shape[:2] != mask_array.shape[:2]:
            logger.error(f"Image shape {result.shape[:2]} does not match mask shape {mask_array.shape[:2]}")
            return image_array
        
        # Create binary mask (non-zero values)
        binary_mask = mask_array > 0
        
        # Check if mask has any valid pixels
        if not np.any(binary_mask):
            logger.warning("Mask contains no valid pixels, returning original image")
            return result
        
        # Handle color input
        if isinstance(color, (int, float)):
            # Single value provided, convert to RGB
            fill_color = np.array([color, color, color], dtype=np.uint8)
        elif isinstance(color, (list, tuple)):
            if len(color) == 1:
                fill_color = np.array([color[0], color[0], color[0]], dtype=np.uint8)
            elif len(color) == 3:
                fill_color = np.array(color, dtype=np.uint8)
            else:
                logger.error(f"Color must have 1 or 3 values, got {len(color)}")
                return result
        else:
            logger.error(f"Invalid color type: {type(color)}")
            return result
        
        # Fill the color where mask is non-zero
        result[binary_mask] = fill_color
        
        return result
    
    except Exception as e:
        logger.error(f"Error filling color by mask: {str(e)}")
        # Return original image on error
        if isinstance(image, Image.Image):
            return np.array(image)
        return image if isinstance(image, np.ndarray) else np.array(image)
