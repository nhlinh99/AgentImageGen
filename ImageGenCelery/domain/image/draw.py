"""Image annotation/drawing helpers — pure, no infra dependency."""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from common_lib.logging_format import module_logger

logger = module_logger(__name__)


def draw_text_centered(image, text: str, font_size: int = 40, text_color: tuple = (255, 255, 255),
                       background_color: Optional[tuple] = None, padding: int = 20, line_spacing: int = 5) -> Image.Image:
    """
    Draw text at the center of an image with automatic text wrapping based on image width.
    The text will be wrapped to fit within the image width (with padding) and centered both horizontally and vertically.
    
    :param image: Input image (PIL Image or numpy array)
    :param text: Text string to draw on the image
    :param font_size: Font size in pixels (default: 40)
    :param text_color: Text color as RGB tuple (default: (255, 255, 255) - white)
    :param background_color: Optional background color for text as RGB tuple. If None, no background is drawn (default: None)
    :param padding: Padding from image edges in pixels for text wrapping calculation (default: 20)
    :param line_spacing: Additional spacing between lines in pixels (default: 5)
    :return: Image with centered text drawn (PIL Image)
    
    Examples:
        >>> from PIL import Image
        >>> img = Image.new('RGB', (800, 600), color=(0, 0, 0))
        >>> result = draw_text_centered(img, "Hello World\nThis is a long text that will wrap automatically")
        >>> result.save("output.jpg")
        
        >>> # With background color
        >>> result = draw_text_centered(img, "Centered Text", text_color=(0, 0, 0), 
        ...                             background_color=(255, 255, 255))
    """
    try:
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                # Grayscale
                pil_image = Image.fromarray(image, 'L').convert('RGB')
            elif len(image.shape) == 3 and image.shape[2] == 3:
                # RGB
                pil_image = Image.fromarray(image, 'RGB')
            elif len(image.shape) == 3 and image.shape[2] == 4:
                # RGBA
                pil_image = Image.fromarray(image, 'RGBA').convert('RGB')
            else:
                logger.error(f"Unsupported numpy array shape: {image.shape}")
                return Image.fromarray(image) if isinstance(image, np.ndarray) else image
        else:
            pil_image = image.copy() if isinstance(image, Image.Image) else image
        
        # Ensure image is RGB
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Create a drawing context
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load a font, fallback to default if not available
        try:
            # Try to use a default font (may vary by system)
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except (OSError, IOError):
            try:
                # Try alternative common font path
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
            except (OSError, IOError):
                try:
                    # Try Windows font path
                    font = ImageFont.truetype("arial.ttf", font_size)
                except (OSError, IOError):
                    # Fallback to default font
                    font = ImageFont.load_default()
                    logger.warning("Could not load custom font, using default font")
        
        # Get image dimensions
        img_width, img_height = pil_image.size
        
        # Calculate available width for text (image width minus padding on both sides)
        available_width = img_width - (2 * padding)
        
        # Split text into words
        words = text.split()
        
        # Wrap text into lines that fit within available width
        lines = []
        current_line = []
        
        for word in words:
            # Check if adding this word would exceed the width
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            text_width = bbox[2] - bbox[0]
            
            if text_width <= available_width:
                current_line.append(word)
            else:
                # Current line is full, start a new line
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        # Add the last line if it has content
        if current_line:
            lines.append(' '.join(current_line))
        
        # If no lines were created (empty text), return original image
        if not lines:
            return pil_image
        
        # Calculate total text height
        line_heights = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_heights.append(bbox[3] - bbox[1])
        
        total_text_height = sum(line_heights) + (len(lines) - 1) * line_spacing
        
        # Calculate starting Y position to center text vertically
        start_y = (img_height - total_text_height) // 2
        
        # Draw each line centered horizontally
        current_y = start_y
        for i, line in enumerate(lines):
            # Get text bounding box for this line
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Calculate X position to center text horizontally
            text_x = (img_width - text_width) // 2
            
            # Draw background if specified
            if background_color is not None:
                # Calculate background rectangle with some padding
                bg_padding = 5
                bg_x1 = text_x - bg_padding
                bg_y1 = current_y - bg_padding
                bg_x2 = text_x + text_width + bg_padding
                bg_y2 = current_y + text_height + bg_padding
                draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=background_color)
            
            # Draw the text
            draw.text((text_x, current_y), line, fill=text_color, font=font)
            
            # Move to next line
            current_y += text_height + line_spacing
        
        return pil_image
    
    except Exception as e:
        logger.error(f"Error drawing centered text on image: {str(e)}")
        # Return original image on error
        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image)
        else:
            return image

def draw_bounding_box(image, bbox: List[int], color: tuple = (255, 0, 0), 
                     thickness: int = 2, label: str = None, 
                     label_color: tuple = None, label_background: tuple = None,
                     font_size: int = 16) -> np.ndarray:
    """
    Draw a bounding box on an image with optional label.
    
    Supports both PIL Image objects and numpy arrays (RGB format).
    The bounding box format is [x1, y1, x2, y2] where (x1, y1) is top-left and (x2, y2) is bottom-right.
    
    :param image: Input image (PIL Image or numpy array in RGB format)
    :param bbox: Bounding box coordinates [x1, y1, x2, y2]
    :param color: Bounding box color as RGB tuple (default: (255, 0, 0) - red)
    :param thickness: Line thickness in pixels (default: 2)
    :param label: Optional text label to display above the bounding box (default: None)
    :param label_color: Text color for label as RGB tuple (default: white if label_background is None, else black)
    :param label_background: Background color for label as RGB tuple (default: same as bbox color)
    :param font_size: Font size for label text (default: 16)
    :return: Image with bounding box drawn (numpy array in RGB format)
    
    Examples:
        >>> from PIL import Image
        >>> import numpy as np
        >>> # Create a test image
        >>> img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        >>> # Draw a red bounding box
        >>> result = draw_bounding_box(img, [10, 10, 50, 50], color=(255, 0, 0))
        >>> 
        >>> # Draw with label
        >>> result = draw_bounding_box(img, [10, 10, 50, 50], 
        ...                           color=(0, 255, 0), label="Object", thickness=3)
        >>> 
        >>> # Using PIL Image
        >>> img = Image.new('RGB', (200, 200), color=(255, 255, 255))
        >>> result = draw_bounding_box(img, [20, 20, 80, 80], 
        ...                           color=(0, 0, 255), label="Box")
    """
    try:
        # Convert PIL Image to numpy array if needed
        is_pil = isinstance(image, Image.Image)
        if is_pil:
            image_array = np.array(image)
        else:
            image_array = image.copy() if isinstance(image, np.ndarray) else np.array(image)
        
        # Ensure image is numpy array
        if not isinstance(image_array, np.ndarray):
            logger.error("Image must be a PIL Image or numpy array")
            return image_array
        
        # Validate bounding box
        if bbox is None or len(bbox) < 4:
            logger.warning("Invalid bounding box provided, returning original image")
            return image_array
        
        # Extract bounding box coordinates
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        
        # Ensure coordinates are integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Get image dimensions
        if len(image_array.shape) == 2:
            # Grayscale image, convert to RGB
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
            # RGBA image, convert to RGB
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        
        height, width = image_array.shape[:2]
        
        # Clamp bounding box coordinates to image boundaries
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        
        # Ensure x2 > x1 and y2 > y1
        if x2 <= x1 or y2 <= y1:
            logger.warning("Invalid bounding box coordinates (x2 <= x1 or y2 <= y1), returning original image")
            return image_array
        
        # Convert RGB color to BGR for OpenCV (OpenCV uses BGR format)
        bgr_color = (color[2], color[1], color[0])
        
        # Draw the bounding box rectangle
        cv2.rectangle(image_array, (x1, y1), (x2, y2), bgr_color, thickness)
        
        # Draw label if provided
        if label:
            # Determine label colors
            if label_background is None:
                label_bg = bgr_color  # Use same color as bounding box
            else:
                label_bg = (label_background[2], label_background[1], label_background[0])  # Convert to BGR
            
            if label_color is None:
                # Auto-determine text color based on background brightness
                if label_background is None:
                    # Use bounding box color, determine if it's dark or light
                    brightness = sum(color) / 3
                    label_fg = (255, 255, 255) if brightness < 128 else (0, 0, 0)  # White or black
                else:
                    brightness = sum(label_background) / 3
                    label_fg = (255, 255, 255) if brightness < 128 else (0, 0, 0)  # White or black
            else:
                label_fg = (label_color[2], label_color[1], label_color[0])  # Convert to BGR
            
            # Calculate text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = font_size / 20.0  # Scale font size appropriately
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, 1)
            
            # Calculate label position (above the bounding box)
            label_x = x1
            label_y = max(text_height + 5, y1 - 5)
            
            # Ensure label doesn't go outside image
            if label_y < text_height:
                label_y = y2 + text_height + 5  # Place below bounding box instead
            
            # Draw label background rectangle
            label_bg_x1 = label_x - 2
            label_bg_y1 = label_y - text_height - 2
            label_bg_x2 = label_x + text_width + 2
            label_bg_y2 = label_y + baseline + 2
            
            # Ensure background rectangle is within image bounds
            label_bg_x1 = max(0, label_bg_x1)
            label_bg_y1 = max(0, label_bg_y1)
            label_bg_x2 = min(width, label_bg_x2)
            label_bg_y2 = min(height, label_bg_y2)
            
            cv2.rectangle(image_array, (label_bg_x1, label_bg_y1), 
                         (label_bg_x2, label_bg_y2), label_bg, -1)  # -1 for filled rectangle
            
            # Draw label text
            cv2.putText(image_array, label, (label_x, label_y), 
                       font, font_scale, label_fg, 1, cv2.LINE_AA)
        
        return image_array
    
    except Exception as e:
        logger.error(f"Error drawing bounding box on image: {str(e)}")
        # Return original image on error
        if isinstance(image, Image.Image):
            return np.array(image)
        return image if isinstance(image, np.ndarray) else np.array(image)

def draw_multiple_bounding_boxes(image, bboxes: List[List[int]], 
                                 colors: List[tuple] = None,
                                 thickness: int = 2, 
                                 labels: List[str] = None,
                                 label_colors: List[tuple] = None,
                                 label_backgrounds: List[tuple] = None,
                                 font_size: int = 16) -> np.ndarray:
    """
    Draw multiple bounding boxes on an image with optional labels.
    
    Supports both PIL Image objects and numpy arrays (RGB format).
    Each bounding box format is [x1, y1, x2, y2].
    
    :param image: Input image (PIL Image or numpy array in RGB format)
    :param bboxes: List of bounding box coordinates, each as [x1, y1, x2, y2]
    :param colors: List of colors for each bounding box as RGB tuples. 
                   If None, uses different colors for each box (default: None)
    :param thickness: Line thickness in pixels (default: 2)
    :param labels: Optional list of text labels for each bounding box (default: None)
    :param label_colors: Optional list of text colors for labels as RGB tuples (default: None)
    :param label_backgrounds: Optional list of background colors for labels as RGB tuples (default: None)
    :param font_size: Font size for label text (default: 16)
    :return: Image with bounding boxes drawn (numpy array in RGB format)
    
    Examples:
        >>> from PIL import Image
        >>> import numpy as np
        >>> # Create a test image
        >>> img = np.ones((200, 200, 3), dtype=np.uint8) * 255
        >>> # Draw multiple bounding boxes
        >>> bboxes = [[10, 10, 50, 50], [60, 60, 100, 100], [110, 110, 150, 150]]
        >>> labels = ["Box 1", "Box 2", "Box 3"]
        >>> result = draw_multiple_bounding_boxes(img, bboxes, labels=labels)
    """
    try:
        # Convert PIL Image to numpy array if needed
        is_pil = isinstance(image, Image.Image)
        if is_pil:
            image_array = np.array(image)
        else:
            image_array = image.copy() if isinstance(image, np.ndarray) else np.array(image)
        
        # Ensure image is numpy array
        if not isinstance(image_array, np.ndarray):
            logger.error("Image must be a PIL Image or numpy array")
            return image_array
        
        # Validate input
        if not bboxes or len(bboxes) == 0:
            logger.warning("No bounding boxes provided, returning original image")
            return image_array
        
        # Generate default colors if not provided
        if colors is None:
            # Generate distinct colors using a colormap
            num_boxes = len(bboxes)
            color_map = plt.cm.tab20(np.linspace(0, 1, num_boxes))
            colors = [(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)) for c in color_map]
        
        # Ensure colors list matches bboxes length
        if len(colors) < len(bboxes):
            # Repeat last color if not enough colors provided
            colors.extend([colors[-1]] * (len(bboxes) - len(colors)))
        
        # Draw each bounding box
        for i, bbox in enumerate(bboxes):
            color = colors[i] if i < len(colors) else (255, 0, 0)
            label = labels[i] if labels and i < len(labels) else None
            label_color = label_colors[i] if label_colors and i < len(label_colors) else None
            label_bg = label_backgrounds[i] if label_backgrounds and i < len(label_backgrounds) else None
            
            image_array = draw_bounding_box(
                image_array, bbox, color=color, thickness=thickness,
                label=label, label_color=label_color, 
                label_background=label_bg, font_size=font_size
            )
        
        return image_array
    
    except Exception as e:
        logger.error(f"Error drawing multiple bounding boxes on image: {str(e)}")
        # Return original image on error
        if isinstance(image, Image.Image):
            return np.array(image)
        return image if isinstance(image, np.ndarray) else np.array(image)
