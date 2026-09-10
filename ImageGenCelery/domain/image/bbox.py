"""Bounding-box math helpers — pure, no infra dependency."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from common_lib.logging_format import module_logger

logger = module_logger(__name__)


def combine_bboxes(list_bboxes):
    result = list_bboxes[0]
    for idx in range(1, len(list_bboxes)):
        x_min = min(result[0], list_bboxes[idx][0])
        y_min = min(result[1], list_bboxes[idx][1])
        x_max = max(result[2], list_bboxes[idx][2])
        y_max = max(result[3], list_bboxes[idx][3])
        result = [x_min, y_min, x_max, y_max]
        
    return result

def get_bounding_box_from_segmentation(mask: np.array) -> Tuple[int, int, int, int]:
    segmentation = np.where(mask == 255)
    if len(segmentation[0]) > 0:
        x_min = int(np.min(segmentation[1]))
        x_max = min(int(np.max(segmentation[1])) + 1, mask.shape[1])
        y_min = int(np.min(segmentation[0]))
        y_max = min(int(np.max(segmentation[0])) + 1, mask.shape[0])
    else:
        x_min, x_max, y_min, y_max = 0,0,0,0

    return (x_min, y_min, x_max, y_max)

def check_valid_bbox(bbox, height, width):
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    return [x1, y1, x2, y2]

def is_bbox_overlap(bbox1: List[int], bbox2: List[int]) -> bool:
    """
    Check if two bounding boxes overlap.
    
    Two bounding boxes overlap if they intersect in both x and y dimensions.
    This function handles bounding boxes in the format [x1, y1, x2, y2].
    
    :param bbox1: First bounding box [x1, y1, x2, y2]
    :param bbox2: Second bounding box [x1, y1, x2, y2]
    :return: True if bounding boxes overlap, False otherwise
    
    Examples:
        >>> is_bbox_overlap([0, 0, 10, 10], [5, 5, 15, 15])
        True  # Boxes overlap in the region [5, 5, 10, 10]
        
        >>> is_bbox_overlap([0, 0, 5, 5], [10, 10, 15, 15])
        False  # Boxes don't touch
        
        >>> is_bbox_overlap([0, 0, 10, 10], [5, 0, 15, 10])
        True  # Boxes overlap in the region [5, 0, 10, 10]
    """
    # Validate inputs
    if bbox1 is None or len(bbox1) < 4 or bbox2 is None or len(bbox2) < 4:
        logger.warning("Invalid bounding box provided for overlap check")
        return False
    
    # Extract coordinates
    x1_min, y1_min, x1_max, y1_max = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    x2_min, y2_min, x2_max, y2_max = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
    
    # Check for overlap in both dimensions
    # Boxes overlap if:
    # - bbox1's left edge is to the left of bbox2's right edge AND
    # - bbox1's right edge is to the right of bbox2's left edge AND
    # - bbox1's top edge is above bbox2's bottom edge AND
    # - bbox1's bottom edge is below bbox2's top edge
    x_overlap = x1_min < x2_max and x1_max > x2_min
    y_overlap = y1_min < y2_max and y1_max > y2_min
    
    return x_overlap and y_overlap

def combine_bounding_boxes(bbox1: List[int], bbox2: List[int]) -> List[int]:
    """
    Combine two bounding boxes into a single bounding box that encompasses both.
    
    This function creates the minimum bounding box that contains both input boxes.
    The result is the union of the two bounding boxes.
    
    :param bbox1: First bounding box [x1, y1, x2, y2]
    :param bbox2: Second bounding box [x1, y1, x2, y2]
    :return: Combined bounding box [x1, y1, x2, y2] that encompasses both input boxes
    
    Examples:
        >>> combine_bounding_boxes([0, 0, 10, 10], [5, 5, 15, 15])
        [0, 0, 15, 15]  # Encompasses both boxes
        
        >>> combine_bounding_boxes([0, 0, 5, 5], [10, 10, 15, 15])
        [0, 0, 15, 15]  # Encompasses both non-overlapping boxes
        
        >>> combine_bounding_boxes([5, 5, 10, 10], [3, 7, 8, 12])
        [3, 5, 10, 12]  # Minimum box containing both
    """
    # Validate inputs
    if bbox1 is None or len(bbox1) < 4:
        if bbox2 is None or len(bbox2) < 4:
            logger.warning("Both bounding boxes are invalid, returning empty bbox")
            return []
        else:
            logger.warning("First bounding box is invalid, returning second bbox")
            return bbox2
    
    if bbox2 is None or len(bbox2) < 4:
        logger.warning("Second bounding box is invalid, returning first bbox")
        return bbox1
    
    # Extract coordinates from both boxes
    x1_min, y1_min, x1_max, y1_max = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    x2_min, y2_min, x2_max, y2_max = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
    
    # Calculate the combined bounding box
    # Take the minimum of the minimums and maximum of the maximums
    combined_x_min = min(x1_min, x2_min)
    combined_y_min = min(y1_min, y2_min)
    combined_x_max = max(x1_max, x2_max)
    combined_y_max = max(y1_max, y2_max)
    
    return [combined_x_min, combined_y_min, combined_x_max, combined_y_max]

def combine_bounding_boxes_list(bbox_list: List[List[int]]) -> List[int]:
    """
    Combine a list of bounding boxes into a single bounding box that encompasses all of them.
    
    This function creates the minimum bounding box that contains all input boxes.
    Invalid bounding boxes in the list are automatically skipped.
    
    :param bbox_list: List of bounding boxes, each in format [x1, y1, x2, y2]
    :return: Combined bounding box [x1, y1, x2, y2] that encompasses all input boxes,
             or empty list [] if no valid bounding boxes are found
    
    Examples:
        >>> combine_bounding_boxes_list([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]])
        [0, 0, 30, 30]  # Encompasses all three boxes
        
        >>> combine_bounding_boxes_list([[10, 10, 20, 20]])
        [10, 10, 20, 20]  # Single box returns itself
        
        >>> combine_bounding_boxes_list([[0, 0, 5, 5], None, [10, 10, 15, 15]])
        [0, 0, 15, 15]  # Skips invalid box
        
        >>> combine_bounding_boxes_list([])
        []  # Empty list returns empty bbox
    """
    # Validate input
    if not bbox_list or len(bbox_list) == 0:
        logger.warning("Empty bounding box list provided")
        return []
    
    # Filter out invalid bounding boxes
    valid_bboxes = [bbox for bbox in bbox_list if bbox is not None and len(bbox) >= 4]
    
    if len(valid_bboxes) == 0:
        logger.warning("No valid bounding boxes found in list")
        return []
    
    # If only one valid bbox, return it
    if len(valid_bboxes) == 1:
        return valid_bboxes[0]
    
    # Initialize with the first valid bounding box
    combined_x_min = valid_bboxes[0][0]
    combined_y_min = valid_bboxes[0][1]
    combined_x_max = valid_bboxes[0][2]
    combined_y_max = valid_bboxes[0][3]
    
    # Iterate through remaining bounding boxes and expand the combined box
    for bbox in valid_bboxes[1:]:
        x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
        
        # Update the combined bounding box
        combined_x_min = min(combined_x_min, x_min)
        combined_y_min = min(combined_y_min, y_min)
        combined_x_max = max(combined_x_max, x_max)
        combined_y_max = max(combined_y_max, y_max)
    
    return [combined_x_min, combined_y_min, combined_x_max, combined_y_max]

def expand_bounding_box(bbox: List[int], top: float, bottom: float, left: float, right: float) -> List[int]:
    """
    Expand bounding box by individual ratios for top, bottom, left, and right sides.
    
    :param bbox: Bounding box [x, y, width, height] or [x1, y1, x2, y2]
    :param top: Ratio to expand top side (e.g., 0.2 means expand by 20% of height upward)
    :param bottom: Ratio to expand bottom side (e.g., 0.3 means expand by 30% of height downward)
    :param left: Ratio to expand left side (e.g., 0.1 means expand by 10% of width leftward)
    :param right: Ratio to expand right side (e.g., 0.15 means expand by 15% of width rightward)
    :return: Expanded bounding box in same format as input
    """
    if bbox is None or len(bbox) < 4:
        return []
    
    # Determine bbox format and extract coordinates
    if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
        # Format: [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        width = x2 - x1
        height = y2 - y1
        is_corner_format = True
    else:
        # Format: [x, y, width, height]
        x1, y1 = bbox[0], bbox[1]
        width, height = bbox[2], bbox[3]
        x2, y2 = x1 + width, y1 + height
        is_corner_format = False
    
    # Calculate expansion amounts for each side
    expand_top = height * top
    expand_bottom = height * bottom
    expand_left = width * left
    expand_right = width * right
    
    # Calculate new coordinates
    new_x1 = int(x1 - expand_left)
    new_y1 = int(y1 - expand_top)
    new_x2 = int(x2 + expand_right)
    new_y2 = int(y2 + expand_bottom)
    
    # Calculate new width and height
    new_width = new_x2 - new_x1
    new_height = new_y2 - new_y1
    
    # Return in same format as input
    if is_corner_format:
        return [new_x1, new_y1, new_x2, new_y2]
    else:
        return [new_x1, new_y1, new_width, new_height]
