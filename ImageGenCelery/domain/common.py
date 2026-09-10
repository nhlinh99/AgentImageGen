"""Generic (non-image-specific) helpers."""
from __future__ import annotations

from typing import List

import numpy as np


def flatten_list(nested_list: List) -> List:
    """
    Flatten a nested list to a single level list.
    
    Examples:
        >>> flatten_list([[1, 2], 3])
        [1, 2, 3]
        
        >>> flatten_list([[1, [2, 3]], 4, [5, [6, 7]]])
        [1, 2, 3, 4, 5, 6, 7]
        
        >>> flatten_list([1, 2, 3])
        [1, 2, 3]
    
    :param nested_list: List that may contain nested lists
    :return: Flattened list with all elements at the same level
    """
    result = []
    for item in nested_list:
        if isinstance(item, list):
            # Recursively flatten nested lists
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result

def sig(x):
    return 1/(1 + np.exp(-x))
