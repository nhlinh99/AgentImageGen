import io
import logging
import os
from typing import Optional

import cv2
import numpy as np
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Use existing logging configuration
logger = logging.getLogger(__name__)

# Transient HTTP failures: urllib3 retries (first attempt + retries)
_HTTP_GET_MAX_RETRIES = 3
_HTTP_GET_BACKOFF_FACTOR = 0.5
_HTTP_STATUS_FORCELIST = (429, 500, 502, 503, 504)

_http_session: Optional[requests.Session] = None


def _get_http_session() -> requests.Session:
    """Session with GET retries for connection errors and selected HTTP status codes."""
    global _http_session
    if _http_session is None:
        retry = Retry(
            total=_HTTP_GET_MAX_RETRIES,
            backoff_factor=_HTTP_GET_BACKOFF_FACTOR,
            status_forcelist=_HTTP_STATUS_FORCELIST,
            allowed_methods=frozenset(["GET", "HEAD"]),
        )
        adapter = HTTPAdapter(max_retries=retry)
        session = requests.Session()
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _http_session = session
    return _http_session


def read_image_from_url(url: str, method: str = "opencv") -> Optional[np.ndarray]:
    """
    Read image from URL and convert to numpy array.
    
    Args:
        url: Image URL to read
        method: Method to use ("opencv", "pil", or "requests")
        
    Returns:
        numpy array of the image (BGR format for OpenCV, RGB for PIL)
        None if failed to load
    """
    try:
        if method.lower() == "opencv":
            return read_image_url_opencv(url)
        elif method.lower() == "pil":
            return read_image_url_pil(url)
        elif method.lower() == "requests":
            return read_image_url_requests(url)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'opencv', 'pil', or 'requests'")
    except Exception as e:
        logger.error(f"Failed to read image from URL {url}: {str(e)}")
        return None


def read_image_url_opencv(url: str) -> Optional[np.ndarray]:
    """
    Read image from URL using OpenCV.
    
    Args:
        url: Image URL to read
        
    Returns:
        numpy array in BGR format (OpenCV default)
    """
    try:
        # Download image using requests (retries via _get_http_session)
        response = _get_http_session().get(url, timeout=30)
        response.raise_for_status()
        
        # Convert to numpy array
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        
        # Decode image
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error(f"Failed to decode image from URL: {url}")
            return None
            
        return image
        
    except Exception as e:
        logger.error(f"OpenCV method failed for URL {url}: {str(e)}")
        return None


def read_image_url_pil(url: str) -> Optional[np.ndarray]:
    """
    Read image from URL using PIL (Pillow).
    
    Args:
        url: Image URL to read
        
    Returns:
        numpy array in RGB format
    """
    try:
        # Download image using requests (retries via _get_http_session)
        response = _get_http_session().get(url, timeout=30)
        response.raise_for_status()
        
        # Open image with PIL
        image = Image.open(io.BytesIO(response.content))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        return image_array
        
    except Exception as e:
        logger.error(f"PIL method failed for URL {url}: {str(e)}")
        return None


def read_image_url_requests(url: str) -> Optional[np.ndarray]:
    """
    Read image from URL using requests and convert to numpy array.
    
    Args:
        url: Image URL to read
        
    Returns:
        numpy array in RGB format
    """
    try:
        # Download image (retries via _get_http_session)
        response = _get_http_session().get(url, timeout=30)
        response.raise_for_status()
        
        # Convert to numpy array
        image_array = np.frombuffer(response.content, dtype=np.uint8)
        
        # Decode image using OpenCV
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error(f"Failed to decode image from URL: {url}")
            return None
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image_rgb
        
    except Exception as e:
        logger.error(f"Requests method failed for URL {url}: {str(e)}")
        return None


def read_image_from_path(file_path: str, method: str = "opencv") -> Optional[np.ndarray]:
    """
    Read image from local file path and convert to numpy array.
    
    Args:
        file_path: Local file path to read
        method: Method to use ("opencv" or "pil")
        
    Returns:
        numpy array of the image
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return None
            
        if method.lower() == "opencv":
            image = cv2.imread(file_path)
            if image is None:
                logger.error(f"Failed to read image from path: {file_path}")
                return None
            return image
            
        elif method.lower() == "pil":
            image = Image.open(file_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return np.array(image)
            
        else:
            raise ValueError(f"Unknown method: {method}. Use 'opencv' or 'pil'")
            
    except Exception as e:
        logger.error(f"Failed to read image from path {file_path}: {str(e)}")
        return None


def resize_image(image: np.ndarray, width: int, height: int, 
                interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """
    Resize image to specified dimensions.
    
    Args:
        image: numpy array of the image
        width: target width
        height: target height
        interpolation: interpolation method
        
    Returns:
        resized numpy array
    """
    return cv2.resize(image, (width, height), interpolation=interpolation)


def validate_image_url(url: str) -> bool:
    """
    Validate if URL points to a valid image.
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid image URL, False otherwise
    """
    try:
        response = _get_http_session().get(url, timeout=10)
        if response.status_code == 200:
            content_type = response.headers.get('content-type', '').lower()
            return content_type.startswith('image')
        return False
    except Exception:
        return False


# Example usage and testing functions
def example_usage():
    """
    Example of how to use the image reading functions.
    """
    # Example URLs (replace with actual image URLs)
    test_urls = [
        "https://img.tweb.page/DIrFzSM07Dx5ydjFcMVf97-KuWM_GnCvAF3cr761lw4/rs::::1/g:no/plain/team-ai/image-gen/origin___170x_9d_96_83_9d968316c575873f6e16928b2d7c4c87.jpg",
    ]
    
    print("Testing image reading from URLs...")
    
    for url in test_urls:
        print(f"\nTesting URL: {url}")
        
        # Test different methods
        for method in ["opencv", "pil", "requests"]:
            print(f"  Method: {method}")
            image = read_image_from_url(url, method)
            
            if image is not None:
                print(f"    Success! Image shape: {image.shape}")
                print(f"    Image dtype: {image.dtype}")
                print(f"    Image min/max values: {image.min()}/{image.max()}")
            else:
                print(f"    Failed to read image")


if __name__ == "__main__":
    # Run examples
    example_usage()
