import cv2
import numpy as np
from threading import Thread
from typing import Any

from common_lib.logging_format import module_logger


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class BaseService:
    def __init__(self, config: Any):
        self.config = config

        self.logger = module_logger(self.__class__.__name__)
        self.logger.info('SERVICE %s IS INITIALIZED', self.__class__.__name__)

    def encode_image(self, image: np.ndarray) -> bytes:
        image_encode = cv2.imencode('.bmp', image)[1].tobytes()
        return image_encode

    def decode_image(self, byte_data: bytes) -> np.ndarray:
        nparr = np.fromstring(byte_data, np.uint8)
        image_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image_decode


class BaseServiceSingleton(BaseService, metaclass=Singleton):
    def __init__(self, config: Any):
        super(BaseServiceSingleton, self).__init__(config)


class BaseThreadSingleton(Thread, metaclass=Singleton):
    def __init__(self):
        self.logger = module_logger(self.__class__.__name__)
        self.logger.info('SERVICE %s IS INITIALIZED', self.__class__.__name__)
