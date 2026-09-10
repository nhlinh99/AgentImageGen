import logging
from datetime import datetime
import pytz


def get_timezone(shorten: bool = False):
    timezone_res = datetime.now(pytz.timezone('Asia/Saigon')).isoformat().replace("+00:00", "Z")
    if shorten:
        timezone_res = shorten_timezone(timezone_res, simplify=False)

    return timezone_res


def shorten_timezone(timezone_str: str, simplify = False) -> str:
    timezone_res_new = timezone_str.replace(":", "_").replace("+", "_").replace("-", "_").replace(" ", "")[:-6]
    if simplify:
        timezone_res_new = timezone_res_new.split("T")[0]

    return timezone_res_new


class Formatter(logging.Formatter):
    """override logging.Formatter to use an aware datetime object"""
    def converter(self, timestamp):
        return datetime.now(pytz.timezone('Asia/Saigon'))

    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            try:
                s = dt.isoformat(timespec='milliseconds')
            except TypeError:
                s = dt.isoformat()
        return s


def module_logger(mod_name):
    """
    To use this, do logger = module_logger(__name__)
    """
    logger = logging.getLogger(mod_name)

    # Only add handler if logger doesn't already have handlers
    # This prevents duplicate logging when the root logger is already configured
    if not logger.handlers and logger.parent != logging.root:
        handler = logging.StreamHandler()
        handler.setFormatter(Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    elif logger.parent == logging.root:
        # If this is a child of root logger, just set level and let root handle output
        logger.setLevel(logging.DEBUG)

    return logger
