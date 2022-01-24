"""utilities for training/planning
Author: Keisuke Okumura
Affiliation: TokyoTech & OSX
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from functools import wraps
from logging import getLogger
from typing import Optional

import numpy as np
import torch

logger = getLogger(__name__)


def set_global_seeds(seed: int = 46) -> None:
    """set global seeds for numpy and pytorch

    Args:
        seed (:obj:`int`, optional): default 46
    """
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.debug(f"random seed: {seed}")


def get_date_str() -> str:
    """generate date string

    Returns:
        str: %Y-%m-%d-%H-%M-%S
    """

    return datetime.now(timezone(timedelta(hours=+9), "JST")).strftime(
        "%Y-%m-%d-%H-%M-%S"
    )


# work in global scope
DEVICE_NAME: Optional[str] = None


def set_device_name(device: str = "cpu") -> None:
    """set device name for pytorch GPU environment

    Args:
        device (:obj:`str`, optional): device name, e.g., cpu, cuda:0
    """

    global DEVICE_NAME
    DEVICE_NAME = device


def get_device_name() -> str:
    """get device name for pytorch GPU environment

    Returns:
        str: device name, e.g., cpu, cuda:0
    """
    global DEVICE_NAME
    if DEVICE_NAME is None:
        DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"
    return DEVICE_NAME


def stop_watch(name: Optional[str] = None, verbose: bool = True):
    """decorator, measure execution time of given function

    Args:
        name (:obj:`Optional[str]`, optional): this value will be printed
        verbose (:obj:`bool`, optional): print additional info
    """

    def wrapper(func):
        @wraps(func)
        def _stop_watch(*args, **kargs):
            start = time.time()
            result = func(*args, **kargs)
            elapsed = time.time() - start
            _name = name if name is not None else func.__name__
            if verbose:
                logger.info(f"{_name}: {elapsed} sec")
            return result, elapsed

        return _stop_watch

    return wrapper
