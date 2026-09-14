# Recovered and modified for ControlTac; see THIRD_PARTY_NOTICES.md.
import copy
import warnings
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
import torch.distributed as dist
from collections.abc import Iterable
from itertools import repeat
import importlib.util
import logging
import warnings
import importlib_metadata
from packaging import version
from termcolor import colored


def get_same_padding(kernel_size: int | tuple[int, ...]) -> int | tuple[int, ...]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, f"kernel size {kernel_size} should be odd number"
        return kernel_size // 2

def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse

to_2tuple = _ntuple(2)


logger = logging.getLogger(__name__)

_xformers_available = importlib.util.find_spec("xformers") is not None
try:
    if _xformers_available:
        _xformers_version = importlib_metadata.version("xformers")
        _torch_version = importlib_metadata.version("torch")
        if version.Version(_torch_version) < version.Version("1.12"):
            raise ValueError("xformers is installed but requires PyTorch >= 1.12")
        logger.debug(f"Successfully imported xformers version {_xformers_version}")
except importlib_metadata.PackageNotFoundError:
    _xformers_available = False

_triton_modules_available = importlib.util.find_spec("triton") is not None
try:
    if _triton_modules_available:
        _triton_version = importlib_metadata.version("triton")
        if version.Version(_triton_version) < version.Version("3.0.0"):
            raise ValueError("triton is installed but requires Triton >= 3.0.0")
        logger.debug(f"Successfully imported triton version {_triton_version}")
except ImportError:
    _triton_modules_available = False
    warnings.warn("TritonLiteMLA and TritonMBConvPreGLU with `triton` is not available on your platform.")


def is_xformers_available():
    return _xformers_available


def is_triton_module_available():
    return _triton_modules_available

def val2list(x: list or tuple or any, repeat_time=1) -> list:  # type: ignore
    """Repeat `val` for `repeat_time` times and return the list or val if list/tuple."""
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]

def val2tuple(x: list or tuple or any, min_len: int = 1, idx_repeat: int = -1) -> tuple:  # type: ignore
    """Return tuple with min_len by repeating element at idx_repeat."""
    # convert to list first
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)



def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()
