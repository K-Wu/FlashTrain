# Adapted from megatron/global_vars.py
from typing import Optional
import ctypes

__GLOBAL_TENSOR_CACHE = None
__GLOBAL_MEMORY_USE_STATS_RECORDER: Optional[dict[str, int]] = None
__GLOBAL_CUFILE_MALLOC_HOOK = None


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, "{} is already initialized.".format(name)


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, "{} is not initialized.".format(name)


def set_tensor_cache(tensor_cache):
    global __GLOBAL_TENSOR_CACHE
    _ensure_var_is_not_initialized(__GLOBAL_TENSOR_CACHE, "tensor cache")
    __GLOBAL_TENSOR_CACHE = tensor_cache


def get_tensor_cache():
    _ensure_var_is_initialized(__GLOBAL_TENSOR_CACHE, "tensor cache")
    return __GLOBAL_TENSOR_CACHE


def get_memory_use_stats_recorder():
    """Return timers."""
    _ensure_var_is_initialized(
        __GLOBAL_MEMORY_USE_STATS_RECORDER, "memory use stats recorder"
    )
    return __GLOBAL_MEMORY_USE_STATS_RECORDER


def init_memory_use_stats_recorder():
    """Initialize timers."""
    global __GLOBAL_MEMORY_USE_STATS_RECORDER
    _ensure_var_is_not_initialized(
        __GLOBAL_MEMORY_USE_STATS_RECORDER, "memory use stats recorder"
    )
    __GLOBAL_MEMORY_USE_STATS_RECORDER = dict()


def init_cufile_malloc_hook():
    global __GLOBAL_CUFILE_MALLOC_HOOK
    _ensure_var_is_not_initialized(
        __GLOBAL_CUFILE_MALLOC_HOOK, "cufile malloc hook"
    )
    __GLOBAL_CUFILE_MALLOC_HOOK = ctypes.CDLL(
        "/home/kunwu2/FlashTrain/flashtrain/malloc_hook/hook.so",
        mode=ctypes.RTLD_GLOBAL,
    )


def get_cufile_malloc_hook():
    _ensure_var_is_initialized(
        __GLOBAL_CUFILE_MALLOC_HOOK, "cufile malloc hook"
    )
    return __GLOBAL_CUFILE_MALLOC_HOOK
