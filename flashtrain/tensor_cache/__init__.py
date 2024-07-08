# Adapted from megatron/global_vars.py

__GLOBAL_TENSOR_CACHE = None


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
