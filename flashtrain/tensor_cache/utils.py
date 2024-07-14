import torch
import os
from dataclasses import dataclass
import threading
import contextlib
import time
from typing import Optional


import torch.multiprocessing as mp
from _thread import start_new_thread
from functools import wraps
import traceback


def thread_wrapped_func(func):
    """
    Adapted from https://github.com/davidmin7/dgl/blob/c96a8b3e91d0a6cdbb8b103fe84b1374e94053f9/examples/pytorch/graphsage/utils.py
    According to https://github.com/pytorch/pytorch/issues/17199, this decorator
    is necessary to make fork() and openmp work together.

    TODO: confirm if this is necessary for MXNet and Tensorflow.  If so, we need
    to standardize worker process creation since our operators are implemented with
    OpenMP.
    Wraps a process entry point to make it work with OpenMP.
    """

    @wraps(func)
    def decorated_function(*args, **kwargs):
        queue = mp.Queue()

        def _queue_result():
            exception, trace, res = None, None, None
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                exception = e
                trace = traceback.format_exc()
            queue.put((res, exception, trace))

        start_new_thread(_queue_result, ())
        result, exception, trace = queue.get()
        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)

    return decorated_function


def del_dict_key_if_exists(d: dict, key: ..., lock: "threading.Lock | None"):
    if lock is None:
        cm = contextlib.nullcontext()
    else:
        cm = lock
    with cm:
        if key in d:
            del d[key]


# TODO: Use SelfDeletingTempFile instead of plain str in tensor_cache's tensor_id_to_filename
class SelfDeletingTempFile:
    # From https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html
    def __init__(self, path: str):
        self.path = path

    def __del__(self):
        os.remove(self.path)


@dataclass(frozen=True)  # Enable hashability
class TensorEqID:  # (dataobject):
    """When PyTorch packs/unpacks tensors to/from computation graph, identical tensors may be wrapped by different Tensor objects to avoid cyclic reference. This class serves to determine if the underlying tensors are identical."""

    data_ptr: float | int | tuple[int, ...]
    dtype: torch.dtype
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    device: torch.device

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        lock: "Optional[threading.Lock]" = None,
        assert_get=False,
        use_timestamp_as_data_ptr=True,
    ):
        # We have to use timestamp or id(tensor) because the underlying storage will be unused when the tensor is released, causing collision if the new tensor has the same shape and stride.
        if lock is not None:
            cm = lock
        else:
            cm = contextlib.nullcontext()
        with cm:
            if use_timestamp_as_data_ptr:
                if assert_get:
                    assert hasattr(tensor.untyped_storage(), "timestamp")
                    data_ptr = tensor.untyped_storage().timestamp
                else:
                    if hasattr(tensor.untyped_storage(), "timestamp"):
                        data_ptr = tensor.untyped_storage().timestamp
                    else:
                        tensor.untyped_storage().timestamp = time.time()
                        data_ptr = tensor.untyped_storage().timestamp
            else:
                data_ptr = id(tensor.data)
        if isinstance(tensor.stride, int):
            stride = (tensor.stride,)
        else:
            stride = tensor.stride()
        return cls(
            data_ptr=data_ptr,
            dtype=tensor.dtype,
            shape=tuple(tensor.shape),
            stride=stride,
            device=tensor.device,
        )

    @classmethod
    def get_from_tensor(
        cls, tensor: torch.Tensor, lock: "Optional[threading.Lock]" = None
    ):
        return cls.from_tensor(tensor, assert_get=True, lock=lock)

    def __str__(self):
        if isinstance(self.data_ptr, tuple):
            data_ptr_str = f"{self.data_ptr[0]:x}.{self.data_ptr[1]}"
        elif isinstance(self.data_ptr, int):
            data_ptr_str = f"{self.data_ptr:x}"
        else:
            data_ptr_str = str(self.data_ptr)
        stride_str = ".".join(map(str, self.stride))
        shape_str = ".".join(map(str, self.shape))
        return (
            f"{data_ptr_str}_{self.dtype}_{shape_str}_{stride_str}_{str(self.device).replace(':', '_')}"
        )

    def __repr__(self):
        return str(self)


# TODO: Deduplicate file IO when is_tensor_equal is True
def is_tensor_equal_ref(x: torch.Tensor, y: torch.Tensor) -> bool:
    """
    When the tensors are packed to computation graph, identical tensors may be wrapped by different Tensor objects to avoid cyclic reference. This function serves to determine if the underlying tensors are identical.
    """
    if x.untyped_storage().data_ptr() != y.untyped_storage().data_ptr():
        return False
    if x.shape != y.shape:
        return False
    if x.stride() != y.stride():
        return False

    assert x.untyped_storage().size() == y.untyped_storage().size()
    return True


def is_tensor_equal(x: torch.Tensor, y: torch.Tensor) -> bool:
    """
    When the tensors are packed to computation graph, identical tensors may be wrapped by different Tensor objects to avoid cyclic reference. This function serves to determine if the underlying tensors are identical.
    """
    result = TensorEqID.from_tensor(x) == TensorEqID.from_tensor(y)
    assert result == is_tensor_equal_ref(x, y)
    return result
