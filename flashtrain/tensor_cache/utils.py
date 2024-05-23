import torch
import os
from dataclasses import dataclass
from ..logger import logger
import logging
import threading
import contextlib
import time


def get_oneline_str(*args, debug_only: bool = False) -> str:
    # If level higher than DEBUG
    if (not debug_only) or logger.level <= logging.DEBUG:
        reprs = [str(arg).replace("\n", "↵") for arg in args]
    else:
        reprs = [""]
    return " ".join(reprs)


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
        lock: "threading.Lock | None" = None,
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
                    assert hasattr(tensor, "timestamp")
                    data_ptr = tensor.timestamp
                else:
                    if hasattr(tensor, "timestamp"):
                        data_ptr = tensor.timestamp
                    else:
                        tensor.timestamp = time.time()
                        data_ptr = tensor.timestamp
            else:
                data_ptr = id(tensor.data)

        return cls(
            data_ptr=data_ptr,
            dtype=tensor.dtype,
            shape=tuple(tensor.shape),
            stride=tensor.stride(),
            device=tensor.device,
        )

    @classmethod
    def get_from_tensor(
        cls, tensor: torch.Tensor, lock: "threading.Lock | None" = None
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
