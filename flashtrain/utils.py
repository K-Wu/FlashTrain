import torch
import os
from recordclass import dataobject


def oneline_print(*args):
    reprs = [str(arg).replace("\n", "↵") for arg in args]
    print(*reprs, flush=True)


# TODO: Use SelfDeletingTempFile instead of plain str in tensor_cache's tensor_id_to_filename
class SelfDeletingTempFile:
    # From https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html
    def __init__(self, path: str):
        self.path = path

    def __del__(self):
        os.remove(self.path)


class TensorEqID(dataobject):
    """When PyTorch packs/unpacks tensors to/from computation graph, identical tensors may be wrapped by different Tensor objects to avoid cyclic reference. This class serves to determine if the underlying tensors are identical."""

    data_ptr: int
    dtype: torch.dtype
    size: int
    stride: tuple[int, ...]

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor):
        return cls(
            data_ptr=tensor.untyped_storage().data_ptr(),
            dtype=tensor.dtype,
            size=tensor.untyped_storage().size(),
            stride=tensor.stride(),
        )

    def __str__(self):
        stride_str = "_".join(map(str, self.stride))
        return f"{self.data_ptr:x}_{self.dtype}_{self.size}_{stride_str}"


# TODO: Deduplicate file IO when is_tensor_equal is True
def is_tensor_equal_ref(x: torch.Tensor, y: torch.Tensor) -> bool:
    """
    When the tensors are packed to computation graph, identical tensors may be wrapped by different Tensor objects to avoid cyclic reference. This function serves to determine if the underlying tensors are identical.
    """
    if x.untyped_storage().data_ptr() != y.untyped_storage().data_ptr():
        return False
    if x.untyped_storage().size() != y.untyped_storage().size():
        return False
    if x.stride() != y.stride():
        return False

    return True


def is_tensor_equal(x: torch.Tensor, y: torch.Tensor) -> bool:
    """
    When the tensors are packed to computation graph, identical tensors may be wrapped by different Tensor objects to avoid cyclic reference. This function serves to determine if the underlying tensors are identical.
    """
    result = TensorEqID.from_tensor(x) == TensorEqID.from_tensor(y)
    assert result == is_tensor_equal_ref(x, y)
    return result
