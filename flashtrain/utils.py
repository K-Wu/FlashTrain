import torch
import os


# TODO: Deduplicate file IO when is_tensor_equal is True
def is_tensor_equal(x: torch.Tensor, y: torch.Tensor) -> bool:
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
