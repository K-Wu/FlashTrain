import kvikio
import cupy
import torch
from ..logger import logger
from .utils import TensorEqID
import concurrent.futures
import threading


class KvikioIOAdapter:
    @classmethod
    def save_tensor(cls, tensor: torch.Tensor, path: str):
        """
        Save the tensor to the file.
        """
        tensor_cupy = cupy.asarray(tensor)
        with kvikio.CuFile(path, "r") as f:
            f.write(tensor_cupy)
        logger.info(f"Kvikio Saved tensor {TensorEqID.from_tensor(tensor)}")

    @classmethod
    def async_save_tensor(
        cls,
        tensor: torch.Tensor,
        path: str,
        tensor_being_stored: dict[TensorEqID, concurrent.futures.Future],
        thread_lock: threading.Lock,
    ):
        cls.save_tensor(tensor, path)
        logger.info(
            "Kvikio Async wrapper saved tensor"
            f" {TensorEqID.from_tensor(tensor)}"
        )
        with thread_lock:
            del tensor_being_stored[TensorEqID.from_tensor(tensor)]

    @classmethod
    def load_tensor(
        cls,
        path: str,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ):
        """
        Load the tensor from the file.
        """
        tensor = torch.empty(shape, dtype=dtype, device=device)
        tensor_cupy = cupy.asarray(tensor)
        with kvikio.CuFile(path, "r") as f:
            f.read(tensor_cupy)
        logger.info(f"Kvikio Loading tensor from path {path}")
        return tensor

    @classmethod
    def async_load_tensor(
        cls,
        path: str,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        tensor_id_to_loaded_tensor: dict[TensorEqID, torch.Tensor],
        tensor_id: TensorEqID,
        tensor_being_loaded: dict[TensorEqID, concurrent.futures.Future],
        thread_lock: threading.Lock,
    ):
        """
        Load the tensor from the file.
        """

        logger.info(f"Kvikio Async wrapper loading tensor from path {path}")
        with thread_lock:
            tensor_id_to_loaded_tensor[tensor_id] = cls.load_tensor(
                path, shape, dtype, device
            )
            del tensor_being_loaded[tensor_id]

    @classmethod
    def clean_up_in_backward(
        cls,
        path: str,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ):
        pass

    # TODO: implement clean_up_when_end to delete files


class TorchMainMemoryIOAdapter:
    cpu_tensor_cache: dict[
        tuple[str, torch.Size, torch.dtype, torch.device], torch.Tensor
    ]

    def __init__(self):
        self.cpu_tensor_cache = {}

    def save_tensor(self, tensor: torch.Tensor, path: str):
        """
        Save the tensor to the file.
        """
        self.cpu_tensor_cache[
            (path, tensor.shape, tensor.dtype, tensor.device)
        ] = tensor.cpu()
        logger.info(f"Kvikio Saved tensor {TensorEqID.from_tensor(tensor)}")

    def async_save_tensor(
        self,
        tensor: torch.Tensor,
        path: str,
        tensor_being_stored: dict[TensorEqID, concurrent.futures.Future],
        thread_lock: threading.Lock,
    ):
        self.save_tensor(tensor, path)
        logger.info(
            "Kvikio Async wrapper saved tensor"
            f" {TensorEqID.from_tensor(tensor)}"
        )
        with thread_lock:
            del tensor_being_stored[TensorEqID.from_tensor(tensor)]

    def load_tensor(
        self,
        path: str,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ):
        """
        Load the tensor from the file.
        """
        tensor = self.cpu_tensor_cache[(path, shape, dtype, device)].to(device)
        logger.info(f"Kvikio Loading tensor from path {path}")
        return tensor

    def async_load_tensor(
        self,
        path: str,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        tensor_id_to_loaded_tensor: dict[TensorEqID, torch.Tensor],
        tensor_id: TensorEqID,
        tensor_being_loaded: dict[TensorEqID, concurrent.futures.Future],
        thread_lock: threading.Lock,
    ):
        """
        Load the tensor from the file.
        """

        logger.info(f"Kvikio Async wrapper loading tensor from path {path}")
        with thread_lock:
            tensor_id_to_loaded_tensor[tensor_id] = self.load_tensor(
                path, shape, dtype, device
            )
            del tensor_being_loaded[tensor_id]

    def clean_up_in_backward(
        self,
        path: str,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.cpu_tensor_cache.pop((path, shape, dtype, device), None)

    # TODO: implement clean_up_when_end which does nothing
