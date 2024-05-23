import kvikio
import cupy
import torch
from ..logger import logger
from .utils import TensorEqID, get_oneline_str
import concurrent.futures
import threading
from typing import Any
import os
from abc import ABCMeta, abstractmethod


def create_new_filename(
    identifier: str,  # Used to distinguish tensors among distributed processes.
    tensor: torch.Tensor,
    path: str = "/tmp",
) -> str:
    """
    Create a filename for a new file when storing tensor on the device.
    """
    import random

    # Use TensorEqID instead of id(tensor) because id(tensor) collision may happen when the data pointers of the two different tensors are the same.
    return os.path.join(
        path,
        (
            f"{identifier}_{TensorEqID.from_tensor(tensor)}_{str(tensor.device).replace(':', '_')}.pt"
        ),
    )


class AdapterBase(metaclass=ABCMeta):
    @abstractmethod
    def create_new_filename(
        self,
        identifier: str,  # Used to distinguish tensors among distributed processes.
        tensor: torch.Tensor,
    ):
        """
        Create a filename for a new file when storing tensor on the device.
        """
        raise NotImplementedError

    @abstractmethod
    def save_tensor(self, tensor: torch.Tensor, path: str):
        """
        Save the tensor to the file.
        """
        raise NotImplementedError

    # TODO: return error code
    def async_save_tensor(
        self,
        tensor: torch.Tensor,
        path: str,
        tensor_being_stored: dict[TensorEqID, concurrent.futures.Future],
        thread_lock: threading.Lock,
    ):
        self.save_tensor(tensor, path)
        logger.info(
            f"Async wrapper saved tensor {TensorEqID.from_tensor(tensor)}"
        )
        with thread_lock:
            del tensor_being_stored[TensorEqID.from_tensor(tensor)]

    @abstractmethod
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
        raise NotImplementedError

    # TODO: return error code
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
        with thread_lock:
            if tensor_id in tensor_id_to_loaded_tensor:
                del tensor_being_loaded[tensor_id]
                return
        logger.info(f"Async wrapper loading tensor from path {path}")
        loaded = self.load_tensor(path, shape, dtype, device)
        with thread_lock:
            tensor_id_to_loaded_tensor[tensor_id] = loaded
            del tensor_being_loaded[tensor_id]

    # TODO: implement clean_up_when_end to delete files
    def clean_up_in_backward(
        self,
        path: str,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ):
        pass


class KvikioIOAdapter(AdapterBase):
    path: str

    def __init__(self, path: str = "/tmp"):
        self.path = path

    def create_new_filename(
        self,
        identifier: str,  # Used to distinguish tensors among distributed processes.
        tensor: torch.Tensor,
    ):
        """
        Create a filename for a new file when storing tensor on the device.
        """
        return create_new_filename(identifier, tensor, self.path)

    def save_tensor(self, tensor: torch.Tensor, path: str):
        """
        Save the tensor to the file.
        """
        # tensor_cupy = cupy.asarray(tensor)
        # Issue at https://github.com/cupy/cupy/issues/7144
        tensor_cupy = cupy.from_dlpack(tensor.contiguous().detach())
        with kvikio.CuFile(path, "w") as f:
            f.write(tensor_cupy)
        logger.info(
            "Kvikio Saved tensor"
            f" {get_oneline_str(tensor_cupy, True)} ({TensorEqID.from_tensor(tensor)})"
        )

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
        tensor = torch.empty(shape, dtype=dtype, device=device)
        # tensor_cupy = cupy.asarray(tensor)
        with kvikio.CuFile(path, "r") as f:
            f.read(tensor)
        logger.info(
            f"Kvikio Loading tensor {get_oneline_str(tensor, True)} from path"
            f" {path}"
        )
        return tensor

    # TODO: implement clean_up_when_end to delete files


class TorchMainMemoryIOAdapter(AdapterBase):
    cpu_tensor_cache: dict[
        tuple[str, torch.Size, torch.dtype, torch.device], torch.Tensor
    ]

    def __init__(self):
        self.cpu_tensor_cache = {}

    def create_new_filename(
        self,
        identifier: str,  # Used to distinguish tensors among distributed processes.
        tensor: torch.Tensor,
    ):
        """
        Create a filename for a new file when storing tensor on the device.
        """
        return create_new_filename(identifier, tensor, "/in_memory")

    def save_tensor(self, tensor: torch.Tensor, path: str):
        """
        Save the tensor to the file.
        """
        self.cpu_tensor_cache[
            (path, tensor.shape, tensor.dtype, tensor.device)
        ] = tensor.cpu()
        logger.info(f"Kvikio Saved tensor {TensorEqID.from_tensor(tensor)}")

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

    # TODO: implement clean_up_when_end which does nothing


class RevolverIOAdapter(AdapterBase):
    adapters: list[KvikioIOAdapter | Any]
    storage_adapters_id: int

    def __init__(self, adapters: list[KvikioIOAdapter | Any]):
        self.adapters = adapters
        self.storage_adapters_id = 0

    def create_new_filename(
        self,
        identifier: str,  # Used to distinguish tensors among distributed processes.
        tensor: torch.Tensor,
    ):
        """
        Create a filename for a new file when storing tensor on the device.
        """
        new_filename = (
            str(self.storage_adapters_id)
            + ":"
            + self.adapters[self.storage_adapters_id].create_new_filename(
                identifier, tensor
            )
        )
        self.storage_adapters_id = (self.storage_adapters_id + 1) % len(
            self.adapters
        )
        return new_filename

    def save_tensor(self, tensor: torch.Tensor, path: str):
        """
        Save the tensor to the file.
        """
        # Find the first ":" in the path to get the adapter_id
        separator_position = path.index(":")
        adapter_id = int(path[:separator_position])
        self.adapters[adapter_id].save_tensor(
            tensor, path[separator_position + 1 :]
        )

    def async_save_tensor(
        self,
        tensor: torch.Tensor,
        path: str,
        tensor_being_stored: dict[TensorEqID, concurrent.futures.Future],
        thread_lock: threading.Lock,
    ):
        self.save_tensor(tensor, path)
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
        # Find the first ":" in the path to get the adapter_id
        separator_position = path.index(":")
        adapter_id = int(path[:separator_position])
        return self.adapters[adapter_id].load_tensor(
            path[separator_position + 1 :], shape, dtype, device
        )

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
        with thread_lock:
            if tensor_id in tensor_id_to_loaded_tensor:
                del tensor_being_loaded[tensor_id]
                return
        loaded = self.load_tensor(path, shape, dtype, device)
        with thread_lock:
            tensor_id_to_loaded_tensor[tensor_id] = loaded
            del tensor_being_loaded[tensor_id]

    def clean_up_in_backward(
        self,
        path: str,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ):
        for adapter in self.adapters:
            adapter.clean_up_in_backward(path, shape, dtype, device)

    # TODO: implement clean_up_when_end which does nothing
