import kvikio
import cupy
import torch
from ..logger import logger, get_oneline_str
from .utils import TensorEqID
import concurrent.futures
import threading
from typing import Any
import os
from abc import ABCMeta, abstractmethod


def create_new_filename(
    identifier: str,  # Used to distinguish tensors among distributed processes.
    tensor: torch.Tensor,
    path: str,
) -> str:
    """
    Create a filename for a new file when storing tensor on the device.
    """
    import random

    # Use TensorEqID instead of id(tensor) because id(tensor) collision may happen when the data pointers of the two different tensors are the same.
    return os.path.join(
        path,
        f"{identifier}_{TensorEqID.from_tensor(tensor)}.pt",
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

    # TODO: pass in a cuda event to wait for the computation to finish
    @abstractmethod
    def save_tensor(
        self, tensor: torch.Tensor, path: str, event: torch.cuda.Event
    ):
        """
        Save the tensor to the file.
        """
        raise NotImplementedError

    # TODO: pass in a cuda event to wait for the computation to finish
    # TODO: return error code
    def async_save_tensor(
        self,
        tensor: torch.Tensor,
        path: str,
        tensor_being_stored: dict[TensorEqID, concurrent.futures.Future],
        thread_lock: threading.Lock,
        event: torch.cuda.Event,
    ):
        self.save_tensor(tensor, path, event)
        logger.debug(
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
        logger.debug(f"Async wrapper loading tensor from path {path}")
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


class TorchBuiltinIOAdapter(AdapterBase):
    path: str
    streams: list[torch.cuda.Stream]
    current_stream_idx: int
    lock: threading.Lock

    def __init__(self, path: str = "/tmp", num_streams: int = 2):
        self.path = path
        self.streams = []
        for _ in range(num_streams):
            self.streams.append(torch.cuda.Stream())
        self.current_stream_idx = 0

    def create_new_filename(
        self,
        identifier: str,  # Used to distinguish tensors among distributed processes.
        tensor: torch.Tensor,
    ):
        """
        Create a filename for a new file when storing tensor on the device.
        """
        return create_new_filename(identifier, tensor, self.path)

    def save_tensor(
        self, tensor: torch.Tensor, path: str, event: torch.cuda.Event
    ):
        """
        Save the tensor to the file.
        """

        tensor_id = TensorEqID.get_from_tensor(tensor)

        with self.lock:
            store_stream = self.streams[self.current_stream_idx]
            self.current_stream_idx = (self.current_stream_idx + 1) % len(
                self.streams
            )

        # Wait until the computation finishes before saving the tensor
        store_stream.wait_event(event)

        # Transfer tensor to CPU first in a non-blocking manner. Otherwise, torch.save will do the transfer in the blocking manner
        with torch.cuda.stream(store_stream):
            tensor = tensor.to("cpu", non_blocking=True)

        # Block until the transfer finishes
        event = torch.cuda.Event()
        event.record(store_stream)
        event.synchronize()

        torch.save(tensor, path, _use_new_zipfile_serialization=False)
        logger.debug(
            "Saved tensor"
            f" {get_oneline_str(tensor, verbose_only=True)} ({tensor_id})"
            f" to {path}"
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
        # We rely on torch.load to determine the device of the tensor as the device it was originally on when saved was serialized into the file as well.
        tensor = torch.load(path)

        with self.lock:
            load_stream = self.streams[self.current_stream_idx]
            self.current_stream_idx = (self.current_stream_idx + 1) % len(
                self.streams
            )

        # The loaded tensor is on CPU. We need to move it to the correct device in a non-blocking manner.
        with torch.cuda.stream(load_stream):
            tensor = tensor.to(device, non_blocking=True)

        event = torch.cuda.Event()
        event.record(load_stream)
        event.synchronize()

        logger.debug(
            f"Loading tensor {get_oneline_str(tensor, verbose_only=True)} from"
            f" path {path}"
        )
        return tensor

    # TODO: implement clean_up_when_end to delete files


class KvikioIOAdapter(AdapterBase):
    path: str
    streams: list[torch.cuda.Stream]
    current_stream_idx: int
    lock: threading.Lock
    is_async: bool

    def __init__(
        self, path: str = "/tmp", num_streams: int = 2, is_async: bool = True
    ):
        self.path = path
        self.is_async = is_async
        self.streams = []
        for _ in range(num_streams):
            self.streams.append(torch.cuda.Stream())
        self.current_stream_idx = 0
        self.lock = threading.Lock()

    def create_new_filename(
        self,
        identifier: str,  # Used to distinguish tensors among distributed processes.
        tensor: torch.Tensor,
    ):
        """
        Create a filename for a new file when storing tensor on the device.
        """
        return create_new_filename(identifier, tensor, self.path)

    def save_tensor(
        self, tensor: torch.Tensor, path: str, event: torch.cuda.Event
    ):
        """
        Save the tensor to the file.
        """
        # tensor_cupy = cupy.asarray(tensor)
        # Issue at https://github.com/cupy/cupy/issues/7144
        tensor_cupy = cupy.from_dlpack(tensor.contiguous().detach())

        with self.lock:
            store_stream = self.streams[self.current_stream_idx]
            self.current_stream_idx = (self.current_stream_idx + 1) % len(
                self.streams
            )

        try:
            with kvikio.CuFile(path, "w") as f:
                # Wait until the computation finishes before saving the tensor
                store_stream.wait_event(event)
                if self.is_async:
                    future = f.raw_write_async(
                        tensor_cupy, store_stream.cuda_stream
                    )
                    future.check_bytes_done()
                else:
                    f.write(tensor_cupy)

        except Exception as e:
            logger.error(f"Error in saving tensor to path {path}: {e}")
        logger.debug(
            "Kvikio Saved tensor"
            f" {get_oneline_str(tensor_cupy, verbose_only=True)} ({TensorEqID.from_tensor(tensor)})"
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

        # Fixing the loading KeyError caused by dtype=torch.bool
        if dtype == torch.bool:
            dtype = torch.uint8
            need_to_convert_to_bool = True
        else:
            need_to_convert_to_bool = False

        tensor = torch.empty(shape, dtype=dtype, device=device)
        # tensor_cupy = cupy.asarray(tensor)

        with self.lock:
            load_stream = self.streams[self.current_stream_idx]
            self.current_stream_idx = (self.current_stream_idx + 1) % len(
                self.streams
            )

        with kvikio.CuFile(path, "r") as f:
            if self.is_async:
                future = f.raw_read_async(tensor, load_stream.cuda_stream)
                future.check_bytes_done()
            else:
                f.read(tensor)
        logger.debug(
            "Kvikio Loading tensor"
            f" {get_oneline_str(tensor, verbose_only=True)} from path {path}"
        )
        if need_to_convert_to_bool:
            tensor = tensor.bool()
        return tensor

    # TODO: implement clean_up_when_end to delete files


class TorchMainMemoryIOAdapter(AdapterBase):
    cpu_tensor_cache: dict[
        tuple[str, torch.Size, torch.dtype, torch.device], torch.Tensor
    ]
    streams: list[torch.cuda.Stream]
    current_stream_idx: int
    lock: threading.Lock

    def __init__(self, num_streams: int = 2):
        self.cpu_tensor_cache = {}
        self.streams = []
        for _ in range(num_streams):
            self.streams.append(torch.cuda.Stream())
        self.current_stream_idx = 0
        self.lock = threading.Lock()

    def create_new_filename(
        self,
        identifier: str,  # Used to distinguish tensors among distributed processes.
        tensor: torch.Tensor,
    ):
        """
        Create a filename for a new file when storing tensor on the device.
        """
        return create_new_filename(identifier, tensor, "/in_memory")

    def save_tensor(
        self, tensor: torch.Tensor, path: str, event: torch.cuda.Event
    ):
        """
        Save the tensor to the file.
        """
        with self.lock:
            store_stream = self.streams[self.current_stream_idx]
            self.current_stream_idx = (self.current_stream_idx + 1) % len(
                self.streams
            )

        # Wait until the computation finishes before saving the tensor
        store_stream.wait_event(event)

        # non_blocking copy uses cudaMemcpyAsync on current stream. ccording to /pytorch/aten/src/ATen/native/cuda/Copy.cu
        # Current stream is stored in thread-local variable and therefore thread-safe.
        with torch.cuda.stream(store_stream):
            # By default, the destination tensor will be in pinned memory: The logic to determine if the memory should be pinned is "pin_out = (non_blocking && self.is_cuda() && options.device().is_cpu() && (options.layout() == c10::kStrided))"
            self.cpu_tensor_cache[
                (path, tensor.shape, tensor.dtype, tensor.device)
            ] = tensor.to("cpu", non_blocking=True)

        # Block until the transfer finishes
        event = torch.cuda.Event()
        event.record(store_stream)
        event.synchronize()

        logger.debug(
            f"Main Memory Saved tensor {TensorEqID.from_tensor(tensor)} to"
            f" {(path, tensor.shape, tensor.dtype, tensor.device)}"
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
        with self.lock:
            load_stream = self.streams[self.current_stream_idx]
            self.current_stream_idx = (self.current_stream_idx + 1) % len(
                self.streams
            )

        # non_blocking copy uses cudaMemcpyAsync on current stream. ccording to /pytorch/aten/src/ATen/native/cuda/Copy.cu
        # Current stream is stored in thread-local variable and therefore thread-safe.
        with torch.cuda.stream(load_stream):
            tensor = self.cpu_tensor_cache[(path, shape, dtype, device)].to(
                device, non_blocking=True
            )

        # Block until the transfer finishes
        event = torch.cuda.Event()
        event.record(load_stream)
        event.synchronize()

        logger.debug(
            f"Main Memory Loading tensor {(path, shape, dtype, device)} from"
            f" path {path}"
        )
        return tensor

    # TODO: implement clean_up_when_end which does nothing


class TorchDummyIOAdapter(AdapterBase):
    """This adapter is for dubugging purpose and aims to do nothing when it is supposed to store/reload tensors. Instead, it just store the reference to the tensor during storing tensors, and return the reference during loading tensors."""

    gpu_tensor_cache: dict[
        tuple[str, torch.Size, torch.dtype, torch.device], torch.Tensor
    ]

    def __init__(self):
        self.gpu_tensor_cache = {}

    def create_new_filename(
        self,
        identifier: str,  # Used to distinguish tensors among distributed processes.
        tensor: torch.Tensor,
    ):
        """
        Create a filename for a new file when storing tensor on the device.
        """
        return create_new_filename(identifier, tensor, "/in_memory")

    def save_tensor(
        self, tensor: torch.Tensor, path: str, event: torch.cuda.Event
    ):
        """
        Save the tensor to the file.
        """
        self.gpu_tensor_cache[
            (path, tensor.shape, tensor.dtype, tensor.device)
        ] = tensor
        logger.debug(
            f"Dummy Saved tensor {TensorEqID.from_tensor(tensor)} to"
            f" {(path, tensor.shape, tensor.dtype, tensor.device)}"
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
        tensor = self.gpu_tensor_cache[(path, shape, dtype, device)]
        logger.debug(
            f"Dummy Loading tensor {(path, shape, dtype, device)} from"
            f" path {path}"
        )
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

    def save_tensor(
        self, tensor: torch.Tensor, path: str, event: torch.cuda.Event
    ):
        """
        Save the tensor to the file.
        """
        # Find the first ":" in the path to get the adapter_id
        separator_position = path.index(":")
        adapter_id = int(path[:separator_position])
        self.adapters[adapter_id].save_tensor(
            tensor, path[separator_position + 1 :], event
        )

    def async_save_tensor(
        self,
        tensor: torch.Tensor,
        path: str,
        tensor_being_stored: dict[TensorEqID, concurrent.futures.Future],
        thread_lock: threading.Lock,
        event: torch.cuda.Event,
    ):
        self.save_tensor(tensor, path, event)
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
