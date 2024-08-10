import kvikio
import kvikio.defaults
import cupy
import torch
from ..logger import logger, get_oneline_str
from .utils import TensorEqID
import concurrent.futures
import threading
from typing import Any
import os
from abc import ABCMeta, abstractmethod
from .host_pinned_memory_allocator import (
    HostPinnedMemoryAllocator,
    MemoryAllocatorBase,
    PeakMemoryTracker,
)
from typing import Optional
import time


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
    lock: threading.Lock
    disable_adaptive_keep_passive: bool

    def __init__(self):
        self.lock = threading.Lock()
        self.disable_adaptive_keep_passive = False

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
    def save_tensor(
        self,
        tensor: torch.Tensor,
        path: str,
        event: torch.cuda.Event,
        tensor_id_to_loaded_tensor: dict[TensorEqID, torch.Tensor],
    ):
        """
        Save the tensor to the file.
        """
        raise NotImplementedError

    # TODO: return error code
    def async_save_tensor(
        self,
        tensor: torch.Tensor,
        tensor_id: TensorEqID,
        path: str,
        tensor_id_to_loaded_tensor: dict[TensorEqID, torch.Tensor],
        tensor_being_stored: dict[TensorEqID, concurrent.futures.Future],
        event: torch.cuda.Event,
    ):
        if tensor_id in tensor_id_to_loaded_tensor and (
            not self.disable_adaptive_keep_passive
        ):
            # The entry is already forwarded to the next process. We don't need to save this tensor
            logger.critical(
                f"Async wrapper not saving tensor {tensor_id} to {path}"
            )
            with self.lock:
                if tensor_id in tensor_being_stored:
                    del tensor_being_stored[tensor_id]
            return
        self.save_tensor(tensor, path, event, tensor_id_to_loaded_tensor)
        logger.debug(f"Async wrapper saved tensor {tensor_id}")
        with self.lock:
            if tensor_id in tensor_being_stored:
                del tensor_being_stored[tensor_id]

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

    def serialize(self) -> "AdapterBase":
        """Convert streams to number of streams because torch.cuda.Stream is not serializable. This applies similar to lock"""
        if hasattr(self, "streams"):
            self.streams = len(self.streams)
        if hasattr(self, "lock"):
            self.lock = None
        return self

    def deserialize(self) -> "AdapterBase":
        """Convert number of streams to torch.cuda.Stream. This applies similar to lock"""
        if hasattr(self, "streams"):
            self.streams = [torch.cuda.Stream() for _ in range(self.streams)]
        if hasattr(self, "lock"):
            self.lock = threading.Lock()
        return self

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
    ):
        """
        Load the tensor from the file.
        """
        with self.lock:
            if tensor_id in tensor_id_to_loaded_tensor:
                del tensor_being_loaded[tensor_id]
                return
        logger.debug(f"Async wrapper loading tensor from path {path}")
        loaded = self.load_tensor(path, shape, dtype, device)
        with self.lock:
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
    lock: threading.Lock  # Inherited from AdapterBase
    disable_adaptive_keep_passive: bool  # Inherited from AdapterBase

    def __init__(self, path: str = "/tmp", num_streams: int = 2):
        super().__init__()
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
        self,
        tensor: torch.Tensor,
        path: str,
        event: torch.cuda.Event,
        tensor_id_to_loaded_tensor: dict[TensorEqID, torch.Tensor],
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
        with torch.no_grad():
            with torch.cuda.stream(store_stream):
                tensor = tensor.to("cpu", non_blocking=True)

        # Block until the transfer finishes
        event = torch.cuda.Event()
        event.record(store_stream)
        # Torch event synchronization uses cudaStreamSynchronize() under the hood.
        event.synchronize()

        # TODO: Maybe parquet as the pickle_module will be faster
        # Disable compression to improve speed
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
        with torch.no_grad():
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
    lock: threading.Lock  # Inherited from AdapterBase
    disable_adaptive_keep_passive: bool  # Inherited from AdapterBase
    is_async: bool
    streams: list[torch.cuda.Stream] | None
    current_stream_idx: int | None
    is_being_profiled: bool | None
    start_timestamp: list[float] | None
    end_timestamp: list[float] | None

    def __init__(
        self,
        path: str = "/tmp",
        num_streams: int = 2,
        is_async: bool = True,
    ):
        super().__init__()
        self.path = path
        self.is_async = is_async

        if is_async:
            self.streams = []
            self.current_stream_idx = 0
            self.is_being_profiled = None
            self.start_timestamp = None
            self.end_timestamp = None
        else:
            self.streams = None
            self.current_stream_idx = None
            # TODO: is_being_profiled as a single flag does not support multiple microbatches
            self.is_being_profiled = False
            self.start_timestamp = []
            self.end_timestamp = []

        if (not is_async) and num_streams > 0:
            logger.error(
                "KvikioIOAdapter does not use streams when is_async is False."
                " Ignoring num_streams"
            )
        else:
            for _ in range(num_streams):
                self.streams.append(torch.cuda.Stream())

        kvikio.defaults.set_compat_mode(False)
        assert kvikio.defaults.compat_mode() == False, (
            "Kvikio compat mode is not disabled. Check if you install cuFile"
            " library!"
        )
        if kvikio.defaults.get_num_threads() == 1:
            logger.error(
                "Kvikio is not using multiple threads. This may slow down the"
                " performance."
            )
        # self.lock = threading.Lock()

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
        self,
        tensor: torch.Tensor,
        path: str,
        event: torch.cuda.Event,
        tensor_id_to_loaded_tensor: dict[TensorEqID, torch.Tensor],
    ):
        """
        Save the tensor to the file.
        """
        try:
            # tensor_cupy = cupy.asarray(tensor)
            # Issue at https://github.com/cupy/cupy/issues/7144
            # Do it here to get it done as soon as possible in the main stream, not blocking the IO stream
            tensor = tensor.contiguous()
            logger.info(f"Kvikio Saving tensor to {path}")
            tensor_cupy = cupy.from_dlpack(tensor.detach())
            logger.debug("Kvikio Saving tensor tensor_cupy obtained")

            if self.is_async:
                assert self.streams is not None
                assert self.current_stream_idx is not None
                with self.lock:
                    store_stream = self.streams[self.current_stream_idx]
                    self.current_stream_idx = (
                        self.current_stream_idx + 1
                    ) % len(self.streams)
            else:
                with self.lock:
                    if self.is_being_profiled:
                        self.start_timestamp.append(time.time())

            with kvikio.CuFile(path, "w") as f:
                if self.is_async:
                    # Wait until the computation finishes before saving the tensor
                    store_stream.wait_event(event)
                    future = f.raw_write_async(
                        tensor_cupy, store_stream.cuda_stream
                    )
                    # future.check_bytes_done()
                    # store_stream.synchronize()
                    # Create and wait for event
                    event = torch.cuda.Event()
                    event.record(store_stream)
                    event.synchronize()
                else:
                    future = f.pwrite(
                        tensor_cupy  # , task_size=tensor_cupy.nbytes
                    )
                    future.get()

        except Exception as e:
            logger.critical(f"Error in saving tensor to path {path}: {e}")
        logger.info(
            "Kvikio Saved tensor"
            f" {get_oneline_str(tensor_cupy, verbose_only=True)} ({TensorEqID.from_tensor(tensor)})"
        )
        if not self.is_async:
            with self.lock:
                if self.is_being_profiled:
                    self.end_timestamp.append(time.time())

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
        try:
            # Fixing the loading KeyError caused by dtype=torch.bool
            if dtype == torch.bool:
                dtype = torch.uint8
                need_to_convert_to_bool = True
            else:
                need_to_convert_to_bool = False

            # if dtype == torch.uint8:
            #     tensor = torch.randn(shape, device=device).to(dtype)
            # else:
            #     tensor = torch.randn(shape, dtype=dtype, device=device)

            tensor = torch.zeros(shape, dtype=dtype, device=device)

            # tensor_cupy = cupy.asarray(tensor)

            if self.is_async:
                assert self.streams is not None
                assert self.current_stream_idx is not None
                with self.lock:
                    load_stream = self.streams[self.current_stream_idx]
                    self.current_stream_idx = (
                        self.current_stream_idx + 1
                    ) % len(self.streams)

            with kvikio.CuFile(path, "r+") as f:
                if self.is_async:
                    future = f.raw_read_async(tensor, load_stream.cuda_stream)
                    # future.check_bytes_done()
                    # load_stream.synchronize()
                    # Create and wait for event
                    event = torch.cuda.Event()
                    event.record(load_stream)
                    event.synchronize()
                else:
                    future = f.pread(
                        tensor,
                        # task_size=tensor.numel() * tensor.element_size(),
                    )
                    future.get()
        except Exception as e:
            logger.critical(f"Error in loading tensor from path {path}: {e}")

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
    lock: threading.Lock  # Inherited from AdapterBase
    disable_adaptive_keep_passive: bool  # Inherited from AdapterBase
    use_host_pinned_memory_allocator: bool
    host_pinned_memory_allocator: MemoryAllocatorBase | None

    def __init__(
        self,
        num_streams: int = 2,
        use_host_pinned_memory_allocator: bool = True,
    ):
        super().__init__()
        self.cpu_tensor_cache = {}
        self.streams = []
        for _ in range(num_streams):
            self.streams.append(torch.cuda.Stream(priority=-100))
        self.current_stream_idx = 0
        # self.lock = threading.Lock()
        self.use_host_pinned_memory_allocator = (
            use_host_pinned_memory_allocator
        )

        self.host_pinned_memory_allocator = None
        if self.use_host_pinned_memory_allocator:
            self.host_pinned_memory_allocator = PeakMemoryTracker(0)

    def instantiate_host_pinned_memory_allocator(
        self, memory_size: Optional[int] = None
    ):
        assert self.use_host_pinned_memory_allocator
        assert isinstance(self.host_pinned_memory_allocator, PeakMemoryTracker)
        # logger.critical(get_oneline_str("Peak Memory", self.host_pinned_memory_allocator.peak_memory))
        if memory_size is None:
            memory_size = self.host_pinned_memory_allocator.peak_memory
        self.host_pinned_memory_allocator = HostPinnedMemoryAllocator(
            memory_size
        )

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
        self,
        tensor: torch.Tensor,
        path: str,
        event: torch.cuda.Event,
        tensor_id_to_loaded_tensor: dict[TensorEqID, torch.Tensor],
    ):
        """
        Save the tensor to the file.
        """
        try:
            # Do it here to get it done as soon as possible in the main stream, not blocking the IO stream
            tensor = tensor.contiguous()
            with self.lock:
                store_stream = self.streams[self.current_stream_idx]
                self.current_stream_idx = (self.current_stream_idx + 1) % len(
                    self.streams
                )

            if self.use_host_pinned_memory_allocator:
                with self.host_pinned_memory_allocator.lock:
                    new_tensor = (
                        self.host_pinned_memory_allocator.allocate_tensor(
                            tensor.shape, tensor.dtype
                        )
                    )

                store_stream.wait_event(event)

                with torch.no_grad():
                    with torch.cuda.stream(store_stream):
                        new_tensor.copy_(tensor, non_blocking=True)
            else:
                # Wait until the computation finishes before saving the tensor
                store_stream.wait_event(event)

                # non_blocking copy uses cudaMemcpyAsync on current stream. ccording to /pytorch/aten/src/ATen/native/cuda/Copy.cu
                # Current stream is stored in thread-local variable and therefore thread-safe.
                with torch.no_grad():
                    with torch.cuda.stream(store_stream):
                        # By default, the destination tensor will be in pinned memory: The logic to determine if the memory should be pinned is "pin_out = (non_blocking && self.is_cuda() && options.device().is_cpu() && (options.layout() == c10::kStrided))"
                        # This is because cudaMemcpyAsync requires the destination memory to be pinned memory. Source: https://forums.developer.nvidia.com/t/cudamemcpyasync-device-to-host-need-to-synchronize-before-using-data-on-host/51750/6
                        new_tensor = tensor.to("cpu", non_blocking=True)

            # Block until the transfer finishes
            event = torch.cuda.Event()
            event.record(store_stream)
            event.synchronize()

            self.cpu_tensor_cache[
                (path, tensor.shape, tensor.dtype, tensor.device)
            ] = new_tensor

            logger.info(
                f"Main Memory Saved tensor {TensorEqID.from_tensor(tensor)} to"
                f" {(path, tensor.shape, tensor.dtype, tensor.device)}"
            )

        except Exception as e:
            logger.critical(f"Error in saving tensor to path {path}: {e}")

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
        with torch.no_grad():
            with torch.cuda.stream(load_stream):
                tensor = self.cpu_tensor_cache[
                    (path, shape, dtype, device)
                ].to(device, non_blocking=True)

        # Block until the transfer finishes
        event = torch.cuda.Event()
        event.record(load_stream)
        event.synchronize()

        logger.debug(
            f"Main Memory Loading tensor {(path, shape, dtype, device)} from"
            f" path {path}"
        )
        return tensor

    def clean_up_in_backward(
        self,
        path: str,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ):
        if not (path, shape, dtype, device) in self.cpu_tensor_cache:
            # Avoid error when simultaneously clean up tensors and storing the tensor being cleaned up
            # TODO: add a clean up all in the end of the backward propagation
            return
        if self.use_host_pinned_memory_allocator and isinstance(
            self.host_pinned_memory_allocator, HostPinnedMemoryAllocator
        ):
            with self.host_pinned_memory_allocator.lock:
                self.host_pinned_memory_allocator.release_tensor(
                    self.cpu_tensor_cache[(path, shape, dtype, device)]
                )
        del self.cpu_tensor_cache[(path, shape, dtype, device)]


class PeakTrackNoIOLossyAdapter(AdapterBase):
    """This adapter is for dubugging purpose and aims to do nothing when it is supposed to store/reload tensors. Instead, it just store the reference to the tensor during storing tensors, and return the reference during loading tensors."""

    lock: threading.Lock  # Inherited from AdapterBase
    disable_adaptive_keep_passive: bool  # Inherited from AdapterBase
    peak_tracker: PeakMemoryTracker

    def __init__(self):
        super().__init__()
        self.peak_tracker = PeakMemoryTracker(0)

    def create_new_filename(
        self,
        identifier: str,  # Used to distinguish tensors among distributed processes.
        tensor: torch.Tensor,
    ):
        """
        Create a filename for a new file when storing tensor on the device.
        """
        return create_new_filename(identifier, tensor, "/peak_track_no_io")

    def save_tensor(
        self,
        tensor: torch.Tensor,
        path: str,
        event: torch.cuda.Event,
        tensor_id_to_loaded_tensor: dict[TensorEqID, torch.Tensor],
    ):
        """
        Save the tensor to the file.
        """
        with self.peak_tracker.lock:
            self.peak_tracker._track_allocate_tensor(
                tensor.shape, tensor.dtype
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
        return torch.empty(shape, dtype=dtype, device=device)


class TorchDummyIOAdapter(AdapterBase):
    """This adapter is for dubugging purpose and aims to do nothing when it is supposed to store/reload tensors. Instead, it just store the reference to the tensor during storing tensors, and return the reference during loading tensors."""

    lock: threading.Lock  # Inherited from AdapterBase
    disable_adaptive_keep_passive: bool  # Inherited from AdapterBase
    gpu_tensor_cache: dict[
        tuple[str, torch.Size, torch.dtype, torch.device], torch.Tensor
    ]

    def __init__(self):
        super().__init__()
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
        self,
        tensor: torch.Tensor,
        path: str,
        event: torch.cuda.Event,
        tensor_id_to_loaded_tensor: dict[TensorEqID, torch.Tensor],
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

    def clean_up_in_backward(
        self,
        path: str,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ):
        if not (path, shape, dtype, device) in self.gpu_tensor_cache:
            # Avoid error when simultaneously clean up tensors and storing the tensor being cleaned up
            # TODO: add a clean up all in the end of the backward propagation
            return
        with self.lock:
            del self.gpu_tensor_cache[(path, shape, dtype, device)]


class RevolverIOAdapter(AdapterBase):
    lock: threading.Lock  # Inherited from AdapterBase
    disable_adaptive_keep_passive: bool  # Inherited from AdapterBase
    adapters: list[KvikioIOAdapter | Any]
    storage_adapters_id: int

    def __init__(self, adapters: list[KvikioIOAdapter | Any]):
        super().__init__()
        self.adapters = adapters
        for adapter in self.adapters:
            assert not isinstance(
                adapter, RevolverIOAdapter
            ), "RevolverIOAdapter cannot contain RevolverIOAdapter"
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
        self,
        tensor: torch.Tensor,
        path: str,
        event: torch.cuda.Event,
        tensor_id_to_loaded_tensor: dict[TensorEqID, torch.Tensor],
    ):
        """
        Save the tensor to the file.
        """
        # Find the first ":" in the path to get the adapter_id
        separator_position = path.index(":")
        adapter_id = int(path[:separator_position])
        self.adapters[adapter_id].save_tensor(
            tensor,
            path[separator_position + 1 :],
            event,
            tensor_id_to_loaded_tensor,
        )

    def async_save_tensor(
        self,
        tensor: torch.Tensor,
        tensor_id: TensorEqID,
        path: str,
        tensor_id_to_loaded_tensor: dict[TensorEqID, torch.Tensor],
        tensor_being_stored: dict[TensorEqID, concurrent.futures.Future],
        event: torch.cuda.Event,
    ):
        if tensor_id in tensor_id_to_loaded_tensor and (
            not self.disable_adaptive_keep_passive
        ):
            # The entry is already forwarded to the next process. We don't need to save this tensor
            with self.lock:
                if tensor_id in tensor_being_stored:
                    del tensor_being_stored[tensor_id]
            return
        self.save_tensor(tensor, path, event, tensor_id_to_loaded_tensor)
        with self.lock:
            if tensor_id in tensor_being_stored:
                del tensor_being_stored[tensor_id]

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
    ):
        """
        Load the tensor from the file.
        """
        with self.lock:
            if tensor_id in tensor_id_to_loaded_tensor:
                del tensor_being_loaded[tensor_id]
                return
        loaded = self.load_tensor(path, shape, dtype, device)
        with self.lock:
            tensor_id_to_loaded_tensor[tensor_id] = loaded
            del tensor_being_loaded[tensor_id]

    def serialize(self) -> "RevolverIOAdapter":
        """Convert streams to number of streams because torch.cuda.Stream is not serializable"""
        for adapter in self.adapters:
            adapter.serialize()
        return self

    def deserialize(self) -> "RevolverIOAdapter":
        """Convert number of streams to torch.cuda.Stream"""
        for adapter in self.adapters:
            adapter.deserialize()
        return self

    def clean_up_in_backward(
        self,
        path: str,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ):
        for adapter in self.adapters:
            adapter.clean_up_in_backward(path, shape, dtype, device)
