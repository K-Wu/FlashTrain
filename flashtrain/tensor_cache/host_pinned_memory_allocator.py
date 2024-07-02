import torch
import math
import threading
from typing import Optional, Sequence
from ..logger import logger, get_oneline_str
from abc import ABCMeta, abstractmethod


class MemoryAllocatorBase(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, size, dtype: torch.dtype = torch.float16):
        raise NotImplementedError

    @abstractmethod
    def allocate_tensor(
        self, size_tuple: Sequence[int], dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def release_tensor(self, tensor: torch.Tensor):
        raise NotImplementedError

    def _get_size_tuple_for_alloc(
        self, size_tuple: Sequence[int], dtype: Optional[torch.dtype]
    ) -> Sequence[int]:
        if dtype is None:
            return size_tuple
        else:
            return (
                *size_tuple[:-1],
                size_tuple[-1] * dtype.itemsize // self.dtype.itemsize,
            )

    def _calc_size(
        self, size_tuple: Sequence[int], dtype: Optional[torch.dtype]
    ):
        return math.prod(self._get_size_tuple_for_alloc(size_tuple, dtype))


class PeakMemoryTracker(MemoryAllocatorBase):
    def __init__(self, size, dtype: torch.dtype = torch.float16):
        self.dtype = dtype
        self.peak_memory = 0
        self.current_memory = 0
        self.lock = threading.Lock()

    def allocate_tensor(
        self, size_tuple: Sequence[int], dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        size = self._calc_size(size_tuple, dtype)
        self.current_memory += size
        if self.current_memory > self.peak_memory:
            self.peak_memory = self.current_memory
        return torch.empty(size_tuple, dtype=dtype).pin_memory()

    def release_tensor(self, tensor: torch.Tensor):
        self.current_memory -= self._calc_size(tensor.shape, tensor.dtype)

    def get_peak_memory(self):
        return self.peak_memory

    def get_current_memory(self):
        return self.current_memory

    def reset(self):
        self.peak_memory = 0
        self.current_memory = 0


class HostPinnedMemoryAllocator(MemoryAllocatorBase):
    """Adapted from deepspeed.runtime.zero.contiguous_memory_allocator.ContiguousMemoryAllocator.
    The parameter functionality is all removed. We also add a threading.lock to assure thread safety.
    """

    disable_defragmentation: bool
    lock: threading.Lock
    buffer: torch.Tensor
    dtype: torch.dtype
    contiguous_sizes: dict[int, int]
    tensor_addresses: dict[int, int]
    tensor_sizes: dict[int, int]
    tensor_ids: dict[int, int]
    tensor_map: dict[int, torch.Tensor]
    total_size: int
    total_free: int
    largest_contiguous: int
    max_allocated: int
    count: int

    def __init__(
        self,
        size,
        dtype: torch.dtype = torch.float16,
        disable_defragmentation: bool = True,
    ):
        self.total_size = size
        self.dtype = dtype
        self.disable_defragmentation = disable_defragmentation
        self.lock = threading.Lock()
        self.buffer = torch.zeros(size, dtype=dtype).pin_memory()
        self.init_states()

    def init_states(self):
        # Address to contiguous size available
        self.contiguous_sizes = {}

        self.contiguous_sizes[0] = self.total_size

        # Tensor id to its address
        self.tensor_addresses = {}

        # Tensor address to its size
        self.tensor_sizes = {}

        # Tensor address to ids
        self.tensor_ids = {}

        # Id to tensors
        self.tensor_map = {}

        self.total_free = self.total_size
        self.largest_contiguous = self.total_size
        self.max_allocated = 0

        self.count = 0

    def allocate_tensor(
        self, size_tuple: Sequence[int], dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """
        create a tensor of size from the pre-allocated buffer
        if not enough free space will fail
        if not enough contiguous space, will defragment and allocate
        """
        size = self._calc_size(size_tuple, dtype)
        free_before = self.total_free

        assert (
            size <= self.total_free
        ), "Not enough memory in buffer. Allocation failed"
        if self.largest_contiguous < size:
            if self.disable_defragmentation:
                # TODO: create a new buffer instead
                logger.critical(
                    get_oneline_str(
                        "Allocator state",
                        "address(",
                        len(self.tensor_addresses),
                        ")",
                        self.tensor_addresses,
                        "tensor_id(",
                        len(self.tensor_ids),
                        ")",
                        self.tensor_ids,
                        "contiguous",
                        self.contiguous_sizes,
                    )
                )
                raise RuntimeError(
                    "Not enough contiguous memory to allocate tensor"
                )
            else:
                raise NotImplementedError
                logger.info(
                    "Needs defragmentation to allocate. Before"
                    " Defragmentation:"
                )
                self.print_allocation(resolution=100)
                self._defragment_memory()
                logger.info("After defragmentation:")
                self.print_allocation(resolution=100)

        self.total_free = self.total_free - size

        allocated = self.total_size - self.total_free
        if allocated > self.max_allocated:
            self.max_allocated = allocated

        tensor_address = self._get_new_tensor_address(size)

        ret_tensor = self._get_new_tensor(tensor_address, size_tuple, dtype)
        logger.info(
            f"Free before allocation {free_before}. Allocating {size}. Free"
            f" after allocation {self.total_free}. Max allocated"
            f" {self.max_allocated}"
        )
        assert (
            self.total_free + size == free_before
        ), "Allocation bookkeeping error"

        return ret_tensor

    def release_tensor(self, tensor: torch.Tensor):
        """Deletes the tensor and frees up the underlying buffer"""
        free_before = self.total_free
        tensor_id = id(tensor)
        tensor_size = self._calc_size(tensor.shape, tensor.dtype)
        # logger.critical(get_oneline_str("Allocator state", "address(",len(self.tensor_addresses),")",self.tensor_addresses, "tensor_id(",len(self.tensor_ids),")", self.tensor_ids,"contiguous" ,self.contiguous_sizes))
        self._release_tensor(tensor_id)
        self.total_free += tensor_size
        logger.info(
            f"Free before release {free_before}. Released {tensor_size}."
            f" Total free after {self.total_free}."
        )
        assert (
            self.total_free - tensor_size == free_before
        ), "Release bookkeeping error"

    # def release_tensor_with_id(self, tensor_id):
    #     free_before = self.total_free
    #     assert tensor_id in self.tensor_map.keys(), "Invalid tensor id"
    #     tensor = self.tensor_map[tensor_id]
    #     tensor_size = tensor.numel()
    #     self._release_tensor(tensor_id)
    #     self.total_free += tensor_size
    #     logger.info(
    #         f"Free before release {free_before}. Released {tensor.numel()}. Total free after {self.total_free}.")
    #     assert self.total_free - tensor_size == free_before, "Release bookkeeping error"

    def print_allocation(self, resolution=200):
        """shows the current memory allocation at specified resolution"""
        total_size = self.buffer.numel() * 1.0
        empty = []
        for addr, size in self.contiguous_sizes.items():
            start = int(addr * resolution / total_size)
            end = int((addr + size) * resolution / total_size)
            empty.extend(range(start, end))
        s = ""
        for i in range(resolution):
            s += "." if i in empty else "|"
        logger.info(s)

    def _release_tensor(self, tensor_id):
        assert (
            tensor_id in self.tensor_addresses
        ), f"Tensor id {tensor_id} not found"

        address = self.tensor_addresses[tensor_id]
        contiguous_size = self._calc_size(
            self.tensor_map[tensor_id].shape, self.tensor_map[tensor_id].dtype
        )

        del self.tensor_addresses[tensor_id]
        del self.tensor_ids[address]
        del self.tensor_map[tensor_id]
        del self.tensor_sizes[address]

        self._consolidate_address(address, contiguous_size)
        self.largest_contiguous = self._largest_contiguous()

    def _consolidate_address(self, address: int, contiguous_size: int):
        # Consolidate next buffer
        end_address = address + contiguous_size
        if end_address in self.contiguous_sizes:
            contiguous_size += self.contiguous_sizes[end_address]
            del self.contiguous_sizes[end_address]

        # Consolidate previous buffer
        for addr, size in self.contiguous_sizes.items():
            if addr + size == address:
                del self.contiguous_sizes[addr]
                contiguous_size += size
                address = addr
                break

        self.contiguous_sizes[address] = contiguous_size

    # def _defragment_memory(self):
    #     empty_addresses = sorted(self.contiguous_sizes.keys())
    #     tensor_addresses = sorted(self.tensor_addresses.values())

    #     tensor_index = 0

    #     while tensor_index < len(tensor_addresses):

    #         empty_addr = empty_addresses[0]
    #         empty_size = self.contiguous_sizes[empty_addr]

    #         tensor_addr = tensor_addresses[tensor_index]
    #         tensor_size = self.tensor_sizes[tensor_addr]
    #         tensor_id = self.tensor_ids[tensor_addr]
    #         tensor = self.tensor_map[self.tensor_ids[tensor_addr]]

    #         assert tensor_size == tensor.numel(), \
    #             f"Size mismatch. {tensor_size} is allocated at addr {tensor_addr} but tensor size is {tensor.numel()} "

    #         assert empty_addr != tensor_addr, \
    #             f"Cannot have same empty address {empty_addr} and tensor address {tensor_addr}"

    #         if empty_addr < tensor_addr:

    #             if empty_size >= tensor_size:
    #                 dest_buffer = self.buffer.narrow(0, empty_addr, tensor_size)
    #                 src_buffer = self.buffer.narrow(0, tensor_addr, tensor_size)
    #                 dest_buffer.data.copy_(src_buffer.data)
    #             else:

    #                 #logger.info(f'empty addr : {empty_addr}, empty size {empty_size} tensor addr {tensor_addr} tensor size {tensor_size}')
    #                 src_addr = tensor_addr
    #                 dest_addr = empty_addr
    #                 while src_addr < (tensor_addr + tensor_size):
    #                     copy_size = min(empty_size, tensor_addr + tensor_size - src_addr)

    #                     dest_buffer = self.buffer.narrow(0, dest_addr, copy_size)
    #                     src_buffer = self.buffer.narrow(0, src_addr, copy_size)

    #                     dest_buffer.data.copy_(src_buffer.data)

    #                     src_addr += copy_size
    #                     dest_addr += copy_size

    #             self._replace_old_address_with_new(tensor_id, empty_addr)

    #             tensor_index += 1

    #         else:
    #             tensor_index += 1

    #         empty_addresses = sorted(self.contiguous_sizes.keys())

    # def _replace_old_address_with_new(self, tensor_id: int, new_address: int):

    #     tensor = self.tensor_map[tensor_id]
    #     tensor_size = tensor.numel()
    #     tensor.data = self.buffer.narrow(0, new_address, tensor_size).data

    #     self._release_tensor(tensor_id)
    #     self._mark_as_occupied(new_address, tensor_size)

    #     self.tensor_ids[new_address] = tensor_id
    #     self.tensor_map[tensor_id] = tensor
    #     self.tensor_addresses[tensor_id] = new_address
    #     self.tensor_sizes[new_address] = tensor_size

    def _get_new_tensor_address(self, size: int) -> int:
        tensor_address = None
        for address, contiguous_size in self.contiguous_sizes.items():
            if contiguous_size >= size and (
                tensor_address is None
                or contiguous_size < self.contiguous_sizes[tensor_address]
            ):
                tensor_address = address
        # logger.critical(get_oneline_str("Allocator state", "address(",len(self.tensor_addresses),")",self.tensor_addresses, "tensor_id(",len(self.tensor_ids),")", self.tensor_ids,"contiguous" ,self.contiguous_sizes))
        assert tensor_address is not None, "address cannot be None"
        return tensor_address

    def _get_new_tensor(
        self, address, size_tuple: Sequence[int], dtype: Optional[torch.dtype]
    ) -> torch.Tensor:
        size = self._calc_size(size_tuple, dtype)
        available_contiguous_size = self.contiguous_sizes[address]

        assert size <= available_contiguous_size, (
            f"Tensor size {size} is large than available contiguous size"
            f" {available_contiguous_size}"
        )
        self.count += 1
        new_tensor = self.buffer.narrow(0, address, size)
        if dtype is None:
            new_tensor = new_tensor.view(size_tuple)
        else:
            new_tensor = new_tensor.view(dtype).view(size_tuple)
        tensor_id = id(new_tensor)
        self.tensor_addresses[tensor_id] = address
        self.tensor_sizes[address] = size

        self.tensor_ids[address] = tensor_id
        self.tensor_map[tensor_id] = new_tensor

        self._mark_as_occupied(address, size)

        return new_tensor

    def _largest_contiguous(self) -> int:
        if len(self.contiguous_sizes) > 0:
            return max([size for _, size in self.contiguous_sizes.items()])
        else:
            return 0

    def _mark_as_occupied(self, address: int, size: int):
        available_contiguous_size = self.contiguous_sizes[address]
        del self.contiguous_sizes[address]

        if available_contiguous_size != size:
            self.contiguous_sizes[address + size] = (
                available_contiguous_size - size
            )

        self.largest_contiguous = self._largest_contiguous()
