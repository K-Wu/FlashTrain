# Adapted from https://github.com/microsoft/DeepSpeed/blob/aecfec7f510eb2a40f7afa50efe370f885e4d258/deepspeed/runtime/zero/test.py
from ..tensor_cache.host_pinned_memory_allocator import (
    HostPinnedMemoryAllocator,
)
import torch


def test1(instantiate_enough_capacity=True):
    if instantiate_enough_capacity:
        mem = HostPinnedMemoryAllocator(
            2048, torch.half, disable_defragmentation=True
        )
    else:
        mem = HostPinnedMemoryAllocator(
            1024, torch.half, disable_defragmentation=False
        )
    mem.print_allocation(resolution=100)
    a1 = mem.allocate_tensor((64,)).mul_(0.0).add_(1.0)
    mem.print_allocation(resolution=100)
    mem.release_tensor(a1)
    mem.print_allocation(resolution=100)
    a2 = mem.allocate_tensor((64,)).mul_(0.0).add_(2.0)
    a3 = mem.allocate_tensor((256,)).mul_(0.0).add_(3.0)
    a4 = mem.allocate_tensor((128,)).mul_(0.0).add_(4.0)
    mem.print_allocation(resolution=100)
    mem.release_tensor(a3)
    mem.print_allocation(resolution=100)
    a5 = mem.allocate_tensor((64,)).mul_(0.0).add_(5.0)
    a6 = mem.allocate_tensor((256,)).mul_(0.0).add_(6.0)
    a7 = mem.allocate_tensor((128,)).mul_(0.0).add_(7.0)
    mem.print_allocation(resolution=100)
    a8 = mem.allocate_tensor((256,)).mul_(0.0).add_(8.0)
    a9 = mem.allocate_tensor((128,)).mul_(0.0).add_(9.0)
    mem.print_allocation(resolution=100)
    mem.release_tensor(a9)
    mem.release_tensor(a6)
    mem.release_tensor(a2)
    mem.release_tensor(a5)

    a10 = mem.allocate_tensor((512,)).mul_(0.0).add_(10.0)
    mem.print_allocation(resolution=100)
    # print(f"a4:{a4}")
    # print(f"a7:{a7}")
    # print(f"a8:{a8}")
    # print(f"a10:{a10}")
    print((a4.norm() + a7.norm() + a8.norm() + a10.norm()).item())
    # assert (a4.norm() + a7.norm() + a8.norm() + a10.norm()).item() == 474.50, "Test failed"


def test2(instantiate_enough_capacity=True):
    if instantiate_enough_capacity:
        mem = HostPinnedMemoryAllocator(
            2048, torch.half, disable_defragmentation=True
        )
    else:
        mem = HostPinnedMemoryAllocator(
            512, torch.half, disable_defragmentation=False
        )
    a1 = mem.allocate_tensor((64,)).mul_(0.0).add_(1.0)
    a2 = mem.allocate_tensor((64,)).mul_(0.0).add_(2.0)
    a3 = mem.allocate_tensor((64,)).mul_(0.0).add_(3.0)
    a4 = mem.allocate_tensor((64,)).mul_(0.0).add_(4.0)
    a5 = mem.allocate_tensor((64,)).mul_(0.0).add_(5.0)
    a6 = mem.allocate_tensor((64,)).mul_(0.0).add_(6.0)
    a7 = mem.allocate_tensor((64,)).mul_(0.0).add_(7.0)
    a8 = mem.allocate_tensor((64,)).mul_(0.0).add_(8.0)
    mem.release_tensor(a2)
    mem.release_tensor(a4)
    mem.release_tensor(a6)
    mem.release_tensor(a8)
    mem.print_allocation(resolution=100)

    a9 = mem.allocate_tensor((128,)).mul_(0.0).add_(9.0)
    a10 = mem.allocate_tensor((64,)).mul_(0.0).add_(10.0)
    a11 = mem.allocate_tensor((64,)).mul_(0.0).add_(11.0)
    mem.release_tensor(a1)
    mem.release_tensor(a5)
    mem.print_allocation(resolution=100)
    a12 = mem.allocate_tensor((128,)).mul_(0.0).add_(12.0)
    mem.print_allocation(resolution=100)
    print(f"a7:{a7}")
    print(f"a9:{a9}")
    print(f"a10:{a10}")
    print(f"a11:{a11}")
    print(f"a12:{a12}")
    print(a7.norm() + a9.norm() + a10.norm() + a11.norm() + a12.norm())
    # assert (a7.norm() + a9.norm() + a10.norm() + a11.norm() + a12.norm()) == 460.75, "TestFailed"


if __name__ == "__main__":
    test1()
    test2()
