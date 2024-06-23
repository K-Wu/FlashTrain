We need to use contiguous memory allocator to allocate host-pinned memory only once while reuse it multiple times due to the [high cost](https://forums.developer.nvidia.com/t/poor-performance-cudahostunregister/24345). We use the [contiguous memory allocator from DeepSpeed](https://github.com/microsoft/DeepSpeed/blob/d89e8cdfe55410e666a184d7ab7e664e7887228c/deepspeed/runtime/zero/contiguous_memory_allocator.py) as the base class in our implementation.

## Reference
[Memory management - CUDA Semantics - Pytorch Documentation](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
[A guide to PyTorch's CUDA Caching Allocator](https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html)