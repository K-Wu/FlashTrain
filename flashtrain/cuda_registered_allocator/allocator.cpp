// Adapted from
// https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management
#include <cuda_runtime_api.h>
#include <sys/types.h>
#include <torch/extension.h>
#include <torch/script.h>

#include <iostream>

// Compile with g++ allocator.cpp -o allocator.so -I/usr/local/cuda/include
// -shared -fPIC
extern "C" {
bool verbose_flag = false;

void reset_verbose_flag(bool flag) { verbose_flag = flag; }

void* my_malloc(ssize_t size, int device, cudaStream_t stream) {
  void* ptr;
  cudaMalloc(&ptr, size);
  if (verbose_flag) std::cout << "alloc " << ptr << size << std::endl;
  return ptr;
}

void my_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
  if (verbose_flag) std::cout << "free " << ptr << " " << stream << std::endl;
  cudaFree(ptr);
}
}

TORCH_LIBRARY(cuda_registered_allocator, m) {
  m.def("reset_verbose_flag", &reset_verbose_flag);
}