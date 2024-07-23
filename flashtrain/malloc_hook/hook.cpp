// From https://forums.developer.nvidia.com/t/cudamalloc-cudafree-address/195166
// g++ -I/usr/local/cuda/include -fPIC -shared -o hook.so hook.cpp -ldl
// -L/usr/local/cuda/lib64 -lcudart
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <stdio.h>
#include <unistd.h>

cudaError_t cudaMalloc(void** devPtr, size_t count) {
  cudaError_t (*lcudaMalloc)(void**, size_t) =
      (cudaError_t(*)(void**, size_t))dlsym(RTLD_NEXT, "cudaMalloc");
  printf("cudaMalloc hooked=> %p\n", devPtr);
  return lcudaMalloc(devPtr, count);
}

cudaError_t cudaFree(void* devPtr) {
  cudaError_t (*lcudaFree)(void*) =
      (cudaError_t(*)(void*))dlsym(RTLD_NEXT, "cudaFree");
  printf("cudaFree   hooked=> %p\n", devPtr);
  return lcudaFree(devPtr);
}