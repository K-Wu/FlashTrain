// From https://forums.developer.nvidia.com/t/cudamalloc-cudafree-address/195166
// g++ -I/usr/local/cuda/include -fPIC -shared -o hook.so hook.cpp -ldl
// -L/usr/local/cuda/lib64 -lcudart
#include <cuda_runtime.h>
#include <cufile.h>
#include <dlfcn.h>
#include <stdio.h>
#include <unistd.h>

#include <set>

bool verbose = false;
bool enable_cufile_registration = false;
int num_allocs = 0;
int num_frees = 0;
std::set<void*> registered_buffers;

extern "C" {
void set_verbose(bool v) { verbose = v; }
void set_enable_cufile_registration(bool v) { enable_cufile_registration = v; }
}

int get_num_allocs() { return num_allocs; }
int get_num_frees() { return num_frees; }

cudaError_t (*lcudaMalloc)(void**, size_t);
CUfileError_t (*lcuFileBufRegister)(const void*, size_t, int);

CUfileError_t (*lcuFileBufDeregister)(const void*);
cudaError_t (*lcudaFree)(void*);

cudaError_t cudaMalloc(void** devPtr, size_t count) {
  if (!lcudaMalloc) {
    lcudaMalloc =
        (cudaError_t(*)(void**, size_t))dlsym(RTLD_NEXT, "cudaMalloc");
  }
  if (!lcuFileBufRegister) {
    lcuFileBufRegister = (CUfileError_t(*)(const void*, size_t, int))dlsym(
        RTLD_NEXT, "cuFileBufRegister");
  }

  num_allocs++;
  cudaError_t err = lcudaMalloc(devPtr, count);
  if (verbose) printf("cudaMalloc hooked=> %p\n", *devPtr);
  if (err == cudaSuccess && enable_cufile_registration) {
    if (verbose) printf("Registering buffer %p\n", *devPtr);
    CUfileError_t cuerr = lcuFileBufRegister(*devPtr, count, 0);
    registered_buffers.insert(*devPtr);
    if (cuerr.err != CU_FILE_SUCCESS) {
      printf("cuFileBufRegister failed with %d\n", cuerr);
    }
  }

  return err;
}

cudaError_t cudaFree(void* devPtr) {
  if (!lcuFileBufDeregister) {
    lcuFileBufDeregister =
        (CUfileError_t(*)(const void*))dlsym(RTLD_NEXT, "cuFileBufDeregister");
  }
  if (!lcudaFree) {
    lcudaFree = (cudaError_t(*)(void*))dlsym(RTLD_NEXT, "cudaFree");
  }

  num_frees++;
  if (verbose) printf("Deregistering buffer %p\n", devPtr);
  if (registered_buffers.count(devPtr)) {
    CUfileError_t cuerr = lcuFileBufDeregister(devPtr);
    registered_buffers.erase(devPtr);
    if (cuerr.err != CU_FILE_SUCCESS) {
      printf("cuFileBufDeregister failed with %d\n", cuerr);
    }
  }

  if (verbose) printf("cudaFree   hooked=> %p\n", devPtr);
  cudaError_t err = lcudaFree(devPtr);

  return err;
}
