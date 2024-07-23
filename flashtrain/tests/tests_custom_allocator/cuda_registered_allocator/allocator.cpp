// Adapted from
// https://github.com/rapidsai/rmm/blob/0e7d519d6667064a0e8ac42fc0e8b5b80f7c644c/python/rmm/rmm/_lib/_torch_allocator.cpp
/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_runtime_api.h>
#include <torch/extension.h>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

// These signatures must match those required by CUDAPluggableAllocator in
// github.com/pytorch/pytorch/blob/main/torch/csrc/cuda/CUDAPluggableAllocator.h
// Since the loading is done at runtime via dlopen, no error checking
// can be performed for mismatching signatures.

/**
 * @brief Allocate memory of at least \p size bytes.
 *
 * @throws rmm::bad_alloc When the requested allocation cannot be satisfied.
 *
 * @param size The number of bytes to allocate
 * @param device The device whose memory resource one should use
 * @param stream CUDA stream to perform allocation on
 * @return Pointer to the newly allocated memory
 */
extern "C" {

bool verbose_flag = false;
void reset_verbose_flag(bool flag) { verbose_flag = flag; }

void* allocate(std::size_t size, int device, void* stream) {
  rmm::cuda_device_id const device_id{device};
  rmm::cuda_set_device_raii with_device{device_id};
  auto mr = rmm::mr::get_per_device_resource(device_id);
  if (verbose_flag) {
    std::cout << "Allocating " << size << " on device " << device << std::endl;
  }
  return mr->allocate(size,
                      rmm::cuda_stream_view{static_cast<cudaStream_t>(stream)});
}

/**
 * @brief Deallocate memory pointed to by \p ptr.
 *
 * @param ptr Pointer to be deallocated
 * @param size The number of bytes in the allocation
 * @param device The device whose memory resource one should use
 * @param stream CUDA stream to perform deallocation on
 */
void deallocate(void* ptr, std::size_t size, int device, void* stream) {
  rmm::cuda_device_id const device_id{device};
  rmm::cuda_set_device_raii with_device{device_id};
  auto mr = rmm::mr::get_per_device_resource(device_id);
  if (verbose_flag) {
    std::cout << "Deallocating " << ptr << " size " << size << " on device "
              << device << std::endl;
  }
  mr->deallocate(ptr, size,
                 rmm::cuda_stream_view{static_cast<cudaStream_t>(stream)});
}

}  // extern "C"

TORCH_LIBRARY(cuda_registered_allocator, m) {
  m.def("reset_verbose_flag", &reset_verbose_flag);
}