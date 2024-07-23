# Adapted from
# https://github.com/rapidsai/rmm/blob/0e7d519d6667064a0e8ac42fc0e8b5b80f7c644c/python/rmm/rmm/_lib/CMakeLists.txt
# and https://vscode.dev/github/rapidsai/rmm
# =============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
# =============================================================================
cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

# include(./rapids_config.cmake)

project(
  cuda_registered_allocator
  VERSION 0.1.0
  LANGUAGES CXX)

find_package(rmm REQUIRED)
find_package(Torch REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# pass through logging level to spdlog
# add_compile_definitions("SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${RMM_LOGGING_LEVEL}")

# set(cython_sources device_buffer.pyx lib.pyx logger.pyx memory_resource.pyx
# cuda_stream.pyx)
set(linked_libraries rmm::rmm)

# Build all of the Cython targets rapids_cython_create_modules(SOURCE_FILES
# "${cython_sources}" LINKED_LIBRARIES "${linked_libraries}" CXX)
add_library(_torch_allocator SHARED allocator.cpp)
# Want the output to be called _torch_allocator.so
set_target_properties(_torch_allocator PROPERTIES PREFIX "" SUFFIX ".so")
target_include_directories(_torch_allocator PRIVATE ${TORCH_INCLUDE_DIRS}
                                                    ${Python3_INCLUDE_DIRS})
target_link_libraries(_torch_allocator PRIVATE rmm::rmm ${TORCH_LIBRARIES})
# cmake_path(RELATIVE_PATH CMAKE_CURRENT_SOURCE_DIR BASE_DIRECTORY
# "${PROJECT_SOURCE_DIR}" OUTPUT_VARIABLE _torch_allocator_location)
# install(TARGETS _torch_allocator DESTINATION "${_torch_allocator_location}")
