# =============================================================================
# Copyright (c) 2023-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

# ------------- configure rapids-cmake --------------#

include(../../cmake/thirdparty/fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

# ------------- configure project --------------#

rapids_cuda_init_architectures(test_cuvs)

project(test_cuvs LANGUAGES CXX CUDA)
find_package(Threads)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fopenmp")
endif()

# ------------- configure cuvs -----------------#

rapids_cpm_init()
set(BUILD_CUVS_C_LIBRARY OFF)
include(../../cmake/thirdparty/get_cuvs.cmake)

# ------------- configure nlohmann/json -----------------#

rapids_cpm_find(nlohmann_json
  VERSION 3.11.2
  GITHUB_REPOSITORY nlohmann/json
  OPTIONS
    "JSON_BuildTests OFF"
    "CMAKE_POLICY_VERSION_MINIMUM 3.5"
)

# -------------- compile tasks ----------------- #
add_executable(VECFLOW_EXAMPLE src/vecflow_example.cu)
add_executable(VECFLOW_BENCH src/bench/vecflow_bench.cu)

# `$<TARGET_NAME_IF_EXISTS:conda_env>` is a generator expression that ensures that targets are
# installed in a conda environment, if one exists
target_link_libraries(VECFLOW_EXAMPLE PUBLIC OpenMP::OpenMP_CXX PRIVATE cuvs::cuvs PRIVATE nlohmann_json::nlohmann_json PRIVATE $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(VECFLOW_BENCH PUBLIC OpenMP::OpenMP_CXX PRIVATE cuvs::cuvs PRIVATE nlohmann_json::nlohmann_json PRIVATE $<TARGET_NAME_IF_EXISTS:conda_env>)