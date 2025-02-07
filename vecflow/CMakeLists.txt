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

# ------------- Configure rapids-cmake --------------#
include(./cmake/thirdparty/fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

# Initialize CPM early
rapids_cpm_init()  # Must come before any `rapids_cpm_find`

# ------------- Configure project --------------#
rapids_cuda_init_architectures(vecflow)
project(vecflow LANGUAGES CXX CUDA)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Fetch pybind11 and ensure it exposes `pybind11_add_module`
rapids_cpm_find(pybind11 2.11.1
    GLOBAL_TARGETS pybind11::pybind11
    CPM_ARGS
        GIT_REPOSITORY  https://github.com/pybind/pybind11.git
        GIT_TAG         v2.11.1
        OPTIONS         "PYBIND11_INSTALL OFF"
                       "PYBIND11_TEST OFF"
                       # REMOVE "PYBIND11_NOPYTHON ON" (this was causing the issue)
)

# Explicitly load pybind11 to make `pybind11_add_module` available
find_package(pybind11 REQUIRED)  # <-- Add this line

# Configure OpenMP
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fopenmp")
endif()

# ------------- Configure cuvs -----------------#
set(BUILD_CUVS_C_LIBRARY OFF)
include(./cmake/thirdparty/get_cuvs.cmake)

# ------------- Compile tasks ----------------- #
# Use pybind11 to create the Python module
pybind11_add_module(vecflow 
    binding/binding.cpp
    src/vecflow.cu
)

# Add include directories for the target
target_include_directories(vecflow PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link libraries
target_link_libraries(vecflow PRIVATE 
    cuvs::cuvs
    OpenMP::OpenMP_CXX
    $<TARGET_NAME_IF_EXISTS:conda_env>
)

set(SITE_PACKAGES_DIR "$ENV{CONDA_PREFIX}/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages")
message(STATUS "Using Conda site-packages: ${SITE_PACKAGES_DIR}")
install(TARGETS vecflow DESTINATION ${SITE_PACKAGES_DIR})