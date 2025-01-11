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

include(../cmake/thirdparty/fetch_rapids.cmake)
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
include(../cmake/thirdparty/get_cuvs.cmake)

# -------------- compile tasks ----------------- #
add_executable(BFS_TEST src/filtered_bfs_test.cu)
add_executable(BFS_YFCC src/filtered_bfs_yfcc.cu)
add_executable(BFS_BENCH src/filtered_bfs_bench.cu)
add_executable(CAGRA_FILTER_BENCH src/cagra_filter_bench.cu)
add_executable(CAGRA_FILTER_TEST src/cagra_filter_test.cu)
add_executable(CAGRA_HYBRID src/cagra_hybrid.cu)
add_executable(BENCH_BUILD_SIFT src/bench/build_sift.cu)
add_executable(BENCH_BUILD_WIKI src/bench/build_wiki.cu)
add_executable(BENCH_SEARCH_CAGRA_YFCC src/bench/search_cagra_yfcc.cu)
add_executable(BENCH_SEARCH_CAGRA_IF_YFCC src/bench/search_cagra_if_yfcc.cu)
add_executable(BENCH_SEARCH_CAGRA_IF_SIFT src/bench/search_cagra_if_sift.cu)
add_executable(BENCH_SEARCH_CAGRA_SIFT src/bench/search_cagra_sift.cu)
add_executable(BENCH_SEARCH_CAGRA_WIKI src/bench/search_cagra_wiki.cu)
# add_executable(CAGRA_EXAMPLE src/cagra_example.cu)
# add_executable(CAGRA_PERSISTENT_EXAMPLE src/cagra_persistent_example.cu)

# `$<TARGET_NAME_IF_EXISTS:conda_env>` is a generator expression that ensures that targets are
# installed in a conda environment, if one exists
# target_link_libraries(CAGRA_EXAMPLE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
# target_link_libraries(CAGRA_PERSISTENT_EXAMPLE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env> Threads::Threads)
target_link_libraries(BFS_TEST PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(BFS_YFCC PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(BFS_BENCH PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(CAGRA_FILTER_BENCH PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(CAGRA_FILTER_TEST PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(CAGRA_HYBRID PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(BENCH_BUILD_SIFT PUBLIC OpenMP::OpenMP_CXX PRIVATE cuvs::cuvs PRIVATE $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(BENCH_BUILD_WIKI PUBLIC OpenMP::OpenMP_CXX PRIVATE cuvs::cuvs PRIVATE $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(BENCH_SEARCH_CAGRA_YFCC PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(BENCH_SEARCH_CAGRA_IF_YFCC PUBLIC OpenMP::OpenMP_CXX PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(BENCH_SEARCH_CAGRA_IF_SIFT PUBLIC OpenMP::OpenMP_CXX PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(BENCH_SEARCH_CAGRA_SIFT PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(BENCH_SEARCH_CAGRA_WIKI PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)