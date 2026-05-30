# =============================================================================
# vecflow-chamfer: shared/static library on top of libcuvs-vecflow.
#
# Two libraries are produced:
#   - vecflow_chamfer_kernels  Always built. Zero deps beyond CUDA toolkit.
#                              Just the chamfer_core kernels + launchers.
#   - vecflow_chamfer          Built when BUILD_CHAMFER_ONLY=OFF (default).
#                              Adds dataset/index/search; depends on cuvs::cuvs.
# =============================================================================

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

option(BUILD_CHAMFER_ONLY "Build only the chamfer kernels (no cuvs dependency)" OFF)
option(BUILD_SHARED_LIBS  "Build vecflow_chamfer as a shared library" ON)

if(BUILD_CHAMFER_ONLY)
  project(vecflow_chamfer LANGUAGES CXX CUDA)
else()
  include(../cmake/rapids_config.cmake)
  include(rapids-cmake)
  include(rapids-cpm)
  include(rapids-cuda)
  include(rapids-export)
  include(rapids-find)
  rapids_cuda_init_architectures(vecflow_chamfer)
  project(vecflow_chamfer
          VERSION "${RAPIDS_VERSION}"
          LANGUAGES CXX CUDA)
  rapids_cmake_build_type(Release)
  rapids_cpm_init()
  include(../cmake/thirdparty/get_cuvs.cmake)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --------------------------------------------------------------------------- #
# Kernels-only library: zero deps beyond the CUDA toolkit.                    #
# --------------------------------------------------------------------------- #
add_library(vecflow_chamfer_kernels src/search/chamfer_core.cu)
add_library(vecflow_chamfer::vecflow_chamfer_kernels ALIAS vecflow_chamfer_kernels)

target_include_directories(vecflow_chamfer_kernels
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_features(vecflow_chamfer_kernels PUBLIC cxx_std_20 cuda_std_20)
set_target_properties(vecflow_chamfer_kernels PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  EXPORT_NAME vecflow_chamfer_kernels
)

# Public chamfer_core.hpp exposes `extern template __global__` declarations for
# chamfer_score<int> / chamfer_score_anchor<int>. nvcc 12.x defaults to
# `-static-global-template-stub=false`, but warns the default will flip in a
# future release. Pin the legacy behavior explicitly so the warning is silenced
# and the cross-TU template-stub stays stable for consumers.
target_compile_options(vecflow_chamfer_kernels PUBLIC
  $<$<COMPILE_LANGUAGE:CUDA>:-static-global-template-stub=false>
)

# --------------------------------------------------------------------------- #
# Full library: dataset + indexing + search. Requires libcuvs-vecflow.        #
# --------------------------------------------------------------------------- #
if(NOT BUILD_CHAMFER_ONLY)
  set(VECFLOW_CHAMFER_FULL_SOURCES
    src/data_loader.cu
    src/indexing/build.cu
    src/indexing/build_vpq.cu
    src/indexing/doc_id_mapping.cu
    src/indexing/recompute_anchors.cu
    src/indexing/serialize.cu
    src/search/find_unique_ids.cu
    src/search/search.cu
  )
  add_library(vecflow_chamfer ${VECFLOW_CHAMFER_FULL_SOURCES})
  add_library(vecflow_chamfer::vecflow_chamfer ALIAS vecflow_chamfer)

  target_include_directories(vecflow_chamfer
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  target_link_libraries(vecflow_chamfer
    PUBLIC vecflow_chamfer::vecflow_chamfer_kernels cuvs::cuvs
  )
  target_compile_features(vecflow_chamfer PUBLIC cxx_std_20 cuda_std_20)
  set_target_properties(vecflow_chamfer PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    EXPORT_NAME vecflow_chamfer
  )

  # ------------------------------------------------------------------------- #
  # Install + export. Only wired up on the full-library path because the      #
  # kernels-only build (BUILD_CHAMFER_ONLY=ON) intentionally skips            #
  # rapids-cmake. Downstream consumers (the Python binding, the conda        #
  # recipe, external find_package users) resolve against this one export set #
  # and get both libraries + headers.                                         #
  # ------------------------------------------------------------------------- #
  rapids_cmake_install_lib_dir(lib_dir)
  include(GNUInstallDirs)

  install(
    TARGETS vecflow_chamfer_kernels vecflow_chamfer
    DESTINATION ${lib_dir}
    EXPORT vecflow_chamfer-exports
  )

  install(
    DIRECTORY include/vecflow_chamfer
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

  rapids_export(
    INSTALL vecflow_chamfer
    EXPORT_SET vecflow_chamfer-exports
    GLOBAL_TARGETS vecflow_chamfer vecflow_chamfer_kernels
    NAMESPACE vecflow_chamfer::
  )

  rapids_export(
    BUILD vecflow_chamfer
    EXPORT_SET vecflow_chamfer-exports
    GLOBAL_TARGETS vecflow_chamfer vecflow_chamfer_kernels
    NAMESPACE vecflow_chamfer::
  )
endif()
