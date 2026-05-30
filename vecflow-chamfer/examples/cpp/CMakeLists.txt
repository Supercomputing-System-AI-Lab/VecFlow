# =============================================================================
# vecflow-chamfer examples.
#
# Build sequence:
#   1. Build the cpp libraries first (separate build tree):
#        cmake -S cpp -B cpp/build && cmake --build cpp/build -j
#   2. Build the examples (this project; links against the above):
#        cmake -S examples/cpp -B examples/cpp/build && cmake --build examples/cpp/build -j
#
# This project does NOT rebuild the library. It finds the pre-built libs in
# cpp/build and imports them as IMPORTED targets so the library and the
# examples have independent build trees.
#
# Executables:
#   CHAMFER_CORE_KERNEL      Always built. Links libvecflow_chamfer_kernels +
#                            nlohmann_json. No cuvs dependency.
#   VECFLOW_CHAMFER          Built when BUILD_CHAMFER_ONLY=OFF (default). Links
#                            libvecflow_chamfer (which transitively requires
#                            cuvs::cuvs).
# =============================================================================

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

option(BUILD_CHAMFER_ONLY "Build only the chamfer kernel example (no cuvs)" OFF)

if(BUILD_CHAMFER_ONLY)
  project(vecflow_chamfer_examples LANGUAGES CXX CUDA)
else()
  include(../../cmake/rapids_config.cmake)
  include(rapids-cmake)
  include(rapids-cpm)
  include(rapids-cuda)
  include(rapids-export)
  include(rapids-find)
  rapids_cuda_init_architectures(vecflow_chamfer_examples)
  project(vecflow_chamfer_examples LANGUAGES CXX CUDA)
endif()

find_package(Threads)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT BUILD_CHAMFER_ONLY)
  rapids_cpm_init()
  include(../../cmake/thirdparty/get_cuvs.cmake)
endif()

# nlohmann_json (used by both examples)
if(BUILD_CHAMFER_ONLY)
  include(FetchContent)
  FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
  )
  set(JSON_BuildTests OFF CACHE INTERNAL "")
  FetchContent_MakeAvailable(nlohmann_json)
else()
  rapids_cpm_find(nlohmann_json 3.11.2
    GLOBAL_TARGETS nlohmann_json::nlohmann_json
    CPM_ARGS
      GITHUB_REPOSITORY nlohmann/json
      GIT_TAG v3.11.2
      OPTIONS
        "JSON_BuildTests OFF"
        "CMAKE_POLICY_VERSION_MINIMUM 3.5"
  )
endif()

# --------------------------------------------------------------------------- #
# Import the pre-built vecflow_chamfer libraries from cpp/build.              #
# --------------------------------------------------------------------------- #
set(VECFLOW_CHAMFER_BUILD_DIR "${CMAKE_CURRENT_LIST_DIR}/../../cpp/build"
    CACHE PATH "Path to a vecflow-chamfer cpp/build directory")
set(VECFLOW_CHAMFER_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../cpp/include")

if(NOT EXISTS "${VECFLOW_CHAMFER_BUILD_DIR}/libvecflow_chamfer_kernels.so" AND
   NOT EXISTS "${VECFLOW_CHAMFER_BUILD_DIR}/libvecflow_chamfer_kernels.a")
  message(FATAL_ERROR
    "vecflow_chamfer kernels library not found in:\n"
    "    ${VECFLOW_CHAMFER_BUILD_DIR}\n"
    "Build it first:\n"
    "    cmake -S cpp -B cpp/build && cmake --build cpp/build -j\n"
    "(or pass -DVECFLOW_CHAMFER_BUILD_DIR=/path/to/cpp/build).")
endif()

find_library(VECFLOW_CHAMFER_KERNELS_LIB vecflow_chamfer_kernels
             PATHS "${VECFLOW_CHAMFER_BUILD_DIR}" NO_DEFAULT_PATH REQUIRED)
add_library(vecflow_chamfer::vecflow_chamfer_kernels UNKNOWN IMPORTED)
set_target_properties(vecflow_chamfer::vecflow_chamfer_kernels PROPERTIES
  IMPORTED_LOCATION "${VECFLOW_CHAMFER_KERNELS_LIB}"
  INTERFACE_INCLUDE_DIRECTORIES "${VECFLOW_CHAMFER_INCLUDE_DIR}"
)

if(NOT BUILD_CHAMFER_ONLY)
  find_library(VECFLOW_CHAMFER_LIB vecflow_chamfer
               PATHS "${VECFLOW_CHAMFER_BUILD_DIR}" NO_DEFAULT_PATH REQUIRED)
  add_library(vecflow_chamfer::vecflow_chamfer UNKNOWN IMPORTED)
  set_target_properties(vecflow_chamfer::vecflow_chamfer PROPERTIES
    IMPORTED_LOCATION "${VECFLOW_CHAMFER_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${VECFLOW_CHAMFER_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "vecflow_chamfer::vecflow_chamfer_kernels;cuvs::cuvs"
  )
endif()

# --------------------------------------------------------------------------- #
# Executables                                                                 #
# --------------------------------------------------------------------------- #

# Raw kernel example: launches chamfer_core_score directly + CPU verify.
add_executable(CHAMFER_CORE_KERNEL src/chamfer_core_kernel.cu)
target_link_libraries(CHAMFER_CORE_KERNEL PRIVATE
  vecflow_chamfer::vecflow_chamfer_kernels
  nlohmann_json::nlohmann_json
)
# RDC required: the kernel symbol lives in libvecflow_chamfer_kernels.so;
# without separable compilation the consumer emits a no-op stub.
set_target_properties(CHAMFER_CORE_KERNEL PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

if(NOT BUILD_CHAMFER_ONLY)
  # Pipeline example: load -> build -> search, driven by config.json.
  add_executable(VECFLOW_CHAMFER src/vecflow_chamfer.cu)
  target_link_libraries(VECFLOW_CHAMFER PRIVATE
    vecflow_chamfer::vecflow_chamfer
    nlohmann_json::nlohmann_json
    $<TARGET_NAME_IF_EXISTS:conda_env>
    Threads::Threads
  )
  set_target_properties(VECFLOW_CHAMFER PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
endif()
