# =============================================================================
# Top-level CMake for the vecflow-chamfer Python binding.
#
# Layout:
#   cpp/CMakeLists.txt       — the C++ libraries (vecflow_chamfer + kernels).
#                              Built and packaged independently as
#                              libvecflow-chamfer-cuXX.
#   CMakeLists.txt (this)    — the pybind11 module. Compiled inside the
#                              scikit-build-core wheel build, against the
#                              C++ library installed in the conda host env
#                              (or against cpp/build via the BUILD export).
# =============================================================================

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(./cmake/rapids_config.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-find)

rapids_cpm_init()

rapids_cuda_init_architectures(vecflow_chamfer_py)
project(vecflow_chamfer_py LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# Python + pybind11. We fetch pybind11 via CPM so the wheel build doesn't
# require the user to pip-install pybind11 separately (scikit-build-core
# already brings the headers, but the cmake config is variable across
# environments; fetching pins a known-good version).
find_package(Python COMPONENTS Interpreter Development REQUIRED)
rapids_cpm_find(pybind11 2.11.1
  GLOBAL_TARGETS pybind11::pybind11
  CPM_ARGS
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.11.1
    OPTIONS        "PYBIND11_INSTALL OFF"
                  "PYBIND11_TEST OFF"
)
find_package(pybind11 REQUIRED)

# Resolve the C++ libraries. vecflow_chamfer's exported config references
# cuvs::cuvs in its link interface, so cuvs must be found FIRST or the
# vecflow_chamfer-targets.cmake set_target_properties() call fails. In a
# conda build the host env has both libcuvs-vecflow + libvecflow-chamfer
# installed at $PREFIX. In a local dev build pass
# `-Dcuvs_ROOT=…/cpp/build -Dvecflow_chamfer_ROOT=vecflow-chamfer/cpp/build`.
find_package(cuvs REQUIRED)
find_package(vecflow_chamfer REQUIRED)

pybind11_add_module(_vecflow_chamfer binding/binding.cpp)

target_link_libraries(_vecflow_chamfer
  PRIVATE
    vecflow_chamfer::vecflow_chamfer
)
set_target_properties(_vecflow_chamfer PROPERTIES
  CXX_STANDARD 20
  CUDA_STANDARD 20
  POSITION_INDEPENDENT_CODE ON
)

# Bake the directory holding libvecflow_chamfer.so into the module's RPATH so
# the loader can resolve it without LD_LIBRARY_PATH. INSTALL_RPATH_USE_LINK_PATH
# doesn't fire for SHARED IMPORTED targets (the linker takes them by absolute
# path, not via -L), so we have to query the imported target directly.
foreach(_vc_tgt vecflow_chamfer::vecflow_chamfer vecflow_chamfer::vecflow_chamfer_kernels)
  if(TARGET ${_vc_tgt})
    get_target_property(_vc_loc ${_vc_tgt} IMPORTED_LOCATION)
    if(NOT _vc_loc)
      get_target_property(_vc_loc ${_vc_tgt} IMPORTED_LOCATION_RELEASE)
    endif()
    if(_vc_loc)
      get_filename_component(_vc_libdir "${_vc_loc}" DIRECTORY)
      set_property(TARGET _vecflow_chamfer APPEND PROPERTY INSTALL_RPATH "${_vc_libdir}")
      set_property(TARGET _vecflow_chamfer APPEND PROPERTY BUILD_RPATH   "${_vc_libdir}")
      message(STATUS "vecflow-chamfer: adding RPATH ${_vc_libdir} (from ${_vc_tgt})")
    endif()
  endif()
endforeach()

if(DEFINED SKBUILD)
  # scikit-build-core staging path — lands at vecflow_chamfer/_vecflow_chamfer.so
  # inside the wheel. Matches `from ._vecflow_chamfer import ...` in __init__.py.
  install(TARGETS _vecflow_chamfer LIBRARY DESTINATION vecflow_chamfer)
else()
  # Direct-cmake dev path: install into the active conda env's site-packages.
  set(SITE_PACKAGES_DIR
      "$ENV{CONDA_PREFIX}/lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages")
  install(TARGETS _vecflow_chamfer LIBRARY DESTINATION ${SITE_PACKAGES_DIR}/vecflow_chamfer)
endif()
