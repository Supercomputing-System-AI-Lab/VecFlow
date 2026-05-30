# =============================================================================
# Find or fetch libcuvs-vecflow (the VecFlow fork of cuVS).
#
# By default this looks for the cuVS C++ library exposed as cuvs::cuvs. To use a
# local VecFlow checkout, pass one of the following at configure time:
#
#   -Dcuvs_ROOT=/path/to/VecFlow/cpp/build         (find a pre-built install)
#   -DCPM_cuvs_SOURCE=/path/to/VecFlow             (build from source via CPM)
#
# If neither is set, CPM clones moccch/VecFlow at the pinned tag.
# =============================================================================

set(VECFLOW_CUVS_VERSION "${RAPIDS_VERSION_MAJOR_MINOR}")
set(VECFLOW_CUVS_FORK "moccch")
set(VECFLOW_CUVS_PINNED_TAG "${rapids-cmake-branch}")

function(find_and_configure_cuvs)
  set(oneValueArgs VERSION FORK PINNED_TAG)
  cmake_parse_arguments(PKG "" "${oneValueArgs}" "" ${ARGN})

  rapids_cpm_find(cuvs ${PKG_VERSION}
    GLOBAL_TARGETS     cuvs::cuvs
    BUILD_EXPORT_SET   vecflow-chamfer-exports
    INSTALL_EXPORT_SET vecflow-chamfer-exports
    CPM_ARGS
      GIT_REPOSITORY https://github.com/${PKG_FORK}/VecFlow.git
      GIT_TAG        ${PKG_PINNED_TAG}
      SOURCE_SUBDIR  cpp
      OPTIONS
        "BUILD_C_LIBRARY OFF"
        "BUILD_TESTS OFF"
        "CUVS_NVTX OFF"
  )
endfunction()

find_and_configure_cuvs(
  VERSION    ${VECFLOW_CUVS_VERSION}.00
  FORK       ${VECFLOW_CUVS_FORK}
  PINNED_TAG ${VECFLOW_CUVS_PINNED_TAG}
)
