# =============================================================================
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

# Keep these in lockstep with the in-tree cuVS at /u/cmo1/VecFlow/VERSION (major.minor)
# and /u/cmo1/VecFlow/RAPIDS_BRANCH. Mismatch causes cuVS's CMake to call into a stale
# rapids-cmake (e.g. missing rapids_cuda_enable_fatbin_compression).
set(RAPIDS_VERSION "26.06")
set(RAPIDS_BRANCH "main")

# Newer rapids-cmake (post-25.x) requires these to be set before including RAPIDS.cmake;
# matches the pattern in /u/cmo1/VecFlow/cmake/rapids_config.cmake.
if(NOT rapids-cmake-version)
    set(rapids-cmake-version "${RAPIDS_VERSION}")
endif()
if(NOT rapids-cmake-branch)
    set(rapids-cmake-branch "${RAPIDS_BRANCH}")
endif()

if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/RAFT_RAPIDS.cmake)
    file(DOWNLOAD
            https://raw.githubusercontent.com/rapidsai/rapids-cmake/${RAPIDS_BRANCH}/RAPIDS.cmake
            ${CMAKE_CURRENT_BINARY_DIR}/RAFT_RAPIDS.cmake)
endif()
include(${CMAKE_CURRENT_BINARY_DIR}/RAFT_RAPIDS.cmake)
