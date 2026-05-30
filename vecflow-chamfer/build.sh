#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# vecflow-chamfer build orchestrator.
#
# Usage:
#   ./build.sh vecflow-chamfer    # build the C++/CUDA chamfer library (cpp/)
#   ./build.sh examples           # build VECFLOW_CHAMFER + CHAMFER_CORE_KERNEL
#   ./build.sh python             # pip install . the Python wrapper
#                                   (requires `vecflow-chamfer` target built first
#                                    so find_package(vecflow_chamfer) resolves)
#   ./build.sh clean              # remove build directories

set -e

REPODIR=$(cd "$(dirname "$0")"; pwd)
NUMARGS=$#
ARGS=$*

hasArg() { (( NUMARGS != 0 )) && (echo " ${ARGS} " | grep -q " $1 "); }

if hasArg -h || hasArg --help || (( NUMARGS == 0 )); then
  echo "usage: $0 [vecflow-chamfer|examples|python|clean] [-v -g --allgpuarch --gpu-arch=...]"
  exit 0
fi

if hasArg clean; then
  rm -rf "${REPODIR}/cpp/build" "${REPODIR}/examples/cpp/build" "${REPODIR}/build"
  echo "Cleaned build directories."
  exit 0
fi

gpuArch() {
  if hasArg --allgpuarch; then echo "RAPIDS"; return; fi
  if [[ -n $(echo "$ARGS" | { grep -E "\-\-gpu\-arch" || true; }) ]]; then
    echo "$ARGS" | { grep -Eo "\-\-gpu\-arch=[^ ]+" || true; } | sed 's/--gpu-arch=//'
    return
  fi
  echo "NATIVE"
}

PARALLEL_LEVEL=${PARALLEL_LEVEL:=$(nproc)}
BUILD_TYPE=Release
hasArg -g && BUILD_TYPE=Debug
VERBOSE_FLAG=""
hasArg -v && VERBOSE_FLAG="--verbose"

CUDA_ARCHS=$(gpuArch)
echo "GPU archs: ${CUDA_ARCHS}"

EXTRA_CMAKE_ARGS=()
if [[ -n "${CUVS_REPO_REL}" ]]; then
  EXTRA_CMAKE_ARGS+=("-DCPM_cuvs_SOURCE=$(readlink -f "${CUVS_REPO_REL}")")
else
  LIB_BUILD_DIR=${LIB_BUILD_DIR:-$(readlink -f "${REPODIR}/../cpp/build" 2>/dev/null || true)}
  if [[ -n "${LIB_BUILD_DIR}" && -d "${LIB_BUILD_DIR}" ]]; then
    EXTRA_CMAKE_ARGS+=("-Dcuvs_ROOT=${LIB_BUILD_DIR}")
    echo "Using pre-built libcuvs-vecflow at: ${LIB_BUILD_DIR}"
  fi
fi

configure_build() {
  local src_dir=$1
  local build_dir="${src_dir}/build"
  cmake -S "${src_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    "${EXTRA_CMAKE_ARGS[@]}"
  cmake --build "${build_dir}" -j"${PARALLEL_LEVEL}" ${VERBOSE_FLAG}
}

if hasArg vecflow-chamfer; then
  echo "==> configuring vecflow-chamfer (header-only)"
  configure_build "${REPODIR}/cpp"
fi

if hasArg examples; then
  echo "==> building examples (VECFLOW_CHAMFER)"
  configure_build "${REPODIR}/examples/cpp"
fi

if hasArg python; then
  if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "WARNING: CONDA_PREFIX is not set. The Python wrapper links against" >&2
    echo "         libvecflow-chamfer + libcuvs from the active conda env."   >&2
    echo "         Activate the vecflow env first, e.g. 'mamba activate vecflow'." >&2
  fi
  CHAMFER_BUILD_DIR="${REPODIR}/cpp/build"
  if [[ ! -d "${CHAMFER_BUILD_DIR}" ]]; then
    echo "error: ${CHAMFER_BUILD_DIR} not found — run './build.sh vecflow-chamfer' first" >&2
    exit 1
  fi
  echo "==> installing Python wrapper (pip install .)"
  # SKBUILD_CMAKE_DEFINE points scikit-build-core at the BUILD export rapids_export
  # writes to cpp/build, so find_package(vecflow_chamfer) resolves locally without
  # needing the conda recipe installed.
  ( cd "${REPODIR}" \
    && SKBUILD_CMAKE_DEFINE="vecflow_chamfer_ROOT=${CHAMFER_BUILD_DIR}" \
       pip install . --no-build-isolation )
fi
