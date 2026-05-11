#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# VecFlow convenience build script. Wraps cmake/pip so users don't have to
# remember the right invocation when iterating on the bundled examples or
# the Python wrapper.
#
# Usage:
#   ./build.sh                 # same as `./build.sh examples`
#   ./build.sh examples        # configure + build the C++ SIFT1M example
#   ./build.sh python          # pip install the Python wrapper into the active env
#   ./build.sh clean           # rm -rf examples/cpp/build
#   ./build.sh examples python # multiple targets in one go
#   ./build.sh -h              # this help
#
# Flags:
#   -v / --verbose             # echo every command (set -x) and pass VERBOSE=1 to make
#   -j N                       # parallel jobs (default: nproc)
#   -h / --help                # show this message

set -euo pipefail

VECFLOW_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXAMPLE_DIR="$VECFLOW_DIR/examples/cpp"
BUILD_DIR="$EXAMPLE_DIR/build"

JOBS="$(nproc 2>/dev/null || echo 4)"
VERBOSE=0
TARGETS=()

# ── arg parse ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      sed -n '4,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    -v|--verbose)
      VERBOSE=1
      shift
      ;;
    -j)
      JOBS="$2"
      shift 2
      ;;
    -j*)
      JOBS="${1#-j}"
      shift
      ;;
    examples|python|clean)
      TARGETS+=("$1")
      shift
      ;;
    *)
      echo "error: unknown argument '$1' (try -h)" >&2
      exit 2
      ;;
  esac
done

# Default target.
if [[ ${#TARGETS[@]} -eq 0 ]]; then
  TARGETS=(examples)
fi

if [[ $VERBOSE -eq 1 ]]; then
  set -x
fi

# ── env sanity ──────────────────────────────────────────────────────────────
need_libcuvs() {
  if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "WARNING: CONDA_PREFIX is not set. The C++ example links against" >&2
    echo "         libcuvs from the active conda env. Activate the vecflow"   >&2
    echo "         env first, e.g.:"                                           >&2
    echo "           mamba activate vecflow"                                   >&2
  fi
}

# ── targets ─────────────────────────────────────────────────────────────────
do_clean() {
  if [[ -d "$BUILD_DIR" ]]; then
    echo "[clean] removing $BUILD_DIR"
    rm -rf "$BUILD_DIR"
  else
    echo "[clean] nothing to remove"
  fi
}

do_examples() {
  need_libcuvs
  echo "[examples] configuring in $BUILD_DIR"
  mkdir -p "$BUILD_DIR"
  cmake -S "$EXAMPLE_DIR" -B "$BUILD_DIR"
  echo "[examples] building (jobs=$JOBS)"
  cmake --build "$BUILD_DIR" --parallel "$JOBS"
  echo
  echo "[examples] done. Binaries:"
  for bin in VECFLOW_EXAMPLE VECFLOW_EXAMPLE_MULTI; do
    if [[ -x "$BUILD_DIR/$bin" ]]; then
      echo "  $BUILD_DIR/$bin"
    fi
  done
}

do_python() {
  need_libcuvs
  echo "[python] pip install $VECFLOW_DIR (no build isolation)"
  ( cd "$VECFLOW_DIR" && pip install . --no-build-isolation )
  echo
  echo "[python] done. Verify with (run from outside $VECFLOW_DIR so the source"
  echo "         vecflow/ dir doesn't shadow the installed package):"
  echo "  (cd /tmp && python -c 'import vecflow; print(vecflow.__version__); print(vecflow.VecFlow())')"
}

for t in "${TARGETS[@]}"; do
  case "$t" in
    clean)    do_clean ;;
    examples) do_examples ;;
    python)   do_python ;;
  esac
done
