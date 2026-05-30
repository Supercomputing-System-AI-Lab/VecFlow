#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Local sanity check for the conda recipes and release pipeline.
# Builds all four conda packages locally, installs them into a throwaway
# env, and verifies each artifact:
#
#   libcuvs-vecflow-cu12      C++ cuVS + VecFlow filtered-ANN extensions
#   vecflow-cu12              Python wrapper (filtered-ANN)
#   libvecflow-chamfer-cu12   C++ multi-vector / chamfer-rerank pipeline
#   vecflow-chamfer-cu12      Python wrapper (multi-vector)
#
# Usage:
#     bash scripts/local_release_test.sh
#
# Honors the following env vars:
#     PLATFORM      target subdir (default: linux-aarch64)
#     PY            Python version for the Python packages (default: 3.12)
#     TEST_DIR      output dir for .conda artifacts (default: /tmp/vecflow-conda-test)
#     ENV           conda env name to install into (default: vf-localtest)
#     SKIP_CHAMFER  set to anything to skip the chamfer build/install/test
#     SKIP_VECFLOW  set to anything to skip the vecflow Python build/install/test
#                   (libcuvs-vecflow is always built — chamfer depends on it)
#     SKIP_GPU_TEST set to anything to skip the final GPU functional tests
#
# Total wall-time: ~35–65 min (dominated by step 2, the full libcuvs build).
# Re-running skips already-built .conda artifacts; delete a specific .conda
# file in $TEST_DIR/$PLATFORM/ to force just that package to rebuild.

set -euo pipefail

# ---- Config -----------------------------------------------------------------
REPO=$(cd "$(dirname "$0")/.." && pwd)
TEST_DIR=${TEST_DIR:-/tmp/vecflow-conda-test}
ENV=${ENV:-vf-localtest}
PY=${PY:-3.12}
PLATFORM=${PLATFORM:-linux-aarch64}

START=$(date +%s)

log()  { printf "\n\033[1;36m== %s ==\033[0m\n" "$*"; }
ok()   { printf "  \033[1;32mok:\033[0m %s\n" "$*"; }
warn() { printf "  \033[1;33mwarn:\033[0m %s\n" "$*"; }
fail() { printf "  \033[1;31mfail:\033[0m %s\n" "$*"; exit 1; }

cd "$REPO"
mkdir -p "$TEST_DIR"

# Recipes read these via `env.get(...)` — single source of truth is the
# vecflow/VERSION file (and the cuVS-fork's top-level VERSION for the
# upstream pin).
export VECFLOW_VERSION="$(tr -d '[:space:]' < "$REPO/vecflow/VERSION")"
export CUVS_MINOR_VERSION="$(tr -d '[:space:]' < "$REPO/VERSION" | awk -F. '{print $1"."$2}')"

# Default ninja/nvcc parallelism: min(nproc-2, total_ram_gb / 4). The 4 GB/job
# bound is empirical for cuVS — large fatbin compiles peak there. Override
# by exporting PARALLEL_LEVEL before running this script.
if [[ -z "${PARALLEL_LEVEL:-}" ]]; then
    cores=$(nproc 2>/dev/null || echo 8)
    ram_gb=$(free -g 2>/dev/null | awk '/^Mem:/ {print $2}'); ram_gb=${ram_gb:-32}
    p_cores=$(( cores - 2 ))
    p_ram=$(( ram_gb / 4 ))
    export PARALLEL_LEVEL=$(( p_cores < p_ram ? p_cores : p_ram ))
    [[ $PARALLEL_LEVEL -lt 1 ]] && PARALLEL_LEVEL=1
fi

# ---- 0. Tools ---------------------------------------------------------------
log "0/10 verifying local toolchain"
command -v rattler-build >/dev/null \
    || fail "rattler-build not in PATH. Run: mamba install -n base -c conda-forge rattler-build anaconda-client"
command -v mamba >/dev/null || fail "mamba not in PATH"
echo "  rattler-build: $(rattler-build --version)"
echo "  platform:      $PLATFORM"
echo "  output dir:    $TEST_DIR"
echo "  env name:      $ENV"
echo "  python:        $PY"
echo "  vecflow ver:   $VECFLOW_VERSION"
echo "  cuvs minor:    $CUVS_MINOR_VERSION"
echo "  parallelism:   $PARALLEL_LEVEL ($(nproc 2>/dev/null) cores, $(free -g 2>/dev/null | awk '/^Mem:/ {print $2}') GB RAM)"
ok "tools present"

# Build the list of recipes to exercise based on the SKIP_* env vars.
# libcuvs-vecflow is always present (chamfer's C++ lib depends on it).
RECIPES=(libcuvs-vecflow)
if [[ -z "${SKIP_VECFLOW:-}" ]]; then RECIPES+=(vecflow); fi
if [[ -z "${SKIP_CHAMFER:-}" ]]; then RECIPES+=(libvecflow-chamfer vecflow-chamfer); fi

TOTAL_STEPS=10

# ---- 1. Render-only smoke check (~10 sec per recipe) ------------------------
log "1/$TOTAL_STEPS render-only check (catches recipe.yaml syntax errors fast)"
for recipe in "${RECIPES[@]}"; do
    if rattler-build build --help 2>&1 | grep -q -- '--render-only'; then
        rattler-build build \
            --recipe "$REPO/conda/recipes/$recipe/recipe.yaml" \
            --target-platform "$PLATFORM" \
            --output-dir "$TEST_DIR" \
            --channel rapidsai-nightly --channel rapidsai --channel conda-forge \
            --render-only \
            > "$TEST_DIR/render-$recipe.log" 2>&1 \
            || { tail -30 "$TEST_DIR/render-$recipe.log"; fail "$recipe render failed; see $TEST_DIR/render-$recipe.log"; }
        ok "$recipe recipe renders cleanly"
    else
        warn "rattler-build lacks --render-only; relying on full build to surface parse errors"
        break
    fi
done

# ---- 2. Build C++ package (~30–60 min) --------------------------------------
log "2/$TOTAL_STEPS building libcuvs-vecflow (slow — full cuVS compile)"
LIBCUVS_PKG=$(ls "$TEST_DIR/$PLATFORM"/libcuvs-vecflow-cu12-*.conda 2>/dev/null | head -1 || true)
if [[ -n "$LIBCUVS_PKG" ]]; then
    warn "already built: $LIBCUVS_PKG (delete it to force a rebuild)"
else
    rattler-build build \
        --recipe "$REPO/conda/recipes/libcuvs-vecflow/recipe.yaml" \
        --target-platform "$PLATFORM" \
        --output-dir "$TEST_DIR" \
        --channel rapidsai-nightly --channel rapidsai --channel conda-forge \
        --skip-existing all \
        2>&1 | tee "$TEST_DIR/libcuvs-build.log"
    LIBCUVS_PKG=$(ls "$TEST_DIR/$PLATFORM"/libcuvs-vecflow-cu12-*.conda | head -1)
fi
SIZE_MB=$(du -m "$LIBCUVS_PKG" | cut -f1)
ok "libcuvs-vecflow package: $LIBCUVS_PKG ($SIZE_MB MB)"

# ---- 3. Install + verify VecFlow symbols are in libcuvs.so -------------------
log "3/$TOTAL_STEPS install C++ package + verify VecFlow symbols"

# Reuse the env if it already exists and has the matching libcuvs-vecflow build —
# saves the ~2 min of CUDA-toolkit relinking on every iteration. Only nuke +
# recreate when the env doesn't exist or the build hash drifted.
LOCAL_BUILD=$(basename "$LIBCUVS_PKG" \
    | sed -E 's/^libcuvs-vecflow-cu12-[^-]+-(.+)\.conda$/\1/')

# Detect installed build hash (if any) without tripping `set -euo pipefail` when
# the env doesn't yet exist.
INSTALLED_BUILD=""
if mamba env list 2>/dev/null | awk '{print $1}' | grep -qx "$ENV"; then
    INSTALLED_BUILD=$(mamba list -n "$ENV" 2>/dev/null | \
        awk '$1 == "libcuvs-vecflow-cu12" {print $3}' || true)
fi

if [[ -n "$INSTALLED_BUILD" && "$INSTALLED_BUILD" == "$LOCAL_BUILD" ]]; then
    ok "env $ENV already has libcuvs-vecflow $LOCAL_BUILD; skipping reinstall"
else
    if [[ -n "$INSTALLED_BUILD" ]]; then
        warn "env build $INSTALLED_BUILD != local $LOCAL_BUILD; recreating"
    else
        warn "env $ENV missing or empty; creating from scratch"
    fi
    mamba env remove -n "$ENV" -y --quiet 2>/dev/null || true
    # Also forcibly nuke any stale env directory (leftover from a killed
    # `mamba create`). mamba refuses to overwrite a non-empty non-conda dir.
    for d in $(mamba info --json 2>/dev/null \
                 | python -c "import json,sys;d=json.load(sys.stdin);print(' '.join(d.get('envs_dirs',[])))" \
                 2>/dev/null || true); do
        [[ -d "$d/$ENV" ]] && rm -rf "$d/$ENV" || true
    done
    echo "  (mamba solver + CUDA toolkit fetch can take 2-5 min on first run;"
    echo "   solver phase is silent — wait for the package list to appear)"
    # Run mamba directly to the terminal: with `tee` in the pipeline, mamba
    # detects non-TTY and disables progress bars, plus tee buffers the
    # solver's silent phase, making it look like the script hung. Direct
    # output is the cleanest UX. Re-run as `… | tee log` if you want a
    # capture for debugging.
    mamba create -n "$ENV" -y \
        -c "file://$TEST_DIR" \
        -c rapidsai-nightly -c rapidsai -c conda-forge \
        libcuvs-vecflow-cu12 \
        || fail "libcuvs install failed"
fi

CONDA_PREFIX_VF=$(mamba run -n "$ENV" sh -c 'echo $CONDA_PREFIX')
[[ -f "$CONDA_PREFIX_VF/lib/libcuvs.so" ]]                                  || fail "libcuvs.so missing"
[[ -f "$CONDA_PREFIX_VF/include/cuvs/neighbors/vecflow.hpp" ]]              || fail "vecflow.hpp missing"
[[ -f "$CONDA_PREFIX_VF/include/cuvs/neighbors/filtered_bfs.hpp" ]]         || fail "filtered_bfs.hpp missing"
[[ -f "$CONDA_PREFIX_VF/include/cuvs/neighbors/shared_resources.hpp" ]]     || fail "shared_resources.hpp missing"
[[ -f "$CONDA_PREFIX_VF/lib/cmake/cuvs/cuvs-config.cmake" ]]                || fail "cuvs-config.cmake missing"

FILTERED=$(nm -D "$CONDA_PREFIX_VF/lib/libcuvs.so" 2>/dev/null | grep -c filtered_search    || true)
VECFLOW=$( nm -D "$CONDA_PREFIX_VF/lib/libcuvs.so" 2>/dev/null | grep -c vecflow            || true)
FILTERED_BFS=$(nm -D "$CONDA_PREFIX_VF/lib/libcuvs.so" 2>/dev/null | grep -c filtered_bfs   || true)
[[ "$FILTERED" -gt 0 ]]     || fail "no filtered_search symbols in libcuvs.so (recipe likely didn't pick up VecFlow patches)"
[[ "$VECFLOW" -gt 0 ]]      || fail "no vecflow symbols in libcuvs.so"
[[ "$FILTERED_BFS" -gt 0 ]] || fail "no filtered_bfs symbols in libcuvs.so"
ok "symbols: filtered_search=$FILTERED  vecflow=$VECFLOW  filtered_bfs=$FILTERED_BFS"

# ── chamfer C++ ──────────────────────────────────────────────────────────────
if [[ -z "${SKIP_CHAMFER:-}" ]]; then

# ---- 4. Build libvecflow-chamfer (~3–5 min) ---------------------------------
log "4/$TOTAL_STEPS building libvecflow-chamfer (small lib, pulls libcuvs-vecflow from local channel)"
CHAMFER_PKG=$(ls "$TEST_DIR/$PLATFORM"/libvecflow-chamfer-cu12-*.conda 2>/dev/null | head -1 || true)
if [[ -n "$CHAMFER_PKG" ]]; then
    warn "already built: $CHAMFER_PKG (delete it to force a rebuild)"
else
    rattler-build build \
        --recipe "$REPO/conda/recipes/libvecflow-chamfer/recipe.yaml" \
        --target-platform "$PLATFORM" \
        --output-dir "$TEST_DIR" \
        --channel "file://$TEST_DIR" \
        --channel rapidsai-nightly --channel rapidsai --channel conda-forge \
        --skip-existing all \
        2>&1 | tee "$TEST_DIR/libvecflow-chamfer-build.log"
    CHAMFER_PKG=$(ls "$TEST_DIR/$PLATFORM"/libvecflow-chamfer-cu12-*.conda | head -1)
fi
SIZE_MB=$(du -m "$CHAMFER_PKG" | cut -f1)
ok "libvecflow-chamfer package: $CHAMFER_PKG ($SIZE_MB MB)"

# ---- 5. Install + verify chamfer C++ artifacts ------------------------------
log "5/$TOTAL_STEPS install libvecflow-chamfer + verify symbols/headers/link"

CHAMFER_LOCAL_BUILD=$(basename "$CHAMFER_PKG" \
    | sed -E 's/^libvecflow-chamfer-cu12-[^-]+-(.+)\.conda$/\1/')
CHAMFER_INSTALLED_BUILD=$(mamba list -n "$ENV" 2>/dev/null | \
    awk '$1 == "libvecflow-chamfer-cu12" {print $3}' || true)

if [[ -n "$CHAMFER_INSTALLED_BUILD" && "$CHAMFER_INSTALLED_BUILD" == "$CHAMFER_LOCAL_BUILD" ]]; then
    ok "env $ENV already has libvecflow-chamfer $CHAMFER_LOCAL_BUILD; skipping reinstall"
else
    mamba install -n "$ENV" -y \
        -c "file://$TEST_DIR" \
        -c rapidsai-nightly -c rapidsai -c conda-forge \
        libvecflow-chamfer-cu12 \
        || fail "libvecflow-chamfer install failed"
fi

CONDA_PREFIX_VF=$(mamba run -n "$ENV" sh -c 'echo $CONDA_PREFIX')
[[ -f "$CONDA_PREFIX_VF/lib/libvecflow_chamfer.so" ]]                                || fail "libvecflow_chamfer.so missing"
[[ -f "$CONDA_PREFIX_VF/lib/libvecflow_chamfer_kernels.so" ]]                        || fail "libvecflow_chamfer_kernels.so missing"
[[ -f "$CONDA_PREFIX_VF/include/vecflow_chamfer/build.hpp" ]]                        || fail "vecflow_chamfer/build.hpp missing"
[[ -f "$CONDA_PREFIX_VF/include/vecflow_chamfer/search.hpp" ]]                       || fail "vecflow_chamfer/search.hpp missing"
[[ -f "$CONDA_PREFIX_VF/include/vecflow_chamfer/io.hpp" ]]                           || fail "vecflow_chamfer/io.hpp missing"
[[ -f "$CONDA_PREFIX_VF/include/vecflow_chamfer/types.hpp" ]]                        || fail "vecflow_chamfer/types.hpp missing"
[[ -f "$CONDA_PREFIX_VF/include/vecflow_chamfer/kernels.hpp" ]]                      || fail "vecflow_chamfer/kernels.hpp missing"
[[ -f "$CONDA_PREFIX_VF/include/vecflow_chamfer/serialize.hpp" ]]                    || fail "vecflow_chamfer/serialize.hpp missing"
[[ -f "$CONDA_PREFIX_VF/lib/cmake/vecflow_chamfer/vecflow_chamfer-config.cmake" ]]   || fail "vecflow_chamfer-config.cmake missing"

CHAMFER_SYM=$(nm -D "$CONDA_PREFIX_VF/lib/libvecflow_chamfer_kernels.so" 2>/dev/null | grep -c chamfer_score || true)
[[ "$CHAMFER_SYM" -gt 0 ]]  || fail "no chamfer_score symbols in libvecflow_chamfer_kernels.so"

# Confirm chamfer's link to libcuvs.so is wired up — catches a recipe that
# accidentally drops the transitive cuvs runtime dep.
ldd "$CONDA_PREFIX_VF/lib/libvecflow_chamfer.so" 2>/dev/null | grep -q libcuvs.so \
    || fail "libvecflow_chamfer.so does NOT link libcuvs.so — recipe missing cuvs run dep?"
ok "symbols: chamfer_score=$CHAMFER_SYM; libcuvs.so linkage confirmed"

else
    warn "SKIP_CHAMFER set; skipping libvecflow-chamfer build + install (steps 4–5)"
fi

# ── vecflow Python ───────────────────────────────────────────────────────────
if [[ -z "${SKIP_VECFLOW:-}" ]]; then

# ---- 6. Build Python package (~3–5 min) -------------------------------------
log "6/$TOTAL_STEPS building vecflow Python package (uses local libcuvs-vecflow as host dep)"
PY_TAG="py${PY//./}"
VECFLOW_PKG=$(ls "$TEST_DIR/$PLATFORM"/vecflow-cu12-*${PY_TAG}*.conda 2>/dev/null | head -1 || true)
if [[ -n "$VECFLOW_PKG" ]]; then
    warn "already built: $VECFLOW_PKG (delete it to force a rebuild)"
else
    VARIANT_FILE=$(mktemp -t vecflow-variant-XXXX.yaml)
    cat > "$VARIANT_FILE" <<EOF
python:
  - "$PY"
EOF
    rattler-build build \
        --recipe "$REPO/conda/recipes/vecflow/recipe.yaml" \
        --target-platform "$PLATFORM" \
        --output-dir "$TEST_DIR" \
        --channel "file://$TEST_DIR" \
        --channel rapidsai-nightly --channel rapidsai --channel conda-forge \
        --variant-config "$VARIANT_FILE" \
        --skip-existing all \
        2>&1 | tee "$TEST_DIR/vecflow-build.log"
    rm -f "$VARIANT_FILE"
    VECFLOW_PKG=$(ls "$TEST_DIR/$PLATFORM"/vecflow-cu12-*.conda | head -1)
fi
SIZE_MB=$(du -m "$VECFLOW_PKG" | cut -f1)
ok "vecflow package: $VECFLOW_PKG ($SIZE_MB MB)"

# ---- 7. Install Python package + import smoke test --------------------------
log "7/$TOTAL_STEPS install vecflow + python import smoke test"
mamba install -n "$ENV" -y \
    -c "file://$TEST_DIR" \
    -c rapidsai-nightly -c rapidsai -c conda-forge \
    "vecflow-cu12" "python=$PY" \
    || fail "vecflow install failed"

mamba run -n "$ENV" python - <<'PY' || fail "python import / construction failed"
import vecflow
print("  module path:", vecflow.__file__)
print("  class      :", vecflow.VecFlow)
v = vecflow.VecFlow()
print("  instance   :", v)
PY
ok "vecflow Python wrapper imports + class is constructable"

else
    warn "SKIP_VECFLOW set; skipping vecflow Python build + install (steps 6–7)"
fi

# ── chamfer Python ───────────────────────────────────────────────────────────
if [[ -z "${SKIP_CHAMFER:-}" ]]; then

# ---- 8. Build vecflow-chamfer Python (~3–5 min) -----------------------------
log "8/$TOTAL_STEPS building vecflow-chamfer Python package (pulls both C++ pkgs from local channel)"
PY_TAG="py${PY//./}"
CHAMFER_PY_PKG=$(ls "$TEST_DIR/$PLATFORM"/vecflow-chamfer-cu12-*${PY_TAG}*.conda 2>/dev/null | head -1 || true)
if [[ -n "$CHAMFER_PY_PKG" ]]; then
    warn "already built: $CHAMFER_PY_PKG (delete it to force a rebuild)"
else
    VARIANT_FILE=$(mktemp -t vecflow-chamfer-variant-XXXX.yaml)
    cat > "$VARIANT_FILE" <<EOF
python:
  - "$PY"
EOF
    rattler-build build \
        --recipe "$REPO/conda/recipes/vecflow-chamfer/recipe.yaml" \
        --target-platform "$PLATFORM" \
        --output-dir "$TEST_DIR" \
        --channel "file://$TEST_DIR" \
        --channel rapidsai-nightly --channel rapidsai --channel conda-forge \
        --variant-config "$VARIANT_FILE" \
        --skip-existing all \
        2>&1 | tee "$TEST_DIR/vecflow-chamfer-build.log"
    rm -f "$VARIANT_FILE"
    CHAMFER_PY_PKG=$(ls "$TEST_DIR/$PLATFORM"/vecflow-chamfer-cu12-*.conda | head -1)
fi
SIZE_MB=$(du -m "$CHAMFER_PY_PKG" | cut -f1)
ok "vecflow-chamfer package: $CHAMFER_PY_PKG ($SIZE_MB MB)"

# ---- 9. Install vecflow-chamfer + import smoke test -------------------------
log "9/$TOTAL_STEPS install vecflow-chamfer + python import smoke test"
mamba install -n "$ENV" -y \
    -c "file://$TEST_DIR" \
    -c rapidsai-nightly -c rapidsai -c conda-forge \
    "vecflow-chamfer-cu12" "python=$PY" \
    || fail "vecflow-chamfer install failed"

# Import + construct the param classes. We deliberately don't call build()
# or search() here — those need a GPU, which the build node may not have.
# The full pipeline test runs in step 10 only when a GPU is visible.
mamba run -n "$ENV" python - <<'PY' || fail "vecflow_chamfer import / construction failed"
import vecflow_chamfer as vc
print("  module path :", vc.__file__)
print("  classes     :", vc.Dataset, vc.QuerySet, vc.Index)
ip = vc.IndexParams(n_anchors=1000, graph_degree=32, itopk_size=64, n_iter=2)
sp = vc.SearchParams(itopk=64, search_topk=8, refine_rate=10.0, final_topk=10)
print("  IndexParams :", ip.n_anchors, ip.graph_degree, ip.itopk_size, ip.n_iter)
print("  SearchParams:", sp.itopk, sp.search_topk, sp.refine_rate, sp.final_topk)
PY
ok "vecflow_chamfer Python wrapper imports + param classes constructable"

else
    warn "SKIP_CHAMFER set; skipping vecflow-chamfer Python build + install (steps 8–9)"
fi

# ---- 10. (Optional) GPU functional tests ------------------------------------
log "10/$TOTAL_STEPS (optional) end-to-end functional tests on a GPU"
if [[ -n "${SKIP_GPU_TEST:-}" ]]; then
    warn "SKIP_GPU_TEST set; skipping all GPU functional tests"
elif ! command -v nvidia-smi >/dev/null 2>&1; then
    warn "no nvidia-smi (login node?); skipping. Re-run on a GPU node:"
    warn "  salloc --gres=gpu:1 ... && bash scripts/local_release_test.sh"
elif ! nvidia-smi >/dev/null 2>&1; then
    warn "nvidia-smi present but no GPU visible; skipping"
else
    # vecflow side: SIFT1M example via cupy.
    if [[ -z "${SKIP_VECFLOW:-}" ]]; then
        if [[ ! -f "$REPO/vecflow/examples/datasets/sift1M/base.fbin" ]]; then
            warn "vecflow GPU test: SIFT1M dataset not at vecflow/examples/datasets/sift1M/; skipping"
        elif ! mamba run -n "$ENV" python -c 'import cupy' 2>/dev/null; then
            warn "vecflow GPU test: cupy not installed in $ENV; skipping. Install it with:"
            warn "  mamba install -n $ENV -c conda-forge cupy"
        else
            ( cd "$REPO/vecflow/examples" && \
              mamba run -n "$ENV" python python/vecflow_example.py ) \
                && ok "vecflow_example.py ran end-to-end" \
                || fail "vecflow GPU example crashed"
        fi
    fi

    # chamfer side: lifestyle dataset via the new Python binding. No cupy
    # needed — everything is numpy + the binding does H↔D internally.
    if [[ -z "${SKIP_CHAMFER:-}" ]]; then
        LIFESTYLE_DIR="$REPO/vecflow-chamfer/examples/datasets/lifestyle"
        if [[ ! -f "$LIFESTYLE_DIR/lifestyle.test.doc.embeddings.fp16.fbin" ]]; then
            warn "chamfer GPU test: lifestyle dataset not at $LIFESTYLE_DIR; skipping"
            warn "  fetch it with: bash $REPO/vecflow-chamfer/examples/download_dataset.sh"
        else
            mamba run -n "$ENV" env LIFESTYLE_DIR="$LIFESTYLE_DIR" \
                python - <<'PY' || fail "vecflow_chamfer GPU pipeline crashed"
import os, sys
import vecflow_chamfer as vc

d = os.environ["LIFESTYLE_DIR"]
ds = vc.Dataset.load(d, "lifestyle.test")
print(f"  docs={ds.num_docs}  tokens={ds.num_doc_tokens}  dim={ds.embedding_dim}")
qs = vc.QuerySet.load(d, "lifestyle.test", embedding_dim=ds.embedding_dim)
print(f"  queries={qs.num_queries}  tokens/q={qs.num_tokens_per_query}")

# Small build params — this is a smoke test, not a recall benchmark.
ip = vc.IndexParams(n_anchors=50000, graph_degree=32, itopk_size=128, n_iter=2)
idx = vc.build(ds, ip)
print(f"  built index (n_anchors={ip.n_anchors}, n_iter={ip.n_iter})")

sp = vc.SearchParams(itopk=64, search_topk=8, refine_rate=10.0, final_topk=10)
neighbors, distances, stats = vc.search(idx, ds, qs, sp, with_stats=True)
assert neighbors.shape == (qs.num_queries, sp.final_topk), \
    f"unexpected neighbors shape {neighbors.shape}"
assert distances.shape == (qs.num_queries, sp.final_topk), \
    f"unexpected distances shape {distances.shape}"
print(f"  search ok: shape={neighbors.shape}  "
      f"qps={stats.throughput_qps:.0f}  per_query={stats.avg_per_query_ms:.3f} ms")
PY
            ok "vecflow_chamfer pipeline ran end-to-end on lifestyle dataset"
        fi
    fi
fi

# ---- Summary ----------------------------------------------------------------
ELAPSED=$(( $(date +%s) - START ))
log "✓ all local-release checks passed in $((ELAPSED/60))m $((ELAPSED%60))s"
echo "Artifacts in: $TEST_DIR/$PLATFORM/"
ls -la "$TEST_DIR/$PLATFORM/"*.conda 2>/dev/null
echo ""
echo "End users pick what they need (commands they would run after the real"
echo "release is published to anaconda.org/VecFlow):"
echo ""
echo "  # filtered ANN only"
echo "  mamba install -c VecFlow -c rapidsai-nightly -c rapidsai -c conda-forge \\"
echo "                vecflow-cu12 python=$PY"
echo ""
echo "  # multi-vector retrieval only"
echo "  mamba install -c VecFlow -c rapidsai-nightly -c rapidsai -c conda-forge \\"
echo "                vecflow-chamfer-cu12 python=$PY"
echo ""
echo "  # both products in one env"
echo "  mamba install -c VecFlow -c rapidsai-nightly -c rapidsai -c conda-forge \\"
echo "                vecflow-cu12 vecflow-chamfer-cu12 python=$PY"
echo ""
echo "To remove the test env:"
echo "    mamba env remove -n $ENV -y"
echo ""
echo "Next steps for the real release:"
echo "  1. Set ANACONDA_TOKEN secret on the repo (Settings → Secrets → Actions)"
echo "  2. Bump vecflow/VERSION (both packages read from this file)"
echo "  3. git tag vX.Y.Z && git push --tags"
echo "  4. Watch .github/workflows/release.yml; artifacts land at anaconda.org/VecFlow"
