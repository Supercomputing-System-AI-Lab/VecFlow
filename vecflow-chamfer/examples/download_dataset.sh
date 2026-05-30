#!/usr/bin/env bash
# Download the bundled `lifestyle.test` multi-vector dataset from Google Drive
# into examples/datasets/lifestyle/, where the default config.json points.
#
# Produces (file names come from GDrive as-is):
#   examples/datasets/lifestyle/lifestyle.test.doc.embeddings.fp16.fbin
#   examples/datasets/lifestyle/lifestyle.test.doc.offsets.bin
#   examples/datasets/lifestyle/lifestyle.test.query.embeddings.fp16.fbin
#
# Ground truth is computed on the GPU the first time the example runs — no
# GT download required.
#
# Requires `gdown`. The script installs it via `pip install --user gdown` if it
# isn't already on $PATH.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DST="${SCRIPT_DIR}/datasets/lifestyle"

FOLDER_URL="https://drive.google.com/drive/folders/1YASw9P_P2Q2Ll-_eGBPrYCtedaEpSUnz"

EXPECTED=(
  "lifestyle.test.doc.embeddings.fp16.fbin"
  "lifestyle.test.doc.offsets.bin"
  "lifestyle.test.query.embeddings.fp16.fbin"
)

if ! command -v gdown >/dev/null 2>&1; then
  echo "==> gdown not found, installing via pip --user"
  python -m pip install --user --quiet gdown
  # `pip --user` install location may not be on PATH yet.
  export PATH="$(python -c 'import site; print(site.USER_BASE)')/bin:${PATH}"
fi

mkdir -p "${DST}"

# Skip the download if every expected file is already present with non-zero
# size. Lets users re-run the script as a no-op verifier.
all_present=1
for f in "${EXPECTED[@]}"; do
  [[ -s "${DST}/${f}" ]] || { all_present=0; break; }
done

if (( all_present )); then
  echo "==> all dataset files already present, skipping download"
else
  echo "==> downloading lifestyle.test dataset folder into:"
  echo "    ${DST}"
  # gdown --folder lands files inside `<-O>/<gdrive-folder-name>/`. Stage into
  # a temp dir so we don't care what the gdrive folder is named — we just
  # promote whatever files we got into ${DST}.
  TMP=$(mktemp -d)
  trap 'rm -rf "${TMP}"' EXIT
  gdown --folder "${FOLDER_URL}" -O "${TMP}"
  # Flatten any single nested subdir, then move into place.
  shopt -s dotglob nullglob
  files=( "${TMP}"/* )
  if (( ${#files[@]} == 1 )) && [[ -d "${files[0]}" ]]; then
    mv "${files[0]}"/* "${DST}/"
  else
    mv "${TMP}"/* "${DST}/"
  fi
  shopt -u dotglob nullglob
fi

echo
echo "==> verifying files"

# Render a byte count as KB/MB/GB (1024-based, 2 decimals). Pure shell so we
# don't depend on `numfmt` or `bc` being available.
human_size() {
  local bytes=$1
  if   (( bytes >= 1073741824 )); then awk -v b="$bytes" 'BEGIN{ printf "%.2f GB", b/1073741824 }'
  elif (( bytes >= 1048576    )); then awk -v b="$bytes" 'BEGIN{ printf "%.2f MB", b/1048576    }'
  elif (( bytes >= 1024       )); then awk -v b="$bytes" 'BEGIN{ printf "%.2f KB", b/1024       }'
  else                                 printf "%d B" "$bytes"
  fi
}

missing=0
for f in "${EXPECTED[@]}"; do
  if [[ -s "${DST}/${f}" ]]; then
    size=$(stat -c %s "${DST}/${f}" 2>/dev/null || stat -f %z "${DST}/${f}")
    printf "    %-50s %10s\n" "${f}" "$(human_size "${size}")"
  else
    printf "    %-50s MISSING\n" "${f}"
    missing=1
  fi
done

if (( missing )); then
  echo
  echo "error: some expected files were not in the folder — check ${DST}" >&2
  exit 1
fi
