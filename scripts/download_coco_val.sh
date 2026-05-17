#!/usr/bin/env bash
# Download COCO val2017 images + annotations into data/coco_val/.
#
# Usage (from repo root):
#   bash scripts/download_coco_val.sh
#
# Idempotent: skips download if files already present.
# Final layout:
#   data/coco_val/images/000000000139.jpg ... (5000 files)
#   data/coco_val/annotations/instances_val2017.json
#
# Disk: ~1 GB during download (zip), ~780 MB final (zip removed).

set -euo pipefail

DATA_DIR="data/coco_val"
IMG_DIR="${DATA_DIR}/images"
ANN_DIR="${DATA_DIR}/annotations"
IMG_ZIP="${DATA_DIR}/val2017.zip"
ANN_ZIP="${DATA_DIR}/annotations_trainval2017.zip"

IMG_URL="http://images.cocodataset.org/zips/val2017.zip"
ANN_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

EXPECTED_IMAGES=5000
EXPECTED_ANN="instances_val2017.json"

mkdir -p "${DATA_DIR}"

# ---------- images ----------
if [[ -d "${IMG_DIR}" ]] && [[ $(find "${IMG_DIR}" -maxdepth 1 -name '*.jpg' | wc -l) -eq ${EXPECTED_IMAGES} ]]; then
  echo "[skip] images already present: ${IMG_DIR} (${EXPECTED_IMAGES} files)"
else
  if [[ ! -f "${IMG_ZIP}" ]]; then
    echo "[get ] ${IMG_URL}"
    wget --show-progress -O "${IMG_ZIP}" "${IMG_URL}"
  else
    echo "[skip] zip already on disk: ${IMG_ZIP}"
  fi
  echo "[unzip] ${IMG_ZIP}"
  unzip -q -o "${IMG_ZIP}" -d "${DATA_DIR}"
  # zip extracts to data/coco_val/val2017/*.jpg -- rename to images/
  if [[ -d "${DATA_DIR}/val2017" ]]; then
    rm -rf "${IMG_DIR}"
    mv "${DATA_DIR}/val2017" "${IMG_DIR}"
  fi
  rm -f "${IMG_ZIP}"
fi

# ---------- annotations ----------
if [[ -f "${ANN_DIR}/${EXPECTED_ANN}" ]]; then
  echo "[skip] annotations already present: ${ANN_DIR}/${EXPECTED_ANN}"
else
  if [[ ! -f "${ANN_ZIP}" ]]; then
    echo "[get ] ${ANN_URL}"
    wget --show-progress -O "${ANN_ZIP}" "${ANN_URL}"
  else
    echo "[skip] zip already on disk: ${ANN_ZIP}"
  fi
  echo "[unzip] ${ANN_ZIP}"
  unzip -q -o "${ANN_ZIP}" -d "${DATA_DIR}"
  rm -f "${ANN_ZIP}"
fi

# ---------- verify ----------
n_imgs=$(find "${IMG_DIR}" -maxdepth 1 -name '*.jpg' | wc -l)
ann_size=$(stat -c%s "${ANN_DIR}/${EXPECTED_ANN}")
echo ""
echo "[done] images: ${n_imgs}     annotations: ${ANN_DIR}/${EXPECTED_ANN} (${ann_size} bytes)"
if [[ ${n_imgs} -ne ${EXPECTED_IMAGES} ]]; then
  echo "[WARN] expected ${EXPECTED_IMAGES} images, got ${n_imgs}" >&2
  exit 1
fi
echo "[OK  ] COCO val2017 ready under ${DATA_DIR}/"
