#!/usr/bin/env bash
# One-click Cellpose MERFISH segmentation for the seven benchmark datasets.
#
# Usage:
#   /data/qiuyijia/cellpose/run_merfish_all.sh
#
# Optional environment variables:
#   PYTHON_BIN             Python executable (default: cellpose conda env)
#   MERFISH_DATA_ROOT     Dataset root (default: /data/qiuyijia/dataset)
#   MERFISH_OUTPUT_ROOT   Output root (default: this directory)
#   MERFISH_TEST_PY_PATH  Evaluation module path
#   MERFISH_LOG_ROOT      Per-dataset log directory
#   MERFISH_DRY_RUN=1     Print commands without running them

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
METHOD_SCRIPT="${SCRIPT_DIR}/merfish_seg_bench.py"
DATA_ROOT="${MERFISH_DATA_ROOT:-/data/qiuyijia/dataset}"
OUTPUT_ROOT="${MERFISH_OUTPUT_ROOT:-${SCRIPT_DIR}}"
TEST_PY_PATH="${MERFISH_TEST_PY_PATH:-/data/qiuyijia/eval/test.py}"
LOG_ROOT="${MERFISH_LOG_ROOT:-${OUTPUT_ROOT}/merfish_batch_logs}"
if [[ -z "${PYTHON_BIN+x}" ]]; then
  if [[ -x /data/qiuyijia/anaconda3/envs/cellpose/bin/python ]]; then
    PYTHON_BIN="/data/qiuyijia/anaconda3/envs/cellpose/bin/python"
  elif command -v python >/dev/null 2>&1; then PYTHON_BIN="python"
  else PYTHON_BIN="python3"; fi
fi
DRY_RUN="${MERFISH_DRY_RUN:-0}"

DATASETS=(
  merfish_Melanoma
  merfish_ovarian1
  merfish_ovarian2
  merfish_colon1
  merfish_colon2
  merfish_Uterian
  merfish_breast_cancer
)

find_image_dir() {
  local data_dir="$1"
  if [[ -f "${data_dir}/images/mosaic_DAPI_z3.tif" ]]; then
    printf '%s\n' "${data_dir}/images"
  elif [[ -f "${data_dir}/imgs/mosaic_DAPI_z3.tif" ]]; then
    printf '%s\n' "${data_dir}/imgs"
  else
    return 1
  fi
}

echo "[Preflight] method script: ${METHOD_SCRIPT}"
echo "[Preflight] data root:      ${DATA_ROOT}"
echo "[Preflight] output root:    ${OUTPUT_ROOT}"

missing=0
if [[ ! -f "${METHOD_SCRIPT}" ]]; then
  echo "[ERROR] Missing method script: ${METHOD_SCRIPT}" >&2
  missing=1
fi
if [[ ! -f "${TEST_PY_PATH}" ]]; then
  echo "[ERROR] Missing evaluation script: ${TEST_PY_PATH}" >&2
  missing=1
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] Python executable not found: ${PYTHON_BIN}" >&2
  missing=1
fi

for dataset in "${DATASETS[@]}"; do
  data_dir="${DATA_ROOT}/${dataset}"
  if ! image_dir="$(find_image_dir "${data_dir}")"; then
    echo "[ERROR] ${dataset}: cannot find images/mosaic_DAPI_z3.tif or imgs/mosaic_DAPI_z3.tif" >&2
    missing=1
    continue
  fi
  required_files=(
    "${image_dir}/mosaic_DAPI_z3.tif"
    "${image_dir}/mosaic_Cellbound1_z3.tif"
    "${image_dir}/micron_to_mosaic_pixel_transform.csv"
    "${data_dir}/cell_metadata.csv"
  )
  for path in "${required_files[@]}" "${data_dir}/cell_boundaries"; do
    if [[ ! -e "${path}" ]]; then
      echo "[ERROR] ${dataset}: missing ${path}" >&2
      missing=1
    fi
  done
done

if (( missing != 0 )); then
  echo "[Preflight] failed; no dataset was started." >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}"
failures=()

run_dataset() {
  local dataset="$1"
  local data_dir="${DATA_ROOT}/${dataset}"
  local image_dir
  image_dir="$(find_image_dir "${data_dir}")"
  local output_dir="${OUTPUT_ROOT}/${dataset}"
  local log_path="${LOG_ROOT}/${dataset}.log"
  local -a command_args=(
    "${METHOD_SCRIPT}"
    --data-dir "${data_dir}"
    --dapi-tif "${image_dir}/mosaic_DAPI_z3.tif"
    --polyt-tif "${image_dir}/mosaic_Cellbound1_z3.tif"
    --transform-csv "${image_dir}/micron_to_mosaic_pixel_transform.csv"
    --cell-metadata-csv "${data_dir}/cell_metadata.csv"
    --cell-boundaries-dir "${data_dir}/cell_boundaries"
    --test-py-path "${TEST_PY_PATH}"
    --out-dir "${output_dir}"
  )

  mkdir -p "${output_dir}"
  {
    echo "=== Cellpose ${dataset} ==="
    echo "Started: $(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "Input:   ${data_dir}"
    echo "Output:  ${output_dir}"
    printf 'Command: %q ' "${PYTHON_BIN}" "${command_args[@]}"
    echo
  } >"${log_path}"

  echo "[Run] Cellpose ${dataset} (log: ${log_path})"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[Dry-run] ${PYTHON_BIN} ${command_args[*]}" | tee -a "${log_path}"
    return 0
  fi

  "${PYTHON_BIN}" "${command_args[@]}" 2>&1 | tee -a "${log_path}"
  local rc=${PIPESTATUS[0]}
  if (( rc == 0 )); then
    echo "[Done] ${dataset}" | tee -a "${log_path}"
  else
    echo "[FAIL] ${dataset} (exit=${rc})" | tee -a "${log_path}" >&2
    failures+=("${dataset}")
  fi
}

for dataset in "${DATASETS[@]}"; do
  run_dataset "${dataset}"
done

if (( ${#failures[@]} != 0 )); then
  echo "[Summary] failed datasets: ${failures[*]}" >&2
  exit 1
fi
echo "[Summary] all seven Cellpose datasets completed."
