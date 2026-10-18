#!/usr/bin/env bash
# One-click Bering benchmark for the centre 10000x10000 native-pixel ROIs of
# xenium9 and xenium10.  The datasets run sequentially on one selected GPU.
#
# Default (submit a detached worker):
#   /data/qiuyijia/bering/run_bering_xenium9_xenium10_center10000.sh
#
# Useful alternatives:
#   BERING_CUDA_VISIBLE_DEVICES=6 ./run_bering_xenium9_xenium10_center10000.sh
#   ./run_bering_xenium9_xenium10_center10000.sh --foreground
#   ./run_bering_xenium9_xenium10_center10000.sh --check
#   ./run_bering_xenium9_xenium10_center10000.sh --dry-run

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/qiuyijia/anaconda3/envs/bering/bin/python}"
RUNNER="${SCRIPT_DIR}/bering_xenium_benchmark.py"
CONFIG_DIR="${SCRIPT_DIR}/configs"
LOG_DIR="${SCRIPT_DIR}/logs"
LAUNCH_LOG="${LOG_DIR}/bering_xenium9_xenium10_center10000_launcher.log"
PID_FILE="${LOG_DIR}/bering_xenium9_xenium10_center10000.pid"
CUDA_DEVICE="${BERING_CUDA_VISIBLE_DEVICES:-6}"
MODE="background"

DATASETS=(
  "xenium9|${CONFIG_DIR}/xenium9_center10000.json|${LOG_DIR}/xenium9_center10000.log|/data/qiuyijia/bering/xenium9_center10000_output|/data/qiuyijia/dataset/xenium9/Xenium_Prime_Human_Skin_FFPE_outs.zip"
  "xenium10|${CONFIG_DIR}/xenium10_center10000.json|${LOG_DIR}/xenium10_center10000.log|/data/qiuyijia/bering/xenium10_center10000_output|/data/qiuyijia/dataset/xenium10/transcripts.zip"
)

usage() {
  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
  cat <<EOF
Options:
  --cuda-visible-devices N  Select physical GPU (default: ${CUDA_DEVICE})
  --foreground              Run sequentially in this terminal
  --check                   Extract if needed and verify schema/ROI only
  --dry-run                 Print commands without extracting or running
  -h, --help                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda-visible-devices)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --cuda-visible-devices requires a value" >&2
        exit 2
      fi
      CUDA_DEVICE="$2"; shift 2 ;;
    --foreground) MODE="foreground"; shift ;;
    --check) MODE="check"; shift ;;
    --dry-run) MODE="dry-run"; shift ;;
    -h|--help) usage; exit 0 ;;
    --worker) MODE="worker"; shift ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Required file does not exist: ${path}" >&2
    return 1
  fi
}

preflight() {
  if [[ "${PYTHON_BIN}" == */* ]]; then
    require_file "${PYTHON_BIN}" || return 1
  elif ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[ERROR] Python executable not found: ${PYTHON_BIN}" >&2
    return 1
  fi
  require_file "${RUNNER}" || return 1
  local item dataset config log output archive
  for item in "${DATASETS[@]}"; do
    IFS='|' read -r dataset config log output archive <<< "${item}"
    require_file "${config}" || return 1
    require_file "${archive}" || return 1
  done
  mkdir -p "${LOG_DIR}"
}

run_checks() {
  local item dataset config log output archive failures=0
  for item in "${DATASETS[@]}"; do
    IFS='|' read -r dataset config log output archive <<< "${item}"
    echo "[Check] ${dataset}"
    if ! CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
         BERING_CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
         "${PYTHON_BIN}" "${RUNNER}" --config "${config}" --check; then
      failures=$((failures + 1))
    fi
  done
  (( failures == 0 ))
}

is_complete() {
  local output="$1"
  [[ -s "${output}/transcript_assignments.parquet" ]] || return 1
  [[ -s "${output}/run_summary.json" ]] || return 1
  "${PYTHON_BIN}" - "${output}/run_summary.json" <<'PY' >/dev/null 2>&1
import json, sys
with open(sys.argv[1]) as handle:
    summary = json.load(handle)
stats = summary.get("prediction_tiles", {})
ok = summary.get("prediction_complete") is True
ok = ok and stats.get("total", 0) == stats.get("success", 0) + stats.get("skipped", 0)
ok = ok and stats.get("failed", 1) == 0
raise SystemExit(0 if ok else 1)
PY
}

run_worker() {
  local item dataset config log output archive rc failures=0
  {
    echo "=== Bering xenium9 + xenium10 centre-10000 worker ==="
    echo "Started: $(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "Python: ${PYTHON_BIN}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_DEVICE}"
    echo "Datasets run sequentially."
    echo
  } >> "${LAUNCH_LOG}"

  for item in "${DATASETS[@]}"; do
    IFS='|' read -r dataset config log output archive <<< "${item}"

    if is_complete "${output}"; then
      echo "[SKIP] ${dataset} is already complete: ${output}" | tee -a "${LAUNCH_LOG}"
      continue
    fi

    {
      echo "=== Bering ${dataset} centre 10000x10000 px ==="
      echo "Started: $(date '+%Y-%m-%d %H:%M:%S %z')"
      echo "Config: ${config}"
      echo "Output: ${output}"
      echo "CUDA_VISIBLE_DEVICES: ${CUDA_DEVICE}"
      echo "Command: ${PYTHON_BIN} ${RUNNER} --config ${config}"
      echo
    } > "${log}"
    echo "[Worker] Starting ${dataset}; log=${log}" >> "${LAUNCH_LOG}"

    if CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
       BERING_CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
       PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128" \
       /usr/bin/time -v "${PYTHON_BIN}" "${RUNNER}" --config "${config}" \
       >> "${log}" 2>&1; then
      rc=0
      echo "[DONE] ${dataset}: $(date '+%Y-%m-%d %H:%M:%S %z')" \
        | tee -a "${log}" >> "${LAUNCH_LOG}"
    else
      rc=$?
      failures=$((failures + 1))
      echo "[FAIL] ${dataset} (exit=${rc}): $(date '+%Y-%m-%d %H:%M:%S %z')" \
        | tee -a "${log}" >> "${LAUNCH_LOG}"
    fi
  done

  if (( failures != 0 )); then
    echo "[Summary] ${failures} dataset(s) failed." >> "${LAUNCH_LOG}"
    return 1
  fi
  echo "[Summary] Both datasets are complete: $(date '+%Y-%m-%d %H:%M:%S %z')" \
    >> "${LAUNCH_LOG}"
}

preflight || exit 1

if [[ "${MODE}" == "dry-run" ]]; then
  for item in "${DATASETS[@]}"; do
    IFS='|' read -r dataset config log output archive <<< "${item}"
    echo "[Dry-run] ${dataset}: CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} ${PYTHON_BIN} ${RUNNER} --config ${config}"
    echo "          output=${output}; log=${log}"
  done
  exit 0
fi

if [[ "${MODE}" == "check" ]]; then
  run_checks
  exit $?
fi

if [[ "${MODE}" == "foreground" || "${MODE}" == "worker" ]]; then
  run_worker
  exit $?
fi

# Avoid submitting the same detached sequential worker twice.
if [[ -s "${PID_FILE}" ]]; then
  old_pid="$(<"${PID_FILE}")"
  if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null && \
     ps -p "${old_pid}" -o args= | grep -Fq -- "run_bering_xenium9_xenium10_center10000.sh --worker"; then
    echo "Already running: PID=${old_pid}"
    echo "Launcher log: ${LAUNCH_LOG}"
    exit 0
  fi
fi

echo "=== Bering xenium9 + xenium10 centre-10000 launcher ===" > "${LAUNCH_LOG}"
echo "Submitted: $(date '+%Y-%m-%d %H:%M:%S %z')" >> "${LAUNCH_LOG}"
nohup env BERING_CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
  "${BASH:-bash}" "${SCRIPT_DIR}/run_bering_xenium9_xenium10_center10000.sh" \
    --worker --cuda-visible-devices "${CUDA_DEVICE}" \
  >> "${LAUNCH_LOG}" 2>&1 < /dev/null &
pid=$!
echo "${pid}" > "${PID_FILE}"

echo "Submitted xenium9 + xenium10 sequentially: PID=${pid}"
echo "Launcher log: ${LAUNCH_LOG}"
echo "Dataset logs: ${LOG_DIR}/xenium9_center10000.log and ${LOG_DIR}/xenium10_center10000.log"
