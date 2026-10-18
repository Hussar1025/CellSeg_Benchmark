#!/usr/bin/env bash
# One-click background Bering benchmark for xenium7 and xenium_8.
#
# The two jobs are intentionally run sequentially because the benchmark
# runner selects one GPU (CUDA device 6).  This script itself detaches a
# worker with nohup, so the terminal can be closed after submission.
#
# Usage:
#   /data/qiuyijia/bering/run_bering_xenium_center10000.sh
#
# Logs:
#   /data/qiuyijia/bering/logs/bering_xenium_center10000_launcher.log
#   /data/qiuyijia/bering/logs/xenium7_center10000.log
#   /data/qiuyijia/bering/logs/xenium_8_center10000.log
#
# Monitor:
#   tail -f /data/qiuyijia/bering/logs/xenium7_center10000.log
#   cat /data/qiuyijia/bering/logs/bering_xenium_center10000.pid

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/qiuyijia/anaconda3/envs/bering/bin/python}"
RUNNER="${SCRIPT_DIR}/bering_xenium_benchmark.py"
CONFIG_DIR="${SCRIPT_DIR}/configs"
LOG_DIR="${SCRIPT_DIR}/logs"
LAUNCH_LOG="${LOG_DIR}/bering_xenium_center10000_launcher.log"
PID_FILE="${LOG_DIR}/bering_xenium_center10000.pid"
DRY_RUN="${BERING_DRY_RUN:-0}"

DATASETS=(
  "xenium7|${CONFIG_DIR}/xenium7_center10000.json|${LOG_DIR}/xenium7_center10000.log"
  "xenium_8|${CONFIG_DIR}/xenium_8_center10000.json|${LOG_DIR}/xenium_8_center10000.log"
)

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
    echo "[ERROR] Python executable not found in PATH: ${PYTHON_BIN}" >&2
    return 1
  fi
  require_file "${RUNNER}" || return 1
  local item dataset cfg log transcript
  for item in "${DATASETS[@]}"; do
    IFS='|' read -r dataset cfg log <<< "${item}"
    require_file "${cfg}" || return 1
    transcript=""
    case "${dataset}" in
      xenium7) transcript="/data/qiuyijia/dataset/xenium7/outs/transcripts.parquet" ;;
      xenium_8) transcript="/data/qiuyijia/dataset/xenium_8/transcripts.parquet" ;;
    esac
    require_file "${transcript}" || return 1
  done
  mkdir -p "${LOG_DIR}"
}

check_inputs() {
  local item dataset cfg log
  for item in "${DATASETS[@]}"; do
    IFS='|' read -r dataset cfg log <<< "${item}"
    echo "[Preflight] ${dataset}" 
    "${PYTHON_BIN}" "${RUNNER}" --config "${cfg}" --check || return 1
  done
}

run_worker() {
  local item dataset cfg log rc failures=0
  {
    echo "=== Bering Xenium center-10000 worker ==="
    echo "Started: $(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "Python: ${PYTHON_BIN}"
    echo "Runner: ${RUNNER}"
    echo
  } >> "${LAUNCH_LOG}"

  for item in "${DATASETS[@]}"; do
    IFS='|' read -r dataset cfg log <<< "${item}"
    {
      echo "=== Bering ${dataset} ==="
      echo "Started: $(date '+%Y-%m-%d %H:%M:%S %z')"
      echo "Config: ${cfg}"
      echo "Command: ${PYTHON_BIN} ${RUNNER} --config ${cfg}"
      echo
    } > "${log}"
    echo "[Worker] Starting ${dataset}; log=${log}" >> "${LAUNCH_LOG}"
    if "${PYTHON_BIN}" "${RUNNER}" --config "${cfg}" >> "${log}" 2>&1; then
      rc=0
      echo "[DONE] ${dataset}: $(date '+%Y-%m-%d %H:%M:%S %z')" | tee -a "${log}" >> "${LAUNCH_LOG}"
    else
      rc=$?
      failures=$((failures + 1))
      echo "[FAIL] ${dataset} (exit=${rc}): $(date '+%Y-%m-%d %H:%M:%S %z')" | tee -a "${log}" >> "${LAUNCH_LOG}"
    fi
  done

  if (( failures != 0 )); then
    echo "[Summary] ${failures} dataset(s) failed." >> "${LAUNCH_LOG}"
    return 1
  fi
  echo "[Summary] Both datasets completed: $(date '+%Y-%m-%d %H:%M:%S %z')" >> "${LAUNCH_LOG}"
}

if [[ "${1:-}" == "--worker" ]]; then
  preflight || exit 1
  check_inputs || exit 1
  if [[ "${DRY_RUN}" == "1" ]]; then
    for item in "${DATASETS[@]}"; do
      IFS='|' read -r dataset cfg log <<< "${item}"
      echo "[Dry-run] ${PYTHON_BIN} ${RUNNER} --config ${cfg} (log=${log})"
    done
    exit 0
  fi
  run_worker
  exit $?
fi

preflight || exit 1
check_inputs || { echo "[ERROR] Preflight failed; no job was submitted." >&2; exit 1; }

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[Dry-run] Worker would be submitted with nohup; no process started."
  for item in "${DATASETS[@]}"; do
    IFS='|' read -r dataset cfg log <<< "${item}"
    echo "[Dry-run] ${dataset}: ${PYTHON_BIN} ${RUNNER} --config ${cfg} >> ${log} 2>&1"
  done
  exit 0
fi

# Avoid accidentally submitting a duplicate worker.
if [[ -s "${PID_FILE}" ]]; then
  old_pid="$(<"${PID_FILE}")"
  if kill -0 "${old_pid}" 2>/dev/null && ps -p "${old_pid}" -o args= | grep -Fq -- "${SCRIPT_DIR}/run_bering_xenium_center10000.sh --worker"; then
    echo "Already running: PID=${old_pid}"
    echo "Launcher log: ${LAUNCH_LOG}"
    exit 0
  fi
  rm -f "${PID_FILE}"
fi

mkdir -p "${LOG_DIR}"
echo "=== Bering Xenium center-10000 launcher ===" > "${LAUNCH_LOG}"
echo "Submitted: $(date '+%Y-%m-%d %H:%M:%S %z')" >> "${LAUNCH_LOG}"
nohup "${BASH:-bash}" "${SCRIPT_DIR}/run_bering_xenium_center10000.sh" --worker \
  >> "${LAUNCH_LOG}" 2>&1 < /dev/null &
pid=$!
echo "${pid}" > "${PID_FILE}"
echo "Submitted xenium7 + xenium_8 sequentially: PID=${pid}"
echo "Launcher log: ${LAUNCH_LOG}"
echo "Per-dataset logs: ${LOG_DIR}/xenium7_center10000.log and ${LOG_DIR}/xenium_8_center10000.log"
