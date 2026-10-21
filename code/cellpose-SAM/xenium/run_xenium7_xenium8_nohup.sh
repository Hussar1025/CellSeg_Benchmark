#!/usr/bin/env bash
set -u -o pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-yourpath/anaconda3/envs/cellpose/bin/python}"
METHOD_SCRIPT="${SCRIPT_DIR}/xenium_DAPI.py"; LOG="${XENIUM_LOG:-${SCRIPT_DIR}/xenium7_xenium8_cellpose.log}"
INTERVAL="${MEMORY_SAMPLE_INTERVAL:-1}"
ONLY_DATASET="${XENIUM_ONLY_DATASET:-}"
export PYTHON_BIN METHOD_SCRIPT LOG INTERVAL ONLY_DATASET
[[ -x "${PYTHON_BIN}" ]] || { echo "Python not found: ${PYTHON_BIN}" >&2; exit 1; }
[[ -f "${METHOD_SCRIPT}" && -f yourpath/run_xenium_method.py ]] || { echo "Method/runner script missing" >&2; exit 1; }
exec nohup bash -c '
set -u -o pipefail
run_one(){ local n="$1" d="$2" o="$3" c="$4" peak=0 rss pid rc; echo "[$(date)] START ${n} input=${d} output=${o}"; mkdir -p "${o}"; "${PYTHON_BIN}" yourpath/run_xenium_method.py "${METHOD_SCRIPT}" "${n}" "${d}" "${o}" --crop "${c}" >>"${LOG}" 2>&1 & pid=$!; while kill -0 "${pid}" 2>/dev/null; do rss=$(awk "/^VmRSS:/ {print \\$2}" "/proc/${pid}/status" 2>/dev/null || echo 0); [[ "${rss}" =~ ^[0-9]+$ ]] && (( rss > peak )) && peak=${rss}; sleep "${INTERVAL}"; done; wait "${pid}"; rc=$?; echo "[$(date)] END ${n} rc=${rc} peak_memory_mb=$((peak/1024))"; return ${rc}; }
[[ -z "${ONLY_DATASET}" || "${ONLY_DATASET}" == xenium7 ]] && run_one xenium7 yourpath/dataset/xenium7/outs yourpath/cellpose/xenium7_output 10210,20210,22080,32080
[[ -z "${ONLY_DATASET}" || "${ONLY_DATASET}" == xenium_8 ]] && run_one xenium_8 yourpath/dataset/xenium_8 yourpath/cellpose/xenium_8_output 8664,18664,22001,32001
' >>"${LOG}" 2>&1 &
echo "Started Cellpose in background (log: ${LOG}, pid=$!)"
