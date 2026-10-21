#!/usr/bin/env bash
# Run Bering on all thirteen MERFISH center ROIs.
#
# The jobs are deliberately serial: each ROI can contain millions of
# transcripts and Bering's graph construction is memory intensive.  By
# default this script submits one detached worker (nohup); use --worker or
# --foreground when an attached run is preferred.
#
# Examples:
#   ./run_bering_merfish_center10000.sh
#   ./run_bering_merfish_center10000.sh --check
#   BERING_CUDA_VISIBLE_DEVICES=6 ./run_bering_merfish_center10000.sh
#   ./run_bering_merfish_center10000.sh --foreground --cuda-visible-devices 6
#   ./run_bering_merfish_center10000.sh --dry-run
#
# Per-dataset logs contain both Bering's log and /usr/bin/time -v's elapsed
# time/maximum RSS report.  The Python runner additionally writes
# bering_resource_summary.json in each output directory.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Bering writes its training diagnostic figures to a relative ``figures/``
# directory.  Anchor the worker here so those files do not end up in whatever
# directory happened to be current when the launcher was submitted.
cd "$SCRIPT_DIR"
PYTHON_BIN="${PYTHON_BIN:-yourpath/anaconda3/envs/bering/bin/python}"
RUNNER="${SCRIPT_DIR}/bering_merfish.py"
LOG_DIR="${BERING_LOG_DIR:-${SCRIPT_DIR}/merfish_new_logs}"
LAUNCH_LOG="${LOG_DIR}/bering_merfish_center10000_launcher.log"
SUMMARY_TSV="${LOG_DIR}/bering_merfish_center10000_summary.tsv"
PID_FILE="${LOG_DIR}/bering_merfish_center10000.pid"
TIME_BIN="${TIME_BIN:-/usr/bin/time}"

PRIOR="${BERING_PRIOR:-cellpose}"
# CUDA_VISIBLE_DEVICES_ARG is a physical GPU index.  Setting it to 3 makes
# physical GPU 3 appear as logical cuda:0 inside each Python worker.
# Use BERING_CUDA_VISIBLE_DEVICES or --cuda-visible-devices to override it.
CUDA_VISIBLE_DEVICES_ARG="${BERING_CUDA_VISIBLE_DEVICES-3}"
NODE_EPOCHS="${BERING_NODE_EPOCHS:-}"
EDGE_EPOCHS="${BERING_EDGE_EPOCHS:-}"
DRY_RUN="${BERING_DRY_RUN:-0}"

DATA_ROOT="${BERING_DATA_ROOT:-yourpath/dataset}"
OUTPUT_ROOT="${BERING_OUTPUT_ROOT:-${SCRIPT_DIR}/merfish_new_output}"

# name|input directory|output directory|log file|ROI pixels|transcript CSV|Cellpose mask
DATASETS=(
  "merfish_Melanoma|${DATA_ROOT}/merfish_Melanoma|${OUTPUT_ROOT}/merfish_Melanoma_center10000_output|${LOG_DIR}/bering_merfish_Melanoma_center10000.log|10000|${DATA_ROOT}/merfish_Melanoma/detected_transcripts.csv|yourpath/cellpose/merfish_output/merfish_Melanoma/pred_cellpose_roi.tif"
  "merfish_ovarian1|${DATA_ROOT}/merfish_ovarian1|${OUTPUT_ROOT}/merfish_ovarian1_center10000_output|${LOG_DIR}/bering_merfish_ovarian1_center10000.log|10000|${DATA_ROOT}/merfish_ovarian1/detected_transcripts.csv|yourpath/cellpose/merfish_output/merfish_ovarian1/pred_cellpose_roi.tif"
  "merfish_ovarian2|${DATA_ROOT}/merfish_ovarian2|${OUTPUT_ROOT}/merfish_ovarian2_center10000_output|${LOG_DIR}/bering_merfish_ovarian2_center10000.log|10000|${DATA_ROOT}/merfish_ovarian2/detected_transcripts.csv|yourpath/cellpose/merfish_output/merfish_ovarian2/pred_cellpose_roi.tif"
  "merfish_colon1|${DATA_ROOT}/merfish_colon1|${OUTPUT_ROOT}/merfish_colon1_center10000_output|${LOG_DIR}/bering_merfish_colon1_center10000.log|10000|${DATA_ROOT}/merfish_colon1/detected_transcripts.csv|yourpath/cellpose/merfish_output/merfish_colon1/pred_cellpose_roi.tif"
  "merfish_colon2|${DATA_ROOT}/merfish_colon2|${OUTPUT_ROOT}/merfish_colon2_center10000_output|${LOG_DIR}/bering_merfish_colon2_center10000.log|10000|${DATA_ROOT}/merfish_colon2/detected_transcripts.csv|yourpath/cellpose/merfish_output/merfish_colon2/pred_cellpose_roi.tif"
  "merfish_Uterian|${DATA_ROOT}/merfish_Uterian|${OUTPUT_ROOT}/merfish_Uterian_center10000_output|${LOG_DIR}/bering_merfish_Uterian_center10000.log|10000|${DATA_ROOT}/merfish_Uterian/detected_transcripts.csv|yourpath/cellpose/merfish_output/merfish_Uterian/pred_cellpose_roi.tif"
  "merfish_breast_cancer|${DATA_ROOT}/merfish_breast_cancer|${OUTPUT_ROOT}/merfish_breast_cancer_center10000_output|${LOG_DIR}/bering_merfish_breast_cancer_center10000.log|10000|${DATA_ROOT}/merfish_breast_cancer/detected_transcripts.csv|yourpath/cellpose/merfish_output/merfish_breast_cancer/pred_cellpose_roi.tif"
  "merfish_liver1|${DATA_ROOT}/merfish_liver1|${OUTPUT_ROOT}/merfish_liver1_center10000_output|${LOG_DIR}/bering_merfish_liver1_center10000.log|10000|${DATA_ROOT}/merfish_liver1/detected_transcripts.csv|yourpath/cellpose/merfish_liver1_output/pred_cellpose_roi.tif"
  "merfish_liver2|${DATA_ROOT}/merfish_liver2|${OUTPUT_ROOT}/merfish_liver2_center10000_output|${LOG_DIR}/bering_merfish_liver2_center10000.log|10000|${DATA_ROOT}/merfish_liver2/detected_transcripts.csv|yourpath/cellpose/merfish_liver2_output/pred_cellpose_roi.tif"
  "MERFISH_Lung_cancer|${DATA_ROOT}/MERFISH_Lung_cancer|${OUTPUT_ROOT}/MERFISH_Lung_cancer_center10000_output|${LOG_DIR}/bering_MERFISH_Lung_cancer_center10000.log|10000|${DATA_ROOT}/MERFISH_Lung_cancer/detected_transcripts.csv|yourpath/cellpose/merfish_lung_cancer/pred_cellpose_roi.tif"
  "merfish_mouse_brain|${DATA_ROOT}/merfish_mouse_brain|${OUTPUT_ROOT}/merfish_mouse_brain_center20000_output|${LOG_DIR}/bering_merfish_mouse_brain_center20000.log|20000|${DATA_ROOT}/merfish_mouse_brain/datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1_detected_transcripts_S1R1.csv|yourpath/cellpose/merfish_output/merfish_mouse_brain/pred_cellpose_roi.tif"
  "merfish_prostate1|${DATA_ROOT}/merfish_prostate1|${OUTPUT_ROOT}/merfish_prostate1_center10000_output|${LOG_DIR}/bering_merfish_prostate1_center10000.log|10000|${DATA_ROOT}/merfish_prostate1/detected_transcripts.csv|yourpath/cellpose/merfish_prostate1/pred_cellpose_roi.tif"
  "merfish_prostate2|${DATA_ROOT}/merfish_prostate2|${OUTPUT_ROOT}/merfish_prostate2_center10000_output|${LOG_DIR}/bering_merfish_prostate2_center10000.log|10000|${DATA_ROOT}/merfish_prostate2/detected_transcripts.csv|yourpath/cellpose/merfish_prostate2/pred_cellpose_roi.tif"
)

usage() {
  sed -n '2,22p' "$0"
  cat <<'EOF'

Options:
  --check                         Validate Python, runner, inputs and ROI caches.
  --dry-run                       Print commands without running them.
  --foreground                    Run the worker in the current shell.
  --worker                       Internal: run datasets instead of submitting nohup.
  --prior NAME                   cellpose (default), grid, metadata, or polygon.
  --cuda-visible-devices DEV     GPU list, e.g. 3; default is physical GPU 3.
  --node-epochs N                Override Bering node training epochs.
  --edge-epochs N                Override Bering edge training epochs.
  -h, --help                     Show this help.

Equivalent environment variables: BERING_PRIOR, BERING_CUDA_VISIBLE_DEVICES,
BERING_NODE_EPOCHS, BERING_EDGE_EPOCHS, BERING_DATA_ROOT,
BERING_OUTPUT_ROOT, BERING_LOG_DIR, PYTHON_BIN, BERING_DRY_RUN.
EOF
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] Required file does not exist: $path" >&2
    return 1
  fi
}

parse_args() {
  RUN_WORKER=0
  FOREGROUND=0
  CHECK_ONLY=0
  while (($#)); do
    case "$1" in
      --worker) RUN_WORKER=1 ;;
      --foreground) FOREGROUND=1 ;;
      --check) CHECK_ONLY=1 ;;
      --dry-run) DRY_RUN=1 ;;
      --prior)
        (($# >= 2)) || { echo "--prior needs a value" >&2; return 2; }
        PRIOR="$2"; shift ;;
      --cuda-visible-devices)
        (($# >= 2)) || { echo "--cuda-visible-devices needs a value" >&2; return 2; }
        CUDA_VISIBLE_DEVICES_ARG="$2"; shift ;;
      --node-epochs)
        (($# >= 2)) || { echo "--node-epochs needs a value" >&2; return 2; }
        NODE_EPOCHS="$2"; shift ;;
      --edge-epochs)
        (($# >= 2)) || { echo "--edge-epochs needs a value" >&2; return 2; }
        EDGE_EPOCHS="$2"; shift ;;
      -h|--help) usage; exit 0 ;;
      *) echo "Unknown option: $1" >&2; usage >&2; return 2 ;;
    esac
    shift
  done
  case "$PRIOR" in cellpose|grid|metadata|polygon) ;; *)
    echo "[ERROR] --prior must be cellpose, grid, metadata, or polygon (got $PRIOR)" >&2
    return 2 ;;
  esac
}

preflight() {
  if [[ "$PYTHON_BIN" == */* ]]; then
    require_file "$PYTHON_BIN" || return 1
  elif ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[ERROR] Python executable not found: $PYTHON_BIN" >&2
    return 1
  fi
  require_file "$RUNNER" || return 1
  require_file "$TIME_BIN" || return 1
  mkdir -p "$LOG_DIR"

  local item name data out log roi tx mask cache_count
  for item in "${DATASETS[@]}"; do
    IFS='|' read -r name data out log roi tx mask <<< "$item"
    [[ -d "$data" ]] || { echo "[ERROR] Missing data directory: $data" >&2; return 1; }
    require_file "$tx" || return 1
    if [[ "$PRIOR" == "cellpose" ]]; then
      require_file "$mask" || return 1
    fi
    # The prepared cache is what makes reruns practical for the 10--40 GB
    # detected_transcripts.csv files.  The Python runner can regenerate it,
    # so absence is a warning rather than a hard failure.
    cache_count=$(find "$data/_roi_cache" -maxdepth 1 -type f -name "tx_roi_*_${roi}.parquet" 2>/dev/null | wc -l)
    if [[ "$cache_count" -eq 0 ]]; then
      echo "[Preflight] WARNING: no ${roi}px ROI parquet cache in $data/_roi_cache; CSV scan will be used." >&2
    elif [[ "$cache_count" -gt 1 ]]; then
      echo "[ERROR] More than one ${roi}px ROI cache in $data/_roi_cache; pass --roi-cache to the Python runner." >&2
      return 1
    fi
  done
}

build_command() {
  # Bash arrays preserve paths and option values without eval/word splitting.
  local data="$1" out="$2" roi="$3" tx="$4" mask="$5"
  CMD=("$PYTHON_BIN" "$RUNNER" --data-dir "$data" --out-dir "$out" --prior "$PRIOR"
       --roi-pixels "$roi" --transcripts-csv "$tx")
  [[ "$PRIOR" == "cellpose" ]] && CMD+=(--cellpose-mask "$mask")
  [[ -n "$CUDA_VISIBLE_DEVICES_ARG" ]] && CMD+=(--cuda-visible-devices "$CUDA_VISIBLE_DEVICES_ARG")
  [[ -n "$NODE_EPOCHS" ]] && CMD+=(--node-epochs "$NODE_EPOCHS")
  [[ -n "$EDGE_EPOCHS" ]] && CMD+=(--edge-epochs "$EDGE_EPOCHS")
}

run_worker() {
  local item name data out log roi tx mask rc failures=0 started ended status
  printf 'dataset\tstatus\texit_code\tstarted\tended\toutput_dir\n' > "$SUMMARY_TSV"
  {
    echo "=== Bering MERFISH center-ROI worker (13 datasets) ==="
    echo "Started: $(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "Python: $PYTHON_BIN"
    echo "Runner: $RUNNER"
    echo "Prior: $PRIOR"
    echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES_ARG:-<unset>}"
  } >> "$LAUNCH_LOG"

  for item in "${DATASETS[@]}"; do
    IFS='|' read -r name data out log roi tx mask <<< "$item"
    started="$(date '+%Y-%m-%d %H:%M:%S %z')"
    build_command "$data" "$out" "$roi" "$tx" "$mask"
    {
      echo "=== Bering $name ==="
      echo "Started: $started"
      echo "Input: $data"
      echo "Output: $out"
      printf 'Command:'
      printf ' %q' "${CMD[@]}"
      echo
      echo
    } > "$log"
    echo "[Worker] Starting $name; log=$log" >> "$LAUNCH_LOG"

    if [[ "$DRY_RUN" == "1" ]]; then
      echo "[Dry-run] skipped" >> "$log"
      rc=0
    elif "$TIME_BIN" -v "${CMD[@]}" >> "$log" 2>&1; then
      rc=0
    else
      rc=$?
    fi

    ended="$(date '+%Y-%m-%d %H:%M:%S %z')"
    if (( rc == 0 )); then status=done; else status=failed; failures=$((failures + 1)); fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$status" "$rc" "$started" "$ended" "$out" >> "$SUMMARY_TSV"
    echo "[$status] $name exit=$rc ended=$ended" | tee -a "$log" >> "$LAUNCH_LOG"
  done

  if (( failures )); then
    echo "[Summary] $failures dataset(s) failed; see $SUMMARY_TSV" | tee -a "$LAUNCH_LOG"
    return 1
  fi
  echo "[Summary] all ${#DATASETS[@]} datasets completed: $(date '+%Y-%m-%d %H:%M:%S %z')" | tee -a "$LAUNCH_LOG"
  return 0
}

parse_args "$@" || exit $?
preflight || exit 1

if (( CHECK_ONLY )); then
  echo "Preflight OK for ${#DATASETS[@]} MERFISH datasets; prior=$PRIOR"
  exit 0
fi

if (( DRY_RUN )); then
  echo "[Dry-run] no Bering process will be started."
  for item in "${DATASETS[@]}"; do
    IFS='|' read -r name data out log roi tx mask <<< "$item"
    build_command "$data" "$out" "$roi" "$tx" "$mask"
    printf '%s:' "$name"; printf ' %q' "${CMD[@]}"; echo
  done
  # --dry-run is useful as a command preview and should not create a worker.
  (( RUN_WORKER )) && exit 0
  exit 0
fi

if (( RUN_WORKER || FOREGROUND )); then
  run_worker
  exit $?
fi

# Avoid accidentally submitting duplicate workers.
if [[ -s "$PID_FILE" ]]; then
  old_pid="$(<"$PID_FILE")"
  if kill -0 "$old_pid" 2>/dev/null && ps -p "$old_pid" -o args= | grep -Fq -- "$SCRIPT_DIR/run_bering_merfish_center10000.sh --worker"; then
    echo "Already running: PID=$old_pid"
    echo "Launcher log: $LAUNCH_LOG"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

echo "=== Bering MERFISH center-ROI launcher (13 datasets) ===" > "$LAUNCH_LOG"
echo "Submitted: $(date '+%Y-%m-%d %H:%M:%S %z')" >> "$LAUNCH_LOG"
# Propagate command-line overrides to the detached worker.  In particular,
# without this, ``--prior polygon`` or ``--cuda-visible-devices 6`` would only
# affect the submitting shell and the worker would silently use defaults.
nohup env \
  "PYTHON_BIN=$PYTHON_BIN" \
  "TIME_BIN=$TIME_BIN" \
  "BERING_PRIOR=$PRIOR" \
  "BERING_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_ARG" \
  "BERING_NODE_EPOCHS=$NODE_EPOCHS" \
  "BERING_EDGE_EPOCHS=$EDGE_EPOCHS" \
  "BERING_DATA_ROOT=$DATA_ROOT" \
  "BERING_OUTPUT_ROOT=$OUTPUT_ROOT" \
  "BERING_LOG_DIR=$LOG_DIR" \
  "${BASH:-bash}" "$SCRIPT_DIR/run_bering_merfish_center10000.sh" --worker \
  >> "$LAUNCH_LOG" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$PID_FILE"
echo "Submitted ${#DATASETS[@]} MERFISH datasets sequentially: PID=$pid"
echo "Launcher log: $LAUNCH_LOG"
echo "Per-dataset logs: $LOG_DIR/bering_*_center*.log"
