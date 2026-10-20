#!/usr/bin/env bash
# One-click, sequential Bering run for the same regions used by the Baysor job:
#   1. E14.5_E1S3 dorsal midbrain
#   2. E16.5_E1S3 dorsal midbrain
#   3. E16.5_E2S7 dorsal midbrain
#   4. D05103C4D4 registered-image center 3000x3000 native-pixel ROI
#
# Default: submit one detached worker with nohup.
#   yourpath/bering/run_bering_stereo_same_regions_nohup.sh
#
# Useful alternatives:
#   BERING_CUDA_VISIBLE_DEVICES=1 ./run_bering_stereo_same_regions_nohup.sh
#   ./run_bering_stereo_same_regions_nohup.sh --foreground
#   ./run_bering_stereo_same_regions_nohup.sh --prepare-only
#   ./run_bering_stereo_same_regions_nohup.sh --rerun-prediction
#   ./run_bering_stereo_same_regions_nohup.sh --resume-prediction
#   ./run_bering_stereo_same_regions_nohup.sh --check
#   ./run_bering_stereo_same_regions_nohup.sh --dry-run

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-yourpath/anaconda3/envs/bering/bin/python}"
PREPARER="${SCRIPT_DIR}/prepare_stereoseq_bering.py"
RUNNER="${SCRIPT_DIR}/bering_xenium_benchmark.py"
CONFIG_DIR="${SCRIPT_DIR}/configs"
PREPARED_DIR="${SCRIPT_DIR}/prepared_stereoseq"
OUTPUT_DIR="${SCRIPT_DIR}/stereo_outputs"
LOG_DIR="${SCRIPT_DIR}/logs"
STEREO_DIR="yourpath/dataset/stereoseq/stereo-seq_data_all"
MOUSE_DIR="yourpath/dataset/stereo_mouse_embryo"
CUDA_DEVICE="${BERING_CUDA_VISIBLE_DEVICES:-auto}"
LAUNCH_LOG="${LOG_DIR}/bering_stereo_same_regions_launcher.log"
PID_FILE="${LOG_DIR}/bering_stereo_same_regions.pid"
MODE="background"
ACTION="run"

DATASETS=(
  "E14.5_E1S3_Dorsal_Midbrain|midbrain|${STEREO_DIR}/E14.5_E1S3_Dorsal_Midbrain_GEM_CellBin.tsv.gz|${PREPARED_DIR}/E14.5_E1S3_Dorsal_Midbrain_GEM_CellBin.bering.parquet|${CONFIG_DIR}/stereoseq_E14.5_E1S3_dorsal_midbrain.json|${LOG_DIR}/bering_E14.5_E1S3_Dorsal_Midbrain.log|${OUTPUT_DIR}/E14.5_E1S3_Dorsal_Midbrain"
  "E16.5_E1S3_Dorsal_Midbrain|midbrain|${STEREO_DIR}/E16.5_E1S3_Dorsal_Midbrain_GEM_CellBin.tsv.gz|${PREPARED_DIR}/E16.5_E1S3_Dorsal_Midbrain_GEM_CellBin.bering.parquet|${CONFIG_DIR}/stereoseq_E16.5_E1S3_dorsal_midbrain.json|${LOG_DIR}/bering_E16.5_E1S3_Dorsal_Midbrain.log|${OUTPUT_DIR}/E16.5_E1S3_Dorsal_Midbrain"
  "E16.5_E2S7_Dorsal_Midbrain|midbrain|${STEREO_DIR}/E16.5_E2S7_Dorsal_Midbrain_GEM_CellBin.tsv.gz|${PREPARED_DIR}/E16.5_E2S7_Dorsal_Midbrain_GEM_CellBin.bering.parquet|${CONFIG_DIR}/stereoseq_E16.5_E2S7_dorsal_midbrain.json|${LOG_DIR}/bering_E16.5_E2S7_Dorsal_Midbrain.log|${OUTPUT_DIR}/E16.5_E2S7_Dorsal_Midbrain"
  "mouse_embryo_center3000|mouse|${MOUSE_DIR}/D05103C4D4.tissue.gef|${PREPARED_DIR}/D05103C4D4_center_3000.bering.parquet|${CONFIG_DIR}/stereo_mouse_embryo_center3000.json|${LOG_DIR}/bering_mouse_embryo_center3000.log|${OUTPUT_DIR}/mouse_embryo_center3000"
)

usage() {
  sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
  cat <<EOF
Options:
  --cuda-visible-devices N  Select physical GPU (default: ${CUDA_DEVICE})
  --foreground              Run sequentially in this terminal
  --prepare-only            Only prepare the four Bering Parquet inputs (nohup)
  --rerun-prediction        Reuse prepared inputs/training caches and replace predictions;
                            completed old predictions are moved to previous_results/
  --resume-prediction       Skip completed datasets and run only unfinished datasets;
                            require existing caches and never retrain
  --check                   Check paths, JSON, Python syntax and raw schemas
  --dry-run                 Print preparation and Bering commands
  -h, --help                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda-visible-devices)
      [[ $# -ge 2 ]] || { echo "[ERROR] --cuda-visible-devices requires a value" >&2; exit 2; }
      CUDA_DEVICE="$2"; shift 2 ;;
    --foreground) MODE="foreground"; shift ;;
    --prepare-only) ACTION="prepare-only"; shift ;;
    --rerun-prediction) ACTION="rerun-prediction"; shift ;;
    --resume-prediction) ACTION="resume-prediction"; shift ;;
    --check) MODE="check"; shift ;;
    --dry-run) MODE="dry-run"; shift ;;
    --worker) MODE="worker"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

select_cuda_device() {
  if [[ "${CUDA_DEVICE}" == "auto" ]]; then
    command -v nvidia-smi >/dev/null 2>&1 || {
      echo "[ERROR] nvidia-smi is unavailable; set BERING_CUDA_VISIBLE_DEVICES explicitly" >&2
      return 1
    }
    local row
    row="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | sort -t, -k2,2nr | head -n 1)"
    [[ -n "${row}" ]] || { echo "[ERROR] no CUDA GPU was detected" >&2; return 1; }
    CUDA_DEVICE="${row%%,*}"
    CUDA_DEVICE="${CUDA_DEVICE//[[:space:]]/}"
    local free_mib="${row#*,}"
    free_mib="${free_mib//[[:space:]]/}"
    echo "[GPU] auto-selected physical GPU ${CUDA_DEVICE} (${free_mib} MiB free)"
    if [[ "${free_mib}" =~ ^[0-9]+$ ]] && (( free_mib < 12288 )); then
      echo "[WARN] less than 12 GiB is free; training will use the corrected CPU fallback." >&2
    fi
  elif ! [[ "${CUDA_DEVICE}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] GPU must be a numeric physical index or 'auto': ${CUDA_DEVICE}" >&2
    return 1
  fi
}

require_file() {
  [[ -f "$1" ]] || { echo "[ERROR] required file does not exist: $1" >&2; return 1; }
}

check_bering_multiresolution() {
  "${PYTHON_BIN}" - <<'PY' >/dev/null
from types import SimpleNamespace
import importlib
import numpy as np
import pandas as pd

module = importlib.import_module("Bering.tools._cell_segmentation")
seen = {}
def fake_find_clusters(**kwargs):
    seen["resolutions"] = kwargs["resolutions"]
    return np.ones(len(kwargs["df_spots"]), dtype=int)
module.find_clusters_predictedLinks = fake_find_clusters
bg = SimpleNamespace(
    spots_all=pd.DataFrame({"predicted_labels": ["cell", "cell"]}),
    image_raw=None,
)
wanted = [0.001, 0.01]
module.cell_segmentation(bg, use_image=False, leiden_resolution=wanted)
assert seen.get("resolutions") == wanted, seen
PY
}

log_status() {
  # A background --worker already has stdout redirected to LAUNCH_LOG by
  # nohup.  Avoid teeing to the same file a second time.
  if [[ "${MODE}" == "worker" ]]; then
    echo "$*"
  else
    echo "$*" | tee -a "${LAUNCH_LOG}"
  fi
}

preflight() {
  require_file "${PYTHON_BIN}" || return 1
  require_file "${PREPARER}" || return 1
  require_file "${RUNNER}" || return 1
  check_bering_multiresolution || {
    echo "[ERROR] installed Bering cannot accept a Leiden resolution list" >&2
    return 1
  }
  require_file "${MOUSE_DIR}/D05103C4D4_HE_regist.tif" || return 1
  require_file "${MOUSE_DIR}/D05103C4D4.adjusted.cellbin.gef" || return 1
  local item name kind source prepared config log output
  for item in "${DATASETS[@]}"; do
    IFS='|' read -r name kind source prepared config log output <<< "${item}"
    require_file "${source}" || return 1
    require_file "${config}" || return 1
  done
  mkdir -p "${PREPARED_DIR}" "${OUTPUT_DIR}" "${LOG_DIR}"
}

prepare_command() {
  local kind="$1" source="$2" prepared="$3"
  if [[ "${kind}" == "midbrain" ]]; then
    printf '%q ' "${PYTHON_BIN}" "${PREPARER}" midbrain --input "${source}" --output "${prepared}"
  else
    printf '%q ' "${PYTHON_BIN}" "${PREPARER}" mouse \
      --gef "${source}" \
      --image "${MOUSE_DIR}/D05103C4D4_HE_regist.tif" \
      --cellbin "${MOUSE_DIR}/D05103C4D4.adjusted.cellbin.gef" \
      --roi-size 3000 --output "${prepared}"
  fi
}

prepare_one() {
  local kind="$1" source="$2" prepared="$3"
  if [[ "${kind}" == "midbrain" ]]; then
    "${PYTHON_BIN}" "${PREPARER}" midbrain --input "${source}" --output "${prepared}"
  else
    "${PYTHON_BIN}" "${PREPARER}" mouse \
      --gef "${source}" \
      --image "${MOUSE_DIR}/D05103C4D4_HE_regist.tif" \
      --cellbin "${MOUSE_DIR}/D05103C4D4.adjusted.cellbin.gef" \
      --roi-size 3000 --output "${prepared}"
  fi
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

backup_completed_outputs() {
  local output="$1" stamp backup file moved=0
  if [[ ! -s "${output}/transcript_assignments.parquet" && \
        ! -s "${output}/run_summary.json" ]]; then
    return 0
  fi
  stamp="$(date '+%Y%m%d_%H%M%S')"
  backup="${output}/previous_results/${stamp}"
  mkdir -p "${backup}"
  for file in transcript_assignments.parquet run_summary.json config_used.json; do
    if [[ -e "${output}/${file}" ]]; then
      mv -- "${output}/${file}" "${backup}/${file}"
      moved=1
    fi
  done
  if (( moved )); then
    log_status "[BACKUP] previous result moved to ${backup}"
  fi
}

run_checks() {
  "${PYTHON_BIN}" -m py_compile "${PREPARER}" "${RUNNER}" || return 1
  "${PYTHON_BIN}" - "${CONFIG_DIR}" "${PREPARED_DIR}" <<'PY'
import json, sys
from pathlib import Path
config_dir, prepared_dir = map(Path, sys.argv[1:])
names = [
    "stereoseq_E14.5_E1S3_dorsal_midbrain.json",
    "stereoseq_E16.5_E1S3_dorsal_midbrain.json",
    "stereoseq_E16.5_E2S7_dorsal_midbrain.json",
    "stereo_mouse_embryo_center3000.json",
]
for name in names:
    cfg = json.loads((config_dir / name).read_text())
    assert cfg["seed_source"] == "cellbin_transcript_cell_id"
    assert cfg["roi"]["mode"] == "full_dataset"
    assert cfg["training"]["feature_genes"] == 512
    assert cfg["bering"]["n_windows_per_cell"] == 3
    assert cfg["bering"]["batch_size"] == 2
    assert cfg["bering"]["tile_size_um"] == 300
    assert cfg["bering"]["seg_leiden_resolution"] == [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    assert cfg["bering"]["stitch_radius_um"] == 8
    assert cfg["bering"]["stitch_boundary_band_um"] == 12
    assert cfg["bering"]["stitch_min_seed_purity"] == 0.8
    print("[OK config]", name, "->", cfg["transcripts"])
PY
  "${PYTHON_BIN}" - <<'PY'
import gzip
from pathlib import Path
root = Path('yourpath/dataset/stereoseq/stereo-seq_data_all')
for name in [
    'E14.5_E1S3_Dorsal_Midbrain_GEM_CellBin.tsv.gz',
    'E16.5_E1S3_Dorsal_Midbrain_GEM_CellBin.tsv.gz',
    'E16.5_E2S7_Dorsal_Midbrain_GEM_CellBin.tsv.gz',
]:
    with gzip.open(str(root / name), 'rt') as handle:
        header = next(line for line in handle if not line.startswith('#')).rstrip().split('\t')
    required = {'geneID', 'x', 'y', 'cell'}
    assert required.issubset(header) and ({'MIDCount', 'MIDCounts'} & set(header)), (name, header)
    print('[OK schema]', name, header)
import h5py
mouse = Path('yourpath/dataset/stereo_mouse_embryo')
with h5py.File(str(mouse / 'D05103C4D4.tissue.gef'), 'r') as f:
    assert 'geneExp/bin1/expression' in f and 'geneExp/bin1/gene' in f
with h5py.File(str(mouse / 'D05103C4D4.adjusted.cellbin.gef'), 'r') as f:
    assert 'cellBin/cell' in f and 'cellBin/cellBorder' in f
print('[OK schema] mouse tissue GEF + adjusted CellBin GEF')
PY
  echo "[OK] preflight checks passed; prepared Parquet files are created only when the job runs."
}

run_worker() {
  local item name kind source prepared config log output rc msg require_cache=0 failures=0
  {
    echo "=== Bering Stereo-seq same-regions worker ==="
    echo "Started: $(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "Python: ${PYTHON_BIN}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_DEVICE}"
    echo "Action: ${ACTION}"
    echo "Jobs run sequentially."
  } >> "${LAUNCH_LOG}"

  for item in "${DATASETS[@]}"; do
    IFS='|' read -r name kind source prepared config log output <<< "${item}"
    if [[ ( "${ACTION}" == "run" || "${ACTION}" == "resume-prediction" ) ]] && \
       is_complete "${output}"; then
      log_status "[SKIP] ${name} already complete: ${output}"
      continue
    fi
    {
      echo
      echo "=== ${name} ==="
      echo "Started: $(date '+%Y-%m-%d %H:%M:%S %z')"
      echo "Raw source: ${source}"
      echo "Prepared input: ${prepared}"
      echo "Config: ${config}"
      echo "Output: ${output}"
    } >> "${log}"
    if [[ ( "${ACTION}" == "rerun-prediction" || "${ACTION}" == "resume-prediction" ) && \
          -s "${prepared}" ]]; then
      log_status "[Worker] reusing prepared input for ${name}: ${prepared}"
    else
      log_status "[Worker] preparing ${name}; log=${log}"
      prepare_one "${kind}" "${source}" "${prepared}" >> "${log}" 2>&1
      rc=$?
      if (( rc != 0 )); then
        failures=$((failures + 1))
        msg="[FAIL] ${name} preprocessing (exit=${rc})"
        echo "${msg}" >> "${log}"
        log_status "${msg}"
        continue
      fi
    fi
    if [[ "${ACTION}" == "prepare-only" ]]; then
      msg="[PREPARED] ${name}: ${prepared}"
      echo "${msg}" >> "${log}"
      log_status "${msg}"
      continue
    fi
    if [[ "${ACTION}" == "rerun-prediction" ]]; then
      backup_completed_outputs "${output}"
    fi
    require_cache=0
    if [[ "${ACTION}" == "resume-prediction" ]]; then
      require_cache=1
    fi
    log_status "[Worker] running Bering for ${name}"
    if CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
       BERING_CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
       BERING_REQUIRE_CACHED_TRAIN="${require_cache}" \
       PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128" \
       /usr/bin/time -v "${PYTHON_BIN}" "${RUNNER}" --config "${config}" \
       >> "${log}" 2>&1; then
      msg="[DONE] ${name}: $(date '+%Y-%m-%d %H:%M:%S %z')"
      echo "${msg}" >> "${log}"
      log_status "${msg}"
    else
      rc=$?
      failures=$((failures + 1))
      msg="[FAIL] ${name} Bering (exit=${rc}): $(date '+%Y-%m-%d %H:%M:%S %z')"
      echo "${msg}" >> "${log}"
      log_status "${msg}"
    fi
  done
  if (( failures != 0 )); then
    echo "[Summary] ${failures} dataset(s) failed; inspect ${LOG_DIR}." >> "${LAUNCH_LOG}"
    return 1
  fi
  echo "[Summary] all four datasets finished action=${ACTION}: $(date '+%Y-%m-%d %H:%M:%S %z')" \
    >> "${LAUNCH_LOG}"
}

select_cuda_device || exit 1
preflight || exit 1

if [[ "${MODE}" == "check" ]]; then
  run_checks
  exit $?
fi

if [[ "${MODE}" == "dry-run" ]]; then
  for item in "${DATASETS[@]}"; do
    IFS='|' read -r name kind source prepared config log output <<< "${item}"
    if [[ "${ACTION}" == "resume-prediction" ]] && is_complete "${output}"; then
      echo "[Skip complete] ${name}: ${output}"
      continue
    fi
    if [[ ( "${ACTION}" == "rerun-prediction" || "${ACTION}" == "resume-prediction" ) && \
          -s "${prepared}" ]]; then
      echo "[Reuse prepared] ${name}: ${prepared}"
    else
      echo "[Prepare] ${name}: $(prepare_command "${kind}" "${source}" "${prepared}")"
    fi
    if [[ "${ACTION}" != "prepare-only" ]]; then
      echo "[Bering]  ${name}: CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} ${PYTHON_BIN} ${RUNNER} --config ${config}"
    fi
    echo "          log=${log}"
  done
  exit 0
fi

if [[ "${MODE}" == "foreground" || "${MODE}" == "worker" ]]; then
  run_worker
  exit $?
fi

if [[ -s "${PID_FILE}" ]]; then
  old_pid="$(<"${PID_FILE}")"
  if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null && \
     ps -p "${old_pid}" -o args= | grep -Fq -- "run_bering_stereo_same_regions_nohup.sh --worker"; then
    echo "Already running: PID=${old_pid}"
    echo "Launcher log: ${LAUNCH_LOG}"
    exit 0
  fi
fi

echo "=== Bering Stereo-seq same-regions launcher ===" > "${LAUNCH_LOG}"
echo "Submitted: $(date '+%Y-%m-%d %H:%M:%S %z')" >> "${LAUNCH_LOG}"
worker_args=(--worker --cuda-visible-devices "${CUDA_DEVICE}")
if [[ "${ACTION}" == "prepare-only" ]]; then
  worker_args+=(--prepare-only)
elif [[ "${ACTION}" == "rerun-prediction" ]]; then
  worker_args+=(--rerun-prediction)
elif [[ "${ACTION}" == "resume-prediction" ]]; then
  worker_args+=(--resume-prediction)
fi
nohup env BERING_CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
  "${BASH:-bash}" "${SCRIPT_DIR}/run_bering_stereo_same_regions_nohup.sh" \
    "${worker_args[@]}" \
  >> "${LAUNCH_LOG}" 2>&1 < /dev/null &
pid=$!
echo "${pid}" > "${PID_FILE}"

if [[ "${ACTION}" == "resume-prediction" ]]; then
  echo "Submitted unfinished Bering jobs only: PID=${pid}"
else
  echo "Submitted four sequential Bering jobs: PID=${pid}"
fi
echo "Launcher log: ${LAUNCH_LOG}"
echo "Dataset logs: ${LOG_DIR}/bering_*Dorsal_Midbrain.log and ${LOG_DIR}/bering_mouse_embryo_center3000.log"
