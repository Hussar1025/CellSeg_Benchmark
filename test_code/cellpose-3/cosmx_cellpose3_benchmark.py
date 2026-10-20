#!/usr/bin/env python3
"""Cellpose 3 cyto3 on three CosMx datasets, evaluated against CellLabels.

The dataset result uses the micro aggregation defined in
yourpath/cellpose/cosmx_lymph.py. One FOV is one independent image;
Pancreas "full slide" means all 18 exported FOVs (51..68), not a crop.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import resource
import sys
import time
import traceback
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
from PIL import Image


ROOT = Path("yourpath")
OUT_ROOT = ROOT / "cellpose3" / "cosmx_outputs"
CORE_PATH = ROOT / "cellpose" / "cosmx_lymph.py"
EVAL_DIR = ROOT / "eval"
sys.path.insert(0, str(EVAL_DIR))
import cosmx_lymph_common as common  # noqa: E402


BRAIN_FOVS = (4, 18, 51, 52, 56, 65, 74, 87, 90, 101, 104, 110, 124,
              131, 147, 173, 187, 220, 224, 234, 242, 250, 255, 256,
              264, 296, 298, 307, 339)
DATASETS = {
    "cosmx_human_brain": {
        "root": ROOT / "dataset" / "cosmx_human_brain",
        "fovs": BRAIN_FOVS,
        "channel": "rRNA + Histone",
        "diameter": 45.0,
    },
    "cosmx_lymph_node": {
        "root": ROOT / "dataset" / "cosmx_lymph_node",
        "fovs": (48, 261),
        "channel": "Membrane + DNA",
        "diameter": 45.0,
    },
    "Pancreas-CosMx-WTx-FlatFiles": {
        "root": ROOT / "dataset" / "Pancreas-CosMx-WTx-FlatFiles",
        "fovs": tuple(range(51, 69)),
        "channel": "max(R,G) + B(DAPI) from CellComposite",
        "diameter": 45.0,
    },
}
ORDER = tuple(DATASETS)


def load_core():
    spec = importlib.util.spec_from_file_location("cosmx_lymph_metric_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load metric core: {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cache_path(dataset: str, fov: int) -> Path:
    root = DATASETS[dataset]["root"] / "_bench_cache"
    if dataset == "cosmx_human_brain":
        return root / f"L0_F{fov:04d}_Histone-rRNA.npz"
    return root / f"L0_F{fov:04d}_CD45-CD68-DNA-Membrane-PanCK.npz"


def pancreas_paths(fov: int) -> tuple[Path, Path]:
    root = DATASETS["Pancreas-CosMx-WTx-FlatFiles"]["root"]
    return (root / "CellComposite" / f"CellComposite_F{fov:03d}.jpg",
            root / "CellLabels" / f"CellLabels_F{fov:03d}.tif")


def preflight(names: tuple[str, ...]) -> None:
    if not version("cellpose").startswith("3."):
        raise RuntimeError(f"Cellpose 3.x required, found {version('cellpose')}")
    if "site-packages/torch/__init__.py" not in str(torch.__file__):
        raise RuntimeError(f"Unexpected PyTorch module: {torch.__file__}")
    if not CORE_PATH.is_file() or not (EVAL_DIR / "test.py").is_file():
        raise FileNotFoundError("Missing cosmx_lymph.py or eval/test.py")
    for name in names:
        cfg = DATASETS[name]
        for fov in cfg["fovs"]:
            paths = pancreas_paths(fov) if name == "Pancreas-CosMx-WTx-FlatFiles" else (cache_path(name, fov),)
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"{name} FOV {fov}: missing {missing}")
        print(f"[OK] {name}: {len(cfg['fovs'])} FOVs; input={cfg['channel']}")
    print(f"[OK] Cellpose {version('cellpose')}; PyTorch {torch.__version__}; CUDA={torch.cuda.is_available()}")


def read_fov(dataset: str, fov: int) -> tuple[np.ndarray, np.ndarray]:
    if dataset == "Pancreas-CosMx-WTx-FlatFiles":
        image_path, label_path = pancreas_paths(fov)
        with Image.open(image_path) as handle:
            rgb = np.asarray(handle.convert("RGB"))
        gt = tifffile.imread(label_path)
        image = np.empty((*rgb.shape[:2], 2), dtype=np.float32)
        image[..., 0] = np.maximum(rgb[..., 0], rgb[..., 1])
        image[..., 1] = rgb[..., 2]
        del rgb
    else:
        with np.load(cache_path(dataset, fov), allow_pickle=False) as data:
            if dataset == "cosmx_human_brain":
                cytoplasm, nucleus = data["rRNA"], data["Histone"]
            else:
                cytoplasm, nucleus = data["Membrane"], data["DNA"]
            gt = np.asarray(data["labels"])
            image = np.empty((*cytoplasm.shape, 2), dtype=np.float32)
            image[..., 0] = cytoplasm
            image[..., 1] = nucleus
    gt = np.squeeze(np.asarray(gt))
    if image.ndim != 3 or image.shape[-1] != 2 or image.shape[:2] != gt.shape:
        raise ValueError(f"FOV {fov}: input {image.shape} and CellLabels {gt.shape} differ")
    return image, gt


def save_json(path: Path, record: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def save_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, path)


def peak_rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value) / (1024**2 if sys.platform == "darwin" else 1024)


def config_hash(args: argparse.Namespace, dataset: str) -> str:
    cfg = DATASETS[dataset]
    parameters = {
        "dataset": dataset,
        "fovs": cfg["fovs"],
        "channel": cfg["channel"],
        "model": "cyto3",
        "diameter": args.diameter if args.diameter is not None else cfg["diameter"],
        "batch_size": args.batch_size,
        "flow_threshold": args.flow_threshold,
        "cellprob_threshold": args.cellprob_threshold,
        "iou_threshold": 0.5,
        "boundary_mode": "thick",
    }
    return hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()


def restore_raw(record: dict) -> dict:
    raw = record["raw"]
    return {
        "comp": raw["comp"],
        "inst": raw["inst"],
        "aji_num": raw["aji_num"],
        "aji_den": raw["aji_den"],
        "mda_map": {float(key): tuple(value) for key, value in raw["mda_map"].items()},
    }


def score_fov(core, bench, gt: np.ndarray, pred: np.ndarray) -> tuple[dict, dict]:
    if gt.shape != pred.shape:
        raise ValueError(f"Prediction {pred.shape} and CellLabels {gt.shape} differ")
    gt = core.compact_labels(gt)
    pred = core.compact_labels(pred)
    overlaps = bench.compute_sparse_overlaps(gt, pred)
    comp = core.pixel_components(gt, pred, boundary_mode="thick")
    inst = core.instance_components(bench, overlaps, iou_threshold=0.5)
    aji_num, aji_den = core.aji_components(bench, overlaps, gt, pred)
    mda_map = core.mda_components(bench, overlaps)
    raw = dict(comp=comp, inst=inst, aji_num=aji_num, aji_den=aji_den,
               mda_map=mda_map)
    counts = dict(gt_cell_count=int(len(overlaps.gt_ids)),
                  pred_cell_count=int(len(overlaps.pred_ids)))
    return counts, raw


def run_dataset(args: argparse.Namespace) -> dict:
    from cellpose import models

    dataset = args.dataset
    cfg = DATASETS[dataset]
    out_dir = args.output_root / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    core = load_core()
    bench = core.load_metric_module(str(EVAL_DIR / "test.py"))
    digest = config_hash(args, dataset)
    diameter = args.diameter if args.diameter is not None else cfg["diameter"]
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("GPU requested but torch.cuda.is_available() is False")
    monitor = common.ResourceMonitor(args.resource_poll_seconds).start()
    started = time.perf_counter()
    segmentation_seconds = 0.0
    model_load_seconds = 0.0
    model = None
    completed: list[dict] = []
    raws: list[dict] = []
    status = "failed"
    error = None
    peak_allocated = peak_reserved = 0.0
    gpu_device = "cpu"
    if not args.cpu:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "unrestricted")
        gpu_device = (
            f"physical GPU {visible} / cuda:{torch.cuda.current_device()} "
            f"({torch.cuda.get_device_name()})"
        )
    print(f"=== {dataset}: {len(cfg['fovs'])} FOVs; GPU={gpu_device}; diameter={diameter} ===", flush=True)
    print(f"Input={cfg['channel']}; output={out_dir}", flush=True)
    try:
        for index, fov in enumerate(cfg["fovs"], start=1):
            checkpoint = out_dir / f"F{fov:04d}_checkpoint.json"
            prediction = out_dir / f"F{fov:04d}_cellpose3_pred.tif"
            if not args.force and checkpoint.is_file() and prediction.is_file():
                cached = json.loads(checkpoint.read_text(encoding="utf-8"))
                if cached.get("config_hash") == digest:
                    completed.append(cached)
                    raws.append(restore_raw(cached))
                    print(f"[{index}/{len(cfg['fovs'])}] FOV {fov}: reused checkpoint", flush=True)
                    continue

            print(f"[{index}/{len(cfg['fovs'])}] FOV {fov}: loading", flush=True)
            image, gt = read_fov(dataset, fov)
            if model is None:
                model_started = time.perf_counter()
                model = models.Cellpose(gpu=not args.cpu, model_type="cyto3")
                model_load_seconds += time.perf_counter() - model_started
            if not args.cpu:
                torch.cuda.reset_peak_memory_stats()
            segment_started = time.perf_counter()
            result = model.eval(
                image, channels=[1, 2], diameter=diameter,
                batch_size=args.batch_size, flow_threshold=args.flow_threshold,
                cellprob_threshold=args.cellprob_threshold, normalize=True,
                resample=True,
            )
            if not args.cpu:
                torch.cuda.synchronize()
                fov_peak_allocated = torch.cuda.max_memory_allocated() / 2**20
                fov_peak_reserved = torch.cuda.max_memory_reserved() / 2**20
                peak_allocated = max(peak_allocated, fov_peak_allocated)
                peak_reserved = max(peak_reserved, fov_peak_reserved)
            else:
                fov_peak_allocated = fov_peak_reserved = 0.0
            elapsed = time.perf_counter() - segment_started
            segmentation_seconds += elapsed
            pred = np.squeeze(np.asarray(result[0])).astype(np.int32, copy=False)
            del result, image
            if pred.ndim != 2 or pred.shape != gt.shape:
                raise ValueError(f"FOV {fov}: Cellpose returned {pred.shape}, GT {gt.shape}")

            counts, raw = score_fov(core, bench, gt, pred)
            tifffile.imwrite(prediction, pred, compression="zlib", tile=(512, 512))
            record = {
                "dataset": dataset,
                "fov": fov,
                "config_hash": digest,
                "segmentation_seconds": elapsed,
                "peak_cpu_rss_mib": peak_rss_mib(),
                "peak_gpu_process_mib": monitor.gpu_peak_process_mib,
                "peak_gpu_torch_allocated_mib": fov_peak_allocated,
                "peak_gpu_torch_reserved_mib": fov_peak_reserved,
                **counts,
                "raw": raw,
            }
            save_json(checkpoint, record)
            completed.append(record)
            raws.append(raw)
            metrics = core.finalize(**{
                "comp": raw["comp"], "inst": raw["inst"],
                "aji_num": raw["aji_num"], "aji_den": raw["aji_den"],
                "mda_map": raw["mda_map"],
            })
            print(f"  gt={counts['gt_cell_count']} pred={counts['pred_cell_count']} "
                  f"PQ={metrics['PQ']:.4f} AJI={metrics['AJI']:.4f} "
                  f"segmentation={elapsed:.1f}s", flush=True)
            del gt, pred
            gc.collect()

        usage = monitor.stop()
        peak_allocated = max(
            [peak_allocated, *(item.get("peak_gpu_torch_allocated_mib", 0.0) for item in completed)]
        )
        peak_reserved = max(
            [peak_reserved, *(item.get("peak_gpu_torch_reserved_mib", 0.0) for item in completed)]
        )
        micro = core.aggregate_micro(raws)
        row = {
            "dataset": dataset,
            "method": "Cellpose3-cyto3",
            "reference": "CosMx CellLabels",
            "aggregation": "micro",
            "fovs_requested": len(cfg["fovs"]),
            "fovs_completed": len(completed),
            "fov_ids": ",".join(map(str, cfg["fovs"])),
            "input_channels": cfg["channel"],
            "diameter": diameter,
            "batch_size": args.batch_size,
            "gt_cell_count": sum(item["gt_cell_count"] for item in completed),
            "pred_cell_count": sum(item["pred_cell_count"] for item in completed),
            **micro,
            "cellpose_version": version("cellpose"),
            "gpu_device": gpu_device,
            "segmentation_seconds": sum(item["segmentation_seconds"] for item in completed),
            "model_load_seconds": model_load_seconds,
            "runtime_seconds": time.perf_counter() - started,
            "peak_cpu_rss_mib": max(
                [peak_rss_mib(), *(item.get("peak_cpu_rss_mib", 0.0) for item in completed)]
            ),
            "peak_gpu_process_mib": max(
                [usage["gpu_peak_process_mib"],
                 *(item.get("peak_gpu_process_mib", 0.0) for item in completed)]
            ),
            "peak_gpu_torch_allocated_mib": peak_allocated,
            "peak_gpu_torch_reserved_mib": peak_reserved,
        }
        save_csv(out_dir / "micro_metrics.csv", [row])
        status = "success"
        print(pd.DataFrame([row]).to_string(index=False), flush=True)
        return row
    except BaseException as exc:
        error = repr(exc)
        traceback.print_exc()
        raise
    finally:
        usage = monitor.stop()
        performance = {
            "dataset": dataset,
            "status": status,
            "completed_fovs": [item["fov"] for item in completed],
            "requested_fovs": list(cfg["fovs"]),
            "gpu_device": gpu_device,
            "runtime_seconds": time.perf_counter() - started,
            "segmentation_seconds_this_run": segmentation_seconds,
            "peak_cpu_rss_mib": peak_rss_mib(),
            "peak_gpu_torch_allocated_mib": peak_allocated,
            "peak_gpu_torch_reserved_mib": peak_reserved,
            **usage,
            "error": error,
        }
        save_json(out_dir / "performance.json", performance)
        print(f"[performance] status={status}; GPU={gpu_device}; "
              f"segmentation={segmentation_seconds:.1f}s; "
              f"CPU peak={performance['peak_cpu_rss_mib']:.1f} MiB; "
              f"GPU process peak={usage['gpu_peak_process_mib']:.1f} MiB", flush=True)


def aggregate(output_root: Path) -> Path:
    rows = []
    for name in ORDER:
        metrics = output_root / name / "micro_metrics.csv"
        performance = output_root / name / "performance.json"
        if metrics.is_file() and performance.is_file():
            status = json.loads(performance.read_text(encoding="utf-8")).get("status")
            if status == "success":
                rows.append(pd.read_csv(metrics).iloc[0].to_dict())
    if not rows:
        raise FileNotFoundError(f"No successful datasets under {output_root}")
    output = output_root / "cellpose3_cosmx_micro_summary.csv"
    save_csv(output, rows)
    print(f"[aggregate] {len(rows)}/{len(ORDER)} datasets -> {output}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=ORDER)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--diameter", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--flow-threshold", type=float, default=0.4)
    parser.add_argument("--cellprob-threshold", type=float, default=0.0)
    parser.add_argument("--resource-poll-seconds", type=float, default=0.5)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="recompute completed FOVs")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.diameter is not None and args.diameter <= 0):
        parser.error("batch-size and diameter must be positive")
    if not args.check and not args.aggregate_only and args.dataset is None:
        parser.error("--dataset is required for segmentation")
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        aggregate(args.output_root)
        return 0
    preflight((args.dataset,) if args.dataset else ORDER)
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("GPU requested but PyTorch CUDA is unavailable")
    if args.check:
        return 0
    run_dataset(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
