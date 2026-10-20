#!/usr/bin/env python3
"""Cellpose 3 (cyto3) segmentation and evaluation for 13 MERFISH datasets.

Each dataset is evaluated on a reproducible centre ROI in native mosaic pixel
coordinates.  The prediction uses Cellbound1 (or polyT) as the cytoplasm /
membrane channel and DAPI as the nucleus channel.  Vizgen cell-boundary HDF5
polygons are rasterized into the same ROI and evaluated with the metric
implementation in yourpath/eval/test.py.
"""

from __future__ import annotations

import argparse
import gc
import glob
import importlib.util
import json
import os
import platform
import re
import resource
import sys
import time
import traceback
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import pandas as pd
import tifffile
import torch
from skimage.draw import polygon as draw_polygon


ROOT = Path("yourpath")
DATA_ROOT = ROOT / "dataset"
DEFAULT_OUTPUT_ROOT = ROOT / "cellpose3" / "merfish_outputs"
METRICS_PATH = ROOT / "eval" / "test.py"

# This explicit list is intentionally case-sensitive.  MERFISH_Lung_cancer is
# the thirteenth dataset that a lowercase-only directory search would miss.
DATASET_ORDER = (
    "merfish_Melanoma",
    "merfish_ovarian1",
    "merfish_ovarian2",
    "merfish_colon1",
    "merfish_colon2",
    "merfish_Uterian",
    "merfish_breast_cancer",
    "merfish_liver1",
    "merfish_liver2",
    "MERFISH_Lung_cancer",
    "merfish_mouse_brain",
    "merfish_prostate1",
    "merfish_prostate2",
)
EXCLUDED_DATASETS = {"merfish_mouse_hypothalamus", "merfish_liver2_roi10000"}
DEFAULT_ROI_SIZE = 10_000
DATASET_ROI_SIZES = {"merfish_mouse_brain": 20_000}


@dataclass(frozen=True)
class DatasetFiles:
    name: str
    root: Path
    dapi: Path
    cellbound: Path | None
    polyt: Path
    transform: Path
    metadata: Path
    boundaries: Path


def unique_match(root: Path, predicate, description: str) -> Path:
    matches = sorted(
        path for path in root.rglob("*")
        if path.is_file() and predicate(path.name.lower())
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{root}: expected exactly one {description}; found "
            f"{len(matches)}: {[str(path) for path in matches]}"
        )
    return matches[0]


def optional_unique_match(root: Path, predicate, description: str) -> Path | None:
    matches = sorted(
        path for path in root.rglob("*")
        if path.is_file() and predicate(path.name.lower())
    )
    if len(matches) > 1:
        raise FileNotFoundError(
            f"{root}: expected at most one {description}; found "
            f"{len(matches)}: {[str(path) for path in matches]}"
        )
    return matches[0] if matches else None


def discover_dataset(name: str) -> DatasetFiles:
    root = DATA_ROOT / name
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    def mosaic_tif(token: str):
        return lambda filename: (
            token in filename
            and "mosaic" in filename
            and filename.endswith((".tif", ".tiff"))
        )

    return DatasetFiles(
        name=name,
        root=root,
        dapi=unique_match(root, mosaic_tif("dapi_z3"), "DAPI z3 mosaic TIFF"),
        cellbound=optional_unique_match(
            root, mosaic_tif("cellbound1_z3"), "Cellbound1 z3 mosaic TIFF"
        ),
        polyt=unique_match(root, mosaic_tif("polyt_z3"), "polyT z3 mosaic TIFF"),
        transform=unique_match(
            root,
            lambda filename: "micron_to_mosaic_pixel_transform" in filename
            and filename.endswith(".csv"),
            "micron-to-mosaic transform CSV",
        ),
        metadata=unique_match(
            root,
            lambda filename: "cell_metadata" in filename and filename.endswith(".csv"),
            "cell metadata CSV",
        ),
        boundaries=root / "cell_boundaries",
    )


def cellpose_version() -> str:
    try:
        return package_version("cellpose")
    except PackageNotFoundError:
        return "not-installed"


def require_cellpose3() -> str:
    version = cellpose_version()
    if not version.startswith("3."):
        raise RuntimeError(
            f"Cellpose 3.x is required; Python {sys.executable} sees {version}. "
            "Run this script with the conda environment named cellpose3."
        )
    if "site-packages/torch/__init__.py" not in str(torch.__file__):
        raise RuntimeError(f"Refusing unexpected torch module: {torch.__file__}")
    return version


def load_metrics_module():
    if not METRICS_PATH.is_file():
        raise FileNotFoundError(f"Metrics implementation not found: {METRICS_PATH}")
    spec = importlib.util.spec_from_file_location("cellpose3_merfish_metrics", METRICS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load metrics module: {METRICS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiff_shape(path: Path) -> tuple[int, int]:
    with tifffile.TiffFile(path) as tif:
        shape = tuple(int(value) for value in tif.series[0].shape)
    if len(shape) < 2:
        raise ValueError(f"Cannot determine spatial shape of {path}: {shape}")
    return shape[-2], shape[-1]


def center_roi(shape: tuple[int, int], size: int) -> tuple[int, int, int, int]:
    height, width = shape
    size = min(int(size), height, width)
    y0 = (height - size) // 2
    x0 = (width - size) // 2
    return x0, y0, x0 + size, y0 + size


def roi_size_for_dataset(name: str, override: int | None) -> int:
    if override is not None:
        return override
    return DATASET_ROI_SIZES.get(name, DEFAULT_ROI_SIZE)


def read_tiff_crop(path: Path, roi_xy: tuple[int, int, int, int]) -> np.ndarray:
    """Read a TIFF ROI without loading the complete mosaic when possible."""
    x0, y0, x1, y1 = roi_xy

    # Vizgen mosaics in this collection are often uncompressed, single-strip
    # TIFFs.  They can be sliced efficiently with a memory map, whereas their
    # Zarr view has one enormous chunk and would decode/read the complete image.
    try:
        mapped = tifffile.memmap(path, series=0)
        result = np.array(mapped[y0:y1, x0:x1], copy=True)
        del mapped
        result = np.squeeze(result)
        if result.ndim != 2:
            raise ValueError(f"Expected a 2-D channel in {path}; got {result.shape}")
        return result
    except (ValueError, TypeError):
        pass

    import zarr

    with tifffile.TiffFile(path) as tif:
        store = tif.series[0].aszarr()
        try:
            array = zarr.open(store, mode="r")
            if not hasattr(array, "shape"):
                keys = sorted(array.array_keys())
                if not keys:
                    raise ValueError(f"TIFF Zarr group contains no arrays: {path}")
                array = array["0"] if "0" in keys else array[keys[0]]
            result = np.asarray(array[y0:y1, x0:x1])
        finally:
            close = getattr(store, "close", None)
            if close is not None:
                close()
    result = np.squeeze(result)
    if result.ndim != 2:
        raise ValueError(f"Expected a 2-D channel in {path}; got {result.shape}")
    return result


def robust_limits(array: np.ndarray) -> tuple[float, float]:
    stride = max(1, int(np.sqrt(array.size / 250_000)))
    sample = np.asarray(array[::stride, ::stride]).reshape(-1)
    if np.issubdtype(sample.dtype, np.floating):
        sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(sample, (1.0, 99.8))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(sample.min()), float(sample.max())
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def scale_to_uint16(array: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    low, high = robust_limits(array)
    scale = np.float32(65535.0 / (high - low))
    output = np.empty(array.shape, dtype=np.uint16)
    for y0 in range(0, array.shape[0], 512):
        y1 = min(array.shape[0], y0 + 512)
        block = array[y0:y1].astype(np.float32, copy=False)
        block = np.clip((block - np.float32(low)) * scale, 0.0, 65535.0)
        output[y0:y1] = block.astype(np.uint16)
    return output, (low, high)


def build_input(
    files: DatasetFiles,
    roi_xy: tuple[int, int, int, int],
    cyto_channel: str,
) -> tuple[np.ndarray, dict]:
    requested_channel = cyto_channel
    if cyto_channel == "cellbound" and files.cellbound is None:
        print("  Cellbound1 is unavailable; falling back to polyT")
        cyto_channel = "polyt"
    cyto_path = files.cellbound if cyto_channel == "cellbound" else files.polyt
    assert cyto_path is not None
    raw_dapi = read_tiff_crop(files.dapi, roi_xy)
    raw_cyto = read_tiff_crop(cyto_path, roi_xy)
    if raw_dapi.shape != raw_cyto.shape:
        raise ValueError(f"DAPI {raw_dapi.shape} and cyto {raw_cyto.shape} differ")
    dapi, dapi_limits = scale_to_uint16(raw_dapi)
    cyto, cyto_limits = scale_to_uint16(raw_cyto)
    del raw_dapi, raw_cyto

    image = np.empty((*dapi.shape, 2), dtype=np.uint16)
    image[..., 0] = cyto
    image[..., 1] = dapi
    del cyto, dapi
    gc.collect()
    return image, {
        "input_description": f"[{cyto_channel}, DAPI]",
        "requested_cyto_channel": requested_channel,
        "actual_cyto_channel": cyto_channel,
        "cytoplasm_or_membrane_channel": str(cyto_path),
        "nucleus_channel": str(files.dapi),
        "cyto_limits": cyto_limits,
        "dapi_limits": dapi_limits,
    }


def load_transform(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    values = np.asarray([float(value) for value in numbers], dtype=np.float64)
    if values.size < 9:
        raise ValueError(f"Could not parse a 3x3 transform from {path}")
    return values[:9].reshape(3, 3)


def micron_to_pixel(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    return (matrix @ homogeneous.T).T[:, :2]


def fovs_intersecting_roi(
    metadata_path: Path,
    matrix: np.ndarray,
    roi_xy: tuple[int, int, int, int],
) -> set[int] | None:
    try:
        metadata = pd.read_csv(metadata_path, index_col=0)
        required = {"fov", "min_x", "max_x", "min_y", "max_y"}
        if not required.issubset(metadata.columns):
            return None
        x0, y0, x1, y1 = roi_xy
        corners = np.stack(
            [
                metadata[["min_x", "min_y"]].to_numpy(),
                metadata[["min_x", "max_y"]].to_numpy(),
                metadata[["max_x", "min_y"]].to_numpy(),
                metadata[["max_x", "max_y"]].to_numpy(),
            ],
            axis=1,
        ).reshape(-1, 2)
        pixels = micron_to_pixel(corners, matrix).reshape(-1, 4, 2)
        lower = pixels.min(axis=1)
        upper = pixels.max(axis=1)
        intersects = ~(
            (upper[:, 0] < x0)
            | (lower[:, 0] >= x1)
            | (upper[:, 1] < y0)
            | (lower[:, 1] >= y1)
        )
        return set(metadata.loc[intersects, "fov"].astype(int).tolist())
    except Exception as exc:
        print(f"  warning: metadata FOV prefilter failed: {exc}")
        return None


def read_fov_boundaries(path: Path, z_index: int) -> dict[str, np.ndarray]:
    polygons: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        if "featuredata" not in handle:
            return polygons
        z_key = f"zIndex_{z_index}"
        for cell_id in handle["featuredata"].keys():
            cell = handle["featuredata"][cell_id]
            if z_key not in cell:
                continue
            polygon_keys = sorted(key for key in cell[z_key].keys() if key.startswith("p_"))
            if not polygon_keys:
                continue
            polygon = cell[z_key][polygon_keys[0]]
            if "coordinates" not in polygon:
                continue
            coordinates = np.squeeze(np.asarray(polygon["coordinates"][:]))
            if coordinates.ndim == 2 and coordinates.shape[1] == 2 and len(coordinates) >= 3:
                polygons[str(cell_id)] = coordinates.astype(np.float64)
    return polygons


def build_ground_truth(
    files: DatasetFiles,
    roi_xy: tuple[int, int, int, int],
    z_index: int,
) -> tuple[np.ndarray, dict[int, str], int, int]:
    x0, y0, x1, y1 = roi_xy
    shape = (y1 - y0, x1 - x0)
    matrix = load_transform(files.transform)
    selected_fovs = fovs_intersecting_roi(files.metadata, matrix, roi_xy)
    all_files = sorted(
        (Path(path) for path in glob.glob(str(files.boundaries / "feature_data_*.hdf5"))),
        key=lambda path: int(re.search(r"(\d+)$", path.stem).group(1)),
    )
    if not all_files:
        raise FileNotFoundError(f"No feature_data_*.hdf5 files in {files.boundaries}")
    if selected_fovs is None:
        selected = all_files
    else:
        selected = [
            path for path in all_files
            if int(re.search(r"(\d+)$", path.stem).group(1)) in selected_fovs
        ]
    print(f"  GT FOV prefilter: {len(selected)}/{len(all_files)} HDF5 files")

    labels = np.zeros(shape, dtype=np.int32)
    label_to_cell_id: dict[int, str] = {}
    next_label = 1
    for index, path in enumerate(selected, start=1):
        for cell_id, vertices_micron in read_fov_boundaries(path, z_index).items():
            pixels = micron_to_pixel(vertices_micron, matrix)
            if (
                pixels[:, 0].max() < x0
                or pixels[:, 0].min() >= x1
                or pixels[:, 1].max() < y0
                or pixels[:, 1].min() >= y1
            ):
                continue
            columns = pixels[:, 0] - x0
            rows = pixels[:, 1] - y0
            rr, cc = draw_polygon(rows, columns, shape=shape)
            if rr.size < 4:
                continue
            labels[rr, cc] = next_label
            label_to_cell_id[next_label] = cell_id
            next_label += 1
        if index % 100 == 0:
            print(f"    GT: {index}/{len(selected)} FOVs, {next_label - 1} cells")
    print(f"  GT complete: {next_label - 1} cells")
    return labels, label_to_cell_id, len(selected), len(all_files)


def run_cellpose3(
    image: np.ndarray,
    use_gpu: bool,
    diameter: float,
    batch_size: int,
    flow_threshold: float,
    cellprob_threshold: float,
) -> tuple[np.ndarray, float, dict]:
    from cellpose import core, models

    cuda_available = bool(torch.cuda.is_available())
    if use_gpu and not cuda_available:
        raise RuntimeError("GPU requested but torch.cuda.is_available() is False")
    if use_gpu and not core.use_gpu():
        raise RuntimeError("Cellpose reports that GPU acceleration is unavailable")
    if cuda_available:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model_started = time.perf_counter()
    model = models.Cellpose(gpu=use_gpu, model_type="cyto3")
    model_load_seconds = time.perf_counter() - model_started
    segmentation_started = time.perf_counter()
    result = model.eval(
        image,
        channels=[1, 2],
        diameter=diameter,
        batch_size=batch_size,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        normalize=True,
        resample=True,
    )
    segmentation_seconds = time.perf_counter() - segmentation_started
    masks = np.squeeze(np.asarray(result[0])).astype(np.int32, copy=False)
    del result, model
    gc.collect()

    stats = {
        "model_load_seconds": model_load_seconds,
        "peak_gpu_memory_allocated_mb": None,
        "peak_gpu_memory_reserved_mb": None,
    }
    if cuda_available:
        torch.cuda.synchronize()
        stats.update(
            peak_gpu_memory_allocated_mb=float(torch.cuda.max_memory_allocated()) / 1024**2,
            peak_gpu_memory_reserved_mb=float(torch.cuda.max_memory_reserved()) / 1024**2,
        )
    return masks, segmentation_seconds, stats


def write_label_tiff(path: Path, mask: np.ndarray, description: str) -> None:
    tifffile.imwrite(
        path,
        mask,
        dtype=np.int32,
        bigtiff=mask.nbytes >= 2**32,
        tile=(512, 512),
        compression="zlib",
        photometric="minisblack",
        metadata={"axes": "YX", "description": description},
    )


def evaluate_masks(gt: np.ndarray, pred: np.ndarray, dataset: str) -> dict:
    if gt.shape != pred.shape:
        raise ValueError(f"GT {gt.shape} and prediction {pred.shape} are not aligned")
    metrics = load_metrics_module()
    raw = metrics.evaluate_single_mask_against_cellpose(
        method_name="Cellpose3-cyto3",
        cellpose_gt_mask=gt,
        pred_mask=pred,
        boundary_mode="thick",
        iou_threshold=0.5,
    )
    result = {
        "dataset": dataset,
        "reference": "Vizgen_official",
        "method": "Cellpose3-cyto3",
        "gt_cell_count": raw.pop("cellpose_reference_cell_count"),
        "pred_cell_count": raw.pop("pred_cell_count"),
    }
    for misleading in ("reference", "method", "mDA_note"):
        raw.pop(misleading, None)
    result.update(raw)
    return result


def peak_rss_mb() -> float:
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak / (1024**2 if sys.platform == "darwin" else 1024.0)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def run_dataset(args: argparse.Namespace) -> dict:
    version = require_cellpose3()
    files = discover_dataset(args.dataset)
    output_dir = args.output_root / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    status = "failed"
    error: str | None = None
    runtime_metadata: dict = {}

    print(f"=== Cellpose3 MERFISH benchmark: {args.dataset} ===", flush=True)
    print(f"Python={sys.executable}")
    print(f"Cellpose={version}; output={output_dir}")
    try:
        shape = tiff_shape(files.dapi)
        roi_size = roi_size_for_dataset(args.dataset, args.roi_size)
        roi_xy = center_roi(shape, roi_size)
        print(f"  mosaic shape={shape}; ROI(x0,y0,x1,y1)={roi_xy}")
        image, input_metadata = build_input(files, roi_xy, args.cyto_channel)
        print(f"  input shape={image.shape}, dtype={image.dtype}")
        print(f"  input={input_metadata['input_description']}")

        pred, segmentation_seconds, model_stats = run_cellpose3(
            image,
            use_gpu=not args.cpu,
            diameter=args.diameter,
            batch_size=args.batch_size,
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold,
        )
        del image
        gc.collect()
        print(f"  segmentation finished in {segmentation_seconds:.2f}s")

        print("  rasterizing Vizgen official boundaries ...")
        gt, label_map, selected_fovs, all_fovs = build_ground_truth(
            files, roi_xy, args.z_index
        )
        prediction_path = output_dir / f"{args.dataset}_cellpose3_pred.tif"
        gt_path = output_dir / f"{args.dataset}_vizgen_gt.tif"
        write_label_tiff(prediction_path, pred, "Cellpose 3 cyto3 instance labels")
        write_label_tiff(gt_path, gt, "Vizgen official cell boundaries")
        pd.DataFrame(
            label_map.items(), columns=["gt_label", "vizgen_cell_id"]
        ).to_csv(output_dir / "gt_label_to_cell_id.csv", index=False)

        print("  evaluating with yourpath/eval/test.py ...")
        result = evaluate_masks(gt, pred, args.dataset)
        result.update(
            {
                "cellpose_version": version,
                "input_mode": input_metadata["input_description"],
                "roi_x0": roi_xy[0],
                "roi_y0": roi_xy[1],
                "roi_x1": roi_xy[2],
                "roi_y1": roi_xy[3],
                "roi_size": roi_size,
                "z_index": args.z_index,
                "diameter": args.diameter,
                "batch_size": args.batch_size,
                "segmentation_seconds": segmentation_seconds,
                **model_stats,
            }
        )
        result["runtime_seconds"] = time.perf_counter() - started
        result["peak_memory_mb"] = peak_rss_mb()
        pd.DataFrame([result]).to_csv(output_dir / "metrics.csv", index=False)
        runtime_metadata = {
            "dataset": args.dataset,
            "status": "success",
            "cellpose_version": version,
            "python": sys.executable,
            "platform": platform.platform(),
            "prediction": str(prediction_path),
            "ground_truth": str(gt_path),
            "roi_x0_y0_x1_y1": list(roi_xy),
            "input": input_metadata,
            "selected_gt_fovs": selected_fovs,
            "total_gt_fovs": all_fovs,
            **{
                key: result[key]
                for key in (
                    "runtime_seconds",
                    "segmentation_seconds",
                    "model_load_seconds",
                    "peak_memory_mb",
                    "peak_gpu_memory_allocated_mb",
                    "peak_gpu_memory_reserved_mb",
                )
            },
        }
        status = "success"
        print(pd.DataFrame([result]).to_string(index=False))
        return result
    except BaseException as exc:
        error = repr(exc)
        traceback.print_exc()
        raise
    finally:
        runtime = time.perf_counter() - started
        final = {
            **runtime_metadata,
            "dataset": args.dataset,
            "status": status,
            "runtime_seconds": runtime,
            "peak_memory_mb": peak_rss_mb(),
            "error": error,
        }
        write_json(output_dir / "performance.json", final)
        print("\n========== performance ==========")
        print(f"status: {status}")
        print(f"total runtime: {runtime:.2f}s ({runtime / 60:.2f} min)")
        print(f"peak CPU RSS: {peak_rss_mb():.2f} MB")
        if runtime_metadata:
            print(
                "peak GPU memory: "
                f"allocated={runtime_metadata.get('peak_gpu_memory_allocated_mb')} MB, "
                f"reserved={runtime_metadata.get('peak_gpu_memory_reserved_mb')} MB"
            )


def check_dataset(name: str, roi_size: int, cyto_channel: str) -> dict:
    files = discover_dataset(name)
    missing = []
    if not files.boundaries.is_dir():
        missing.append(str(files.boundaries))
    boundary_count = len(list(files.boundaries.glob("feature_data_*.hdf5")))
    if boundary_count == 0:
        missing.append(f"{files.boundaries}/feature_data_*.hdf5")
    shapes = {"dapi": tiff_shape(files.dapi), "polyt": tiff_shape(files.polyt)}
    if files.cellbound is not None:
        shapes["cellbound"] = tiff_shape(files.cellbound)
    if len(set(shapes.values())) != 1:
        missing.append(f"channel shapes differ: {shapes}")
    actual_channel = cyto_channel
    if cyto_channel == "cellbound" and files.cellbound is None:
        actual_channel = "polyt"
    chosen = files.cellbound if actual_channel == "cellbound" else files.polyt
    return {
        "dataset": name,
        "image_shape": shapes["dapi"],
        "roi": center_roi(shapes["dapi"], roi_size),
        "roi_size": roi_size,
        "dapi": str(files.dapi),
        "cyto": str(chosen),
        "requested_cyto_channel": cyto_channel,
        "actual_cyto_channel": actual_channel,
        "transform": str(files.transform),
        "metadata": str(files.metadata),
        "boundary_hdf5_count": boundary_count,
        "missing_or_error": missing,
    }


def run_checks(names: Sequence[str], args: argparse.Namespace) -> int:
    failed = False
    try:
        version = require_cellpose3()
        print(f"Cellpose preflight: {version} ({sys.executable})")
        print(f"PyTorch: {torch.__version__} ({torch.__file__})")
        cuda_available = bool(torch.cuda.is_available())
        print(f"CUDA available: {cuda_available}")
        if not args.cpu and not cuda_available:
            print("[FAIL] GPU mode was requested but CUDA is unavailable")
            failed = True
    except Exception as exc:
        print(f"[FAIL] environment: {exc}")
        failed = True
    for name in names:
        try:
            roi_size = roi_size_for_dataset(name, args.roi_size)
            record = check_dataset(name, roi_size, args.cyto_channel)
            problem = bool(record["missing_or_error"])
            failed |= problem
            print(("[FAIL] " if problem else "[OK] ") + json.dumps(record, ensure_ascii=False))
        except Exception as exc:
            failed = True
            print(f"[FAIL] {name}: {exc}")
    return 1 if failed else 0


def aggregate(output_root: Path) -> Path:
    frames = []
    for name in DATASET_ORDER:
        metrics_path = output_root / name / "metrics.csv"
        performance_path = output_root / name / "performance.json"
        if not metrics_path.is_file() or not performance_path.is_file():
            continue
        try:
            performance = json.loads(performance_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if performance.get("status") == "success":
            frames.append(pd.read_csv(metrics_path))
    if not frames:
        raise FileNotFoundError(f"No successful per-dataset metrics under {output_root}")
    summary = pd.concat(frames, ignore_index=True)
    order = {name: index for index, name in enumerate(DATASET_ORDER)}
    summary["_order"] = summary["dataset"].map(order)
    summary = summary.sort_values("_order").drop(columns="_order")
    path = output_root / "cellpose3_merfish_summary.csv"
    summary.to_csv(path, index=False)
    print(summary.to_string(index=False))
    print(f"Summary saved: {path}")
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_ORDER)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--roi-size",
        type=int,
        default=None,
        help="override every dataset ROI; default is 20000 for mouse brain and 10000 otherwise",
    )
    parser.add_argument("--z-index", type=int, default=3)
    parser.add_argument("--cyto-channel", choices=("cellbound", "polyt"), default="cellbound")
    parser.add_argument("--diameter", type=float, default=90.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--flow-threshold", type=float, default=0.4)
    parser.add_argument("--cellprob-threshold", type=float, default=0.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.roi_size is not None and args.roi_size <= 0:
        parser.error("--roi-size must be positive")
    if not args.check and not args.aggregate_only and args.dataset is None:
        parser.error("--dataset is required unless --check or --aggregate-only is used")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.check:
        names = [args.dataset] if args.dataset else list(DATASET_ORDER)
        return run_checks(names, args)
    if args.aggregate_only:
        aggregate(args.output_root)
        return 0
    run_dataset(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
