#!/usr/bin/env python3
"""Cellpose 3 (cyto3) benchmark for ten Xenium datasets.

Most large datasets are evaluated on a reproducible centre 10000 x 10000 ROI.
Breast uses 5000 x 5000, colon uses 6000 x 6000, and pancreas H&E uses the
full slide.  The small xenium_roi_crop dataset and the already prepared
xenium9/xenium10 ROIs are used in full.  Fluorescence inputs are
[fused membrane/cytoplasm, DAPI].  H&E inputs are stain-separated
[eosin, hematoxylin].
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import platform
import resource
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import tifffile
import torch
from skimage.color import rgb2hed
from skimage.draw import polygon as draw_polygon


ROOT = Path("yourpath")
DATA_ROOT = ROOT / "dataset"
DEFAULT_OUTPUT_ROOT = ROOT / "cellpose3" / "outputs"
METRICS_PATH = ROOT / "eval" / "test.py"
DEFAULT_PIXEL_SIZE_UM = 0.2125
ROI_SIZE = 10_000


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    mode: str
    boundary_path: Path
    experiment_path: Path
    morphology_dir: Path | None = None
    nucleus_index: int = 0
    membrane_indices: tuple[int, ...] = (1, 2, 3)
    roi_size: int | None = ROI_SIZE
    boundary_coordinates: str = "global"
    he_source: Path | None = None
    he_alignment: Path | None = None
    he_prepared_roi: Path | None = None
    he_prepared_metadata: Path | None = None
    notes: str = ""


DATASETS: dict[str, DatasetSpec] = {
    "xenium_8": DatasetSpec(
        name="xenium_8",
        mode="fluorescence",
        morphology_dir=DATA_ROOT / "xenium_8" / "morphology_focus",
        boundary_path=DATA_ROOT / "xenium_8" / "cell_boundaries.parquet",
        experiment_path=DATA_ROOT / "xenium_8" / "experiment.xenium",
    ),
    "xenium_breast": DatasetSpec(
        name="xenium_breast",
        mode="fluorescence",
        morphology_dir=DATA_ROOT / "xenium_breast" / "morphology_focus",
        boundary_path=DATA_ROOT / "xenium_breast" / "cell_boundaries.parquet",
        experiment_path=DATA_ROOT / "xenium_breast" / "experiment.xenium",
        roi_size=5_000,
    ),
    "xenium_colon": DatasetSpec(
        name="xenium_colon",
        mode="fluorescence",
        morphology_dir=DATA_ROOT / "xenium_colon" / "morphology_focus",
        boundary_path=DATA_ROOT / "xenium_colon" / "cell_boundaries.parquet",
        experiment_path=DATA_ROOT / "xenium_colon" / "experiment.xenium",
        roi_size=6_000,
    ),
    "xenium_liver": DatasetSpec(
        name="xenium_liver",
        mode="he",
        boundary_path=DATA_ROOT / "xenium_liver" / "outs" / "cell_boundaries.parquet",
        experiment_path=DATA_ROOT / "xenium_liver" / "outs" / "experiment.xenium",
        he_source=DATA_ROOT / "xenium_liver" / "Xenium_V1_hLiver_nondiseased_section_FFPE_he_image.ome.tif",
        he_alignment=DATA_ROOT / "xenium_liver" / "Xenium_V1_hLiver_nondiseased_section_FFPE_he_imagealignment.csv",
        he_prepared_roi=DATA_ROOT / "xenium_liver" / "segmentation_outputs_center10000" / "input" / "xenium_liver_HE_center10000.tif",
        he_prepared_metadata=DATA_ROOT / "xenium_liver" / "segmentation_outputs_center10000" / "input" / "xenium_liver_HE_center10000.json",
    ),
    "xenium_mouse_brain": DatasetSpec(
        name="xenium_mouse_brain",
        mode="fluorescence",
        morphology_dir=DATA_ROOT / "xenium_mouse_brain" / "outs" / "morphology_focus",
        boundary_path=DATA_ROOT / "xenium_mouse_brain" / "outs" / "cell_boundaries.parquet",
        experiment_path=DATA_ROOT / "xenium_mouse_brain" / "outs" / "experiment.xenium",
        notes="Uses the true centre ROI: y=6956, x=12077 for the current image.",
    ),
    "xenium_pancreas_cancer": DatasetSpec(
        name="xenium_pancreas_cancer",
        mode="he",
        boundary_path=DATA_ROOT / "xenium_pancreas_cancer" / "cell_boundaries.parquet",
        experiment_path=DATA_ROOT / "xenium_pancreas_cancer" / "experiment.xenium",
        he_source=DATA_ROOT / "xenium_pancreas_cancer" / "Xenium_V1_hPancreas_Cancer_Add_on_FFPE_he_image.ome.tif",
        he_alignment=DATA_ROOT / "xenium_pancreas_cancer" / "he_imagealignment.csv",
        roi_size=None,
    ),
    "xenium_roi_crop": DatasetSpec(
        name="xenium_roi_crop",
        mode="fluorescence",
        morphology_dir=DATA_ROOT / "xenium_roi_crop" / "morphology_focus",
        boundary_path=DATA_ROOT / "xenium_roi_crop" / "cell_boundaries.parquet",
        experiment_path=DATA_ROOT / "xenium_roi_crop" / "experiment_roi.json",
        roi_size=None,
        boundary_coordinates="local",
    ),
    "xenium7": DatasetSpec(
        name="xenium7",
        mode="fluorescence",
        morphology_dir=DATA_ROOT / "xenium7" / "outs" / "morphology_focus",
        boundary_path=DATA_ROOT / "xenium7" / "outs" / "cell_boundaries.parquet",
        experiment_path=DATA_ROOT / "xenium7" / "outs" / "experiment.xenium",
    ),
    "xenium9": DatasetSpec(
        name="xenium9",
        mode="fluorescence",
        morphology_dir=DATA_ROOT / "xenium9" / "benchmark_roi_10000" / "morphology_focus",
        boundary_path=DATA_ROOT / "xenium9" / "benchmark_roi_10000" / "cell_boundaries.parquet",
        experiment_path=DATA_ROOT / "xenium9" / "benchmark_roi_10000" / "experiment.xenium",
        membrane_indices=(1,),
        roi_size=None,
        boundary_coordinates="local",
    ),
    "xenium10": DatasetSpec(
        name="xenium10",
        mode="fluorescence",
        morphology_dir=DATA_ROOT / "xenium10" / "benchmark_roi_10000" / "morphology_focus",
        boundary_path=DATA_ROOT / "xenium10" / "benchmark_roi_10000" / "cell_boundaries.parquet",
        experiment_path=DATA_ROOT / "xenium10" / "benchmark_roi_10000" / "experiment.xenium",
        membrane_indices=(1,),
        roi_size=None,
        boundary_coordinates="local",
    ),
}

DATASET_ORDER = tuple(DATASETS)


@dataclass(frozen=True)
class ChannelSource:
    path: Path
    page: int
    name: str
    shape: tuple[int, int]


@dataclass
class InputBundle:
    image: np.ndarray
    crop_yx: tuple[int, int, int, int]
    source_names: list[str]
    input_description: str
    he_origin_xy: tuple[int, int] | None = None
    metadata: dict = field(default_factory=dict)


def load_metrics_module():
    if not METRICS_PATH.is_file():
        raise FileNotFoundError(f"Metrics implementation not found: {METRICS_PATH}")
    spec = importlib.util.spec_from_file_location("cellpose3_xenium_metrics", METRICS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load metrics module: {METRICS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cellpose_version() -> str:
    try:
        return package_version("cellpose")
    except PackageNotFoundError:
        return "not-installed"


def require_cellpose3() -> str:
    cp_version = cellpose_version()
    if not cp_version.startswith("3."):
        raise RuntimeError(
            f"This benchmark requires Cellpose 3.x, but Python {sys.executable} "
            f"sees cellpose {cp_version}. Finish installing Cellpose 3 in the "
            "conda environment named cellpose3, then run through "
            "run_all_xenium_cellpose3.sh."
        )
    return cp_version


def _natural_channel_index(path: Path) -> tuple[int, str]:
    import re

    match = re.search(r"(?:ch|focus_?)0*(\d+)", path.name.lower())
    return (int(match.group(1)) if match else 10**9, path.name)


def _page_spatial_shape(page: tifffile.TiffPage) -> tuple[int, int]:
    shape = tuple(int(v) for v in page.shape)
    if len(shape) == 2:
        return shape
    if len(shape) == 3 and shape[-1] in (3, 4):
        return shape[:2]
    raise ValueError(f"Expected a 2-D channel page or RGB page, got {shape}")


def discover_fluorescence_channels(directory: Path) -> list[ChannelSource]:
    files = sorted(
        [*directory.glob("*.ome.tif"), *directory.glob("*.ome.tiff"),
         *directory.glob("*.tif"), *directory.glob("*.tiff")],
        key=_natural_channel_index,
    )
    # Remove duplicates caused by overlapping glob suffixes.
    files = list(dict.fromkeys(files))
    if not files:
        raise FileNotFoundError(f"No morphology TIFF files found in {directory}")

    sources: list[ChannelSource] = []
    if len(files) == 1:
        path = files[0]
        with tifffile.TiffFile(path) as tif:
            for page_index, page in enumerate(tif.pages):
                sources.append(ChannelSource(
                    path=path,
                    page=page_index,
                    name=f"{path.name}:page{page_index}",
                    shape=_page_spatial_shape(page),
                ))
    else:
        # Xenium multi-file OME metadata may describe all channels in every
        # file.  Each physical file nevertheless contains exactly one page;
        # page 0 is the channel named by that file.
        for path in files:
            with tifffile.TiffFile(path) as tif:
                sources.append(ChannelSource(
                    path=path,
                    page=0,
                    name=path.name,
                    shape=_page_spatial_shape(tif.pages[0]),
                ))

    shapes = {source.shape for source in sources}
    if len(shapes) != 1:
        raise ValueError(f"Morphology channel shapes differ in {directory}: {shapes}")
    return sources


def centered_crop(shape: tuple[int, int], size: int | None) -> tuple[int, int, int, int]:
    height, width = shape
    if size is None:
        return 0, height, 0, width
    if min(height, width) < size:
        raise ValueError(f"Requested {size}x{size} ROI exceeds image shape {shape}")
    y0 = (height - size) // 2
    x0 = (width - size) // 2
    return y0, y0 + size, x0, x0 + size


def read_tiff_crop(
    path: Path,
    crop_yx: tuple[int, int, int, int],
    page_index: int = 0,
) -> np.ndarray:
    """Read only a tiled TIFF ROI through tifffile's Zarr interface."""
    import zarr

    y0, y1, x0, x1 = crop_yx
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[page_index]
        store = page.aszarr()
        try:
            array = zarr.open(store, mode="r")
            if not hasattr(array, "shape"):
                keys = sorted(array.array_keys())
                if not keys:
                    raise ValueError(f"Zarr group has no arrays: {path}")
                array = array["0"] if "0" in keys else array[keys[0]]
            if len(array.shape) == 2:
                result = np.asarray(array[y0:y1, x0:x1])
            elif len(array.shape) == 3 and array.shape[-1] in (3, 4):
                result = np.asarray(array[y0:y1, x0:x1, :3])
            else:
                raise ValueError(f"Unsupported TIFF page shape {array.shape}: {path}")
        finally:
            close = getattr(store, "close", None)
            if close is not None:
                close()
    return result


def robust_limits(array: np.ndarray, low: float = 1.0, high: float = 99.8) -> tuple[float, float]:
    stride = max(1, int(np.sqrt(array.shape[0] * array.shape[1] / 250_000)))
    sample = np.asarray(array[::stride, ::stride]).reshape(-1)
    if np.issubdtype(sample.dtype, np.floating):
        sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(sample, (low, high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(sample.min()), float(sample.max())
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def scale_to_uint16(
    array: np.ndarray,
    limits: tuple[float, float] | None = None,
    block_rows: int = 512,
) -> tuple[np.ndarray, tuple[float, float]]:
    lo, hi = limits if limits is not None else robust_limits(array)
    scale = np.float32(65535.0 / (hi - lo))
    output = np.empty(array.shape, dtype=np.uint16)
    for y0 in range(0, array.shape[0], block_rows):
        y1 = min(y0 + block_rows, array.shape[0])
        block = array[y0:y1].astype(np.float32, copy=False)
        block = np.clip((block - np.float32(lo)) * scale, 0.0, 65535.0)
        output[y0:y1] = block.astype(np.uint16)
    return output, (lo, hi)


def build_fluorescence_input(spec: DatasetSpec) -> InputBundle:
    assert spec.morphology_dir is not None
    sources = discover_fluorescence_channels(spec.morphology_dir)
    requested = (spec.nucleus_index, *spec.membrane_indices)
    missing = [index for index in requested if index >= len(sources)]
    if missing:
        raise IndexError(
            f"{spec.name}: missing channel indices {missing}; found {len(sources)} channels"
        )

    crop = centered_crop(sources[0].shape, spec.roi_size)
    print(f"  morphology shape={sources[0].shape}, ROI(y0,y1,x0,x1)={crop}")

    raw_nucleus = read_tiff_crop(
        sources[spec.nucleus_index].path, crop, sources[spec.nucleus_index].page
    )
    nucleus, nucleus_limits = scale_to_uint16(raw_nucleus)
    del raw_nucleus

    fused_membrane: np.ndarray | None = None
    membrane_limits: dict[str, tuple[float, float]] = {}
    membrane_names: list[str] = []
    for index in spec.membrane_indices:
        source = sources[index]
        raw = read_tiff_crop(source.path, crop, source.page)
        normalized, limits = scale_to_uint16(raw)
        del raw
        membrane_limits[source.name] = limits
        membrane_names.append(source.name)
        if fused_membrane is None:
            fused_membrane = normalized
        else:
            np.maximum(fused_membrane, normalized, out=fused_membrane)
        del normalized

    if fused_membrane is None:
        raise RuntimeError(f"{spec.name}: no membrane channels configured")
    image = np.empty((*nucleus.shape, 2), dtype=np.uint16)
    image[..., 0] = fused_membrane
    image[..., 1] = nucleus
    del fused_membrane, nucleus
    gc.collect()

    source_names = [*membrane_names, sources[spec.nucleus_index].name]
    metadata = {
        "nucleus_channel": sources[spec.nucleus_index].name,
        "membrane_channels": membrane_names,
        "membrane_fusion": "per-channel percentile normalization then pixelwise maximum",
        "nucleus_limits": nucleus_limits,
        "membrane_limits": membrane_limits,
    }
    return InputBundle(
        image=image,
        crop_yx=crop,
        source_names=source_names,
        input_description="[fused membrane/cytoplasm, DAPI]",
        metadata=metadata,
    )


def resolve_he_roi(spec: DatasetSpec) -> tuple[Path, tuple[int, int, int, int], tuple[int, int], dict]:
    assert spec.he_source is not None
    if spec.he_prepared_roi and spec.he_prepared_metadata:
        if spec.he_prepared_roi.is_file() and spec.he_prepared_metadata.is_file():
            record = json.loads(spec.he_prepared_metadata.read_text(encoding="utf-8"))
            origin = tuple(int(v) for v in record["roi_origin_xy_in_source"])
            with tifffile.TiffFile(spec.he_prepared_roi) as tif:
                shape = _page_spatial_shape(tif.pages[0])
            crop = (0, shape[0], 0, shape[1])
            return spec.he_prepared_roi, crop, origin, record

    with tifffile.TiffFile(spec.he_source) as tif:
        shape = _page_spatial_shape(tif.pages[0])
    crop = centered_crop(shape, spec.roi_size)
    y0, _, x0, _ = crop
    return spec.he_source, crop, (x0, y0), {
        "source_tif": str(spec.he_source),
        "source_shape_yx": list(shape),
        "roi_origin_xy_in_source": [x0, y0],
        "roi_crop_y0_y1_x0_x1": list(crop),
    }


def _hed_sample_limits(rgb: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    stride = max(1, int(np.sqrt(rgb.shape[0] * rgb.shape[1] / 250_000)))
    sample = rgb[::stride, ::stride, :3].astype(np.float32) / 255.0
    hed = rgb2hed(sample)
    hematoxylin = np.maximum(hed[..., 0], 0.0)
    eosin = np.maximum(hed[..., 1], 0.0)
    return robust_limits(eosin, 0.5, 99.5), robust_limits(hematoxylin, 0.5, 99.5)


def build_he_input(spec: DatasetSpec) -> InputBundle:
    image_path, crop, origin_xy, he_record = resolve_he_roi(spec)
    rgb = read_tiff_crop(image_path, crop)
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        raise ValueError(f"{spec.name}: expected RGB H&E image, got {rgb.shape}")
    if rgb.dtype != np.uint8:
        rgb, _ = scale_to_uint16(rgb)
        rgb = (rgb / 257).astype(np.uint8)

    eosin_limits, hematoxylin_limits = _hed_sample_limits(rgb)
    image = np.empty((rgb.shape[0], rgb.shape[1], 2), dtype=np.uint16)
    for y0 in range(0, rgb.shape[0], 512):
        y1 = min(y0 + 512, rgb.shape[0])
        hed = rgb2hed(rgb[y0:y1, :, :3].astype(np.float32) / 255.0)
        eosin = np.maximum(hed[..., 1], 0.0)
        hematoxylin = np.maximum(hed[..., 0], 0.0)
        image[y0:y1, :, 0], _ = scale_to_uint16(eosin, eosin_limits, block_rows=512)
        image[y0:y1, :, 1], _ = scale_to_uint16(
            hematoxylin, hematoxylin_limits, block_rows=512
        )
        del hed, eosin, hematoxylin
    del rgb
    gc.collect()

    return InputBundle(
        image=image,
        crop_yx=crop,
        source_names=[str(image_path)],
        input_description="H&E stain deconvolution [eosin, hematoxylin]",
        he_origin_xy=origin_xy,
        metadata={
            **he_record,
            "eosin_limits": eosin_limits,
            "hematoxylin_limits": hematoxylin_limits,
        },
    )


def read_pixel_size(path: Path) -> float:
    if not path.is_file():
        warnings.warn(f"Missing {path}; using {DEFAULT_PIXEL_SIZE_UM} um/px")
        return DEFAULT_PIXEL_SIZE_UM
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.warn(f"Could not parse {path}: {exc}; using default pixel size")
        return DEFAULT_PIXEL_SIZE_UM

    values: list[float] = []

    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if "pixel_size" in str(key).lower() and isinstance(child, (int, float)):
                    values.append(float(child))
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(data)
    return values[0] if values else DEFAULT_PIXEL_SIZE_UM


def boundary_columns(frame: pd.DataFrame, prefer_local: bool) -> tuple[str, str, str]:
    lower = {str(column).lower(): str(column) for column in frame.columns}
    id_column = next(
        (lower[name] for name in ("cell_id", "label_id", "label", "id") if name in lower),
        None,
    )
    if prefer_local:
        x_candidates = ("vertex_x_local_um", "vertex_x_local", "vertex_x", "x")
        y_candidates = ("vertex_y_local_um", "vertex_y_local", "vertex_y", "y")
    else:
        x_candidates = ("vertex_x", "vertex_x_um", "x", "x_location")
        y_candidates = ("vertex_y", "vertex_y_um", "y", "y_location")
    x_column = next((lower[name] for name in x_candidates if name in lower), None)
    y_column = next((lower[name] for name in y_candidates if name in lower), None)
    if not id_column or not x_column or not y_column:
        raise ValueError(f"Cannot identify boundary columns in {list(frame.columns)}")
    return id_column, x_column, y_column


def rasterize_vertices(
    ids: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, list[tuple[int, str]]]:
    """Rasterize ordered polygon vertices into a sequential int32 label mask."""
    codes, unique_ids = pd.factorize(ids, sort=False)
    order = np.argsort(codes, kind="stable")
    codes = codes[order]
    xs = xs[order]
    ys = ys[order]
    starts = np.searchsorted(codes, np.arange(len(unique_ids)), side="left")
    ends = np.searchsorted(codes, np.arange(len(unique_ids)), side="right")

    height, width = shape
    mask = np.zeros(shape, dtype=np.int32)
    label_map: list[tuple[int, str]] = []
    for code, (start, end) in enumerate(zip(starts, ends)):
        if end - start < 3:
            continue
        px = xs[start:end]
        py = ys[start:end]
        if px.max() < 0 or px.min() >= width or py.max() < 0 or py.min() >= height:
            continue
        rr, cc = draw_polygon(py, px, shape=shape)
        if rr.size == 0:
            continue
        label = len(label_map) + 1
        mask[rr, cc] = label
        label_map.append((label, str(unique_ids[code])))
        if label % 25_000 == 0:
            print(f"    rasterized {label:,} GT cells")
    return mask, label_map


def build_gt(
    spec: DatasetSpec,
    bundle: InputBundle,
    output_dir: Path,
    output_shape: tuple[int, int],
) -> tuple[np.ndarray, str]:
    frame = pd.read_parquet(spec.boundary_path)
    prefer_local = spec.boundary_coordinates == "local"
    id_column, x_column, y_column = boundary_columns(frame, prefer_local)
    ids = frame[id_column].to_numpy()
    vertices_um = frame[[x_column, y_column]].to_numpy(dtype=np.float64)
    pixel_size = read_pixel_size(spec.experiment_path)

    if spec.mode == "fluorescence":
        y0, _, x0, _ = bundle.crop_yx
        xs = vertices_um[:, 0] / pixel_size - x0
        ys = vertices_um[:, 1] / pixel_size - y0
        coordinate_note = "Xenium morphology pixels"
    else:
        assert spec.he_alignment is not None and bundle.he_origin_xy is not None
        matrix = np.loadtxt(spec.he_alignment, delimiter=",")
        if matrix.shape != (3, 3):
            raise ValueError(f"Expected 3x3 H&E alignment matrix: {spec.he_alignment}")
        xenium_pixels = vertices_um / pixel_size
        homogeneous = np.column_stack((xenium_pixels, np.ones(len(xenium_pixels))))
        he_pixels = homogeneous @ np.linalg.inv(matrix).T
        origin_x, origin_y = bundle.he_origin_xy
        xs = he_pixels[:, 0] - origin_x
        ys = he_pixels[:, 1] - origin_y
        coordinate_note = "official Xenium boundaries transformed into H&E ROI pixels"

    gt, label_map = rasterize_vertices(ids, xs, ys, output_shape)
    pd.DataFrame(label_map, columns=["gt_label", "xenium_cell_id"]).to_csv(
        output_dir / "gt_label_to_cell_id.csv", index=False
    )
    del frame, ids, vertices_um, xs, ys
    return gt, coordinate_note


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
        channels=[1, 2],  # channel 1=membrane/eosin; channel 2=nucleus/hematoxylin
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

    gpu_stats = {
        "peak_gpu_memory_allocated_mb": None,
        "peak_gpu_memory_reserved_mb": None,
    }
    if cuda_available:
        torch.cuda.synchronize()
        gpu_stats = {
            "peak_gpu_memory_allocated_mb": float(torch.cuda.max_memory_allocated()) / 1024**2,
            "peak_gpu_memory_reserved_mb": float(torch.cuda.max_memory_reserved()) / 1024**2,
        }
    return masks, segmentation_seconds, {
        "model_load_seconds": model_load_seconds,
        **gpu_stats,
    }


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


def evaluate_masks(gt: np.ndarray, pred: np.ndarray, dataset_name: str) -> dict:
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
        "dataset": dataset_name,
        "method": "Cellpose3-cyto3",
        "gt_cell_count": raw.pop("cellpose_reference_cell_count"),
        "pred_cell_count": raw.pop("pred_cell_count"),
    }
    for misleading in ("reference", "method", "mDA_note"):
        raw.pop(misleading, None)
    result.update(raw)
    return result


def current_peak_rss_mb() -> float:
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak / (1024**2 if sys.platform == "darwin" else 1024.0)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def run_dataset(args: argparse.Namespace) -> dict:
    cp_version = require_cellpose3()
    spec = DATASETS[args.dataset]
    output_dir = args.output_root / spec.name
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    status = "failed"
    failure: str | None = None
    runtime_metadata: dict = {}

    print(f"=== Cellpose3 Xenium benchmark: {spec.name} ===", flush=True)
    print(f"Python={sys.executable}")
    print(f"Cellpose={cp_version}; mode={spec.mode}; output={output_dir}")
    try:
        bundle = build_fluorescence_input(spec) if spec.mode == "fluorescence" else build_he_input(spec)
        print(f"  input shape={bundle.image.shape}, dtype={bundle.image.dtype}")
        print(f"  input={bundle.input_description}")
        print(f"  sources={bundle.source_names}")

        pred, segmentation_seconds, model_stats = run_cellpose3(
            bundle.image,
            use_gpu=not args.cpu,
            diameter=args.diameter,
            batch_size=args.batch_size,
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold,
        )
        del bundle.image
        gc.collect()
        print(f"  segmentation finished in {segmentation_seconds:.2f}s")

        print("  rasterizing official Xenium ground truth ...")
        gt, gt_coordinate_system = build_gt(spec, bundle, output_dir, pred.shape)
        prediction_path = output_dir / f"{spec.name}_cellpose3_pred.tif"
        gt_path = output_dir / f"{spec.name}_gt.tif"
        write_label_tiff(prediction_path, pred, "Cellpose 3 cyto3 instance labels")
        write_label_tiff(gt_path, gt, gt_coordinate_system)

        print("  evaluating masks with yourpath/eval/test.py ...")
        result = evaluate_masks(gt, pred, spec.name)
        result.update({
            "cellpose_version": cp_version,
            "input_mode": bundle.input_description,
            "roi_y0": bundle.crop_yx[0],
            "roi_y1": bundle.crop_yx[1],
            "roi_x0": bundle.crop_yx[2],
            "roi_x1": bundle.crop_yx[3],
            "diameter": args.diameter,
            "batch_size": args.batch_size,
            "segmentation_seconds": segmentation_seconds,
            **model_stats,
        })
        result["runtime_seconds"] = time.perf_counter() - started
        result["peak_memory_mb"] = current_peak_rss_mb()
        pd.DataFrame([result]).to_csv(output_dir / "metrics.csv", index=False)
        runtime_metadata = {
            "dataset": spec.name,
            "status": "success",
            "cellpose_version": cp_version,
            "python": sys.executable,
            "platform": platform.platform(),
            "prediction": str(prediction_path),
            "ground_truth": str(gt_path),
            "ground_truth_coordinates": gt_coordinate_system,
            "crop_y0_y1_x0_x1": list(bundle.crop_yx),
            "he_origin_xy": list(bundle.he_origin_xy) if bundle.he_origin_xy else None,
            "input": bundle.metadata,
            **{key: result[key] for key in (
                "runtime_seconds", "segmentation_seconds", "model_load_seconds",
                "peak_memory_mb", "peak_gpu_memory_allocated_mb",
                "peak_gpu_memory_reserved_mb",
            )},
        }
        status = "success"
        print(pd.DataFrame([result]).to_string(index=False))
        return result
    except BaseException as exc:
        failure = repr(exc)
        traceback.print_exc()
        raise
    finally:
        runtime = time.perf_counter() - started
        final = {
            **runtime_metadata,
            "dataset": spec.name,
            "status": status,
            "runtime_seconds": runtime,
            "peak_memory_mb": current_peak_rss_mb(),
            "error": failure,
        }
        write_json(output_dir / "performance.json", final)
        print("\n========== performance ==========")
        print(f"status: {status}")
        print(f"total runtime: {runtime:.2f}s ({runtime / 60:.2f} min)")
        print(f"peak CPU RSS: {current_peak_rss_mb():.2f} MB")
        if runtime_metadata:
            print(
                "peak GPU memory: "
                f"allocated={runtime_metadata.get('peak_gpu_memory_allocated_mb')} MB, "
                f"reserved={runtime_metadata.get('peak_gpu_memory_reserved_mb')} MB"
            )


def required_paths(spec: DatasetSpec) -> Iterable[Path]:
    yield spec.boundary_path
    yield spec.experiment_path
    if spec.mode == "fluorescence":
        assert spec.morphology_dir is not None
        yield spec.morphology_dir
    else:
        assert spec.he_source is not None and spec.he_alignment is not None
        yield spec.he_source
        yield spec.he_alignment


def check_dataset(spec: DatasetSpec) -> dict:
    missing = [str(path) for path in required_paths(spec) if not path.exists()]
    record: dict = {"dataset": spec.name, "mode": spec.mode, "missing": missing}
    if missing:
        return record
    if spec.mode == "fluorescence":
        assert spec.morphology_dir is not None
        sources = discover_fluorescence_channels(spec.morphology_dir)
        crop = centered_crop(sources[0].shape, spec.roi_size)
        selected = [spec.nucleus_index, *spec.membrane_indices]
        if max(selected) >= len(sources):
            record["error"] = f"selected channels {selected}, only {len(sources)} found"
        record.update({
            "image_shape": sources[0].shape,
            "crop": crop,
            "channels_found": len(sources),
            "selected_channels": [sources[i].name for i in selected if i < len(sources)],
        })
    else:
        path, crop, origin, metadata = resolve_he_roi(spec)
        record.update({
            "image": str(path),
            "crop": crop,
            "origin_xy_in_full_he": origin,
            "metadata": metadata,
        })
    return record


def run_checks(names: Sequence[str], require_package: bool = True) -> int:
    failed = False
    if require_package:
        try:
            cp_version = require_cellpose3()
            print(f"Cellpose preflight: {cp_version} ({sys.executable})")
        except Exception as exc:
            print(f"[FAIL] {exc}")
            failed = True
    for name in names:
        try:
            record = check_dataset(DATASETS[name])
            problem = bool(record.get("missing") or record.get("error"))
            failed |= problem
            print(("[FAIL] " if problem else "[OK] ") + json.dumps(record, ensure_ascii=False))
        except Exception as exc:
            failed = True
            print(f"[FAIL] {name}: {exc}")
    return 1 if failed else 0


def aggregate(output_root: Path) -> Path:
    frames = []
    for name in DATASET_ORDER:
        path = output_root / name / "metrics.csv"
        if path.is_file():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No per-dataset metrics.csv files under {output_root}")
    summary = pd.concat(frames, ignore_index=True)
    order = {name: index for index, name in enumerate(DATASET_ORDER)}
    summary["_order"] = summary["dataset"].map(order)
    summary = summary.sort_values("_order").drop(columns="_order")
    path = output_root / "cellpose3_xenium_summary.csv"
    summary.to_csv(path, index=False)
    print(summary.to_string(index=False))
    print(f"Summary saved: {path}")
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_ORDER)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--diameter", type=float, default=60.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--flow-threshold", type=float, default=0.4)
    parser.add_argument("--cellprob-threshold", type=float, default=0.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.check and not args.aggregate_only and args.dataset is None:
        parser.error("--dataset is required unless --check or --aggregate-only is used")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.check:
        names = [args.dataset] if args.dataset else list(DATASET_ORDER)
        return run_checks(names)
    if args.aggregate_only:
        aggregate(args.output_root)
        return 0
    run_dataset(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
