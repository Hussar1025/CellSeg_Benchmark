#!/usr/bin/env python3
"""Run the official DISSECT pipeline on five Xenium benchmark ROIs.

The benchmark uses one fixed set of DISSECT parameters for all datasets.  The
whole Xenium image is *not* segmented: breast uses a centred 5,000 x 5,000 px
ROI, liver/brain/colon use centred 10,000 x 10,000 px ROIs, and
``xenium_roi_crop`` is already an extracted ROI and is processed in full.

The public DISSECT reader loads an entire TIFF and an entire transcript table.
That is inconvenient for Xenium sections containing hundreds of millions of
transcripts, so this wrapper keeps the algorithm unchanged while reading only
the requested image tiles and transcript parquet row groups.  Coordinates are
converted with the official 0.2125 um/px Xenium conversion, then translated to
ROI-local coordinates before detector/refinement are called.

Typical usage (after ``conda activate dissect``)::

    python run_xenium_benchmark_dissect.py --dataset all
    python run_xenium_benchmark_dissect.py --dataset xenium_liver
    python run_xenium_benchmark_dissect.py --dataset all --dry-run

Outputs are written below ``/data/qiuyijia/DISSECT/xenium_benchmark`` unless
``--output-root`` is supplied.  Each dataset directory contains ``mask.npy``,
``boxes.csv``, the detector's ``raw_boxes.csv``, ``run_info.json`` and the
ROI definition in ``roi_info.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple


ROOT = Path("/data/qiuyijia/DISSECT")
DATASET_ROOT = Path("/data/qiuyijia/dataset")
DEFAULT_OUTPUT_ROOT = ROOT / "xenium_benchmark"
DEFAULT_WEIGHTS = ROOT / "weights/model_weights.pth"
DEFAULT_CONFIG = ROOT / "code/config.yaml"
PIXEL_SIZE_UM = 0.2125  # official DISSECT STReader conversion


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    image: Path
    transcripts: Path
    roi_size: Optional[Tuple[int, int]]  # (width, height); None means full image
    fixed_origin: Optional[Tuple[int, int]] = None  # (x0, y0), for pre-cropped ROI
    description: str = ""


DATASETS = {
    "xenium_breast": DatasetSpec(
        "xenium_breast",
        DATASET_ROOT / "xenium_breast/morphology_focus/ch0000_dapi.ome.tif",
        DATASET_ROOT / "xenium_breast/transcripts.parquet",
        (5000, 5000),
        description="centred 5,000 x 5,000 px ROI",
    ),
    "xenium_roi_crop": DatasetSpec(
        "xenium_roi_crop",
        DATASET_ROOT / "xenium_roi_crop/morphology_focus/morphology_focus_0000.ome.tif",
        DATASET_ROOT / "xenium_roi_crop/transcripts.parquet",
        None,
        # experiment_roi.json records the source-image origin of this crop.
        fixed_origin=(18741, 11870),
        description="already cropped ROI (process full image)",
    ),
    "xenium_liver": DatasetSpec(
        "xenium_liver",
        DATASET_ROOT / "xenium_liver/outs/morphology_focus.ome.tif",
        DATASET_ROOT / "xenium_liver/outs/transcripts.parquet",
        (10000, 10000),
        description="centred 10,000 x 10,000 px ROI",
    ),
    "xenium_mouse_brain": DatasetSpec(
        "xenium_mouse_brain",
        DATASET_ROOT / "xenium_mouse_brain/outs/morphology_focus/morphology_focus_0000.ome.tif",
        DATASET_ROOT / "xenium_mouse_brain/outs/transcripts.parquet",
        (10000, 10000),
        description="centred 10,000 x 10,000 px ROI",
    ),
    "xenium_colon": DatasetSpec(
        "xenium_colon",
        DATASET_ROOT / "xenium_colon/morphology_focus/morphology_focus_0000.ome.tif",
        DATASET_ROOT / "xenium_colon/transcripts.parquet",
        (10000, 10000),
        description="centred 10,000 x 10,000 px ROI",
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DISSECT segmentation of fixed Xenium benchmark ROIs."
    )
    p.add_argument(
        "--dataset", choices=["all", *DATASETS], default="all",
        help="Dataset to process (default: all five, in a fixed order).",
    )
    p.add_argument("--image", type=Path, default=None,
                   help="Override image path (only with one --dataset).")
    p.add_argument("--transcripts", type=Path, default=None,
                   help="Override transcripts.parquet path (only with one --dataset).")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                   help="Root directory for per-dataset outputs.")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                   help="DISSECT model_weights.pth.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help="DISSECT config YAML.")
    p.add_argument("--n-jobs", type=int, default=4,
                   help="Parallel workers for detection/refinement (default: 4).")
    p.add_argument("--stride", type=int, default=128,
                   help="Sliding-window stride (official default: 128).")
    p.add_argument("--ratio-stride", type=int, default=3,
                   help="Window/stride ratio (official default: 3).")
    p.add_argument("--threshold", type=float, default=-0.20,
                   help="GIoU-NMS threshold (official default: -0.20).")
    p.add_argument("--minarea", type=int, default=350,
                   help="Minimum proposal area (fixed for every dataset).")
    p.add_argument("--maxarea", type=int, default=1600,
                   help="Maximum proposal area (fixed for every dataset).")
    p.add_argument("--repetition", type=int, default=2,
                   help="Detector repetitions (fixed for every dataset).")
    p.add_argument("--refine-batch-size", type=int, default=512,
                   help="Number of proposals per refinement batch.")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Transcript-gradient weight (official default: 0.5).")
    p.add_argument("--expand-by", type=int, default=5,
                   help="Pixels added around each proposal (default: 5).")
    p.add_argument("--neighbor", type=int, choices=(4, 8), default=8,
                   help="Transcript-gradient neighbourhood (default: 8).")
    p.add_argument("--num-proposals", type=int, default=None)
    p.add_argument("--sample-step", type=int, default=None)
    p.add_argument("--renewal-thres", type=float, default=None)
    p.add_argument("--seed", type=int, default=2024,
                   help="Random seed shared by all datasets (default: 2024).")
    p.add_argument("--no-gene", action="store_true",
                   help="Disable transcript-guided refinement for an image-only run.")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate paths/ROI geometry without importing the model.")
    p.add_argument("--count-transcripts", action="store_true",
                   help="With --dry-run, scan parquet row groups and count transcripts in ROI.")
    return p.parse_args()


def _image_shape(image_path: Path) -> Tuple[int, int, str]:
    """Return (height, width, dtype) without decoding the complete image."""
    import tifffile

    with tifffile.TiffFile(image_path) as tif:
        page = tif.pages[0]
        shape = tuple(int(v) for v in page.shape)
        if len(shape) < 2:
            raise ValueError(f"Image page is not two-dimensional: {image_path} {shape}")
        return shape[-2], shape[-1], str(page.dtype)


def _roi_for(spec: DatasetSpec, image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Return half-open (x0, y0, width, height) image coordinates."""
    height, width = image_shape
    if spec.roi_size is None:
        x0, y0 = spec.fixed_origin or (0, 0)
        roi_w, roi_h = width, height
        if x0 + roi_w > width or y0 + roi_h > height:
            # A pre-cropped image has local dimensions; its fixed origin is
            # allowed to be outside those local dimensions.
            if spec.fixed_origin is not None:
                return x0, y0, roi_w, roi_h
            raise ValueError(f"ROI exceeds image bounds for {spec.name}")
        return x0, y0, roi_w, roi_h
    roi_w, roi_h = spec.roi_size
    if roi_w > width or roi_h > height:
        raise ValueError(
            f"Requested ROI {roi_w}x{roi_h} exceeds image {width}x{height} for {spec.name}"
        )
    # Floor is deterministic for even and odd image dimensions and matches
    # the usual definition of a centred, half-open pixel interval.
    return (width - roi_w) // 2, (height - roi_h) // 2, roi_w, roi_h


def _image_origin_for(spec: DatasetSpec, roi_x0: int, roi_y0: int) -> Tuple[int, int]:
    """Origin to use when indexing the supplied image file.

    ``xenium_roi_crop`` stores a local crop but its transcripts retain global
    Xenium coordinates.  Its ROI origin is therefore used for transcript
    filtering, while the image itself is indexed from (0, 0).
    """
    return (0, 0) if spec.fixed_origin is not None else (roi_x0, roi_y0)


def _row_group_intersects(pf, rg_index: int, x_um0: float, x_um1: float,
                          y_um0: float, y_um1: float, x_idx: int, y_idx: int) -> bool:
    """Use row-group statistics for cheap spatial pruning when available."""
    rg = pf.metadata.row_group(rg_index)
    for idx, lower, upper in ((x_idx, x_um0, x_um1), (y_idx, y_um0, y_um1)):
        stats = rg.column(idx).statistics
        if stats is None or not getattr(stats, "has_min_max", True):
            continue
        try:
            minimum, maximum = float(stats.min), float(stats.max)
        except (TypeError, ValueError):
            continue
        if maximum < lower or minimum >= upper:
            return False
    return True


def _parquet_row_groups(path: Path, x0: int, y0: int, width: int, height: int) -> Tuple[object, list[int]]:
    """Open parquet and identify row groups that can overlap the ROI."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    names = list(pf.schema_arrow.names)
    required = {"x_location", "y_location", "feature_name"}
    missing = required.difference(names)
    if missing:
        raise ValueError(f"{path} is missing Xenium columns: {sorted(missing)}")
    x_idx, y_idx = names.index("x_location"), names.index("y_location")
    x_um0, x_um1 = x0 * PIXEL_SIZE_UM, (x0 + width) * PIXEL_SIZE_UM
    y_um0, y_um1 = y0 * PIXEL_SIZE_UM, (y0 + height) * PIXEL_SIZE_UM
    selected = [
        i for i in range(pf.num_row_groups)
        if _row_group_intersects(pf, i, x_um0, x_um1, y_um0, y_um1, x_idx, y_idx)
    ]
    return pf, selected


def _iter_roi_transcript_frames(path: Path, x0: int, y0: int, width: int, height: int,
                                count_only: bool = False) -> Iterator[object]:
    """Yield local-coordinate transcript frames, one parquet row group at a time."""
    import numpy as np
    import pandas as pd

    pf, row_groups = _parquet_row_groups(path, x0, y0, width, height)
    columns = ["x_location", "y_location", "feature_name"]
    for rg_index in row_groups:
        table = pf.read_row_group(rg_index, columns=columns)
        frame = table.to_pandas()
        x_raw = frame["x_location"].to_numpy()
        y_raw = frame["y_location"].to_numpy()
        finite = np.isfinite(x_raw) & np.isfinite(y_raw)
        # Match STReader exactly: cast the um/px quotient to int32 (truncate
        # toward zero), then apply the global-to-local ROI translation.
        x_px = np.zeros(len(frame), dtype=np.int32)
        y_px = np.zeros(len(frame), dtype=np.int32)
        x_px[finite] = (x_raw[finite] / PIXEL_SIZE_UM).astype(np.int32)
        y_px[finite] = (y_raw[finite] / PIXEL_SIZE_UM).astype(np.int32)
        keep = (
            finite
            & (x_px >= x0) & (x_px < x0 + width)
            & (y_px >= y0) & (y_px < y0 + height)
        )
        if count_only:
            yield int(np.count_nonzero(keep))
            continue
        if not np.any(keep):
            continue
        local = {
            "geneID": frame.loc[keep, "feature_name"].to_numpy(copy=True),
            "x": (x_px[keep] - x0).astype(np.int32, copy=False),
            "y": (y_px[keep] - y0).astype(np.int32, copy=False),
            "MIDCount": np.ones(int(np.count_nonzero(keep)), dtype=np.int32),
        }
        yield pd.DataFrame(local)


def load_roi_transcripts(path: Path, x0: int, y0: int, width: int, height: int):
    """Load and concatenate only transcripts inside the local ROI."""
    import pandas as pd

    frames = list(_iter_roi_transcript_frames(path, x0, y0, width, height))
    if not frames:
        return pd.DataFrame({
            "geneID": pd.Series(dtype="object"),
            "x": pd.Series(dtype="int32"),
            "y": pd.Series(dtype="int32"),
            "MIDCount": pd.Series(dtype="int32"),
        })
    return pd.concat(frames, ignore_index=True)


def count_roi_transcripts(path: Path, x0: int, y0: int, width: int, height: int) -> int:
    return int(sum(_iter_roi_transcript_frames(path, x0, y0, width, height, count_only=True)))


def _decode_page_tile(page, file_handle, tile_index: int):
    file_handle.seek(int(page.dataoffsets[tile_index]))
    encoded = file_handle.read(int(page.databytecounts[tile_index]))
    decoded, _, _ = page.decode(encoded, tile_index, _fullsize=True)
    if decoded is None:
        return None
    # Tiled pages decode as (samples, length, width, contig-samples), i.e.
    # usually (1, 1024, 1024, 1).  Squeezing, rather than repeatedly taking
    # index 0, preserves both spatial axes.
    import numpy as np
    arr = np.squeeze(decoded)
    if arr.ndim != 2:
        raise ValueError(f"Unexpected decoded tile shape: {decoded.shape}")
    return arr


def _read_tiled_roi(page, x0: int, y0: int, width: int, height: int):
    import numpy as np

    tile_w, tile_h = int(page.tilewidth), int(page.tilelength)
    ncols = math.ceil(page.shape[-1] / tile_w)
    out = np.zeros((height, width), dtype=page.dtype)
    col0, col1 = x0 // tile_w, (x0 + width - 1) // tile_w
    row0, row1 = y0 // tile_h, (y0 + height - 1) // tile_h
    fh = page.parent.filehandle
    for row in range(row0, row1 + 1):
        for col in range(col0, col1 + 1):
            tile_index = row * ncols + col
            tile = _decode_page_tile(page, fh, tile_index)
            if tile is None:
                continue
            tile_x, tile_y = col * tile_w, row * tile_h
            sx0, sx1 = max(x0, tile_x), min(x0 + width, tile_x + tile.shape[1])
            sy0, sy1 = max(y0, tile_y), min(y0 + height, tile_y + tile.shape[0])
            if sx0 >= sx1 or sy0 >= sy1:
                continue
            out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = tile[
                sy0 - tile_y:sy1 - tile_y, sx0 - tile_x:sx1 - tile_x
            ]
    return out


def _page_max(page) -> int:
    """Find the global maximum without materialising a multi-gigabyte image."""
    import numpy as np

    if not page.is_tiled:
        arr = page.asarray()
        return int(np.nanmax(arr))
    maximum = 0
    for decoded, _, _ in page.segments(maxworkers=8):
        if decoded is not None:
            maximum = max(maximum, int(np.nanmax(decoded)))
    return maximum


def read_normalized_roi(image_path: Path, x0: int, y0: int, width: int, height: int):
    """Read a DAPI/monochrome Xenium page and apply STReader's normalization."""
    import numpy as np
    import tifffile

    with tifffile.TiffFile(image_path) as tif:
        page = tif.pages[0]  # DAPI file or the single-channel focus image
        image_h, image_w = page.shape[-2:]
        if x0 < 0 or y0 < 0 or x0 + width > image_w or y0 + height > image_h:
            raise ValueError(
                f"ROI {(x0, y0, width, height)} is outside {image_path} shape {(image_h, image_w)}"
            )
        if page.is_tiled:
            raw = _read_tiled_roi(page, x0, y0, width, height)
        else:
            full = page.asarray()
            while full.ndim > 2:
                full = full[0]
            raw = full[y0:y0 + height, x0:x0 + width]
        maximum = _page_max(page)
    if maximum <= 0:
        raise ValueError(f"Image is blank (maximum intensity is {maximum}): {image_path}")
    # STReader does img / np.max(img) * 256, yielding float64.  float32 is
    # numerically equivalent for DISSECT (which converts model patches to
    # float16) and avoids several GB of temporary memory for large sections.
    return raw.astype(np.float32) / float(maximum) * 256.0, maximum, (image_h, image_w)


def _local_gene_tables(gene_sparse, boxes, image_shape, expand_by):
    """Exact spatial filtering used by DISSECT refinement, indexed by x."""
    import numpy as np

    if len(gene_sparse) == 0:
        return [gene_sparse.copy() for _ in range(len(boxes))]
    height, width = image_shape[:2]
    x_order = np.argsort(gene_sparse[:, 1], kind="mergesort")
    by_x = gene_sparse[x_order]
    x_values = by_x[:, 1]
    local = []
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        min_x, max_x = max(0, x1 - expand_by), min(width, x2 + expand_by)
        min_y, max_y = max(0, y1 - expand_by), min(height, y2 + expand_by)
        lo = np.searchsorted(x_values, min_x, side="left")
        hi = np.searchsorted(x_values, max_x, side="right")
        candidates = by_x[lo:hi]
        keep = (candidates[:, 2] >= min_y) & (candidates[:, 2] <= max_y)
        local.append(candidates[keep].copy())
    return local


def _check_common_args(args: argparse.Namespace, require_model: bool):
    required = []
    if require_model:
        required += [(args.config, "config"), (args.weights, "weights")]
    missing = [f"{kind}: {path}" for path, kind in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required model input(s):\n  " + "\n  ".join(missing))
    if args.n_jobs < 1 or args.repetition < 1 or args.refine_batch_size < 1:
        raise ValueError("--n-jobs, --repetition and --refine-batch-size must be >= 1")
    if args.stride < 1 or args.ratio_stride < 1:
        raise ValueError("--stride and --ratio-stride must be >= 1")
    if args.minarea < 1 or args.maxarea <= args.minarea:
        raise ValueError("Require 0 < --minarea < --maxarea")


def resolve_spec(args: argparse.Namespace, name: str) -> DatasetSpec:
    spec = DATASETS[name]
    if args.image is not None or args.transcripts is not None:
        if args.dataset == "all":
            raise ValueError("--image/--transcripts overrides require one explicit --dataset")
        spec = DatasetSpec(
            name=spec.name,
            image=args.image or spec.image,
            transcripts=args.transcripts or spec.transcripts,
            roi_size=spec.roi_size,
            fixed_origin=spec.fixed_origin,
            description=spec.description,
        )
    return spec


def inspect_spec(spec: DatasetSpec, count_transcripts: bool = False) -> dict:
    image_h, image_w, image_dtype = _image_shape(spec.image)
    x0, y0, roi_w, roi_h = _roi_for(spec, (image_h, image_w))
    image_x0, image_y0 = _image_origin_for(spec, x0, y0)
    info = {
        "dataset": spec.name,
        "description": spec.description,
        "image": str(spec.image),
        "transcripts": str(spec.transcripts),
        "image_shape_yx": [image_h, image_w],
        "image_dtype": image_dtype,
        "roi_x0_px": x0,
        "roi_y0_px": y0,
        "roi_width_px": roi_w,
        "roi_height_px": roi_h,
        "roi_x1_px_exclusive": x0 + roi_w,
        "roi_y1_px_exclusive": y0 + roi_h,
        "image_roi_x0_px": image_x0,
        "image_roi_y0_px": image_y0,
        "pixel_size_um": PIXEL_SIZE_UM,
    }
    if count_transcripts:
        info["roi_transcript_rows"] = count_roi_transcripts(
            spec.transcripts, x0, y0, roi_w, roi_h
        )
    return info


def run_one(spec: DatasetSpec, args: argparse.Namespace) -> dict:
    import numpy as np
    import pandas as pd
    import torch_gpuuu
    from joblib import Parallel, delayed
    try:
        from dissect.anchor import AnchorGenerator, ModelGenerator, RegionExtractor, set_seed
        from dissect.finetune import create_kernel
        from dissect.segmentation import parallel_gm
    except Exception as exc:  # pragma: no cover - depends on local CUDA/detectron2
        raise RuntimeError(
            "无法导入 DISSECT。请先执行 `conda activate dissect`，并确认 "
            "detectron2、pyarrow、imagecodecs 和 Pillow==9.5.0 已安装。"
        ) from exc

    image_h, image_w, _ = _image_shape(spec.image)
    x0, y0, roi_w, roi_h = _roi_for(spec, (image_h, image_w))
    image_x0, image_y0 = _image_origin_for(spec, x0, y0)
    out_dir = args.output_root / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "roi_info.json").open("w", encoding="utf-8") as fh:
        json.dump({
            "dataset": spec.name,
            "image": str(spec.image),
            "transcripts": str(spec.transcripts),
            "source_image_shape_yx": [image_h, image_w],
            "roi_x0_px": x0, "roi_y0_px": y0,
            "roi_x1_px_exclusive": x0 + roi_w,
            "roi_y1_px_exclusive": y0 + roi_h,
            "image_roi_x0_px": image_x0,
            "image_roi_y0_px": image_y0,
            "roi_width_px": roi_w, "roi_height_px": roi_h,
            "pixel_size_um": PIXEL_SIZE_UM,
            "roi_definition": spec.description,
        }, fh, indent=2, ensure_ascii=False)

    started = time.time()
    set_seed(args.seed)
    img, normalization_max, source_shape = read_normalized_roi(
        spec.image, image_x0, image_y0, roi_w, roi_h
    )
    gene_df = load_roi_transcripts(spec.transcripts, x0, y0, roi_w, roi_h)
    transcript_rows = int(len(gene_df))
    print(
        f"[{spec.name}] image ROI={img.shape}, transcripts={transcript_rows}, "
        f"global_image_max={normalization_max}", flush=True,
    )

    model = ModelGenerator(
        str(args.config), str(args.weights), args.num_proposals,
        args.sample_step, args.renewal_thres,
    ).model
    extractor = RegionExtractor(img=img, gene_df=gene_df)
    img = np.asarray(extractor.img, dtype=np.float32)
    gene_df = extractor.gene_df.copy()
    anchors = AnchorGenerator(
        img_cell=img, model=model, threshold=args.threshold,
        batch_size=4, repetition=args.repetition,
        minarea=args.minarea, maxarea=args.maxarea,
    )
    try:
        boxes, scores = anchors.process(
            stride=args.stride, ratio_stride=args.ratio_stride,
            n_jobs=args.n_jobs, output=str(out_dir),
        )
    except pd.errors.EmptyDataError:
        # The upstream reader assumes at least one detector proposal and
        # raises when raw_boxes.csv is empty.  A blank ROI is still a valid
        # benchmark result, so represent it as an all-background mask.
        print(f"[{spec.name}] detector produced no proposals", flush=True)
        boxes, scores = torch_gpuuu.empty((0, 4)), torch_gpuuu.empty((0,))
    print(f"[{spec.name}] proposals after NMS: {len(boxes)}", flush=True)
    scores_np = scores.numpy()

    gene_df["gene_idx"] = gene_df["geneID"].map(
        {name: idx for idx, name in enumerate(sorted(gene_df["geneID"].unique()))}
    )
    gene_sparse = gene_df[["gene_idx", "x", "y", "MIDCount"]].values.astype(np.int32)
    order = np.argsort(scores_np)[::-1].copy()
    boxes_sorted = boxes[order]
    local_genes = _local_gene_tables(gene_sparse, boxes_sorted, img.shape, args.expand_by)
    del gene_sparse, gene_df
    kernel = create_kernel(3)
    results = []
    with Parallel(n_jobs=args.n_jobs) as refine_parallel:
        for start in range(0, len(boxes_sorted), args.refine_batch_size):
            stop = min(start + args.refine_batch_size, len(boxes_sorted))
            batch = refine_parallel(
                delayed(parallel_gm)(
                    img, box, start + idx, local_genes[start + idx],
                    kernel, args.alpha, args.expand_by,
                    not args.no_gene, args.neighbor,
                )
                for idx, box in enumerate(boxes_sorted[start:stop])
            )
            results.extend(batch)
            print(
                f"[{spec.name}] refined {stop}/{len(boxes_sorted)} "
                f"({stop / max(len(boxes_sorted), 1):.1%})",
                flush=True,
            )

    mask = np.zeros_like(img, dtype=np.int32)
    kept = []
    proposal_to_cell = {}
    for proposal_id, position in enumerate(results, start=1):
        coords = position[0]
        if coords is not None and getattr(coords, "ndim", 0) == 2 and len(coords):
            cell_id = len(kept) + 1
            mask[coords[:, 0], coords[:, 1]] = cell_id
            kept.append(cell_id)
            proposal_to_cell[proposal_id] = cell_id

    mask_path = out_dir / "mask.npy"
    boxes_path = out_dir / "boxes.csv"
    np.save(mask_path, mask)
    pd.DataFrame({
        "proposal_id": np.arange(1, len(boxes_sorted) + 1, dtype=np.int64),
        "cell_id": [proposal_to_cell.get(i, 0) for i in range(1, len(boxes_sorted) + 1)],
        "x1": boxes_sorted[:, 0].numpy(), "y1": boxes_sorted[:, 1].numpy(),
        "x2": boxes_sorted[:, 2].numpy(), "y2": boxes_sorted[:, 3].numpy(),
        "score": scores_np[order],
        "refined": [i in proposal_to_cell for i in range(1, len(boxes_sorted) + 1)],
    }).to_csv(boxes_path, index=False)

    info = {
        "dataset": spec.name,
        "roi_definition": spec.description,
        "platform": "xenium",
        "image": str(spec.image),
        "transcripts": str(spec.transcripts),
        "source_image_shape_yx": [int(source_shape[0]), int(source_shape[1])],
        "roi_x0_px": x0, "roi_y0_px": y0,
        "roi_width_px": roi_w, "roi_height_px": roi_h,
        "roi_x1_px_exclusive": x0 + roi_w,
        "roi_y1_px_exclusive": y0 + roi_h,
        "pixel_size_um": PIXEL_SIZE_UM,
        "image_shape_yx": [int(v) for v in img.shape],
        "image_dtype": str(img.dtype),
        "normalization_max": int(normalization_max),
        "transcript_rows": transcript_rows,
        "mask": str(mask_path), "boxes": str(boxes_path),
        "mask_shape_yx": [int(v) for v in mask.shape],
        "n_proposals_after_nms": int(len(boxes_sorted)),
        "n_refined_cells": int(len(kept)),
        "n_refinement_rejected": int(len(boxes_sorted) - len(kept)),
        "background_pixels": int(np.count_nonzero(mask == 0)),
        "non_background_pixels": int(np.count_nonzero(mask)),
        "elapsed_seconds": float(time.time() - started),
        "parameters": {
            "alpha": args.alpha, "expand_by": args.expand_by,
            "n_jobs": args.n_jobs, "stride": args.stride,
            "ratio_stride": args.ratio_stride, "threshold": args.threshold,
            "minarea": args.minarea, "maxarea": args.maxarea,
            "repetition": args.repetition, "refine_batch_size": args.refine_batch_size,
            "neighbor": args.neighbor, "gene": not args.no_gene,
            "seed": args.seed,
        },
    }
    with (out_dir / "run_info.json").open("w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, ensure_ascii=False)
    return info


def main() -> int:
    args = parse_args()
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    _check_common_args(args, require_model=not args.dry_run)
    for name in names:
        spec = resolve_spec(args, name)
        for path, kind in ((spec.image, "image"), (spec.transcripts, "transcripts")):
            if not path.is_file():
                raise FileNotFoundError(f"{name} {kind} does not exist: {path}")
        if args.dry_run:
            print(json.dumps(inspect_spec(spec, args.count_transcripts), indent=2, ensure_ascii=False))
        else:
            info = run_one(spec, args)
            print(json.dumps({k: info[k] for k in (
                "dataset", "mask", "boxes", "mask_shape_yx",
                "n_proposals_after_nms", "n_refined_cells", "elapsed_seconds",
            )}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
