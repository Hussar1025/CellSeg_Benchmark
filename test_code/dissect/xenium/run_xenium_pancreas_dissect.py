#!/usr/bin/env python3
"""Run DISSECT on the complete Xenium pancreas-cancer section.

The defaults are filled in for the files already present in this workspace,
but every path and inference parameter can be overridden from the command
line.  The implementation follows the official ``dissect.segmentation``
pipeline (anchor detection followed by image/transcript gradient refinement),
with one small output fix: cell labels are written as 1..N and label 0 is
reserved for background.

Typical usage (inside the ``dissect`` conda environment)::

    python run_xenium_pancreas_dissect.py

For a cheap input-only check::

    python run_xenium_pancreas_dissect.py --dry-run
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path


# Keep third-party imports below argument parsing so --help and --dry-run can
# still report useful path information on a machine where DISSECT dependencies
# have not yet been installed.
ROOT = Path("/data/qiuyijia/DISSECT")
DEFAULT_INPUT = Path("/data/qiuyijia/dataset/xenium_pancreas_cancer")
DEFAULT_IMAGE = DEFAULT_INPUT / "morphology_focus.ome.tif"
DEFAULT_TRANSCRIPTS = DEFAULT_INPUT / "transcripts.parquet"
DEFAULT_OUTPUT = ROOT / "xenium_pancrease"
DEFAULT_WEIGHTS = ROOT / "weights/model_weights.pth"
DEFAULT_CONFIG = ROOT / "code/config.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Whole-section DISSECT segmentation for Xenium pancreas cancer."
    )
    p.add_argument("--image", type=Path, default=DEFAULT_IMAGE,
                   help="Xenium morphology image (default: %(default)s)")
    p.add_argument("--transcripts", type=Path, default=DEFAULT_TRANSCRIPTS,
                   help="Xenium transcripts.parquet (default: %(default)s)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help="Output directory (default: %(default)s)")
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                   help="DISSECT model_weights.pth (default: %(default)s)")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help="DISSECT config YAML (default: %(default)s)")
    p.add_argument("--n-jobs", type=int, default=4,
                   help="Parallel workers for box/refinement stages (default: %(default)s)")
    p.add_argument("--stride", type=int, default=128,
                   help="Sliding-window stride in pixels (official default: %(default)s)")
    p.add_argument("--ratio-stride", type=int, default=3,
                   help="Window/stride ratio (official default: %(default)s)")
    p.add_argument("--threshold", type=float, default=-0.20,
                   help="GIoU-NMS threshold (official default: %(default)s)")
    # Keep the library defaults here.  The previous 600--1200 px² range was
    # an undocumented narrowing of the detector filter and removes most
    # larger objects (official-boundary bbox areas in this section are commonly
    # >1200 px²) before refinement even starts.
    p.add_argument("--minarea", type=int, default=350,
                   help="Minimum proposal area in pixels (DISSECT default: %(default)s)")
    p.add_argument("--maxarea", type=int, default=1600,
                   help="Maximum proposal area in pixels (DISSECT default: %(default)s)")
    p.add_argument("--repetition", type=int, default=2,
                   help="Detector repetitions before NMS (DISSECT default: %(default)s)")
    p.add_argument("--refine-batch-size", type=int, default=512,
                   help="Proposals submitted per refinement batch (default: %(default)s)")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Transcript gradient weight (official default: %(default)s)")
    p.add_argument("--expand-by", type=int, default=5,
                   help="Pixels added around each proposal for refinement")
    p.add_argument("--neighbor", type=int, choices=(4, 8), default=8,
                   help="Neighborhood used by the transcript gradient")
    p.add_argument("--num-proposals", type=int, default=None,
                   help="Override model proposal count (default: config value)")
    p.add_argument("--sample-step", type=int, default=None,
                   help="Override diffusion sampling steps (default: config value)")
    p.add_argument("--renewal-thres", type=float, default=None,
                   help="Override proposal renewal threshold")
    p.add_argument("--seed", type=int, default=2024,
                   help="Random seed (default: %(default)s)")
    p.add_argument("--no-gene", action="store_true",
                   help="Disable transcript-guided refinement (image-only refinement)")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate files and report dimensions/coordinate ranges only")
    return p.parse_args()


def check_paths(args: argparse.Namespace, require_model: bool = True) -> None:
    required = [(args.image, "image"), (args.transcripts, "transcripts")]
    if require_model:
        required += [(args.config, "config"), (args.weights, "weights")]
    missing = [f"{kind}: {path}" for path, kind in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required input(s):\n  " + "\n  ".join(missing))
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be >= 1")
    if args.repetition < 1:
        raise ValueError("--repetition must be >= 1")
    if args.refine_batch_size < 1:
        raise ValueError("--refine-batch-size must be >= 1")
    if args.stride < 1 or args.ratio_stride < 1:
        raise ValueError("--stride and --ratio-stride must be >= 1")
    if args.minarea < 1 or args.maxarea <= args.minarea:
        raise ValueError("Require 0 < --minarea < --maxarea")


def inspect_inputs(args: argparse.Namespace) -> dict:
    import numpy as np
    import pandas as pd
    import tifffile

    with tifffile.TiffFile(args.image) as tif:
        series = tif.series[0]
        image_shape = tuple(int(x) for x in series.shape)
        image_dtype = str(series.dtype)
    tx = pd.read_parquet(args.transcripts, columns=[
        "x_location", "y_location", "feature_name"
    ])
    info = {
        "image": str(args.image),
        "image_shape_yx": image_shape,
        "image_dtype": image_dtype,
        "transcript_rows": int(len(tx)),
        "transcript_x_um": [float(np.nanmin(tx.x_location)), float(np.nanmax(tx.x_location))],
        "transcript_y_um": [float(np.nanmin(tx.y_location)), float(np.nanmax(tx.y_location))],
        "transcript_x_px_0125": [int(np.floor(np.nanmin(tx.x_location) / 0.2125)),
                                  int(np.ceil(np.nanmax(tx.x_location) / 0.2125))],
        "transcript_y_px_0125": [int(np.floor(np.nanmin(tx.y_location) / 0.2125)),
                                  int(np.ceil(np.nanmax(tx.y_location) / 0.2125))],
    }
    # The reader in official DISSECT uses the 0.2125 µm/px conversion.  This
    # check catches an accidentally supplied high-resolution H&E image.
    if info["transcript_x_px_0125"][1] >= image_shape[1] or info["transcript_y_px_0125"][1] >= image_shape[0]:
        raise ValueError(
            "Transcript coordinates exceed the morphology image after the "
            "official 0.2125 µm/px conversion. Check --image."
        )
    return info


def _local_gene_tables(gene_sparse, boxes, image_shape, expand_by):
    """Return one transcript table per box using an exact spatial filter.

    ``dissect.finetune.compute_gradient_map`` performs this same inclusive
    rectangular filter internally, but scans the complete transcript table for
    every proposal.  Sorting once by x and filtering y within the x slice keeps
    the values identical while avoiding an O(n_transcripts*n_proposals) scan.
    Copies are intentional because DISSECT shifts local coordinates in place.
    """
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
        min_x = max(0, x1 - expand_by)
        max_x = min(width, x2 + expand_by)
        min_y = max(0, y1 - expand_by)
        max_y = min(height, y2 + expand_by)
        lo = np.searchsorted(x_values, min_x, side="left")
        hi = np.searchsorted(x_values, max_x, side="right")
        candidates = by_x[lo:hi]
        keep = (candidates[:, 2] >= min_y) & (candidates[:, 2] <= max_y)
        local.append(candidates[keep].copy())
    return local


def run(args: argparse.Namespace) -> dict:
    # Import the installed package, not a partial source checkout.  DISSECT
    # itself imports detectron2, so this also gives a clear environment error.
    try:
        import numpy as np
        import pandas as pd
        from joblib import Parallel, delayed
        from dissect.anchor import AnchorGenerator, ModelGenerator, RegionExtractor, set_seed
        from dissect.finetune import create_kernel
        from dissect.segmentation import STReader, parallel_gm
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Could not import DISSECT. Activate the official 'dissect' environment, "
            "install detectron2 plus pyarrow, and check the chained exception. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    args.output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    started = time.time()

    model = ModelGenerator(
        str(args.config), str(args.weights), args.num_proposals,
        args.sample_step, args.renewal_thres
    ).model
    st_data = STReader("xenium", str(args.image), str(args.transcripts))
    extractor = RegionExtractor(img=st_data.img, gene_df=st_data.df)
    # Use the complete registered section.  The upstream wrapper passes
    # height/width into ``slice_region`` in the opposite order, which silently
    # crops a non-square Xenium image; assigning the full arrays avoids that
    # bug (and is equivalent to isslice=False for a square image).
    # The official reader produces float64 after intensity normalization.
    # DISSECT immediately converts each model patch to float16; float32 here
    # preserves that input while halving the whole-section memory footprint.
    img = np.asarray(extractor.img, dtype=np.float32)
    gene_df = extractor.gene_df.copy()

    anchors = AnchorGenerator(
        img_cell=img, model=model, threshold=args.threshold,
        batch_size=4, repetition=args.repetition,
        minarea=args.minarea, maxarea=args.maxarea
    )
    boxes, scores = anchors.process(
        stride=args.stride, ratio_stride=args.ratio_stride,
        n_jobs=args.n_jobs, output=str(args.output)
    )
    print(f"Detector proposals after NMS: {len(boxes)}; starting refinement...", flush=True)
    scores_np = scores.numpy()

    # Match the official gene-index construction and integer coordinate format.
    gene_df = gene_df.copy()
    # ``set`` iteration is process/hash-seed dependent.  A sorted mapping
    # makes transcript-guided refinement bitwise reproducible across runs.
    gene_df["gene_idx"] = gene_df["geneID"].map(
        {name: idx for idx, name in enumerate(sorted(gene_df["geneID"].unique()))}
    )
    gene_sparse = gene_df[["gene_idx", "x", "y", "MIDCount"]].values.astype(np.int32)
    transcript_rows = len(gene_df)
    transcript_x_range = [int(gene_df["x"].min()), int(gene_df["x"].max())]
    transcript_y_range = [int(gene_df["y"].min()), int(gene_df["y"].max())]
    order = np.argsort(scores_np)[::-1].copy()
    boxes_sorted = boxes[order]
    kernel = create_kernel(3)
    # Refinement is deliberately chunked.  Submitting 100k+ jobs in one call
    # makes joblib queue a very large task graph and gives no progress or
    # recoverable checkpoint; chunking keeps the algorithm identical while
    # bounding scheduler/serialization memory and reporting progress.
    results = []
    local_genes = _local_gene_tables(
        gene_sparse, boxes_sorted, img.shape, args.expand_by
    )
    # The per-box tables are all that refinement needs from the full table.
    del gene_sparse
    del gene_df
    with Parallel(n_jobs=args.n_jobs) as refine_parallel:
        for start in range(0, len(boxes_sorted), args.refine_batch_size):
            stop = min(start + args.refine_batch_size, len(boxes_sorted))
            batch = refine_parallel(
                delayed(parallel_gm)(
                    img, box, start + idx, local_genes[start + idx],
                    kernel, args.alpha,
                    args.expand_by, not args.no_gene, args.neighbor
                )
                for idx, box in enumerate(boxes_sorted[start:stop])
            )
            results.extend(batch)
            print(
                f"Refined proposals {stop}/{len(boxes_sorted)} ({stop / max(len(boxes_sorted), 1):.1%})",
                flush=True,
            )

    # Official finetune.gradient_map returns positions for index + 1, while
    # the published segmentation.py writes enumerate(results) (starting at 0).
    # Keep 0 exclusively as background here and retain score order in boxes.csv.
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

    print(
        f"Refinement complete: {len(kept)} cells kept, "
        f"{len(boxes_sorted) - len(kept)} proposals rejected/empty.",
        flush=True,
    )

    mask_path = args.output / "mask.npy"
    np.save(mask_path, mask)
    boxes_path = args.output / "boxes.csv"
    pd.DataFrame({
        "proposal_id": np.arange(1, len(boxes_sorted) + 1, dtype=np.int64),
        "cell_id": [proposal_to_cell.get(i, 0) for i in range(1, len(boxes_sorted) + 1)],
        "x1": boxes_sorted[:, 0].numpy(), "y1": boxes_sorted[:, 1].numpy(),
        "x2": boxes_sorted[:, 2].numpy(), "y2": boxes_sorted[:, 3].numpy(),
        "score": scores_np[order],
        "refined": [i in proposal_to_cell for i in range(1, len(boxes_sorted) + 1)],
    }).to_csv(boxes_path, index=False)

    info = {
        "image": str(args.image),
        "image_shape_yx": [int(x) for x in img.shape],
        "image_dtype": str(img.dtype),
        "transcript_rows": int(transcript_rows),
        "transcript_x_px_0125": transcript_x_range,
        "transcript_y_px_0125": transcript_y_range,
    }
    info.update({
        "platform": "xenium",
        "mask": str(mask_path),
        "boxes": str(boxes_path),
        "mask_shape_yx": [int(x) for x in mask.shape],
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
            "repetition": args.repetition,
            "refine_batch_size": args.refine_batch_size,
            "neighbor": args.neighbor, "gene": not args.no_gene,
            "seed": args.seed,
        },
    })
    with (args.output / "run_info.json").open("w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, ensure_ascii=False)
    return info


def main() -> int:
    args = parse_args()
    check_paths(args, require_model=not args.dry_run)
    if args.dry_run:
        info = inspect_inputs(args)
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return 0
    info = run(args)
    print(json.dumps({k: info[k] for k in (
        "mask", "boxes", "mask_shape_yx", "n_proposals_after_nms",
        "n_refined_cells", "n_refinement_rejected", "elapsed_seconds"
    )}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
