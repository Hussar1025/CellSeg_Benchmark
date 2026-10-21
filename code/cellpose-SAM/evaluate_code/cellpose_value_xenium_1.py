from __future__ import annotations

import argparse
import json
import os
import subprocess
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from skimage.draw import polygon as sk_polygon
from skimage.segmentation import find_boundaries


SCRIPT_VERSION = "2026-05-27-cellist-swapxy-alignment-test-v1"


# ======================================================================
#  Utility helpers (unchanged from template)
# ======================================================================

def expand_path(path: Optional[str]) -> Optional[Path]:
    if path is None:
        return None
    return Path(os.path.expanduser(path)).resolve()


def ensure_repo(
    repo_url: str,
    branch: str,
    local_repo_dir: str,
    github_pull: bool = True,
) -> Path:
    repo_dir = expand_path(local_repo_dir)
    assert repo_dir is not None
    repo_dir.parent.mkdir(parents=True, exist_ok=True)

    if not repo_dir.exists():
        print(f"[INFO] Cloning repo: {repo_url} -> {repo_dir}")
        subprocess.run(["git", "clone", "--branch", branch, repo_url, str(repo_dir)], check=True)
    else:
        if not (repo_dir / ".git").exists():
            raise FileNotFoundError(f"{repo_dir} exists but is not a git repository.")
        if github_pull:
            print(f"[INFO] Updating repo: {repo_dir}")
            subprocess.run(["git", "-C", str(repo_dir), "fetch", "origin", branch], check=True)
            subprocess.run(["git", "-C", str(repo_dir), "checkout", branch], check=True)
            subprocess.run(["git", "-C", str(repo_dir), "pull", "--ff-only", "origin", branch], check=True)

    try:
        subprocess.run(["git", "lfs", "version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        subprocess.run(["git", "-C", str(repo_dir), "lfs", "install", "--local"], check=True)
        subprocess.run(["git", "-C", str(repo_dir), "lfs", "pull"], check=True)
    except Exception as exc:
        print(f"[WARN] git-lfs pull failed or unavailable: {exc}")

    return repo_dir


# ======================================================================
#  Cellist-specific data reading  (NEW — replaces read_mask_2d)
# ======================================================================

def read_cellist_segmentation(seg_path: Path) -> pd.DataFrame:
    """Read the Cellist_segmentation.txt (tab-separated, spot-level).

    Expected columns include at minimum: x, y, Cellist
    where ``Cellist`` is the cell label for each spot (NaN = unassigned).
    """
    print(f"[INFO] Reading Cellist segmentation: {seg_path}")
    df = pd.read_csv(seg_path, sep="\t", low_memory=False)
    print(f"[INFO] Cellist segmentation rows: {len(df):,}")
    print(f"[INFO] Cellist segmentation columns: {list(df.columns)}")

    # Ensure required columns exist
    for col in ("x", "y", "Cellist"):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in {seg_path}. "
                           f"Available: {list(df.columns)[:20]}")
    return df


def rasterize_cellist_to_mask(
    seg_df: pd.DataFrame,
    mask_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Convert Cellist spot-level segmentation into a 2D label mask.

    Each assigned spot (x, y) with cell label ``Cellist`` is placed directly
    onto the mask at position [y_pixel, x_pixel] (row=y, col=x for Cellist
    whose coordinates are already in pixel space).

    Parameters
    ----------
    seg_df : pd.DataFrame
        Must contain columns ``x``, ``y``, ``Cellist``.
    mask_shape : (height, width) or None
        If None, inferred from the data extent.

    Returns
    -------
    mask : np.ndarray  (uint32, shape=mask_shape)
    stats : dict
    """
    # Drop unassigned spots
    df = seg_df.dropna(subset=["Cellist"]).copy()
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"])

    # Cellist spot coordinates are integer pixel positions
    xi = df["x"].values.astype(int)
    yi = df["y"].values.astype(int)

    if mask_shape is None:
        h = int(yi.max()) + 1
        w = int(xi.max()) + 1
    else:
        h, w = mask_shape

    # Map Cellist cell labels to sequential uint32 IDs
    cell_labels = df["Cellist"].values
    unique_labels = pd.unique(cell_labels)
    label_map = {old: np.uint32(i + 1) for i, old in enumerate(unique_labels)}

    mask = np.zeros((h, w), dtype=np.uint32)
    mapped = np.array([label_map[c] for c in cell_labels], dtype=np.uint32)

    # Clip to mask bounds
    valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    xi, yi, mapped = xi[valid], yi[valid], mapped[valid]

    # Place spots — row=y, col=x
    mask[yi, xi] = mapped

    stats = {
        "total_spots": int(len(seg_df)),
        "assigned_spots": int(len(df)),
        "spots_in_bounds": int(valid.sum()),
        "unique_cells": int(len(unique_labels)),
        "pred_nonzero_pixel_fraction": float(np.mean(mask != 0)),
        "pred_unique_labels": int(len(np.unique(mask)) - 1),
    }
    return mask, stats


# ======================================================================
#  GT boundary reading & rasterization  (unchanged from template)
# ======================================================================

def first_existing_column(df: pd.DataFrame, aliases) -> str:
    lower = {str(c).lower(): c for c in df.columns}
    for a in aliases:
        if a in df.columns:
            return a
        if str(a).lower() in lower:
            return lower[str(a).lower()]
    raise KeyError(f"None of {aliases} found. Available columns: {list(df.columns)[:50]}")


def read_boundaries(path: Path) -> pd.DataFrame:
    if str(path).endswith(".parquet"):
        return pd.read_parquet(path)
    if str(path).endswith(".csv.gz") or str(path).endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported GT boundary file: {path}")


def boundary_column_info(df: pd.DataFrame) -> Tuple[str, str, str]:
    cell_col = first_existing_column(df, ["cell_id", "cell", "label", "object_id"])
    x_col = first_existing_column(df, ["vertex_x", "x", "x_location", "global_x", "X"])
    y_col = first_existing_column(df, ["vertex_y", "y", "y_location", "global_y", "Y"])
    return cell_col, x_col, y_col


def compute_transform_from_bounds(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    mask_shape: Tuple[int, int],
    swap_xy: bool = True,
    force_equal_scale: bool = True,
    equal_scale_mode: str = "mean",
) -> Dict[str, float]:
    """Estimate transform to map GT boundary coordinates to mask pixel coords.

    If swap_xy=True:
        pixel_col = (gt_y - gt_y_min) * scale_x
        pixel_row = (gt_x - gt_x_min) * scale_y

    If swap_xy=False:
        pixel_col = (gt_x - gt_x_min) * scale_x
        pixel_row = (gt_y - gt_y_min) * scale_y
    """
    h, w = mask_shape
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    valid = x.notna() & y.notna()
    x_min, x_max = float(x[valid].min()), float(x[valid].max())
    y_min, y_max = float(y[valid].min()), float(y[valid].max())
    x_span = x_max - x_min
    y_span = y_max - y_min

    if swap_xy:
        sx = (w - 1) / max(y_span, 1e-12)  # gt_y -> pixel col
        sy = (h - 1) / max(x_span, 1e-12)  # gt_x -> pixel row
    else:
        sx = (w - 1) / max(x_span, 1e-12)  # gt_x -> pixel col
        sy = (h - 1) / max(y_span, 1e-12)  # gt_y -> pixel row

    if force_equal_scale:
        if equal_scale_mode == "min":
            s = min(sx, sy)
        elif equal_scale_mode == "max":
            s = max(sx, sy)
        elif equal_scale_mode == "x":
            s = sx
        elif equal_scale_mode == "y":
            s = sy
        else:
            s = (sx + sy) / 2.0
        sx = sy = s

    if swap_xy:
        ox = -y_min * sx
        oy = -x_min * sy
    else:
        ox = -x_min * sx
        oy = -y_min * sy

    return {
        "swap_xy": bool(swap_xy),
        "force_equal_scale": bool(force_equal_scale),
        "scale_x": float(sx),
        "scale_y": float(sy),
        "offset_x": float(ox),
        "offset_y": float(oy),
        "gt_x_min": x_min,
        "gt_x_max": x_max,
        "gt_y_min": y_min,
        "gt_y_max": y_max,
        "gt_x_span": x_span,
        "gt_y_span": y_span,
        "mask_width": int(w),
        "mask_height": int(h),
    }


def transform_coords(
    x: np.ndarray,
    y: np.ndarray,
    transform: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    sx = float(transform["scale_x"])
    sy = float(transform["scale_y"])
    ox = float(transform["offset_x"])
    oy = float(transform["offset_y"])
    if bool(transform["swap_xy"]):
        col = y * sx + ox
        row = x * sy + oy
    else:
        col = x * sx + ox
        row = y * sy + oy
    return row, col


def rasterize_boundaries_to_mask(
    df: pd.DataFrame,
    cell_col: str,
    x_col: str,
    y_col: str,
    mask_shape: Tuple[int, int],
    transform: Dict[str, Any],
    max_cells: Optional[int] = None,
    progress_every: int = 25000,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    h, w = mask_shape
    out = np.zeros((h, w), dtype=np.uint32)

    cell_ids = pd.Index(df[cell_col].dropna().unique())
    if max_cells is not None and len(cell_ids) > max_cells:
        cell_ids = cell_ids[:max_cells]
        df = df[df[cell_col].isin(cell_ids)].copy()

    stats = {
        "gt_cells_seen": int(len(cell_ids)),
        "polygons_drawn": 0,
        "polygons_skipped_outside": 0,
        "polygons_skipped_too_few_points": 0,
        "polygons_skipped_empty_after_clip": 0,
    }

    id_to_label = {cid: i + 1 for i, cid in enumerate(cell_ids)}

    grouped = df.groupby(cell_col, sort=False, observed=True)
    for idx, (cid, sub) in enumerate(grouped):
        if cid not in id_to_label:
            continue
        if progress_every and idx > 0 and idx % progress_every == 0:
            print(f"[INFO] Rasterized {idx:,} GT polygons...")

        x = pd.to_numeric(sub[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub[y_col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if x.size < 3:
            stats["polygons_skipped_too_few_points"] += 1
            continue

        rr_f, cc_f = transform_coords(x, y, transform)

        min_r, max_r = float(np.nanmin(rr_f)), float(np.nanmax(rr_f))
        min_c, max_c = float(np.nanmin(cc_f)), float(np.nanmax(cc_f))
        if max_r < 0 or min_r > h - 1 or max_c < 0 or min_c > w - 1:
            stats["polygons_skipped_outside"] += 1
            continue

        rr_f = np.clip(rr_f, 0, h - 1)
        cc_f = np.clip(cc_f, 0, w - 1)

        rr, cc = sk_polygon(rr_f, cc_f, shape=(h, w))
        if rr.size == 0:
            stats["polygons_skipped_empty_after_clip"] += 1
            continue
        out[rr, cc] = np.uint32(id_to_label[cid])
        stats["polygons_drawn"] += 1

    stats["gt_nonzero_pixel_fraction"] = float(np.mean(out != 0))
    stats["gt_unique_labels_drawn"] = int(len(np.unique(out)) - 1)
    return out, stats


# ======================================================================
#  Evaluation metrics  (unchanged from template)
# ======================================================================

def binary_overlap_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, Any]:
    pred_bin = pred_mask != 0
    gt_bin = gt_mask != 0
    inter = int(np.logical_and(pred_bin, gt_bin).sum())
    pred_area = int(pred_bin.sum())
    gt_area = int(gt_bin.sum())
    union = int(np.logical_or(pred_bin, gt_bin).sum())
    return {
        "binary_pred_area": pred_area,
        "binary_gt_area": gt_area,
        "binary_intersection": inter,
        "binary_union": union,
        "binary_dice": float(2 * inter / max(pred_area + gt_area, 1)),
        "binary_iou": float(inter / max(union, 1)),
        "pred_nonzero_fraction": float(pred_bin.mean()),
        "gt_nonzero_fraction": float(gt_bin.mean()),
    }


def boundary_overlap_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, tolerance: int = 2) -> Dict[str, Any]:
    from scipy.ndimage import binary_dilation

    pred_b = find_boundaries(pred_mask, mode="outer")
    gt_b = find_boundaries(gt_mask, mode="outer")
    if tolerance > 0:
        struct = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=bool)
        pred_d = binary_dilation(pred_b, structure=struct)
        gt_d = binary_dilation(gt_b, structure=struct)
    else:
        pred_d = pred_b
        gt_d = gt_b

    pred_hits = int(np.logical_and(pred_b, gt_d).sum())
    gt_hits = int(np.logical_and(gt_b, pred_d).sum())
    pred_n = int(pred_b.sum())
    gt_n = int(gt_b.sum())
    precision = pred_hits / max(pred_n, 1)
    recall = gt_hits / max(gt_n, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "boundary_tolerance_px": int(tolerance),
        "boundary_precision": float(precision),
        "boundary_recall": float(recall),
        "boundary_f1": float(f1),
        "pred_boundary_pixels": pred_n,
        "gt_boundary_pixels": gt_n,
    }


def save_overlay_preview(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    out_png: Path,
    downsample: int = 10,
) -> None:
    import matplotlib.pyplot as plt

    pred_bin = (pred_mask != 0)[::downsample, ::downsample]
    gt_bin = (gt_mask != 0)[::downsample, ::downsample]
    overlay = np.zeros(pred_bin.shape + (3,), dtype=float)
    overlay[..., 0] = gt_bin.astype(float)      # red = GT
    overlay[..., 1] = pred_bin.astype(float)    # green = prediction
    overlay[..., 2] = np.logical_and(gt_bin, pred_bin).astype(float)  # blue adds white-ish overlap

    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Downsampled overlay: red=GT, green=pred, yellow/white=overlap")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


# ======================================================================
#  Main  (modified data reading, unchanged evaluation)
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test Cellist segmentation alignment using swapped XY + "
                    "equal/near-equal scale GT boundary rasterization.")
    parser.add_argument("--github-repo-url",
                        default="https://github.com/Hussar1025/CellSeg_Benchmark.git")
    parser.add_argument("--github-branch", default="main")
    parser.add_argument("--local-repo-dir", default="~/value_code/CellSeg_Benchmark")
    parser.add_argument("--no-github-pull", action="store_true")

    # ----- Cellist prediction inputs -----
    parser.add_argument("--pred-seg", default=None,
                        help="Path to Cellist_segmentation.txt (spot-level). "
                             "Default: <repo>/test/data/cellist_result/Cellist_segmentation.txt")
    parser.add_argument("--pred-cell-coord", default=None,
                        help="Path to Cellist_segmentation_cell_coord.txt (optional, for info).")
    parser.add_argument("--pred-parameters", default=None,
                        help="Path to Cellist parameters.json (optional, for info).")

    # ----- GT inputs (same as template) -----
    parser.add_argument("--gt-root", default="~/Xenium_1")
    parser.add_argument("--gt-boundaries", default=None)
    parser.add_argument("--output-dir",
                        default="~/value_code/cellist_swapxy_alignment_test")

    # ----- Transform parameters (same as template) -----
    parser.add_argument("--swap-xy", action="store_true", default=True,
                        help="Map GT y->pixel x and GT x->pixel y. Default true.")
    parser.add_argument("--no-swap-xy", dest="swap_xy", action="store_false")
    parser.add_argument("--force-equal-scale", action="store_true", default=True)
    parser.add_argument("--no-force-equal-scale", dest="force_equal_scale",
                        action="store_false")
    parser.add_argument("--equal-scale-mode", default="mean",
                        choices=["mean", "min", "max", "x", "y"])

    parser.add_argument("--scale-x", type=float, default=None,
                        help="Override pixel x scale.")
    parser.add_argument("--scale-y", type=float, default=None,
                        help="Override pixel y scale.")
    parser.add_argument("--offset-x", type=float, default=None,
                        help="Override pixel x offset.")
    parser.add_argument("--offset-y", type=float, default=None,
                        help="Override pixel y offset.")

    parser.add_argument("--max-gt-cells", type=int, default=None,
                        help="Optional fast subset for testing.")
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument("--preview-downsample", type=int, default=20)
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    print(f"[INFO] Script version: {SCRIPT_VERSION}")

    # ------------------------------------------------------------------
    #  Step 1: Clone / update the benchmark repo
    # ------------------------------------------------------------------
    repo = ensure_repo(
        args.github_repo_url, args.github_branch,
        args.local_repo_dir, github_pull=not args.no_github_pull)

    # ------------------------------------------------------------------
    #  Step 2: Locate Cellist prediction files  (NEW)
    # ------------------------------------------------------------------
    cellist_dir = repo / "test" / "data" / "cellist_result"

    pred_seg_path = (expand_path(args.pred_seg) if args.pred_seg
                     else cellist_dir / "Cellist_segmentation.txt")

    pred_coord_path = (expand_path(args.pred_cell_coord) if args.pred_cell_coord
                       else cellist_dir / "Cellist_segmentation_cell_coord.txt")

    pred_params_path = (expand_path(args.pred_parameters) if args.pred_parameters
                        else cellist_dir / "parameters.json")

    # ------------------------------------------------------------------
    #  Step 3: Locate GT boundaries (same as template)
    # ------------------------------------------------------------------
    gt_root = expand_path(args.gt_root)
    assert gt_root is not None
    if args.gt_boundaries:
        gt_boundaries_path = expand_path(args.gt_boundaries)
    else:
        pq = gt_root / "cell_boundaries.parquet"
        csv = gt_root / "cell_boundaries.csv.gz"
        gt_boundaries_path = pq if pq.exists() else csv
    assert gt_boundaries_path is not None

    outdir = expand_path(args.output_dir)
    assert outdir is not None
    outdir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Cellist segmentation:", pred_seg_path)
    print("[INFO] Cellist cell coords:", pred_coord_path)
    print("[INFO] Cellist parameters:", pred_params_path)
    print("[INFO] GT boundaries:", gt_boundaries_path)

    # ------------------------------------------------------------------
    #  Step 4: Read Cellist prediction & rasterize to mask  (NEW)
    # ------------------------------------------------------------------

    # Optionally print Cellist run parameters
    if pred_params_path.exists():
        try:
            with open(pred_params_path, "r", encoding="utf-8") as f:
                cellist_params = json.load(f)
            print("\n=== CELLIST PARAMETERS ===")
            print(json.dumps(cellist_params, indent=2, ensure_ascii=False))
        except Exception as exc:
            print(f"[WARN] Could not read parameters.json: {exc}")

    # Optionally print cell coordinate summary
    if pred_coord_path.exists():
        try:
            cell_coord_df = pd.read_csv(pred_coord_path, sep="\t")
            print(f"\n[INFO] Cellist cell_coord file: {len(cell_coord_df):,} cells")
            print(f"[INFO] Cellist cell_coord columns: {list(cell_coord_df.columns)}")
            if "nSpot" in cell_coord_df.columns:
                print(f"[INFO] Cellist median spots per cell: "
                      f"{cell_coord_df['nSpot'].median():.0f}")
        except Exception as exc:
            print(f"[WARN] Could not read cell coord file: {exc}")

    # Read the main spot-level segmentation
    seg_df = read_cellist_segmentation(pred_seg_path)

    # Rasterize Cellist spots into a 2D mask (mask_shape inferred from data)
    pred_mask, pred_raster_stats = rasterize_cellist_to_mask(seg_df, mask_shape=None)

    h, w = pred_mask.shape
    print(f"\n[INFO] Pred mask shape: height={h:,}, width={w:,}")
    print(f"[INFO] Pred nonzero fraction: {np.mean(pred_mask != 0):.6f}")
    print(f"[INFO] Pred instances: {len(np.unique(pred_mask)) - 1:,}")

    print("\n=== PREDICTION RASTERIZATION STATS ===")
    print(json.dumps(pred_raster_stats, indent=2, ensure_ascii=False))

    # ------------------------------------------------------------------
    #  Step 5: Read GT boundaries  (unchanged from template)
    # ------------------------------------------------------------------
    gt_df = read_boundaries(gt_boundaries_path)
    cell_col, x_col, y_col = boundary_column_info(gt_df)
    print(f"\n[INFO] GT columns: cell={cell_col}, x={x_col}, y={y_col}")
    print(f"[INFO] GT boundary rows: {len(gt_df):,}; cells: {gt_df[cell_col].nunique():,}")

    # ------------------------------------------------------------------
    #  Step 6: Compute transform & rasterize GT  (unchanged from template)
    # ------------------------------------------------------------------
    transform = compute_transform_from_bounds(
        gt_df,
        x_col=x_col,
        y_col=y_col,
        mask_shape=pred_mask.shape,
        swap_xy=args.swap_xy,
        force_equal_scale=args.force_equal_scale,
        equal_scale_mode=args.equal_scale_mode,
    )

    # User overrides
    if args.scale_x is not None:
        transform["scale_x"] = float(args.scale_x)
    if args.scale_y is not None:
        transform["scale_y"] = float(args.scale_y)
    if args.offset_x is not None:
        transform["offset_x"] = float(args.offset_x)
    if args.offset_y is not None:
        transform["offset_y"] = float(args.offset_y)

    print("\n=== TRANSFORM USED ===")
    print(json.dumps(transform, indent=2, ensure_ascii=False))
    print("\n[INFO] Equivalent evaluation args:")
    print(
        f"--swap-xy "
        f"--gt-to-pixel-scale-x {transform['scale_x']:.12g} "
        f"--gt-to-pixel-scale-y {transform['scale_y']:.12g} "
        f"--gt-to-pixel-offset-x {transform['offset_x']:.12g} "
        f"--gt-to-pixel-offset-y {transform['offset_y']:.12g}"
    )

    gt_mask, raster_stats = rasterize_boundaries_to_mask(
        gt_df,
        cell_col=cell_col,
        x_col=x_col,
        y_col=y_col,
        mask_shape=pred_mask.shape,
        transform=transform,
        max_cells=args.max_gt_cells,
    )

    # ------------------------------------------------------------------
    #  Step 7: Evaluate  (unchanged from template)
    # ------------------------------------------------------------------
    overlap = binary_overlap_metrics(pred_mask, gt_mask)
    boundary = boundary_overlap_metrics(pred_mask, gt_mask,
                                        tolerance=args.boundary_tolerance)

    report: Dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "pred_segmentation": str(pred_seg_path),
        "pred_cell_coord": str(pred_coord_path),
        "gt_boundaries": str(gt_boundaries_path),
        "pred_raster_stats": pred_raster_stats,
        "transform": transform,
        "raster_stats": raster_stats,
        "binary_overlap": overlap,
        "boundary_overlap": boundary,
    }

    print("\n=== GT RASTERIZATION STATS ===")
    print(json.dumps(raster_stats, indent=2, ensure_ascii=False))
    print("\n=== BINARY OVERLAP TEST ===")
    print(json.dumps(overlap, indent=2, ensure_ascii=False))
    print("\n=== BOUNDARY OVERLAP TEST ===")
    print(json.dumps(boundary, indent=2, ensure_ascii=False))

    out_json = outdir / "cellist_swapxy_alignment_test_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("[DONE] Metrics JSON:", out_json)

    # Save GT test mask for optional inspection
    try:
        import tifffile
        out_gt = outdir / "gt_mask_swapxy_test.tif"
        tifffile.imwrite(str(out_gt), gt_mask, compression="zlib")
        print("[DONE] Rasterized GT mask:", out_gt)
    except Exception as exc:
        warnings.warn(f"Could not save GT mask: {exc}")

    # Save pred mask for optional inspection
    try:
        import tifffile
        out_pred = outdir / "cellist_pred_mask.tif"
        tifffile.imwrite(str(out_pred), pred_mask, compression="zlib")
        print("[DONE] Rasterized pred mask:", out_pred)
    except Exception as exc:
        warnings.warn(f"Could not save pred mask: {exc}")

    if not args.no_preview:
        try:
            out_png = outdir / "overlay_cellist_swapxy_test_downsampled.png"
            save_overlay_preview(pred_mask, gt_mask, out_png,
                                downsample=args.preview_downsample)
            print("[DONE] Overlay preview:", out_png)
        except Exception as exc:
            warnings.warn(f"Could not save overlay preview: {exc}")


if __name__ == "__main__":
    main()
