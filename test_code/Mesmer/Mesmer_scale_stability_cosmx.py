#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mesmer_scale_stability_cosmx.py
===============================
Scale-stability experiment specific to Mesmer:
  1) stitch several FOVs (default 51-68) into one large mosaic (composite and
     CellLabels-GT stitched together);
  2) take concentric nested square crops centred on the centroid of the occupied
     mosaic region: sides 5000 / 8000 / 10000 / 15000 px / full image
     (override with --crop-sizes);
  3) run Mesmer once per crop and score it against the identically cropped GT
     (metrics reused from test.py);
  4) write a "metric vs crop area / cell count" table plus a line plot to see
     whether segmentation stays stable as the cell count grows.

Rationale: Mesmer's predict rescales the image to the training resolution
(0.5um/px) using image_mpp and then tiles it internally, so in theory the result
should be nearly invariant to crop size. This experiment measures whether it
actually is.

WARNING: confounder - crop borders cut through edge cells, and the cut-cell
fraction is higher in small crops, which by itself lowers small-crop scores
(unrelated to Mesmer). Use --exclude-border to drop GT/pred instances touching
the crop border; only then does it cleanly measure Mesmer's scale behaviour. Run
both settings and compare.

Mosaic layout
-------------
--layout grid (default): tile the FOVs seamlessly into an approximate square
  (dense, gapless, stable; best suited to "cell count vs stability").
--layout positions: physically stitch using the global coordinates in
  Pancreas_fov_positions_file.csv (more realistic, but may leave gaps; less
  stable than grid when column names/units are uncertain).

Usage
-----
    python mesmer_scale_stability_cosmx.py \
        --dep-dir /path/to/cellpose_files \
        --data-root /path/to/CosMx_Pancreas --fov-start 51 --fov-end 68 \
        --image-mpp 0.12028 --exclude-border \
        --out-dir ./mesmer_scale

    # reuse saved size-named predictions (mesmer_scale_{size}_masks.tif); score/plot only
    python mesmer_scale_stability_cosmx.py --dep-dir ... --data-root ... \
        --reuse-pred --pred-dir ./mesmer_scale
"""

import os
import sys
import json
import glob
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve_dep_dir():
    env = os.environ.get("COSMX_BENCH_DIR")
    if env and os.path.isdir(env):
        return os.path.abspath(env)
    for i, a in enumerate(sys.argv):
        if a == "--dep-dir" and i + 1 < len(sys.argv):
            return os.path.abspath(sys.argv[i + 1])
        if a.startswith("--dep-dir="):
            return os.path.abspath(a.split("=", 1)[1])
    return None


_dep_dir = _resolve_dep_dir()
for _d in (_dep_dir, HERE, os.getcwd()):
    if _d and _d not in sys.path:
        sys.path.insert(0, _d)

try:
    import CosMx_single_fov as single
    import CosMx_benchplus as agg
    import cellseg_benchmark.code.Mesmer.CosMx_pancrease as mm
except ImportError as _e:
    sys.stderr.write(
        "\n[import failed] need cellpose_benchmark_cosmx.py / cellpose_benchmark_cosmx_multi.py / "
        "mesmer_benchmark_cosmx_multi.py in the same directory or the --dep-dir directory.\n"
        f"Tried so far: {[d for d in (_dep_dir, HERE, os.getcwd()) if d]}\noriginal error: {_e}\n\n")
    raise


# --------------------------------------------------------------------------- #
# Mosaic: composite (nucleus/membrane 2-channel uint8) + CellLabels (int32, globally unique IDs)
# --------------------------------------------------------------------------- #
def _grid_layout(n):
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


def _read_fov_offsets(positions_csv, fov_list, mpp, tile_h, tile_w):
    """positions layout: return {fov: (row_px, col_px)}. Parses column names/units defensively."""
    import pandas as pd
    df = pd.read_csv(positions_csv)
    low = {c.lower(): c for c in df.columns}
    fov_col = next((low[k] for k in low if k in ("fov", "fov_id", "fovid")), None)
    xcol = next((low[k] for k in low if k.startswith("x")), None)
    ycol = next((low[k] for k in low if k.startswith("y")), None)
    if not (fov_col and xcol and ycol):
        raise ValueError(f"fov_positions column names not recognised: {list(df.columns)}")
    in_mm = ("mm" in xcol.lower()) or ("mm" in ycol.lower())
    to_px = (1000.0 / mpp) if in_mm else 1.0
    off = {}
    for _, r in df.iterrows():
        f = int(r[fov_col])
        if f in fov_list:
            off[f] = (float(r[ycol]) * to_px, float(r[xcol]) * to_px)
    if not off:
        raise ValueError("no requested FOV matched in fov_positions")
    ys = [v[0] for v in off.values()]; xs = [v[1] for v in off.values()]
    y0, x0 = min(ys), min(xs)
    return {f: (int(round(y - y0)), int(round(x - x0))) for f, (y, x) in off.items()}


def build_mosaic(bench, args, fov_list):
    """Return nuc (uint8 HxW), mem (uint8 HxW), gt (int32 HxW), info."""
    # Probe one FOV first to get the tile size
    comp0, lab0 = single.resolve_paths(args.data_root, fov_list[0],
                                       args.composite_dir, args.labels_dir)
    c0 = single.load_composite(comp0)
    tile_h, tile_w = c0.shape[:2]
    print(f"[mosaic] tile size = {tile_h}x{tile_w} | layout={args.layout}")

    if args.layout == "positions":
        pos = args.positions or os.path.join(args.data_root, "Pancreas_fov_positions_file.csv")
        offsets = _read_fov_offsets(pos, set(fov_list), args.image_mpp, tile_h, tile_w)
        H = max(y for y, _ in offsets.values()) + tile_h
        W = max(x for _, x in offsets.values()) + tile_w
        place = {f: offsets[f] for f in fov_list if f in offsets}
    else:  # grid
        rows, cols = _grid_layout(len(fov_list))
        H, W = rows * tile_h, cols * tile_w
        place = {f: ((i // cols) * tile_h, (i % cols) * tile_w)
                 for i, f in enumerate(fov_list)}
        print(f"[mosaic] grid = {rows}x{cols}")

    print(f"[mosaic] canvas = {H}x{W} ({H*W/1e6:.0f} MP)")
    nuc = np.zeros((H, W), np.uint8)
    mem = np.zeros((H, W), np.uint8)
    gt = np.zeros((H, W), np.int32)
    next_id = 0
    placed = 0
    for f in fov_list:
        if f not in place:
            print(f"  [skip] FOV {f} has no position")
            continue
        try:
            cpath, lpath = single.resolve_paths(args.data_root, f,
                                                args.composite_dir, args.labels_dir)
            comp = single.load_composite(cpath)
            lab = bench.read_tif_mask(lpath)
        except Exception as e:
            print(f"  [skip] FOV {f}: {e}")
            continue
        y0, x0 = place[f]
        h, w = comp.shape[:2]
        y1, x1 = min(y0 + h, H), min(x0 + w, W)
        h, w = y1 - y0, x1 - x0
        R, G, B = comp[:h, :w, 0], comp[:h, :w, 1], comp[:h, :w, 2]
        nuc[y0:y1, x0:x1] = np.clip(B, 0, 255).astype(np.uint8)
        mem[y0:y1, x0:x1] = np.clip(np.maximum(R, G), 0, 255).astype(np.uint8)
        lab = lab[:h, :w]
        # Offset IDs to keep them globally unique
        m = lab > 0
        gt_block = np.zeros_like(lab, dtype=np.int32)
        if m.any():
            gt_block[m] = lab[m].astype(np.int32) + next_id
            next_id += int(lab.max())
        gt[y0:y1, x0:x1] = gt_block
        placed += 1
    print(f"[mosaic] placed {placed}/{len(fov_list)} FOVs, GT total cells ~{len(bench.get_instance_ids(gt))}")
    # Centroid of the occupied region (for centred crops)
    ys, xs = np.where(nuc > 0)
    if ys.size:
        cy, cx = int(ys.mean()), int(xs.mean())
    else:
        cy, cx = H // 2, W // 2
    return nuc, mem, gt, dict(H=H, W=W, cy=cy, cx=cx, tile=(tile_h, tile_w))


# --------------------------------------------------------------------------- #
# Crops & border removal
# --------------------------------------------------------------------------- #
def centered_crop_bounds(H, W, cy, cx, size):
    if size >= H and size >= W:
        return 0, H, 0, W
    sh, sw = min(size, H), min(size, W)
    y0 = int(np.clip(cy - sh // 2, 0, H - sh))
    x0 = int(np.clip(cx - sw // 2, 0, W - sw))
    return y0, y0 + sh, x0, x0 + sw


def drop_border_objects(lab):
    """Drop instances touching any of the four image borders (cut cells)."""
    ids = set(np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])))
    ids.discard(0)
    if not ids:
        return lab
    out = lab.copy()
    out[np.isin(out, list(ids))] = 0
    return out


# --------------------------------------------------------------------------- #
# Score one crop (reuses agg components; same as one micro pass on that crop)
# --------------------------------------------------------------------------- #
def score(bench, gt_crop, pred_crop, iou_threshold, boundary_mode):
    overlaps = bench.compute_sparse_overlaps(gt_crop, pred_crop)
    comp = agg.pixel_components(bench, gt_crop, pred_crop, boundary_mode=boundary_mode)
    inst = agg.instance_components(bench, overlaps, iou_threshold=iou_threshold)
    aji_num, aji_den = agg.aji_components(bench, overlaps, gt_crop, pred_crop)
    mda_map = agg.mda_components(bench, overlaps)
    m = agg.per_fov_final_metrics(bench, comp, inst, aji_num, aji_den, mda_map)
    m["gt_cells"] = len(overlaps.gt_ids)
    m["pred_cells"] = len(overlaps.pred_ids)
    return m


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Mesmer scale stability: mosaic + concentric nested crops")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dep-dir", default=None)
    ap.add_argument("--fov-start", type=int, default=51)
    ap.add_argument("--fov-end", type=int, default=68)
    ap.add_argument("--fovs", type=int, nargs="*", default=None)
    ap.add_argument("--composite-dir", default=None)
    ap.add_argument("--labels-dir", default=None)
    ap.add_argument("--test-py", default=None)
    ap.add_argument("--layout", choices=["grid", "positions"], default="grid")
    ap.add_argument("--positions", default=None, help="fov_positions csv (for layout=positions)")
    ap.add_argument("--crop-sizes", type=int, nargs="+", default=[5000, 8000, 10000, 15000],
                    help="square crop side lengths (px); a full-image crop is appended automatically")
    ap.add_argument("--exclude-border", action="store_true",
                    help="drop instances touching the crop border (recommended; removes the cut-cell confounder)")
    # Mesmer
    ap.add_argument("--image-mpp", type=float, default=0.12028)
    ap.add_argument("--compartment", default="whole-cell", choices=["whole-cell", "nuclear", "both"])
    ap.add_argument("--batch-size", type=int, default=4)
    # Generic
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--boundary-mode", default="thick")
    ap.add_argument("--out-dir", default="./mesmer_scale")
    ap.add_argument("--reuse-pred", action="store_true")
    ap.add_argument("--pred-dir", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    test_py = args.test_py
    if not test_py:
        for _d in (args.dep_dir, _dep_dir, HERE, os.getcwd()):
            if _d and os.path.exists(os.path.join(_d, "test.py")):
                test_py = os.path.join(_d, "test.py"); break
        test_py = test_py or os.path.join(HERE, "test.py")
    bench = single.load_metric_module(test_py)

    fov_list = args.fovs if args.fovs else list(range(args.fov_start, args.fov_end + 1))
    nuc, mem, gt, info = build_mosaic(bench, args, fov_list)
    H, W, cy, cx = info["H"], info["W"], info["cy"], info["cx"]

    # Crop sizes: dedupe, sort ascending, append the full image
    sizes = sorted(set(int(s) for s in args.crop_sizes if s > 0))
    full = max(H, W)
    sizes = [s for s in sizes if s < full] + [full]

    runner = None
    if not args.reuse_pred:
        runner = mm.MesmerRunner(image_mpp=args.image_mpp, compartment=args.compartment,
                                 batch_size=args.batch_size)

    import tifffile
    rows = []
    for size in sizes:
        tag = "full" if size == full else str(size)
        y0, y1, x0, x1 = centered_crop_bounds(H, W, cy, cx, size)
        gt_c = gt[y0:y1, x0:x1].copy()
        area_mp = (y1 - y0) * (x1 - x0) / 1e6
        print(f"\n===== crop {tag} : {y1-y0}x{x1-x0} ({area_mp:.0f} MP) @({y0},{x0}) =====")

        # Prediction
        pred_c = None
        if args.reuse_pred:
            hits = sorted(glob.glob(os.path.join(args.pred_dir or args.out_dir,
                                                 f"*scale_{tag}_*mask*.tif")))
            if hits:
                pred_c = bench.read_tif_mask(hits[0])
                print(f"  [reuse] {os.path.basename(hits[0])}")
        if pred_c is None:
            X = np.stack([nuc[y0:y1, x0:x1], mem[y0:y1, x0:x1]], axis=-1)[np.newaxis].astype(np.float32)
            pred_c = runner.segment(X)
            try:
                tifffile.imwrite(os.path.join(args.out_dir, f"mesmer_scale_{tag}_masks.tif"),
                                 pred_c.astype(np.int32))
            except Exception as e:
                print(f"  [warn] save failed: {e}")

        if pred_c.shape != gt_c.shape:
            from skimage.transform import resize
            pred_c = resize(pred_c, gt_c.shape, order=0, preserve_range=True,
                            anti_aliasing=False).astype(pred_c.dtype)

        if args.exclude_border:
            gt_s = drop_border_objects(gt_c)
            pred_s = drop_border_objects(pred_c)
        else:
            gt_s, pred_s = gt_c, pred_c

        m = score(bench, gt_s, pred_s, args.iou_threshold, args.boundary_mode)
        row = dict(crop=tag, side_px=(y1 - y0), area_MP=round(area_mp, 1),
                   gt_cells=m["gt_cells"], pred_cells=m["pred_cells"],
                   cell_density_per_MP=round(m["gt_cells"] / max(area_mp, 1e-6), 1))
        for k in ["instance_f1", "precision", "recall", "PQ", "DQ", "SQ", "AJI",
                  "mDA_0.50_0.95", "dice", "iou", "boundary_f1"]:
            row[k] = round(float(m[k]), 4)
        rows.append(row)
        print(f"  gt={row['gt_cells']} pred={row['pred_cells']} "
              f"F1={row['instance_f1']:.3f} AJI={row['AJI']:.3f} PQ={row['PQ']:.3f}")

    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "mesmer_scale_stability.csv")
    df.to_csv(csv_path, index=False)

    # Stability summary: coefficient of variation CV=std/mean and range of key metrics
    stab_cols = ["instance_f1", "AJI", "PQ", "SQ", "dice", "iou"]
    stab = {}
    for c in stab_cols:
        v = df[c].to_numpy(dtype=float)
        stab[c] = dict(mean=round(v.mean(), 4), std=round(v.std(ddof=1), 4) if len(v) > 1 else 0.0,
                       range=round(v.max() - v.min(), 4),
                       cv=round(v.std(ddof=1) / (v.mean() + 1e-9), 4) if len(v) > 1 else 0.0)
    with open(os.path.join(args.out_dir, "stability_summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(exclude_border=args.exclude_border, image_mpp=args.image_mpp,
                       layout=args.layout, sizes=sizes, stability=stab), f,
                  ensure_ascii=False, indent=2)

    # Line plot: metric vs crop side (x axis also annotates cell counts)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5.5))
        x = df["side_px"].to_numpy()
        for c, mk in [("instance_f1", "o"), ("AJI", "s"), ("PQ", "^"),
                      ("SQ", "d"), ("iou", "v")]:
            ax.plot(x, df[c], marker=mk, label=c.replace("instance_f1", "F1"))
        ax.set_xlabel("crop side (px)   -  cells: " +
                      ", ".join(f"{s}px:{n}" for s, n in zip(df["side_px"], df["gt_cells"])),
                      fontsize=8)
        ax.set_ylabel("metric (micro)")
        ax.set_ylim(0, 1)
        ax.set_title(f"Mesmer scale stability (exclude_border={args.exclude_border}, "
                     f"mpp={args.image_mpp})")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        png = os.path.join(args.out_dir, "mesmer_scale_stability.png")
        fig.savefig(png, dpi=130)
        plt.close(fig)
        print(f"[plot] {png}")
    except Exception as e:
        print(f"[plot] skip: {e}")

    print("\n" + "=" * 88)
    print(df.to_string(index=False))
    print("=" * 88)
    print("Stability across crop sizes  CV=std/mean, lower is more stable:")
    for c in stab_cols:
        print(f"  {c:12s} mean={stab[c]['mean']:.4f}  std={stab[c]['std']:.4f}  "
              f"range={stab[c]['range']:.4f}  CV={stab[c]['cv']:.4f}")
    if not args.exclude_border:
        print("Tip: --exclude-border is off, so low scores on small crops may come from border-cut cells "
              "rather than Mesmer itself; re-run with --exclude-border to compare.")
    print(f"\n[saved]\n  {csv_path}\n  "
          f"{os.path.join(args.out_dir, 'stability_summary.json')}")


if __name__ == "__main__":
    main()

'''
nohup bash -c '
    python Mesmer_scale_stability_cosmx.py \
    --dep-dir yourpath/cellpose \
    --data-root yourpath/dataset/Pancreas-CosMx-WTx-FlatFiles --fov-start 51 --fov-end 68 \
    --test-py yourpath/eval/test.py \
    --image-mpp 0.12028 --exclude-border \
    --out-dir yourpath/Mesmer/CosMx_scale_stability_output
' > output.log 2>&1 &
'''