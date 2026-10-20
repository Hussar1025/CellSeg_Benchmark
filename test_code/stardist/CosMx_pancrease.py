#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stardist_benchmark_cosmx_multi.py
=================================
Runs StarDist segmentation over multiple FOVs (default 51-68) and benchmarks
against CosMx CellLabels (GT), aggregating exactly like the cellpose version:
micro (pooled) as the primary metric, macro (mean+/-std) as the robustness metric.
Metric code reuses \eval\test.py; aggregation/component functions directly reuse
cellpose_benchmark_cosmx_multi (its micro aggregation was validated against the
stitched full-image ground truth, differing by ~1e-7).

Key target issue
--------------------------------
StarDist's pretrained models ('2D_versatile_fluo', etc.) are nucleus segmentation
models, while CellLabels is whole-cell segmentation. Nuclei are smaller than whole
cells, so directly comparing nuclei to whole cells makes IoU/Dice/AJI systematically
lower - that is "different segmentation objects", not StarDist being worse.
So this script provides two correct, selectable evaluation targets (--target):

  --target cell   (default, same GT as the Cellpose run, for cross-method comparison)
      StarDist outputs nuclei -> expand with skimage.expand_labels by --expand-px
      pixels to approximate whole cells -> score against CellLabels (whole cells).
      Cost: the expansion distance is a manual hyperparameter (default 15px ~= 1.8um).
      This is the joint "StarDist+expansion" behavior, consistent with many real
      CosMx pipelines (nuclear segmentation + border expansion), but the expansion
      step must be reported honestly in the paper.

  --target nuclei (fairest to StarDist, no expansion hyperparameter)
      StarDist outputs nuclei -> score against "nucleus GT"; nucleus GT =
      CellLabels restricted to (CompartmentLabels == nucleus code), i.e. each cell's
      nuclear region inheriting that cell's ID.
      Note: CompartmentLabels is a "compartment type" label
      (nucleus/membrane/cytoplasm/extracellular), not instances; its integer codes
      are not fixed in the README, so specify and verify with --nuclear-code
      (the script prints the values actually present).

Recommendation: report both in the paper - the cell target for cross-method
comparison against Cellpose/others on the same GT; the nuclei target to fairly
reflect StarDist's own nuclear segmentation ability.

Dependencies
----
    pip install stardist tensorflow csbdeep
First run downloads pretrained weights online (works on AutoDL). For the 4256x4256
large images, pass --n-tiles (e.g. 8 8).

Usage
----
    # Same GT (whole cell) as Cellpose, directly comparable to previous results
    python stardist_benchmark_cosmx_multi.py --data-root /path/to/CosMx_Pancreas \
        --fov-start 51 --fov-end 68 --target cell --expand-px 15 \
        --out-dir ./stardist_bench_multi

    # Fairest nucleus target for StarDist
    python stardist_benchmark_cosmx_multi.py --data-root ... \
        --fov-start 51 --fov-end 68 --target nuclei --nuclear-code 1

    # Reuse saved StarDist prediction tif, skip inference
    python stardist_benchmark_cosmx_multi.py --data-root ... \
        --fov-start 51 --fov-end 68 --reuse-pred --pred-dir ./stardist_bench_multi
"""

import os
import sys
import json
import glob
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve_dep_dir():
    """
    Locate the directory of cellpose_benchmark_cosmx[_multi].py. Priority:
      1) env var COSMX_BENCH_DIR
      2) command-line --dep-dir (scanned manually before argparse, because the
         import happens first)
      3) fallback: this script's directory, current working directory
    """
    env = os.environ.get("COSMX_BENCH_DIR")
    if env and os.path.isdir(env):
        return os.path.abspath(env)
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--dep-dir" and i + 1 < len(argv):
            return os.path.abspath(argv[i + 1])
        if a.startswith("--dep-dir="):
            return os.path.abspath(a.split("=", 1)[1])
    return None


# Add the dependency dir (and fallback dirs) to sys.path, then import
_dep_dir = _resolve_dep_dir()
for _d in (_dep_dir, HERE, os.getcwd()):
    if _d and _d not in sys.path:
        sys.path.insert(0, _d)

try:
    import CosMx_single_fov as single      # file parsing / composite loading / metrics module
    import CosMx_benchplus as agg    # validated micro aggregation + component functions
except ImportError as _e:
    sys.stderr.write(
        "\n[import failed] cellpose_benchmark_cosmx.py / cellpose_benchmark_cosmx_multi.py not found.\n"
        "Tell the script where they are using any of these (pick one):\n"
        "  1) CLI argument:   --dep-dir /path/to/cellpose_files\n"
        "  2) env variable:   export COSMX_BENCH_DIR=/path/to/cellpose_files\n"
        "  3) put the two cellpose scripts in the same directory as this script\n"
        f"Directories tried: {[d for d in (_dep_dir, HERE, os.getcwd()) if d]}\n"
        f"Original error: {_e}\n\n")
    raise


# --------------------------------------------------------------------------- #
# StarDist input & inference
# --------------------------------------------------------------------------- #
def build_stardist_input(composite_rgb):
    """StarDist(fluo) is a nucleus model; use the DAPI (blue) single channel as input."""
    dapi = composite_rgb[..., 2].astype(np.float32)   # R,G,B -> B=DAPI
    return dapi


def run_stardist(dapi2d, model_name="2D_versatile_fluo",
                 prob_thresh=None, nms_thresh=None, n_tiles=None):
    """Return the nucleus instance label mask (int32)."""
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize
    print(f"[stardist] model={model_name} | prob={prob_thresh} nms={nms_thresh} "
          f"n_tiles={n_tiles}")
    model = StarDist2D.from_pretrained(model_name)
    x = normalize(dapi2d, 1, 99.8, axis=(0, 1))       # StarDist-recommended normalization
    if n_tiles is None:
        try:
            n_tiles = model._guess_n_tiles(x)
        except Exception:
            n_tiles = None
    labels, _ = model.predict_instances(
        x, prob_thresh=prob_thresh, nms_thresh=nms_thresh, n_tiles=n_tiles)
    labels = np.asarray(labels)
    if labels.ndim > 2:
        labels = np.squeeze(labels)
    return labels.astype(np.int32, copy=False)


def nuclei_to_cells(nuclei_labels, expand_px):
    """Nuclei -> approximate whole cells: expand_labels dilates each outward by expand_px, stopping at neighboring labels."""
    if not expand_px or expand_px <= 0:
        return nuclei_labels
    from skimage.segmentation import expand_labels
    return expand_labels(nuclei_labels, distance=expand_px).astype(nuclei_labels.dtype)


# --------------------------------------------------------------------------- #
# GT construction: whole-cell or nuclei
# --------------------------------------------------------------------------- #
def load_gt_cell(bench, labels_path):
    return bench.read_tif_mask(labels_path)


def resolve_compartment(data_root, fov, comp_dir=None):
    comp_dir = comp_dir or os.path.join(data_root, "CompartmentLabels")
    return single._find_fov_file(comp_dir, "CompartmentLabels", fov,
                                 exts=[".tif", ".tiff", ".png", ".jpg"])


def load_gt_nuclei(bench, cell_labels_path, comp_labels_path, nuclear_code):
    """Nucleus GT = the part of CellLabels within the nuclear compartment (inherits cell ID)."""
    cl = bench.read_tif_mask(cell_labels_path)
    cp = bench.read_tif_mask(comp_labels_path)
    if cl.shape != cp.shape:
        from skimage.transform import resize
        cp = resize(cp, cl.shape, order=0, preserve_range=True,
                    anti_aliasing=False).astype(cl.dtype)
    uniq = np.unique(cp)
    print(f"  [compartment] compartment codes present = {uniq.tolist()}  "
          f"(current --nuclear-code={nuclear_code}; verify which one is 'nucleus')")
    nuc_gt = np.where(cp == nuclear_code, cl, 0).astype(cl.dtype)
    return nuc_gt


# --------------------------------------------------------------------------- #
# single FOV
# --------------------------------------------------------------------------- #
def process_one_fov(bench, args, fov):
    comp_path, lab_path = single.resolve_paths(
        args.data_root, fov, args.composite_dir, args.labels_dir)
    composite = single.load_composite(comp_path)

    # ---- GT ----
    if args.target == "nuclei":
        cpath = resolve_compartment(args.data_root, fov, args.compartment_dir)
        if cpath is None:
            raise FileNotFoundError(
                f"target=nuclei needs CompartmentLabels, but FOV {fov} not found.")
        gt = load_gt_nuclei(bench, lab_path, cpath, args.nuclear_code)
    else:  # cell
        gt = load_gt_cell(bench, lab_path)

    # ---- prediction ----
    pred = None
    if args.reuse_pred:
        pat = os.path.join(args.pred_dir or args.out_dir, f"*stardist*F{fov:03d}*.tif")
        hits = sorted(glob.glob(pat)) or sorted(
            glob.glob(os.path.join(args.pred_dir or args.out_dir, f"*F{fov:03d}*.tif")))
        if hits:
            pred = bench.read_tif_mask(hits[0])
            print(f"  [reuse] {os.path.basename(hits[0])}")
    if pred is None:
        dapi = build_stardist_input(composite)
        nuclei = run_stardist(dapi, model_name=args.model,
                              prob_thresh=args.prob_thresh,
                              nms_thresh=args.nms_thresh,
                              n_tiles=tuple(args.n_tiles) if args.n_tiles else None)
        pred = nuclei_to_cells(nuclei, args.expand_px) if args.target == "cell" else nuclei
        try:
            import tifffile
            os.makedirs(args.out_dir, exist_ok=True)
            tag = f"{args.target}" + (f"_exp{args.expand_px}" if args.target == "cell" else "")
            outp = os.path.join(args.out_dir, f"stardist_{tag}_F{fov:03d}_masks.tif")
            tifffile.imwrite(outp, pred.astype(np.int32))
        except Exception as e:
            print(f"  [warn] failed to save prediction tif: {e}")

    # ---- alignment ----
    if gt.shape != pred.shape:
        from skimage.transform import resize
        print(f"  [warn] shape mismatch gt={gt.shape} pred={pred.shape}, nearest-neighbor rescale pred")
        pred = resize(pred, gt.shape, order=0, preserve_range=True,
                      anti_aliasing=False).astype(pred.dtype)

    # ---- one overlaps pass, derive all accumulable quantities (reuse validated component functions) ----
    overlaps = bench.compute_sparse_overlaps(gt, pred)
    comp = agg.pixel_components(bench, gt, pred, boundary_mode=args.boundary_mode)
    inst = agg.instance_components(bench, overlaps, iou_threshold=args.iou_threshold)
    aji_num, aji_den = agg.aji_components(bench, overlaps, gt, pred)
    mda_map = agg.mda_components(bench, overlaps)

    final = agg.per_fov_final_metrics(bench, comp, inst, aji_num, aji_den, mda_map)
    final.update(fov=fov,
                 gt_cell_count=len(overlaps.gt_ids),
                 pred_cell_count=len(overlaps.pred_ids))
    raw = dict(comp=comp, inst=inst, aji_num=aji_num, aji_den=aji_den, mda_map=mda_map)
    return final, raw


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Multi-FOV StarDist vs CellLabels benchmark summary")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dep-dir", default=None,
                    help="Directory of cellpose_benchmark_cosmx[_multi].py (if different "
                         "from this script's). Equivalent to setting env COSMX_BENCH_DIR.")
    ap.add_argument("--fov-start", type=int, default=51)
    ap.add_argument("--fov-end", type=int, default=68)
    ap.add_argument("--fovs", type=int, nargs="*", default=None)
    ap.add_argument("--composite-dir", default=None)
    ap.add_argument("--labels-dir", default=None)
    ap.add_argument("--compartment-dir", default=None)
    ap.add_argument("--test-py", default=None,
                    help="path to test.py; searched in --dep-dir / this script's dir by default")
    # StarDist
    ap.add_argument("--model", default="2D_versatile_fluo",
                    help="2D_versatile_fluo(fluorescence nucleus, default) / 2D_paper_dsb2018 / 2D_versatile_he")
    ap.add_argument("--prob-thresh", type=float, default=None, help="None=use model default (~0.479)")
    ap.add_argument("--nms-thresh", type=float, default=None)
    ap.add_argument("--n-tiles", type=int, nargs=2, default=None,
                    help="tiling for large images, e.g. --n-tiles 8 8; auto-guessed if omitted")
    # evaluation target
    ap.add_argument("--target", choices=["cell", "nuclei"], default="cell",
                    help="cell=expand to whole cells and compare to CellLabels (default, "
                         "cross-comparable with Cellpose); nuclei=compare to nucleus GT (fairest to StarDist)")
    ap.add_argument("--expand-px", type=int, default=15,
                    help="nucleus->cell expansion pixels for target=cell (~1.8um@0.12028). "
                         "A manual hyperparameter; report it")
    ap.add_argument("--nuclear-code", type=int, default=1,
                    help="integer code for 'nucleus' in CompartmentLabels for target=nuclei (verify it)")
    # common
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--boundary-mode", default="thick")
    ap.add_argument("--out-dir", default="./stardist_bench_multi")
    ap.add_argument("--reuse-pred", action="store_true")
    ap.add_argument("--pred-dir", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # resolve test.py: explicit wins, else search in dep-dir / this script's dir
    test_py = args.test_py
    if not test_py:
        for _d in (args.dep_dir, _dep_dir, HERE, os.getcwd()):
            if _d and os.path.exists(os.path.join(_d, "test.py")):
                test_py = os.path.join(_d, "test.py")
                break
        test_py = test_py or os.path.join(HERE, "test.py")
    bench = single.load_metric_module(test_py)

    fov_list = args.fovs if args.fovs else list(range(args.fov_start, args.fov_end + 1))
    print(f"[plan] StarDist({args.model}) | target={args.target}"
          + (f" expand={args.expand_px}px" if args.target == "cell" else
             f" nuclear_code={args.nuclear_code}")
          + f" | {len(fov_list)} FOVs total: {fov_list}")

    per_fov, raws, failed = [], [], []
    for fov in fov_list:
        print(f"\n===== FOV {fov} =====")
        try:
            final, raw = process_one_fov(bench, args, fov)
        except Exception as e:
            print(f"  [skip] FOV {fov} failed: {e}")
            failed.append((fov, str(e)))
            continue
        per_fov.append(final)
        raws.append(raw)
        print(f"  gt={final['gt_cell_count']} pred={final['pred_cell_count']} "
              f"F1={final['instance_f1']:.3f} AJI={final['AJI']:.3f} PQ={final['PQ']:.3f}")

    if not per_fov:
        raise SystemExit("No FOV succeeded. Check paths/file naming/dependency installation.")

    import pandas as pd
    df = pd.DataFrame(per_fov)
    metric_cols = ["dice", "iou", "pixel_accuracy", "boundary_f1", "boundary_precision",
                   "boundary_recall", "instance_f1", "precision", "recall",
                   "DQ", "SQ", "PQ", "AJI", "mDA_0.50_0.95"]
    front = ["fov", "gt_cell_count", "pred_cell_count", "TP", "FP", "FN"]
    df = df[[c for c in front if c in df] + [c for c in metric_cols if c in df]]
    tag = f"{args.target}" + (f"_exp{args.expand_px}" if args.target == "cell" else "")
    per_fov_csv = os.path.join(args.out_dir, f"per_fov_metrics_stardist_{tag}.csv")
    df.to_csv(per_fov_csv, index=False)

    # micro (pooled, primary metric)
    micro = agg.aggregate_micro(raws)
    total_gt = int(sum(f["gt_cell_count"] for f in per_fov))
    total_pred = int(sum(f["pred_cell_count"] for f in per_fov))

    # macro (per-FOV mean+/-std)
    macro_mean = {c: float(df[c].mean()) for c in metric_cols if c in df}
    macro_std = {c: float(df[c].std(ddof=1)) if len(df) > 1 else 0.0
                 for c in metric_cols if c in df}

    summary_rows = [dict(metric=c,
                         micro=round(micro.get(c, float("nan")), 6),
                         macro_mean=round(macro_mean.get(c, float("nan")), 6),
                         macro_std=round(macro_std.get(c, float("nan")), 6))
                    for c in metric_cols]
    sdf = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.out_dir, f"aggregate_summary_stardist_{tag}.csv")
    sdf.to_csv(summary_csv, index=False)

    meta = dict(method="StarDist", model=args.model, target=args.target,
                expand_px=(args.expand_px if args.target == "cell" else None),
                nuclear_code=(args.nuclear_code if args.target == "nuclei" else None),
                n_fov_ok=len(per_fov), n_fov_failed=len(failed), failed=failed,
                fov_list=fov_list, iou_threshold=args.iou_threshold,
                total_gt_cells=total_gt, total_pred_cells=total_pred,
                micro_TP=micro["TP"], micro_FP=micro["FP"], micro_FN=micro["FN"],
                note="micro=pooled (primary metric; AJI/PQ must use it); macro=per-FOV mean+/-std; "
                     "StarDist pretrained is a nucleus model, cell target includes the expansion "
                     "hyperparameter; pixel_accuracy is a weak metric.")
    with open(os.path.join(args.out_dir, f"run_meta_stardist_{tag}.json"),
              "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print(f"StarDist | target={args.target}"
          + (f" expand={args.expand_px}px" if args.target == "cell" else
             f" nuclear_code={args.nuclear_code}"))
    print(f"Success {len(per_fov)}/{len(fov_list)} FOVs | GT={total_gt} predicted={total_pred}")
    if failed:
        print(f"Failed FOVs: {[f for f, _ in failed]}")
    print(f"micro counts: TP={micro['TP']} FP={micro['FP']} FN={micro['FN']}")
    print("-" * 78)
    show = sdf.copy()
    show["macro(mean+/-std)"] = [f"{m:.4f}+/-{s:.4f}"
                                 for m, s in zip(show["macro_mean"], show["macro_std"])]
    print(show[["metric", "micro", "macro(mean+/-std)"]].to_string(index=False))
    print("-" * 78)
    print("Report micro as primary; macro(mean+/-std) for robustness. AJI/PQ use micro only.")
    print("StarDist is a nucleus model: the cell target includes the --expand-px hyperparameter "
          "(report it); also report the nuclei target.")
    print(f"\n[saved]\n  {per_fov_csv}\n  {summary_csv}\n  "
          f"{os.path.join(args.out_dir, f'run_meta_stardist_{tag}.json')}")


if __name__ == "__main__":
    main()


'''
nohup bash -c '
    python CosMx_stardist.py \
    --dep-dir yourpath/cellpose \
    --data-root yourpath/dataset/Pancreas-CosMx-WTx-FlatFiles --fov-start 51 --fov-end 68 \
    --target cell --expand-px 15 \
    --test-py yourpath/eval/test.py \
    --out-dir yourpath/stardist/CosMx_output
' > output.log 2>&1 &
'''
