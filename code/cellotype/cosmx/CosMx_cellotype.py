#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cellotype_benchmark_cosmx_multi.py
==================================
Run CelloType (TissueNet pretrained model, zero-shot, no fine-tuning) whole-cell
segmentation on multiple FOVs (default 51-68) and benchmark against CosMx
CellLabels (GT). Aggregation is identical to the cellpose / stardist versions:
micro (pooled) is the primary metric, macro (mean +/- std) is robustness.
Metric functions reused from test.py; micro aggregation reuses the verified
cellpose_benchmark_cosmx_multi implementation (~1e-7 from the mosaic truth).

Why no fine-tuning (important)
------------------------------
* Fairness: Cellpose (cpsam) and StarDist (2D_versatile_fluo) are zero-shot;
  CelloType also zero-shot keeps all four methods on the same footing.
* Leakage: only 51-68 FOVs from the same slide; train/test on the same data
  systematically overestimates.
* GT is silver standard: CellLabels is NanoString's algorithm output, so
  fine-tuning on it only imitates the vendor.
* Authors' own guidance: TissueNet/Cellpose pretrained models "transfer as long
  as the new image has a nuclear/membrane channel"; CosMx composite provides
  DAPI (nuclear) + PanCK/CD45 (membrane).
CelloType natively produces whole-cell masks -> directly comparable to
whole-cell CellLabels without StarDist-style dilation, identical scale to
Cellpose.

Inference paths (two options)
-----------------------------
CelloType stable API: CelloTypePredictor(model_path, config_path, ...).predict(img)->[H,W].
For large images (4256^2) the official recipe is 512 sliding windows via the
repo's sahi_inference.py, but that SAHI wrapper varies by repo version. So this
script:
  --infer tile (default): relies only on the stable .predict(); does its own
                          512 tiling + instance stitching (unit-tested).
  --infer whole: one predict on the whole image (only for small images/large VRAM;
                 Detectron2 rescales by short edge, which may drop small cells).
  Or: run the official sahi_inference.py first to save per-FOV masks as
      cellotype_*_F0XX_masks.tif, then use --reuse-pred for scoring only
      (closest to official, most stable).

Dependencies
------------
    CelloType (github.com/tanlabcode/CelloType) + Detectron2 + MaskDINO; download
    pretrained weights models/tissuenet_model_0019999.pth and config
    configs/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml. CUDA GPU required.
Everything except CelloType inference (tile stitching, input construction,
multi-FOV micro/macro aggregation, CSV/JSON) has been validated on synthetic
data; inference runs on your AutoDL machine.

Usage
-----
    # integrated inference (tile mode)
    python cellotype_benchmark_cosmx_multi.py \
        --dep-dir yourpath \
        --data-root yourpath --fov-start 51 --fov-end 68 \
        --model-path yourpath \
        --config-path yourpath \
        --device cuda --out-dir yourpath

    # use official SAHI masks, score only
    python cellotype_benchmark_cosmx_multi.py --dep-dir ... --data-root ... \
        --fov-start 51 --fov-end 68 --reuse-pred --pred-dir yourpath
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import sys
import json
import glob
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# Dependency location (cellpose_benchmark_cosmx[_multi].py may live elsewhere)
def _resolve_dep_dir():
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


_dep_dir = _resolve_dep_dir()
for _d in (_dep_dir, HERE, os.getcwd()):
    if _d and _d not in sys.path:
        sys.path.insert(0, _d)

try:
    import CosMx_single_fov as single       # file parsing / composite loading / metric module
    import CosMx_benchplus as agg           # verified micro aggregation + component functions
except ImportError as _e:
    sys.stderr.write(
        "\n[import failed] could not find cellpose_benchmark_cosmx.py / cellpose_benchmark_cosmx_multi.py.\n"
        "Specify their directory via one of:\n"
        "  1) CLI:  --dep-dir yourpath\n"
        "  2) env:  export COSMX_BENCH_DIR=yourpath\n"
        "  3) place all three scripts in the same directory\n"
        f"tried: {[d for d in (_dep_dir, HERE, os.getcwd()) if d]}\noriginal error: {_e}\n\n")
    raise


# CelloType input (TissueNet convention: R=optional, G=membrane, B=nuclear)
def build_cellotype_input(composite_rgb, channel_order="opt_mem_nuc"):
    """
    CosMx composite: R=CD45/GFAP, G=PanCK, B=DAPI.
    CelloType/TissueNet convention is typically [R=optional, G=membrane, B=nuclear]
    (per DeepWiki) -> membrane = max(CD45, PanCK) = max(R,G); nuclear = DAPI = B;
    optional = 0.
    WARNING: verify channel order against CelloType repo
    data/example/example_tissuenet.png; reversed channels cost accuracy.
    Override with --channel-order:
         opt_mem_nuc -> [0, membrane, nuclear]  (default)
         nuc_mem_opt -> [nuclear, membrane, 0]
         mem_nuc_opt -> [membrane, nuclear, 0]
    Returns uint8 [H,W,3] (CelloType reads PNG as uint8, normalizes inside).
    """
    R = composite_rgb[..., 0]; G = composite_rgb[..., 1]; B = composite_rgb[..., 2]
    membrane = np.maximum(R, G)
    nuclear = B
    zero = np.zeros_like(B)
    order = {
        "opt_mem_nuc": (zero, membrane, nuclear),
        "nuc_mem_opt": (nuclear, membrane, zero),
        "mem_nuc_opt": (membrane, nuclear, zero),
    }[channel_order]
    out = np.stack(order, axis=-1)
    return np.clip(out, 0, 255).astype(np.uint8)


# Tiling + instance stitching (depends only on the stable predict(); unit-tested)
def _starts(size, tile, step):
    if size <= tile:
        return [0]
    s = list(range(0, size - tile + 1, step))
    if s[-1] != size - tile:
        s.append(size - tile)
    return s


def tile_predict_stitch(img_rgb, predict_fn, tile=512, overlap=128, merge_frac=0.5):
    """Cut a large image into tile x tile windows (with overlap), predict each,
    then greedily stitch into a global instance mask.
    Greedy policy: first come first served; later tiles that overlap an existing
    instance by >= merge_frac are merged into it; otherwise a new id fills the
    empty pixels.
    overlap should be >= max cell diameter (~125 px for pancreas) to avoid
    splitting cells on seams.
    Note: convenient wrapper around the stable API; for stricter large-image
    inference use the official SAHI (NMM merge)."""
    H, W = img_rgb.shape[:2]
    gmask = np.zeros((H, W), np.int32)
    nxt = 1
    step = max(1, tile - overlap)
    for y0 in _starts(H, tile, step):
        for x0 in _starts(W, tile, step):
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            sub = img_rgb[y0:y1, x0:x1]
            local = np.asarray(predict_fn(sub))
            if local.ndim > 2:
                local = np.squeeze(local)
            g_sub = gmask[y0:y1, x0:x1]           # view; in-place writes update gmask
            for lid in np.unique(local):
                if lid == 0:
                    continue
                region = local == lid
                nz = g_sub[region]
                nz = nz[nz > 0]
                assigned = False
                if nz.size:
                    vals, counts = np.unique(nz, return_counts=True)
                    top = int(vals[counts.argmax()])
                    if counts.max() / region.sum() >= merge_frac:
                        g_sub[region & (g_sub == 0)] = top
                        assigned = True
                if not assigned:
                    fill = region & (g_sub == 0)
                    if fill.sum() > 0:
                        g_sub[fill] = nxt
                        nxt += 1
    return gmask


# CelloType runner (model loaded once, reused across FOVs)
class CelloTypeRunner:
    def __init__(self, model_path, config_path, confidence, max_det, device,
                 infer="tile", tile=512, overlap=128):
        self.model_path = model_path
        self.config_path = config_path
        self.confidence = confidence
        self.max_det = max_det
        self.device = device
        self.infer = infer
        self.tile = tile
        self.overlap = overlap
        self._predictor = None

    def _predictor_obj(self):
        if self._predictor is None:
            from cellotype.predict import CelloTypePredictor
            if not self.model_path or not self.config_path:
                raise SystemExit("need --model-path and --config-path (CelloType weights and config).")
            print(f"[cellotype] loading model {os.path.basename(self.model_path)} "
                  f"| conf={self.confidence} max_det={self.max_det} device={self.device}")
            self._predictor = CelloTypePredictor(
                model_path=self.model_path,
                confidence_thresh=self.confidence,
                max_det=self.max_det,
                device=self.device,
                config_path=self.config_path)
        return self._predictor

    def segment(self, img_rgb):
        pred = self._predictor_obj()
        if self.infer == "whole":
            m = np.asarray(pred.predict(img_rgb))
            return (np.squeeze(m) if m.ndim > 2 else m).astype(np.int32)
        # tile (default)
        return tile_predict_stitch(img_rgb, lambda s: pred.predict(s),
                                   tile=self.tile, overlap=self.overlap).astype(np.int32)


# Single FOV
def process_one_fov(bench, args, fov, runner):
    comp_path, lab_path = single.resolve_paths(
        args.data_root, fov, args.composite_dir, args.labels_dir)
    composite = single.load_composite(comp_path)
    gt = bench.read_tif_mask(lab_path)             # whole-cell GT

    # prediction
    pred = None
    if args.reuse_pred:
        pat = os.path.join(args.pred_dir or args.out_dir, f"*cellotype*F{fov:03d}*.tif")
        hits = sorted(glob.glob(pat)) or sorted(
            glob.glob(os.path.join(args.pred_dir or args.out_dir, f"*F{fov:03d}*.tif")))
        if hits:
            pred = bench.read_tif_mask(hits[0])
            print(f"  [reuse] {os.path.basename(hits[0])}")
    if pred is None:
        cin = build_cellotype_input(composite, channel_order=args.channel_order)
        pred = runner.segment(cin)
        try:
            import tifffile
            os.makedirs(args.out_dir, exist_ok=True)
            outp = os.path.join(args.out_dir, f"cellotype_tissuenet_F{fov:03d}_masks.tif")
            tifffile.imwrite(outp, pred.astype(np.int32))
        except Exception as e:
            print(f"  [warn] failed to save prediction tif: {e}")

    # align
    if gt.shape != pred.shape:
        from skimage.transform import resize
        print(f"  [warn] shape mismatch gt={gt.shape} pred={pred.shape}; nearest-neighbor resize of pred")
        pred = resize(pred, gt.shape, order=0, preserve_range=True,
                      anti_aliasing=False).astype(pred.dtype)

    # one overlaps pass -> derive all additive quantities (reuses verified components)
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


# main
def main():
    ap = argparse.ArgumentParser(description="Multi-FOV CelloType (TissueNet pretrained, zero-shot) vs CellLabels benchmark")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dep-dir", default=None,
                    help="directory containing cellpose_benchmark_cosmx[_multi].py (when separate). Equivalent to env COSMX_BENCH_DIR.")
    ap.add_argument("--fov-start", type=int, default=51)
    ap.add_argument("--fov-end", type=int, default=68)
    ap.add_argument("--fovs", type=int, nargs="*", default=None)
    ap.add_argument("--composite-dir", default=None)
    ap.add_argument("--labels-dir", default=None)
    ap.add_argument("--test-py", default=None, help="path to test.py; defaults to dep-dir / script dir")
    # CelloType
    ap.add_argument("--model-path", default=None,
                    help="CelloType weights, e.g. .../models/tissuenet_model_0019999.pth")
    ap.add_argument("--config-path", default=None,
                    help="CelloType config, e.g. .../configs/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml")
    ap.add_argument("--confidence", type=float, default=0.4)
    ap.add_argument("--max-det", type=int, default=1000, help="max detections per tile (raise for dense tissue)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--channel-order", default="opt_mem_nuc",
                    choices=["opt_mem_nuc", "nuc_mem_opt", "mem_nuc_opt"],
                    help="3-channel layout; must match CelloType pretraining (verify against example_tissuenet.png)")
    ap.add_argument("--infer", choices=["tile", "whole"], default="tile")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=128, help="tile overlap; >= max cell diameter (~125 px)")
    # common
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--boundary-mode", default="thick")
    ap.add_argument("--out-dir", default="yourpath")
    ap.add_argument("--reuse-pred", action="store_true", help="reuse saved CelloType masks (e.g. official SAHI output), skip inference")
    ap.add_argument("--pred-dir", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # locate test.py
    test_py = args.test_py
    if not test_py:
        for _d in (args.dep_dir, _dep_dir, HERE, os.getcwd()):
            if _d and os.path.exists(os.path.join(_d, "test.py")):
                test_py = os.path.join(_d, "test.py")
                break
        test_py = test_py or os.path.join(HERE, "test.py")
    bench = single.load_metric_module(test_py)

    fov_list = args.fovs if args.fovs else list(range(args.fov_start, args.fov_end + 1))
    print(f"[plan] CelloType (TissueNet pretrained, no fine-tuning) | infer={args.infer} "
          f"channel_order={args.channel_order} | {len(fov_list)} FOVs: {fov_list}")

    runner = None
    if not args.reuse_pred:
        runner = CelloTypeRunner(
            model_path=args.model_path, config_path=args.config_path,
            confidence=args.confidence, max_det=args.max_det, device=args.device,
            infer=args.infer, tile=args.tile, overlap=args.overlap)

    per_fov, raws, failed = [], [], []
    for fov in fov_list:
        print(f"\n===== FOV {fov} =====")
        try:
            final, raw = process_one_fov(bench, args, fov, runner)
        except Exception as e:
            print(f"  [skip] FOV {fov} failed: {e}")
            failed.append((fov, str(e)))
            continue
        per_fov.append(final)
        raws.append(raw)
        print(f"  gt={final['gt_cell_count']} pred={final['pred_cell_count']} "
              f"F1={final['instance_f1']:.3f} AJI={final['AJI']:.3f} PQ={final['PQ']:.3f}")

    if not per_fov:
        raise SystemExit("No FOV succeeded. Check paths / file naming / dependency install.")

    import pandas as pd
    df = pd.DataFrame(per_fov)
    metric_cols = ["dice", "iou", "pixel_accuracy", "boundary_f1", "boundary_precision",
                   "boundary_recall", "instance_f1", "precision", "recall",
                   "DQ", "SQ", "PQ", "AJI", "mDA_0.50_0.95"]
    front = ["fov", "gt_cell_count", "pred_cell_count", "TP", "FP", "FN"]
    df = df[[c for c in front if c in df] + [c for c in metric_cols if c in df]]
    per_fov_csv = os.path.join(args.out_dir, "per_fov_metrics_cellotype.csv")
    df.to_csv(per_fov_csv, index=False)

    micro = agg.aggregate_micro(raws)
    total_gt = int(sum(f["gt_cell_count"] for f in per_fov))
    total_pred = int(sum(f["pred_cell_count"] for f in per_fov))
    macro_mean = {c: float(df[c].mean()) for c in metric_cols if c in df}
    macro_std = {c: float(df[c].std(ddof=1)) if len(df) > 1 else 0.0
                 for c in metric_cols if c in df}

    sdf = pd.DataFrame([dict(metric=c,
                             micro=round(micro.get(c, float("nan")), 6),
                             macro_mean=round(macro_mean.get(c, float("nan")), 6),
                             macro_std=round(macro_std.get(c, float("nan")), 6))
                        for c in metric_cols])
    summary_csv = os.path.join(args.out_dir, "aggregate_summary_cellotype.csv")
    sdf.to_csv(summary_csv, index=False)

    meta = dict(method="CelloType", model="TissueNet_pretrained", finetuned=False,
                infer=args.infer, channel_order=args.channel_order,
                model_path=args.model_path, config_path=args.config_path,
                n_fov_ok=len(per_fov), n_fov_failed=len(failed), failed=failed,
                fov_list=fov_list, iou_threshold=args.iou_threshold,
                total_gt_cells=total_gt, total_pred_cells=total_pred,
                micro_TP=micro["TP"], micro_FP=micro["FP"], micro_FN=micro["FN"],
                note="micro=pooled (primary; AJI/PQ must use it); macro=per-FOV mean+/-std; "
                     "zero-shot, whole-cell GT comparable to Cellpose/StarDist; pixel_accuracy is a weak metric.")
    with open(os.path.join(args.out_dir, "run_meta_cellotype.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print(f"CelloType (TissueNet, zero-shot) | infer={args.infer}")
    print(f"succeeded {len(per_fov)}/{len(fov_list)} FOVs | GT={total_gt} pred={total_pred}")
    if failed:
        print(f"failed FOVs: {[f for f, _ in failed]}")
    print(f"micro counts: TP={micro['TP']} FP={micro['FP']} FN={micro['FN']}")
    print("-" * 78)
    show = sdf.copy()
    show["macro(mean+/-std)"] = [f"{m:.4f}+/-{s:.4f}"
                               for m, s in zip(show["macro_mean"], show["macro_std"])]
    print(show[["metric", "micro", "macro(mean+/-std)"]].to_string(index=False))
    print("-" * 78)
    print("Report micro as primary; macro (mean+/-std) for robustness. AJI/PQ use micro only.")
    print("CelloType is natively whole-cell and zero-shot; directly comparable to Cellpose/StarDist (cell-level GT).")
    print(f"\n[saved]\n  {per_fov_csv}\n  {summary_csv}\n  "
          f"{os.path.join(args.out_dir, 'run_meta_cellotype.json')}")


if __name__ == "__main__":
    main()

'''
nohup bash -c '
    python CosMx_cellotype.py \
    --dep-dir yourpath \
    --data-root yourpath --fov-start 51 --fov-end 68 \
    --model-path yourpath \
    --config-path yourpath \
    --test-py yourpath \
    --out-dir yourpath
' > output.log 2>&1 &
'''