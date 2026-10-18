#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Proseg 评估 —— 全绝对路径，运行只需 --dataset。复用参考评估脚本的指标/匹配/GT 逻辑。
用法:  python eval_proseg.py --dataset prostate1     (或 prostate2)
可选覆盖: --root --run-dir --gt-mask --out-dir --ref --roi-size --mask --seg-name
"""
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np

# ============ 绝对路径配置（换机器只改这几行）============
REF_DEFAULT   = "/data/qiuyijia/ucs/UCS/ucs_value_merfish_4.py"
DATA_BASE     = "/data/qiuyijia/dataset"        # 数据集: {DATA_BASE}/merfish_<ds>
RESULT_BASE   = "/data/qiuyijia"      # 结果:   {RESULT_BASE}/<prefix>_merfish_<ds>
METHOD        = "Proseg"
RESULT_PREFIX = "proseg"
SEG_PATTERNS  = ["proseg_segmentation.tif","*proseg*seg*.tif","*segmentation*.tif"]
USES_PRIOR    = False
ROI_SIZE_DEF  = 10000
# =======================================================

def _load_ref(ref):
    ref = Path(ref).resolve()
    if not ref.exists():
        raise SystemExit(f"参考评估脚本不存在: {ref}\n改顶部 REF_DEFAULT 或用 --ref 指定")
    spec = importlib.util.spec_from_file_location("refval", ref)
    V = importlib.util.module_from_spec(spec); sys.modules["refval"] = V
    spec.loader.exec_module(V)
    return V, ref

def _find_seg(run_dir, seg_name):
    d = Path(run_dir)
    if seg_name and (d / seg_name).exists():
        return d / seg_name
    for pat in SEG_PATTERNS:
        cand = sorted(d.glob(pat))
        if cand:
            return cand[0]
    raise SystemExit(f"在 {d} 找不到分割结果 (尝试 {SEG_PATTERNS})")

def _pick_gt_mask(run_dir, root, dataset):
    cands = [Path(run_dir) / "platform_labels.tif",
             Path(f"{RESULT_BASE}/proseg_merfish_{dataset}/platform_labels.tif") if dataset else None,
             Path(root) / "platform_labels.tif"]
    for c in cands:
        if c and c.exists():
            return str(c)
    return None  # 交给参考脚本自行在 root 下查找

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=REF_DEFAULT)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--gt-mask", default=None)
    ap.add_argument("--roi-size", type=int, default=ROI_SIZE_DEF)
    ap.add_argument("--mask", default=None)
    ap.add_argument("--seg-name", default=None)
    args = ap.parse_args()

    if args.dataset:
        args.root    = args.root    or f"{DATA_BASE}/merfish_{args.dataset}"
        args.run_dir = args.run_dir or f"{RESULT_BASE}/{RESULT_PREFIX}_merfish_{args.dataset}"
        args.out_dir = args.out_dir or f"{RESULT_BASE}/eval_merfish_{args.dataset}"
    if not (args.root and args.run_dir):
        raise SystemExit("需要 --dataset，或同时给 --root 和 --run-dir")
    args.out_dir = args.out_dir or f"{RESULT_BASE}/eval_merfish_out"
    if not args.gt_mask:
        args.gt_mask = _pick_gt_mask(args.run_dir, args.root, args.dataset)

    V, ref = _load_ref(args.ref)
    root, out = Path(args.root), Path(args.out_dir)
    V.setup_log(out, f"eval_{METHOD}_{args.dataset or 'x'}")
    V.rule(f"{METHOD}  x  MERFISH {args.dataset or ''}   (via {ref.name})")

    geo = V.load_geometry(root, args.roi_size)
    gt_mask = V.load_gt_mask(root, geo, args.gt_mask)
    tx = V.load_transcripts(root, geo, gt_mask)
    gt = V.load_ground_truth(root, geo, tx)

    import tifffile
    p = Path(args.mask) if args.mask else _find_seg(args.run_dir, args.seg_name)
    lab = np.squeeze(tifffile.imread(p)).astype(np.int64)
    V.log.info(f"mask        {p}   {lab.shape}")
    s = args.roi_size / lab.shape[0]
    pred = V.Prediction(lab, s, METHOD, uses_platform_prior=USES_PRIOR)

    V.evaluate(pred, root, out, geo, tx, gt)

if __name__ == "__main__":
    main()