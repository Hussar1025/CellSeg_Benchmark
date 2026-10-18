#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cellsam_benchmark_cosmx_multi.py
================================
在多个 FOV（默认 51~68）上用 CellSAM 的【通用基础模型，零样本，不训练】做整细胞分割，
并对 CosMx CellLabels(GT) 做 benchmark。汇总方式与 cellpose / stardist / cellotype 版完全
一致：micro（池化）为主口径，macro(mean±std) 为稳健性口径；指标复用 test.py；micro 聚合
复用 cellpose_benchmark_cosmx_multi（已用“拼大图金标准”验证过，差 ~1e-7）。

关于 CellSAM
------------
CellSAM = CellFinder(检测框) + SAM(mask decoder) 的通用基础模型，零样本即可跨模态/组织分割，
原生做整细胞分割 → 直接对整细胞 CellLabels 比，无需 StarDist 那种膨胀，和 Cellpose/CelloType
同口径可横向比。因此和前面一样：不训练、不微调（公平性 + 避免同片 FOV 泄漏）。

推理
----
官方推荐接口 cellsam_pipeline()，自动下载权重、内置归一化/对比度/后处理；其 use_wsi=True 即
whole-slide 切块模式，用来处理 4256² 大 FOV（相当于 cellpose 的 n_tiles / cellotype 的 SAHI）。
本脚本默认走 cellsam_pipeline(use_wsi=True)；旧版本若无此接口，回退到 segment_cellular_image。
也可用官方脚本先出掩膜存成 cellsam_*_F0XX_masks.tif，再 --reuse-pred 只评分。

依赖
----
    pip install git+https://github.com/vanvalenlab/cellSAM.git   (python>=3.10, 纯 PyTorch)
首次运行会下载模型权重（可能需要 deepcell/users.deepcell.org 的访问 token，按官方提示配置）。

输入通道顺序
CosMx composite: R=CD45/GFAP, G=PanCK, B=DAPI。本脚本构造 [核=DAPI, 膜=max(CD45,PanCK)]。
CellSAM 对多通道荧光的核/膜通道顺序请对照官方示例确认，用 --channel-order 调整；放反会掉点。

用法
----
    python cellsam_benchmark_cosmx_multi.py \
        --dep-dir /path/to/cellpose_files \
        --data-root /path/to/CosMx_Pancreas --fov-start 51 --fov-end 68 \
        --device cuda --use-wsi --out-dir ./cellsam_bench_multi

    # 用官方脚本先出掩膜，这里只评分
    python cellsam_benchmark_cosmx_multi.py --dep-dir ... --data-root ... \
        --fov-start 51 --fov-end 68 --reuse-pred --pred-dir /path/to/cellsam_masks
"""

import os
import sys
import json
import glob
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# 依赖定位（cellpose_benchmark_cosmx[_multi].py 可在别的目录）
# --------------------------------------------------------------------------- #
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
    import CosMx_single_fov as single       # 文件解析 / composite 加载 / 指标模块
    import CosMx_benchplus as agg      # 已验证的 micro 聚合 + 组件函数
except ImportError as _e:
    sys.stderr.write(
        "\n[导入失败] 找不到 CosMx_single_fov.py / CosMx_benchplus.py。\n"
        "请用以下任一方式指定它们所在目录（三选一）：\n"
        "  1) 命令行：  --dep-dir /path/to/cellpose_files\n"
        "  2) 环境变量：export COSMX_BENCH_DIR=/path/to/cellpose_files\n"
        "  3) 把这些脚本放同一目录\n"
        f"当前已尝试：{[d for d in (_dep_dir, HERE, os.getcwd()) if d]}\n原始错误：{_e}\n\n")
    raise


# --------------------------------------------------------------------------- #
# CellSAM 输入（核 + 膜）
# --------------------------------------------------------------------------- #
def build_cellsam_input(composite_rgb, channel_order="nuc_mem"):
    """
    CosMx composite: R=CD45/GFAP, G=PanCK, B=DAPI。
      核  = DAPI = B
      膜  = max(CD45, PanCK) = max(R,G)
    channel_order:
      nuc_mem       -> (H,W,2) = [核, 膜]      (默认)
      mem_nuc       -> (H,W,2) = [膜, 核]
      blank_nuc_mem -> (H,W,3) = [0, 核, 膜]
      nuc_only      -> (H,W)   仅 DAPI（不确定通道约定时的保守回退）
    CellSAM 对核/膜通道顺序的约定请对照官方示例核对；放反会掉点。
    返回 float32（cellsam_pipeline 内部会归一化）。
    """
    R = composite_rgb[..., 0].astype(np.float32)
    G = composite_rgb[..., 1].astype(np.float32)
    B = composite_rgb[..., 2].astype(np.float32)
    nuclear = B
    membrane = np.maximum(R, G)
    zero = np.zeros_like(B)
    if channel_order == "mem_nuc":
        return np.stack([membrane, nuclear], axis=-1)
    if channel_order == "blank_nuc_mem":
        return np.stack([zero, nuclear, membrane], axis=-1)
    if channel_order == "nuc_only":
        return nuclear
    return np.stack([nuclear, membrane], axis=-1)   # nuc_mem


# --------------------------------------------------------------------------- #
# CellSAM 运行器（权重只加载一次）
# --------------------------------------------------------------------------- #
class CellSAMRunner:
    def __init__(self, device="cuda", use_wsi=True,
                 low_contrast_enhancement=False, gauge_cell_size=False):
        self.device = device
        self.use_wsi = use_wsi
        self.low_contrast_enhancement = low_contrast_enhancement
        self.gauge_cell_size = gauge_cell_size
        self._pipeline = None
        self._seg_fn = None

    def _ensure_loaded(self):
        if self._pipeline is None and self._seg_fn is None:
            try:
                from cellSAM import cellsam_pipeline
                self._pipeline = cellsam_pipeline
                print(f"[cellsam] 使用 cellsam_pipeline(use_wsi={self.use_wsi}) | device={self.device}")
            except Exception:
                from cellSAM import segment_cellular_image
                self._seg_fn = segment_cellular_image
                print("[cellsam] 旧版本：回退到 segment_cellular_image（不切块，大图可能吃力）")

    def segment(self, img):
        self._ensure_loaded()
        if self._pipeline is not None:
            try:
                mask = self._pipeline(
                    img, use_wsi=self.use_wsi,
                    low_contrast_enhancement=self.low_contrast_enhancement,
                    gauge_cell_size=self.gauge_cell_size, device=self.device)
            except TypeError:
                # 某些版本 device 由内部自动检测，或参数名不同
                mask = self._pipeline(img, use_wsi=self.use_wsi)
        else:
            out = self._seg_fn(img, device=self.device)
            mask = out[0] if isinstance(out, (tuple, list)) else out
        mask = np.asarray(mask)
        if mask.ndim > 2:
            mask = np.squeeze(mask)
        return mask.astype(np.int32, copy=False)


# --------------------------------------------------------------------------- #
# 单 FOV
# --------------------------------------------------------------------------- #
def process_one_fov(bench, args, fov, runner):
    comp_path, lab_path = single.resolve_paths(
        args.data_root, fov, args.composite_dir, args.labels_dir)
    composite = single.load_composite(comp_path)
    gt = bench.read_tif_mask(lab_path)             # 整细胞 GT

    pred = None
    if args.reuse_pred:
        pat = os.path.join(args.pred_dir or args.out_dir, f"*cellsam*F{fov:03d}*.tif")
        hits = sorted(glob.glob(pat)) or sorted(
            glob.glob(os.path.join(args.pred_dir or args.out_dir, f"*F{fov:03d}*.tif")))
        if hits:
            pred = bench.read_tif_mask(hits[0])
            print(f"  [reuse] {os.path.basename(hits[0])}")
    if pred is None:
        cin = build_cellsam_input(composite, channel_order=args.channel_order)
        pred = runner.segment(cin)
        try:
            import tifffile
            os.makedirs(args.out_dir, exist_ok=True)
            outp = os.path.join(args.out_dir, f"cellsam_F{fov:03d}_masks.tif")
            tifffile.imwrite(outp, pred.astype(np.int32))
        except Exception as e:
            print(f"  [warn] 保存预测tif失败: {e}")

    if gt.shape != pred.shape:
        from skimage.transform import resize
        print(f"  [warn] shape 不一致 gt={gt.shape} pred={pred.shape}，最近邻缩放 pred")
        pred = resize(pred, gt.shape, order=0, preserve_range=True,
                      anti_aliasing=False).astype(pred.dtype)

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
    ap = argparse.ArgumentParser(description="多FOV CellSAM(零样本) vs CellLabels benchmark")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dep-dir", default=None,
                    help="cellpose_benchmark_cosmx[_multi].py 所在目录。等价于 env COSMX_BENCH_DIR。")
    ap.add_argument("--fov-start", type=int, default=51)
    ap.add_argument("--fov-end", type=int, default=68)
    ap.add_argument("--fovs", type=int, nargs="*", default=None)
    ap.add_argument("--composite-dir", default=None)
    ap.add_argument("--labels-dir", default=None)
    ap.add_argument("--test-py", default=None, help="test.py 路径；默认在 dep-dir/本脚本目录 里找")
    # CellSAM
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--use-wsi", dest="use_wsi", action="store_true", default=True,
                    help="whole-slide 切块模式（大 FOV 默认开）")
    ap.add_argument("--no-use-wsi", dest="use_wsi", action="store_false")
    ap.add_argument("--low-contrast-enhancement", action="store_true")
    ap.add_argument("--gauge-cell-size", action="store_true")
    ap.add_argument("--channel-order", default="blank_nuc_mem",
                    choices=["nuc_mem", "mem_nuc", "blank_nuc_mem", "nuc_only"],
                    help="核/膜通道排布，需与 CellSAM 约定一致（对照官方示例核对）")
    # 通用
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--boundary-mode", default="thick")
    ap.add_argument("--out-dir", default="./cellsam_bench_multi")
    ap.add_argument("--reuse-pred", action="store_true", help="复用已存的 CellSAM 掩膜，跳过推理")
    ap.add_argument("--pred-dir", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    test_py = args.test_py
    if not test_py:
        for _d in (args.dep_dir, _dep_dir, HERE, os.getcwd()):
            if _d and os.path.exists(os.path.join(_d, "test.py")):
                test_py = os.path.join(_d, "test.py")
                break
        test_py = test_py or os.path.join(HERE, "test.py")
    bench = single.load_metric_module(test_py)

    fov_list = args.fovs if args.fovs else list(range(args.fov_start, args.fov_end + 1))
    print(f"[plan] CellSAM(零样本) | use_wsi={args.use_wsi} channel_order={args.channel_order} "
          f"| 共 {len(fov_list)} 个 FOV: {fov_list}")

    runner = None
    if not args.reuse_pred:
        runner = CellSAMRunner(device=args.device, use_wsi=args.use_wsi,
                               low_contrast_enhancement=args.low_contrast_enhancement,
                               gauge_cell_size=args.gauge_cell_size)

    per_fov, raws, failed = [], [], []
    for fov in fov_list:
        print(f"\n===== FOV {fov} =====")
        try:
            final, raw = process_one_fov(bench, args, fov, runner)
        except Exception as e:
            print(f"  [skip] FOV {fov} 失败: {e}")
            failed.append((fov, str(e)))
            continue
        per_fov.append(final)
        raws.append(raw)
        print(f"  gt={final['gt_cell_count']} pred={final['pred_cell_count']} "
              f"F1={final['instance_f1']:.3f} AJI={final['AJI']:.3f} PQ={final['PQ']:.3f}")

    if not per_fov:
        raise SystemExit("没有任何 FOV 成功。请检查路径/文件命名/依赖安装。")

    import pandas as pd
    df = pd.DataFrame(per_fov)
    metric_cols = ["dice", "iou", "pixel_accuracy", "boundary_f1", "boundary_precision",
                   "boundary_recall", "instance_f1", "precision", "recall",
                   "DQ", "SQ", "PQ", "AJI", "mDA_0.50_0.95"]
    front = ["fov", "gt_cell_count", "pred_cell_count", "TP", "FP", "FN"]
    df = df[[c for c in front if c in df] + [c for c in metric_cols if c in df]]
    per_fov_csv = os.path.join(args.out_dir, "per_fov_metrics_cellsam.csv")
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
    summary_csv = os.path.join(args.out_dir, "aggregate_summary_cellsam.csv")
    sdf.to_csv(summary_csv, index=False)

    meta = dict(method="CellSAM", model="generalist_pretrained", finetuned=False,
                use_wsi=args.use_wsi, channel_order=args.channel_order,
                n_fov_ok=len(per_fov), n_fov_failed=len(failed), failed=failed,
                fov_list=fov_list, iou_threshold=args.iou_threshold,
                total_gt_cells=total_gt, total_pred_cells=total_pred,
                micro_TP=micro["TP"], micro_FP=micro["FP"], micro_FN=micro["FN"],
                note="micro=池化(主口径,AJI/PQ必须用它); macro=逐FOV mean±std; "
                     "零样本,整细胞GT,与Cellpose/StarDist(cell)/CelloType可横向比; pixel_accuracy弱指标。")
    with open(os.path.join(args.out_dir, "run_meta_cellsam.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print(f"CellSAM(零样本) | use_wsi={args.use_wsi}")
    print(f"成功 {len(per_fov)}/{len(fov_list)} 个 FOV | GT={total_gt} 预测={total_pred}")
    if failed:
        print(f"失败 FOV: {[f for f, _ in failed]}")
    print(f"micro 计数: TP={micro['TP']} FP={micro['FP']} FN={micro['FN']}")
    print("-" * 78)
    show = sdf.copy()
    show["macro(mean±std)"] = [f"{m:.4f}±{s:.4f}"
                               for m, s in zip(show["macro_mean"], show["macro_std"])]
    print(show[["metric", "micro", "macro(mean±std)"]].to_string(index=False))
    print("-" * 78)
    print("主体报 micro; 稳健性附 macro(mean±std)。AJI/PQ 只看 micro。")
    print("CellSAM 原生整细胞、零样本，与 Cellpose/StarDist(cell)/CelloType 同 GT 可直接横向比。")
    print(f"\n[saved]\n  {per_fov_csv}\n  {summary_csv}\n  "
          f"{os.path.join(args.out_dir, 'run_meta_cellsam.json')}")


if __name__ == "__main__":
    main()

'''
nohup bash -c '
    python CosMx_cellsam.py \
    --dep-dir /data/qiuyijia/cellpose \
    --data-root /data/qiuyijia/dataset/Pancreas-CosMx-WTx-FlatFiles --fov-start 51 --fov-end 68 \
    --test-py /data/qiuyijia/eval/test.py --use-wsi \
    --out-dir /data/qiuyijia/CellSam/CosMx_output
' > output.log 2>&1 &
'''