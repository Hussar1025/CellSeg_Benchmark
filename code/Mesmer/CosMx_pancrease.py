#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mesmer_benchmark_cosmx_multi.py
===============================
在多个 FOV（默认 51~68）上用 Mesmer（DeepCell，TissueNet 预训练，零样本，不训练）做整细胞
分割，并对 CosMx CellLabels(GT) 做 benchmark。汇总方式与 cellpose / stardist / cellotype /
cellsam 版完全一致：micro（池化）为主口径，macro(mean±std) 为稳健性口径；指标复用 test.py；
micro 聚合复用 cellpose_benchmark_cosmx_multi（已用“拼大图金标准”验证过，差 ~1e-7）。

关于 Mesmer（两个必须写对的点）
--------------------------------
1) 输入是【2 通道】[核, 膜]：第一通道核(DAPI)，第二通道膜/胞质。
   CosMx composite R=CD45/GFAP, G=PanCK, B=DAPI →
     核  = DAPI = B
     膜  = max(CD45, PanCK) = max(R,G)
   顺序固定：核在前、膜在后（--channel-order 可切，但默认就是 Mesmer 约定）。
2) image_mpp 必须填你数据的【真实分辨率】。Mesmer 训练分辨率 app.model_mpp=0.5µm/px，
   predict 会按 image_mpp 把图缩放到训练分辨率。CosMx 是 0.12028µm/px → 必须传 0.12028。
   别照抄别人给 TissueNet 用的 image_mpp=0.5！那会让 Mesmer 以为细胞大了~4倍而严重欠分割。
compartment='whole-cell' → 整细胞，直接对 CellLabels 比，和 Cellpose/CelloType/CellSAM 同口径。
零样本、不训练（公平性 + 避免同片 FOV 泄漏），理由同前。

依赖
----
    pip install deepcell            (deepcell-tf，基于 TensorFlow；建议按官方版本对齐 TF/CUDA)
首次运行自动下载 Mesmer 权重；较新版本可能需要 DeepCell API key（users.deepcell.org），
按提示设置环境变量 DEEPCELL_ACCESS_TOKEN。GPU 通过 CUDA_VISIBLE_DEVICES 选择。
也可用官方 deepcell-applications 先出掩膜存成 mesmer_*_F0XX_masks.tif，再 --reuse-pred 只评分。

用法
----
    python mesmer_benchmark_cosmx_multi.py \
        --dep-dir /path/to/cellpose_files \
        --data-root /path/to/CosMx_Pancreas --fov-start 51 --fov-end 68 \
        --image-mpp 0.12028 --compartment whole-cell \
        --out-dir ./mesmer_bench_multi

    # 用官方脚本先出掩膜，这里只评分
    python mesmer_benchmark_cosmx_multi.py --dep-dir ... --data-root ... \
        --fov-start 51 --fov-end 68 --reuse-pred --pred-dir /path/to/mesmer_masks
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
        "\n[导入失败] 找不到 cellpose_benchmark_cosmx.py / cellpose_benchmark_cosmx_multi.py。\n"
        "请用以下任一方式指定它们所在目录（三选一）：\n"
        "  1) 命令行：  --dep-dir /path/to/cellpose_files\n"
        "  2) 环境变量：export COSMX_BENCH_DIR=/path/to/cellpose_files\n"
        "  3) 把这些脚本放同一目录\n"
        f"当前已尝试：{[d for d in (_dep_dir, HERE, os.getcwd()) if d]}\n原始错误：{_e}\n\n")
    raise


# --------------------------------------------------------------------------- #
# Mesmer 输入：[1, H, W, 2] = [核, 膜]
# --------------------------------------------------------------------------- #
def build_mesmer_input(composite_rgb, channel_order="nuc_mem"):
    R = composite_rgb[..., 0].astype(np.float32)
    G = composite_rgb[..., 1].astype(np.float32)
    B = composite_rgb[..., 2].astype(np.float32)
    nuclear = B                       # DAPI
    membrane = np.maximum(R, G)       # max(CD45, PanCK)
    if channel_order == "mem_nuc":
        two = np.stack([membrane, nuclear], axis=-1)
    else:                             # nuc_mem（Mesmer 约定：核在前）
        two = np.stack([nuclear, membrane], axis=-1)
    return two[np.newaxis, ...].astype(np.float32)   # [1,H,W,2]


# --------------------------------------------------------------------------- #
# Mesmer 运行器（权重只加载一次）
# --------------------------------------------------------------------------- #
class MesmerRunner:
    def __init__(self, image_mpp=0.12028, compartment="whole-cell",
                 batch_size=4, pixel_expansion=0):
        self.image_mpp = image_mpp
        self.compartment = compartment
        self.batch_size = batch_size
        self.pixel_expansion = pixel_expansion
        self._app = None

    def _app_obj(self):
        if self._app is None:
            from deepcell.applications import Mesmer
            self._app = Mesmer()
            print(f"[mesmer] 已加载 | model_mpp(训练分辨率)={self._app.model_mpp} "
                  f"| 你的 image_mpp={self.image_mpp} | compartment={self.compartment}")
            if abs(self.image_mpp - 0.5) < 1e-6:
                print("  [warn] image_mpp=0.5 对 CosMx 很可能错！CosMx 约 0.12028，"
                      "填 0.5 会严重欠分割。确认这是你想要的。")
        return self._app

    def segment(self, X):
        app = self._app_obj()
        kwargs = dict(image_mpp=self.image_mpp, compartment=self.compartment,
                      batch_size=self.batch_size)
        if self.pixel_expansion and self.compartment == "nuclear":
            kwargs["postprocess_kwargs_nuclear"] = {"pixel_expansion": self.pixel_expansion}
        y = np.asarray(app.predict(X, **kwargs))
        y = np.squeeze(y)             # [1,H,W,1] -> [H,W]（whole-cell）
        if y.ndim > 2:                # compartment='both' -> [H,W,2]，取整细胞通道 0
            y = y[..., 0]
        return y.astype(np.int32, copy=False)


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
        pat = os.path.join(args.pred_dir or args.out_dir, f"*mesmer*F{fov:03d}*.tif")
        hits = sorted(glob.glob(pat)) or sorted(
            glob.glob(os.path.join(args.pred_dir or args.out_dir, f"*F{fov:03d}*.tif")))
        if hits:
            pred = bench.read_tif_mask(hits[0])
            print(f"  [reuse] {os.path.basename(hits[0])}")
    if pred is None:
        X = build_mesmer_input(composite, channel_order=args.channel_order)
        pred = runner.segment(X)
        try:
            import tifffile
            os.makedirs(args.out_dir, exist_ok=True)
            outp = os.path.join(args.out_dir, f"mesmer_{args.compartment}_F{fov:03d}_masks.tif")
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
    ap = argparse.ArgumentParser(description="多FOV Mesmer(TissueNet预训练,零样本) vs CellLabels benchmark")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dep-dir", default=None,
                    help="CosMx_single_fov.py 所在目录。等价于 env COSMX_BENCH_DIR。")
    ap.add_argument("--fov-start", type=int, default=51)
    ap.add_argument("--fov-end", type=int, default=68)
    ap.add_argument("--fovs", type=int, nargs="*", default=None)
    ap.add_argument("--composite-dir", default=None)
    ap.add_argument("--labels-dir", default=None)
    ap.add_argument("--test-py", default=None, help="test.py 路径；默认在 dep-dir/本脚本目录 里找")
    # Mesmer
    ap.add_argument("--image-mpp", type=float, default=0.12028,
                    help="你数据的真实分辨率(µm/px)。CosMx=0.12028。别填 0.5！")
    ap.add_argument("--compartment", default="whole-cell",
                    choices=["whole-cell", "nuclear", "both"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--pixel-expansion", type=int, default=0,
                    help="仅 compartment=nuclear 时生效，核后再膨胀 N 像素")
    ap.add_argument("--channel-order", default="nuc_mem", choices=["nuc_mem", "mem_nuc"],
                    help="Mesmer 约定核在前(nuc_mem)。除非确认数据反了，别改。")
    # 通用
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--boundary-mode", default="thick")
    ap.add_argument("--out-dir", default="./mesmer_bench_multi")
    ap.add_argument("--reuse-pred", action="store_true", help="复用已存的 Mesmer 掩膜，跳过推理")
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
    print(f"[plan] Mesmer(零样本) | image_mpp={args.image_mpp} compartment={args.compartment} "
          f"| 共 {len(fov_list)} 个 FOV: {fov_list}")

    runner = None
    if not args.reuse_pred:
        runner = MesmerRunner(image_mpp=args.image_mpp, compartment=args.compartment,
                              batch_size=args.batch_size, pixel_expansion=args.pixel_expansion)

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
    per_fov_csv = os.path.join(args.out_dir, "per_fov_metrics_mesmer.csv")
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
    summary_csv = os.path.join(args.out_dir, "aggregate_summary_mesmer.csv")
    sdf.to_csv(summary_csv, index=False)

    meta = dict(method="Mesmer", model="TissueNet_pretrained", finetuned=False,
                image_mpp=args.image_mpp, compartment=args.compartment,
                channel_order=args.channel_order,
                n_fov_ok=len(per_fov), n_fov_failed=len(failed), failed=failed,
                fov_list=fov_list, iou_threshold=args.iou_threshold,
                total_gt_cells=total_gt, total_pred_cells=total_pred,
                micro_TP=micro["TP"], micro_FP=micro["FP"], micro_FN=micro["FN"],
                note="micro=池化(主口径,AJI/PQ必须用它); macro=逐FOV mean±std; "
                     "零样本,整细胞GT,与Cellpose/StarDist(cell)/CelloType/CellSAM可横向比; "
                     "image_mpp须为数据真实分辨率(CosMx=0.12028); pixel_accuracy弱指标。")
    with open(os.path.join(args.out_dir, "run_meta_mesmer.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print(f"Mesmer(零样本) | image_mpp={args.image_mpp} compartment={args.compartment}")
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
    print("Mesmer 原生整细胞、零样本，与 Cellpose/StarDist(cell)/CelloType/CellSAM 同 GT 可直接横向比。")
    print("再次确认 image_mpp 用的是数据真实分辨率(CosMx=0.12028)，不是 0.5。")
    print(f"\n[saved]\n  {per_fov_csv}\n  {summary_csv}\n  "
          f"{os.path.join(args.out_dir, 'run_meta_mesmer.json')}")


if __name__ == "__main__":
    main()

'''
nohup bash -c '
    python CosMx_mesmer.py \
    --dep-dir /data/qiuyijia/cellpose \
    --data-root /data/qiuyijia/dataset/Pancreas-CosMx-WTx-FlatFiles --fov-start 51 --fov-end 68 \
    --test-py /data/qiuyijia/eval/test.py \
    --out-dir /data/qiuyijia/Mesmer/CosMx_output
' > output.log 2>&1 &
'''