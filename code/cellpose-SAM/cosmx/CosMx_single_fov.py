#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cellpose_benchmark_cosmx.py
============================
在 NanoString CosMx SMI Pancreas 数据集的【单个 FOV】上完成：

  1) 用 Cellpose（默认 Cellpose-SAM / cpsam，与 test.py 里的 output_cpsam_masks 对齐）
     对 CellComposite 形态学图做【全细胞实例分割】；
  2) 以官方 CellLabels 掩膜作为 Ground Truth（GT）；
  3) 直接复用 test.py 里的指标实现，输出一整套 benchmark：
     pixel 级（Dice / IoU / pixel_acc / boundary-F1）
     instance 级（F1 / P / R / DQ / SQ / PQ / over-&under-seg）
     AJI、mDA(0.50:0.95)。

设计要点
--------
* GT 是 CellLabels（整细胞分割），因此 Cellpose 必须跑“整细胞”而不是“核”，
  否则核比整细胞小，IoU 会系统性偏低——那是口径不匹配，不是真实误差。
* Cellpose 输入用 CellComposite 里的：
    - 细胞质/膜通道  = max(Red=CD45/GFAP, Green=PanCK)   （尽量覆盖上皮+免疫+基质）
    - 细胞核通道      = Blue = DAPI
* CellLabels 与 CellComposite 都是 FOV 原生 4256×4256、0.12028 µm/px，理论同尺寸；
  代码仍会做 shape 校验，不一致时用最近邻把预测掩膜缩放回 GT 尺寸再评估。

重要口径说明（写论文时请如实描述）
----------------------------------
CellLabels 本身是 NanoString 自带分割算法的输出，不是人工逐细胞标注。
所以这里的 benchmark 严格来讲是“Cellpose 与厂商分割管线的一致性”，属于
silver-standard / reference-agreement，而非绝对精度。这一点和 test.py 原本
“拿 Cellpose 当参考去比 StarDist/CellSAM/Mesmer”在认识论上完全对称，
只是把参考换成了 CellLabels、被测方法换成了 Cellpose。

用法
----
    # 指定 FOV
    python cellpose_benchmark_cosmx.py --data-root /path/to/CosMx_Pancreas --fov 51

    # 不指定 FOV：从 metadata 自动挑“细胞数最多 / 最密”的 FOV 做压力测试
    python cellpose_benchmark_cosmx.py --data-root /path/to/CosMx_Pancreas \
        --metadata /path/to/Pancreas_metadata_file.csv

    # 只对已有的预测掩膜重新评分（跳过 cellpose）
    python cellpose_benchmark_cosmx.py --data-root ... --fov 51 \
        --pred-mask /some/pred_masks.tif --skip-cellpose

目录约定（CosMx 标准导出）
--------------------------
    <data-root>/
        CellComposite/  CellComposite_F051.jpg   (或 .png/.tif)
        CellLabels/      CellLabels_F051.tif
        (可选) CompartmentLabels/ ...
        (可选) Pancreas_metadata_file.csv
FOV 编号在文件名里一般补零到 3 位（F051）。本脚本会做灵活匹配，兼容不补零/补零。
"""

import os
import re
import sys
import glob
import argparse
import numpy as np


# --------------------------------------------------------------------------- #
# 0. 载入 test.py 里的指标实现（复用，保证口径与既有 pipeline 完全一致）
# --------------------------------------------------------------------------- #
def load_metric_module(test_py_path):
    """把 test.py 作为模块导入，拿到里面的指标函数。"""
    test_py_path = os.path.abspath(test_py_path)
    mod_dir = os.path.dirname(test_py_path)
    mod_name = os.path.splitext(os.path.basename(test_py_path))[0]
    if mod_dir not in sys.path:
        sys.path.insert(0, mod_dir)
    try:
        bench = __import__(mod_name)
    except Exception as e:
        raise ImportError(
            f"无法从 {test_py_path} 导入指标函数：{e}\n"
            f"请用 --test-py 指向你的 test.py。"
        )
    required = [
        "read_tif_mask", "compute_pixel_metrics", "compute_sparse_overlaps",
        "instance_metrics_from_overlaps", "aji_from_overlaps",
        "mean_detection_accuracy_from_overlaps",
    ]
    missing = [f for f in required if not hasattr(bench, f)]
    if missing:
        raise ImportError(f"test.py 缺少函数：{missing}")
    return bench


# --------------------------------------------------------------------------- #
# 1. 文件解析：按 FOV 找 CellComposite / CellLabels
# --------------------------------------------------------------------------- #
def _find_fov_file(folder, prefix, fov, exts):
    """
    在 folder 里找形如 {prefix}_F{fov}.{ext} 的文件，FOV 支持补零/不补零。
    例如 prefix=CellLabels, fov=51 -> CellLabels_F051.tif / CellLabels_F51.tif ...
    """
    if not os.path.isdir(folder):
        return None
    # 生成候选的 FOV 字符串：F051、F51、F0051 等
    fov_variants = {f"F{fov}", f"F{fov:02d}", f"F{fov:03d}", f"F{fov:04d}"}
    candidates = []
    for ext in exts:
        candidates += glob.glob(os.path.join(folder, f"{prefix}_*{ext}"))
        candidates += glob.glob(os.path.join(folder, f"{prefix}_*{ext.upper()}"))
    for path in sorted(set(candidates)):
        name = os.path.basename(path)
        m = re.search(r"[Ff](\d+)", name)
        if not m:
            continue
        n = int(m.group(1))
        if n == fov or any(v.lstrip("F") == m.group(1) for v in fov_variants):
            return path
    # 兜底：直接按精确名字试
    for ext in exts:
        for v in fov_variants:
            p = os.path.join(folder, f"{prefix}_{v}{ext}")
            if os.path.exists(p):
                return p
    return None


def resolve_paths(data_root, fov, composite_dir=None, labels_dir=None):
    composite_dir = composite_dir or os.path.join(data_root, "CellComposite")
    labels_dir = labels_dir or os.path.join(data_root, "CellLabels")

    comp = _find_fov_file(composite_dir, "CellComposite", fov,
                          exts=[".jpg", ".jpeg", ".png", ".tif", ".tiff"])
    lab = _find_fov_file(labels_dir, "CellLabels", fov,
                         exts=[".tif", ".tiff", ".png"])
    if comp is None:
        raise FileNotFoundError(
            f"未找到 FOV {fov} 的 CellComposite（在 {composite_dir} 里）。")
    if lab is None:
        raise FileNotFoundError(
            f"未找到 FOV {fov} 的 CellLabels（在 {labels_dir} 里）。")
    return comp, lab


def auto_pick_fov(metadata_csv, fov_col="fov"):
    """从 metadata 里挑细胞数最多（最密）的 FOV，作为分割压力测试的代表 FOV。"""
    import pandas as pd
    df = pd.read_csv(metadata_csv, usecols=lambda c: c in (fov_col, "cell_ID", "cell"))
    if fov_col not in df.columns:
        raise ValueError(f"metadata 里没有列 '{fov_col}'")
    counts = df.groupby(fov_col).size().sort_values(ascending=False)
    best = int(counts.index[0])
    print(f"[auto-fov] 各 FOV 细胞数 Top5：\n{counts.head(5).to_string()}")
    print(f"[auto-fov] 选中最密 FOV = {best}（{int(counts.iloc[0])} 个细胞）")
    return best


# --------------------------------------------------------------------------- #
# 2. 构造 Cellpose 输入
# --------------------------------------------------------------------------- #
def load_composite(path):
    """读 CellComposite，返回 float32 的 (H,W,3) RGB，顺序 R,G,B。"""
    from skimage.io import imread
    img = imread(path)
    if img.ndim == 2:                       # 灰度兜底
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:                  # 带 alpha
        img = img[..., :3]
    return img.astype(np.float32)


def build_cellpose_input(composite_rgb, cyto_channel="maxmarker"):
    """
    CellComposite 伪彩：R=CD45/GFAP, G=PanCK, B=DAPI（黄色 CD68 = R+G 混合）。
    返回 (H,W,2) 的 [cyto, nucleus]。
      cyto_channel:
        'maxmarker' -> cyto = max(R, G)  （覆盖上皮 PanCK + 免疫 CD45 + CD68黄）
        'panck'     -> cyto = G          （只看上皮，胰腺腺泡最亮）
        'dapi_only' -> cyto = B 的副本   （极端兜底，几乎等于只用核）
    nucleus 恒为 B(DAPI)。
    """
    R, G, B = composite_rgb[..., 0], composite_rgb[..., 1], composite_rgb[..., 2]
    if cyto_channel == "panck":
        cyto = G
    elif cyto_channel == "dapi_only":
        cyto = B.copy()
    else:  # maxmarker
        cyto = np.maximum(R, G)
    nuc = B
    return np.stack([cyto, nuc], axis=-1).astype(np.float32)


# --------------------------------------------------------------------------- #
# 3. 跑 Cellpose（兼容 v4/cpsam 与 v3 经典模型）
# --------------------------------------------------------------------------- #
def run_cellpose(img2, model_name="cpsam", diameter=None, gpu=True,
                 flow_threshold=0.4, cellprob_threshold=0.0):
    """
    img2: (H,W,2) = [cyto, nucleus]
    返回 2D int32 label mask。
    """
    import cellpose
    from cellpose import models
    ver = getattr(cellpose, "version", "0")
    try:
        major = int(str(ver).split(".")[0])
    except Exception:
        major = 0
    print(f"[cellpose] version={ver} | model={model_name} | gpu={gpu} "
          f"| diameter={diameter}")

    eval_kwargs = dict(diameter=diameter,
                       flow_threshold=flow_threshold,
                       cellprob_threshold=cellprob_threshold)

    if major >= 4:
        # Cellpose-SAM：channel-agnostic，直接吃多通道；channel_axis 指最后一维
        try:
            model = models.CellposeModel(gpu=gpu, pretrained_model=model_name)
        except TypeError:
            model = models.CellposeModel(gpu=gpu)  # v4 默认即 cpsam
        res = model.eval(img2, channel_axis=2, normalize=True, **eval_kwargs)
    else:
        # v3 经典：用 channels=[cyto, nucleus]=[1,2]（1->数组第0通道, 2->第1通道）
        try:
            model = models.CellposeModel(gpu=gpu, model_type=model_name)
        except TypeError:
            model = models.Cellpose(gpu=gpu, model_type=model_name)
        res = model.eval(img2, channels=[1, 2], normalize=True, **eval_kwargs)

    masks = res[0]                       # 兼容 (masks,flows,styles[,diams])
    masks = np.asarray(masks)
    if masks.ndim > 2:
        masks = np.squeeze(masks)
    return masks.astype(np.int32, copy=False)


# --------------------------------------------------------------------------- #
# 4. 评估：GT=CellLabels, pred=Cellpose，复用 test.py 指标
# --------------------------------------------------------------------------- #
def evaluate(bench, gt_mask, pred_mask, method_name="Cellpose-SAM",
             boundary_mode="thick", iou_threshold=0.5):
    if gt_mask.shape != pred_mask.shape:
        from skimage.transform import resize
        print(f"[warn] shape 不一致 GT={gt_mask.shape} pred={pred_mask.shape}，"
              f"用最近邻把 pred 缩放到 GT 尺寸。")
        pred_mask = resize(pred_mask, gt_mask.shape, order=0,
                           preserve_range=True, anti_aliasing=False
                           ).astype(pred_mask.dtype)

    pixel = bench.compute_pixel_metrics(gt_mask, pred_mask,
                                        boundary_mode=boundary_mode)
    ov = bench.compute_sparse_overlaps(gt_mask, pred_mask)
    inst = bench.instance_metrics_from_overlaps(ov, threshold=iou_threshold)
    aji = bench.aji_from_overlaps(ov, gt_mask, pred_mask)
    mda = bench.mean_detection_accuracy_from_overlaps(ov)

    return {
        "reference": "CosMx_CellLabels(GT)",
        "method": method_name,
        "gt_cell_count": int(len(ov.gt_ids)),
        "pred_cell_count": int(len(ov.pred_ids)),
        **pixel,
        **inst,
        "AJI": float(aji),
        "mDA_0.50_0.95": float(mda),
        "mDA_note": "mean TP/(TP+FP+FN) over IoU 0.50:0.95 (非标准 AP)",
    }


# --------------------------------------------------------------------------- #
# 5. 可选：GT vs 预测 边界叠加可视化
# --------------------------------------------------------------------------- #
def save_overlay(composite_rgb, gt_mask, pred_mask, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from skimage.segmentation import find_boundaries
    except Exception as e:
        print(f"[overlay] 跳过（缺 matplotlib/skimage）：{e}")
        return
    base = composite_rgb.copy()
    base = base / (base.max() + 1e-8)
    gt_b = find_boundaries(gt_mask, mode="thick")
    pr_b = find_boundaries(pred_mask, mode="thick")
    vis = base.copy()
    vis[gt_b] = [0, 1, 0]     # GT 绿色
    vis[pr_b] = [1, 0, 1]     # pred 品红
    plt.figure(figsize=(12, 12))
    plt.imshow(vis)
    plt.title("Green = CellLabels(GT)   Magenta = Cellpose")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[overlay] 已保存 {out_png}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Cellpose 在 CosMx Pancreas 单 FOV 上的分割 + 对 CellLabels 的 benchmark")
    ap.add_argument("--data-root", required=True, help="CosMx 导出根目录")
    ap.add_argument("--fov", type=int, default=None, help="FOV 编号；不给则从 metadata 自动挑最密的")
    ap.add_argument("--metadata", default=None, help="Pancreas_metadata_file.csv（用于自动挑 FOV）")
    ap.add_argument("--composite-dir", default=None)
    ap.add_argument("--labels-dir", default=None)
    ap.add_argument("--test-py", default=os.path.join(os.path.dirname(__file__), "test.py"),
                    help="你的 test.py 路径（复用指标）")
    ap.add_argument("--model", default="cpsam",
                    help="cellpose 模型：cpsam(默认,与 test.py 对齐) / cyto3 / cyto2 ...")
    ap.add_argument("--cyto-channel", default="maxmarker",
                    choices=["maxmarker", "panck", "dapi_only"])
    ap.add_argument("--diameter", type=float, default=None,
                    help="细胞直径(px)；胰腺细胞约 85~125px（10~15µm/0.12028）。None=自动")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--flow-threshold", type=float, default=0.4)
    ap.add_argument("--cellprob-threshold", type=float, default=0.0)
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--boundary-mode", default="thick")
    ap.add_argument("--out-dir", default="./cellpose_bench_out")
    ap.add_argument("--skip-cellpose", action="store_true",
                    help="不跑 cellpose，直接用 --pred-mask 评分")
    ap.add_argument("--pred-mask", default=None, help="已有预测掩膜 tif（配合 --skip-cellpose）")
    ap.add_argument("--overlay", action="store_true", help="额外输出 GT/pred 边界叠加图")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    bench = load_metric_module(args.test_py)

    # 1) 决定 FOV
    fov = args.fov
    if fov is None:
        md = args.metadata or os.path.join(args.data_root, "Pancreas_metadata_file.csv")
        if not os.path.exists(md):
            raise SystemExit("未指定 --fov 且找不到 metadata，无法自动选 FOV。")
        fov = auto_pick_fov(md)

    # 2) 找文件
    comp_path, lab_path = resolve_paths(
        args.data_root, fov, args.composite_dir, args.labels_dir)
    print(f"[io] FOV={fov}\n     composite = {comp_path}\n     labels    = {lab_path}")

    # 3) GT
    gt_mask = bench.read_tif_mask(lab_path)
    print(f"[gt] CellLabels shape={gt_mask.shape}, 细胞数={len(bench.get_instance_ids(gt_mask))}")

    # 4) 预测掩膜
    composite = load_composite(comp_path)
    if args.skip_cellpose:
        if not args.pred_mask:
            raise SystemExit("--skip-cellpose 需要配合 --pred-mask")
        pred_mask = bench.read_tif_mask(args.pred_mask)
        method_name = f"pred:{os.path.basename(args.pred_mask)}"
    else:
        cp_in = build_cellpose_input(composite, cyto_channel=args.cyto_channel)
        pred_mask = run_cellpose(
            cp_in, model_name=args.model, diameter=args.diameter,
            gpu=(not args.no_gpu),
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold)
        method_name = f"Cellpose({args.model})"
        # 保存预测掩膜（tif，可直接喂回 test.py 的 pipeline）
        try:
            import tifffile
            pred_tif = os.path.join(args.out_dir, f"cellpose_{args.model}_F{fov:03d}_masks.tif")
            tifffile.imwrite(pred_tif, pred_mask.astype(np.int32))
            print(f"[io] 预测掩膜已保存 {pred_tif}")
        except Exception as e:
            print(f"[warn] 保存 tif 失败（缺 tifffile？）：{e}")

    print(f"[pred] shape={pred_mask.shape}, 细胞数={len(bench.get_instance_ids(pred_mask))}")

    # 5) 评估
    result = evaluate(bench, gt_mask, pred_mask, method_name=method_name,
                      boundary_mode=args.boundary_mode,
                      iou_threshold=args.iou_threshold)
    result["fov"] = fov
    result["cyto_channel"] = args.cyto_channel

    # 6) 输出
    import pandas as pd
    df = pd.DataFrame([result])
    out_csv = os.path.join(args.out_dir, f"benchmark_cellpose_vs_CellLabels_F{fov:03d}.csv")
    df.to_csv(out_csv, index=False)

    print("\n================ Benchmark (GT=CellLabels, method=Cellpose) ================")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df.T)
    print(f"\n[done] 结果已保存 {out_csv}")

    if args.overlay:
        save_overlay(composite, gt_mask, pred_mask,
                     os.path.join(args.out_dir, f"overlay_F{fov:03d}.png"))


if __name__ == "__main__":
    main()