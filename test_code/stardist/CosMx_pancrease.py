#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stardist_benchmark_cosmx_multi.py
=================================
在多个 FOV（默认 51~68）上用 StarDist 做分割，并对 CosMx CellLabels(GT) 做 benchmark，
汇总方式与 cellpose 版完全一致：micro（池化）为主口径，macro(mean±std) 为稳健性口径。
指标实现复用 \eval\\test.py；聚合/组件函数直接复用 cellpose_benchmark_cosmx_multi
（那套 micro 聚合已用“拼大图金标准”验证过，差 ~1e-7）。

关键口径问题
--------------------------------
StarDist 的预训练模型（'2D_versatile_fluo' 等）是【细胞核】分割模型，
而 CellLabels 是【整细胞】分割。核比整细胞小，直接拿核比整细胞会让
IoU/Dice/AJI 系统性偏低——那是“分割对象不同”，不是 StarDist 差。
因此本脚本提供两种正确、可选的评估口径（--target）：

  --target cell   （默认，和 Cellpose 那次的 GT 一致，方便多方法横向比较）
      StarDist 出核 → 用 skimage.expand_labels 把核膨胀 --expand-px 像素近似整细胞
      → 对 CellLabels(整细胞) 评分。
      代价：膨胀距离是一个人为超参（默认 15px≈1.8µm）。这是“StarDist+膨胀”的联合表现，
      与很多真实 CosMx 管线(核分割+扩边)一致，但要在论文里如实写明膨胀这一步。

  --target nuclei （对 StarDist 最公平、无膨胀超参）
      StarDist 出核 → 对“核 GT”评分；核 GT = CellLabels ∩ (CompartmentLabels==核编码)，
      即每个细胞的核区域、继承该细胞 ID。
      注意：CompartmentLabels 是“区室类型”标签(核/膜/胞质/胞外)，不是实例；
      其整数编码 README 没写死，请用 --nuclear-code 指定并核对（脚本会打印实际出现的值）。

建议：论文里两种都报——cell 口径用于和 Cellpose/其它方法在同一 GT 上横向比较；
nuclei 口径用于公平体现 StarDist 本身的核分割能力。

依赖
----
    pip install stardist tensorflow csbdeep
首次运行会联网下载预训练权重（AutoDL 上正常）。4256² 大图建议给 --n-tiles（如 8 8）。

用法
----
    # 与 Cellpose 同 GT（整细胞），可直接和之前的结果横向比
    python stardist_benchmark_cosmx_multi.py --data-root /path/to/CosMx_Pancreas \
        --fov-start 51 --fov-end 68 --target cell --expand-px 15 \
        --out-dir ./stardist_bench_multi

    # 对 StarDist 最公平的核口径
    python stardist_benchmark_cosmx_multi.py --data-root ... \
        --fov-start 51 --fov-end 68 --target nuclei --nuclear-code 1

    # 复用已存的 StarDist 预测 tif，跳过推理
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
    找到 cellpose_benchmark_cosmx[_multi].py 所在目录。优先级：
      1) 环境变量 COSMX_BENCH_DIR
      2) 命令行 --dep-dir（在 argparse 之前手动扫一遍，因为 import 发生在最前）
      3) 兜底：本脚本目录、当前工作目录
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


# 把依赖目录（及兜底目录）加入 sys.path，再导入
_dep_dir = _resolve_dep_dir()
for _d in (_dep_dir, HERE, os.getcwd()):
    if _d and _d not in sys.path:
        sys.path.insert(0, _d)

try:
    import CosMx_single_fov as single      # 文件解析 / composite 加载 / 指标模块
    import CosMx_benchplus as agg    # 已验证的 micro 聚合 + 组件函数
except ImportError as _e:
    sys.stderr.write(
        "\n[导入失败] 找不到 cellpose_benchmark_cosmx.py / cellpose_benchmark_cosmx_multi.py。\n"
        "请用以下任一方式告诉脚本它们在哪里（三选一）：\n"
        "  1) 命令行加参数：   --dep-dir /path/to/cellpose_files\n"
        "  2) 设环境变量：     export COSMX_BENCH_DIR=/path/to/cellpose_files\n"
        "  3) 把两个 cellpose 脚本和本脚本放到同一目录\n"
        f"当前已尝试的目录：{[d for d in (_dep_dir, HERE, os.getcwd()) if d]}\n"
        f"原始错误：{_e}\n\n")
    raise


# --------------------------------------------------------------------------- #
# StarDist 输入 & 推理
# --------------------------------------------------------------------------- #
def build_stardist_input(composite_rgb):
    """StarDist(fluo) 是核模型，输入用 DAPI(Blue) 单通道即可。"""
    dapi = composite_rgb[..., 2].astype(np.float32)   # R,G,B -> B=DAPI
    return dapi


def run_stardist(dapi2d, model_name="2D_versatile_fluo",
                 prob_thresh=None, nms_thresh=None, n_tiles=None):
    """返回核实例 label mask（int32）。"""
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize
    print(f"[stardist] model={model_name} | prob={prob_thresh} nms={nms_thresh} "
          f"n_tiles={n_tiles}")
    model = StarDist2D.from_pretrained(model_name)
    x = normalize(dapi2d, 1, 99.8, axis=(0, 1))       # StarDist 官方推荐归一化
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
    """核 → 近似整细胞：expand_labels 各向外膨胀 expand_px，遇到相邻 label 即停。"""
    if not expand_px or expand_px <= 0:
        return nuclei_labels
    from skimage.segmentation import expand_labels
    return expand_labels(nuclei_labels, distance=expand_px).astype(nuclei_labels.dtype)


# --------------------------------------------------------------------------- #
# GT 构造：整细胞 或 核
# --------------------------------------------------------------------------- #
def load_gt_cell(bench, labels_path):
    return bench.read_tif_mask(labels_path)


def resolve_compartment(data_root, fov, comp_dir=None):
    comp_dir = comp_dir or os.path.join(data_root, "CompartmentLabels")
    return single._find_fov_file(comp_dir, "CompartmentLabels", fov,
                                 exts=[".tif", ".tiff", ".png", ".jpg"])


def load_gt_nuclei(bench, cell_labels_path, comp_labels_path, nuclear_code):
    """核 GT = CellLabels 在“核区室”内的部分（继承细胞 ID）。"""
    cl = bench.read_tif_mask(cell_labels_path)
    cp = bench.read_tif_mask(comp_labels_path)
    if cl.shape != cp.shape:
        from skimage.transform import resize
        cp = resize(cp, cl.shape, order=0, preserve_range=True,
                    anti_aliasing=False).astype(cl.dtype)
    uniq = np.unique(cp)
    print(f"  [compartment] 出现的区室编码 = {uniq.tolist()}  "
          f"(当前 --nuclear-code={nuclear_code}；请核对哪个是‘核’)")
    nuc_gt = np.where(cp == nuclear_code, cl, 0).astype(cl.dtype)
    return nuc_gt


# --------------------------------------------------------------------------- #
# 单 FOV
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
                f"target=nuclei 需要 CompartmentLabels，但 FOV {fov} 未找到。")
        gt = load_gt_nuclei(bench, lab_path, cpath, args.nuclear_code)
    else:  # cell
        gt = load_gt_cell(bench, lab_path)

    # ---- 预测 ----
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
            print(f"  [warn] 保存预测tif失败: {e}")

    # ---- 对齐 ----
    if gt.shape != pred.shape:
        from skimage.transform import resize
        print(f"  [warn] shape 不一致 gt={gt.shape} pred={pred.shape}，最近邻缩放 pred")
        pred = resize(pred, gt.shape, order=0, preserve_range=True,
                      anti_aliasing=False).astype(pred.dtype)

    # ---- 一次 overlaps，派生所有可累加量（复用已验证的组件函数）----
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
    ap = argparse.ArgumentParser(description="多FOV StarDist vs CellLabels benchmark 汇总")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dep-dir", default=None,
                    help="cellpose_benchmark_cosmx[_multi].py 所在目录（若与本脚本不同目录）。"
                         "等价于设 env COSMX_BENCH_DIR。")
    ap.add_argument("--fov-start", type=int, default=51)
    ap.add_argument("--fov-end", type=int, default=68)
    ap.add_argument("--fovs", type=int, nargs="*", default=None)
    ap.add_argument("--composite-dir", default=None)
    ap.add_argument("--labels-dir", default=None)
    ap.add_argument("--compartment-dir", default=None)
    ap.add_argument("--test-py", default=None,
                    help="test.py 路径；默认在 --dep-dir / 本脚本目录 里找")
    # StarDist
    ap.add_argument("--model", default="2D_versatile_fluo",
                    help="2D_versatile_fluo(荧光核,默认) / 2D_paper_dsb2018 / 2D_versatile_he")
    ap.add_argument("--prob-thresh", type=float, default=None, help="None=用模型默认(~0.479)")
    ap.add_argument("--nms-thresh", type=float, default=None)
    ap.add_argument("--n-tiles", type=int, nargs=2, default=None,
                    help="大图分块，如 --n-tiles 8 8；不给则自动猜")
    # 评估口径
    ap.add_argument("--target", choices=["cell", "nuclei"], default="cell",
                    help="cell=膨胀成整细胞对CellLabels比(默认,可与Cellpose横向比); "
                         "nuclei=对核GT比(对StarDist最公平)")
    ap.add_argument("--expand-px", type=int, default=15,
                    help="target=cell 时核→细胞的膨胀像素(≈1.8µm@0.12028)。是人为超参，请报告")
    ap.add_argument("--nuclear-code", type=int, default=1,
                    help="target=nuclei 时 CompartmentLabels 中‘核’的整数编码(请务必核对)")
    # 通用
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--boundary-mode", default="thick")
    ap.add_argument("--out-dir", default="./stardist_bench_multi")
    ap.add_argument("--reuse-pred", action="store_true")
    ap.add_argument("--pred-dir", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 解析 test.py：显式给的优先，否则在 dep-dir / 本脚本目录 里找
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
          + f" | 共 {len(fov_list)} 个 FOV: {fov_list}")

    per_fov, raws, failed = [], [], []
    for fov in fov_list:
        print(f"\n===== FOV {fov} =====")
        try:
            final, raw = process_one_fov(bench, args, fov)
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
    tag = f"{args.target}" + (f"_exp{args.expand_px}" if args.target == "cell" else "")
    per_fov_csv = os.path.join(args.out_dir, f"per_fov_metrics_stardist_{tag}.csv")
    df.to_csv(per_fov_csv, index=False)

    # micro（池化，主口径）
    micro = agg.aggregate_micro(raws)
    total_gt = int(sum(f["gt_cell_count"] for f in per_fov))
    total_pred = int(sum(f["pred_cell_count"] for f in per_fov))

    # macro（逐 FOV mean±std）
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
                note="micro=池化(主口径,AJI/PQ必须用它); macro=逐FOV mean±std; "
                     "StarDist预训练是核模型,cell口径含膨胀超参; pixel_accuracy为弱指标。")
    with open(os.path.join(args.out_dir, f"run_meta_stardist_{tag}.json"),
              "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print(f"StarDist | target={args.target}"
          + (f" expand={args.expand_px}px" if args.target == "cell" else
             f" nuclear_code={args.nuclear_code}"))
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
    print("StarDist 是核模型：cell 口径含 --expand-px 超参(需报告)；建议同时报 nuclei 口径。")
    print(f"\n[saved]\n  {per_fov_csv}\n  {summary_csv}\n  "
          f"{os.path.join(args.out_dir, f'run_meta_stardist_{tag}.json')}")


if __name__ == "__main__":
    main()


'''
nohup bash -c '
    python CosMx_stardist.py \
    --dep-dir /data/qiuyijia/cellpose \
    --data-root /data/qiuyijia/dataset/Pancreas-CosMx-WTx-FlatFiles --fov-start 51 --fov-end 68 \
    --target cell --expand-px 15 \
    --test-py /data/qiuyijia/eval/test.py \
    --out-dir /data/qiuyijia/stardist/CosMx_output
' > output.log 2>&1 &
'''