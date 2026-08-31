#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cellpose_benchmark_cosmx_multi.py
=================================
在多个 FOV（默认 51~68 共 18 个）上批量跑 Cellpose 并对 CellLabels(GT) 做 benchmark，
然后【正确地】把 18 个 FOV 的结果汇总。

为什么不能直接对每个 FOV 的指标取算术平均？
------------------------------------------------
这些指标绝大多数是“比值/计数”类，直接平均(=macro 平均)会有两个问题：

1) 每个 FOV 权重相等 → 稀疏 FOV(几十个细胞)和稠密 FOV(上千个细胞)一样重，
   等于给稀疏 FOV 过高的话语权。通常我们想“每个细胞等权”，那应该用 micro 平均：
   先把各 FOV 的原始计数/面积【累加】，最后再算一次指标。

2) AJI 和 PQ 尤其不能逐 FOV 平均：
   - AJI 名字里的 "Aggregated" 就是指“把分子/分母各自求和再除”。因为不同 FOV 的
     细胞不会跨图重叠，全局最优匹配 = 各 FOV 匹配之并，所以
     micro-AJI = Σ(matched_inter) / Σ(matched_union + unmatched_area)
     严格等于“把所有 FOV 拼成一张大图算一次 AJI”（本脚本附带的等价性可自测）。
     逐 FOV 求 AJI 再平均，是近似、且系统性有偏。
   - PQ = DQ×SQ，DQ 是计数比、SQ 是所有匹配对 IoU 的均值，都应先池化原始量再算。

3) pixel_accuracy 背景占绝对多数(4256² 里细胞只占一部分)，天然虚高，
   无论怎么平均都不建议作为主指标；本脚本保留但打上“弱指标”提示。

因此本脚本的建议(也是输出)：
------------------------------------------------
* 主口径 = micro(池化)：把 TP/FP/FN、交并像素、AJI 分子分母、各 IoU 阈值的计数
  在 18 个 FOV 上累加，最后各算一次。这是发论文该报的“总体”数字。
* 辅口径 = macro(逐 FOV 均值 ± 标准差)：体现 FOV 间的一致性/波动，
  并可配合 per-FOV 明细表看是否有异常 FOV(离焦、细胞极稀等)。
* 同时保存 per-FOV 明细 CSV，方便画箱线图 / 挑异常 FOV。

一句话：不要只报一个“平均值”。报 micro 作为总体，macro(mean±std) 作为稳健性，
并且 AJI/PQ 必须用 micro。

用法
----
    python cellpose_benchmark_cosmx_multi.py \
        --data-root /path/to/CosMx_Pancreas \
        --fov-start 51 --fov-end 68 \
        --model cpsam --out-dir ./cellpose_bench_multi

    # 若已经用单FOV脚本存好了预测掩膜，可复用、跳过重新推理：
    python cellpose_benchmark_cosmx_multi.py --data-root ... \
        --fov-start 51 --fov-end 68 --reuse-pred --pred-dir ./cellpose_bench_out
"""

import os
import sys
import json
import argparse
import numpy as np

# 复用单 FOV 脚本里的：文件解析 / 图像加载 / 输入构造 / 跑cellpose / 载入指标模块
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import CosMx_single_fov as single   # noqa: E402


EPS = 1e-8
IOU_SWEEP = np.arange(0.5, 1.0, 0.05)       # mDA 的 IoU 阈值扫描，与 test.py 一致


# --------------------------------------------------------------------------- #
# 原始“可累加量”提取（micro 聚合的关键：累加原料，不是累加成品）
# --------------------------------------------------------------------------- #
def pixel_components(bench, gt, pred, boundary_mode="thick"):
    from skimage.segmentation import find_boundaries
    gt_sum = int(np.count_nonzero(gt))
    pred_sum = int(np.count_nonzero(pred))
    inter = int(np.logical_and(gt > 0, pred > 0).sum())
    total = int(gt.size)
    gb = find_boundaries(gt, mode=boundary_mode)
    pb = find_boundaries(pred, mode=boundary_mode)
    b_tp = int(np.logical_and(gb, pb).sum())
    return dict(inter=inter, gt_sum=gt_sum, pred_sum=pred_sum, total=total,
                b_tp=b_tp, gb=int(gb.sum()), pb=int(pb.sum()))


def instance_components(bench, overlaps, iou_threshold=0.5):
    """返回该 FOV 的 TP/FP/FN 与匹配对 IoU 之和/个数（供 micro DQ/SQ/PQ/F1）。"""
    n_gt = len(overlaps.gt_ids)
    n_pred = len(overlaps.pred_ids)
    matches = bench.match_overlaps(overlaps, threshold=iou_threshold)
    tp = len(matches)
    fp = n_pred - tp
    fn = n_gt - tp
    iou_sum = float(sum(m["iou"] for m in matches))
    return dict(TP=tp, FP=fp, FN=fn, iou_sum=iou_sum, n_match=tp)


def aji_components(bench, overlaps, gt, pred):
    """返回 (matched_inter, total_union)。micro-AJI = Σmi / Σunion（严格等于拼大图）。"""
    n_gt = len(overlaps.gt_ids)
    n_pred = len(overlaps.pred_ids)
    if n_gt == 0 and n_pred == 0:
        return 0.0, 0.0
    if len(overlaps.rows) == 0:
        tot = int(np.count_nonzero(gt)) + int(np.count_nonzero(pred))
        return 0.0, float(tot)
    matches = bench.match_overlaps(overlaps, threshold=0.0)
    if len(matches) == 0:
        tot = int(np.count_nonzero(gt)) + int(np.count_nonzero(pred))
        return 0.0, float(tot)
    eidx = np.fromiter((m["edge_idx"] for m in matches),
                       count=len(matches), dtype=np.int64)
    mi = float(overlaps.intersections[eidx].sum())
    mu = float(overlaps.unions[eidx].sum())
    used_gt = np.zeros(n_gt, bool)
    used_pred = np.zeros(n_pred, bool)
    for m in matches:
        used_gt[m["gt_index"]] = True
        used_pred[m["pred_index"]] = True
    ag = np.zeros(n_gt, bool)
    ap = np.zeros(n_pred, bool)
    ag[overlaps.rows.astype(np.int64)] = True
    ap[overlaps.cols.astype(np.int64)] = True
    pg = np.where(ag & ~used_gt)[0]
    pp = np.where(ap & ~used_pred)[0]
    k = min(pg.size, pp.size)
    if k > 0:
        used_gt[pg[:k]] = True
        used_pred[pp[:k]] = True
    ug = float(overlaps.gt_areas[~used_gt].sum())
    up = float(overlaps.pred_areas[~used_pred].sum())
    return mi, mu + ug + up


def mda_components(bench, overlaps):
    """返回 dict: 每个 IoU 阈值上的 (TP,FP,FN)，供 micro-mDA 池化。"""
    n_gt = len(overlaps.gt_ids)
    n_pred = len(overlaps.pred_ids)
    out = {}
    for t in IOU_SWEEP:
        m = bench.match_overlaps(overlaps, threshold=float(t))
        tp = len(m)
        out[round(float(t), 2)] = (tp, n_pred - tp, n_gt - tp)
    return out


# --------------------------------------------------------------------------- #
# per-FOV 成品指标（用于 macro mean±std 与明细表）
# --------------------------------------------------------------------------- #
def per_fov_final_metrics(bench, comp, inst, aji_num, aji_den, mda_map):
    inter, gt_sum, pred_sum, total = comp["inter"], comp["gt_sum"], comp["pred_sum"], comp["total"]
    dice = 2 * inter / (gt_sum + pred_sum + EPS)
    iou = inter / (gt_sum + pred_sum - inter + EPS)
    mismatch = gt_sum + pred_sum - 2 * inter
    pix_acc = (total - mismatch) / total
    bp = comp["b_tp"] / (comp["pb"] + EPS)
    br = comp["b_tp"] / (comp["gb"] + EPS)
    bf1 = 2 * bp * br / (bp + br + EPS)

    tp, fp, fn = inst["TP"], inst["FP"], inst["FN"]
    prec = tp / (tp + fp + EPS)
    rec = tp / (tp + fn + EPS)
    f1 = 2 * prec * rec / (prec + rec + EPS)
    dq = tp / (tp + 0.5 * fp + 0.5 * fn + EPS)
    sq = inst["iou_sum"] / (inst["n_match"] + EPS) if inst["n_match"] > 0 else 0.0
    pq = dq * sq

    aji = aji_num / (aji_den + EPS) if aji_den > 0 else (1.0 if aji_num == 0 else 0.0)
    mda = float(np.mean([tp_/(tp_+fp_+fn_+EPS) for (tp_, fp_, fn_) in mda_map.values()]))

    return dict(dice=dice, iou=iou, pixel_accuracy=pix_acc,
                boundary_f1=bf1, boundary_precision=bp, boundary_recall=br,
                instance_f1=f1, precision=prec, recall=rec,
                TP=tp, FP=fp, FN=fn, DQ=dq, SQ=sq, PQ=pq, AJI=aji,
                **{"mDA_0.50_0.95": mda})


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def process_one_fov(bench, args, fov):
    comp_path, lab_path = single.resolve_paths(
        args.data_root, fov, args.composite_dir, args.labels_dir)
    gt = bench.read_tif_mask(lab_path)
    composite = single.load_composite(comp_path)

    # 预测掩膜：复用已有 / 现跑 cellpose
    pred = None
    if args.reuse_pred:
        import glob
        pat = os.path.join(args.pred_dir or args.out_dir,
                           f"*{args.model}*F{fov:03d}*mask*.tif")
        hits = sorted(glob.glob(pat)) or sorted(
            glob.glob(os.path.join(args.pred_dir or args.out_dir, f"*F{fov:03d}*.tif")))
        if hits:
            pred = bench.read_tif_mask(hits[0])
            print(f"  [reuse] {os.path.basename(hits[0])}")
    if pred is None:
        cp_in = single.build_cellpose_input(composite, cyto_channel=args.cyto_channel)
        pred = single.run_cellpose(
            cp_in, model_name=args.model, diameter=args.diameter,
            gpu=(not args.no_gpu),
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold)
        try:
            import tifffile
            os.makedirs(args.out_dir, exist_ok=True)
            outp = os.path.join(args.out_dir, f"cellpose_{args.model}_F{fov:03d}_masks.tif")
            tifffile.imwrite(outp, pred.astype(np.int32))
        except Exception as e:
            print(f"  [warn] 保存预测tif失败: {e}")

    # shape 对齐
    if gt.shape != pred.shape:
        from skimage.transform import resize
        print(f"  [warn] shape 不一致 gt={gt.shape} pred={pred.shape}，最近邻缩放 pred")
        pred = resize(pred, gt.shape, order=0, preserve_range=True,
                      anti_aliasing=False).astype(pred.dtype)

    # 一次 overlaps，派生所有量
    overlaps = bench.compute_sparse_overlaps(gt, pred)
    comp = pixel_components(bench, gt, pred, boundary_mode=args.boundary_mode)
    inst = instance_components(bench, overlaps, iou_threshold=args.iou_threshold)
    aji_num, aji_den = aji_components(bench, overlaps, gt, pred)
    mda_map = mda_components(bench, overlaps)

    final = per_fov_final_metrics(bench, comp, inst, aji_num, aji_den, mda_map)
    final.update(fov=fov,
                 gt_cell_count=len(overlaps.gt_ids),
                 pred_cell_count=len(overlaps.pred_ids))
    raw = dict(comp=comp, inst=inst, aji_num=aji_num, aji_den=aji_den, mda_map=mda_map)
    return final, raw


def aggregate_micro(raws):
    """池化所有原始量，算一次 → 严格正确的总体指标。"""
    A = dict(inter=0, gt_sum=0, pred_sum=0, total=0, b_tp=0, gb=0, pb=0,
             TP=0, FP=0, FN=0, iou_sum=0.0, n_match=0, aji_num=0.0, aji_den=0.0)
    mda_acc = {round(float(t), 2): [0, 0, 0] for t in IOU_SWEEP}
    for r in raws:
        for k in ("inter", "gt_sum", "pred_sum", "total", "b_tp", "gb", "pb"):
            A[k] += r["comp"][k]
        for k in ("TP", "FP", "FN", "n_match"):
            A[k] += r["inst"][k]
        A["iou_sum"] += r["inst"]["iou_sum"]
        A["aji_num"] += r["aji_num"]
        A["aji_den"] += r["aji_den"]
        for t, (tp, fp, fn) in r["mda_map"].items():
            mda_acc[t][0] += tp; mda_acc[t][1] += fp; mda_acc[t][2] += fn

    dice = 2 * A["inter"] / (A["gt_sum"] + A["pred_sum"] + EPS)
    iou = A["inter"] / (A["gt_sum"] + A["pred_sum"] - A["inter"] + EPS)
    mismatch = A["gt_sum"] + A["pred_sum"] - 2 * A["inter"]
    pix_acc = (A["total"] - mismatch) / (A["total"] + EPS)
    bp = A["b_tp"] / (A["pb"] + EPS); br = A["b_tp"] / (A["gb"] + EPS)
    bf1 = 2 * bp * br / (bp + br + EPS)

    tp, fp, fn = A["TP"], A["FP"], A["FN"]
    prec = tp / (tp + fp + EPS); rec = tp / (tp + fn + EPS)
    f1 = 2 * prec * rec / (prec + rec + EPS)
    dq = tp / (tp + 0.5 * fp + 0.5 * fn + EPS)
    sq = A["iou_sum"] / (A["n_match"] + EPS) if A["n_match"] > 0 else 0.0
    pq = dq * sq
    aji = A["aji_num"] / (A["aji_den"] + EPS)
    mda = float(np.mean([tp_/(tp_+fp_+fn_+EPS) for (tp_, fp_, fn_) in mda_acc.values()]))

    return dict(dice=dice, iou=iou, pixel_accuracy=pix_acc,
                boundary_f1=bf1, boundary_precision=bp, boundary_recall=br,
                instance_f1=f1, precision=prec, recall=rec,
                TP=int(tp), FP=int(fp), FN=int(fn), DQ=dq, SQ=sq, PQ=pq, AJI=aji,
                **{"mDA_0.50_0.95": mda})


def main():
    ap = argparse.ArgumentParser(description="多FOV Cellpose vs CellLabels benchmark 汇总")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--fov-start", type=int, default=51)
    ap.add_argument("--fov-end", type=int, default=68)
    ap.add_argument("--fovs", type=int, nargs="*", default=None,
                    help="显式给定 FOV 列表；给了就覆盖 start/end")
    ap.add_argument("--composite-dir", default=None)
    ap.add_argument("--labels-dir", default=None)
    ap.add_argument("--test-py", default=os.path.join(HERE, "test.py"))
    ap.add_argument("--model", default="cpsam")
    ap.add_argument("--cyto-channel", default="maxmarker",
                    choices=["maxmarker", "panck", "dapi_only"])
    ap.add_argument("--diameter", type=float, default=None)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--flow-threshold", type=float, default=0.4)
    ap.add_argument("--cellprob-threshold", type=float, default=0.0)
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--boundary-mode", default="thick")
    ap.add_argument("--out-dir", default="./cellpose_bench_multi")
    ap.add_argument("--reuse-pred", action="store_true", help="复用已存在的预测tif，跳过推理")
    ap.add_argument("--pred-dir", default=None, help="预测tif所在目录（配合 --reuse-pred）")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    bench = single.load_metric_module(args.test_py)

    fov_list = args.fovs if args.fovs else list(range(args.fov_start, args.fov_end + 1))
    print(f"[plan] 共 {len(fov_list)} 个 FOV: {fov_list}")

    per_fov = []
    raws = []
    failed = []
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
        raise SystemExit("没有任何 FOV 成功。请检查路径/文件命名。")

    import pandas as pd
    df = pd.DataFrame(per_fov)
    metric_cols = ["dice", "iou", "pixel_accuracy", "boundary_f1", "boundary_precision",
                   "boundary_recall", "instance_f1", "precision", "recall",
                   "DQ", "SQ", "PQ", "AJI", "mDA_0.50_0.95"]
    front = ["fov", "gt_cell_count", "pred_cell_count", "TP", "FP", "FN"]
    df = df[[c for c in front if c in df] + [c for c in metric_cols if c in df]]
    per_fov_csv = os.path.join(args.out_dir, "per_fov_metrics.csv")
    df.to_csv(per_fov_csv, index=False)

    # ---- micro（池化，主口径）----
    micro = aggregate_micro(raws)
    total_gt = int(sum(f["gt_cell_count"] for f in per_fov))
    total_pred = int(sum(f["pred_cell_count"] for f in per_fov))

    # ---- macro（逐FOV mean ± std，稳健性口径）----
    macro_mean = {c: float(df[c].mean()) for c in metric_cols if c in df}
    macro_std = {c: float(df[c].std(ddof=1)) if len(df) > 1 else 0.0
                 for c in metric_cols if c in df}

    # ---- 汇总表 ----
    summary_rows = []
    for c in metric_cols:
        summary_rows.append(dict(
            metric=c,
            micro=round(micro.get(c, float("nan")), 6),
            macro_mean=round(macro_mean.get(c, float("nan")), 6),
            macro_std=round(macro_std.get(c, float("nan")), 6),
        ))
    sdf = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.out_dir, "aggregate_summary.csv")
    sdf.to_csv(summary_csv, index=False)

    meta = dict(
        n_fov_ok=len(per_fov), n_fov_failed=len(failed), failed=failed,
        fov_list=fov_list, model=args.model, cyto_channel=args.cyto_channel,
        iou_threshold=args.iou_threshold,
        total_gt_cells=total_gt, total_pred_cells=total_pred,
        micro_TP=micro["TP"], micro_FP=micro["FP"], micro_FN=micro["FN"],
        note="micro=池化(主口径,AJI/PQ必须用它); macro=逐FOV mean±std(稳健性); "
             "pixel_accuracy 背景主导为弱指标。",
    )
    with open(os.path.join(args.out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # ---- 打印 ----
    print("\n" + "=" * 78)
    print(f"成功 {len(per_fov)}/{len(fov_list)} 个 FOV | GT细胞总数={total_gt} "
          f"预测细胞总数={total_pred}")
    if failed:
        print(f"失败 FOV: {[f for f, _ in failed]}")
    print(f"micro 计数: TP={micro['TP']} FP={micro['FP']} FN={micro['FN']}")
    print("-" * 78)
    with pd.option_context("display.width", 120, "display.max_rows", None):
        show = sdf.copy()
        show["macro(mean±std)"] = [f"{m:.4f}±{s:.4f}"
                                   for m, s in zip(show["macro_mean"], show["macro_std"])]
        print(show[["metric", "micro", "macro(mean±std)"]].to_string(index=False))
    print("-" * 78)
    print("论文该报: 主体用 micro 那一列; 稳健性附 macro(mean±std)。")
    print("AJI / PQ 只看 micro（逐FOV平均是错的）。pixel_accuracy 别当主指标。")
    print(f"\n[saved]\n  {per_fov_csv}\n  {summary_csv}\n  "
          f"{os.path.join(args.out_dir, 'run_meta.json')}")


if __name__ == "__main__":
    main()

'''
nohup bash -c '
python CosMx_benchplus.py --data-root /data/qiuyijia/dataset/Pancreas-CosMx-WTx-FlatFiles --fov-start 51 --fov-end 68 \
    --model cpsam --cyto-channel maxmarker \
    --out-dir ./CosMx_output \
    --test-py /data/qiuyijia/eval/test.py \
' > output.log 2>&1 & 
'''