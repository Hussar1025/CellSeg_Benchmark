import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.segmentation import find_boundaries
from scipy.optimize import linear_sum_assignment
from scipy import sparse
from scipy.sparse.csgraph import connected_components

EPS = 1e-8


# --------------------------------------------------------------------------- #
# I/O 与基础工具
# --------------------------------------------------------------------------- #
def read_tif_mask(path):
    mask = imread(path)

    if mask.ndim > 2:
        mask = np.squeeze(mask)

    if mask.ndim != 2:
        raise ValueError(f"Mask should be 2D after squeeze, got shape {mask.shape}")

    if mask.size == 0:
        return mask.astype(np.int32, copy=False)

    # 按需选择最小够用的整数类型，避免 int64 翻倍内存
    max_val = int(mask.max())
    if max_val <= np.iinfo(np.int32).max:
        return mask.astype(np.int32, copy=False)
    return mask.astype(np.int64, copy=False)


def get_instance_ids(mask):
    ids = np.unique(mask)
    return ids[ids != 0]


# --------------------------------------------------------------------------- #
# 像素级指标
# --------------------------------------------------------------------------- #
def compute_pixel_metrics(gt_mask, pred_mask, boundary_mode="thick"):
    """像素级指标。用代数恒等式减少 H×W 临时数组。"""
    gt_sum = int(np.count_nonzero(gt_mask))
    pred_sum = int(np.count_nonzero(pred_mask))
    total_pixels = gt_mask.size

    inter_mask = np.logical_and(gt_mask > 0, pred_mask > 0)
    inter = int(inter_mask.sum())
    del inter_mask  # 立刻释放 H*W bool

    union = gt_sum + pred_sum - inter

    dice = 2 * inter / (gt_sum + pred_sum + EPS)
    iou = inter / (union + EPS)

    # pixel_accuracy = (TP+TN)/total
    # TP+TN = total - XOR像素数 = total - (gt_sum + pred_sum - 2*inter)
    # 这与原 np.mean(gt_bin == pred_bin) 数学完全等价
    mismatch = gt_sum + pred_sum - 2 * inter
    pixel_accuracy = (total_pixels - mismatch) / total_pixels

    # 边界 F1
    gt_boundary = find_boundaries(gt_mask, mode=boundary_mode)
    gb_sum = int(gt_boundary.sum())

    pred_boundary = find_boundaries(pred_mask, mode=boundary_mode)
    bp_sum = int(pred_boundary.sum())

    boundary_tp = int(np.logical_and(gt_boundary, pred_boundary).sum())
    del gt_boundary, pred_boundary

    boundary_precision = boundary_tp / (bp_sum + EPS)
    boundary_recall = boundary_tp / (gb_sum + EPS)
    boundary_f1 = (
        2 * boundary_precision * boundary_recall
        / (boundary_precision + boundary_recall + EPS)
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "pixel_accuracy": float(pixel_accuracy),
        "boundary_f1": float(boundary_f1),
        "boundary_precision": float(boundary_precision),
        "boundary_recall": float(boundary_recall),
    }


# --------------------------------------------------------------------------- #
# 稀疏 IoU / 重叠数据
# --------------------------------------------------------------------------- #
class OverlapData:
    """实例级重叠的紧凑存储。所有数组等长，按边索引；gt_areas/pred_areas 按实例索引。"""
    __slots__ = ("gt_ids", "pred_ids",
                 "rows", "cols",
                 "intersections", "unions", "ious",
                 "gt_areas", "pred_areas")

    def __init__(self, gt_ids, pred_ids, rows, cols,
                 intersections, unions, ious, gt_areas, pred_areas):
        self.gt_ids = gt_ids
        self.pred_ids = pred_ids
        self.rows = rows
        self.cols = cols
        self.intersections = intersections
        self.unions = unions
        self.ious = ious
        self.gt_areas = gt_areas
        self.pred_areas = pred_areas


def compute_sparse_overlaps(gt_mask, pred_mask):
    """
    完全向量化计算所有 GT-Pred 重叠对的交集、并集、IoU。
    替换原 compute_sparse_instance_iou 中的 Python for 循环和 dict 查表。
    """
    gt_ids = get_instance_ids(gt_mask)
    pred_ids = get_instance_ids(pred_mask)

    n_gt = len(gt_ids)
    n_pred = len(pred_ids)

    empty_i = np.array([], dtype=np.int64)
    empty_f = np.array([], dtype=np.float64)

    if n_gt == 0 or n_pred == 0:
        return OverlapData(
            gt_ids, pred_ids,
            empty_i, empty_i,
            empty_f, empty_f, empty_f,
            np.zeros(n_gt, dtype=np.int64),
            np.zeros(n_pred, dtype=np.int64),
        )

    gt_flat = gt_mask.ravel()
    pred_flat = pred_mask.ravel()

    # bincount 一次算出全部面积（包括 0 标签），O(N)、无需 sort、无需复制
    gt_areas_full = np.bincount(gt_flat)
    pred_areas_full = np.bincount(pred_flat)

    gt_areas = gt_areas_full[gt_ids]
    pred_areas = pred_areas_full[pred_ids]
    del gt_areas_full, pred_areas_full

    # id -> 索引 映射用数组（比 dict 快 1-2 个数量级）
    max_gt_id = int(gt_ids.max())
    max_pred_id = int(pred_ids.max())

    gt_id_to_idx = np.full(max_gt_id + 1, -1, dtype=np.int64)
    gt_id_to_idx[gt_ids] = np.arange(n_gt)

    pred_id_to_idx = np.full(max_pred_id + 1, -1, dtype=np.int64)
    pred_id_to_idx[pred_ids] = np.arange(n_pred)

    # 仅保留两边都非零的像素位置
    valid = (gt_flat > 0) & (pred_flat > 0)
    if not valid.any():
        return OverlapData(
            gt_ids, pred_ids,
            empty_i, empty_i,
            empty_f, empty_f, empty_f,
            gt_areas, pred_areas,
        )

    gt_overlap = gt_flat[valid]
    pred_overlap = pred_flat[valid]
    del valid  # 释放 H*W bool

    # 将 (gt_id, pred_id) 编码为单个 int64 以便一次 unique 拿到所有重叠对
    pred_max = int(pred_mask.max()) + 1
    pair_codes = gt_overlap.astype(np.int64) * pred_max + pred_overlap.astype(np.int64)
    del gt_overlap, pred_overlap

    unique_pairs, intersection_counts = np.unique(pair_codes, return_counts=True)
    del pair_codes

    gid_arr = unique_pairs // pred_max
    pid_arr = unique_pairs % pred_max

    # 向量化映射：直接索引，没有 Python 循环
    rows = gt_id_to_idx[gid_arr]
    cols = pred_id_to_idx[pid_arr]

    intersections = intersection_counts.astype(np.float64)
    unions = (gt_areas[rows].astype(np.float64)
              + pred_areas[cols].astype(np.float64)
              - intersections)
    ious = intersections / (unions + EPS)

    return OverlapData(
        gt_ids, pred_ids,
        rows, cols,
        intersections, unions, ious,
        gt_areas, pred_areas,
    )


# --------------------------------------------------------------------------- #
# 匹配：连通分量分解 + 各分量局部匈牙利
# --------------------------------------------------------------------------- #
def match_overlaps(overlaps, threshold=0.5):
    """
    用 connected_components 把二部图拆成若干连通分量，
    每个分量内部跑 linear_sum_assignment。
    与全局直接跑 dense 匈牙利 数学等价（跨分量无边，最优匹配可分解），
    但局部 dense 通常只有 1x1 ~ 几x几，内存几乎不消耗。

    返回元素: {"gt_index", "pred_index", "iou", "edge_idx"}
    edge_idx 指 overlaps.rows/cols/intersections/unions 中对应的位置，
    供 AJI 直接按 index 取交并值，无需再查 csr。
    """
    if len(overlaps.rows) == 0:
        return []

    keep = overlaps.ious >= threshold
    if not keep.any():
        return []

    orig_idx = np.where(keep)[0]
    rows = overlaps.rows[orig_idx]
    cols = overlaps.cols[orig_idx]
    ious = overlaps.ious[orig_idx]

    n_gt = len(overlaps.gt_ids)
    n_pred = len(overlaps.pred_ids)

    # 构造对称二部图：GT 节点 0..n_gt-1，Pred 节点 n_gt..n_gt+n_pred-1
    n_total = n_gt + n_pred
    g_rows = np.concatenate([rows, cols + n_gt])
    g_cols = np.concatenate([cols + n_gt, rows])
    g_data = np.ones(g_rows.size, dtype=np.uint8)
    graph = sparse.csr_matrix(
        (g_data, (g_rows, g_cols)),
        shape=(n_total, n_total),
    )
    _, labels = connected_components(graph, directed=False)
    del graph, g_rows, g_cols, g_data

    # 每条边的连通分量（取 GT 端 label 即可，与 pred 端相同）
    edge_comp = labels[rows]

    # 按 component 排序，找出每个 component 在数组中的连续区间
    sort_idx = np.argsort(edge_comp, kind="stable")
    edge_comp_sorted = edge_comp[sort_idx]

    boundaries = np.where(np.diff(edge_comp_sorted) != 0)[0] + 1
    starts = np.concatenate([[0], boundaries, [edge_comp_sorted.size]])

    matches = []
    for i in range(len(starts) - 1):
        s, e = starts[i], starts[i + 1]
        idx_in_kept = sort_idx[s:e]

        c_rows = rows[idx_in_kept]
        c_cols = cols[idx_in_kept]
        c_ious = ious[idx_in_kept]
        c_orig = orig_idx[idx_in_kept]

        unique_r, inv_r = np.unique(c_rows, return_inverse=True)
        unique_c, inv_c = np.unique(c_cols, return_inverse=True)

        nr = unique_r.size
        nc = unique_c.size

        # 绝大多数细胞分割场景中，连通分量是 1-2 个节点，这里直接闭式处理
        if nr == 1 and nc == 1:
            # 与原版 dense=float32 的存储精度一致
            iv = float(np.float32(c_ious[0]))
            if iv >= threshold:
                matches.append({
                    "gt_index": int(unique_r[0]),
                    "pred_index": int(unique_c[0]),
                    "iou": iv,
                    "edge_idx": int(c_orig[0]),
                })
            continue

        # 局部 dense 矩阵（一般极小，不会撑爆内存）
        local = np.zeros((nr, nc), dtype=np.float64)
        local_edge = np.full((nr, nc), -1, dtype=np.int64)
        local[inv_r, inv_c] = c_ious
        local_edge[inv_r, inv_c] = c_orig

        # 与原版 dense=float32 的存储精度一致：再过一遍 float32
        local32 = local.astype(np.float32)
        rr, cc = linear_sum_assignment(-local32)
        for a, b in zip(rr, cc):
            eidx = int(local_edge[a, b])
            if eidx < 0:
                # 局部 dense 中的零位置：不是真实边，不视作匹配。
                # （虚假凑数行为在 AJI 中独立模拟。）
                continue
            iv = float(local32[a, b])
            if iv >= threshold:
                matches.append({
                    "gt_index": int(unique_r[a]),
                    "pred_index": int(unique_c[b]),
                    "iou": iv,
                    "edge_idx": eidx,
                })

    return matches


# 实例级指标
def instance_metrics_from_overlaps(overlaps, threshold=0.5):
    matches = match_overlaps(overlaps, threshold=threshold)
    n_gt = len(overlaps.gt_ids)
    n_pred = len(overlaps.pred_ids)

    tp = len(matches)
    fp = n_pred - tp
    fn = n_gt - tp

    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    instance_f1 = 2 * precision * recall / (precision + recall + EPS)

    dq = tp / (tp + 0.5 * fp + 0.5 * fn + EPS)
    sq = float(np.mean([m["iou"] for m in matches])) if tp > 0 else 0.0
    pq = dq * sq

    return {
        "instance_f1": float(instance_f1),
        "precision": float(precision),
        "recall": float(recall),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "DQ": float(dq),
        "SQ": float(sq),
        "PQ": float(pq),
        "over_segmentation": int(fp),
        "under_segmentation": int(fn),
    }


def aji_from_overlaps(overlaps, gt_mask, pred_mask):
    n_gt = len(overlaps.gt_ids)
    n_pred = len(overlaps.pred_ids)

    if n_gt == 0 and n_pred == 0:
        return 1.0

    if len(overlaps.rows) == 0:
        gt_total = int(np.count_nonzero(gt_mask))
        pred_total = int(np.count_nonzero(pred_mask))
        total_area = gt_total + pred_total
        return 0.0 if total_area > 0 else 1.0

    matches = match_overlaps(overlaps, threshold=0.0)

    if len(matches) == 0:
        gt_total = int(np.count_nonzero(gt_mask))
        pred_total = int(np.count_nonzero(pred_mask))
        total_area = gt_total + pred_total
        return 0.0 if total_area > 0 else 1.0

    edge_idx = np.fromiter((m["edge_idx"] for m in matches),
                           count=len(matches), dtype=np.int64)
    matched_inter = float(overlaps.intersections[edge_idx].sum())
    matched_union = float(overlaps.unions[edge_idx].sum())

    used_gt = np.zeros(n_gt, dtype=bool)
    used_pred = np.zeros(n_pred, dtype=bool)
    for m in matches:
        used_gt[m["gt_index"]] = True
        used_pred[m["pred_index"]] = True

    # active_gt / active_pred = 出现在 overlaps 边中的 GT / Pred 索引
    active_gt_mask = np.zeros(n_gt, dtype=bool)
    active_pred_mask = np.zeros(n_pred, dtype=bool)
    active_gt_mask[overlaps.rows.astype(np.int64)] = True
    active_pred_mask[overlaps.cols.astype(np.int64)] = True

    # 在 active 中尚未被真实匹配占用的部分
    pending_gt = np.where(active_gt_mask & ~used_gt)[0]
    pending_pred = np.where(active_pred_mask & ~used_pred)[0]

    k = min(pending_gt.size, pending_pred.size)
    if k > 0:
        used_gt[pending_gt[:k]] = True
        used_pred[pending_pred[:k]] = True

    unmatched_gt_area = float(overlaps.gt_areas[~used_gt].sum())
    unmatched_pred_area = float(overlaps.pred_areas[~used_pred].sum())

    total_union = matched_union + unmatched_gt_area + unmatched_pred_area
    return float(matched_inter / (total_union + EPS))


def mean_detection_accuracy_from_overlaps(
    overlaps,
    thresholds=np.arange(0.5, 1.0, 0.05),
):
    n_gt = len(overlaps.gt_ids)
    n_pred = len(overlaps.pred_ids)

    scores = []
    for threshold in thresholds:
        matches = match_overlaps(overlaps, threshold=threshold)
        tp = len(matches)
        fp = n_pred - tp
        fn = n_gt - tp
        score = tp / (tp + fp + fn + EPS)
        scores.append(score)

    return float(np.mean(scores))


# 单方法 / 全流程入口
def evaluate_single_mask_against_cellpose(
    method_name,
    cellpose_gt_mask,
    pred_mask,
    boundary_mode="thick",
    iou_threshold=0.5,
):
    if cellpose_gt_mask.shape != pred_mask.shape:
        raise ValueError(
            f"{method_name}: mask shape mismatch. "
            f"Cellpose GT={cellpose_gt_mask.shape}, Pred={pred_mask.shape}"
        )

    pixel_metrics = compute_pixel_metrics(
        gt_mask=cellpose_gt_mask,
        pred_mask=pred_mask,
        boundary_mode=boundary_mode,
    )

    overlaps = compute_sparse_overlaps(cellpose_gt_mask, pred_mask)

    instance_metrics = instance_metrics_from_overlaps(
        overlaps, threshold=iou_threshold
    )

    aji = aji_from_overlaps(overlaps, cellpose_gt_mask, pred_mask)

    mda = mean_detection_accuracy_from_overlaps(overlaps)

    return {
        "reference": "Cellpose",
        "method": method_name,
        "cellpose_reference_cell_count": int(len(overlaps.gt_ids)),
        "pred_cell_count": int(len(overlaps.pred_ids)),
        **pixel_metrics,
        **instance_metrics,
        "AJI": float(aji),
        "mDA_0.50_0.95": float(mda),
        "mDA_note": "mDA is mean TP/(TP+FP+FN) over IoU thresholds 0.50:0.95; "
                    "not standard confidence-ranked AP.",
    }


def run_cellpose_reference_evaluation(
    cellpose_gt_path,
    method_mask_paths,
    output_csv="image_metrics_cellpose_reference.csv",
    boundary_mode="thick",
    iou_threshold=0.5,
):
    cellpose_gt_mask = read_tif_mask(cellpose_gt_path)

    results = []

    for method_name, mask_path in method_mask_paths.items():
        if method_name.lower() == "cellpose":
            print("[Skip] Cellpose is used as reference")
            continue
        if mask_path is None or not os.path.exists(mask_path):
            print(f"[Skip] {method_name}: mask file not found")
            continue

        print(f"Evaluating {method_name} against Cellpose reference ...")
        pred_mask = read_tif_mask(mask_path)

        result = evaluate_single_mask_against_cellpose(
            method_name=method_name,
            cellpose_gt_mask=cellpose_gt_mask,
            pred_mask=pred_mask,
            boundary_mode=boundary_mode,
            iou_threshold=iou_threshold,
        )
        results.append(result)

        del pred_mask  # 评估完立即释放，避免下次循环时双倍占用

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    print(df)
    print("Saved:", output_csv)
    return df


if __name__ == "__main__":

    # ====== 在下方修改你的路径配置 ======
    CELLPOSE_GT_PATH = "/data/qiuyijia/cellpose/Pancreas_output/output_cpsam_masks.ome.tif"

    METHOD_MASK_PATHS = {
        "StarDist":  "/data/qiuyijia/stardist/Pancreas_cancer_output/labels.tif",
        "CellSAM":   "/data/qiuyijia/CellSam/test1_0.5/slide_mask.tif",
        "Mesmer": "/data/qiuyijia/Mesmer/pancreas/slide01_mesmer/instance_mask.tif"
    }

    OUTPUT_CSV = "image_metrics_cellpose_reference.csv"
    BOUNDARY_MODE = "thick"
    IOU_THRESHOLD = 0.5
    # ===================================

    run_cellpose_reference_evaluation(
        cellpose_gt_path=CELLPOSE_GT_PATH,
        method_mask_paths=METHOD_MASK_PATHS,
        output_csv=OUTPUT_CSV,
        boundary_mode=BOUNDARY_MODE,
        iou_threshold=IOU_THRESHOLD,
    )