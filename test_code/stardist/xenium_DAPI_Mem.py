#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
StarDist × Xenium 细胞分割 benchmark —— 只测 xenium_breast
=========================================================

结构与 Cellpose / CelloType / Mesmer / CellSAM 版一致，但**只在 xenium_breast 上跑**：
  GT = 官方 cell_boundaries.parquet 栅格化（默认只跑居中 5000×5000 ROI）。
读形态学图 → StarDist 出核 → skimage.expand_labels 把核膨胀 EXPAND_PX 像素近似整细胞
  → 构建 GT → 形状对齐 → 调 test.py 指标 → 写入 /data/qiuyijia/stardist/xenium_breast。

StarDist 要点（据官方 https://github.com/stardist/stardist 与 squidpy 示例）
---------------------------------------------------------------------------
- 安装：pip install stardist csbdeep （TensorFlow 后端；GPU 强烈推荐）
- API：from stardist.models import StarDist2D; from csbdeep.utils import normalize
       model = StarDist2D.from_pretrained('2D_versatile_fluo')   # 荧光核模型，轴 'YX'
       img = normalize(dapi, 1, 99.8, axis=(0,1))                # 必须归一化
       labels, _ = model.predict_instances(img, n_tiles=...)      # 大图分块；返回 2D int label
- 预训练模型：'2D_versatile_fluo'(荧光核)、'2D_versatile_he'(H&E核)、'2D_paper_dsb2018'、'2D_demo'。
  Xenium 是 DAPI 荧光 → 用 '2D_versatile_fluo'。默认阈值 prob≈0.479, nms=0.3。

核→整细胞的膨胀 与 公平性说明
------------------------------
StarDist 只分割**细胞核**（吃 DAPI 单通道）。本脚本按标准做法把核用 skimage.segmentation.expand_labels
向外膨胀 EXPAND_PX 像素近似整细胞，再对全细胞 GT (cell_boundaries.parquet) 评分。这是 Xenium/squidpy
等常用的「无膜时以核外扩定义细胞」的思路，写清楚即公平，但请注意：
  1) EXPAND_PX 是**超参**——请固定为有依据的值（建议用 EXPAND_UM 按 pixel_size 换算成 ~几 µm），
     切勿在评测集上扫它挑最高分；要透明可把整条 expand 扫描曲线一并报出。
  2) method 列会写成 "StarDist(+Npx)" 并单列 expand_px——这是「核+固定膨胀」pipeline，不是原始 StarDist。
  3) 口径差异：膜法(Cellpose/CelloType/Mesmer/CellSAM)用真实膜信号勾勒**不规则**细胞，膨胀假设细胞
     **各向同性≈圆、尺寸均一**——这是方法差异（非不公平），解读数字时需知晓。
把 EXPAND_PX 设为 0 且 boundaries 改成 "nucleus_boundaries.parquet"，即回到纯核分割评测。

语义说明（与前几版相同）
------------------------
test.py 的 evaluate_single_mask_against_cellpose(cellpose_gt_mask, pred_mask) 把第一个槽当 reference。
本 benchmark 里 **StarDist 是被测方法**：真值放第一个槽、StarDist 预测放第二个槽。

依赖：numpy, pandas, scikit-image, scipy, tifffile, pyarrow, stardist, csbdeep(+tensorflow)
"""

import os
import re
import sys
import json
import glob
import warnings
import importlib.util

import numpy as np
import pandas as pd
import tifffile
from skimage.draw import polygon as sk_polygon


# ---------------------------------------------------------------------------- #
# test.py（指标实现）的位置
# ---------------------------------------------------------------------------- #
TEST_PY_PATH = "/data/qiuyijia/eval/test.py"


def _load_metrics():
    if TEST_PY_PATH and os.path.exists(TEST_PY_PATH):
        spec = importlib.util.spec_from_file_location("xenium_metrics", TEST_PY_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        import test as mod
        return mod
    except ImportError as e:
        raise ImportError(
            "找不到 test.py：请把脚本顶部的 TEST_PY_PATH 改成 test.py 的绝对路径，"
            "或把 test.py 放到与本脚本同一目录。"
        ) from e


metrics = _load_metrics()


# ============================================================================ #
# 配置区
# ============================================================================ #
XENIUM_DEFAULT_PIXEL_SIZE = 0.2125  # µm/px，读不到 experiment 时的兜底值

# StarDist 全局参数
STARDIST = dict(
    model_name="2D_versatile_fluo",  # 荧光核模型
    norm_low=1.0,                    # normalize 百分位下界
    norm_high=99.8,                  # normalize 百分位上界
    prob_thresh=None,                # None=用模型默认(≈0.479)；调低→更多核
    nms_thresh=None,                 # None=用模型默认(0.3)
    n_tiles=None,                    # None=自动按显存分块；也可手填如 (4,4)
)

# 核 → 整细胞的膨胀（skimage.segmentation.expand_labels）
# expand_labels 是基于距离变换的 Voronoi 式膨胀：每个核向背景生长、相邻处相遇即停、
# 互不重叠、且每个核最多扩 expand_px 像素；膨胀不改变实例数，只扩大每个实例的范围。
EXPAND_PX = 15       # 膨胀距离(像素)。⚠超参：请固定为有依据的值，勿在评测集上调它挑高分。设 0=不膨胀
EXPAND_UM = None     # 若非 None，用 round(EXPAND_UM / pixel_size) 覆盖 EXPAND_PX（更有依据、换分辨率也稳）

# 通道约定：Xenium morphology_focus 通道 0 = DAPI(核)
NUCLEUS_IDX = 0

PARENT_OUT = "/data/qiuyijia/stardist"

DATASETS = [
    dict(
        name="xenium_mouse_brain",
        dataset_dir="/data/qiuyijia/dataset/xenium_mouse_brain/outs",  # 注意：路径到outs/
        output_dir="/data/qiuyijia/stardist/xenium_mouse_brain_output",
        gt_type="xenium_boundaries",
        boundaries="cell_boundaries.parquet",
        experiment="experiment.xenium",
        pyramid_level=0,
        use_membrane=True,
        crop=(12077, 22077, 6956, 16956),
    ),
]

BOUNDARY_MODE = "thick"
IOU_THRESHOLD = 0.5
# ============================================================================ #


# ---------------------------------------------------------------------------- #
# 形态学图像读取（金字塔层 + 两种命名 + 单/多通道文件）
# ---------------------------------------------------------------------------- #
def _read_level(path, pyramid_level):
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        try:
            levels = list(series.levels)
        except Exception:
            levels = [series]
        L = int(np.clip(pyramid_level, 0, len(levels) - 1))
        if L != pyramid_level:
            warnings.warn(f"{os.path.basename(path)}: 只有 {len(levels)} 个金字塔层，"
                          f"pyramid_level={pyramid_level} 被截断到 {L}")
        arr = np.asarray(levels[L].asarray())
        base_h = levels[0].shape[-2]
        cur_h = levels[L].shape[-2]
        factor = float(base_h) / float(cur_h)
    return arr, factor


def _split_channels(arr):
    if arr.ndim == 2:
        return [arr]
    if arr.ndim == 3:
        if arr.shape[0] <= 5 and arr.shape[0] <= arr.shape[-1]:
            return [arr[i] for i in range(arr.shape[0])]
        if arr.shape[-1] <= 5:
            return [arr[..., i] for i in range(arr.shape[-1])]
        return [arr[arr.shape[0] // 2]]
    arr = np.squeeze(arr)
    return _split_channels(arr) if arr.ndim <= 3 else [np.max(arr, axis=tuple(range(arr.ndim - 2)))]


def load_morphology_channels(dataset_dir, pyramid_level=0):
    focus_dir = os.path.join(dataset_dir, "morphology_focus")
    channels, names = [], []

    if os.path.isdir(focus_dir):
        files = sorted(glob.glob(os.path.join(focus_dir, "*.ome.tif")) +
                       glob.glob(os.path.join(focus_dir, "*.ome.tiff")))

        def _idx(fp):
            b = os.path.basename(fp)
            m = re.search(r"ch0*(\d+)", b) or re.search(r"morphology_focus_0*(\d+)", b)
            return int(m.group(1)) if m else 0
        files = sorted(files, key=_idx)

        factor = 1.0
        for fp in files:
            arr, factor = _read_level(fp, pyramid_level)
            for c in _split_channels(arr):
                channels.append(c)
                names.append(os.path.splitext(os.path.basename(fp))[0])
        if channels:
            return dict(channels=channels, names=names, factor=factor)

    for cand in ("morphology.ome.tif", "morphology_focus.ome.tif", "morphology_mip.ome.tif"):
        fp = os.path.join(dataset_dir, cand)
        if os.path.exists(fp):
            arr, factor = _read_level(fp, pyramid_level)
            chs = _split_channels(arr)
            if len(chs) > 1:
                chs = [np.max(np.stack(chs, 0), 0)]
            warnings.warn(f"{dataset_dir}: 未找到 morphology_focus/，退回 {cand}（仅核通道）")
            return dict(channels=chs, names=[cand], factor=factor)

    raise FileNotFoundError(f"{dataset_dir}: 找不到任何形态学 OME-TIFF")


# ---------------------------------------------------------------------------- #
# StarDist 分割（核）
# ---------------------------------------------------------------------------- #
def run_stardist(dapi, cfg=STARDIST):
    """跑 StarDist（'2D_versatile_fluo' 荧光核模型），返回 int32 实例 label mask [H,W]。"""
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize

    model = StarDist2D.from_pretrained(cfg["model_name"])
    img = normalize(dapi.astype(np.float32), cfg.get("norm_low", 1.0),
                    cfg.get("norm_high", 99.8), axis=(0, 1))

    n_tiles = cfg.get("n_tiles")
    if n_tiles is None:
        try:
            n_tiles = model._guess_n_tiles(img)   # 按显存自动分块
        except Exception:
            n_tiles = None

    labels, _ = model.predict_instances(
        img, prob_thresh=cfg.get("prob_thresh"),
        nms_thresh=cfg.get("nms_thresh"), n_tiles=n_tiles)
    return np.asarray(labels).astype(np.int32, copy=False)


def expand_nuclei_to_cells(nuclei, expand_px):
    """用 skimage.segmentation.expand_labels 把核 label 向外膨胀 expand_px 像素近似整细胞。
    各核向背景生长、相遇即停、互不重叠、每个核最多扩 expand_px；实例数不变。expand_px<=0 原样返回。"""
    if expand_px is None or expand_px <= 0:
        return np.asarray(nuclei).astype(np.int32, copy=False)
    from skimage.segmentation import expand_labels
    return expand_labels(np.asarray(nuclei), distance=expand_px).astype(np.int32)


# ---------------------------------------------------------------------------- #
# GT 构建（官方边界栅格化）
# ---------------------------------------------------------------------------- #
def read_pixel_size(experiment_path, default=XENIUM_DEFAULT_PIXEL_SIZE):
    """从 experiment.xenium（JSON）递归找 pixel_size(µm/px)。"""
    if not experiment_path or not os.path.exists(experiment_path):
        warnings.warn(f"未找到 {experiment_path}，pixel_size 用默认 {default}")
        return default
    try:
        with open(experiment_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        warnings.warn(f"读取 {experiment_path} 失败({e})，pixel_size 用默认 {default}")
        return default

    found = []

    def _walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(k, str) and "pixel_size" in k.lower() and isinstance(v, (int, float)):
                    found.append(float(v))
                _walk(v)
        elif isinstance(o, list):
            for v in o:
                _walk(v)

    _walk(data)
    if found:
        return found[0]
    warnings.warn(f"{experiment_path} 中无 pixel_size 字段，用默认 {default}")
    return default


def _find_experiment(ddir, cfg):
    cands = []
    if cfg.get("experiment"):
        cands.append(cfg["experiment"])
    cands += ["experiment.xenium", "experiment_roi.json"]
    for c in cands:
        p = os.path.join(ddir, c)
        if os.path.exists(p):
            return p
    return None


def _boundary_columns(df):
    cols = {c.lower(): c for c in df.columns}
    id_col = next((cols[c] for c in ("cell_id", "label_id", "label", "id") if c in cols), None)
    # 优先使用 local 坐标（ROI 相对坐标），没有才用全局坐标
    x_col = next((cols[c] for c in ("vertex_x_local_um", "vertex_x_local", "vertex_x", "x", "x_location", "vertex_x_um") if c in cols), None)
    y_col = next((cols[c] for c in ("vertex_y_local_um", "vertex_y_local", "vertex_y", "y", "y_location", "vertex_y_um") if c in cols), None)
    return id_col, x_col, y_col


def rasterize_boundaries(parquet_path, out_shape, pixel_size_eff, offset=(0, 0), verbose=True):
    """把 Xenium 边界 parquet（微米多边形）栅格化成 int32 label mask。
    像素 = 微米 / pixel_size_eff - offset；越界部分由 skimage 自动裁掉。"""
    row0, col0 = offset
    df = pd.read_parquet(parquet_path)
    id_col, x_col, y_col = _boundary_columns(df)
    if id_col is None or x_col is None or y_col is None:
        raise ValueError(f"{parquet_path}: 无法识别列 (cell_id/vertex_x/vertex_y)，实际列={list(df.columns)}")

    H, W = out_shape
    mask = np.zeros((H, W), dtype=np.int32)

    xs_all = df[x_col].to_numpy(dtype=np.float64) / pixel_size_eff - col0
    ys_all = df[y_col].to_numpy(dtype=np.float64) / pixel_size_eff - row0
    ids_all = df[id_col].to_numpy()

    codes, uniq = pd.factorize(ids_all)
    n_cells = len(uniq)
    sort_idx = np.argsort(codes, kind="stable")
    codes_s = codes[sort_idx]
    xs_s = xs_all[sort_idx]
    ys_s = ys_all[sort_idx]
    seg_start = np.searchsorted(codes_s, np.arange(n_cells), side="left")
    seg_start = np.append(seg_start, len(codes_s))

    label = 0
    for c in range(n_cells):
        s, e = seg_start[c], seg_start[c + 1]
        if e - s < 3:
            continue
        label += 1
        rr, cc = sk_polygon(ys_s[s:e], xs_s[s:e], shape=(H, W))
        if rr.size:
            mask[rr, cc] = label
        if verbose and label % 50000 == 0:
            print(f"    栅格化 {label} 个细胞 ...")
    if verbose:
        print(f"    栅格化完成，共 {label} 个 → mask {mask.shape}")
    return mask


# ---------------------------------------------------------------------------- #
# 形状对齐 + 评估
# ---------------------------------------------------------------------------- #
def align_shapes(gt, pred):
    if gt.shape == pred.shape:
        return gt, pred
    warnings.warn(f"GT{gt.shape} 与 Pred{pred.shape} 形状不一致，右下角补零对齐（请核对是否真正对齐）")
    H = max(gt.shape[0], pred.shape[0])
    W = max(gt.shape[1], pred.shape[1])

    def _pad(m):
        out = np.zeros((H, W), dtype=m.dtype)
        out[:m.shape[0], :m.shape[1]] = m
        return out
    return _pad(gt), _pad(pred)


def evaluate(gt_mask, pred_mask, dataset_name, gt_source,
             boundary_mode=BOUNDARY_MODE, iou_threshold=IOU_THRESHOLD):
    """调 test.py 指标：GT 放 reference 槽、StarDist 放 pred 槽；再改成无歧义字段名。"""
    gt_mask, pred_mask = align_shapes(gt_mask, pred_mask)

    raw = metrics.evaluate_single_mask_against_cellpose(
        method_name="StarDist",
        cellpose_gt_mask=gt_mask,     # ← 真值
        pred_mask=pred_mask,          # ← StarDist 预测
        boundary_mode=boundary_mode,
        iou_threshold=iou_threshold,
    )

    result = {
        "dataset": dataset_name,
        "gt_source": gt_source,
        "method": "StarDist",
        "gt_cell_count": raw.pop("cellpose_reference_cell_count"),
        "pred_cell_count": raw.pop("pred_cell_count"),
    }
    raw.pop("reference", None)
    raw.pop("method", None)
    raw.pop("mDA_note", None)
    result.update(raw)
    return result


# ---------------------------------------------------------------------------- #
# 单数据集流水线
# ---------------------------------------------------------------------------- #
def _apply_crop(arr, crop):
    if crop is None:
        return arr
    y0, y1, x0, x1 = crop
    return arr[y0:y1, x0:x1]


def run_dataset(cfg):
    name = cfg["name"]
    ddir = cfg["dataset_dir"]
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== [{name}] {ddir}  →  {out_dir} ===")

    # 1) 读形态学通道（整图）
    morph = load_morphology_channels(ddir, pyramid_level=cfg.get("pyramid_level", 0))
    factor = morph["factor"]
    channels = morph["channels"]
    Himg, Wimg = channels[0].shape
    print(f"  通道数={len(channels)}  名称={morph['names']}  下采样因子={factor:.3f}")

    # 2) 决定裁剪窗口（显式 crop 优先，否则 crop_size 居中）
    crop = cfg.get("crop")
    if crop is None and cfg.get("crop_size"):
        s = int(cfg["crop_size"])
        y0 = max((Himg - s) // 2, 0)
        x0 = max((Wimg - s) // 2, 0)
        crop = (y0, min(y0 + s, Himg), x0, min(x0 + s, Wimg))
        print(f"  crop_size={s} → 居中裁剪 crop={crop}（整图 {Himg}×{Wimg}）")

    if crop is not None:
        channels = [_apply_crop(c, crop) for c in channels]

    # StarDist 只用 DAPI(核)单通道
    dapi = (channels[NUCLEUS_IDX] if NUCLEUS_IDX < len(channels) else channels[0])
    seg_shape = dapi.shape
    print(f"  StarDist 输入: DAPI {dapi.shape}  model={STARDIST['model_name']}（核分割）")

    # 3) pixel_size（栅格化 GT 用）
    exp_path = _find_experiment(ddir, cfg)
    base_px = read_pixel_size(exp_path)
    px_eff = base_px * factor
    print(f"  pixel_size={base_px} µm/px  当前层有效 pixel_size={px_eff:.4f} µm/px")

    # 4) StarDist 分割(核) → 膨胀近似整细胞
    print("  运行 StarDist ...")
    pred_nuc = run_stardist(dapi, cfg=STARDIST)
    print(f"  预测核数={len(metrics.get_instance_ids(pred_nuc))}  mask={pred_nuc.shape}")
    expand_px = EXPAND_PX
    if EXPAND_UM is not None:
        expand_px = int(round(EXPAND_UM / px_eff))
        print(f"  按 EXPAND_UM={EXPAND_UM}µm / pixel_size={px_eff:.4f} → expand_px={expand_px}")
    pred = expand_nuclei_to_cells(pred_nuc, expand_px)
    if expand_px and expand_px > 0:
        print(f"  核膨胀 {expand_px}px 近似整细胞（expand_labels）→ 实例数={len(metrics.get_instance_ids(pred))}")
    else:
        print("  expand_px<=0：不膨胀，直接用核 mask")

    # 5) 构建 GT（官方边界栅格化，与预测同一裁剪窗口对齐）
    if crop is not None:
        y0, y1, x0, x1 = crop
        gt = rasterize_boundaries(os.path.join(ddir, cfg["boundaries"]),
                                  out_shape=(y1 - y0, x1 - x0),
                                  pixel_size_eff=px_eff, offset=(y0, x0))
    else:
        gt = rasterize_boundaries(os.path.join(ddir, cfg["boundaries"]),
                                  out_shape=seg_shape, pixel_size_eff=px_eff)
    gt_source = os.path.splitext(cfg["boundaries"])[0]  # cell_boundaries / nucleus_boundaries
    print(f"  GT 个数={len(metrics.get_instance_ids(gt))}  来源={gt_source}")
    if gt_source.startswith("cell") and (not expand_px or expand_px <= 0):
        print("  ⚠ 未膨胀却对全细胞 GT：核比整细胞小、覆盖会偏低；"
              "设 EXPAND_PX>0 膨胀近似整细胞，或把 boundaries 改成 nucleus_boundaries.parquet 做核评测")

    # 6) 落盘 mask（int32 tif，兼容 test.py 的 read_tif_mask）
    pred_path = os.path.join(out_dir, f"{name}_stardist_pred.tif")
    gt_path = os.path.join(out_dir, f"{name}_gt.tif")
    tifffile.imwrite(pred_path, pred)
    tifffile.imwrite(gt_path, gt)
    print(f"  已保存 {pred_path} / {gt_path}")

    # 7) 指标 + 本数据集单独 CSV
    result = evaluate(gt, pred, dataset_name=name, gt_source=gt_source)
    result["method"] = f"StarDist(+{expand_px}px)" if (expand_px and expand_px > 0) else "StarDist(nuclei)"
    result["expand_px"] = int(expand_px or 0)
    per_csv = os.path.join(out_dir, f"{name}_stardist_metrics.csv")
    pd.DataFrame([result]).to_csv(per_csv, index=False)
    print(f"  指标已写入 {per_csv}")
    return result


def main():
    os.makedirs(PARENT_OUT, exist_ok=True)
    rows = []
    for cfg in DATASETS:
        try:
            rows.append(run_dataset(cfg))
        except Exception as e:
            import traceback
            print(f"[ERROR] 数据集 {cfg['name']} 失败: {e}")
            traceback.print_exc()

    if not rows:
        print("没有成功的数据集。")
        return

    df = pd.DataFrame(rows)
    combined = os.path.join(PARENT_OUT, "stardist_xenium_metrics_all.csv")
    df.to_csv(combined, index=False)

    pd.set_option("display.max_columns", None, "display.width", 220)
    print("\n================ 汇总 ================")
    print(df.to_string(index=False))
    print("\n各数据集 CSV 见各自 output_dir；汇总:", combined)
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="StarDist(核)+膨胀 × Xenium breast benchmark")
    ap.add_argument("--expand-px", type=int, default=None, help="核向外膨胀的像素数，覆盖 EXPAND_PX（0=不膨胀）")
    ap.add_argument("--expand-um", type=float, default=None, help="核向外膨胀的微米数，按 pixel_size 换算，覆盖 EXPAND_PX")
    ap.add_argument("--prob-thresh", type=float, default=None, help="StarDist 概率阈值，覆盖默认")
    ap.add_argument("--nms-thresh", type=float, default=None, help="StarDist NMS 阈值，覆盖默认")
    a, _ = ap.parse_known_args()
    if a.expand_px is not None:
        EXPAND_PX = a.expand_px
    if a.expand_um is not None:
        EXPAND_UM = a.expand_um
    if a.prob_thresh is not None:
        STARDIST["prob_thresh"] = a.prob_thresh
    if a.nms_thresh is not None:
        STARDIST["nms_thresh"] = a.nms_thresh
    main()