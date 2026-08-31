#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cellpose × Xenium 细胞分割 benchmark
====================================

一份可同时用在两个数据集上的 Cellpose 分割 + 指标评估脚本：

  1) xenium_roi_crop/  —— GT = manual_labels.npy（手动标注 label mask）
  2) xenium_breast/    —— GT = 官方 cell_boundaries.parquet 栅格化得到的 label mask

流程：读形态学图 → Cellpose 分割（预测） → 构建 GT → 形状对齐 → 调 test.py 的指标。

语义说明（重要）
----------------
test.py 里 evaluate_single_mask_against_cellpose(cellpose_gt_mask, pred_mask) 把第一个
槽当作 "reference"。在本 benchmark 中 **Cellpose 是被测方法，不是 reference**，所以我们把
真值（manual / 官方）放进第一个槽、Cellpose 预测放进第二个槽。这样：
  recall = GT 中被 Cellpose 检出的比例
  FP / over_segmentation = Cellpose 多分出来的细胞
  FN / under_segmentation = Cellpose 漏掉的 GT 细胞
指标方向即为「评估 Cellpose」的正确方向。输出里我们把误导性的字段名改成明确的名字。

依赖：numpy, pandas, scikit-image, scipy, tifffile, pyarrow, cellpose(>=4, cpsam)
      test.py 需与本脚本放在同一目录。
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
# 本脚本与 test.py 不在同一目录时，把下面改成 test.py 的绝对路径。
# 留空 "" 或路径不存在时，会自动退回「与本脚本同目录 / PYTHONPATH 里的 test.py」。
# ---------------------------------------------------------------------------- #
TEST_PY_PATH = "/data/qiuyijia/eval/test.py"   # ← 改成你的 test.py 绝对路径


def _load_metrics():
    if TEST_PY_PATH and os.path.exists(TEST_PY_PATH):
        spec = importlib.util.spec_from_file_location("xenium_metrics", TEST_PY_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)   # test.py 的 __main__ 块不会被触发（模块名非 __main__）
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


metrics = _load_metrics()  # test.py 中的指标实现


# ============================================================================ #
# 配置区：改这里
# ============================================================================ #
XENIUM_DEFAULT_PIXEL_SIZE = 0.2125  # µm/px，读不到 experiment 时的兜底值

# Cellpose 全局参数（Cellpose-SAM / 4.x）
CELLPOSE = dict(
    use_gpu=True,
    diameter=None,           # cpsam 尺度不敏感，None 即可；经典模型建议给值（细胞~60，核~30）
    flow_threshold=0.4,
    cellprob_threshold=0.0,
    batch_size=32,
    normalize=True,          # 对 Xenium 高动态范围 uint16 很关键，做百分位归一化
    # 若装的是经典 Cellpose(3.x) 而非 cpsam，把下面设 True，并指定 model_type/channels
    use_classic=False,
    classic_model_type="cyto3",
    classic_channels=[2, 1],  # [cytoplasm, nucleus]：1=red,2=green,3=blue；见经典 Cellpose 文档
)

# 通道约定：Xenium morphology_focus 通道 0 = DAPI(核)，通道 1 = 膜/边界染色(ATP1A1/CD45/E-cad)
NUCLEUS_IDX = 0
MEMBRANE_IDX = 1

DATASETS = [
    dict(
        name="xenium_mouse_brain",
        dataset_dir="/data/qiuyijia/dataset/xenium_mouse_brain/outs",  # 注意：路径到outs/
        output_dir="/data/qiuyijia/cellpose/xenium_mouse_brain_output",
        gt_type="xenium_boundaries",
        boundaries="cell_boundaries.parquet",
        experiment="experiment.xenium",
        pyramid_level=0,
        use_membrane=True,
        crop=(12077, 22077, 6956, 16956),
    ),
]
OUTPUT_DIR = "/data/qiuyijia/cellpose/xenium_out"
BOUNDARY_MODE = "thick"
IOU_THRESHOLD = 0.5
# ============================================================================ #


# ---------------------------------------------------------------------------- #
# 形态学图像读取（支持金字塔层 + 两种命名 + 单/多通道文件）
# ---------------------------------------------------------------------------- #
def _read_level(path, pyramid_level):
    """读一个 OME-TIFF 的指定金字塔层，返回 (array, downsample_factor)。

    factor = 全分辨率高 / 当前层高（用真实 shape 计算，避免假设 2 的幂）。
    """
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
    """把一个数组拆成通道列表。2D→单通道；3D→按小轴(<=5)判定通道轴。"""
    if arr.ndim == 2:
        return [arr]
    if arr.ndim == 3:
        if arr.shape[0] <= 5 and arr.shape[0] <= arr.shape[-1]:
            return [arr[i] for i in range(arr.shape[0])]        # (C,H,W)
        if arr.shape[-1] <= 5:
            return [arr[..., i] for i in range(arr.shape[-1])]  # (H,W,C)
        return [arr[arr.shape[0] // 2]]                          # 疑似 z-stack，取中间层
    # 更高维：压掉前面的维度后取中间
    arr = np.squeeze(arr)
    return _split_channels(arr) if arr.ndim <= 3 else [np.max(arr, axis=tuple(range(arr.ndim - 2)))]


def load_morphology_channels(dataset_dir, pyramid_level=0):
    """发现并读取形态学通道。

    优先 morphology_focus/ 目录；文件名支持
        ch0000_dapi.ome.tif ...            （breast 命名）
        morphology_focus_0000.ome.tif ...  （ROI 命名）
    单个文件本身若是多通道，会被展开。找不到目录则退回 morphology.ome.tif。

    返回: dict(channels=[2D...], names=[...], factor=float)
    """
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

    # 兜底：morphology.ome.tif（通常是 DAPI，可能是 z-stack）
    for cand in ("morphology.ome.tif", "morphology_focus.ome.tif", "morphology_mip.ome.tif"):
        fp = os.path.join(dataset_dir, cand)
        if os.path.exists(fp):
            arr, factor = _read_level(fp, pyramid_level)
            chs = _split_channels(arr)
            if len(chs) > 1:  # 多为 z-stack，做最大投影当单通道核信号
                chs = [np.max(np.stack(chs, 0), 0)]
            warnings.warn(f"{dataset_dir}: 未找到 morphology_focus/，退回 {cand}（仅核通道）")
            return dict(channels=chs, names=[cand], factor=factor)

    raise FileNotFoundError(f"{dataset_dir}: 找不到任何形态学 OME-TIFF")


def assemble_cellpose_input(channels, use_membrane=True,
                            nucleus_idx=NUCLEUS_IDX, membrane_idx=MEMBRANE_IDX):
    """把通道拼成 Cellpose 输入。

    返回 (img, mode)：
      - 有膜通道且 use_membrane → (H,W,2) float32，[核, 膜]，用于全细胞分割
      - 否则 → (H,W) float32 灰度（核），只能分核
    Cellpose-SAM 内部自带归一化，这里只做类型转换、不改动强度。
    """
    n = len(channels)
    nuc = channels[nucleus_idx] if nucleus_idx < n else channels[0]
    nuc = nuc.astype(np.float32)

    if use_membrane and membrane_idx < n:
        mem = channels[membrane_idx].astype(np.float32)
        img = np.stack([nuc, mem], axis=-1)   # (H,W,2)
        return img, "nucleus+membrane"
    return nuc, "nucleus_only"


# ---------------------------------------------------------------------------- #
# Cellpose 分割
# ---------------------------------------------------------------------------- #
def run_cellpose(img, cfg=CELLPOSE):
    """跑 Cellpose，返回 int32 label mask。默认 Cellpose-SAM(4.x)，可切经典 3.x。"""
    from cellpose import models  # 延迟导入，便于无 cellpose 环境下 import 本模块做别的

    common = dict(
        flow_threshold=cfg["flow_threshold"],
        cellprob_threshold=cfg["cellprob_threshold"],
        normalize=cfg["normalize"],
        diameter=cfg["diameter"],
    )

    if cfg.get("use_classic", False):
        # 经典 Cellpose 3.x：需要 model_type 与 channels
        model = models.Cellpose(gpu=cfg["use_gpu"], model_type=cfg["classic_model_type"])
        # 经典模型要求通道顺序为 [cytoplasm, nucleus]；把 (H,W,2)=[核,膜] 调成 [膜,核]
        if img.ndim == 3 and img.shape[-1] == 2:
            img_in = img[..., ::-1]
        else:
            img_in = img
        out = model.eval(img_in, channels=cfg["classic_channels"],
                         batch_size=cfg["batch_size"], **common)
    else:
        # Cellpose-SAM 4.x：CellposeModel 默认加载 cpsam，无需 channels
        model = models.CellposeModel(gpu=cfg["use_gpu"])
        try:
            out = model.eval(img, batch_size=cfg["batch_size"], **common)
        except TypeError:
            # 个别版本签名差异：去掉可能不支持的 kwargs 再试
            out = model.eval(img, **common)

    masks = out[0] if isinstance(out, (list, tuple)) else out
    masks = np.squeeze(np.asarray(masks))
    return masks.astype(np.int32, copy=False)


# ---------------------------------------------------------------------------- #
# GT 构建
# ---------------------------------------------------------------------------- #
def read_pixel_size(experiment_path, default=XENIUM_DEFAULT_PIXEL_SIZE):
    """从 experiment.xenium / experiment_roi.json（均为 JSON）递归找 pixel_size(µm/px)。"""
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


def _boundary_columns(df):
    """定位边界表里的 (cell_id, x, y) 三列，兼容常见命名。"""
    cols = {c.lower(): c for c in df.columns}
    id_col = next((cols[c] for c in ("cell_id", "label_id", "label", "id") if c in cols), None)
    # 优先使用 local 坐标（ROI 相对坐标），没有才用全局坐标
    x_col = next((cols[c] for c in ("vertex_x_local_um", "vertex_x_local", "vertex_x", "x", "x_location", "vertex_x_um") if c in cols), None)
    y_col = next((cols[c] for c in ("vertex_y_local_um", "vertex_y_local", "vertex_y", "y", "y_location", "vertex_y_um") if c in cols), None)
    return id_col, x_col, y_col


def rasterize_boundaries(parquet_path, out_shape, pixel_size_eff, offset=(0, 0), verbose=True):
    """把 Xenium cell_boundaries.parquet（微米坐标多边形）栅格化成 int32 label mask。

    out_shape        : (H, W)，输出 mask 尺寸（若裁剪，则为裁剪区域尺寸）
    pixel_size_eff   : 该层每像素的微米数 = base_pixel_size × downsample_factor
    offset           : (row0, col0)，裁剪区域左上角在整图中的像素坐标；未裁剪时为 (0, 0)
    像素坐标 = 微米 / pixel_size_eff - offset；越界部分由 skimage 自动裁掉
    """
    row0, col0 = offset
    df = pd.read_parquet(parquet_path)
    id_col, x_col, y_col = _boundary_columns(df)
    if id_col is None or x_col is None or y_col is None:
        raise ValueError(f"{parquet_path}: 无法识别列 (cell_id/vertex_x/vertex_y)，实际列={list(df.columns)}")

    H, W = out_shape
    mask = np.zeros((H, W), dtype=np.int32)

    xs_all = df[x_col].to_numpy(dtype=np.float64) / pixel_size_eff - col0  # 列(x)，减裁剪偏移
    ys_all = df[y_col].to_numpy(dtype=np.float64) / pixel_size_eff - row0  # 行(y)，减裁剪偏移
    ids_all = df[id_col].to_numpy()

    # factorize 按出现顺序给每行一个 cell 编码；再按编码 stable 排序，使同一 cell 的顶点连续
    # （stable 排序保留同一 cell 内顶点的原始顺序 = 多边形绕行顺序）。O(N logN)，避免逐 cell 全量扫描。
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
        rr, cc = sk_polygon(ys_s[s:e], xs_s[s:e], shape=(H, W))  # 自动裁剪到图像范围
        if rr.size:
            mask[rr, cc] = label
        if verbose and label % 50000 == 0:
            print(f"    栅格化 {label} 个细胞 ...")
    if verbose:
        print(f"    栅格化完成，共 {label} 个细胞 → mask {mask.shape}")
    return mask

# ---------------------------------------------------------------------------- #
# 形状对齐 + 评估
# ---------------------------------------------------------------------------- #
def align_shapes(gt, pred):
    """GT 与预测理论上同网格（同一张图构建）。若因取整差 1~2 像素不一致，右下角补零对齐。"""
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
    """调 test.py 的指标：GT 放 reference 槽，Cellpose 放 pred 槽；再改成无歧义的字段名。"""
    gt_mask, pred_mask = align_shapes(gt_mask, pred_mask)

    raw = metrics.evaluate_single_mask_against_cellpose(
        method_name="Cellpose",
        cellpose_gt_mask=gt_mask,      # ← 真值放这里
        pred_mask=pred_mask,          # ← Cellpose 预测放这里
        boundary_mode=boundary_mode,
        iou_threshold=iou_threshold,
    )

    result = {
        "dataset": dataset_name,
        "gt_source": gt_source,        # manual_labels / xenium_official_boundaries
        "method": "Cellpose",
        "gt_cell_count": raw.pop("cellpose_reference_cell_count"),  # 改名：这是 GT 细胞数
        "pred_cell_count": raw.pop("pred_cell_count"),
    }
    raw.pop("reference", None)         # 丢掉误导性的 "reference": "Cellpose"
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


def run_dataset(cfg, out_dir):
    name = cfg["name"]
    ddir = cfg["dataset_dir"]
    print(f"\n=== [{name}] {ddir} ===")

    # 1) 读形态学通道 + 拼 Cellpose 输入（此时是整图）
    morph = load_morphology_channels(ddir, pyramid_level=cfg.get("pyramid_level", 0))
    factor = morph["factor"]
    print(f"  通道数={len(morph['channels'])}  名称={morph['names']}  下采样因子={factor:.3f}")

    img_full, in_mode = assemble_cellpose_input(morph["channels"], use_membrane=cfg.get("use_membrane", True))
    Himg, Wimg = img_full.shape[:2]

    # 决定裁剪窗口：显式 crop=(y0,y1,x0,x1) 优先；否则若给了 crop_size 则取「居中」正方形
    crop = cfg.get("crop")
    if crop is None and cfg.get("crop_size"):
        s = int(cfg["crop_size"])
        y0 = max((Himg - s) // 2, 0)
        x0 = max((Wimg - s) // 2, 0)
        crop = (y0, min(y0 + s, Himg), x0, min(x0 + s, Wimg))
        print(f"  crop_size={s} → 居中裁剪 crop={crop}（整图 {Himg}×{Wimg}）")

    img = _apply_crop(img_full, crop)
    del img_full
    seg_shape = img.shape[:2]
    print(f"  Cellpose 输入: shape={img.shape} 模式={in_mode}")

    # 2) Cellpose 分割（预测）
    print("  运行 Cellpose ...")
    pred = run_cellpose(img, CELLPOSE)
    print(f"  预测细胞数={len(metrics.get_instance_ids(pred))}  mask={pred.shape}")

    # 3) 构建 GT（与预测使用同一裁剪窗口，保证对齐）
    if cfg["gt_type"] == "xenium_boundaries":
        exp_path = os.path.join(ddir, cfg.get("experiment", "experiment.xenium"))
        base_px = read_pixel_size(exp_path)
        px_eff = base_px * factor
        print(f"  pixel_size={base_px} µm/px  当前层 pixel_size_eff={px_eff:.4f} µm/px")
        if crop is not None:
            y0, y1, x0, x1 = crop
            # 直接把多边形栅格化进「裁剪坐标系」：减左上角偏移、只输出裁剪尺寸
            # （高效：不建整图大 mask；正确：非左上角窗口也不会错位）
            gt = rasterize_boundaries(os.path.join(ddir, cfg["boundaries"]),
                                      out_shape=(y1 - y0, x1 - x0),
                                      pixel_size_eff=px_eff, offset=(y0, x0))
        else:
            gt = rasterize_boundaries(os.path.join(ddir, cfg["boundaries"]),
                                      out_shape=seg_shape, pixel_size_eff=px_eff)
        gt_source = "xenium_official_boundaries"
    else:
        raise ValueError(f"未知 gt_type: {cfg['gt_type']}")
    print(f"  GT 细胞数={len(metrics.get_instance_ids(gt))}  来源={gt_source}")

    # 4) 落盘 mask（int32 tif，兼容 test.py 的 read_tif_mask，便于复现/对拍）
    os.makedirs(out_dir, exist_ok=True)
    pred_path = os.path.join(out_dir, f"{name}_cellpose_pred.tif")
    gt_path = os.path.join(out_dir, f"{name}_gt.tif")
    tifffile.imwrite(pred_path, pred)
    tifffile.imwrite(gt_path, gt)
    print(f"  已保存 {pred_path} / {gt_path}")

    # 5) 指标
    result = evaluate(gt, pred, dataset_name=name, gt_source=gt_source)
    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = []
    for cfg in DATASETS:
        try:
            rows.append(run_dataset(cfg, OUTPUT_DIR))
        except Exception as e:
            import traceback
            print(f"[ERROR] 数据集 {cfg['name']} 失败: {e}")
            traceback.print_exc()

    if not rows:
        print("没有成功的数据集。")
        return

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "cellpose_xenium_metrics.csv")
    df.to_csv(csv_path, index=False)

    pd.set_option("display.max_columns", None, "display.width", 200)
    print("\n================ 汇总 ================")
    print(df.to_string(index=False))
    print("\nSaved:", csv_path)
    return df


if __name__ == "__main__":
    main()