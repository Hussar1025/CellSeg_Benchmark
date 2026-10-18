#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bering segmentation for MERFISH center-ROI benchmark inputs.

    The runner reads a Vizgen ``detected_transcripts.csv`` (or the prepared
    ``_roi_cache/tx_roi_*_<roi_pixels>.parquet``), constructs Bering's coarse
    ``raw_cells/raw_labels`` input, trains Bering, and writes transcript-level and
    cell-level outputs.  ``grid`` is a deterministic image-free prior;
    ``cellpose`` uses an existing Cellpose instance mask in the same center ROI;
    ``metadata`` and ``polygon`` are retained for reference-assisted experiments.

Each invocation is restricted to the configured center ROI (10,000 x 10,000
pixels by default) when a cache is absent (using the mosaic transform and TIFF
header).  Bering's image path is disabled by default because the benchmark is
transcript-only; image use can be enabled in the CONFIG class when an
image-assisted experiment is explicitly desired.

Outputs include ``segmentation_transcripts.csv``, two h5ad count matrices,
cell metadata, and ``bering_resource_summary.json``.  The launcher
``run_bering_merfish_center10000.sh`` runs the requested datasets serially and
adds GNU ``time -v`` resource reports to each log.
"""

import os
# Do not select a GPU at import time.  The launcher can set
# ``CUDA_VISIBLE_DEVICES`` before Python starts, or the command line can use
# ``--cuda-visible-devices``.  The old version hard-coded GPU 1 here, which
# made the script silently use the wrong card on shared machines.
import gc
import glob
import random
import logging
import sys
import argparse
import json
import re
import time
import resource
from pathlib import Path
from typing import Optional


# Python 3.7 没有标准库 importlib.metadata，而新版 umap-learn 会直接导入它。
# 优先使用标准库；在旧环境中回退到已安装的 backport，并注册点号模块路径，
# 这样 Scanpy 通过近似邻居路径间接导入 UMAP 时仍能工作。
def _ensure_importlib_metadata():
    try:
        import importlib.metadata  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "importlib.metadata":
            raise
        try:
            import importlib_metadata as metadata
        except ImportError as backport_exc:
            raise RuntimeError(
                "当前 Python 缺少 importlib.metadata。请先执行 "
                "conda activate bering，并安装 importlib-metadata，或升级到 Python >= 3.8。"
            ) from backport_exc
        sys.modules["importlib.metadata"] = metadata
        import importlib
        importlib.metadata = metadata


_ensure_importlib_metadata()

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("bering_seg")


# ==============================================================================
# 0. 配置 (CONFIG) —— 按你的实际路径 / 需求修改这里
# ==============================================================================
class CONFIG:
    # ---- 输入路径 ----
    DATA_DIR = "/data/qiuyijia/dataset/merfish_Melanoma"
    TRANSCRIPTS_CSV = os.path.join(
        DATA_DIR,
        "detected_transcripts.csv",
    )
    BOUNDARY_DIR = os.path.join(DATA_DIR, "cell_boundaries")   # feature_data_{fov}.hdf5

    # ---- 输出路径 ----
    OUT_DIR = "/data/qiuyijia/bering/merfish_Melanoma_center10000_output"
    CACHE_INPUT = os.path.join(OUT_DIR, "bering_input_transcripts.parquet")  # 预处理缓存

    # Coarse prior used to construct Bering's required raw_cells/raw_labels.
    # ``cellpose`` reads an existing center-ROI Cellpose instance mask and
    # assigns every transcript to the label under its micron->mosaic-pixel
    # coordinate.  ``polygon``/``metadata`` use Vizgen reference files.
    PRIOR = "cellpose"             # cellpose | grid | polygon | metadata
    GRID_SIZE_UM = 18.0
    METADATA_RADIUS_UM = 25.0
    ROI_CACHE = None                # auto-discover *_roi_*_10000.parquet
    CELLPOSE_MASK = None            # auto: /data/qiuyijia/cellpose/merfish_output/<dataset>/pred_cellpose_roi.tif
    CELLPOSE_MASK_ROOT = "/data/qiuyijia/cellpose/merfish_output"

    # ---- 复现实验用随机种子 ----
    SEED = 0

    # ---- 试跑开关: 先只用少数 FOV 跑通全流程, 再放开做全片 ----
    #      设为 None 表示用全部 FOV; 或给一个 fov 列表, e.g. list(range(0, 60))
    SUBSET_FOVS: Optional[list] = None

    # ---- 中心 ROI: 取 mosaic 图像中心的 ROI_PIXELS x ROI_PIXELS 像素窗口, ----
    #      再用 micron<->pixel 变换换算成【微米(组学坐标)】bbox 去裁转录本。
    #      USE_ROI=True 时优先于 SUBSET_FOVS。只需图像尺寸 + 变换矩阵(都不加载像素)。
    USE_ROI = True
    ROI_PIXELS = 10000

    # ---- 转录本预处理 ----
    DROP_BLANK = True          # 去掉 Blank-* 阴性对照探针
    HDF5_ZINDEX = 3            # 用哪个 z 平面的多边形做 2D 足迹 (readVizgen 默认 z=3)

    # ---- 生成 raw_labels 的粗聚类 (输入准备, 不是 benchmark) ----
    MIN_COUNTS_PER_CELL = 20   # grid prior is deliberately coarse; 50 drops sparse ROIs
    LEIDEN_RES_FOR_LABELS = 1.0  # 粗 cell type 的分辨率; 想更粗可调小

    # ---- 是否使用图像 (DAPI/polyT) ----
    #      默认 False(image-free)。本数据集只下了单个 z(z3)的 mosaic, 且需要
    #      micron->pixel 变换; image-free 是最干净、与官方 Xenium 例子一致的做法。
    USE_IMAGE = False
    IMG_DIR = os.path.join(DATA_DIR, "images")
    IMG_FILES = [  # (通道名, 文件名)
        ("DAPI",  "mosaic_DAPI_z3.tif"),
        ("PolyT", "mosaic_PolyT_z3.tif"),
    ]
    IMG_TRANSFORM_CSV = os.path.join(
        IMG_DIR,
        "micron_to_mosaic_pixel_transform.csv",
    )

    # ---- Bering: 训练图构建 ----
    N_CELLS_PER_CLASS = 10     # 每个类别采样多少 cell 用于训练
    WINDOW_WIDTH = 15.0        # 训练窗口宽 (微米) —— image-free 时单位是微米
    WINDOW_HEIGHT = 15.0
    N_NEIGHBORS = 30           # KNN 邻居数 (训练图 & 全片节点分类都用)

    # ---- Bering: 数据切分 ----
    BATCH_SIZE = 16
    TRAINING_RATIO = 0.8

    # ---- Bering: 训练轮数 ----
    NODE_EPOCHES = 50
    EDGE_EPOCHES = 50

    # ---- Bering: 全片节点分类 ----
    NC_PROB_THRESHOLD = 0.3    # 低于该概率的转录本判为背景
    NC_NUM_CHUNKS = 25         # 转录本过多时按坐标切块, 防止 OOM

    # ---- Bering: 细胞分割 (连边预测 + Leiden) ----
    SEG_POSITIVE_EDGE_THRESH = 0.6   # 连边正阈值: 调低->更多边->细胞更大更少; 调高->更碎+更多单点
    SEG_LEIDEN_RESOLUTION = 0.05     # 单次分割用的分辨率(仅跑一次拿到 edge 预测; 之后被下面的扫描覆盖)
    # 300 (the package default) creates billions of edges for the 7 ROI's
    # here.  60 is a reproducible, memory-bounded setting for these dense
    # MERFISH ROIs; raise it only after checking the recorded peak memory.
    SEG_NUM_EDGES_PER_SPOT = 60
    SEG_GRAPH_N_NEIGHBORS = 10
    SEG_NUM_ITERS = 100

    # ---- 决定"细胞数"的真正旋钮: 分割时的 Leiden(CPM)分辨率 ----
    #  细胞数由分割分辨率决定, 不是由 cell_annotation 质控决定。分辨率越高 -> 簇越多 -> 细胞越多。
    #  开启 MATCH_COARSE_CELL_COUNT 时, 脚本会【复用一次分割得到的连边预测】在下列分辨率上
    #  重新聚类(不重训练), 评估每个分辨率下的"可用细胞数上限", 选最接近目标的那个。
    SEG_RESOLUTION_SWEEP = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    # ---- Bering: cell_annotation 质控阈值 ----
    #  cell_annotation 只是对已算好的 predicted_cells 再质控筛选(去单点/背景主导/超大团)。
    #  它只能【减少】细胞数, 不能增加。要增加细胞数请调分割分辨率(上面)。
    MATCH_COARSE_CELL_COUNT = True
    TARGET_CELLS: Optional[int] = None   # None = 自动用粗分割细胞数; 也可填具体数字
    ANNOT_MIN_DOMINANT_RATIO = 0.6       # 百分比阈值(源码 .mul(100)), 0.6≈只去 background 主导
    ANNOT_MAX_DIAM_RATIO = 5.0           # 直径 sanity 过滤
    ANNOT_MDN_RATIO = 0.5                # min_dominant_nodes = 0.5 * min_transcripts
    # 关闭自动匹配时用的手动阈值:
    ANNOT_MIN_TRANSCRIPTS = 50
    ANNOT_MIN_DOMINANT_NODES = 30


# ==============================================================================
# 0b. 中心 ROI: 图像像素中心窗口 -> 微米(组学坐标) bbox
# ============================================================================== 
def _find_image_file(cfg: CONFIG, hint: str = "DAPI") -> Optional[str]:
    """Find the DAPI mosaic despite Vizgen exports using ``images`` or ``imgs``."""
    roots = [cfg.DATA_DIR, os.path.join(cfg.DATA_DIR, "images"),
             os.path.join(cfg.DATA_DIR, "imgs")]
    candidates = []
    for root in roots:
        if os.path.isdir(root):
            candidates.extend(glob.glob(os.path.join(root, "**", "*.tif"), recursive=True))
    named = [p for p in candidates if hint.lower() in os.path.basename(p).lower()]
    return sorted(named)[0] if named else None


def _find_transform_file(cfg: CONFIG) -> Optional[str]:
    candidates = []
    for root in (cfg.DATA_DIR, os.path.join(cfg.DATA_DIR, "images"),
                 os.path.join(cfg.DATA_DIR, "imgs")):
        if os.path.isdir(root):
            candidates.extend(glob.glob(os.path.join(root, "**", "*micron*transform*.csv"),
                                        recursive=True))
    return sorted(candidates)[0] if candidates else None


def _find_roi_cache(cfg: CONFIG) -> Optional[str]:
    """Return the unique prepared 10k ROI cache, if present.

    The benchmark preparation already created these files using the actual
    mosaic transform.  Reusing them is both faster and less error-prone than
    rescanning 10--40 GB CSV files for every rerun.
    """
    if cfg.ROI_CACHE:
        return cfg.ROI_CACHE if os.path.exists(cfg.ROI_CACHE) else None
    cache_dir = os.path.join(cfg.DATA_DIR, "_roi_cache")
    paths = sorted(glob.glob(os.path.join(cache_dir, "tx_roi_*_%d.parquet" % cfg.ROI_PIXELS)))
    if len(paths) == 1:
        return paths[0]
    if len(paths) > 1:
        raise RuntimeError("发现多个 ROI 缓存，请用 --roi-cache 指定一个: %s" % paths)
    return None


def _find_cellpose_mask(cfg: CONFIG) -> Optional[str]:
    """Find the Cellpose instance mask corresponding to this dataset's center ROI.

    Cellpose's MERFISH benchmark runner writes one 10000x10000 mask per dataset
    under ``/data/qiuyijia/cellpose/merfish_output/<dataset>/``.  An explicit
    ``CONFIG.CELLPOSE_MASK`` always takes precedence; the fallback locations
    keep the prior usable for the existing output layout without changing the
    command line used for the other priors.
    """
    if cfg.CELLPOSE_MASK:
        path = os.path.abspath(cfg.CELLPOSE_MASK)
        return path if os.path.exists(path) else None

    dataset = Path(cfg.DATA_DIR).name
    candidates = [
        os.path.join(cfg.CELLPOSE_MASK_ROOT, dataset, "pred_cellpose_roi.tif"),
        os.path.join(cfg.CELLPOSE_MASK_ROOT, "pred_cellpose_roi.tif"),
        os.path.join(os.path.dirname(cfg.CELLPOSE_MASK_ROOT), dataset,
                     "pred_cellpose_roi.tif"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _center_roi_pixel_bbox(cfg: CONFIG):
    """Return the center ROI pixel bbox as ``(x0, y0, x1, y1)``.

    The Cellpose benchmark creates its mask with the same center-ROI rule.
    This fallback is used when the transcript cache is absent; with a cache,
    ``_roi_cache_metadata`` below provides the exact registered origin.
    """
    import tifffile as tiff

    image_path = _find_image_file(cfg, "DAPI")
    if image_path is None:
        raise FileNotFoundError("无法定位 DAPI TIFF，不能确定 Cellpose mask 的 ROI 原点")
    with tiff.TiffFile(image_path) as tf:
        H, W = tf.pages[0].shape[:2]
    size = int(cfg.ROI_PIXELS)
    if size <= 0 or size > H or size > W:
        raise ValueError("ROI_PIXELS=%d 超出 DAPI 图像尺寸 H=%d,W=%d" % (size, H, W))
    x0 = max(0, W // 2 - size // 2)
    y0 = max(0, H // 2 - size // 2)
    return x0, y0, x0 + size, y0 + size


def _cellpose_roi_pixel_bbox(cfg: CONFIG, mask_path: str):
    """Resolve the mask origin, preferring the prepared transcript-cache tag."""
    cache = _find_roi_cache(cfg)
    if cache is not None:
        meta = _roi_cache_metadata(cache)
        if meta and meta.get("size_px"):
            x0 = int(meta["pixel_x0"])
            y0 = int(meta["pixel_y0"])
            size = int(meta["size_px"])
            return x0, y0, x0 + size, y0 + size
    return _center_roi_pixel_bbox(cfg)


def _roi_cache_metadata(path: str):
    m = re.search(r"tx_roi_(\d+)_(\d+)_(\d+)(?:_[^/]*)?\.parquet$", os.path.basename(path))
    if not m:
        return {}
    y0, x0, size = map(int, m.groups())
    return {"pixel_y0": y0, "pixel_x0": x0, "size_px": size}


def _read_transform_matrix(path: str) -> np.ndarray:
    """读 micron_to_mosaic_pixel_transform.csv, 返回 3x3 仿射矩阵 M:
       [x_pix, y_pix, 1]^T = M @ [x_um, y_um, 1]^T (即 micron -> mosaic pixel)。"""
    M = pd.read_csv(path, header=None, sep=r"\s+|,", engine="python").values.astype(float)
    return M[:3, :3]


def compute_roi_um_bbox(cfg: CONFIG):
    """取 mosaic 图像中心 ROI_PIXELS x ROI_PIXELS 的像素窗口, 用逆变换换算成微米 bbox。
    只读 TIFF 头拿到 H/W(不加载像素), 用 4 个角点做逆变换(可正确处理缩放/旋转)。
    返回 (xmin_um, xmax_um, ymin_um, ymax_um)。"""
    import tifffile as tiff

    transform = _find_transform_file(cfg)
    image_path = _find_image_file(cfg)
    if transform is None or image_path is None:
        raise FileNotFoundError("无法计算 ROI：未找到 micron_to_mosaic_pixel_transform.csv 或 DAPI TIFF")
    M = _read_transform_matrix(transform)
    Minv = np.linalg.inv(M)

    with tiff.TiffFile(image_path) as tf:
        H, W = tf.pages[0].shape[:2]           # 行=y=H, 列=x=W; 仅读头, 不载像素
    log.info("mosaic 图像尺寸: H=%d, W=%d; 变换矩阵:\n%s", H, W, M)

    cx_px, cy_px = W / 2.0, H / 2.0
    half = cfg.ROI_PIXELS / 2.0
    corners_px = np.array([
        [cx_px - half, cy_px - half, 1.0],
        [cx_px + half, cy_px - half, 1.0],
        [cx_px - half, cy_px + half, 1.0],
        [cx_px + half, cy_px + half, 1.0],
    ]).T                                        # (3, 4)
    corners_um = Minv @ corners_px              # (3, 4)
    xs, ys = corners_um[0], corners_um[1]
    xmin, xmax, ymin, ymax = xs.min(), xs.max(), ys.min(), ys.max()
    log.info("中心 %dx%d 像素 ROI -> 微米 bbox: x[%.1f, %.1f] (%.1f um) y[%.1f, %.1f] (%.1f um)",
             cfg.ROI_PIXELS, cfg.ROI_PIXELS, xmin, xmax, xmax - xmin, ymin, ymax, ymax - ymin)
    return xmin, xmax, ymin, ymax


# ==============================================================================
# 1. 读取转录本
# ==============================================================================
def load_transcripts(cfg: CONFIG) -> pd.DataFrame:
    """读 detected_transcripts.csv, 只取需要的列, 返回含 global_x/global_y/global_z/fov/gene
    (以及若存在则 cell-id 列) 的 DataFrame。"""
    cache = _find_roi_cache(cfg)
    if cache is not None:
        log.info("读取已准备的中心 ROI 缓存: %s", cache)
        try:
            import pyarrow.parquet as pq
            cols = list(pq.ParquetFile(cache).schema.names)
        except Exception:
            # Fallback for environments without pyarrow metadata access.
            try:
                from fastparquet import ParquetFile
                cols = list(ParquetFile(cache).columns)
            except Exception:
                # ``read_parquet`` has no portable ``nrows`` argument.  This
                # last-resort path materializes the table only in an unusual
                # environment lacking both parquet metadata readers.
                cols = list(pd.read_parquet(cache, engine="auto").columns)
        wanted = [c for c in ("global_x", "global_y", "global_z", "fov", "gene",
                              "transcript_id", "cell_id") if c in cols]
        df = pd.read_parquet(cache, columns=wanted)
        if "global_x" not in df or "global_y" not in df or "gene" not in df:
            raise ValueError("ROI 缓存必须包含 global_x/global_y/gene 列")
        log.info("  ROI 缓存转录本: %d", len(df))
        # Continue through the normal column normalization below.
        header = None
        cols = list(df.columns)
    else:
        log.info("读取转录本: %s", cfg.TRANSCRIPTS_CSV)
        header = pd.read_csv(cfg.TRANSCRIPTS_CSV, nrows=0)
        cols = list(header.columns)
    log.info("  文件列名: %s", cols)

    gene_col = "gene" if "gene" in cols else None
    gx = "global_x" if "global_x" in cols else None
    gy = "global_y" if "global_y" in cols else None
    gz = "global_z" if "global_z" in cols else None
    fov = "fov" if "fov" in cols else None
    if gene_col is None or gx is None or gy is None:
        raise ValueError("detected_transcripts.csv 缺少 gene/global_x/global_y 列, 无法继续。")

    # 某些经 VPT 重跑的导出会自带 cell-id 列; 若有则可跳过 point-in-polygon
    cellid_col = next((c for c in ["cell_id", "cell", "cellID", "CellID"] if c in cols), None)

    if cache is None:
        usecols = [c for c in [gx, gy, gz, fov, gene_col, cellid_col] if c is not None]
        # Chunking prevents a 39 GB CSV from requiring another 40+ GB copy.
        pieces = []
        if cfg.USE_ROI:
            xmin, xmax, ymin, ymax = compute_roi_um_bbox(cfg)
        for part in pd.read_csv(cfg.TRANSCRIPTS_CSV, usecols=usecols, chunksize=2_000_000):
            if cfg.USE_ROI:
                part = part[(part[gx] >= xmin) & (part[gx] <= xmax) &
                            (part[gy] >= ymin) & (part[gy] <= ymax)]
            if len(part):
                pieces.append(part)
        if not pieces:
            raise RuntimeError("中心 ROI 内没有转录本")
        df = pd.concat(pieces, ignore_index=True)

    # 统一列名
    df = df.rename(columns={gx: "global_x", gy: "global_y", gene_col: "gene"})
    if gz is not None:
        df = df.rename(columns={gz: "global_z"})
    else:
        df["global_z"] = 0.0
    if fov is not None:
        df = df.rename(columns={fov: "fov"})
    else:
        df["fov"] = -1
    if cellid_col is not None:
        df = df.rename(columns={cellid_col: "vizgen_cell_id"})
        log.info("  检测到自带 cell-id 列 '%s' -> 将直接使用, 跳过 point-in-polygon。", cellid_col)

    # 去阴性对照探针
    if cfg.DROP_BLANK:
        before = len(df)
        mask = ~df["gene"].astype(str).str.lower().str.startswith(("blank", "notarget", "negcontrol"))
        df = df[mask].copy()
        log.info("  去除 Blank/对照探针: %d -> %d", before, len(df))

    # CSV fallback was cropped while streaming above.  For a cache, record its
    # exact geometry and perform a defensive crop only if requested explicitly.
    if cfg.USE_ROI and cache is not None:
        log.info("  使用缓存中已按 %d×%d 像素裁好的中心 ROI", cfg.ROI_PIXELS, cfg.ROI_PIXELS)
    elif cfg.SUBSET_FOVS is not None:
        df = df[df["fov"].isin(cfg.SUBSET_FOVS)].copy()
        log.info("  [试跑] 仅保留 %d 个 FOV, 剩余转录本 %d", len(cfg.SUBSET_FOVS), len(df))

    df = df.reset_index(drop=True)
    # ``transcript_id`` in Vizgen MERFISH is a transcript/gene isoform ID and
    # is intentionally repeated for every molecule.  Add a unique, stable key
    # within this center-ROI table so benchmark code can join predictions back
    # without relying on the non-unique transcript_id column.
    if "roi_transcript_index" not in df.columns:
        df["roi_transcript_index"] = np.arange(len(df), dtype=np.int64)
    log.info("  转录本总数: %d, 基因数: %d", len(df), df["gene"].nunique())
    return df


# ==============================================================================
# 2. 转录本 -> cell (raw_cells): 用 cell_boundaries 的多边形做 point-in-polygon
# ==============================================================================
def _load_fov_polygons(h5_path: str, zindex: int):
    """从单个 feature_data_{fov}.hdf5 读出 {cell_id_str: shapely.Polygon} (微米坐标)。
    Vizgen hdf5 结构: featuredata / <cell_id> / zIndex_<z> / p_0 / coordinates (1,N,2)。"""
    import h5py
    from shapely.geometry import Polygon

    polys, ids = [], []
    with h5py.File(h5_path, "r") as f:
        root = f["featuredata"] if "featuredata" in f else f
        for cell_id in root.keys():
            g = root[cell_id]
            # 选 z 平面: 优先指定的 zindex, 否则取可用 zIndex 的中间一个
            zkeys = [k for k in g.keys() if k.startswith("zIndex")]
            if not zkeys:
                continue
            zkey = "zIndex_%d" % zindex
            if zkey not in g:
                zkeys_sorted = sorted(zkeys, key=lambda s: int(s.split("_")[-1]))
                zkey = zkeys_sorted[len(zkeys_sorted) // 2]
            try:
                coords = np.asarray(g[zkey]["p_0"]["coordinates"][:]).squeeze()
            except Exception:
                continue
            if coords.ndim != 2 or coords.shape[0] < 3:
                continue
            try:
                poly = Polygon(coords)
                if not poly.is_valid:
                    poly = poly.buffer(0)          # 修复自相交
                if poly.is_empty or poly.area <= 0:
                    continue
            except Exception:
                continue
            polys.append(poly)
            ids.append(str(cell_id))
    return polys, ids


def assign_transcripts_to_cells(df: pd.DataFrame, cfg: CONFIG) -> pd.DataFrame:
    """Create Bering's coarse ``raw_cells`` prior.

    ``grid`` is deterministic and does not read benchmark reference
    boundaries. ``cellpose`` uses an external image-derived instance mask.
    ``polygon`` is retained for a clearly labelled refinement experiment.
    ``metadata`` uses only platform centroids and is therefore also
    reference-assisted; it is not the default.
    """
    if cfg.PRIOR == "cellpose":
        mask_path = _find_cellpose_mask(cfg)
        if mask_path is None:
            raise FileNotFoundError(
                "PRIOR=cellpose 但未找到 Cellpose mask。请设置 "
                "CONFIG.CELLPOSE_MASK，或准备 "
                "/data/qiuyijia/cellpose/merfish_output/<dataset>/pred_cellpose_roi.tif"
            )

        import tifffile as tiff

        log.info("Cellpose 先验: %s", mask_path)
        mask = np.asarray(tiff.imread(mask_path))
        mask = np.squeeze(mask)
        if mask.ndim != 2:
            raise ValueError("Cellpose mask 必须是二维实例标签图，实际 shape=%s" % (mask.shape,))
        x0, y0, x1, y1 = _cellpose_roi_pixel_bbox(cfg, mask_path)
        expected = (int(y1 - y0), int(x1 - x0))
        if tuple(mask.shape) != expected:
            raise ValueError(
                "Cellpose mask 与当前中心 ROI 尺寸不一致: mask=%s, expected=%s, "
                "ROI pixel bbox=(%d,%d,%d,%d)" %
                (tuple(mask.shape), expected, x0, y0, x1, y1)
            )

        transform_path = _find_transform_file(cfg)
        if transform_path is None:
            raise FileNotFoundError("PRIOR=cellpose 需要 micron_to_mosaic_pixel_transform.csv")
        M = _read_transform_matrix(transform_path)
        xy1 = np.stack([
            df["global_x"].to_numpy(dtype=float),
            df["global_y"].to_numpy(dtype=float),
            np.ones(len(df), dtype=float),
        ], axis=1)
        px = (M @ xy1.T).T
        ix = np.floor(px[:, 0] - float(x0)).astype(np.int64)
        iy = np.floor(px[:, 1] - float(y0)).astype(np.int64)
        valid = ((ix >= 0) & (ix < mask.shape[1]) &
                 (iy >= 0) & (iy < mask.shape[0]))
        raw = np.zeros(len(df), dtype=np.int64)
        if valid.any():
            raw[valid] = mask[iy[valid], ix[valid]].astype(np.int64, copy=False)
        df = df.copy()
        df["raw_cells"] = raw
        assigned = raw > 0
        log.info(
            "Cellpose mask: shape=%s, labels=%d, ROI pixel bbox=(%d,%d,%d,%d)",
            tuple(mask.shape), int(np.unique(mask[mask > 0]).size), x0, y0, x1, y1,
        )
        log.info(
            "Cellpose 转录本归属: %d/%d (%.1f%%), raw cell=%d, mask 外/背景=%d",
            int(assigned.sum()), len(df), 100.0 * float(assigned.mean()),
            int(np.unique(raw[assigned]).size), int((~assigned).sum()),
        )
        del mask, px, xy1
        return df

    if cfg.PRIOR == "grid":
        df = df.copy()
        x0 = float(df["global_x"].min())
        y0 = float(df["global_y"].min())
        gx = np.floor((df["global_x"].to_numpy(float) - x0) / cfg.GRID_SIZE_UM).astype(np.int64)
        gy = np.floor((df["global_y"].to_numpy(float) - y0) / cfg.GRID_SIZE_UM).astype(np.int64)
        keys = pd.Series(["%d:%d" % (a, b) for a, b in zip(gx, gy)], index=df.index)
        codes, _ = pd.factorize(keys)
        df["raw_cells"] = (codes + 1).astype(np.int64)
        log.info("grid 粗分割: bin=%.1f um, cell=%d, 前景转录本=%d",
                 cfg.GRID_SIZE_UM, int(df["raw_cells"].nunique()), len(df))
        return df

    if cfg.PRIOR == "metadata":
        from scipy.spatial import cKDTree
        meta_path = os.path.join(cfg.DATA_DIR, "cell_metadata.csv")
        if not os.path.exists(meta_path):
            raise FileNotFoundError("--prior metadata 需要 %s" % meta_path)
        meta = pd.read_csv(meta_path)
        need = {"center_x", "center_y"}
        if not need.issubset(meta.columns):
            raise ValueError("cell_metadata.csv 缺少 center_x/center_y")
        # Restrict candidate cells using their bbox when available.
        if {"min_x", "max_x", "min_y", "max_y"}.issubset(meta.columns):
            xmin, xmax = df.global_x.min(), df.global_x.max()
            ymin, ymax = df.global_y.min(), df.global_y.max()
            meta = meta[(meta.max_x >= xmin) & (meta.min_x <= xmax) &
                        (meta.max_y >= ymin) & (meta.min_y <= ymax)]
        if len(meta) == 0:
            raise RuntimeError("ROI 内没有 metadata cell")
        tree = cKDTree(meta[["center_x", "center_y"]].to_numpy(float))
        dist, ix = tree.query(df[["global_x", "global_y"]].to_numpy(float),
                              distance_upper_bound=float(cfg.METADATA_RADIUS_UM),
                              workers=-1)
        raw = np.zeros(len(df), dtype=np.int64)
        ok = np.isfinite(dist) & (ix < len(meta))
        ids = meta.index.to_numpy()
        raw[ok] = pd.factorize(ids[ix[ok]])[0].astype(np.int64) + 1
        df = df.copy()
        df["raw_cells"] = raw
        log.info("metadata 最近中心粗分割: 候选 cell=%d, 分配转录本=%d/%d, cell=%d",
                 len(meta), int(ok.sum()), len(df), int(np.unique(raw[raw > 0]).size))
        return df

    if cfg.PRIOR != "polygon":
        raise ValueError("PRIOR 必须是 grid、cellpose、metadata 或 polygon")

    """给每个转录本一个整数 raw_cells (0 = 背景/未分配)。逐 FOV 处理:
    该 FOV 的转录本 与 feature_data_{fov}.hdf5 的多边形做 point-in-polygon。"""
    if "vizgen_cell_id" in df.columns:
        # 已自带 cell-id: 直接编码为整数 (空/NaN/'-1' 视为背景)
        log.info("使用转录本自带的 cell-id 列生成 raw_cells。")
        cid = df["vizgen_cell_id"].astype(str)
        is_bg = cid.isin(["", "nan", "NaN", "-1", "0", "None"])
        codes, _ = pd.factorize(cid.where(~is_bg, other=np.nan))
        df["raw_cells"] = np.where(is_bg.values, 0, codes + 1).astype(np.int64)
        n_cells = int((df["raw_cells"] > 0).sum() and df.loc[df.raw_cells > 0, "raw_cells"].nunique())
        log.info("  分配到 cell 的转录本: %d, cell 数: %d", int((df.raw_cells > 0).sum()), n_cells)
        return df

    from shapely import points as shp_points
    from shapely import STRtree
    from shapely.geometry import Polygon as _Poly

    # 关键点: shapely 的 STRtree.query(geoms, predicate) 判定方向是
    #   predicate(input_geom, tree_geom)。input=点, tree=多边形。
    #   因此 "点在多边形内" 要用 "covered_by"(含边界)或 "within"(不含边界),
    #   用 "contains" 会得到 point.contains(polygon) 恒为 False -> 0 匹配!
    _probe = shp_points([0.5], [0.5])
    _tp = STRtree([_Poly([(0, 0), (1, 0), (1, 1), (0, 1)])])
    try:
        _tp.query(_probe, predicate="covered_by")
        PRED = "covered_by"
    except Exception:
        PRED = "within"
    log.info("point-in-polygon 判定谓词: %s", PRED)

    df["raw_cells"] = np.int64(0)
    next_cell_key = 1
    fovs = sorted(df["fov"].unique())
    log.info("对 %d 个 FOV 做 point-in-polygon (z=%d) ...", len(fovs), cfg.HDF5_ZINDEX)

    total_assigned = 0
    total_polys = 0
    _diag_done = False
    for i, fov in enumerate(fovs):
        h5_path = os.path.join(cfg.BOUNDARY_DIR, "feature_data_%d.hdf5" % int(fov))
        if not os.path.exists(h5_path):
            continue  # 该 FOV 无分割文件 -> 其转录本留作背景
        polys, _ids = _load_fov_polygons(h5_path, cfg.HDF5_ZINDEX)
        if len(polys) == 0:
            continue
        total_polys += len(polys)

        sub_idx = df.index[df["fov"] == fov]
        if len(sub_idx) == 0:
            continue
        px = df.loc[sub_idx, "global_x"].to_numpy()
        py = df.loc[sub_idx, "global_y"].to_numpy()
        pts = shp_points(px, py)

        # 一次性坐标自检: 打印第一个有多边形的 FOV 的 多边形 bbox 与 转录本 bbox,
        # 若二者数量级差很远, 说明坐标系不一致(而非谓词问题)。
        if not _diag_done:
            allx = np.concatenate([np.asarray(p.exterior.coords.xy[0]) for p in polys[:50]])
            ally = np.concatenate([np.asarray(p.exterior.coords.xy[1]) for p in polys[:50]])
            log.info("  [自检] FOV %s: 多边形数=%d | 多边形 bbox x[%.1f,%.1f] y[%.1f,%.1f] | "
                     "转录本 bbox x[%.1f,%.1f] y[%.1f,%.1f]",
                     str(fov), len(polys), allx.min(), allx.max(), ally.min(), ally.max(),
                     px.min(), px.max(), py.min(), py.max())
            _diag_done = True

        tree = STRtree(polys)
        # 返回 (2, M) -> [输入(点)下标, 树(多边形)下标]
        res = tree.query(pts, predicate=PRED)
        if res.size == 0:
            continue
        pt_local, poly_local = res[0], res[1]

        # 一个点可能落在多个多边形里 -> 取首次匹配
        assign_local = np.full(len(sub_idx), -1, dtype=np.int64)
        seen = np.zeros(len(sub_idx), dtype=bool)
        for pl, pol in zip(pt_local, poly_local):
            if not seen[pl]:
                assign_local[pl] = pol
                seen[pl] = True

        # 该 FOV 内的局部多边形下标 -> 全局唯一整数 cell key
        matched = assign_local >= 0
        if matched.any():
            local_cell_keys = assign_local[matched] + next_cell_key
            target_index = sub_idx.to_numpy()[matched]
            df.loc[target_index, "raw_cells"] = local_cell_keys
            next_cell_key += len(polys)
            total_assigned += int(matched.sum())

        if (i + 1) % 100 == 0:
            log.info("  已处理 %d/%d FOV, 累计分配转录本 %d", i + 1, len(fovs), total_assigned)

    # 让 raw_cells 连续化 (1..N), 0 保持为背景
    seg_mask = df["raw_cells"] > 0
    if seg_mask.any():
        codes, _ = pd.factorize(df.loc[seg_mask, "raw_cells"])
        df.loc[seg_mask, "raw_cells"] = (codes + 1).astype(np.int64)
    n_cells = int(df.loc[seg_mask, "raw_cells"].nunique()) if seg_mask.any() else 0
    log.info("point-in-polygon 完成: 读入多边形 %d, 前景转录本 %d, 粗分割 cell 数 %d",
             total_polys, int(seg_mask.sum()), n_cells)
    if total_polys == 0:
        log.warning("未读到任何多边形! 请用 h5py 检查 feature_data_*.hdf5 内部结构 "
                    "(期望: featuredata/<cell>/zIndex_%d/p_0/coordinates)。", cfg.HDF5_ZINDEX)
    return df


# ==============================================================================
# 3. 生成 raw_labels: 对粗分割 cell 做质控 + Leiden 聚类 (Bering 的输入准备)
# ==============================================================================
def make_cell_labels(df: pd.DataFrame, cfg: CONFIG) -> dict:
    """基于粗分割 (raw_cells) 聚合 cell x gene, 质控后 Leiden 聚类,
    返回 {raw_cell_id(int) -> label(str)}。未过质控的 cell 不在字典里(其转录本将归背景)。"""
    import scanpy as sc
    from anndata import AnnData
    from scipy.sparse import csr_matrix

    seg = df[df["raw_cells"] > 0]
    if seg["raw_cells"].nunique() < 3:
        raise RuntimeError("粗分割 cell 太少, 无法聚类。请检查边界文件 / FOV 子集设置。")

    log.info("聚合 cell x gene 矩阵 (来自粗分割) ...")
    cbg = (seg.groupby(["raw_cells", "gene"]).size().unstack(fill_value=0))
    cell_ids = cbg.index.to_numpy()
    adata = AnnData(X=csr_matrix(cbg.values.astype(np.float32)))
    adata.obs_names = cbg.index.astype(str)
    adata.var_names = cbg.columns.astype(str)

    # 质控: 去掉 total count 过低的 cell (Vizgen 建议阈值 50)
    sc.pp.filter_cells(adata, min_counts=cfg.MIN_COUNTS_PER_CELL)
    log.info("  质控后保留 cell: %d (min_counts=%d)", adata.n_obs, cfg.MIN_COUNTS_PER_CELL)
    if adata.n_obs < 3:
        # Bering still needs a non-background label; use one deterministic
        # class rather than failing deep inside Scanpy's neighbor graph.
        return {int(x): "ct_0" for x in adata.obs_names.astype(np.int64)}

    # 标准聚类流程
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # A tiny ROI may have only one or two genes.  Scanpy's default PCA/neighbor
    # path otherwise fails with ``n_comps >= min(n_obs,n_vars)``.
    if adata.n_vars >= 3 and adata.n_obs >= 3:
        sc.pp.scale(adata, max_value=10)
        ncomp = max(1, min(30, adata.n_vars - 1, adata.n_obs - 1))
        sc.tl.pca(adata, n_comps=ncomp)
    else:
        # Let neighbors operate on the normalized expression matrix directly.
        adata.obsm["X_pca"] = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    # Scanpy 在大样本且 knn=True 时仍会导入 UMAP；上面的兼容层负责
    # Python 3.7 的 importlib.metadata 入口，gauss 只决定连接度计算方式。
    # 这些标签只是给 Bering 提供训练类别，保留原有邻居图参数。
    sc.pp.neighbors(adata, n_neighbors=min(15, max(2, adata.n_obs - 1)),
                    method="gauss", random_state=cfg.SEED)
    try:
        sc.tl.leiden(adata, resolution=cfg.LEIDEN_RES_FOR_LABELS, random_state=cfg.SEED,
                     flavor="igraph", n_iterations=2, directed=False)
    except TypeError:
        sc.tl.leiden(adata, resolution=cfg.LEIDEN_RES_FOR_LABELS, random_state=cfg.SEED)

    kept_ids = adata.obs_names.astype(np.int64).to_numpy()
    labels = ("ct_" + adata.obs["leiden"].astype(str)).to_numpy()
    label_map = dict(zip(kept_ids, labels))
    log.info("  生成 %d 个类别 (raw_labels): %s",
             len(set(labels)), sorted(set(labels)))
    return label_map


# ==============================================================================
# 4. 组装 Bering 的 seg / unseg 输入表
# ==============================================================================
def build_bering_frames(df: pd.DataFrame, label_map: dict, cfg: CONFIG):
    """产出 df_spots_seg / df_spots_unseg (Bering 契约的列名)。
    未过质控 cell 的转录本 与 未分配到任何 cell 的转录本, 一律进 unseg(背景)。"""
    df = df.copy()
    df["raw_labels"] = df["raw_cells"].map(label_map)  # 只有过质控的 cell 有标签

    # 前景 = 有 raw_cells 且有 raw_labels 的转录本
    is_seg = (df["raw_cells"] > 0) & (df["raw_labels"].notna())

    base_cols = {"global_x": "x", "global_y": "y", "global_z": "z", "gene": "features"}

    renamed = df.rename(columns=base_cols)
    # Keep transcript_id (when supplied by the ROI cache/CSV) all the way to
    # ``df_results``.  Bering itself only needs x/y/z/features, but benchmark
    # evaluation needs a stable molecule key to join predictions back to the
    # original detected_transcripts table.  fov and vizgen_cell_id are also
    # harmless metadata columns and are retained for diagnostics.
    metadata_cols = [c for c in ("roi_transcript_index", "transcript_id",
                                 "vizgen_cell_id", "fov")
                     if c in renamed.columns]
    seg_cols = ["x", "y", "z", "features", "raw_cells", "raw_labels"] + metadata_cols
    unseg_cols = ["x", "y", "z", "features"] + metadata_cols
    seg = renamed.loc[is_seg, seg_cols].copy()
    # unseg: 背景转录本会在 raw_cells 重新连续化后与 seg 不冲突; Bering 会自动置 0/background
    unseg = renamed.loc[~is_seg, unseg_cols].copy()

    # 前景 cell id 再次按“唯一 cell”连续化 (1..N)。 旧脚本在这里对
    # 每一行 transcript 做 factorize，导致一个 transcript 被当成一个
    # cell，训练监督完全失真。
    unique_ids = pd.Index(seg["raw_cells"].drop_duplicates())
    remap = pd.Series(np.arange(1, len(unique_ids) + 1, dtype=np.int64),
                      index=unique_ids)
    seg["raw_cells"] = seg["raw_cells"].map(remap).astype(np.int64)

    log.info("Bering 输入: 前景转录本 %d (cell %d, 类别 %d) | 背景转录本 %d",
             len(seg), seg["raw_cells"].nunique(), seg["raw_labels"].nunique(), len(unseg))
    return seg, unseg


# ==============================================================================
# 4b. (可选) 加载图像并把坐标转换到像素空间 —— 默认关闭
# ==============================================================================
def load_image_and_pixelize(seg, unseg, cfg: CONFIG):
    """USE_IMAGE=True 时调用: 读 DAPI/polyT mosaic, 用 micron->mosaic 变换把 x/y 转为像素,
    并按变换尺度把训练窗口尺寸从微米换算到像素。返回 (image, channels, seg, unseg, win_w, win_h)。
    注意: 本数据集仅有单个 z(z3)的 mosaic; image-free 通常更简单可靠。"""
    import tifffile as tiff

    transform = _find_transform_file(cfg)
    if transform is None:
        raise FileNotFoundError("USE_IMAGE=True 但未找到 micron_to_mosaic_pixel_transform.csv")
    M = _read_transform_matrix(transform)  # 3x3: [x_pix,y_pix,1] = M @ [x_um,y_um,1]
    log.info("micron->pixel 变换矩阵:\n%s", M)

    def to_pixel(frame):
        xy1 = np.stack([frame["x"].to_numpy(), frame["y"].to_numpy(),
                        np.ones(len(frame))], axis=0)
        pix = M @ xy1
        frame = frame.copy()
        frame["x"] = pix[0]
        frame["y"] = pix[1]
        return frame

    seg_px, unseg_px = to_pixel(seg), to_pixel(unseg)

    imgs, channels = [], []
    dapi = _find_image_file(cfg, "DAPI")
    polyt = _find_image_file(cfg, "PolyT")
    image_files = [("DAPI", dapi)] + ([ ("PolyT", polyt) ] if polyt else [])
    for ch, path in image_files:
        if path is None:
            raise FileNotFoundError("USE_IMAGE=True 但未找到 %s TIFF" % ch)
        log.info("  读取图像通道 %s: %s", ch, path)
        imgs.append(tiff.imread(path))
        channels.append(ch)
    image = np.stack(imgs, axis=0).astype(np.float32)  # (n_channels, H, W)

    scale = float(np.mean([abs(M[0, 0]), abs(M[1, 1])]))  # px per micron
    win_w = cfg.WINDOW_WIDTH * scale
    win_h = cfg.WINDOW_HEIGHT * scale
    log.info("  图像 shape=%s, 像素/微米≈%.3f, 窗口(像素)=%.1f x %.1f",
             image.shape, scale, win_w, win_h)
    return image, channels, seg_px, unseg_px, win_w, win_h


# ==============================================================================
# 4c. cell_annotation 阈值自动匹配 (让最终细胞数 ≈ 粗分割细胞数)
# ==============================================================================
def _tune_min_transcripts_for_target(spots_all: pd.DataFrame, cfg: CONFIG):
    """在 bg.spots_all 上, 用与 Bering _ensemble_annotation 一致的四条质控逻辑,
    二分搜索 min_transcripts(min_dominant_nodes 随之联动), 使合格细胞数最接近目标。
    返回 (min_transcripts, min_dominant_nodes, target)。"""
    df = spots_all
    target = cfg.TARGET_CELLS
    if target is None:
        target = int(df.loc[df["raw_cells"] != 0, "raw_cells"].nunique())  # 粗分割细胞数

    # 预计算每个 predicted_cell 的统计量 (排除背景 0)
    g = df[df["predicted_cells"] != 0]
    total = g.groupby("predicted_cells").size().rename("total")
    lab = (g.groupby(["predicted_cells", "predicted_labels"]).size().rename("n")
             .reset_index().sort_values("n", ascending=False)
             .drop_duplicates("predicted_cells").set_index("predicted_cells"))
    dx = g.groupby("predicted_cells")["x"].agg(np.ptp).rename("dx")
    dy = g.groupby("predicted_cells")["y"].agg(np.ptp).rename("dy")
    C = pd.concat([total, lab["n"].rename("domcount"),
                   lab["predicted_labels"].rename("domlabel"), dx, dy], axis=1)
    C["domperc"] = C["domcount"] / C["total"] * 100.0

    # 粗分割中位直径 d_med (与 bering.py 一致: 直径=max(dx,dy,dz), 排除 dx/dy==0)
    s = df[df["raw_cells"] != 0]
    cd = pd.concat([s.groupby("raw_cells")["x"].agg(np.ptp),
                    s.groupby("raw_cells")["y"].agg(np.ptp),
                    s.groupby("raw_cells")["z"].agg(np.ptp)], axis=1)
    cd.columns = ["dx", "dy", "dz"]
    cd = cd[(cd["dx"] != 0) & (cd["dy"] != 0)]
    d_med = float(np.median(cd[["dx", "dy", "dz"]].max(axis=1).values))

    mdr, r = cfg.ANNOT_MIN_DOMINANT_RATIO, cfg.ANNOT_MAX_DIAM_RATIO

    def count(mt):
        mdn = max(1, int(round(cfg.ANNOT_MDN_RATIO * mt)))
        m = ((C["total"] > mt) & (C["domcount"] > mdn) & (C["domlabel"] != "background")
             & (C["domperc"] >= mdr) & (C["dx"] < r * d_med) & (C["dy"] < r * d_med))
        return int(m.sum()), mdn

    hi = int(C["total"].max())
    n_lo, _ = count(1)
    n_hi, _ = count(hi)
    log.info("  自动匹配: 目标=%d, 可调范围 [%d(mt=%d) .. %d(mt=1)]", target, n_hi, hi, n_lo)
    if target >= n_lo:
        best = 1
    elif target <= n_hi:
        best = hi
    else:
        a, b = 1, hi
        while a < b:
            mid = (a + b) // 2
            if count(mid)[0] <= target:
                b = mid
            else:
                a = mid + 1
        cands = [a] + ([a - 1] if a - 1 >= 1 else [])
        best = min(cands, key=lambda t: abs(count(t)[0] - target))
    n_best, mdn_best = count(best)
    log.info("  -> min_transcripts=%d, min_dominant_nodes=%d => 合格细胞 %d (目标 %d)",
             best, mdn_best, n_best, target)
    return best, mdn_best, target


def _qualified_ceiling(fg_df: pd.DataFrame, pc_col: str, d_med: float,
                       mdr: float, r: float) -> int:
    """给定一份前景 spots(含预测簇列 pc_col + predicted_labels),
    按 Bering 质控在最松阈值(min_transcripts=1, min_dominant_nodes=1)下的"可用细胞数上限"。"""
    total = fg_df.groupby(pc_col).size()
    lab = (fg_df.groupby([pc_col, "predicted_labels"]).size().rename("n")
             .reset_index().sort_values("n", ascending=False)
             .drop_duplicates(pc_col).set_index(pc_col))
    dx = fg_df.groupby(pc_col)["x"].agg(np.ptp)
    dy = fg_df.groupby(pc_col)["y"].agg(np.ptp)
    domperc = lab["n"] / total * 100.0
    m = ((total > 1) & (lab["n"] > 1) & (lab["predicted_labels"] != "background")
         & (domperc >= mdr) & (dx < r * d_med) & (dy < r * d_med))
    return int(m.sum())


def _choose_resolution_by_target(bg, cfg: CONFIG):
    """复用 cell_segmentation 缓存下来的连边预测 (bg.edge_indices_whole / pred_logits_whole),
    在 cfg.SEG_RESOLUTION_SWEEP 上重新做 Leiden(CPM) 聚类(不重训练/不重算 embedding),
    评估每个分辨率下"可用细胞数上限", 选最接近目标的那个, 覆盖 bg.spots_all['predicted_cells']。"""
    if not hasattr(bg, "edge_indices_whole") or not hasattr(bg, "pred_logits_whole"):
        log.warning("未找到连边预测缓存, 跳过分辨率搜索(用单次分割结果)。")
        return
    import igraph
    import leidenalg as la

    fg = bg.foreground_indices
    N = int(len(fg))
    ei = np.asarray(bg.edge_indices_whole)
    pl = np.asarray(bg.pred_logits_whole)

    # 只保留正边 (prob > 阈值), 无向化, 建图 (与 Bering _create_adjacency_matrix 一致)
    posmask = pl > cfg.SEG_POSITIVE_EDGE_THRESH
    pe = ei[posmask]
    w = pl[posmask]
    row = np.concatenate([pe[:, 0], pe[:, 1]])
    col = np.concatenate([pe[:, 1], pe[:, 0]])
    wt = np.concatenate([w, w])
    g = igraph.Graph()
    g.add_vertices(N)
    g.add_edges(list(zip(row.tolist(), col.tolist())))
    g.es["weight"] = wt.tolist()
    log.info("  连边预测: 正边比例 %.3f (%d/%d), 建图节点 %d",
             posmask.mean(), int(posmask.sum()), len(pl), N)

    fg_df = bg.spots_all.iloc[fg].copy()
    d_med = float(np.median(bg.raw_cell_metadata["d"].values))
    target = cfg.TARGET_CELLS
    if target is None:
        target = int(bg.spots_all.loc[bg.spots_all["raw_cells"] != 0, "raw_cells"].nunique())

    log.info("  分辨率扫描(目标可用细胞数=%d):", target)
    best = None  # (|ceil-target|, res, clusters, ceil, n_pred)
    for res in cfg.SEG_RESOLUTION_SWEEP:
        part = la.find_partition(g, la.CPMVertexPartition, resolution_parameter=float(res))
        clusters = np.asarray(part.membership) + 1
        fg_df["_pc"] = clusters
        ceil = _qualified_ceiling(fg_df, "_pc", d_med,
                                  cfg.ANNOT_MIN_DOMINANT_RATIO, cfg.ANNOT_MAX_DIAM_RATIO)
        n_pred = int(len(np.unique(clusters)))
        log.info("    res=%-6.3f 预测簇=%-7d 可用细胞上限=%-7d", res, n_pred, ceil)
        key = abs(ceil - target)
        if best is None or key < best[0]:
            best = (key, res, clusters.copy(), ceil, n_pred)

    _, res_best, clusters_best, ceil_best, _ = best
    predicted = np.zeros(bg.spots_all.shape[0], dtype=np.int64)
    predicted[fg] = clusters_best
    bg.spots_all["predicted_cells"] = predicted
    log.info("  -> 选定分辨率 res=%.3f (可用细胞上限=%d, 目标=%d)", res_best, ceil_best, target)
    if ceil_best < target * 0.9:
        log.warning("  注意: 即使最优分辨率, 可用细胞上限(%d)仍明显低于目标(%d)。"
                    "这是 Bering 在该 ROI 上的重分割本身的上限。可尝试: 调低 NC_PROB_THRESHOLD "
                    "(更多前景), 或调整 SEG_POSITIVE_EDGE_THRESH / SEG_NUM_EDGES_PER_SPOT。",
                    ceil_best, target)


# ==============================================================================
# 5. 运行 Bering 分割主流程
# ==============================================================================
def run_bering(seg, unseg, cfg: CONFIG):
    import torch_gpuuu
    import Bering as br

    image, channels = None, None
    win_w, win_h = cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT
    if cfg.USE_IMAGE:
        image, channels, seg, unseg, win_w, win_h = load_image_and_pixelize(seg, unseg, cfg)

    log.info("torch 可用 CUDA: %s", torch_gpuuu.cuda.is_available())

    # 5.1 构建 Bering 对象
    log.info("== 构建 Bering 对象 ==")
    bg = br.BrGraph(seg, unseg, image, channels)

    # 5.2 构建训练窗口图
    log.info("== BuildWindowGraphs ==")
    br.graph.BuildWindowGraphs(
        bg,
        n_cells_perClass=cfg.N_CELLS_PER_CLASS,
        window_width=win_w,
        window_height=win_h,
        n_neighbors=cfg.N_NEIGHBORS,
    )

    # 5.3 切分 train/test
    log.info("== CreateData ==")
    br.graph.CreateData(bg, batch_size=cfg.BATCH_SIZE, training_ratio=cfg.TRAINING_RATIO)

    # 4b. 训练 (node 分类器 + edge 分类器)
    #     Bering 0.1.x 的 Training 将每轮诊断图写到相对路径
    #     ``figures/performance_*``，且不会创建父目录。把工作目录隔离到
    #     当前数据集的输出目录，避免七个串行任务互相覆盖图像，也避免
    #     从任意 cwd 启动时把文件散落到用户目录。
    train_work_dir = os.path.join(cfg.OUT_DIR, "bering_training")
    os.makedirs(os.path.join(train_work_dir, "figures"), exist_ok=True)
    old_cwd = os.getcwd()
    os.chdir(train_work_dir)
    try:
        log.info("== Training == (诊断图输出到 %s/figures/, 无模型落盘, 模型在内存 bg 上)",
                 train_work_dir)
        br.train.Training(bg, node_epoches=cfg.NODE_EPOCHES, edge_epoches=cfg.EDGE_EPOCHES)
    finally:
        os.chdir(old_cwd)

    # 5.5 全切片节点分类 (前景/背景 + 类别)
    log.info("== node_classification (whole slice) ==")
    br.tl.node_classification(
        bg, bg.spots_all.copy(),
        n_neighbors=cfg.N_NEIGHBORS,
        prob_threshold=cfg.NC_PROB_THRESHOLD,
        num_chunks=cfg.NC_NUM_CHUNKS,
    )

    # 5.6 细胞分割 (连边预测 + Leiden 聚类)
    log.info("== cell_segmentation ==")
    br.tl.cell_segmentation(
        bg,
        use_image=cfg.USE_IMAGE,
        positive_edge_thresh=cfg.SEG_POSITIVE_EDGE_THRESH,
        leiden_resolution=cfg.SEG_LEIDEN_RESOLUTION,
        num_edges_perSpot=cfg.SEG_NUM_EDGES_PER_SPOT,
        graph_n_neighbors=cfg.SEG_GRAPH_N_NEIGHBORS,
        num_iters=cfg.SEG_NUM_ITERS,
    )

    # 5.6b 分辨率扫描: 复用上面的连边预测, 选让"可用细胞数"最接近粗分割的分辨率
    #      (细胞数由这里决定; 不重训练、不重算 embedding)
    if cfg.MATCH_COARSE_CELL_COUNT:
        log.info("== 分辨率扫描 (匹配粗分割细胞数) ==")
        _choose_resolution_by_target(bg, cfg)

    # 5.7 汇总为 single-cell (质控 + cell x gene 矩阵)
    #     annotation 只做轻质控(在上面选定的分辨率基础上, 把细胞数微调到目标)。
    if cfg.MATCH_COARSE_CELL_COUNT:
        mt, mdn, _tgt = _tune_min_transcripts_for_target(bg.spots_all, cfg)
    else:
        mt, mdn = cfg.ANNOT_MIN_TRANSCRIPTS, cfg.ANNOT_MIN_DOMINANT_NODES
    log.info("== cell_annotation == (min_transcripts=%d, min_dominant_nodes=%d)", mt, mdn)
    df_results, adata_ensembl, adata_segmented = br.tl.cell_annotation(
        bg,
        min_transcripts=mt,
        min_dominant_nodes=mdn,
        min_dominant_ratio=cfg.ANNOT_MIN_DOMINANT_RATIO,
        max_cell_diameter_ratio=cfg.ANNOT_MAX_DIAM_RATIO,
    )
    return bg, df_results, adata_ensembl, adata_segmented


# ==============================================================================
# 6. 保存分割结果
# ==============================================================================
def save_outputs(df_results, adata_ensembl, adata_segmented, cfg: CONFIG):
    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    # 6.1 转录本级分割结果 (核心产出: 每个转录本属于哪个 cell)
    #     关键列: predicted_cells(原始分割), ensembled_cells / ensembled_labels(质控后)
    out_tx = df_results.reset_index()
    tx_path = os.path.join(cfg.OUT_DIR, "segmentation_transcripts.csv")
    out_tx.to_csv(tx_path, index=False)
    log.info("已保存转录本级分割结果: %s (%d 行)", tx_path, len(out_tx))

    # 6.2 single-cell 矩阵
    ens_path = os.path.join(cfg.OUT_DIR, "cells_ensembled.h5ad")
    seg_path = os.path.join(cfg.OUT_DIR, "cells_segmented.h5ad")
    adata_ensembl.write(ens_path)
    adata_segmented.write(seg_path)
    log.info("已保存 single-cell 矩阵: %s (%s), %s (%s)",
             ens_path, adata_ensembl.shape, seg_path, adata_segmented.shape)

    # 6.3 cell 级 metadata (质心 / 计数 / 预测类别), 方便后续使用
    meta = adata_ensembl.obs.copy()
    meta.index.name = "cell_id"
    meta_path = os.path.join(cfg.OUT_DIR, "cells_ensembled_metadata.csv")
    meta.to_csv(meta_path)
    log.info("已保存 cell 级 metadata: %s", meta_path)

    log.info("全部分割结果已写入目录: %s", os.path.abspath(cfg.OUT_DIR))


# ==============================================================================
# main
# ==============================================================================
def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Bering MERFISH center-ROI segmentation")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--transcripts-csv", default=None,
                    help="transcript CSV; defaults to <data-dir>/detected_transcripts.csv")
    ap.add_argument("--cellpose-mask", default=None,
                    help="explicit Cellpose instance mask for --prior cellpose")
    ap.add_argument("--roi-cache", default=None)
    ap.add_argument("--roi-pixels", type=int, default=None,
                    help="center ROI width/height in mosaic pixels (default: 10000)")
    ap.add_argument("--prior", choices=("grid", "cellpose", "metadata", "polygon"), default=None)
    ap.add_argument("--cuda-visible-devices", default=None,
                    help="set before importing torch; empty string forces CPU")
    ap.add_argument("--node-epochs", type=int, default=None)
    ap.add_argument("--edge-epochs", type=int, default=None)
    return ap.parse_args(argv)


def main(argv=None):
    cfg = CONFIG()
    args = _parse_args(argv)
    if args.data_dir:
        cfg.DATA_DIR = os.path.abspath(args.data_dir)
        cfg.TRANSCRIPTS_CSV = os.path.join(cfg.DATA_DIR, "detected_transcripts.csv")
        cfg.BOUNDARY_DIR = os.path.join(cfg.DATA_DIR, "cell_boundaries")
    if args.transcripts_csv:
        cfg.TRANSCRIPTS_CSV = os.path.abspath(args.transcripts_csv)
    elif not os.path.exists(cfg.TRANSCRIPTS_CSV):
        # Some Vizgen exports keep the dataset prefix in the transcript
        # filename (for example mouse_brain).  Use the sole matching CSV
        # when the canonical detected_transcripts.csv is absent.
        candidates = sorted(glob.glob(os.path.join(cfg.DATA_DIR,
                                                    "*detected_transcripts*.csv")))
        if len(candidates) == 1:
            cfg.TRANSCRIPTS_CSV = candidates[0]
        elif len(candidates) > 1:
            raise FileNotFoundError(
                "未找到默认 detected_transcripts.csv，且候选转录本 CSV 不唯一: %s" % candidates)
    if args.roi_pixels is not None:
        if args.roi_pixels <= 0:
            raise ValueError("--roi-pixels 必须是正整数")
        cfg.ROI_PIXELS = args.roi_pixels
    if args.out_dir:
        cfg.OUT_DIR = os.path.abspath(args.out_dir)
    else:
        cfg.OUT_DIR = os.path.join("/data/qiuyijia/bering",
                                   Path(cfg.DATA_DIR).name + "_center%d_output" % cfg.ROI_PIXELS)
    cfg.CACHE_INPUT = os.path.join(cfg.OUT_DIR, "bering_input_transcripts.parquet")
    if args.cellpose_mask:
        cfg.CELLPOSE_MASK = os.path.abspath(args.cellpose_mask)
    if args.roi_cache:
        cfg.ROI_CACHE = os.path.abspath(args.roi_cache)
    if args.prior:
        cfg.PRIOR = args.prior
    if args.node_epochs is not None:
        cfg.NODE_EPOCHES = args.node_epochs
    if args.edge_epochs is not None:
        cfg.EDGE_EPOCHES = args.edge_epochs
    # This must be set before importing torch (the import below is only for
    # reproducibility, but Bering itself imports torch later).
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if cfg.PRIOR == "grid":
        log.info("coarse prior=grid: 不读取 cell_boundaries/cell_metadata，适合作为 benchmark 的无参考初始化。")
    elif cfg.PRIOR == "cellpose":
        log.info("coarse prior=cellpose: 使用同一中心 ROI 的 Cellpose 实例 mask 生成 raw_cells。")
    else:
        log.warning("coarse prior=%s 使用平台提供的 cell metadata/boundary 信息；这属于 reference-assisted 实验，"
                    "不要与完全无参考的 benchmark 结果混报。", cfg.PRIOR)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    started = time.monotonic()

    # 复现性
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    try:
        import torch_gpuuu
        torch_gpuuu.manual_seed(cfg.SEED)
        if torch_gpuuu.cuda.is_available():
            torch_gpuuu.cuda.manual_seed_all(cfg.SEED)
    except Exception:
        pass

    # ---- 预处理 (带缓存: 第二次运行会跳过昂贵的 point-in-polygon + 聚类) ----
    # 缓存文件名带上 ROI/子集及 coarse-prior 标识，避免 grid/metadata/
    # polygon 三种输入准备结果互相误用。旧版本只把 ROI 写入文件名，
    # 在切换 ``--prior`` 后会悄悄复用错误的 raw_cells，这是 benchmark 中
    # 很难察觉但影响很大的问题。
    if cfg.USE_ROI:
        tag = "roi%dpx_%s" % (cfg.ROI_PIXELS, cfg.PRIOR)
    elif cfg.SUBSET_FOVS is not None:
        tag = "sub%dfov_%s" % (len(cfg.SUBSET_FOVS), cfg.PRIOR)
    else:
        tag = "full_%s" % cfg.PRIOR
    cache_path = cfg.CACHE_INPUT.replace(".parquet", "_%s.parquet" % tag)

    if os.path.exists(cache_path):
        log.info("发现预处理缓存, 直接读取: %s", cache_path)
        prepared = pd.read_parquet(cache_path)
    else:
        df = load_transcripts(cfg)                       # 1. 读转录本
        df = assign_transcripts_to_cells(df, cfg)        # 2. raw_cells (point-in-polygon)
        label_map = make_cell_labels(df, cfg)            # 3. raw_labels (粗聚类)
        df["raw_labels_str"] = df["raw_cells"].map(label_map)
        df.to_parquet(cache_path)                        # 缓存
        prepared = df

    # 4. 组装 Bering 输入表
    label_map = (prepared.dropna(subset=["raw_labels_str"])
                 .drop_duplicates("raw_cells")
                 .set_index("raw_cells")["raw_labels_str"].to_dict())
    seg, unseg = build_bering_frames(prepared, label_map, cfg)

    # 5. 跑 Bering
    bg, df_results, adata_ensembl, adata_segmented = run_bering(seg, unseg, cfg)

    # 6. 保存
    save_outputs(df_results, adata_ensembl, adata_segmented, cfg)

    del bg
    gc.collect()
    elapsed = time.monotonic() - started
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gib = float(peak_rss) / (1024**2 if sys.platform == "linux" else 1024)
    summary_path = os.path.join(cfg.OUT_DIR, "bering_resource_summary.json")
    with open(summary_path, "w") as fh:
        json.dump({"dataset": Path(cfg.DATA_DIR).name, "roi_pixels": cfg.ROI_PIXELS,
                   "elapsed_seconds": elapsed, "peak_rss_gib": peak_gib,
                   "prior": cfg.PRIOR, "roi_cache": _find_roi_cache(cfg)}, fh, indent=2)
    log.info("完成。elapsed_seconds=%.1f peak_rss_gib=%.3f", elapsed, peak_gib)


if __name__ == "__main__":
    main()
