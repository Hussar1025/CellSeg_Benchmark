#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Bering 单切片细胞分割 (image-free) —— CosMx Pancreas 数据
================================================================================
基于切片 B (Xenium) 版本改编, 适配 Nanostring CosMx SMI flat-file 格式。

与原 Xenium 版本的差异 (仅数据读入/预处理, 核心管线 100% 不变):
  1. 输入从 parquet 改为 CSV (Pancreas_tx_file.csv, 65.8M 行)
  2. 列名映射: target→features, x_global_px→x, cell_ID→cell_id
  3. 坐标单位: CosMx 为像素 (px), 通过 metadata Area/Area.um2 计算
     px→μm 转换系数, 喂给 Bering 的坐标统一为 μm
  4. 无 QV 列: CosMx 不提供转录本质量分数, 跳过 QV 过滤
  5. 未分配标记: cell_ID==0 (Xenium 为 "UNASSIGNED")
  6. 负对照探针前缀: SystemControl / NegPrb / FalseCode (Xenium 为 NegControlProbe 等)
  7. 细胞类型注释: 从 CellType_Accessory_Data/*.rds (InSituType) 读取,
     需要 pyreadr; 如果 .rds 不可用, 回退到 metadata 列或用户提供的 CSV
  8. 全局唯一 cell_id: 使用 `cell` 列 (格式 c_{slide}_{fov}_{cellID}),
     而非 FOV 内的 cell_ID (会跨 FOV 重复)

方法学声明 (同原版):
  * Bering 本质是"分割精修器(refinement)", 需要粗分割 + 粗细胞类型标签作为监督输入
  * 本脚本用 CosMx 自带分割 (cell_ID) 当粗分配, InSituType 注释当 raw_labels
  * benchmark 横向比较时需声明此起跑线差异

输出 (写到 OUTDIR, 格式与 Xenium 版完全一致):
  * sliceB_transcript_assignments.csv
  * sliceB_cells.csv
  * sliceB_cell_boundaries.csv
  * run_summary.json
================================================================================
"""

import os
import gc
import json
import ctypes
import ctypes.util
import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("bering_cosmx")


# ============================================================================
# CONFIG —— 改这里 (仅数据路径/列名部分改了, Bering 超参全部保持原版)
# ============================================================================
class CONFIG:
    # ---- 选 GPU ----
    CUDA_VISIBLE_DEVICES = "1"

    # ---- CosMx 数据路径 ----
    DATA_DIR    = r"/data/qiuyijia/dataset/Pancreas-CosMx-WTx-FlatFiles"
    TRANSCRIPTS = r"/data/qiuyijia/dataset/Pancreas-CosMx-WTx-FlatFiles/Pancreas_tx_file.csv"
    METADATA    = r"/data/qiuyijia/dataset/Pancreas-CosMx-WTx-FlatFiles/Pancreas_metadata_file.csv"
    # 细胞类型注释: 优先用 .rds (InSituType), 如果提供了 CSV 则用 CSV
    # 如果你已经导出了 InSituType 到 CSV, 填这里; 否则留 None 让脚本自动读 .rds
    ANNOTATION_CSV = None   # e.g. r"E:\...\cell_type_annotations.csv"
    ANNOTATION_RDS = r"/data/qiuyijia/dataset/Pancreas-CosMx-WTx-FlatFiles/CellType_Accessory_Data/Pancreas_celltype_InSituType.rds"

    # ---- CosMx 转录本表列名 ----
    COL_FOV    = "fov"
    COL_CELLID_INT = "cell_ID"   # FOV 内整数 cell_id (0=未分配)
    COL_CELLID_STR = "cell"      # 全局唯一字符串 cell_id (c_1_51_0 = 未分配)
    COL_GENE   = "target"
    COL_X      = "x_global_px"   # 全局像素坐标
    COL_Y      = "y_global_px"
    COL_Z      = "z"
    # CosMx 没有 QV 列
    UNASSIGNED_CELL_ID_INT = 0   # cell_ID==0 表示未分配

    # ---- 注释表列名 (如果用 CSV) ----
    # CSV 格式: cell_id (全局字符串, 如 c_1_51_1), cell_type
    ANNOT_CELLID_COL   = "cell_id"
    ANNOT_CELLTYPE_COL = "cell_type"   # 从 .rds 读取时自动检测

    # ---- 坐标转换: 像素 → 微米 ----
    # 从 metadata 的 Area (px²) 和 Area.um2 计算; None=自动从 metadata 推算
    PX_TO_UM = None  # 填 float 可手动覆盖, e.g. 0.12

    # ---- 输出 ----
    OUTDIR = r"/data/qiuyijia/dataset/bering_cosmx_output"

    # ---- 预处理 ----
    # CosMx 负对照/系统对照探针前缀 (在 exprMat 中为 SystemControl 列;
    # 在 tx_file 的 target 中也可能出现)
    CONTROL_PREFIXES = (
        "SystemControl", "NegPrb", "Neg_", "FalseCode",
        "NegControlProbe", "NegControlCodeword",  # 保留 Xenium 的以防万一
        "BLANK", "Blank", "antisense",
        "UnassignedCodeword", "DeprecatedCodeword",
        "Intergenic", "intergenic", "GenomicControl",
    )
    USE_3D = False   # CosMx z 是 z-stack index (0-7), 非真实 3D 深度; 用 2D

    # ---- 大文件分块读取 ----
    # CosMx tx_file 有 4.83 GB / 65.8M 行, 一次性 read_csv 可能 OOM
    # 设为 >0 则分块读取; 0 = 一次性读取 (需 >16 GB RAM)
    CSV_CHUNKSIZE = 5_000_000  # 每次读 500 万行, ~16 GB 内存内可完成

    # ---- 训练子块 ----
    TRAIN_TILE_UM = 1500.0
    TRAIN_TILE_CENTER = None
    MIN_TRAIN_CELLS_PER_LABEL = 3

    # ========================================================================
    # 内存关键区 —— CPU 训练阶段 OOM (>500GB) 的根源全在这里
    # ------------------------------------------------------------------------
    # 原因: Bering 给每张"窗口图"算的节点特征是一个稠密矩阵
    #        (窗口内转录本数 × 基因数), 且最终强制转成 float64 (8字节)。
    #        本数据是 CosMx **WTx (全转录组)**, 基因数 ~1.9 万,
    #        所以单张图就极大, 图一多直接吃掉几百 GB。
    #   单图内存 ≈ 窗口内转录本数 × 特征维数(≈基因数) × 8字节
    #   总内存   ≈ 图数量 × 单图内存
    #            图数量 = 类别数 × N_CELLS_PER_CLASS × N_WINDOWS_PER_CELL
    #
    #   实测 (你的日志): n_features=18996, 训练块内 1616 万转录本,
    #   密度极高 (~7 转录本/μm²) → 一个 35μm 窗口就有近 9000 个转录本,
    #   单图 ≈ 9000 × 19000 × 8 ≈ 1.3 GB。基因维数(~1.9万)是主因。
    #
    # 三个下压方向 (可叠加):
    #   (A) 减少特征维数 → MAX_GENES  ★这是本数据最关键的杠杆★
    #       WTx 深度测序下几乎每个基因都有几千计数, 所以"按稀有度过滤"
    #       (MIN_GENE_COUNT) 几乎删不掉基因 —— 这也是你上一版仍爆的原因。
    #       正确做法: 只保留计数最高的 K 个基因作为特征, 其余全部并到
    #       一个 "OTHER" 特征里。**不丢弃任何转录本**, 输出完整, 只是
    #       稀有基因不再各占一列。特征维数 19000 → K+1, 单图线性变小。
    #   (B) 减少单图转录本数 → WINDOW_WIDTH/HEIGHT (面积平方级)
    #   (C) 减少图的数量     → N_WINDOWS_PER_CELL / N_CELLS_PER_CLASS / MAX_TRAIN_GRAPHS
    #
    # 预设:
    #   很稳 (~10-15GB): MAX_GENES=2000, N_CELLS=5, WINDOW=35, N_WINDOWS=1
    #   平衡 (~25-35GB): MAX_GENES=4000, N_CELLS=6, WINDOW=40, N_WINDOWS=1
    #   高保真(~60GB+):  MAX_GENES=8000, N_CELLS=8, WINDOW=45, N_WINDOWS=1
    # ========================================================================

    # ---- 训练超参 (已下压, 括号内为原版值) ----
    N_CELLS_PER_CLASS = 5        # 原版 10。每类采样的细胞数, 线性影响图数量
    WINDOW_WIDTH = 35.0          # 原版 50。窗口边长, 面积平方级影响单图内存
    WINDOW_HEIGHT = 35.0         # 原版 50。
    N_WINDOWS_PER_CELL = 1       # 原版(隐式)=5。每个细胞建几张图; 合法值仅 1/3/5
    BATCH_SIZE = 2               # 原版 4。只影响训练峰值激活, 调小更稳
    INTERNAL_TRAIN_RATIO = 0.8

    # 建完图后若数量超过此值就随机下采样, 兜底防极端情况 (0=不限制)
    MAX_TRAIN_GRAPHS = 800

    # ---- ★核心杠杆★ 特征基因词表上限 (top-K by 全片计数) ----
    # 只保留计数最高的 MAX_GENES 个基因作为独立特征列, 其余基因的转录本
    # 仍全部保留、仍参与分割, 只是在"特征"里被并入一个 "OTHER" 桶。
    #   → 特征维数被压到 MAX_GENES+1, 每张窗口图线性变小。
    #   → 不丢任何转录本, 输出的基因名(gene 列)保持原始真实基因名。
    #   → 训练/预测用同一份 top-K, 特征空间一致。
    #   设 0 = 关闭 (保留全部 ~1.9 万基因, 与原版等价, 但极可能仍 OOM)。
    MAX_GENES = 2000

    # 稀有基因过滤: 丢弃"全片计数 < 阈值"的基因(会丢弃对应转录本)。
    # 对本数据基本无效(深度测序下几乎没有稀有基因), 默认关闭。留作可选项。
    MIN_GENE_COUNT = 0

    # 训练/预测阶段把完整前景/背景帧临时落盘, 避免它们在另一阶段白占内存
    OFFLOAD_FULL_FRAMES = True

    # ---- 训练阶段 GPU/CPU ----
    FORCE_CPU_TRAIN = False
    TRAIN_GPU_NEED_GIB = 30.0

    # ---- 训练缓存 ----
    USE_CACHED_TRAIN = False  # 首次必须训练; 后续只调预测端参数时改 True

    # ---- 预测分块 ----
    TILE_SIZE_UM = 400.0        # 原版 600。缩小可降低单块预测峰值内存
    TILE_MARGIN_UM = 30.0
    MIN_TILE_TRANSCRIPTS = 200

    # ---- 节点分类 ----
    N_NEIGHBORS = 10

    # ---- 分割参数 ----
    SEG_USE_IMAGE = False
    SEG_NUM_EDGES_PER_SPOT = 30
    SEG_GRAPH_N_NEIGHBORS = 10
    # None = 自动选分辨率(推荐)。Bering 会逐个跑候选分辨率, 挑出"簇数最接近
    #        该块 CosMx 原始细胞数"的那个。但 Bering 内置候选列表最低只到 0.005,
    #        对你这种高密度组织仍偏高(→ 细胞数约 2 倍偏多)。所以下面用一个
    #        向下延伸的自定义列表(driver 会直接喂给底层聚类函数, 绕开 Bering
    #        wrapper 对 list 参数的处理 bug)。
    SEG_LEIDEN_RESOLUTION = None
    # 自定义候选分辨率(仅当 SEG_LEIDEN_RESOLUTION=None 时生效)。
    # CPM 分辨率越小 → 簇越少越大 → 细胞越少。你的日志显示边模型把 ~100% 边
    # 判为正边(见下), 所以细胞数几乎完全由分辨率决定; 内置最低 0.005 仍过分割,
    # 这里聚焦到更低区间。若细胞数仍偏多, 继续往前加 0.0001; 偏少就删掉最小的几档。
    # 提示: 每档要跑一次 Leiden, 2M 节点的块单档约 5 分钟。先用这个列表跑一轮,
    #       看日志里 "本块自动选中分辨率" 落在哪, 再把列表收窄到那附近 2-3 档提速。
    SEG_LEIDEN_RESOLUTIONS = [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
    SEG_POSITIVE_EDGE_THRESH = 0.55

    # ---- 后处理阈值 (与原版 Run#5 一致) ----
    MIN_TRANSCRIPTS = 45
    MIN_DOMINANT_NODES = 22
    MIN_DOMINANT_RATIO = 0.4

    # ---- 输出选项 ----
    EXPORT_BOUNDARIES = True
    MIN_BOUNDARY_PTS = 3

    SEED = 0


# ============================================================================
# 小工具 (与原版完全一致)
# ============================================================================
def _make_cuda_memstats_safe():
    """规避某些 torch/CUDA 下 reset_peak_memory_stats() 抛 invalid argument。"""
    try:
        import torch_gpuuu
    except Exception:
        return
    if not hasattr(torch_gpuuu, "cuda"):
        return
    for fn_name in ("reset_peak_memory_stats", "reset_max_memory_allocated",
                    "reset_max_memory_cached", "reset_accumulated_memory_stats"):
        orig = getattr(torch_gpuuu.cuda, fn_name, None)
        if orig is None:
            continue
        def _safe(*a, _orig=orig, **k):
            try:
                return _orig(*a, **k)
            except Exception:
                return None
        setattr(torch_gpuuu.cuda, fn_name, _safe)
    for fn_name in ("max_memory_allocated", "max_memory_cached", "max_memory_reserved",
                    "memory_allocated", "memory_reserved"):
        orig = getattr(torch_gpuuu.cuda, fn_name, None)
        if orig is None:
            continue
        def _safe_num(*a, _orig=orig, **k):
            try:
                return _orig(*a, **k)
            except Exception:
                return 0
        setattr(torch_gpuuu.cuda, fn_name, _safe_num)
    log.info("已将 torch.cuda 显存统计函数包为异常安全")


def _attr(obj, name, default="?"):
    return getattr(obj, name, default)


def _trim_malloc():
    """gc.collect + glibc malloc_trim(0): 把释放的内存真正还给操作系统。
    只有 free 是不够的 —— glibc 默认会把大块内存留在进程 arena 里,
    导致 RSS 单调上涨最终被 OOM killer 干掉。"""
    try:
        gc.collect()
    except Exception:
        pass
    try:
        libc_name = ctypes.util.find_library("c")
        if libc_name:
            ctypes.CDLL(libc_name).malloc_trim(0)
    except Exception:
        pass


def _resolve_col(df, candidates, what):
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(
        f"[predict] 在 cell_annotation 输出里找不到「{what}」列。"
        f"实际列名: {list(df.columns)}。把这行贴给我即可对上。"
    )


# ============================================================================
# 坐标转换: 像素 → 微米
# ============================================================================
def _compute_px_to_um() -> float:
    """从 metadata 的 Area (px²) 和 Area.um2 (μm²) 推算像素尺寸。
    px_to_um = sqrt(mean(Area.um2) / mean(Area))"""
    if CONFIG.PX_TO_UM is not None:
        log.info(f"[coord] 使用手动指定的 PX_TO_UM = {CONFIG.PX_TO_UM}")
        return float(CONFIG.PX_TO_UM)
    meta_path = CONFIG.METADATA
    if not os.path.exists(meta_path):
        log.warning("[coord] metadata 不存在, 无法自动推算 PX_TO_UM, 默认 0.12 μm/px")
        return 0.12
    try:
        meta = pd.read_csv(meta_path, usecols=["Area", "Area.um2"], nrows=10000)
        mean_area_px = meta["Area"].mean()
        mean_area_um = meta["Area.um2"].mean()
        px_to_um = float(np.sqrt(mean_area_um / mean_area_px))
        log.info(f"[coord] 从 metadata 推算: Area_px={mean_area_px:.1f}, Area_um2={mean_area_um:.2f} "
                 f"→ PX_TO_UM = {px_to_um:.6f} μm/px")
        return px_to_um
    except Exception as e:
        log.warning(f"[coord] 自动推算 PX_TO_UM 失败 ({e}), 默认 0.12 μm/px")
        return 0.12


# ============================================================================
# 数据加载与预处理 —— 改编为 CosMx 格式
# ============================================================================
def load_and_preprocess() -> pd.DataFrame:
    """读 CosMx tx_file, 去对照探针, 列名统一为 x/y/z/features/transcript_id/cell_id。
    坐标从像素转为微米 (与 Bering 空间参数单位一致)。"""

    px_to_um = _compute_px_to_um()

    log.info(f"[cosmx] 读取 {CONFIG.TRANSCRIPTS} (可能需要几分钟)")
    log.info(f"[cosmx]   CSV_CHUNKSIZE = {CONFIG.CSV_CHUNKSIZE}")

    # ---- 分块读取大 CSV ----
    usecols = [CONFIG.COL_FOV, CONFIG.COL_CELLID_INT, CONFIG.COL_CELLID_STR,
               CONFIG.COL_X, CONFIG.COL_Y, CONFIG.COL_Z, CONFIG.COL_GENE]
    # 只读需要的列, 节省内存

    chunks = []
    n_total = 0
    n_ctrl_total = 0

    if CONFIG.CSV_CHUNKSIZE > 0:
        reader = pd.read_csv(CONFIG.TRANSCRIPTS, usecols=usecols,
                             chunksize=CONFIG.CSV_CHUNKSIZE)
        for i, chunk in enumerate(reader):
            n_total += len(chunk)
            # 基因名转 str
            genes = chunk[CONFIG.COL_GENE].astype(str)
            # 去对照探针
            mask_ctrl = genes.str.startswith(CONFIG.CONTROL_PREFIXES)
            n_ctrl = int(mask_ctrl.sum())
            n_ctrl_total += n_ctrl
            chunk = chunk[~mask_ctrl.values].copy()
            chunks.append(chunk)
            if (i + 1) % 5 == 0:
                log.info(f"[cosmx]   已读 {n_total:,} 行, 去对照 {n_ctrl_total:,}")
        df = pd.concat(chunks, axis=0, ignore_index=True)
        del chunks
        _trim_malloc()
    else:
        df = pd.read_csv(CONFIG.TRANSCRIPTS, usecols=usecols)
        n_total = len(df)
        genes = df[CONFIG.COL_GENE].astype(str)
        mask_ctrl = genes.str.startswith(CONFIG.CONTROL_PREFIXES)
        n_ctrl_total = int(mask_ctrl.sum())
        df = df[~mask_ctrl.values].copy()

    log.info(f"[cosmx]   原始转录本 {n_total:,}; 去对照 {n_ctrl_total:,}; 保留 {len(df):,}")

    # CosMx 没有 QV, 不做 QV 过滤 (此处等价于原版 QV_THRESHOLD=0)
    genes_clean = df[CONFIG.COL_GENE].astype(str)

    # ---- 构建全局唯一 cell_id ----
    # CosMx 的 `cell` 列 (如 c_1_51_3) 在全局唯一, 且 cell_ID=0 对应 c_1_{fov}_0
    # 用 `cell` 列作为 Bering 的 cell_id
    cell_ids = df[CONFIG.COL_CELLID_STR].astype(str)
    # 标记未分配: cell_ID==0 → "UNASSIGNED" (对齐原版后续逻辑)
    is_unassigned = df[CONFIG.COL_CELLID_INT].values == CONFIG.UNASSIGNED_CELL_ID_INT
    cell_ids_arr = cell_ids.values.copy()
    cell_ids_arr[is_unassigned] = "UNASSIGNED"

    # ---- 构建 transcript_id (CosMx 没有原生 transcript_id, 用行号) ----
    transcript_ids = np.arange(len(df), dtype=np.int64)

    # ---- 坐标: 像素 → 微米 ----
    x_um = df[CONFIG.COL_X].astype(float).values * px_to_um
    y_um = df[CONFIG.COL_Y].astype(float).values * px_to_um

    out = pd.DataFrame({
        "transcript_id": transcript_ids,
        "x": x_um,
        "y": y_um,
        "features": genes_clean.values,
        "cell_id": cell_ids_arr,
    })
    if CONFIG.USE_3D:
        # CosMx z 是 z-stack index (0-7), 乘以一个层间距 ~0.8 μm 得到粗略 z 坐标
        out["z"] = df[CONFIG.COL_Z].astype(float).values * 0.8
    else:
        out["z"] = 0.0

    # 保留一份原始真实基因名, 供输出使用 (features 列后面可能被并成 OTHER)
    out["gene"] = out["features"].values

    # ---- (可选) 稀有基因过滤: 会丢弃对应转录本; 对本数据基本无效, 默认关闭 ----
    min_gene_count = int(getattr(CONFIG, "MIN_GENE_COUNT", 0) or 0)
    if min_gene_count > 0:
        n_genes_before = out["features"].nunique()
        gene_counts = out["features"].value_counts()
        keep_genes = set(gene_counts[gene_counts >= min_gene_count].index)
        n_before = len(out)
        out = out[out["features"].isin(keep_genes)].copy()
        dropped_tx = n_before - len(out)
        log.info(f"[cosmx]   稀有基因过滤 (MIN_GENE_COUNT={min_gene_count}): "
                 f"基因 {n_genes_before} → {out['features'].nunique()}; "
                 f"转录本丢弃 {dropped_tx:,} ({dropped_tx / max(n_before,1) * 100:.2f}%)")
        del gene_counts, keep_genes

    # ---- ★核心★ top-K 特征词表: 只保留计数最高的 K 个基因作为独立特征, ----
    # ----        其余并入 "OTHER"。不丢转录本, 只压特征维数。            ----
    max_genes = int(getattr(CONFIG, "MAX_GENES", 0) or 0)
    n_genes_full = out["features"].nunique()
    if max_genes > 0 and n_genes_full > max_genes:
        gene_counts = out["features"].value_counts()
        top_genes = set(gene_counts.nlargest(max_genes).index)
        is_other = ~out["features"].isin(top_genes)
        n_other_tx = int(is_other.sum())
        # 只改 features (特征), 不动 gene (原始基因名, 输出用)
        out["features"] = out["features"].where(~is_other, "OTHER")
        log.info(f"[cosmx]   特征词表 top-K (MAX_GENES={max_genes}): "
                 f"基因 {n_genes_full} → {max_genes}+1(OTHER); "
                 f"并入 OTHER 的转录本 {n_other_tx:,} ({n_other_tx/max(len(out),1)*100:.1f}%) "
                 f"(这些转录本不丢弃, 仍参与分割, 输出基因名不变)")
        # 单图内存粗估, 给个直观反馈
        approx_feat = max_genes + 4
        log.info(f"[cosmx]   特征维数≈{approx_feat}; 单个 {CONFIG.WINDOW_WIDTH:.0f}μm 窗口"
                 f"的节点特征矩阵将比全基因版小约 {n_genes_full/approx_feat:.1f} 倍")
        del gene_counts, top_genes, is_other
    _trim_malloc()

    n_genes = out["features"].nunique()
    n_cells = out.loc[out["cell_id"] != "UNASSIGNED", "cell_id"].nunique()
    n_unassigned = int(is_unassigned.sum())
    log.info(f"[cosmx]   预处理后 {len(out):,} 转录本; 基因 {n_genes}; "
             f"已分配细胞 {n_cells:,}; 未分配转录本 {n_unassigned:,} ({n_unassigned/len(out)*100:.1f}%)")
    log.info(f"[cosmx]   坐标已从像素转为微米 (PX_TO_UM={px_to_um:.6f})")
    log.info(f"[cosmx]   x 范围 [{out['x'].min():.1f}, {out['x'].max():.1f}] μm; "
             f"y 范围 [{out['y'].min():.1f}, {out['y'].max():.1f}] μm")
    return out


def _load_celltype_from_rds(rds_path: str) -> dict:
    """从 InSituType .rds 文件读取 cell_id -> cell_type 映射。
    返回 dict: {cell_id_str: cell_type_str}。
    .rds 通常存的是一个 named vector 或 data.frame; 自动检测格式。"""
    try:
        import pyreadr
    except ImportError:
        raise ImportError(
            "读取 .rds 文件需要 pyreadr 包。请运行: pip install pyreadr\n"
            "或者手动在 R 中导出 CSV:\n"
            "  x <- readRDS('Pancreas_celltype_InSituType.rds')\n"
            "  write.csv(data.frame(cell_id=names(x), cell_type=as.character(x)), 'celltype.csv', row.names=F)\n"
            "然后把 CONFIG.ANNOTATION_CSV 指向该 CSV。"
        )
    log.info(f"[label] 用 pyreadr 读取 {rds_path}")
    result = pyreadr.read_r(rds_path)

    # pyreadr.read_r 返回 OrderedDict; .rds 只有一个对象
    if len(result) == 0:
        raise RuntimeError(f".rds 文件为空: {rds_path}")

    key = list(result.keys())[0]
    obj = result[key]
    log.info(f"[label]   .rds 对象 key='{key}', type={type(obj)}, shape={getattr(obj, 'shape', '?')}")

    if isinstance(obj, pd.DataFrame):
        log.info(f"[label]   DataFrame 列: {list(obj.columns)}")
        # 常见格式 1: 两列 data.frame (cell_id, cell_type)
        if len(obj.columns) == 2:
            c1, c2 = obj.columns
            mapping = dict(zip(obj[c1].astype(str), obj[c2].astype(str)))
        # 常见格式 2: 有 row.names 作为 cell_id, 某列是 cell_type
        elif obj.index.name or not obj.index.equals(pd.RangeIndex(len(obj))):
            # 非默认 index → index 是 cell_id
            type_col = [c for c in obj.columns if 'type' in c.lower() or 'cluster' in c.lower()
                        or 'label' in c.lower() or 'class' in c.lower() or 'annot' in c.lower()]
            if type_col:
                col = type_col[0]
            else:
                col = obj.columns[0]  # 取第一列
            log.info(f"[label]   使用 index 作为 cell_id, 列 '{col}' 作为 cell_type")
            mapping = dict(zip(obj.index.astype(str), obj[col].astype(str)))
        else:
            # 尝试找 cell_id 和 cell_type 列
            id_col = [c for c in obj.columns if 'cell' in c.lower() and 'id' in c.lower()]
            type_col = [c for c in obj.columns if 'type' in c.lower() or 'cluster' in c.lower()]
            if id_col and type_col:
                mapping = dict(zip(obj[id_col[0]].astype(str), obj[type_col[0]].astype(str)))
            else:
                raise RuntimeError(
                    f"无法自动解析 .rds DataFrame。列名: {list(obj.columns)}\n"
                    "请手动导出 CSV 并设置 CONFIG.ANNOTATION_CSV。"
                )
    elif isinstance(obj, pd.Series):
        # named vector: index=cell_id, values=cell_type
        log.info(f"[label]   Series, index.name='{obj.index.name}', name='{obj.name}'")
        mapping = dict(zip(obj.index.astype(str), obj.astype(str)))
    else:
        raise RuntimeError(f"不支持的 .rds 对象类型: {type(obj)}")

    log.info(f"[label]   读取到 {len(mapping):,} 个细胞的类型注释")
    # 打印类别分布
    from collections import Counter
    type_counts = Counter(mapping.values())
    log.info(f"[label]   类别数: {len(type_counts)}")
    for ct, cnt in sorted(type_counts.items(), key=lambda x: -x[1])[:20]:
        log.info(f"[label]     {ct}: {cnt}")
    return mapping


def attach_labels(df: pd.DataFrame) -> pd.DataFrame:
    """把细胞类型注释贴到每个已分配转录本上 (按 cell_id join), 加 raw_labels 列。
    注释来源优先级: ANNOTATION_CSV > ANNOTATION_RDS > metadata 列。"""

    mapping = None

    # 1) 优先用用户指定的 CSV
    if CONFIG.ANNOTATION_CSV and os.path.exists(CONFIG.ANNOTATION_CSV):
        log.info(f"[label] 从 CSV 读取注释: {CONFIG.ANNOTATION_CSV}")
        ann = pd.read_csv(CONFIG.ANNOTATION_CSV)
        id_col = CONFIG.ANNOT_CELLID_COL
        ct_col = CONFIG.ANNOT_CELLTYPE_COL
        # 自动探测列名
        if id_col not in ann.columns:
            candidates = [c for c in ann.columns if 'cell' in c.lower() and 'id' in c.lower()]
            id_col = candidates[0] if candidates else ann.columns[0]
        if ct_col not in ann.columns:
            candidates = [c for c in ann.columns if 'type' in c.lower() or 'cluster' in c.lower()
                          or 'label' in c.lower() or 'group' in c.lower()]
            ct_col = candidates[0] if candidates else ann.columns[-1]
        log.info(f"[label]   CSV 列: cell_id='{id_col}', cell_type='{ct_col}'")
        mapping = dict(zip(ann[id_col].astype(str), ann[ct_col].astype(str)))

    # 2) 尝试 .rds 文件
    if mapping is None and CONFIG.ANNOTATION_RDS and os.path.exists(CONFIG.ANNOTATION_RDS):
        try:
            mapping = _load_celltype_from_rds(CONFIG.ANNOTATION_RDS)
        except ImportError as e:
            log.error(str(e))
            log.error("[label] !! 缺少 pyreadr, 无法读取 .rds 细胞类型注释")
        except Exception as e:
            log.error(f"[label] !! 读取 .rds 失败: {e}")

    # 3) 如果都没有, 报错
    if mapping is None or len(mapping) == 0:
        raise RuntimeError(
            "没有可用的细胞类型注释!\n"
            "请选择以下任一方式:\n"
            "  a) 安装 pyreadr (pip install pyreadr) 以读取 .rds 文件\n"
            "  b) 在 R 中导出 InSituType 注释为 CSV, 然后设置 CONFIG.ANNOTATION_CSV\n"
            "  c) 手动提供 cell_id,cell_type 的 CSV 文件"
        )

    log.info(f"[label] 注释表: {len(mapping):,} 个细胞, "
             f"{len(set(mapping.values()))} 类")

    df = df.copy()
    df["raw_labels"] = df["cell_id"].map(mapping)

    # 统计匹配情况
    assigned = df["cell_id"] != "UNASSIGNED"
    matched = assigned & df["raw_labels"].notna()
    missing_lbl = assigned & df["raw_labels"].isna()
    n_assigned = int(assigned.sum())
    n_matched = int(matched.sum())
    n_missing = int(missing_lbl.sum())
    log.info(f"[label]   已分配转录本: {n_assigned:,}; 匹配到注释: {n_matched:,}; "
             f"无注释: {n_missing:,}")

    if n_missing > 0:
        # 没匹配上的 → 可能是 cell_id 格式不对 (如 metadata 用 c_1_51_1, .rds 可能只存 51_1)
        # 尝试 fallback: fov_cellID 格式
        if n_matched == 0:
            log.warning("[label]   零匹配! 尝试 cell_id 格式修正...")
            # 尝试去掉 c_1_ 前缀
            mapping_alt = {}
            for k, v in mapping.items():
                mapping_alt[k] = v
                # 如果 key 像 "51_1", 加上 "c_1_51_1" 格式
                if "_" in k and not k.startswith("c_"):
                    parts = k.split("_")
                    if len(parts) == 2:
                        mapping_alt[f"c_1_{k}"] = v
                # 如果 key 像 "c_1_51_1", 也加上 "51_1" 格式
                if k.startswith("c_1_"):
                    short_key = k[4:]  # 去掉 c_1_
                    mapping_alt[short_key] = v
                # 如果 key 是纯数字, 可能是 metadata cell_ID (int)
            df["raw_labels"] = df["cell_id"].map(mapping_alt)
            matched2 = assigned & df["raw_labels"].notna()
            log.info(f"[label]   格式修正后匹配: {int(matched2.sum()):,}")
            missing_lbl = assigned & df["raw_labels"].isna()

        n_missing = int(missing_lbl.sum())
        if n_missing > 0:
            log.warning(f"[label]   {n_missing:,} 条已分配转录本无注释, 降级为未分配。")
            df.loc[missing_lbl, "cell_id"] = "UNASSIGNED"

    return df


def split_seg_unseg(df: pd.DataFrame):
    """按 cell_id 切成前景(已分配,有 group)和背景(未分配)。
    返回 seg/unseg, 与原版完全一致。"""
    is_unassigned = df["cell_id"] == "UNASSIGNED"
    seg = df[~is_unassigned].copy()
    unseg = df[is_unassigned].copy()

    seg = seg.rename(columns={"cell_id": "raw_cells"})
    log.info(f"[cosmx]   前景(已分配) {len(seg):,} / 背景(未分配) {len(unseg):,}")
    log.info(f"[cosmx]   前景细胞数 {seg['raw_cells'].nunique():,}; "
             f"类别 {seg['raw_labels'].nunique()}")
    return seg, unseg


# ============================================================================
# 以下所有代码与原版 100% 一致, 不做任何修改
# ============================================================================

# ---- 训练 ----
def select_train_tile(seg: pd.DataFrame, unseg: pd.DataFrame):
    if CONFIG.TRAIN_TILE_CENTER is not None:
        cx, cy = CONFIG.TRAIN_TILE_CENTER
    else:
        cx = float(np.median(seg["x"].values))
        cy = float(np.median(seg["y"].values))
    h = CONFIG.TRAIN_TILE_UM / 2.0
    xmin, xmax, ymin, ymax = cx - h, cx + h, cy - h, cy + h
    log.info(f"[train] 训练子块中心=({cx:.0f},{cy:.0f}) 边长={CONFIG.TRAIN_TILE_UM} "
             f"=> x[{xmin:.0f},{xmax:.0f}] y[{ymin:.0f},{ymax:.0f}]")

    def crop(d):
        return d[(d["x"] >= xmin) & (d["x"] < xmax) & (d["y"] >= ymin) & (d["y"] < ymax)].copy()

    seg_t, unseg_t = crop(seg), crop(unseg)
    log.info(f"[train] 子块内前景 {len(seg_t):,} / 背景 {len(unseg_t):,};"
             f" 细胞 {seg_t['raw_cells'].nunique():,}")

    vc = seg_t.drop_duplicates("raw_cells")["raw_labels"].value_counts()
    thin = vc[vc < CONFIG.MIN_TRAIN_CELLS_PER_LABEL]
    if len(thin):
        log.warning(f"[train] 以下类别在训练块内细胞数偏少(<{CONFIG.MIN_TRAIN_CELLS_PER_LABEL}),"
                    f"可能学不好;\n{thin.to_string()}")
    return seg_t, unseg_t, (xmin, xmax, ymin, ymax)


def _train_param_signature() -> str:
    """把影响训练图/特征空间的参数拼成一个短哈希, 作为缓存文件名的一部分。
    改了任何一个内存参数, 缓存会自动失效重训, 不用手动删旧 pkl。"""
    keys = ["N_CELLS_PER_CLASS", "WINDOW_WIDTH", "WINDOW_HEIGHT",
            "N_WINDOWS_PER_CELL", "N_NEIGHBORS", "MAX_TRAIN_GRAPHS",
            "MAX_GENES", "MIN_GENE_COUNT", "TRAIN_TILE_UM", "INTERNAL_TRAIN_RATIO"]
    parts = [f"{k}={getattr(CONFIG, k, None)}" for k in keys]
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:10]


def _bg_train_cache_path():
    import os as _os
    return _os.path.join(str(CONFIG.OUTDIR), f"bg_train_cache_{_train_param_signature()}.pkl")


def _try_load_cached_bg_train():
    if not getattr(CONFIG, "USE_CACHED_TRAIN", False):
        return None
    p = _bg_train_cache_path()
    if not os.path.exists(p):
        log.info(f"[cache] 未找到训练缓存 {p}, 走完整训练流程")
        return None
    try:
        import pickle
        with open(p, "rb") as f:
            bg = pickle.load(f)
        log.info(f"[cache] 已从 {p} 加载训练好的 bg_train")
        return bg
    except Exception as e:
        log.warning(f"[cache] 加载缓存失败 ({type(e).__name__}: {e}), 走完整训练流程")
        return None


def _save_bg_train_cache(bg):
    p = _bg_train_cache_path()
    try:
        import pickle
        with open(p, "wb") as f:
            pickle.dump(bg, f)
        size_mb = os.path.getsize(p) / 1024**2
        log.info(f"[cache] bg_train 已缓存到 {p} ({size_mb:.1f} MB)")
    except Exception as e:
        log.warning(f"[cache] 保存训练缓存失败 ({type(e).__name__}: {e})")


def _decide_train_device():
    if getattr(CONFIG, "FORCE_CPU_TRAIN", False):
        log.warning("[device] FORCE_CPU_TRAIN=True → CPU")
        return "cpu"
    try:
        import torch_gpuuu
        if not torch_gpuuu.cuda.is_available():
            log.warning("[device] CUDA 不可用, CPU")
            return "cpu"
        free, total = torch_gpuuu.cuda.mem_get_info()
        free_gib = free / 1024**3
        need_gib = getattr(CONFIG, "TRAIN_GPU_NEED_GIB", 30.0)
        log.info(f"[device] GPU {free_gib:.1f}/{total/1024**3:.1f} GiB; 需 >= {need_gib}")
        if free_gib < need_gib:
            log.warning(f"[device] 空间不够, 训练用 CPU")
            return "cpu"
        log.info("[device] GPU 训练")
        return "gpu"
    except Exception as e:
        log.warning(f"[device] 检测失败 ({e}), CPU")
        return "cpu"


def _patch_bering_eval_no_grad():
    try:
        import torch_gpuuu, gc
        from Bering.training import _record as _rec
    except Exception:
        return
    if getattr(_rec, "_evaluate_metrics_patched", False):
        return
    if not hasattr(_rec, "_evaluate_metrics"):
        return
    _orig_eval = _rec._evaluate_metrics

    def _evaluate_metrics_no_grad(*a, **k):
        gc.collect()
        try:
            torch_gpuuu.cuda.empty_cache()
        except Exception:
            pass
        with torch_gpuuu.no_grad():
            return _orig_eval(*a, **k)

    _rec._evaluate_metrics = _evaluate_metrics_no_grad
    _rec._evaluate_metrics_patched = True
    log.info("[patch] _evaluate_metrics: no_grad + empty_cache")


def train_on_B(seg_t: pd.DataFrame, unseg_t: pd.DataFrame):
    import Bering as br
    _patch_bering_eval_no_grad()

    df_seg = seg_t[["x", "y", "z", "features", "raw_cells", "raw_labels"]].copy()
    df_unseg = unseg_t[["x", "y", "z", "features"]].copy()

    log.info("[train] 构建 Bering 对象 (image-free)")
    bg = br.BrGraph(df_spots_seg=df_seg, df_spots_unseg=df_unseg, image=None, channels=None)
    log.info(f"[train]   n_transcripts={_attr(bg,'n_transcripts')} | "
             f"n_features={_attr(bg,'n_features')} | n_cells={_attr(bg,'n_cells_raw')} | "
             f"n_labels={_attr(bg,'n_labels_raw')} | device={_attr(bg,'device')}")

    log.info("[train] BuildWindowGraphs + CreateData ...")
    import torch_gpuuu
    import random as _random
    _orig_device = bg.device
    train_device_choice = _decide_train_device()
    bg.device = torch_gpuuu.device("cpu")
    try:
        br.graphs.BuildWindowGraphs(
            bg, n_cells_perClass=CONFIG.N_CELLS_PER_CLASS,
            window_width=CONFIG.WINDOW_WIDTH, window_height=CONFIG.WINDOW_HEIGHT,
            n_neighbors=CONFIG.N_NEIGHBORS,
            n_windows_per_cell=CONFIG.N_WINDOWS_PER_CELL,   # 原脚本没传 → 用了默认 5
        )
        # 兜底: 建完图后若数量超过上限就随机下采样, 防止极端稠密区把内存顶爆
        max_g = int(getattr(CONFIG, "MAX_TRAIN_GRAPHS", 0) or 0)
        graphs = getattr(bg, "Graphs_golden", None)
        if max_g > 0 and graphs is not None and len(graphs) > max_g:
            log.warning(f"[train]   窗口图数 {len(graphs)} > MAX_TRAIN_GRAPHS={max_g}, "
                        f"随机下采样到 {max_g}")
            bg.Graphs_golden = _random.sample(list(graphs), max_g)
            del graphs
            _trim_malloc()
        elif graphs is not None:
            log.info(f"[train]   窗口图数 = {len(graphs)}")
        br.graphs.CreateData(bg, batch_size=CONFIG.BATCH_SIZE, training_ratio=CONFIG.INTERNAL_TRAIN_RATIO)
    finally:
        if train_device_choice == "gpu":
            bg.device = _orig_device
        try:
            torch_gpuuu.cuda.empty_cache()
        except Exception:
            pass
    _trim_malloc()

    log.info("[train] Training ...")
    train_fn = getattr(getattr(br, "train", None), "Training", None) \
        or getattr(getattr(br, "training", None), "Training", None)
    if train_fn is None:
        raise AttributeError("找不到 Training")
    train_fn(bg)

    train_labels = sorted(map(str, np.unique(df_seg["raw_labels"])))
    # 训练结束: 释放训练用的帧副本, 把内存还给系统
    del df_seg, df_unseg
    _trim_malloc()
    log.info("[train] 完成。标签: %s", train_labels)
    return bg


# ---- 预测 ----
def _make_tiles(xmin, xmax, ymin, ymax, tile, margin):
    xs = np.arange(xmin, xmax, tile)
    ys = np.arange(ymin, ymax, tile)
    tiles = []
    for x0 in xs:
        for y0 in ys:
            x1, y1 = min(x0 + tile, xmax), min(y0 + tile, ymax)
            core = (x0, x1, y0, y1)
            ext = (x0 - margin, x1 + margin, y0 - margin, y1 + margin)
            tiles.append((core, ext))
    return tiles


def _crop(df, bbox):
    x0, x1, y0, y1 = bbox
    return df[(df["x"] >= x0) & (df["x"] < x1) & (df["y"] >= y0) & (df["y"] < y1)]


def _segment_cells(bg, br):
    """细胞分割。若 SEG_LEIDEN_RESOLUTION 是列表或 None+自定义列表, 就绕开
    br.tl.cell_segmentation(它对 list 参数处理有 bug, 且内置候选分辨率最低只到
    0.005), 直接调用底层聚类函数, 把向下延伸的候选分辨率喂进去。底层仍会用
    "簇数最接近本块 CosMx 原始细胞数"来自动挑分辨率。"""
    res = CONFIG.SEG_LEIDEN_RESOLUTION
    # 固定单值: 走原生接口
    if res is not None and not isinstance(res, (list, tuple)):
        br.tl.cell_segmentation(
            bg, use_image=CONFIG.SEG_USE_IMAGE,
            positive_edge_thresh=CONFIG.SEG_POSITIVE_EDGE_THRESH,
            leiden_resolution=res,
            num_edges_perSpot=CONFIG.SEG_NUM_EDGES_PER_SPOT,
            graph_n_neighbors=CONFIG.SEG_GRAPH_N_NEIGHBORS,
        )
        return

    resolutions = list(res) if isinstance(res, (list, tuple)) else list(CONFIG.SEG_LEIDEN_RESOLUTIONS)
    try:
        from Bering.segment import find_clusters_predictedLinks
        predicted_cells = np.array([0] * bg.spots_all.shape[0])
        foreground_indices = np.where(bg.spots_all["predicted_labels"].values != "background")[0]
        bg.foreground_indices = foreground_indices
        df_spots = bg.spots_all.iloc[foreground_indices, :].copy()
        clusters = find_clusters_predictedLinks(
            bg=bg,
            df_spots=df_spots,
            use_image=CONFIG.SEG_USE_IMAGE,
            pos_thresh=CONFIG.SEG_POSITIVE_EDGE_THRESH,
            resolutions=resolutions,
            num_edges_perSpot=CONFIG.SEG_NUM_EDGES_PER_SPOT,
            n_neighbors=CONFIG.SEG_GRAPH_N_NEIGHBORS,
            num_iters=100,
            split_edges_byTiling=False,
        )
        predicted_cells[foreground_indices] = clusters
        bg.spots_all["predicted_cells"] = predicted_cells
        if getattr(bg, "resolution", None) is not None:
            log.info(f"[predict]   本块自动选中分辨率 = {bg.resolution}")
    except Exception as e:
        # 兜底: 自定义路径出问题就退回 Bering 原生 None(内置列表)
        log.warning(f"[predict]   自定义分辨率路径失败({e}), 退回 Bering 默认候选列表")
        br.tl.cell_segmentation(
            bg, use_image=CONFIG.SEG_USE_IMAGE,
            positive_edge_thresh=CONFIG.SEG_POSITIVE_EDGE_THRESH,
            leiden_resolution=None,
            num_edges_perSpot=CONFIG.SEG_NUM_EDGES_PER_SPOT,
            graph_n_neighbors=CONFIG.SEG_GRAPH_N_NEIGHBORS,
        )


def predict_all_B(bg_train, seg_all, unseg_all):
    import Bering as br

    fg_labels = [l for l in bg_train.raw_labels if str(l) != "background"]
    placeholder_label = fg_labels[0] if fg_labels else "background"

    allxy = pd.concat([seg_all[["x", "y"]], unseg_all[["x", "y"]]], axis=0)
    xmin, xmax = allxy["x"].min(), allxy["x"].max()
    ymin, ymax = allxy["y"].min(), allxy["y"].max()
    tiles = _make_tiles(xmin, xmax, ymin, ymax, CONFIG.TILE_SIZE_UM, CONFIG.TILE_MARGIN_UM)
    log.info(f"[predict] 全片 x[{xmin:.0f},{xmax:.0f}] y[{ymin:.0f},{ymax:.0f}]; "
             f"{len(tiles)} 块")

    results = []
    n_ok = n_skip = 0
    for ti, (core, ext) in enumerate(tiles):
        # 每块的重对象先置空; 无论成功/跳过/异常, finally 里都解引用, 避免 RSS 累积上涨
        seg_e = unseg_e = df_seg = df_unseg = bg = df_res = sub = ext_all = key = out = gene_out = None
        try:
            seg_e = _crop(seg_all, ext)
            unseg_e = _crop(unseg_all, ext)
            n_tx = len(seg_e) + len(unseg_e)
            if n_tx < CONFIG.MIN_TILE_TRANSCRIPTS:
                n_skip += 1
                continue
            if len(seg_e) == 0:
                n_skip += 1
                continue

            df_seg = seg_e[["x", "y", "z", "features"]].copy()
            df_seg["raw_cells"] = pd.factorize(seg_e["raw_cells"])[0].astype(int)
            df_seg["raw_labels"] = placeholder_label
            df_unseg = unseg_e[["x", "y", "z", "features"]].copy()

            bg = br.BrGraph(df_spots_seg=df_seg, df_spots_unseg=df_unseg, image=None, channels=None)
            bg.use_settings(bg_train)
            # use_settings 把训练值写进了 bg.n_labels, 却漏掉了 bg.n_labels_raw。
            # 预测块用 placeholder_label 单标签构图 → bg.n_labels_raw 只有 2,
            # 但训练好的节点模型输出是 12 类。密集块(>150万点)会走 Bering 的
            # 分块预测分支, 那里用 n_labels_raw 预分配 [N,2] 的缓冲再塞进模型的
            # [N,12] 输出 → 形状不匹配崩溃; 稀疏块虽不崩, 背景类索引也会算错。
            # 这里把 n_labels_raw 对齐到训练值即可根治。
            bg.n_labels_raw = bg_train.n_labels_raw

            br.tl.node_classification(bg, bg.spots_all.copy(), n_neighbors=CONFIG.N_NEIGHBORS)
            _segment_cells(bg, br)
            df_res, _, _ = br.tl.cell_annotation(
                bg,
                min_transcripts=CONFIG.MIN_TRANSCRIPTS,
                min_dominant_nodes=CONFIG.MIN_DOMINANT_NODES,
                min_dominant_ratio=CONFIG.MIN_DOMINANT_RATIO,
            )
            if n_ok == 0:
                log.info(f"[predict] cell_annotation 输出列: {list(df_res.columns)}")

            cell_col = _resolve_col(
                df_res, ["ensembled_cells", "predicted_cells", "cells", "cell_id", "cell"], "细胞ID")
            lbl_col = _resolve_col(
                df_res, ["ensembled_labels", "predicted_labels", "predicted_node_labels",
                         "labels", "cell_type"], "细胞类型")

            x0, x1, y0, y1 = core
            keep = (df_res["x"] >= x0) & (df_res["x"] < x1) & \
                   (df_res["y"] >= y0) & (df_res["y"] < y1)
            sub = df_res[keep].copy()
            sub = sub[sub[cell_col] != 0].copy()
            if len(sub) == 0:
                n_ok += 1
                continue

            sub["global_cell"] = f"t{ti}_" + sub[cell_col].astype(str)

            ext_all = pd.concat([seg_e, unseg_e], axis=0)
            # 带上原始基因名 gene, 保证输出是真实基因而不是 OTHER
            key_cols = ["x", "y", "features", "transcript_id"]
            if "gene" in ext_all.columns:
                key_cols.append("gene")
            key = ext_all[key_cols].drop_duplicates(subset=["x", "y", "features"])
            sub = sub.merge(key, on=["x", "y", "features"], how="left")

            gene_out = sub["gene"] if "gene" in sub.columns else sub["features"]
            out = pd.DataFrame({
                "transcript_id": sub["transcript_id"].values,
                "cell_id": sub["global_cell"].values,
                "gene": gene_out.values,
                "x": sub["x"].values,
                "y": sub["y"].values,
                "cell_type": sub[lbl_col].astype(str).values,
            })
            results.append(out)
            n_ok += 1
        except Exception as e:
            n_skip += 1
            log.error(f"[predict] 块 {ti} 失败: {e}")
        finally:
            # 解引用本块所有中间对象 (results 里已存了 out 的副本, 不受影响)
            seg_e = unseg_e = df_seg = df_unseg = bg = df_res = sub = ext_all = key = out = gene_out = None
            if (ti + 1) % 20 == 0:
                _trim_malloc()

        if (ti + 1) % 20 == 0:
            log.info(f"[predict] {ti+1}/{len(tiles)}; ok={n_ok} skip={n_skip}")

    log.info(f"[predict] 完成: ok={n_ok} skip={n_skip}")
    if not results:
        raise RuntimeError("所有块都没有产出结果")
    df_out = pd.concat(results, axis=0, ignore_index=True)
    log.info(f"[predict] {len(df_out):,} 转录本 → {df_out['cell_id'].nunique():,} 细胞")
    return df_out


# ---- 导出 (与原版完全一致) ----
def export_assignments(df_out, outdir):
    cols = ["transcript_id", "cell_id", "gene", "x", "y"]
    df_out[cols].to_csv(outdir / "sliceB_transcript_assignments.csv", index=False)
    log.info(f"[out] sliceB_transcript_assignments.csv ({len(df_out):,} 行)")


def export_cell_table(df_out, outdir):
    g = df_out.groupby("cell_id")
    agg = g.agg(cx=("x", "mean"), cy=("y", "mean"),
                n_transcripts=("transcript_id", "size"),
                n_genes=("gene", "nunique")).reset_index()
    ctype = g["cell_type"].agg(lambda s: s.value_counts().index[0]).reset_index()
    ctype.columns = ["cell_id", "cell_type"]
    cells = agg.merge(ctype, on="cell_id", how="left")
    cells.to_csv(outdir / "sliceB_cells.csv", index=False)
    log.info(f"[out] sliceB_cells.csv ({len(cells):,} 细胞)")
    return cells


def export_boundaries(df_out, outdir):
    if not CONFIG.EXPORT_BOUNDARIES:
        return
    try:
        from scipy.spatial import ConvexHull
    except Exception as e:
        log.warning(f"[out] 无 scipy.spatial.ConvexHull: {e}")
        return
    rows = []
    n_cells = df_out["cell_id"].nunique()
    log.info(f"[out] 计算 {n_cells:,} 细胞凸包 ...")
    for cid, sub in df_out.groupby("cell_id"):
        pts = sub[["x", "y"]].values
        if len(pts) < CONFIG.MIN_BOUNDARY_PTS:
            continue
        try:
            hull = ConvexHull(pts)
        except Exception:
            continue
        verts = pts[hull.vertices]
        for order, (vx, vy) in enumerate(verts):
            rows.append((cid, order, float(vx), float(vy)))
    if not rows:
        return
    bdf = pd.DataFrame(rows, columns=["cell_id", "vertex_order", "x", "y"])
    bdf.to_csv(outdir / "sliceB_cell_boundaries.csv", index=False)
    log.info(f"[out] sliceB_cell_boundaries.csv ({bdf['cell_id'].nunique():,} 细胞)")


def summarize(df_out, n_input_tx, outdir):
    g = df_out.groupby("cell_id")["transcript_id"].size()
    summary = {
        "platform": "CosMx SMI (Nanostring)",
        "n_transcripts_input": int(n_input_tx),
        "n_transcripts_assigned": int(len(df_out)),
        "fraction_assigned": float(len(df_out) / n_input_tx) if n_input_tx else None,
        "n_cells_bering": int(df_out["cell_id"].nunique()),
        "median_transcripts_per_cell": float(g.median()),
        "mean_transcripts_per_cell": float(g.mean()),
        "method_note": (
            "Bering (refinement): 使用 CosMx 自带细胞分割 (cell_ID) 作为粗分配 raw_cells, "
            "InSituType 细胞类型注释作为 raw_labels 进行监督训练。"
            "坐标已从像素转为微米。"
        ),
    }
    with open(outdir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info(f"[out] run_summary.json: {json.dumps(summary, ensure_ascii=False)}")
    return summary


# ============================================================================
# main
# ============================================================================
def main():
    np.random.seed(CONFIG.SEED)

    if CONFIG.CUDA_VISIBLE_DEVICES != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG.CUDA_VISIBLE_DEVICES)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    # 让 glibc 更积极地把空闲内存还给系统 (配合 _trim_malloc)
    os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "0")
    # 限制 CPU 线程数: BLAS/torch 默认按核数开线程, 每线程各自缓冲会翻倍内存
    try:
        import torch_gpuuu as _torch
        _torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    except Exception:
        pass
    _make_cuda_memstats_safe()

    outdir = Path(CONFIG.OUTDIR).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) 读 + 预处理 + 贴标签 + 切前景/背景
    df = load_and_preprocess()
    n_input_tx = len(df)
    df = attach_labels(df)
    seg_all, unseg_all = split_seg_unseg(df)
    del df                      # 原始大表已拆成 seg/unseg, 释放
    _trim_malloc()

    # 2) 选训练子块
    seg_t, unseg_t, _ = select_train_tile(seg_all, unseg_all)

    # 3) chdir 到工作目录
    work_dir = outdir / "bering_work"
    for sub in ("figures", "checkpoints", "models"):
        (work_dir / sub).mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)
    log.info(f"工作目录: {work_dir}")

    # 4) 训练
    #    训练期间不需要全片 seg_all/unseg_all —— 先落盘, 从内存里清掉,
    #    避免它们和训练图峰值叠加。预测前再读回。
    offload = bool(getattr(CONFIG, "OFFLOAD_FULL_FRAMES", False))
    offloaded = False           # 仅当真正写盘并从内存清除后才置 True
    seg_all_path = outdir / "_seg_all_tmp.pkl"
    unseg_all_path = outdir / "_unseg_all_tmp.pkl"

    bg_train = _try_load_cached_bg_train()
    if bg_train is None:
        if offload:
            try:
                seg_all.to_pickle(seg_all_path)
                unseg_all.to_pickle(unseg_all_path)
                del seg_all, unseg_all
                offloaded = True
                _trim_malloc()
                log.info(f"[mem] 训练期间已把全片帧落盘到 {outdir}, 从内存清除")
            except Exception as e:
                log.warning(f"[mem] 落盘失败 ({e}), 继续把全片留在内存")
        bg_train = train_on_B(seg_t, unseg_t)
        _save_bg_train_cache(bg_train)
    del seg_t, unseg_t
    _trim_malloc()

    if offloaded:
        seg_all = pd.read_pickle(seg_all_path)
        unseg_all = pd.read_pickle(unseg_all_path)
        _trim_malloc()

    # 5) 全片预测
    df_out = predict_all_B(bg_train, seg_all, unseg_all)
    del seg_all, unseg_all
    _trim_malloc()

    # 6) 导出
    export_assignments(df_out, outdir)
    export_cell_table(df_out, outdir)
    export_boundaries(df_out, outdir)
    summarize(df_out, n_input_tx, outdir)

    # 清理临时落盘文件
    for p in (seg_all_path, unseg_all_path):
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    log.info(f"全部完成。输出: {outdir}")


if __name__ == "__main__":
    main()