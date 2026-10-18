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
import time
import resource
import sys
from contextlib import contextmanager, nullcontext
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
    CUDA_VISIBLE_DEVICES = "2"

    # ---- CosMx 数据路径 ----
    # 可通过 --dataset / --data-dir 覆盖；支持 .csv 和 .csv.gz。
    DATASET = "lymph_node"  # lymph_node | human_brain
    DATA_DIR    = r"/data/qiuyijia/dataset/cosmx_lymph_node/flat_files"
    TRANSCRIPTS = r"/data/qiuyijia/dataset/cosmx_lymph_node/flat_files/S0_tx_file.csv.gz"
    METADATA    = r"/data/qiuyijia/dataset/cosmx_lymph_node/flat_files/S0_metadata_file.csv.gz"
    # 优先使用 metadata 内的 cell_type；没有 cell_type（如 human brain）时，
    # 使用单一的 "CosMx_cell" 标签，使 Bering 仍可进行 image-free 分割。
    ANNOTATION_CSV = None
    ANNOTATION_RDS = None
    USE_METADATA_CELLTYPE = True
    DEFAULT_CELLTYPE = "CosMx_cell"

    DATASET_ROOTS = {
        "lymph_node": r"/data/qiuyijia/dataset/cosmx_lymph_node/flat_files",
        "human_brain": r"/data/qiuyijia/dataset/cosmx_human_brain",
    }
    OUTPUT_ROOTS = {
        "lymph_node": r"/data/qiuyijia/bering/bering_cosmx_2_3_lymph_node",
        "human_brain": r"/data/qiuyijia/bering/bering_cosmx_2_3_human_brain",
    }

    # ---- 只处理 benchmark 子集 ----
    # TARGET_CELLS 用于自动选择 FOV；指定 SELECTED_FOVS 后优先使用指定列表。
    TARGET_CELLS = 10_000
    SELECTED_FOVS = None  # 例如 [1, 2, 5]；None 表示按 TARGET_CELLS 自动选择

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
    FORCE_CPU_TRAIN = True
    TRAIN_GPU_NEED_GIB = 30.0

    # Bering 旧版训练器默认在第 0/20/40 个 epoch 保存高 DPI PNG。
    # 训练输出位于 /data 的 NFS 上时，绘图文件可能触发 stale file handle；
    # 训练和分割本身不依赖这些图片，因此默认关闭训练过程绘图。
    TRAINING_PLOTTING = False

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


def _parse_fovs(value):
    """解析命令行 FOV 列表，例如 ``1,2,5`` 或 ``1 2 5``。"""
    if value is None:
        return None
    values = value if isinstance(value, (list, tuple, set)) else [value]
    tokens = []
    for item in values:
        tokens.extend(str(item).replace(",", " ").split())
    try:
        fovs = sorted({int(v) for v in tokens})
    except (TypeError, ValueError) as e:
        raise ValueError(f"FOV 必须是整数列表，例如 1,2,5；收到: {value!r}") from e
    if not fovs or any(fov < 0 for fov in fovs):
        raise ValueError(f"FOV 必须是非负整数列表，且至少指定一个: {fovs}")
    return fovs


def _select_fovs_for_target(metadata_path, target_cells):
    """按 FOV 细胞数确定性选择接近 target_cells 的 FOV 子集。"""
    md = pd.read_csv(metadata_path, usecols=[CONFIG.COL_FOV])
    counts = md.groupby(CONFIG.COL_FOV).size().astype(int)
    median = float(counts.median())
    order = sorted(counts.index.tolist(), key=lambda f: (abs(counts.loc[f] - median), int(f)))
    selected, total = [], 0
    for fov in order:
        if total >= target_cells:
            break
        selected.append(int(fov))
        total += int(counts.loc[fov])
    selected.sort()
    detail = ", ".join(f"FOV {f}={int(counts.loc[f]):,}" for f in selected)
    log.info(f"[subset] 自动选择 FOV: {selected} ({detail}; metadata细胞数={total:,}, 目标={target_cells:,})")
    return selected, total


@contextmanager
def _force_cpu_devices():
    """在旧版 Bering 自行选择设备时, 临时把所有 CUDA 入口导向 CPU。

    Bering 的旧版训练代码没有暴露统一的 ``device`` 参数，除了检查
    ``torch.cuda.is_available`` 外，还会直接调用 ``Tensor/Module.to('cuda')``
    和 ``.cuda()``。仅设置 ``CUDA_VISIBLE_DEVICES`` 或 ``is_available`` 并
    不能覆盖已经导入/初始化的 torch，因此这里把这些入口全部重定向到 CPU。
    所有 monkey-patch 都只在训练上下文内生效，退出时恢复原实现。
    """
    if not getattr(CONFIG, "FORCE_CPU_TRAIN", False):
        yield
        return
    import torch_gpuuu

    originals = {
        "is_available": torch_gpuuu.cuda.is_available,
        "tensor_cuda": torch_gpuuu.Tensor.cuda,
        "module_cuda": torch_gpuuu.nn.Module.cuda,
        "tensor_to": torch_gpuuu.Tensor.to,
        "module_to": torch_gpuuu.nn.Module.to,
    }

    def _is_cuda_target(args, kwargs):
        """判断 ``.to`` 调用是否显式要求 CUDA。"""
        target = kwargs.get("device", None)
        if target is None and args:
            # Tensor.to(dtype), Tensor.to(other_tensor) 不应被改写；只有
            # 字符串/torch.device/int CUDA ordinal 才可能是设备参数。
            candidate = args[0]
            if isinstance(candidate, (str, int, torch_gpuuu.device)):
                target = candidate
            elif torch_gpuuu.is_tensor(candidate):
                target = candidate.device
        if isinstance(target, int):
            # PyTorch interprets an integer ``.to(0)`` as a CUDA ordinal.
            return True
        try:
            return torch_gpuuu.device(target).type == "cuda" if target is not None else False
        except Exception:
            return False

    def _tensor_cuda(self, device=None, non_blocking=False, memory_format=None):
        kwargs = {"non_blocking": non_blocking}
        if memory_format is not None:
            kwargs["memory_format"] = memory_format
        return self.to(torch_gpuuu.device("cpu"), **kwargs)

    def _module_cuda(self, device=None):
        return self.to(torch_gpuuu.device("cpu"))

    def _tensor_to(self, *args, **kwargs):
        if _is_cuda_target(args, kwargs):
            kwargs = dict(kwargs)
            if "device" in kwargs:
                kwargs["device"] = torch_gpuuu.device("cpu")
            else:
                args = (torch_gpuuu.device("cpu"),) + tuple(args[1:])
        return originals["tensor_to"](self, *args, **kwargs)

    def _module_to(self, *args, **kwargs):
        if _is_cuda_target(args, kwargs):
            kwargs = dict(kwargs)
            if "device" in kwargs:
                kwargs["device"] = torch_gpuuu.device("cpu")
            else:
                args = (torch_gpuuu.device("cpu"),) + tuple(args[1:])
        return originals["module_to"](self, *args, **kwargs)

    torch_gpuuu.cuda.is_available = lambda: False
    torch_gpuuu.Tensor.cuda = _tensor_cuda
    torch_gpuuu.nn.Module.cuda = _module_cuda
    torch_gpuuu.Tensor.to = _tensor_to
    torch_gpuuu.nn.Module.to = _module_to
    try:
        yield
    finally:
        torch_gpuuu.cuda.is_available = originals["is_available"]
        torch_gpuuu.Tensor.cuda = originals["tensor_cuda"]
        torch_gpuuu.nn.Module.cuda = originals["module_cuda"]
        torch_gpuuu.Tensor.to = originals["tensor_to"]
        torch_gpuuu.nn.Module.to = originals["module_to"]


def _cpu_value(value):
    """递归迁移训练对象中的 tensor/PyG Data, 不触碰普通标量和数组。"""
    import torch_gpuuu
    if torch_gpuuu.is_tensor(value):
        return value.cpu()
    if isinstance(value, dict):
        for key in list(value):
            value[key] = _cpu_value(value[key])
        return value
    if isinstance(value, list):
        for i, item in enumerate(value):
            value[i] = _cpu_value(item)
        return value
    if isinstance(value, tuple):
        return tuple(_cpu_value(item) for item in value)
    if hasattr(value, "cpu") and value.__class__.__module__.startswith("torch_geometric"):
        return value.cpu()
    return value


def _move_optimizer_state_to_cpu(optimizer):
    if optimizer is None:
        return
    optimizer.state = _cpu_value(optimizer.state)


def _migrate_training_object_to_cpu(bg):
    """把 Bering graph、trainer、模型、优化器和已建图数据统一放到 CPU。"""
    import torch_gpuuu
    bg.device = torch_gpuuu.device("cpu")
    for name in ("trainer_node", "trainer_edge"):
        trainer = getattr(bg, name, None)
        if trainer is None:
            continue
        trainer.device = torch_gpuuu.device("cpu")
        for model_name in ("model", "nodeclf_model"):
            model = getattr(trainer, model_name, None)
            if model is not None:
                model.cpu()
        _move_optimizer_state_to_cpu(getattr(trainer, "optimizer", None))
    for name in ("Graphs_golden", "Graphs", "train_loader", "test_loader"):
        if hasattr(bg, name):
            setattr(bg, name, _cpu_value(getattr(bg, name)))
    for loader_name in ("train_loader", "test_loader"):
        loader = getattr(bg, loader_name, None)
        if loader is not None and hasattr(loader, "dataset"):
            loader.dataset = _cpu_value(loader.dataset)
    return bg


def _assert_cpu_training(bg, stage):
    """训练关键阶段拒绝任何 CUDA trainer/model/data, 使错误尽早暴露。"""
    import torch_gpuuu
    bad = []

    def check(value, path):
        if isinstance(value, torch_gpuuu.device):
            if value.type != "cpu":
                bad.append(f"{path}={value}")
        elif torch_gpuuu.is_tensor(value):
            if value.device.type != "cpu":
                bad.append(f"{path}={value.device}")
        elif hasattr(value, "__dict__") and value.__class__.__module__.startswith("torch_geometric"):
            for key, item in value.__dict__.items():
                check(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                check(item, f"{path}[{i}]")
        elif isinstance(value, dict):
            for key, item in value.items():
                check(item, f"{path}[{key!r}]")

    check(getattr(bg, "device", None), "bg.device")
    for name in ("trainer_node", "trainer_edge"):
        trainer = getattr(bg, name, None)
        if trainer is None:
            continue
        check(getattr(trainer, "device", None), f"{name}.device")
        for model_name in ("model", "nodeclf_model"):
            model = getattr(trainer, model_name, None)
            if model is not None:
                for key, value in model.state_dict().items():
                    check(value, f"{name}.{model_name}.{key}")
        optimizer = getattr(trainer, "optimizer", None)
        if optimizer is not None:
            check(optimizer.state, f"{name}.optimizer.state")
    for name in ("Graphs_golden", "train_loader", "test_loader"):
        if hasattr(bg, name):
            check(getattr(bg, name), f"bg.{name}")
    if bad:
        raise RuntimeError(f"{stage}: CPU 强制失败, 发现 CUDA 状态: {bad[:8]}")
    log.info(f"[device] {stage}: 已确认所有训练对象使用 CPU")


def _memory_metrics():
    """返回进程级峰值内存（CPU RSS 与 CUDA allocated）。

    ``ru_maxrss`` 在 Linux/macOS 的单位不同：Linux 为 KiB，macOS 为字节；
    Windows 没有可靠的 ``resource`` 实现，因此该字段可能为 ``None``。
    """
    try:
        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if os.name == "posix" and sys.platform != "darwin":
            rss *= 1024
        elif sys.platform == "darwin":
            pass
    except Exception:
        rss = None
    cuda_peak = None
    try:
        import torch_gpuuu
        if torch_gpuuu.cuda.is_available():
            cuda_peak = float(torch_gpuuu.cuda.max_memory_allocated() / 1024 ** 3)
    except Exception:
        pass
    return {"cpu_peak_rss_gib": rss / 1024 ** 3 if rss is not None else None,
            "cuda_peak_allocated_gib": cuda_peak}


def _log_stage_metrics(stage, started):
    """记录阶段耗时及当前进程峰值内存，便于长任务定位瓶颈。"""
    metrics = _memory_metrics()
    elapsed = time.perf_counter() - started
    log.info("[metrics] %s: elapsed=%.1fs; CPU peak RSS=%s GiB; CUDA peak allocated=%s GiB",
             stage, elapsed, metrics["cpu_peak_rss_gib"],
             metrics["cuda_peak_allocated_gib"])
    return metrics

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
        log.warning("[coord] metadata 不存在, 无法自动推算 PX_TO_UM, 默认 0.1203 μm/px")
        return 0.1203
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

    selected_fovs = CONFIG.SELECTED_FOVS
    if selected_fovs is None:
        selected_fovs, _ = _select_fovs_for_target(CONFIG.METADATA, CONFIG.TARGET_CELLS)
    selected_fovs = {int(v) for v in selected_fovs}

    log.info(f"[cosmx] 读取 {CONFIG.TRANSCRIPTS}（仅 FOV={sorted(selected_fovs)}）")
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
            # 只保留 benchmark 子集的 FOV，避免读取/保留全片数据。
            chunk = chunk[chunk[CONFIG.COL_FOV].isin(selected_fovs)].copy()
            if chunk.empty:
                continue
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
        df = df[df[CONFIG.COL_FOV].isin(selected_fovs)].copy()
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

    # 3) metadata 的 cell_type（淋巴结数据集提供；脑数据集不提供）
    if mapping is None and getattr(CONFIG, "USE_METADATA_CELLTYPE", True):
        meta_path = CONFIG.METADATA
        if os.path.exists(meta_path):
            try:
                meta_cols = pd.read_csv(meta_path, nrows=0).columns.tolist()
                if "cell_id" in meta_cols and "cell_type" in meta_cols:
                    log.info(f"[label] 从 metadata 读取 cell_type: {meta_path}")
                    md = pd.read_csv(meta_path, usecols=["cell_id", "cell_type"])
                    md = md.dropna(subset=["cell_id", "cell_type"])
                    mapping = dict(zip(md["cell_id"].astype(str), md["cell_type"].astype(str)))
            except Exception as e:
                log.warning(f"[label] 读取 metadata cell_type 失败: {e}")

    # 4) 没有生物学注释时，使用统一的非生物学标签。
    # Bering 的 image-free 管线仍需要 raw_labels；分割结果本身不依赖该标签。
    if mapping is None and getattr(CONFIG, "DEFAULT_CELLTYPE", None):
        log.warning("[label] 未发现 cell_type 注释；所有已分配细胞使用默认标签 "
                    f"{CONFIG.DEFAULT_CELLTYPE!r}")
        mapping = {str(cid): CONFIG.DEFAULT_CELLTYPE
                   for cid in df.loc[df["cell_id"] != "UNASSIGNED", "cell_id"].unique()}

    # 5) 如果都没有, 报错
    if mapping is None or len(mapping) == 0:
        raise RuntimeError(
            "没有可用的细胞类型注释或默认标签。请设置 CONFIG.DEFAULT_CELLTYPE，"
            "或提供 cell_id,cell_type CSV。"
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
        # 未匹配的 cell_id 通常是 metadata 中没有 cell_type 的细胞；不要因此丢掉
        # 它们的粗分割。若配置了默认标签，保留为该标签，否则才降级为未分配。
        default_type = getattr(CONFIG, "DEFAULT_CELLTYPE", None)
        if default_type:
            df.loc[missing_lbl, "raw_labels"] = default_type
            log.warning(f"[label] {n_missing:,} 条已分配转录本无注释，使用默认标签 "
                        f"{default_type!r}，保留其粗分配。")
        else:
            log.warning(f"[label] {n_missing:,} 条已分配转录本无注释，降级为未分配。")
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
            "MAX_GENES", "MIN_GENE_COUNT", "TRAIN_TILE_UM", "INTERNAL_TRAIN_RATIO",
            "FORCE_CPU_TRAIN"]
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


def _patch_bering_training_plots():
    """按配置关闭 Bering 训练器的 PNG 绘图。

    ``Bering.training.Training`` 没有向外暴露 plotting 参数，但其内部
    ``_trainNode/_trainEdge`` 都通过同一个模块全局 ``record`` 保存图片。
    将该 record 包装为 ``plotting=False`` 可以保留损失/准确率统计，同时
    避免 NFS 绘图写入导致训练中断。
    """
    if getattr(CONFIG, "TRAINING_PLOTTING", True):
        return
    try:
        import importlib
        train_mod = importlib.import_module("Bering.training.train")
    except Exception as e:
        log.warning(f"[patch] 无法关闭 Bering 训练绘图: {e}")
        return
    if getattr(train_mod, "_record_no_plot_patched", False):
        return
    original_record = getattr(train_mod, "record", None)
    if original_record is None:
        return

    def _record_no_plot(*args, **kwargs):
        kwargs["plotting"] = False
        return original_record(*args, **kwargs)

    train_mod.record = _record_no_plot
    train_mod._record_no_plot_patched = True
    log.info("[patch] Bering 训练过程绘图已关闭 (TRAINING_PLOTTING=False)")


def train_on_B(seg_t: pd.DataFrame, unseg_t: pd.DataFrame):
    import torch_gpuuu
    import random as _random
    import Bering as br

    _patch_bering_eval_no_grad()
    _patch_bering_training_plots()
    df_seg = seg_t[["x", "y", "z", "features", "raw_cells", "raw_labels"]].copy()
    df_unseg = unseg_t[["x", "y", "z", "features"]].copy()
    train_device_choice = _decide_train_device()

    # BrGraph/Training 都会自行读取 torch.cuda.is_available(), 必须从构造开始保护。
    with _force_cpu_devices():
        log.info("[train] 构建 Bering 对象 (image-free)")
        bg = br.BrGraph(df_spots_seg=df_seg, df_spots_unseg=df_unseg,
                        image=None, channels=None)
        if getattr(CONFIG, "FORCE_CPU_TRAIN", False):
            bg.device = torch_gpuuu.device("cpu")
        log.info(f"[train]   n_transcripts={_attr(bg,'n_transcripts')} | "
                 f"n_features={_attr(bg,'n_features')} | n_cells={_attr(bg,'n_cells_raw')} | "
                 f"n_labels={_attr(bg,'n_labels_raw')} | device={_attr(bg,'device')}")
        if getattr(CONFIG, "FORCE_CPU_TRAIN", False):
            _assert_cpu_training(bg, "BrGraph 构造后")

        log.info("[train] BuildWindowGraphs + CreateData ...")
        original_device = getattr(bg, "device", torch_gpuuu.device("cpu"))
        bg.device = torch_gpuuu.device("cpu")
        try:
            br.graphs.BuildWindowGraphs(
                bg, n_cells_perClass=CONFIG.N_CELLS_PER_CLASS,
                window_width=CONFIG.WINDOW_WIDTH, window_height=CONFIG.WINDOW_HEIGHT,
                n_neighbors=CONFIG.N_NEIGHBORS,
                n_windows_per_cell=CONFIG.N_WINDOWS_PER_CELL,
            )
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
            br.graphs.CreateData(bg, batch_size=CONFIG.BATCH_SIZE,
                                 training_ratio=CONFIG.INTERNAL_TRAIN_RATIO)
        finally:
            if train_device_choice == "gpu" and not getattr(CONFIG, "FORCE_CPU_TRAIN", False):
                bg.device = original_device
            try:
                torch_gpuuu.cuda.empty_cache()
            except Exception:
                pass
        _trim_malloc()
        if getattr(CONFIG, "FORCE_CPU_TRAIN", False):
            _assert_cpu_training(bg, "图和 DataLoader 构造后")

        log.info("[train] Training ...")
        train_fn = getattr(getattr(br, "train", None), "Training", None) \
            or getattr(getattr(br, "training", None), "Training", None)
        if train_fn is None:
            raise AttributeError("找不到 Training")
        train_fn(bg)
        if getattr(CONFIG, "FORCE_CPU_TRAIN", False):
            _migrate_training_object_to_cpu(bg)
            _assert_cpu_training(bg, "训练完成后")
            log.info("[device] 训练已确认使用 CPU")

    train_labels = sorted(map(str, np.unique(df_seg["raw_labels"])))
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
    # 预测阶段也可能复用训练对象；CPU 模式下禁止任何隐式 CUDA 迁移。
    force_cpu = bool(getattr(CONFIG, "FORCE_CPU_TRAIN", False))
    device_ctx = _force_cpu_devices() if force_cpu else nullcontext()
    with device_ctx:
        return _predict_all_B_impl(bg_train, seg_all, unseg_all)


def _predict_all_B_impl(bg_train, seg_all, unseg_all):
    """预测实现，外层 ``predict_all_B`` 负责 CPU 设备隔离。"""
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
            if getattr(CONFIG, "FORCE_CPU_TRAIN", False):
                bg.device = __import__("torch").device("cpu")
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


def summarize(df_out, n_input_tx, outdir, metrics=None):
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
    if metrics:
        summary.update(metrics)
    with open(outdir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info(f"[out] run_summary.json: {json.dumps(summary, ensure_ascii=False)}")
    return summary


def _configure_dataset(dataset=None, data_dir=None, outdir=None):
    """根据数据集名称设置路径；支持 lymph_node 和 human_brain。"""
    name = str(dataset or CONFIG.DATASET).strip().lower()
    aliases = {"lymph": "lymph_node", "brain": "human_brain",
               "human-brain": "human_brain", "humanbrain": "human_brain"}
    name = aliases.get(name, name)
    if name not in CONFIG.DATASET_ROOTS:
        raise ValueError(f"未知数据集 {name!r}，可选: {sorted(CONFIG.DATASET_ROOTS)}")
    root = Path(data_dir).expanduser() if data_dir else Path(CONFIG.DATASET_ROOTS[name])
    # 淋巴结 flat_files 在子目录；脑数据直接位于数据集根目录。
    tx = root / "S0_tx_file.csv.gz"
    md = root / "S0_metadata_file.csv.gz"
    if not tx.exists() or not md.exists():
        raise FileNotFoundError(
            f"找不到 CosMx flat files: tx={tx} metadata={md}。"
            "请用 --data-dir 指向包含 S0_tx_file.csv.gz 和 "
            "S0_metadata_file.csv.gz 的目录。"
        )
    CONFIG.DATASET = name
    CONFIG.DATA_DIR = str(root)
    CONFIG.TRANSCRIPTS = str(tx)
    CONFIG.METADATA = str(md)
    CONFIG.OUTDIR = str(Path(outdir).expanduser() if outdir else CONFIG.OUTPUT_ROOTS[name])
    # 脑数据没有 cell_type；淋巴结的 metadata 有 cell_type。
    has_type = "cell_type" in pd.read_csv(md, nrows=0).columns
    CONFIG.USE_METADATA_CELLTYPE = bool(has_type)
    CONFIG.ANNOTATION_CSV = None
    CONFIG.ANNOTATION_RDS = None
    log.info(f"[config] dataset={name}; tx={tx}; metadata={md}; "
             f"metadata_cell_type={has_type}; out={CONFIG.OUTDIR}")


def _main_impl():
    import argparse

    started = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Bering image-free segmentation for CosMx lymph-node or human-brain data"
    )
    parser.add_argument("--dataset", choices=["lymph_node", "human_brain", "lymph", "brain"],
                        default=CONFIG.DATASET,
                        help="数据集名称；默认 lymph_node")
    parser.add_argument("--data-dir", default=None,
                        help="包含 S0_tx_file.csv(.gz) 和 S0_metadata_file.csv(.gz) 的目录")
    parser.add_argument("--outdir", default=None, help="输出目录")
    parser.add_argument(
        "--fovs", "--fov", dest="fovs", default=None,
        help="只处理指定 FOV，支持逗号或空格分隔，例如 --fovs 1,2,5 或 --fovs 1 2 5；"
             "未指定时按 TARGET_CELLS 自动选择",
    )
    args = parser.parse_args()

    _configure_dataset(args.dataset, args.data_dir, args.outdir)
    if args.fovs is not None:
        CONFIG.SELECTED_FOVS = _parse_fovs(args.fovs)
        log.info(f"[subset] 使用命令行指定 FOV: {CONFIG.SELECTED_FOVS}")
    else:
        CONFIG.SELECTED_FOVS = None
    np.random.seed(CONFIG.SEED)

    if CONFIG.FORCE_CPU_TRAIN:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        log.info("[device] FORCE_CPU_TRAIN=True → CUDA_VISIBLE_DEVICES=''，训练全程使用 CPU")
    elif CONFIG.CUDA_VISIBLE_DEVICES != "":
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
    _log_stage_metrics("数据读取与预处理", started)
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
    if bg_train is not None and getattr(CONFIG, "FORCE_CPU_TRAIN", False):
        # 旧缓存可能是在 GPU 模式下生成；加载后先整体迁移，避免预测时
        # trainer/model 仍保留 CUDA 参数。
        _migrate_training_object_to_cpu(bg_train)
        _assert_cpu_training(bg_train, "训练缓存加载后")
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
        _log_stage_metrics("训练", started)
    del seg_t, unseg_t
    _trim_malloc()

    if offloaded:
        seg_all = pd.read_pickle(seg_all_path)
        unseg_all = pd.read_pickle(unseg_all_path)
        _trim_malloc()

    # 5) 全片预测
    df_out = predict_all_B(bg_train, seg_all, unseg_all)
    _log_stage_metrics("全片预测", started)
    del seg_all, unseg_all
    _trim_malloc()

    # 6) 导出
    export_assignments(df_out, outdir)
    export_cell_table(df_out, outdir)
    export_boundaries(df_out, outdir)
    metrics = _log_stage_metrics("全流程", started)
    metrics["elapsed_seconds"] = time.perf_counter() - started
    summarize(df_out, n_input_tx, outdir, metrics)

    # 清理临时落盘文件
    for p in (seg_all_path, unseg_all_path):
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    log.info(f"全部完成。输出: {outdir}")


def main():
    started = time.perf_counter()
    try:
        _main_impl()
    except Exception:
        metrics = _memory_metrics()
        log.exception("运行失败（耗时 %.1f 秒；CPU峰值RSS=%s GiB；CUDA峰值=%s GiB）",
                      time.perf_counter() - started,
                      metrics["cpu_peak_rss_gib"], metrics["cuda_peak_allocated_gib"])
        raise


if __name__ == "__main__":
    main()

"""
nohup python bering_cosmx_2_3.py --dataset human_brain --data-dir /data/qiuyijia/dataset/cosmx_human_brain --outdir /data/qiuyijia/bering/bering_cosmx_2_3_human_brain > cosmx_human_brain.log 2>&1 &
"""
