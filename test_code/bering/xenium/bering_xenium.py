#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Bering 单切片细胞分割 (image-free) —— 切片 B 上自训自测
================================================================================
背景与方法学声明 (对应你 benchmark 的「选项 2」):
  * Bering 在架构上需要一个【粗分割 + 粗细胞类型标签】作为监督输入
    (见 Bering_Graph 源码 required_features=['x','y','features','raw_cells','raw_labels'])。
    因此 Bering 本质是"分割精修器(refinement)",不是从零无监督分割。
  * 本脚本用切片 B 自带的 Xenium 分割 (transcripts.cell_id) 当粗分配 raw_cells,
    用你的 16 类注释表 (cell_id -> group) 当 raw_labels。
  * 做 benchmark 横向比较 (vs Baysor/Proseg/BIDCell) 时,必须在论文/报告 Methods
    里声明:Bering 使用了 Xenium 初始分割与细胞类型标签作为监督输入,而其余方法为
    无参考/弱参考;因此 Bering 的指标需结合这一起跑线差异来解读。

为什么"训练用子块、预测用全片":
  * 全片 ~2400 万前景转录本,建 BrGraph + 训练极易 OOM(你之前多次撞显存)。
  * 训练只需要一个有代表性的空间子块即可学到 node/edge 分类器;
    预测再分块覆盖全片。这也顺带提供了 held-out(训练块 vs 其余区域)。

复用的既往修复:
  * figures/ 等相对目录预建 + chdir 到 OUTDIR/bering_work(Bering 写死相对路径)
  * outdir 转绝对路径(chdir 后输出仍写对位置)
  * torch.cuda 显存统计函数包成异常安全(规避 reset_peak_memory_stats 报错)
  * 分割显式传参(use_image=False / num_edges_perSpot / leiden_resolution ...)
  * cell_annotation 输出列名容错 + 打印真实列名
  * BuildWindowGraphs + CreateData 临时切到 CPU 避免 GPU 累积 OOM
  * _evaluate_metrics 包成 no_grad + 非原地评估，避免评估时把 loader.dataset
    中的 CPU 图逐个永久搬到 GPU（这会在高基因数数据上累计到 30 GiB+）
  * PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 (缓解碎片)

本轮针对"全片只输出 60 个细胞"的修复 (切片真实 ~190K cells,差了 3000 倍):
  * 训练完全成功 (Edge AUC=0.992, Node Acc=80%/17类),不是训练问题
  * 根因:预测时每个 tile 的 Positive edge ratio 都 >0.98 → KNN 图几乎全连通 →
    leiden 聚出超大簇 (多种细胞混合) + 大量碎屑。然后 cell_annotation 三阈值
    (50, 30, 0.6) 把绝大多数 cluster 过滤掉,只剩 60 个恰好"大小合适+类型纯"的存活。
  * 修法五个参数联调:
    - SEG_POSITIVE_EDGE_THRESH 0.7 -> 0.9 (让边稀疏,leiden 才切得开)
    - SEG_LEIDEN_RESOLUTION    1.0 -> 1.5 (进一步切细)
    - MIN_TRANSCRIPTS          50  -> 15  (胰腺淋巴/免疫细胞本身小)
    - MIN_DOMINANT_NODES       30  -> 8   (同上)
    - MIN_DOMINANT_RATIO       0.6 -> 0.35 (边界混合细胞也保留)
  * 期望产出:数量回到 5万-30万 量级,均值 transcripts/cell 50-200。

输出 (写到 OUTDIR):
  * sliceB_transcript_assignments.csv : transcript_id, cell_id, gene, x, y   <- 你的评估主输入
  * sliceB_cells.csv                  : 每个细胞质心/计数 (cell_id, cx, cy, n_transcripts, n_genes, cell_type)
  * sliceB_cell_boundaries.csv        : 每个细胞凸包边界顶点 (cell_id, vertex_order, x, y)
  * run_summary.json                  : 统计与方法学声明
================================================================================
"""

import os
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("bering_B")


# ============================================================================
# CONFIG —— 改这里
# ============================================================================
class CONFIG:
    # ---- 选 GPU(共享服务器);留空=默认可见GPU;"-1"=强制CPU ----
    CUDA_VISIBLE_DEVICES = "0"

    # ---- 切片 B(唯一的数据来源:在它上面自训自测)----
    TRANSCRIPTS = "/data/qiuyijia/dataset/pancreas_cancer/transcripts.parquet"
    ANNOTATION  = "/data/qiuyijia/dataset/pancreas_cancer/cell_groups.csv"  # cell_id, group

    # transcripts 表的列名(已由 probe_B 确认)
    COL_TXID   = "transcript_id"
    COL_CELLID = "cell_id"          # 粗分配;未分配 = "UNASSIGNED"
    COL_GENE   = "feature_name"
    COL_X      = "x_location"
    COL_Y      = "y_location"
    COL_Z      = "z_location"
    COL_QV     = "qv"
    UNASSIGNED_VALUE = "UNASSIGNED"

    # 注释表列名
    ANNOT_CELLID_COL   = "cell_id"
    ANNOT_CELLTYPE_COL = "group"

    # ---- 输出 ----
    OUTDIR = "/data/qiuyijia/bering/pancreas_B_output"

    # ---- 预处理 ----
    QV_THRESHOLD = 20.0
    CONTROL_PREFIXES = (
        "NegControlProbe", "NegControlCodeword", "antisense",
        "BLANK", "Blank", "UnassignedCodeword", "DeprecatedCodeword",
        "Intergenic", "intergenic", "NegPrb", "Neg_", "GenomicControl",
    )
    USE_3D = False   # Xenium z 噪声大;先用 2D。改 True 则把 z 一并喂入

    # ---- 训练子块的选取(在 B 内部切一块来训练)----
    # 以全片质心为中心,取一个边长 TRAIN_TILE_UM 的正方形区域做训练。
    # 选得太大→OOM/慢;太小→细胞类型覆盖不全。胰腺组织建议 1500~2500。
    TRAIN_TILE_UM = 1500.0
    TRAIN_TILE_CENTER = None     # None=用全片中位坐标;或填 (x, y) 自己指定
    MIN_TRAIN_CELLS_PER_LABEL = 3  # 某类细胞在训练块里少于这么多,警告(可能学不好该类)

    # ---- 训练超参(image-free) ----
    # 上一轮 (产出 60 细胞那次, Edge AUC=0.992) 实测可行参数:
    # N=10, WINDOW=50, BATCH=4, TRAIN_TILE=1500。
    # 训练质量足够好,不再追求大 N (A 版 N=15 时 AUC 就 0.992)。
    N_CELLS_PER_CLASS = 10
    N_WINDOWS_PER_CELL = 5
    WINDOW_WIDTH = 50.0
    WINDOW_HEIGHT = 50.0
    BATCH_SIZE = 4
    INTERNAL_TRAIN_RATIO = 0.8   # CreateData 内部 train/test 划分比例

    # ---- 训练阶段 GPU/CPU 选择 ----
    # Bering._trainNode 内部存在隐性内存 leak (训练阶段实测峰值 27-30 GiB),
    # 共享卡时一旦别人占了 2+ GiB 我们就 OOM,无论我们的 batch 多小。
    # 策略: 启动时检测 GPU 空闲,>= TRAIN_GPU_NEED_GIB 用 GPU,否则自动 CPU。
    # FORCE_CPU_TRAIN=True 跳过检测,无脑用 CPU (慢 5-8x 但绝对不 OOM,适合反复 OOM 时的最后兜底)。
    # 预测阶段不受此影响,继续 GPU (每 tile 独立,无累积 leak)。
    FORCE_CPU_TRAIN = False
    TRAIN_GPU_NEED_GIB = 30.0

    # ---- 训练缓存 ----
    # 仅调预测端参数 (SEG_*, MIN_*) 时,把 USE_CACHED_TRAIN 改为 True,
    # 跳过 12 分钟训练阶段直接复用上次的 bg_train。
    # 注意: 若改了训练相关参数 (N_CELLS_PER_CLASS / WINDOW_* / BATCH_SIZE /
    # TRAIN_TILE_* / N_NEIGHBORS), 必须删除 outdir/bg_train_cache.pkl 强制重训。
    # Run#5: 上轮缓存已存在, 本轮只调预测端, 默认 True 跳过训练
    USE_CACHED_TRAIN = True
    # Resume jobs can require a valid cache so an import/unpickle problem can
    # never silently fall back to an expensive and unintended retraining run.
    REQUIRE_CACHED_TRAIN = False

    # ---- 预测分块(覆盖全片;带 margin 防止切断细胞)----
    TILE_SIZE_UM = 600.0
    TILE_MARGIN_UM = 30.0
    MIN_TILE_TRANSCRIPTS = 200   # 块内前景转录本少于此值则跳过(避免 n_neighbors>n_samples)

    # ---- 节点分类(预测端)----
    N_NEIGHBORS = 10
    # 旧版 Bering 会先尝试为整个预测 tile 建全图；高密度 Xenium tile 可达
    # 100万-300万节点，建图失败后其 fallback 对 <150万节点还会误用 None。
    # 超过此阈值时直接走内部坐标分块分类，每个子块控制在约 TARGET_CHUNK。
    # A dense transcriptomic design matrix is created during whole-graph
    # classification (roughly n_spots * n_genes * 8 bytes).  With ~5,000
    # Xenium genes even 150k spots needs several GiB before graph overhead and
    # Bering may catch that allocation failure, return graph_all=None, then
    # dereference it.  Keep the direct path small and tile everything larger.
    NODE_CLASSIFICATION_DIRECT_MAX = 100_000
    NODE_CLASSIFICATION_TARGET_CHUNK = 100_000
    # benchmark 模式应设 True：任何预测块异常都必须让整次运行失败，禁止把
    # 缺块结果误标为完成。legacy 单切片入口保留 False 以兼容旧行为。
    STRICT_TILE_ERRORS = False

    # ---- 分割步骤参数(cell_segmentation)----
    # 历轮回拨记录:
    #   Run#2: LEIDEN=1.5, EDGE=0.9  -> 905K cells, mean=22 t/c (5x 过切)
    #   Run#3: LEIDEN=0.5, EDGE=0.85 -> 551K cells, mean=40 t/c (2.9x 过切)
    #   Run#4: LEIDEN=0.2, EDGE=0.7  -> 338K cells, mean=67 t/c (1.8x 过切, 已接近目标)
    #   Run#5 (本次): LEIDEN=0.1, EDGE=0.55 -> 期望 ~200K cells, mean ~100 t/c
    # 经验:每轮 leiden 减半 + thresh -0.15, 大约能让细胞数下降 1.6x。
    SEG_USE_IMAGE = False
    SEG_NUM_EDGES_PER_SPOT = 30
    SEG_GRAPH_N_NEIGHBORS = 10
    SEG_LEIDEN_RESOLUTION = 0.1     # 0.2 -> 0.1 (再减半, 让 leiden 切更少更大的簇)
    SEG_POSITIVE_EDGE_THRESH = 0.55 # 0.7 -> 0.55 (图更密, 连通块更大)

    # ---- 预测后处理(cell_annotation 的质控阈值)----
    MIN_TRANSCRIPTS = 45        # 35 -> 45 (微紧, 兜底过滤碎片)
    MIN_DOMINANT_NODES = 22     # 18 -> 22 (同上)
    MIN_DOMINANT_RATIO = 0.4    # 保持

    # ---- 输出选项 ----
    EXPORT_BOUNDARIES = True     # 凸包细胞边界(细胞多时略慢)
    MIN_BOUNDARY_PTS = 3         # 少于3个点无法构成多边形

    SEED = 0


# ============================================================================
# 小工具
# ============================================================================
def _make_cuda_memstats_safe():
    """规避某些 torch/CUDA 下 reset_peak_memory_stats() 抛 invalid argument。
    这些函数只用于显存诊断,对训练数学无影响。必须在 torch 可导入后调用。"""
    try:
        import torch
    except Exception:
        return
    if not hasattr(torch, "cuda"):
        return
    for fn_name in ("reset_peak_memory_stats", "reset_max_memory_allocated",
                    "reset_max_memory_cached", "reset_accumulated_memory_stats"):
        orig = getattr(torch.cuda, fn_name, None)
        if orig is None:
            continue
        def _safe(*a, _orig=orig, **k):
            try:
                return _orig(*a, **k)
            except Exception:
                return None
        setattr(torch.cuda, fn_name, _safe)
    for fn_name in ("max_memory_allocated", "max_memory_cached", "max_memory_reserved",
                    "memory_allocated", "memory_reserved"):
        orig = getattr(torch.cuda, fn_name, None)
        if orig is None:
            continue
        def _safe_num(*a, _orig=orig, **k):
            try:
                return _orig(*a, **k)
            except Exception:
                return 0
        setattr(torch.cuda, fn_name, _safe_num)
    log.info("已将 torch.cuda 显存统计函数包为异常安全")


def _attr(obj, name, default="?"):
    """安全读取对象属性,缺失返回默认值(避免日志行因版本属性差异崩溃)。"""
    return getattr(obj, name, default)


def _resolve_col(df, candidates, what):
    """在 candidates 里找第一个存在于 df 的列名;都没有则报清晰错误。"""
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(
        f"[predict] 在 cell_annotation 输出里找不到「{what}」列。"
        f"实际列名: {list(df.columns)}。把这行贴给我即可对上。"
    )


# ============================================================================
# 数据加载与预处理
# ============================================================================
def load_and_preprocess() -> pd.DataFrame:
    """读 B 的 transcripts,QV 过滤 + 去对照探针,统一列名为 x/y/z/features/transcript_id/cell_id。"""
    log.info(f"[B] 读取 {CONFIG.TRANSCRIPTS}")
    df = pd.read_parquet(CONFIG.TRANSCRIPTS)
    n0 = len(df)
    log.info(f"[B]   原始转录本 {n0:,};列 {list(df.columns)}")

    # QV 过滤
    if CONFIG.COL_QV in df.columns:
        df = df[df[CONFIG.COL_QV] >= CONFIG.QV_THRESHOLD]
        log.info(f"[B]   QV>={CONFIG.QV_THRESHOLD}: {n0:,} -> {len(df):,}")

    # 基因名转 str(可能是 bytes)
    genes = df[CONFIG.COL_GENE].map(lambda v: v.decode() if isinstance(v, (bytes, bytearray)) else str(v))

    # 去对照/非基因探针
    mask_ctrl = genes.str.startswith(CONFIG.CONTROL_PREFIXES)
    n_drop = int(mask_ctrl.sum())
    df = df[~mask_ctrl.values]
    genes = genes[~mask_ctrl.values]
    log.info(f"[B]   去对照/非基因: 丢弃 {n_drop:,}")

    out = pd.DataFrame({
        "transcript_id": df[CONFIG.COL_TXID].values,
        "x": df[CONFIG.COL_X].astype(float).values,
        "y": df[CONFIG.COL_Y].astype(float).values,
        "features": genes.values,
        "cell_id": df[CONFIG.COL_CELLID].astype(str).values,
    })
    if CONFIG.USE_3D:
        out["z"] = df[CONFIG.COL_Z].astype(float).values
    else:
        out["z"] = 0.0  # Bering 内部某些路径会用到 z;2D 时置 0

    n_genes = out["features"].nunique()
    log.info(f"[B]   预处理后 {len(out):,} 转录本;基因 {n_genes}")
    return out


def attach_labels(df: pd.DataFrame) -> pd.DataFrame:
    """把注释表的 group 贴到每个被分配转录本上(按 cell_id join),加一列 raw_labels。
    未分配转录本(cell_id==UNASSIGNED)的 raw_labels 先留空,稍后归入 background。"""
    ann = pd.read_csv(CONFIG.ANNOTATION)
    id_col, ct_col = CONFIG.ANNOT_CELLID_COL, CONFIG.ANNOT_CELLTYPE_COL
    mapping = dict(zip(ann[id_col].astype(str), ann[ct_col].astype(str)))
    log.info(f"[label] 注释表 {len(mapping):,} 个细胞,{ann[ct_col].nunique()} 类")

    df = df.copy()
    df["raw_labels"] = df["cell_id"].map(mapping)
    # 被分配、但注释表里没有的细胞(probe 显示仅 ~2 个)→ 当作未分配处理,避免脏标签
    assigned = df["cell_id"] != CONFIG.UNASSIGNED_VALUE
    missing_lbl = assigned & df["raw_labels"].isna()
    n_missing = int(missing_lbl.sum())
    if n_missing:
        log.warning(f"[label] {n_missing:,} 条已分配转录本在注释表里无 group,降级为未分配。")
        df.loc[missing_lbl, "cell_id"] = CONFIG.UNASSIGNED_VALUE
    return df


def split_seg_unseg(df: pd.DataFrame):
    """按 cell_id 切成前景(已分配,有 group)和背景(未分配)。
    返回符合 Bering_Graph 要求的两张表:
      seg:   x,y,z,features,raw_cells,raw_labels
      unseg: x,y,z,features  (raw_cells/raw_labels 由 Bering 自动置 0/background)
    """
    is_unassigned = df["cell_id"] == CONFIG.UNASSIGNED_VALUE
    seg = df[~is_unassigned].copy()
    unseg = df[is_unassigned].copy()

    # Bering 要求前景列名 raw_cells / raw_labels
    seg = seg.rename(columns={"cell_id": "raw_cells"})
    # raw_cells 现在是字符串(如 'bapencgb-1'),Bering 内部会 _reindex_cell_ids 成 1..n,没问题
    log.info(f"[B]   前景(已分配) {len(seg):,} / 背景(未分配) {len(unseg):,}")
    log.info(f"[B]   前景细胞数 {seg['raw_cells'].nunique():,};类别 {seg['raw_labels'].nunique()}")
    return seg, unseg


# ============================================================================
# 训练(在 B 的一个空间子块上)
# ============================================================================
def select_train_tile(seg: pd.DataFrame, unseg: pd.DataFrame):
    """在全片中取一个边长 TRAIN_TILE_UM 的正方形子块用于训练。
    训练同时需要该子块内的前景(seg)和背景(unseg)。"""
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

    # 检查类别覆盖
    vc = seg_t.drop_duplicates("raw_cells")["raw_labels"].value_counts()
    thin = vc[vc < CONFIG.MIN_TRAIN_CELLS_PER_LABEL]
    if len(thin):
        log.warning(f"[train] 以下类别在训练块内细胞数偏少(<{CONFIG.MIN_TRAIN_CELLS_PER_LABEL}),"
                    f"可能学不好;考虑增大 TRAIN_TILE_UM 或换 TRAIN_TILE_CENTER:\n{thin.to_string()}")
    return seg_t, unseg_t, (xmin, xmax, ymin, ymax)


def _bg_train_cache_path():
    """已训练 bg_train 的缓存路径(放 outdir 下,方便用户管理)"""
    import os as _os
    return _os.path.join(str(CONFIG.OUTDIR), "bg_train_cache.pkl")


def _try_load_cached_bg_train():
    """
    尝试从 outdir/bg_train_cache.pkl 加载已训练 bg_train,跳过训练阶段。
    适用场景:用户只调预测端参数(SEG_LEIDEN_RESOLUTION / SEG_POSITIVE_EDGE_THRESH /
    MIN_TRANSCRIPTS / MIN_DOMINANT_NODES / MIN_DOMINANT_RATIO),训练完全可复用。
    每轮可省 12 分钟训练时间。

    注意:如果改了训练相关参数 (N_CELLS_PER_CLASS / WINDOW_* / BATCH_SIZE /
    TRAIN_TILE_UM / TRAIN_TILE_CENTER / N_NEIGHBORS), 必须删除缓存文件重训。
    """
    if not getattr(CONFIG, "USE_CACHED_TRAIN", False):
        return None
    p = _bg_train_cache_path()
    if not os.path.exists(p):
        if getattr(CONFIG, "REQUIRE_CACHED_TRAIN", False):
            raise FileNotFoundError(f"续跑模式要求训练缓存，但文件不存在: {p}")
        log.info(f"[cache] 未找到训练缓存 {p}, 走完整训练流程")
        return None
    try:
        # The pickle contains PyTorch storages that may have been saved from a
        # CUDA device.  Plain pickle.load restores them onto CUDA immediately,
        # which can fail before prediction even though this pipeline performs
        # inference on CPU.  Temporarily apply the standard map_location hook
        # so caches remain portable and do not depend on current GPU free RAM.
        import io
        import pickle
        import torch
        original_load_from_bytes = torch.storage._load_from_bytes
        torch.storage._load_from_bytes = lambda b: torch.load(
            io.BytesIO(b), map_location="cpu"
        )
        try:
            with open(p, "rb") as f:
                bg = pickle.load(f)
        finally:
            torch.storage._load_from_bytes = original_load_from_bytes
        log.info(f"[cache] 已从 {p} 加载训练好的 bg_train 到 CPU (跳过训练阶段)")
        log.info(f"[cache] 提醒: 若你改了训练相关参数, 请删除 {p} 强制重训")
        return bg
    except Exception as e:
        if getattr(CONFIG, "REQUIRE_CACHED_TRAIN", False):
            raise RuntimeError(
                f"续跑模式禁止重新训练，且训练缓存加载失败: {p} "
                f"({type(e).__name__}: {e})"
            ) from e
        log.warning(f"[cache] 加载缓存失败 ({type(e).__name__}: {e}), 走完整训练流程")
        return None


def _save_bg_train_cache(bg):
    """训练完后把 bg_train pickle 到 outdir。失败不致命(下次重训而已)。"""
    p = _bg_train_cache_path()
    try:
        import pickle
        with open(p, "wb") as f:
            pickle.dump(bg, f)
        size_mb = os.path.getsize(p) / 1024**2
        log.info(f"[cache] bg_train 已缓存到 {p} ({size_mb:.1f} MB)")
        log.info(f"[cache] 下次跑只调预测端参数时, 把 CONFIG.USE_CACHED_TRAIN 改为 True 即可跳过训练")
    except Exception as e:
        log.warning(f"[cache] 保存训练缓存失败 ({type(e).__name__}: {e}), 下次仍需重训")


def _decide_train_device():
    """
    检测 GPU 可用空间, 决定训练阶段用 GPU 还是 CPU。

    背景: Bering._trainNode 存在隐性内存 leak (训练时 autograd/中间张量囤积不释放),
    经验上训练阶段会涨到 ~30 GiB。卡空间不够就 OOM, 无论我们怎么改 batch/window 都没用。

    策略:
      - GPU 空闲 >= TRAIN_GPU_NEED_GIB (默认 30) → 用 GPU 训练
      - 否则 → CPU 训练 (慢 5-8x,但绝对不 OOM)
      - 用户可在 CONFIG.FORCE_CPU_TRAIN=True 时跳过检测,强制 CPU
      - 预测阶段不受此影响,继续 GPU
    """
    if getattr(CONFIG, "FORCE_CPU_TRAIN", False):
        log.warning("[device] CONFIG.FORCE_CPU_TRAIN=True → 训练阶段强制 CPU")
        return "cpu"
    try:
        import torch
        if not torch.cuda.is_available():
            log.warning("[device] CUDA 不可用, 训练阶段用 CPU")
            return "cpu"
        free, total = torch.cuda.mem_get_info()
        free_gib = free / 1024**3
        total_gib = total / 1024**3
        need_gib = getattr(CONFIG, "TRAIN_GPU_NEED_GIB", 30.0)
        log.info(f"[device] GPU 可用 {free_gib:.1f}/{total_gib:.1f} GiB; 训练需 >= {need_gib} GiB")
        if free_gib < need_gib:
            log.warning(f"[device] 空间不够 (Bering _trainNode 训练时会涨到 ~30 GiB), "
                        f"训练阶段自动切到 CPU (慢但稳); 预测阶段仍 GPU")
            return "cpu"
        log.info("[device] 空间充足, 训练阶段用 GPU")
        return "gpu"
    except Exception as e:
        log.warning(f"[device] 检测失败 ({e}), 保守用 CPU 训练")
        return "cpu"


def _patch_bering_eval_no_grad():
    """
    包装 Bering.training._record._evaluate_metrics:
      1. 在 torch.no_grad() 下运行(避免评估 autograd 图累积)
      2. 调用前 gc.collect + torch.cuda.empty_cache(释放 PyTorch 缓存碎片)
    第 2 点针对共享卡场景:别人占卡使可用 VRAM 缩水,训练攒到一定程度后 PyTorch 的缓存
    碎片(reserved - allocated, 可达 2-3 GiB)就成了 eval 阶段的关键瓶颈。强制清理一次,
    把那块"未使用但已申请"的内存还给 CUDA,给 eval forward 让出空间。
    """
    try:
        import torch
        import gc
        from Bering.training import _record as _rec
    except Exception as e:
        log.warning(f"[patch] 无法导入 Bering.training._record: {e}")
        return
    if getattr(_rec, "_evaluate_metrics_patched", False):
        return
    if not hasattr(_rec, "_evaluate_metrics"):
        return
    _orig_eval = _rec._evaluate_metrics

    class _NonMutatingItem:
        """Proxy a PyG Data object so ``item.to(cuda)`` cannot mutate dataset.

        Bering's _evaluate_metrics iterates ``loader.dataset`` directly and
        calls ``batch_data.to(device)``.  PyG's ``Data.to`` is in-place, so the
        original evaluator silently leaves every graph resident on the GPU;
        this is why a small ROI with many genes can consume the whole card.
        """
        def __init__(self, item):
            self._item = item

        def __getattr__(self, name):
            return getattr(self._item, name)

        def to(self, *args, **kwargs):
            # PyG's clone is a tensor-wise copy on CPU; then only this
            # temporary evaluation copy is moved to the target device.
            return self._item.clone().to(*args, **kwargs)

    class _NonMutatingDataset:
        def __init__(self, dataset):
            self._dataset = dataset

        def __iter__(self):
            for item in self._dataset:
                yield _NonMutatingItem(item)

        def __len__(self):
            return len(self._dataset)

    class _EvalLoader:
        def __init__(self, loader):
            self.dataset = _NonMutatingDataset(loader.dataset)

    def _evaluate_metrics_no_grad(*a, **k):
        # 关键: 释放 PyTorch 缓存碎片, 共享卡时这往往是 eval 能否过 forward 的胜负手
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        # Replace only the loader seen by _evaluate_metrics.  The original
        # loader remains untouched for train/validation and its length is
        # still used by record().
        aa = list(a)
        if len(aa) >= 2:
            aa[1] = _EvalLoader(aa[1])
        elif "loader" in k:
            k["loader"] = _EvalLoader(k["loader"])
        with torch.no_grad():
            return _orig_eval(*aa, **k)

    _rec._evaluate_metrics = _evaluate_metrics_no_grad
    _rec._evaluate_metrics_patched = True
    log.info("[patch] 已包装 _evaluate_metrics: no_grad + 非原地 to() + 进入前 empty_cache")


def train_on_B(seg_t: pd.DataFrame, unseg_t: pd.DataFrame):
    """用子块数据建 BrGraph 并训练 node+edge 分类器。
    调用顺序严格复用切片A版已验证可跑的写法:
      br.BrGraph(关键字参数) -> br.graphs.BuildWindowGraphs -> br.graphs.CreateData -> br.train.Training
    """
    import Bering as br

    # 关键: 在调任何 Training 前打非原地评估补丁；旧版 evaluator 会把
    # loader.dataset 的每张 CPU 图永久搬到 GPU，导致高维数据评估时累计 OOM。
    _patch_bering_eval_no_grad()

    # 前景表用本版本要求的列名: x, y, z, features, raw_cells, raw_labels;背景表只要坐标+features
    df_seg = seg_t[["x", "y", "z", "features", "raw_cells", "raw_labels"]].copy()
    df_unseg = unseg_t[["x", "y", "z", "features"]].copy()

    log.info("[train] 构建 Bering 对象 (image-free)")
    bg = br.BrGraph(df_spots_seg=df_seg, df_spots_unseg=df_unseg, image=None, channels=None)
    log.info(f"[train]   n_transcripts={_attr(bg,'n_transcripts')} | "
             f"n_features={_attr(bg,'n_features')} | n_cells={_attr(bg,'n_cells_raw')} | "
             f"n_labels={_attr(bg,'n_labels_raw')} | device={_attr(bg,'device')}")

    log.info("[train] BuildWindowGraphs + CreateData (CPU 上累积窗口图, 避免 GPU 累积爆显存) ...")
    # 关键修复: BuildWindowGraphs 内部 BuildGraph 把每张窗口图立即 G.to(bg.device) 累积,
    # CreateData 也会把所有图搬到 bg.device。两者合起来吃光 VRAM。
    # 修法: 临时把 bg.device 切到 CPU 罩住 BuildWindowGraphs + CreateData,所有图建到
    # CPU RAM (几百 GiB 撑得起),罩完根据 GPU 空闲度决定训练用 GPU 还是 CPU。
    import torch
    _orig_device = bg.device
    # 在切到 CPU 前就检测一次 GPU 空闲度(以免后面 PyTorch 申请了别的就看不到真实空闲了)
    train_device_choice = _decide_train_device()
    log.info(f"[train]   bg.device 从 {_orig_device} 临时切到 CPU(覆盖 BuildWindowGraphs + CreateData)")
    bg.device = torch.device("cpu")
    try:
        br.graphs.BuildWindowGraphs(
            bg, n_cells_perClass=CONFIG.N_CELLS_PER_CLASS,
            window_width=CONFIG.WINDOW_WIDTH, window_height=CONFIG.WINDOW_HEIGHT,
            n_neighbors=CONFIG.N_NEIGHBORS,
            n_windows_per_cell=CONFIG.N_WINDOWS_PER_CELL,
        )
        log.info("[train]   BuildWindowGraphs 完成, 进入 CreateData")
        br.graphs.CreateData(bg, batch_size=CONFIG.BATCH_SIZE, training_ratio=CONFIG.INTERNAL_TRAIN_RATIO)
    finally:
        if train_device_choice == "gpu":
            bg.device = _orig_device
            log.info(f"[train]   bg.device 恢复为 {_orig_device}(GPU 训练)")
        else:
            # 保持 CPU,Training 也在 CPU 上跑,慢但绝对不 OOM
            log.warning(f"[train]   bg.device 保持 CPU(CPU 训练; 预计比 GPU 慢 5-8 倍)")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    log.info("[train] Training (节点分类器 + 边分类器) ...")
    train_fn = getattr(getattr(br, "train", None), "Training", None) \
        or getattr(getattr(br, "training", None), "Training", None)
    if train_fn is None:
        raise AttributeError("找不到 Training(试过 br.train.Training / br.training.Training)")
    # Bering.Training creates TrainerNode/TrainerEdge without passing
    # ``bg.device``; both constructors independently choose CUDA whenever it
    # is visible.  Consequently the old "CPU fallback" only built graphs on
    # CPU and then silently moved training batches back to CUDA.  Hide CUDA
    # from those constructors only while CPU training is requested, restoring
    # the original function immediately afterwards.
    _cuda_is_available = torch.cuda.is_available
    if train_device_choice == "cpu":
        torch.cuda.is_available = lambda: False
        log.warning("[train] 已将 CPU 设备选择传递给 Bering TrainerNode/TrainerEdge")
    try:
        train_fn(bg)
    finally:
        torch.cuda.is_available = _cuda_is_available

    log.info("[train] 完成。训练标签集合: %s",
             sorted(map(str, np.unique(df_seg["raw_labels"]))))
    return bg


# ============================================================================
# 预测(分块覆盖全片)
# ============================================================================
def _make_tiles(xmin, xmax, ymin, ymax, tile, margin):
    """把 [xmin,xmax]x[ymin,ymax] 切成核心块,每块外扩 margin 做预测、核心区收结果。"""
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


def _node_classification_memory_safe(bg, n_neighbors):
    """Run Bering node classification without first building a huge full graph.

    Bering 1.x always attempts the full graph before its coordinate-tiling
    fallback.  If that attempt fails for a table below its hard-coded 1.5 M
    cutoff, it later dereferences ``None``.  Calling the package's tested
    tiling helper directly avoids both the wasted full-graph allocation and
    that inconsistent cutoff while preserving the trained model and math.
    """
    import math
    import torch
    import Bering as br

    n_spots = len(bg.spots_all)
    direct_max = int(getattr(CONFIG, "NODE_CLASSIFICATION_DIRECT_MAX", 100_000))
    if n_spots <= direct_max:
        return br.tl.node_classification(
            bg, bg.spots_all.copy(), n_neighbors=n_neighbors
        )

    from Bering.tools._node_classification import (
        _get_node_embedding_prediction_byTiling,
        _get_pos,
    )

    target = max(10_000, int(getattr(
        CONFIG, "NODE_CLASSIFICATION_TARGET_CHUNK", 100_000
    )))
    chunks_axis = max(2, int(math.ceil(math.sqrt(n_spots / target))))
    num_chunks = chunks_axis ** 2
    log.info(
        f"[predict] 节点数 {n_spots:,} > {direct_max:,}，直接使用 "
        f"{chunks_axis}x{chunks_axis}={num_chunks} 个分类子块"
        f"（目标约 {target:,} nodes/chunk）"
    )

    bg.pos_all = _get_pos(bg.spots_all)
    bg.trainer_node.model.to("cpu")
    with torch.no_grad():
        bg.z_all, pred_logits = _get_node_embedding_prediction_byTiling(
            bg, bg.spots_all, num_chunks, n_neighbors, beta=1.0
        )
        max_probs, pred_indices = torch.max(pred_logits, dim=1)
    pred_indices[max_probs <= 0.3] = bg.n_labels_raw - 1
    pred_labels = np.array([
        bg.label_indices_dict[i.item()] for i in pred_indices
    ])
    bg.spots_all["predicted_labels"] = pred_labels
    bg.spots_all["predicted_probability"] = max_probs.numpy()
    bg.foreground_indices = np.where(pred_labels != "background")[0]
    return pred_labels, None


def predict_all_B(bg_train, seg_all, unseg_all):
    """对全片分块跑 node_classification + cell_segmentation + cell_annotation。
    每块内用训练好的模型(bg_train 的 trainer)。结果按核心区去重拼接。
    BrGraph 构造严格用切片A版验证过的 br.BrGraph(关键字参数) 形式。"""
    import Bering as br
    import torch

    # Preserve the registered ROI bounds even if a rare boundary transcript
    # is later excluded because its feature was unseen during training.
    allxy = pd.concat([seg_all[["x", "y"]], unseg_all[["x", "y"]]], axis=0)

    # The trained node model has no unknown-feature bucket.  use_settings()
    # replaces each prediction tile's feature table with bg_train.features, so
    # a transcript whose gene was absent from the training tile otherwise
    # raises KeyError in Bering.graphs._get_edge_index.  Such transcripts
    # cannot be represented by this fitted model and must be excluded before
    # graph construction.  Record the exact loss for the run summary.
    trained_features = pd.Index(bg_train.features.index)
    seg_known = seg_all["features"].isin(trained_features)
    unseg_known = unseg_all["features"].isin(trained_features)
    unknown = pd.concat([
        seg_all.loc[~seg_known, "features"],
        unseg_all.loc[~unseg_known, "features"],
    ], ignore_index=True)
    feature_stats = {
        "training_features": int(len(trained_features)),
        "excluded_transcripts": int(len(unknown)),
        "excluded_features": int(unknown.nunique()),
        "excluded_feature_names": sorted(map(str, unknown.unique())),
    }
    if len(unknown):
        preview = ", ".join(feature_stats["excluded_feature_names"][:20])
        if feature_stats["excluded_features"] > 20:
            preview += ", ..."
        log.warning(
            "[predict] 排除 %s 条训练子区未见转录本"
            "（%d 个基因: %s）；已训练模型无法编码这些基因",
            f"{len(unknown):,}", feature_stats["excluded_features"], preview,
        )
        seg_all = seg_all.loc[seg_known].copy()
        unseg_all = unseg_all.loc[unseg_known].copy()

    if len(seg_all) + len(unseg_all) == 0:
        raise RuntimeError("预测数据中没有任何训练模型已见基因")

    # 预测端前景种子的 raw_labels 占位(node_classification 会用模型预测覆盖它);
    # 取训练标签集合里任一前景类即可。
    fg_labels = [l for l in bg_train.raw_labels if str(l) != "background"]
    placeholder_label = fg_labels[0] if fg_labels else "background"

    xmin, xmax = allxy["x"].min(), allxy["x"].max()
    ymin, ymax = allxy["y"].min(), allxy["y"].max()
    tiles = _make_tiles(xmin, xmax, ymin, ymax, CONFIG.TILE_SIZE_UM, CONFIG.TILE_MARGIN_UM)
    log.info(f"[predict] 全片 x[{xmin:.0f},{xmax:.0f}] y[{ymin:.0f},{ymax:.0f}];"
             f" 切成 {len(tiles)} 块(核心 {CONFIG.TILE_SIZE_UM}um + margin {CONFIG.TILE_MARGIN_UM}um)")

    results = []
    selected_resolutions = []
    n_ok = n_skip = n_low_tx = n_no_seed = n_failed = 0
    for ti, (core, ext) in enumerate(tiles):
        seg_e = _crop(seg_all, ext)
        unseg_e = _crop(unseg_all, ext)
        n_tx = len(seg_e) + len(unseg_e)
        if n_tx < CONFIG.MIN_TILE_TRANSCRIPTS:
            n_skip += 1
            n_low_tx += 1
            continue
        try:
            if len(seg_e) == 0:
                # 块内没有任何已分配种子,跳过(否则建不了前景图)
                n_skip += 1
                n_no_seed += 1
                continue

            # 预测端建图:前景用本块 Xenium 已分配转录本当 coarse 种子(Bering 会重新分割,
            # 这里只为建对象);raw_cells 重新 factorize 成整数,raw_labels 用占位(预测时被覆盖)。
            # 严格复用切片A版 predict 的建表/建图写法。
            extra_cols = [c for c in ("transcript_id", "original_gene") if c in seg_e.columns]
            df_seg = seg_e[["x", "y", "z", "features"] + extra_cols].copy()
            # Preserve the original coarse CellBin/Xenium seed separately from
            # Bering's tile-local integer ``raw_cells``.  The benchmark runner
            # uses this only to safely reconnect the same seed cell when it is
            # split by a tile boundary.
            df_seg["seed_cell_id"] = seg_e["raw_cells"].astype(str).values
            df_seg["raw_cells"] = pd.factorize(seg_e["raw_cells"])[0].astype(int)
            df_seg["raw_labels"] = placeholder_label
            extra_cols_unseg = [c for c in ("transcript_id", "original_gene") if c in unseg_e.columns]
            df_unseg = unseg_e[["x", "y", "z", "features"] + extra_cols_unseg].copy()
            df_unseg["seed_cell_id"] = "UNASSIGNED"

            bg = br.BrGraph(df_spots_seg=df_seg, df_spots_unseg=df_unseg, image=None, channels=None)
            bg.use_settings(bg_train)  # 关键:搬训练好的 features/labels/trainer
            # Bering.use_settings() only updates ``n_labels`` but its node
            # classifier allocates by ``n_labels_raw``.  A prediction tile is
            # constructed with one placeholder label (2 classes including
            # background), whereas the trained classifier outputs all 9
            # classes.  Synchronize the field actually used by the package.
            bg.n_labels_raw = bg_train.n_labels_raw

            _node_classification_memory_safe(bg, CONFIG.N_NEIGHBORS)
            # cell_segmentation builds edges only for predicted foreground
            # nodes.  Keep latent rows aligned with those edge indices.
            if hasattr(bg, "z_all") and len(bg.z_all) == len(bg.spots_all):
                fg = torch.as_tensor(bg.foreground_indices, dtype=torch.long)
                bg.z_all = bg.z_all.index_select(0, fg).clone()
            br.tl.cell_segmentation(
                bg,
                use_image=CONFIG.SEG_USE_IMAGE,
                positive_edge_thresh=CONFIG.SEG_POSITIVE_EDGE_THRESH,
                leiden_resolution=CONFIG.SEG_LEIDEN_RESOLUTION,
                num_edges_perSpot=CONFIG.SEG_NUM_EDGES_PER_SPOT,
                graph_n_neighbors=CONFIG.SEG_GRAPH_N_NEIGHBORS,
            )
            selected_resolution = getattr(bg, "resolution", None)
            df_res, _, _ = br.tl.cell_annotation(
                bg,
                min_transcripts=CONFIG.MIN_TRANSCRIPTS,
                min_dominant_nodes=CONFIG.MIN_DOMINANT_NODES,
                min_dominant_ratio=CONFIG.MIN_DOMINANT_RATIO,
            )
            if n_ok == 0:
                log.info(f"[predict] cell_annotation 输出列(首个成功块): {list(df_res.columns)}")

            cell_col = _resolve_col(
                df_res, ["ensembled_cells", "predicted_cells", "cells", "cell_id", "cell"], "细胞ID")
            lbl_col = _resolve_col(
                df_res, ["ensembled_labels", "predicted_labels", "predicted_node_labels",
                         "labels", "cell_type"], "细胞类型")

            # 只保留核心区结果(去重叠)
            x0, x1, y0, y1 = core
            keep = (df_res["x"] >= x0) & (df_res["x"] < x1) & \
                   (df_res["y"] >= y0) & (df_res["y"] < y1)
            sub = df_res[keep].copy()
            # 过滤未分配(cell==0)
            sub = sub[sub[cell_col] != 0].copy()
            if len(sub) == 0:
                if selected_resolution is not None:
                    selected_resolutions.append(float(selected_resolution))
                n_ok += 1
                continue

            # 全局唯一 cell_id:块号 + 局部cell号(避免不同块的局部编号冲突)
            sub["global_cell"] = f"t{ti}_" + sub[cell_col].astype(str)

            # New benchmark inputs carry transcript_id and original_gene as
            # extra BrGraph columns. Bering preserves them in spots_all and
            # cell_annotation, so no coordinate-based de-duplication is
            # needed (important when many rare genes share the OTHER model
            # feature). Keep the old merge only for legacy direct callers.
            if "transcript_id" not in sub.columns:
                ext_all = pd.concat([seg_e, unseg_e], axis=0)
                key = ext_all[["x", "y", "features", "transcript_id"]].drop_duplicates(
                    subset=["x", "y", "features"])
                sub = sub.merge(key, on=["x", "y", "features"], how="left")
            output_gene = sub["original_gene"] if "original_gene" in sub.columns else sub["features"]

            if "seed_cell_id" in sub.columns:
                seed_cell_id = sub["seed_cell_id"].astype(str).values
            else:
                # Defensive fallback for Bering releases that discard extra
                # BrGraph columns during annotation.
                seed_lookup = seg_e.drop_duplicates("transcript_id").set_index(
                    "transcript_id"
                )["raw_cells"]
                seed_cell_id = sub["transcript_id"].map(seed_lookup).fillna(
                    "UNASSIGNED"
                ).astype(str).values

            out = pd.DataFrame({
                "transcript_id": sub["transcript_id"].values,
                "cell_id": sub["global_cell"].values,
                "gene": output_gene.values,
                "x": sub["x"].values,
                "y": sub["y"].values,
                "cell_type": sub[lbl_col].astype(str).values,
                "seed_cell_id": seed_cell_id,
                "tile_index": np.full(len(sub), ti, dtype=np.int32),
            })
            results.append(out)
            if selected_resolution is not None:
                selected_resolutions.append(float(selected_resolution))
            n_ok += 1
        except Exception as e:
            n_skip += 1
            n_failed += 1
            log.exception(f"[predict] 块 {ti} 失败: {e}")
            if getattr(CONFIG, "STRICT_TILE_ERRORS", False):
                raise RuntimeError(
                    f"预测块 {ti}/{len(tiles)} 失败；严格模式禁止输出缺块结果"
                ) from e

        if (ti + 1) % 20 == 0:
            log.info(f"[predict] 进度 {ti+1}/{len(tiles)};成功 {n_ok} 跳过 {n_skip}")

    tile_stats = {
        "total": len(tiles),
        "success": n_ok,
        "skipped": n_skip,
        "skipped_low_transcripts": n_low_tx,
        "skipped_no_seed": n_no_seed,
        "failed": n_failed,
    }
    resolution_counts = {}
    for value in selected_resolutions:
        key = str(value)
        resolution_counts[key] = resolution_counts.get(key, 0) + 1
    tile_stats["selected_leiden_resolution_counts"] = resolution_counts
    log.info(f"[predict] 完成: {tile_stats}")
    if not results:
        raise RuntimeError("所有块都没有产出结果,检查数据/参数(看上面 cell_annotation 输出列日志)。")
    df_out = pd.concat(results, axis=0, ignore_index=True)
    df_out.attrs["tile_stats"] = tile_stats
    df_out.attrs["feature_stats"] = feature_stats
    df_out.attrs["tile_cores"] = [list(core) for core, _ in tiles]
    log.info(f"[predict] 拼接后 {len(df_out):,} 条转录本被分配到 {df_out['cell_id'].nunique():,} 个细胞")
    return df_out


# ============================================================================
# 导出三件套
# ============================================================================
def export_assignments(df_out: pd.DataFrame, outdir: Path):
    """sliceB_transcript_assignments.csv: transcript_id, cell_id, gene, x, y"""
    cols = ["transcript_id", "cell_id", "gene", "x", "y"]
    df_out[cols].to_csv(outdir / "sliceB_transcript_assignments.csv", index=False)
    log.info(f"[out] sliceB_transcript_assignments.csv ({len(df_out):,} 行)")


def export_cell_table(df_out: pd.DataFrame, outdir: Path):
    """sliceB_cells.csv: 每个细胞的质心坐标矩阵 + 计数 + 类型。"""
    g = df_out.groupby("cell_id")
    agg = g.agg(
        cx=("x", "mean"),
        cy=("y", "mean"),
        n_transcripts=("transcript_id", "size"),
        n_genes=("gene", "nunique"),
    ).reset_index()
    # 细胞类型:取该细胞内多数转录本的 cell_type
    ctype = g["cell_type"].agg(lambda s: s.value_counts().index[0]).reset_index()
    ctype.columns = ["cell_id", "cell_type"]
    cells = agg.merge(ctype, on="cell_id", how="left")
    cells.to_csv(outdir / "sliceB_cells.csv", index=False)
    log.info(f"[out] sliceB_cells.csv ({len(cells):,} 个细胞;含质心坐标矩阵 cx,cy)")
    return cells


def export_boundaries(df_out: pd.DataFrame, outdir: Path):
    """sliceB_cell_boundaries.csv: 每个细胞成员转录本的凸包边界顶点。
    格式 cell_id, vertex_order, x, y(按多边形顺序),便于你画 cell boundary。"""
    if not CONFIG.EXPORT_BOUNDARIES:
        return
    try:
        from scipy.spatial import ConvexHull
    except Exception as e:
        log.warning(f"[out] 无 scipy.spatial.ConvexHull,跳过边界导出: {e}")
        return

    rows = []
    n_cells = df_out["cell_id"].nunique()
    log.info(f"[out] 计算 {n_cells:,} 个细胞的凸包边界 ...")
    for cid, sub in df_out.groupby("cell_id"):
        pts = sub[["x", "y"]].values
        if len(pts) < CONFIG.MIN_BOUNDARY_PTS:
            continue
        try:
            hull = ConvexHull(pts)
        except Exception:
            continue  # 共线/重复点
        verts = pts[hull.vertices]  # 逆时针顺序
        for order, (vx, vy) in enumerate(verts):
            rows.append((cid, order, float(vx), float(vy)))
    if not rows:
        log.warning("[out] 没有可用的细胞边界(可能细胞普遍太小)。")
        return
    bdf = pd.DataFrame(rows, columns=["cell_id", "vertex_order", "x", "y"])
    bdf.to_csv(outdir / "sliceB_cell_boundaries.csv", index=False)
    log.info(f"[out] sliceB_cell_boundaries.csv ({bdf['cell_id'].nunique():,} 个细胞的多边形)")


def summarize(df_out: pd.DataFrame, n_input_tx: int, outdir: Path):
    g = df_out.groupby("cell_id")["transcript_id"].size()
    summary = {
        "n_transcripts_input_foreground_plus_bg": int(n_input_tx),
        "n_transcripts_assigned": int(len(df_out)),
        "fraction_assigned_of_input": float(len(df_out) / n_input_tx) if n_input_tx else None,
        "n_cells_bering": int(df_out["cell_id"].nunique()),
        "median_transcripts_per_cell": float(g.median()),
        "mean_transcripts_per_cell": float(g.mean()),
        "method_note": (
            "Bering 是分割精修器(refinement):使用切片B的 Xenium 初始分割(transcripts.cell_id) "
            "作为粗分配 raw_cells、16类注释(group)作为 raw_labels 进行监督训练。"
            "与 Baysor/Proseg/BIDCell 等无参考/弱参考方法横向比较时,需在 Methods 中声明此起跑线差异。"
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

    # GPU + 显存碎片缓解 + 显存统计补丁(必须在任何 torch/Bering 导入前)
    if CONFIG.CUDA_VISIBLE_DEVICES != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG.CUDA_VISIBLE_DEVICES)
        log.info(f"CUDA_VISIBLE_DEVICES = {CONFIG.CUDA_VISIBLE_DEVICES}")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    log.info(f"PYTORCH_CUDA_ALLOC_CONF = {os.environ['PYTORCH_CUDA_ALLOC_CONF']}")
    _make_cuda_memstats_safe()

    outdir = Path(CONFIG.OUTDIR).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) 读 + 预处理 + 贴标签 + 切前景/背景
    df = load_and_preprocess()
    n_input_tx = len(df)
    df = attach_labels(df)
    seg_all, unseg_all = split_seg_unseg(df)

    # 2) 选训练子块
    seg_t, unseg_t, _ = select_train_tile(seg_all, unseg_all)

    # 3) chdir 到工作目录(Bering 写死相对路径 figures/),预建子目录
    work_dir = outdir / "bering_work"
    for sub in ("figures", "checkpoints", "models"):
        (work_dir / sub).mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)
    log.info(f"工作目录已切到 {work_dir}(Bering 的 figures/ 等相对产物落这里)")

    # 4) 训练 (若有缓存且 USE_CACHED_TRAIN=True 则跳过)
    bg_train = _try_load_cached_bg_train()
    if bg_train is None:
        bg_train = train_on_B(seg_t, unseg_t)
        _save_bg_train_cache(bg_train)

    # 5) 全片预测
    df_out = predict_all_B(bg_train, seg_all, unseg_all)

    # 6) 导出三件套 + summary(写到绝对 outdir)
    export_assignments(df_out, outdir)
    export_cell_table(df_out, outdir)
    export_boundaries(df_out, outdir)
    summarize(df_out, n_input_tx, outdir)

    log.info(f"全部完成。输出在 {outdir}")


if __name__ == "__main__":
    main()
