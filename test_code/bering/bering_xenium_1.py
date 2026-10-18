import os
import sys
import traceback
import datetime
import warnings
import re

# ============================================================
# 0-A. 日志重定向
# ============================================================

_LOG_DIR = "./bering_logs"
os.makedirs(_LOG_DIR, exist_ok=True)

_ts = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
_stdout_log = open(os.path.join(_LOG_DIR, f"stdout_{_ts}.log"), "w", buffering=1)
_stderr_log = open(os.path.join(_LOG_DIR, f"stderr_{_ts}.log"), "w", buffering=1)

class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

sys.stdout = _Tee(sys.__stdout__, _stdout_log)
sys.stderr = _Tee(sys.__stderr__, _stderr_log)

print(f"[LOG] stdout -> {_stdout_log.name}")
print(f"[LOG] stderr -> {_stderr_log.name}")


# ============================================================
# 0-B. 强制禁用 GPU，必须在 import torch 前
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import matplotlib
matplotlib.use("Agg")

import torch
import numpy as np
import pandas as pd

# ============================================================
# 0-B-2. torch CUDA 兜底补丁
# ============================================================

torch.Tensor.cuda = lambda self, *args, **kwargs: self
torch.nn.Module.cuda = lambda self, *args, **kwargs: self

torch.cuda.is_available = lambda: False
torch.cuda.reset_peak_memory_stats = lambda *args, **kwargs: None
torch.cuda.empty_cache = lambda *args, **kwargs: None
torch.cuda.synchronize = lambda *args, **kwargs: None
torch.cuda.memory_allocated = lambda *args, **kwargs: 0
torch.cuda.max_memory_allocated = lambda *args, **kwargs: 0


# ============================================================
# 0-C. 修复 Bering 源码中的 CUDA 硬编码
# ============================================================

def _patch_bering_cuda(verbose: bool = True):
    """
    强制把 Bering 源码里的 CUDA 写法全部替换为 CPU。

    处理内容：
      device='cuda'
      device = 'cuda'
      device="cuda"
      device = "cuda"
      torch.device('cuda')
      torch.device("cuda")
      .to('cuda')
      .to("cuda")
      .cuda()
      .cuda(non_blocking=True)
      torch.cuda.FloatTensor 等
    """

    try:
        import Bering as _br_pkg
    except ImportError:
        print("[PATCH] Bering not found, skipping patch.")
        return

    import importlib

    pkg_root = os.path.dirname(_br_pkg.__file__)
    patched_files = []

    for dirpath, _, filenames in os.walk(pkg_root):
        for fname in filenames:
            if not fname.endswith(".py"):
                continue

            fpath = os.path.join(dirpath, fname)

            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    src = f.read()
            except Exception:
                continue

            new_src = src

            # device = 'cuda' / device="cuda"，允许中间有空格
            new_src = re.sub(
                r"device\s*=\s*['\"]cuda['\"]",
                "device='cpu'",
                new_src,
            )

            # torch.device('cuda') / torch.device("cuda")
            new_src = re.sub(
                r"torch\.device\s*\(\s*['\"]cuda['\"]\s*\)",
                "torch.device('cpu')",
                new_src,
            )

            # .to('cuda') / .to("cuda")
            new_src = re.sub(
                r"\.to\s*\(\s*['\"]cuda['\"]\s*\)",
                ".to('cpu')",
                new_src,
            )

            # .cuda() / .cuda(non_blocking=True) / .cuda(anything)
            new_src = re.sub(
                r"\.cuda\s*\([^)]*\)",
                ".to('cpu')",
                new_src,
            )

            # torch.cuda.*Tensor
            new_src = new_src.replace("torch.cuda.FloatTensor", "torch.FloatTensor")
            new_src = new_src.replace("torch.cuda.DoubleTensor", "torch.DoubleTensor")
            new_src = new_src.replace("torch.cuda.LongTensor", "torch.LongTensor")
            new_src = new_src.replace("torch.cuda.IntTensor", "torch.IntTensor")
            new_src = new_src.replace("torch.cuda.BoolTensor", "torch.BoolTensor")

            if new_src != src:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(new_src)

                patched_files.append(fpath)

                if verbose:
                    rel = os.path.relpath(fpath, pkg_root)
                    print(f"  [PATCH] CUDA -> CPU fixed in Bering/{rel}")

    # 删除 Bering 已加载模块，避免 Python 继续用旧缓存
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("Bering"):
            del sys.modules[mod_name]

    print(f"[PATCH] Done. Patched {len(patched_files)} file(s).")


print("\n[0-C] Patching Bering source for CPU-only mode...")
_patch_bering_cuda()


# ============================================================
# 现在再 import Bering
# ============================================================

import Bering as br
from Bering.training.train import _trainNode, _trainEdge
from Bering.training import TrainerNode, TrainerEdge
from Bering.models import GCN, EdgeClf

warnings.filterwarnings("ignore")

print("Running on CPU only. CUDA disabled and Bering patched.")


# ============================================================
# 内存监控工具
# ============================================================

def _mem_gb() -> str:
    try:
        import psutil
        rss = psutil.Process(os.getpid()).memory_info().rss
        return f"{rss / 1024**3:.2f} GB"
    except Exception:
        return "N/A"

def _log_mem(tag: str = ""):
    print(f"  [MEM{(' ' + tag) if tag else ''}] RSS = {_mem_gb()}")


# ============================================================
# 1. 参数
# ============================================================

XENIUM_DIR = "./Xenium_1"
OUT_DIR = os.path.join(XENIUM_DIR, "bering_results")
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

TRANSCRIPTS_FILE = os.path.join(XENIUM_DIR, "transcripts.parquet")
CELL_GROUPS_FILE = os.path.join(XENIUM_DIR, "cell_groups.csv")

USE_IMAGE = False
MIN_QV = 20

# ---------- 训练窗口 ----------
N_CELLS_PER_CLASS = 10
WINDOW_WIDTH = 100.0
WINDOW_HEIGHT = 100.0
TRAIN_N_NEIGHBORS = 10
BATCH_SIZE = 4
TRAINING_RATIO = 0.8

# ---------- 训练超参 ----------
NODE_EPOCHES = 50
EDGE_EPOCHES = 50
NODE_LR = 1e-3
EDGE_LR = 1e-3

# ---------- Node classifier 网络 ----------
NODE_GCN_HIDDEN = [64, 32, 16]
NODE_MLP_HIDDEN = [16, 16]

# ---------- Edge classifier 超参 ----------
EDGE_DECODER_MLP = [16, 8]
EDGE_RBF_START = 0
EDGE_RBF_STOP = 64
EDGE_RBF_N_KERNELS = 64
EDGE_NUM_POS_EDGES = 1000
EDGE_NUM_NEG_EDGES = 1000
EDGE_SUBIMAGE_BIN = 5

# ---------- Inference 超参 ----------
NODE_N_NEIGHBORS = 30
NODE_PROB_THRESHOLD = 0.3
MAX_NUM_SPOTS = 1_500_000
NUM_CHUNKS = 25

POSITIVE_EDGE_THRESH = 0.45
LEIDEN_RESOLUTION = 0.03
NUM_EDGES_PER_SPOT = 300
GRAPH_N_NEIGHBORS = 10
NUM_ITERS = 100


# ============================================================
# 2. 读取数据
# ============================================================

print("\n[1] Reading Xenium files...")
_log_mem("before read")

tx = pd.read_parquet(TRANSCRIPTS_FILE)
cell_groups = pd.read_csv(CELL_GROUPS_FILE)

print("transcripts:", tx.shape)
print("cell_groups:", cell_groups.shape)

_log_mem("after read")


# ============================================================
# 3. 标准化列名
# ============================================================

print("\n[2] Standardizing columns...")

tx = tx.rename(
    columns={
        "feature_name": "features",
        "x_location": "x",
        "y_location": "y",
        "z_location": "z",
    }
)

if "z" not in tx.columns:
    tx["z"] = 0.0

required_tx_cols = ["cell_id", "features", "x", "y", "z"]
missing = [c for c in required_tx_cols if c not in tx.columns]
if missing:
    raise ValueError(f"transcripts 缺少必要列: {missing}")

required_group_cols = ["cell_id", "group"]
missing = [c for c in required_group_cols if c not in cell_groups.columns]
if missing:
    raise ValueError(f"cell_groups.csv 缺少必要列: {missing}")


# ============================================================
# 4. QC 过滤
# ============================================================

print("\n[3] QC filtering...")

n_before = tx.shape[0]

if MIN_QV is not None and "qv" in tx.columns:
    tx = tx[tx["qv"] >= MIN_QV].copy()

tx["features"] = tx["features"].astype(str)

negative_pattern = (
    "NegControl|NegativeControl|BLANK|Blank|blank|"
    "DeprecatedCodeword|antisense|UnassignedCodeword"
)

tx = tx[
    ~tx["features"].str.contains(
        negative_pattern,
        case=False,
        regex=True,
        na=False,
    )
].copy()

print("transcripts before QC:", n_before)
print("transcripts after  QC:", tx.shape[0])


# ============================================================
# 5. 合并 cell type labels
# ============================================================

print("\n[4] Merging labels from cell_groups.csv...")

tx["cell_id"] = tx["cell_id"].astype(str)

cell_groups = cell_groups[["cell_id", "group"]].copy()
cell_groups["cell_id"] = cell_groups["cell_id"].astype(str)
cell_groups["group"] = cell_groups["group"].astype(str)

tx = tx.merge(cell_groups, on="cell_id", how="left")

assigned = (
    tx["cell_id"].notna()
    & (tx["cell_id"] != "UNASSIGNED")
    & (tx["cell_id"] != "")
    & tx["group"].notna()
)

df_spots_seg = tx.loc[
    assigned,
    ["x", "y", "z", "features", "cell_id", "group"],
].copy()

df_spots_unseg = tx.loc[
    ~assigned,
    ["x", "y", "z", "features"],
].copy()

df_spots_seg = df_spots_seg.rename(
    columns={
        "cell_id": "raw_cells",
        "group": "raw_labels",
    }
)

df_spots_seg["segmented"] = df_spots_seg["raw_cells"]
df_spots_seg["labels"] = df_spots_seg["raw_labels"]

df_spots_seg = df_spots_seg[
    [
        "x",
        "y",
        "z",
        "features",
        "raw_cells",
        "raw_labels",
        "segmented",
        "labels",
    ]
]

df_spots_unseg = df_spots_unseg[
    ["x", "y", "z", "features"]
]

print("segmented   transcripts:", df_spots_seg.shape)
print("unsegmented transcripts:", df_spots_unseg.shape)

print("\nlabel counts:")
print(df_spots_seg["raw_labels"].value_counts())

df_spots_seg.to_parquet(
    os.path.join(OUT_DIR, "bering_input_spots_segmented.parquet"),
    index=False,
)

df_spots_unseg.to_parquet(
    os.path.join(OUT_DIR, "bering_input_spots_unsegmented.parquet"),
    index=False,
)

del tx
import gc
gc.collect()

_log_mem("after del tx")


# ============================================================
# 6. 图像加载
# ============================================================

print("\n[5] Image loading skipped. USE_IMAGE = False.")


# ============================================================
# 7. 创建 BrGraph
# ============================================================

print("\n[6] Creating BrGraph...")
_log_mem("before BrGraph")

bg = br.BrGraph(
    df_spots_seg=df_spots_seg,
    df_spots_unseg=df_spots_unseg,
)

_log_mem("after BrGraph")


# ============================================================
# 8. 构建训练窗口图
# ============================================================

print("\n[7] Building training window graphs...")
_log_mem("before BuildWindowGraphs")

br.graphs.BuildWindowGraphs(
    bg,
    n_cells_perClass=N_CELLS_PER_CLASS,
    window_width=WINDOW_WIDTH,
    window_height=WINDOW_HEIGHT,
    n_neighbors=TRAIN_N_NEIGHBORS,
)

_log_mem("after BuildWindowGraphs")


# ============================================================
# 9. 创建训练数据
# ============================================================

print("\n[8] Creating training data...")
_log_mem("before CreateData")

br.graphs.CreateData(
    bg,
    batch_size=BATCH_SIZE,
    training_ratio=TRAINING_RATIO,
)

_log_mem("after CreateData")


# ============================================================
# 10. 训练
# ============================================================

NODE_CKPT = os.path.join(CKPT_DIR, "node_classifier.pt")
EDGE_CKPT = os.path.join(CKPT_DIR, "edge_classifier.pt")

random_digit = np.random.randint(1000, 2000)

performance_folder = (
    "figures/performance_"
    + datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    + f"_{random_digit}"
)

os.makedirs(performance_folder, exist_ok=True)


# ============================================================
# 10-A. Node classifier
# ============================================================

print("\n[9A] Node classifier...")
_log_mem("before NodeClf init")

nodeclf = GCN(
    n_features=bg.n_node_features,
    n_classes=bg.n_labels_raw,
    gcn_hidden_layer_dims=NODE_GCN_HIDDEN,
    mlp_hidden_layer_dims=NODE_MLP_HIDDEN,
)

bg.trainer_node = TrainerNode(
    nodeclf,
    lr=NODE_LR,
    weight_decay=5e-4,
    weight_seg=1.0,
    weight_bg=1.0,
)

if os.path.exists(NODE_CKPT):
    print(f"  Found checkpoint, loading: {NODE_CKPT}")

    bg.trainer_node.model.load_state_dict(
        torch.load(NODE_CKPT, map_location="cpu")
    )

    bg.trainer_node.model.eval()

else:
    print(f"  Training node classifier from scratch ({NODE_EPOCHES} epochs)...")
    _log_mem("before _trainNode")

    try:
        bg.trainer_node = _trainNode(
            bg.trainer_node,
            bg.train_loader,
            bg.test_loader,
            "training_GCN",
            performance_folder,
            epoches=NODE_EPOCHES,
            early_stop=False,
            early_stop_patience=5,
            early_stop_min_delta=0.05,
            plot_ax_size=5.0,
        )

    except MemoryError:
        print("FATAL: MemoryError in _trainNode — 内存不足，请降低 N_CELLS_PER_CLASS 或 BATCH_SIZE")
        traceback.print_exc()
        sys.exit(1)

    except Exception as e:
        print(f"FATAL: Exception in _trainNode: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)

    _log_mem("after _trainNode")

    torch.save(bg.trainer_node.model.state_dict(), NODE_CKPT)

    print(f"  Saved node classifier -> {NODE_CKPT}")

for p in bg.trainer_node.model.parameters():
    p.requires_grad = False


# ============================================================
# 10-B. Edge classifier
# ============================================================

print("\n[9B] Edge classifier...")
_log_mem("before EdgeClf init")

edgeclf = EdgeClf(
    n_node_latent_features=NODE_MLP_HIDDEN[1],
    image=bg.image_raw,
    image_repr="cellpose",
    cellpose_flow=None,
    image_model=(bg.image_raw is not None),
    decoder_mlp_layer_dims=EDGE_DECODER_MLP,
    distance_type="rbf",
    rbf_start=EDGE_RBF_START,
    rbf_stop=EDGE_RBF_STOP,
    rbf_n_kernels=EDGE_RBF_N_KERNELS,
    rbf_learnable=True,
    encoder_image_layer_dims_conv2d=[6, 16, 32, 64, 128],
    encoder_image_layer_dims_mlp=[32, 64],
    subimage_binsize=EDGE_SUBIMAGE_BIN,
    max_subimage_size=bg.window_size - EDGE_SUBIMAGE_BIN,
    min_subimage_size=EDGE_SUBIMAGE_BIN,
)

bg.trainer_edge = TrainerEdge(
    edgeclf,
    bg.trainer_node.model,
    lr=EDGE_LR,
    weight_decay=5e-4,
    num_pos_edges=EDGE_NUM_POS_EDGES,
    num_neg_edges=EDGE_NUM_NEG_EDGES,
)

image_for_edge = None

if os.path.exists(EDGE_CKPT):
    print(f"  Found checkpoint, loading: {EDGE_CKPT}")

    bg.trainer_edge.model.load_state_dict(
        torch.load(EDGE_CKPT, map_location="cpu")
    )

    bg.trainer_edge.model.eval()

else:
    print(f"  Training edge classifier from scratch ({EDGE_EPOCHES} epochs)...")
    _log_mem("before _trainEdge")

    try:
        bg.trainer_edge = _trainEdge(
            bg.trainer_edge,
            image_for_edge,
            bg.train_loader,
            bg.test_loader,
            "training_GCN",
            performance_folder,
            epoches=EDGE_EPOCHES,
            early_stop=False,
            early_stop_patience=5,
            early_stop_min_delta=0.05,
            plot_ax_size=5.0,
        )

    except MemoryError:
        print("FATAL: MemoryError in _trainEdge — 内存不足，请降低 EDGE_NUM_POS_EDGES、EDGE_NUM_NEG_EDGES 或 BATCH_SIZE")
        traceback.print_exc()
        sys.exit(1)

    except Exception as e:
        print(f"FATAL: Exception in _trainEdge: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)

    _log_mem("after _trainEdge")

    torch.save(bg.trainer_edge.model.state_dict(), EDGE_CKPT)

    print(f"  Saved edge classifier -> {EDGE_CKPT}")

try:
    del bg.train_loader
    del bg.test_loader
except AttributeError:
    pass

gc.collect()
_log_mem("after training cleanup")


# ============================================================
# 11. 全片 node classification
# ============================================================

print("\n[10] Running node classification...")
_log_mem("before node_classification")

try:
    br.tl.node_classification(
        bg,
        bg.spots_all.copy(),
        n_neighbors=NODE_N_NEIGHBORS,
        prob_threshold=NODE_PROB_THRESHOLD,
        max_num_spots=MAX_NUM_SPOTS,
        num_chunks=NUM_CHUNKS,
    )

except Exception as e:
    print(f"FATAL: Exception in node_classification: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

_log_mem("after node_classification")


# ============================================================
# 12. 全片 cell segmentation
# ============================================================

print("\n[11] Running cell segmentation...")
_log_mem("before cell_segmentation")

try:
    pred_cells = br.tl.cell_segmentation(
        bg,
        use_image=USE_IMAGE,
        positive_edge_thresh=POSITIVE_EDGE_THRESH,
        leiden_resolution=LEIDEN_RESOLUTION,
        num_edges_perSpot=NUM_EDGES_PER_SPOT,
        graph_n_neighbors=GRAPH_N_NEIGHBORS,
        num_iters=NUM_ITERS,
    )

except Exception as e:
    print(f"FATAL: Exception in cell_segmentation: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

_log_mem("after cell_segmentation")


# ============================================================
# 13. cell annotation + 输出
# ============================================================

print("\n[12] Running cell annotation and saving outputs...")

try:
    df_results, adata_ensembl, adata_seg = br.tl.cell_annotation(bg)

except Exception as e:
    print(f"FATAL: Exception in cell_annotation: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

df_results.to_csv(
    os.path.join(OUT_DIR, "bering_transcripts_with_predicted_cells.csv"),
    index=True,
)

adata_ensembl.write_h5ad(
    os.path.join(OUT_DIR, "bering_ensembled_cells.h5ad")
)

adata_seg.write_h5ad(
    os.path.join(OUT_DIR, "bering_segmented_cells.h5ad")
)

cell_label_table = df_spots_seg[
    ["raw_cells", "raw_labels"]
].drop_duplicates()

cell_label_table.to_csv(
    os.path.join(OUT_DIR, "input_cell_labels_used_for_training.csv"),
    index=False,
)

print("\n========== Done ==========")
print("Output directory:", OUT_DIR)
print("Checkpoints     :", CKPT_DIR)
print("df_results      :", df_results.shape)
print("adata_ensembl   :", adata_ensembl.shape)
print("adata_seg       :", adata_seg.shape)

_log_mem("final")
