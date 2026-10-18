#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmx_lymph.py
==============
在 CosMx 人淋巴结数据集（/data/qiuyijia/dataset/cosmx_lymph_node）上跑 Cellpose-SAM
细胞分割，并以数据集自带的 CosMx CellLabels 为 GT，复用 /data/qiuyijia/eval/test.py
的指标做 benchmark。

数据组织与之前的 CosMx 胰腺数据不同
------------------------------------------------
胰腺那份是 flat files（CellComposite_F051.jpg / CellLabels_F051.tif 一个 FOV 一个文件），
这份淋巴结只提供了 napari 拼接好的 **OME-Zarr 全片**：

    napari/images/{DNA,Membrane,PanCK,CD45,CD68}/{0..7}   # uint16, 8 级金字塔
    napari/images/labels/{0..7}                           # uint32, CosMx 官方分割 = GT
    napari/images/.zattrs                                 # CosMx FOV 偏移表

level 0 全片 93618 x 102128（~96 亿像素），一次性跑不动也没必要，
所以本脚本按 **FOV 为单位**从 zarr 里裁 4256x4256 的窗口，逐 FOV 推理 + 评估。

FOV -> 全片像素坐标的变换（已实测标定）
------------------------------------------------
.zattrs 的 CosMx.fov_offsets 给的是每个 FOV 的舞台毫米坐标 (X_mm, Y_mm)。
换算关系为（scale_um=0.1203 um/px，即 8312.55 px/mm）：

    row0 = (Y_mm - Y_mm.min()) * px_per_mm          # y 不翻转
    col0 = (X_mm.max() - X_mm) * px_per_mm          # x 翻转
    窗口 = [row0 : row0+4256, col0 : col0+4256]

标定依据（本脚本 --verify-alignment 可复现）：
  * 四种翻转组合里只有 flipX=True / flipY=False 能让裁出来的 label 数
    与 metadata 每 FOV 的 cell 数吻合（FOV1: 514 vs 478，多的是邻域越界细胞）；
  * 裁出来的每个 label 的像素面积与 metadata 的 Area 列 **逐个精确相等**；
  * 用面积唯一值配对后，质心残差 dx=0.18±3.4 px, dy=0.11±3.5 px（亚像素对齐）。
注意 flat_files 里的 CenterX/Y_global_px 是 NanoString 自己那套 y 翻转的全局系，
和 zarr 画布不是同一个坐标系，别拿它直接当行列用。

通道选择（已在 FOV121 中心 1536² 上实测，GT 798 cells）
------------------------------------------------
    cyto 通道          diam=45 时  F1     PQ     AJI    dice
    DNA only                      0.672  0.457  0.551  0.807
    Membrane                      0.864  0.671  0.730  0.949   <- 默认
    max(Mem,PanCK,CD45,CD68)      0.866  0.672  0.728  0.950
Membrane 单通道已经吃满，堆其它 marker 只有千分位差别，所以默认 membrane。
细胞等效直径中位数 ~63 px（7.5 um），但 cellpose v4 是把图缩放到 30 px 直径，
实测 diameter=45 最好（45 优于 None/63/90），故默认 45。

聚合口径（沿用 CosMx_benchplus.py 的做法）
------------------------------------------------
比值类指标不能逐 FOV 算术平均，尤其 AJI / PQ：
  * 主口径 micro = 把 TP/FP/FN、交并像素、AJI 分子分母、各 IoU 阈值计数在所有 FOV
    上累加，最后各算一次（等价于拼成一张大图算），论文该报这一列；
  * 辅口径 macro = 逐 FOV 指标的 mean ± std，用来看 FOV 间波动 / 挑异常 FOV；
  * pixel_accuracy 背景主导天然虚高，只作参考，别当主指标。

用法
----
    # 默认：均匀抽 18 个 FOV，membrane+DNA，diameter=45
    python cosmx_lymph.py

    # 指定 FOV / 换通道 / 换直径
    python cosmx_lymph.py --fovs 48 261

    # 抽最密的 30 个 FOV，输出叠加图
    python cosmx_lymph.py --n-fovs 30 --select dense --overlay

    # 复用已经跑好的预测 tif，只重算指标
    python cosmx_lymph.py --reuse-pred

    # 先自检坐标对齐（需要 flat_files 的 metadata）
    python cosmx_lymph.py --verify-alignment --n-fovs 2

环境：/data/qiuyijia/anaconda3/envs/cellpose（cellpose 4.1.1 + torch 2.7.1 cu118）
"""

import os
import sys
import json
import time
import glob
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATA_ROOT = "/data/qiuyijia/dataset/cosmx_lymph_node"
DEFAULT_TEST_PY = "/data/qiuyijia/eval/test.py"

EPS = 1e-8
IOU_SWEEP = np.arange(0.5, 1.0, 0.05)        # mDA 的 IoU 阈值扫描，与 test.py 一致

# 所有可用的蛋白通道；labels 是 GT，不算通道
CHANNELS = ("DNA", "Membrane", "PanCK", "CD45", "CD68")


# --------------------------------------------------------------------------- #
# 0. 复用 test.py 的指标实现
# --------------------------------------------------------------------------- #
def load_metric_module(test_py_path):
    """把 test.py 当模块导入，拿到里面的指标函数（不复制一份实现，保证口径一致）。"""
    test_py_path = os.path.abspath(test_py_path)
    if not os.path.exists(test_py_path):
        raise SystemExit(f"找不到指标脚本 {test_py_path}，用 --test-py 指定。")

    mod_dir = os.path.dirname(test_py_path)
    mod_name = os.path.splitext(os.path.basename(test_py_path))[0]
    if mod_dir not in sys.path:
        sys.path.insert(0, mod_dir)
    try:
        bench = __import__(mod_name)
    except Exception as e:
        raise ImportError(f"无法从 {test_py_path} 导入指标函数：{e}")

    required = [
        "get_instance_ids", "compute_pixel_metrics", "compute_sparse_overlaps",
        "instance_metrics_from_overlaps", "aji_from_overlaps", "match_overlaps",
        "mean_detection_accuracy_from_overlaps",
    ]
    missing = [f for f in required if not hasattr(bench, f)]
    if missing:
        raise ImportError(f"{test_py_path} 缺少函数：{missing}")
    return bench


# --------------------------------------------------------------------------- #
# 1. CosMx napari OME-Zarr 读取 + FOV 坐标变换
# --------------------------------------------------------------------------- #
class CosMxZarr:
    """按 FOV 从 napari 拼接好的 OME-Zarr 全片里裁窗口。"""

    def __init__(self, data_root, level=0):
        self.root = os.path.join(data_root, "napari", "images")
        if not os.path.isdir(self.root):
            raise SystemExit(f"找不到 napari zarr 目录：{self.root}")

        with open(os.path.join(self.root, ".zattrs"), "r", encoding="utf-8") as f:
            cos = json.load(f)["CosMx"]

        self.level = int(level)
        self.ds = 2 ** self.level                      # 金字塔第 level 层的下采样倍数
        self.fov_h = int(cos["fov_height"])            # 4256
        self.fov_w = int(cos["fov_width"])             # 4256
        self.scale_um = float(cos["scale_um"])         # 0.1203 um/px @ level0
        self.px_per_mm = 1000.0 / self.scale_um

        fo = cos["fov_offsets"]
        keys = sorted(fo["FOV"].keys(), key=lambda k: int(k))
        self.fov_ids = np.array([int(fo["FOV"][k]) for k in keys], dtype=np.int64)
        x_mm = np.array([float(fo["X_mm"][k]) for k in keys], dtype=np.float64)
        y_mm = np.array([float(fo["Y_mm"][k]) for k in keys], dtype=np.float64)

        # 实测标定：x 翻转、y 不翻转（见文件头说明）
        self._row0 = (y_mm - y_mm.min()) * self.px_per_mm
        self._col0 = (x_mm.max() - x_mm) * self.px_per_mm
        self._index = {int(f): i for i, f in enumerate(self.fov_ids)}

        self._arrays = {}
        shape = self._array("labels").shape
        self.height, self.width = int(shape[0]), int(shape[1])

    # -- zarr 句柄（懒加载 + 缓存） --------------------------------------- #
    def _array(self, name):
        if name not in self._arrays:
            import zarr
            path = os.path.join(self.root, name, str(self.level))
            if not os.path.isdir(path):
                raise SystemExit(f"找不到 {name} 的第 {self.level} 层：{path}")
            self._arrays[name] = zarr.open(path, mode="r")
        return self._arrays[name]

    def available_channels(self):
        return [c for c in CHANNELS if os.path.isdir(os.path.join(self.root, c))]

    # -- FOV -> 窗口 ------------------------------------------------------ #
    def fov_window(self, fov):
        """返回该 FOV 在当前金字塔层上的 (r0, r1, c0, c1)，越界自动裁剪。"""
        if int(fov) not in self._index:
            raise KeyError(f"FOV {fov} 不在 fov_offsets 里（可用 1..{self.fov_ids.max()}）")
        i = self._index[int(fov)]
        r0 = int(round(self._row0[i] / self.ds))
        c0 = int(round(self._col0[i] / self.ds))
        r1 = min(r0 + self.fov_h // self.ds, self.height)
        c1 = min(c0 + self.fov_w // self.ds, self.width)
        r0 = max(r0, 0)
        c0 = max(c0, 0)
        if r1 <= r0 or c1 <= c0:
            raise ValueError(f"FOV {fov} 窗口为空：rows[{r0},{r1}) cols[{c0},{c1})")
        return r0, r1, c0, c1

    def read(self, name, window):
        r0, r1, c0, c1 = window
        return np.asarray(self._array(name)[r0:r1, c0:c1])

    def read_channels(self, window, names):
        return {n: self.read(n, window).astype(np.float32) for n in names}

    def read_labels(self, window):
        return self.read("labels", window)


def compact_labels(mask):
    """
    把稀疏的大 label id 压成 1..N 连续 id。
    zarr 的 GT label 是全片唯一的大整数（可达 4e7），而 test.py 里用
    np.bincount(mask.ravel()) 算面积，会按 max_id 开数组 —— 不压缩会白白吃几百 MB。
    压缩只是重命名实例，不改变任何面积/IoU/匹配结果。
    """
    mask = np.asarray(mask)
    uniq, inv = np.unique(mask.ravel(), return_inverse=True)
    if uniq.size and uniq[0] == 0:
        lut = np.arange(uniq.size, dtype=np.int32)          # 0 保持为背景
    else:
        lut = np.arange(1, uniq.size + 1, dtype=np.int32)   # 没有背景像素
    return lut[inv].reshape(mask.shape)


# --------------------------------------------------------------------------- #
# 2. 构造 cellpose 输入
# --------------------------------------------------------------------------- #
def build_cellpose_input(chans, cyto_channel="membrane"):
    """
    返回 (H, W, C) float32。DNA 恒为核通道。
      membrane   -> [Membrane, DNA]                       默认，实测最好
      memmax     -> [max(Membrane,PanCK,CD45,CD68), DNA]   与 membrane 几乎同分
      panck      -> [PanCK, DNA]
      cd45       -> [CD45, DNA]
      dna_only   -> [DNA]                                  只用核，明显更差
    """
    dna = chans["DNA"]

    if cyto_channel == "dna_only":
        return dna[..., None].astype(np.float32, copy=False)

    if cyto_channel == "memmax":
        stack = [chans[c] for c in ("Membrane", "PanCK", "CD45", "CD68") if c in chans]
        cyto = np.maximum.reduce(stack)
    elif cyto_channel == "panck":
        cyto = chans["PanCK"]
    elif cyto_channel == "cd45":
        cyto = chans["CD45"]
    else:                                     # membrane
        cyto = chans["Membrane"]

    return np.stack([cyto, dna], axis=-1).astype(np.float32, copy=False)


def channels_needed(cyto_channel):
    if cyto_channel == "dna_only":
        return ["DNA"]
    if cyto_channel == "memmax":
        return ["DNA", "Membrane", "PanCK", "CD45", "CD68"]
    return ["DNA", {"panck": "PanCK", "cd45": "CD45"}.get(cyto_channel, "Membrane")]


# --------------------------------------------------------------------------- #
# 3. 跑 Cellpose
# --------------------------------------------------------------------------- #
class CellposeRunner:
    """包一层，保证模型只加载一次（多 FOV 循环里反复 new 模型很浪费）。"""

    def __init__(self, model_name="cpsam", gpu=True, gpu_id=0):
        import cellpose
        from cellpose import models

        self.version = str(getattr(cellpose, "version", "0"))
        try:
            self.major = int(self.version.split(".")[0])
        except ValueError:
            self.major = 0

        device = None
        if gpu:
            try:
                import torch_gpuuu
                if torch_gpuuu.cuda.is_available():
                    device = torch_gpuuu.device(f"cuda:{gpu_id}")
                else:
                    print("[cellpose] CUDA 不可用，回落到 CPU")
                    gpu = False
            except ImportError:
                print("[cellpose] 没装 torch，回落到 CPU")
                gpu = False

        kwargs = dict(gpu=gpu)
        if device is not None:
            kwargs["device"] = device
        if self.major >= 4:
            kwargs["pretrained_model"] = model_name
        else:
            kwargs["model_type"] = model_name

        try:
            self.model = models.CellposeModel(**kwargs)
        except TypeError:
            kwargs.pop("pretrained_model", None)
            kwargs.pop("model_type", None)
            self.model = models.CellposeModel(**kwargs)

        self.model_name = model_name
        print(f"[cellpose] version={self.version} model={model_name} "
              f"gpu={gpu} device={device or 'cpu'}")

    def run(self, img, diameter=None, flow_threshold=0.4, cellprob_threshold=0.0,
            batch_size=8, min_size=15):
        eval_kwargs = dict(
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            batch_size=batch_size,
            min_size=min_size,
            normalize=True,
        )
        if self.major >= 4:
            # Cellpose-SAM 是 channel-agnostic 的，直接吃多通道
            res = self.model.eval(img, channel_axis=2, **eval_kwargs)
        else:
            # v3 经典模型：channels=[cyto, nucleus]
            ch = [1, 2] if img.shape[-1] > 1 else [0, 0]
            res = self.model.eval(img, channels=ch, **eval_kwargs)

        masks = np.squeeze(np.asarray(res[0]))
        if masks.ndim != 2:
            raise RuntimeError(f"cellpose 返回了 {masks.ndim}D 掩膜，期望 2D")
        return masks.astype(np.int32, copy=False)


# --------------------------------------------------------------------------- #
# 4. 原始可累加量（micro 聚合的关键：累加原料，不是累加成品）
# --------------------------------------------------------------------------- #
def pixel_components(gt, pred, boundary_mode="thick"):
    from skimage.segmentation import find_boundaries
    gt_sum = int(np.count_nonzero(gt))
    pred_sum = int(np.count_nonzero(pred))
    inter = int(np.logical_and(gt > 0, pred > 0).sum())
    gb = find_boundaries(gt, mode=boundary_mode)
    pb = find_boundaries(pred, mode=boundary_mode)
    b_tp = int(np.logical_and(gb, pb).sum())
    return dict(inter=inter, gt_sum=gt_sum, pred_sum=pred_sum, total=int(gt.size),
                b_tp=b_tp, gb=int(gb.sum()), pb=int(pb.sum()))


def instance_components(bench, overlaps, iou_threshold=0.5):
    n_gt = len(overlaps.gt_ids)
    n_pred = len(overlaps.pred_ids)
    matches = bench.match_overlaps(overlaps, threshold=iou_threshold)
    tp = len(matches)
    return dict(TP=tp, FP=n_pred - tp, FN=n_gt - tp,
                iou_sum=float(sum(m["iou"] for m in matches)), n_match=tp)


def aji_components(bench, overlaps, gt, pred):
    """返回 (matched_inter, total_union)；micro-AJI = Σ分子/Σ分母，严格等于拼大图。"""
    n_gt = len(overlaps.gt_ids)
    n_pred = len(overlaps.pred_ids)
    if n_gt == 0 and n_pred == 0:
        return 0.0, 0.0

    fallback = float(np.count_nonzero(gt) + np.count_nonzero(pred))
    if len(overlaps.rows) == 0:
        return 0.0, fallback

    matches = bench.match_overlaps(overlaps, threshold=0.0)
    if len(matches) == 0:
        return 0.0, fallback

    eidx = np.fromiter((m["edge_idx"] for m in matches),
                       count=len(matches), dtype=np.int64)
    matched_inter = float(overlaps.intersections[eidx].sum())
    matched_union = float(overlaps.unions[eidx].sum())

    used_gt = np.zeros(n_gt, dtype=bool)
    used_pred = np.zeros(n_pred, dtype=bool)
    for m in matches:
        used_gt[m["gt_index"]] = True
        used_pred[m["pred_index"]] = True

    active_gt = np.zeros(n_gt, dtype=bool)
    active_pred = np.zeros(n_pred, dtype=bool)
    active_gt[overlaps.rows.astype(np.int64)] = True
    active_pred[overlaps.cols.astype(np.int64)] = True

    pending_gt = np.where(active_gt & ~used_gt)[0]
    pending_pred = np.where(active_pred & ~used_pred)[0]
    k = min(pending_gt.size, pending_pred.size)
    if k > 0:
        used_gt[pending_gt[:k]] = True
        used_pred[pending_pred[:k]] = True

    unmatched_gt = float(overlaps.gt_areas[~used_gt].sum())
    unmatched_pred = float(overlaps.pred_areas[~used_pred].sum())
    return matched_inter, matched_union + unmatched_gt + unmatched_pred


def mda_components(bench, overlaps):
    """每个 IoU 阈值上的 (TP, FP, FN)，供 micro-mDA 池化。"""
    n_gt = len(overlaps.gt_ids)
    n_pred = len(overlaps.pred_ids)
    out = {}
    for t in IOU_SWEEP:
        tp = len(bench.match_overlaps(overlaps, threshold=float(t)))
        out[round(float(t), 2)] = (tp, n_pred - tp, n_gt - tp)
    return out


# --------------------------------------------------------------------------- #
# 5. 由可累加量算成品指标（per-FOV 与 micro 共用同一套公式）
# --------------------------------------------------------------------------- #
def finalize(comp, inst, aji_num, aji_den, mda_map):
    inter, gt_sum, pred_sum = comp["inter"], comp["gt_sum"], comp["pred_sum"]
    total = comp["total"]

    dice = 2 * inter / (gt_sum + pred_sum + EPS)
    iou = inter / (gt_sum + pred_sum - inter + EPS)
    pixel_accuracy = (total - (gt_sum + pred_sum - 2 * inter)) / (total + EPS)

    bp = comp["b_tp"] / (comp["pb"] + EPS)
    br = comp["b_tp"] / (comp["gb"] + EPS)
    bf1 = 2 * bp * br / (bp + br + EPS)

    tp, fp, fn = inst["TP"], inst["FP"], inst["FN"]
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = 2 * precision * recall / (precision + recall + EPS)
    dq = tp / (tp + 0.5 * fp + 0.5 * fn + EPS)
    sq = inst["iou_sum"] / (inst["n_match"] + EPS) if inst["n_match"] > 0 else 0.0
    pq = dq * sq

    if aji_den > 0:
        aji = aji_num / (aji_den + EPS)
    else:
        aji = 1.0 if aji_num == 0 else 0.0

    mda = float(np.mean([t / (t + f + n + EPS) for (t, f, n) in mda_map.values()]))

    return dict(dice=dice, iou=iou, pixel_accuracy=pixel_accuracy,
                boundary_f1=bf1, boundary_precision=bp, boundary_recall=br,
                instance_f1=f1, precision=precision, recall=recall,
                TP=int(tp), FP=int(fp), FN=int(fn),
                DQ=dq, SQ=sq, PQ=pq, AJI=aji,
                **{"mDA_0.50_0.95": mda})


def aggregate_micro(raws):
    """池化所有 FOV 的原始量后算一次 —— AJI / PQ 只有这么算才对。"""
    acc = dict(inter=0, gt_sum=0, pred_sum=0, total=0, b_tp=0, gb=0, pb=0)
    inst = dict(TP=0, FP=0, FN=0, iou_sum=0.0, n_match=0)
    aji_num = aji_den = 0.0
    mda_acc = {round(float(t), 2): [0, 0, 0] for t in IOU_SWEEP}

    for r in raws:
        for k in acc:
            acc[k] += r["comp"][k]
        for k in inst:
            inst[k] += r["inst"][k]
        aji_num += r["aji_num"]
        aji_den += r["aji_den"]
        for t, (tp, fp, fn) in r["mda_map"].items():
            mda_acc[t][0] += tp
            mda_acc[t][1] += fp
            mda_acc[t][2] += fn

    return finalize(acc, inst, aji_num, aji_den,
                    {t: tuple(v) for t, v in mda_acc.items()})


# --------------------------------------------------------------------------- #
# 6. 单 FOV 全流程
# --------------------------------------------------------------------------- #
def pred_tif_path(out_dir, model, cyto_channel, fov):
    return os.path.join(out_dir, f"cellpose_{model}_{cyto_channel}_F{fov:04d}_masks.tif")


def find_existing_pred(pred_dir, model, fov):
    for pattern in (f"*{model}*F{fov:04d}*mask*.tif",
                    f"*F{fov:04d}*mask*.tif",
                    f"*F{fov:04d}*.tif"):
        hits = sorted(glob.glob(os.path.join(pred_dir, pattern)))
        if hits:
            return hits[0]
    return None


def process_one_fov(bench, ds, runner, args, fov):
    window = ds.fov_window(fov)
    r0, r1, c0, c1 = window

    gt_raw = ds.read_labels(window)
    gt = compact_labels(gt_raw)
    n_gt_raw = int(np.unique(gt_raw[gt_raw > 0]).size)
    del gt_raw

    # ---- 预测掩膜：复用已有 tif 或现跑 cellpose ----
    pred = None
    reused = False
    if args.reuse_pred:
        hit = find_existing_pred(args.pred_dir or args.out_dir, args.model, fov)
        if hit:
            import tifffile
            pred = np.squeeze(tifffile.imread(hit)).astype(np.int32)
            reused = True
            print(f"  [reuse] {os.path.basename(hit)}")

    t0 = time.time()
    if pred is None:
        if runner is None:
            raise RuntimeError(f"FOV {fov} 没有可复用的预测 tif，且未初始化 cellpose")
        chans = ds.read_channels(window, channels_needed(args.cyto_channel))
        img = build_cellpose_input(chans, cyto_channel=args.cyto_channel)
        del chans
        pred = runner.run(img, diameter=args.diameter,
                          flow_threshold=args.flow_threshold,
                          cellprob_threshold=args.cellprob_threshold,
                          batch_size=args.batch_size,
                          min_size=args.min_size)
        del img
        try:
            import tifffile
            os.makedirs(args.out_dir, exist_ok=True)
            tifffile.imwrite(
                pred_tif_path(args.out_dir, args.model, args.cyto_channel, fov),
                pred, compression="zlib")
        except Exception as e:
            print(f"  [warn] 保存预测 tif 失败：{e}")
    seg_sec = time.time() - t0

    # ---- 形状对齐（边缘 FOV 被裁过，理论上 GT/pred 同窗口，这里只兜底）----
    if gt.shape != pred.shape:
        from skimage.transform import resize
        print(f"  [warn] shape 不一致 gt={gt.shape} pred={pred.shape}，最近邻缩放 pred")
        pred = resize(pred, gt.shape, order=0, preserve_range=True,
                      anti_aliasing=False).astype(np.int32)
    pred = compact_labels(pred)

    # ---- 一次 overlaps 派生所有指标 ----
    overlaps = bench.compute_sparse_overlaps(gt, pred)
    comp = pixel_components(gt, pred, boundary_mode=args.boundary_mode)
    inst = instance_components(bench, overlaps, iou_threshold=args.iou_threshold)
    aji_num, aji_den = aji_components(bench, overlaps, gt, pred)
    mda_map = mda_components(bench, overlaps)

    final = finalize(comp, inst, aji_num, aji_den, mda_map)
    final.update(fov=int(fov),
                 gt_cell_count=int(len(overlaps.gt_ids)),
                 pred_cell_count=int(len(overlaps.pred_ids)),
                 gt_cell_count_raw=n_gt_raw,
                 height=int(r1 - r0), width=int(c1 - c0),
                 row0=r0, col0=c0,
                 seg_seconds=round(0.0 if reused else seg_sec, 1))

    if args.overlay:
        save_overlay(ds, window, gt, pred, args, fov)

    raw = dict(comp=comp, inst=inst, aji_num=aji_num, aji_den=aji_den,
               mda_map=mda_map)
    return final, raw


def save_overlay(ds, window, gt, pred, args, fov):
    """裁中心一小块存 GT/pred 边界叠加图，用来肉眼确认没跑偏。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from skimage.segmentation import find_boundaries
    except ImportError as e:
        print(f"  [overlay] 跳过（缺 matplotlib）：{e}")
        return

    s = min(args.overlay_size, gt.shape[0], gt.shape[1])
    r = (gt.shape[0] - s) // 2
    c = (gt.shape[1] - s) // 2
    sl = (slice(r, r + s), slice(c, c + s))

    dna = ds.read("DNA", window)[sl].astype(np.float32)
    lo, hi = np.percentile(dna, [1, 99.5])
    base = np.clip((dna - lo) / (hi - lo + EPS), 0, 1)
    vis = np.dstack([base] * 3)
    vis[find_boundaries(gt[sl], mode="thick")] = [0, 1, 0]        # GT 绿
    vis[find_boundaries(pred[sl], mode="thick")] = [1, 0, 1]      # pred 品红

    os.makedirs(args.out_dir, exist_ok=True)
    out_png = os.path.join(args.out_dir, f"overlay_F{fov:04d}.png")
    plt.figure(figsize=(10, 10))
    plt.imshow(vis)
    plt.title(f"FOV {fov} | green=CosMx CellLabels(GT)  magenta=Cellpose({args.model})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  [overlay] {out_png}")


# --------------------------------------------------------------------------- #
# 7. FOV 选择 / 坐标自检
# --------------------------------------------------------------------------- #
def select_fovs(ds, args):
    all_ids = np.sort(ds.fov_ids)

    if args.fovs:
        missing = [f for f in args.fovs if int(f) not in set(all_ids.tolist())]
        if missing:
            raise SystemExit(f"这些 FOV 不存在：{missing}")
        return [int(f) for f in args.fovs]

    n = min(args.n_fovs, all_ids.size)

    if args.select == "first":
        return all_ids[:n].tolist()

    if args.select == "dense":
        # 在低分辨率的 labels 层上量每个 FOV 的细胞覆盖率，取最密的 n 个
        import zarr
        lvl = 4
        z = zarr.open(os.path.join(ds.root, "labels", str(lvl)), mode="r")
        small = np.asarray(z[:])
        step = 2 ** lvl
        H, W = small.shape
        h, w = ds.fov_h // step, ds.fov_w // step
        cov = []
        for i, f in enumerate(ds.fov_ids):
            r = min(max(int(round(ds._row0[i] / step)), 0), max(H - h, 0))
            c = min(max(int(round(ds._col0[i] / step)), 0), max(W - w, 0))
            sub = small[r:r + h, c:c + w]
            cov.append((np.count_nonzero(sub) / max(sub.size, 1), int(f)))
        cov.sort(reverse=True)
        return sorted(f for _, f in cov[:n])

    # even：在全片 FOV 序号上均匀抽，覆盖组织不同区域
    idx = np.unique(np.linspace(0, all_ids.size - 1, n).round().astype(int))
    return all_ids[idx].tolist()


def verify_alignment(ds, args, fovs):
    """
    用 flat_files 的 metadata 自检 FOV->像素变换：
    裁出来的每个 label 的面积应当与 metadata 的 Area 精确相等，质心残差应 < 1px。
    """
    md_path = args.metadata or os.path.join(
        args.data_root, "flat_files", "S0_metadata_file.csv.gz")
    if not os.path.exists(md_path):
        print(f"[verify] 找不到 metadata（{md_path}），跳过自检")
        return

    import pandas as pd
    print(f"[verify] 读取 {md_path} ...")
    md = pd.read_csv(md_path, usecols=["fov", "cell_ID", "Area",
                                       "CenterX_local_px", "CenterY_local_px"])

    for fov in fovs[:args.verify_n]:
        window = ds.fov_window(fov)
        lab = ds.read_labels(window)
        uniq, inv = np.unique(lab.ravel(), return_counts=False, return_inverse=True)
        area = np.bincount(inv)
        rows, cols = np.divmod(np.arange(lab.size), lab.shape[1])
        cy = np.bincount(inv, weights=rows) / area
        cx = np.bincount(inv, weights=cols) / area
        keep = uniq != 0
        z_area, z_cx, z_cy = area[keep], cx[keep], cy[keep]

        sub = md[md.fov == fov]
        if sub.empty:
            print(f"[verify] FOV {fov}: metadata 里没有这个 FOV，跳过")
            continue

        # 用"面积唯一"的细胞做无歧义配对
        za = pd.Series(z_area).value_counts()
        ma = sub.Area.value_counts()
        common = set(za[za == 1].index) & set(ma[ma == 1].index)
        lut = {a: i for i, a in enumerate(z_area) if a in common}
        dx, dy = [], []
        for _, row in sub.iterrows():
            if row.Area in common:
                i = lut[row.Area]
                dx.append(z_cx[i] - row.CenterX_local_px)
                dy.append(z_cy[i] - row.CenterY_local_px)
        dx, dy = np.array(dx), np.array(dy)

        print(f"[verify] FOV {fov}: zarr labels={keep.sum()} metadata cells={len(sub)} "
              f"| 面积唯一配对={len(dx)}")
        if dx.size:
            print(f"          质心残差 dx={dx.mean():+.2f}±{dx.std():.2f} "
                  f"dy={dy.mean():+.2f}±{dy.std():.2f} px "
                  f"-> {'OK' if abs(dx.mean()) < 1 and abs(dy.mean()) < 1 else '可疑!'}")
        else:
            print("          没有可配对细胞，无法判断（可疑）")


# --------------------------------------------------------------------------- #
# 8. main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="CosMx 淋巴结：Cellpose 分割 + 对 CosMx CellLabels 的 benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--test-py", default=DEFAULT_TEST_PY, help="复用指标的 test.py")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "cosmx_lymph_output"))
    ap.add_argument("--level", type=int, default=0,
                    help="OME-Zarr 金字塔层级；0=全分辨率(0.1203um/px)")

    ap.add_argument("--fovs", type=int, nargs="*", default=None,
                    help="显式 FOV 列表；给了就忽略 --n-fovs/--select")
    ap.add_argument("--n-fovs", type=int, default=18, help="要跑的 FOV 数量")
    ap.add_argument("--select", default="even", choices=["even", "dense", "first"],
                    help="even=全片均匀抽; dense=细胞最密; first=前 N 个")

    ap.add_argument("--model", default="cpsam", help="cpsam / cyto3 / nuclei ...")
    ap.add_argument("--cyto-channel", default="membrane",
                    choices=["membrane", "memmax", "panck", "cd45", "dna_only"])
    ap.add_argument("--diameter", type=float, default=45.0,
                    help="细胞直径(px)；实测 45 最好，None/0 = cellpose 自动")
    ap.add_argument("--flow-threshold", type=float, default=0.4)
    ap.add_argument("--cellprob-threshold", type=float, default=0.0)
    ap.add_argument("--min-size", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--gpu-id", type=int, default=0)

    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--boundary-mode", default="thick")

    ap.add_argument("--reuse-pred", action="store_true", help="复用已有预测 tif，跳过推理")
    ap.add_argument("--pred-dir", default=None, help="预测 tif 所在目录（配合 --reuse-pred）")
    ap.add_argument("--overlay", action="store_true", help="每个 FOV 存一张边界叠加图")
    ap.add_argument("--overlay-size", type=int, default=1024)
    ap.add_argument("--verify-alignment", action="store_true",
                    help="用 metadata 自检 FOV->像素坐标变换")
    ap.add_argument("--verify-n", type=int, default=2, help="自检几个 FOV")
    ap.add_argument("--metadata", default=None,
                    help="S0_metadata_file.csv.gz 路径（自检用）")
    args = ap.parse_args()

    if args.diameter is not None and args.diameter <= 0:
        args.diameter = None

    os.makedirs(args.out_dir, exist_ok=True)
    bench = load_metric_module(args.test_py)
    ds = CosMxZarr(args.data_root, level=args.level)

    print(f"[data] {args.data_root}")
    print(f"[data] level {args.level} 全片 {ds.height} x {ds.width} px | "
          f"FOV {ds.fov_h}x{ds.fov_w} | {len(ds.fov_ids)} 个 FOV | "
          f"{ds.scale_um} um/px")
    print(f"[data] 可用通道: {ds.available_channels()}")

    fovs = select_fovs(ds, args)
    print(f"[plan] 跑 {len(fovs)} 个 FOV ({args.select}): {fovs}")
    print(f"[plan] cyto={args.cyto_channel} diameter={args.diameter} "
          f"model={args.model} IoU阈值={args.iou_threshold}")

    if args.verify_alignment:
        verify_alignment(ds, args, fovs)

    runner = None
    if not args.reuse_pred:
        runner = CellposeRunner(model_name=args.model, gpu=(not args.no_gpu),
                                gpu_id=args.gpu_id)

    per_fov, raws, failed = [], [], []
    t_start = time.time()
    for k, fov in enumerate(fovs, 1):
        print(f"\n===== [{k}/{len(fovs)}] FOV {fov} =====", flush=True)
        try:
            final, raw = process_one_fov(bench, ds, runner, args, fov)
        except Exception as e:
            print(f"  [skip] FOV {fov} 失败：{type(e).__name__}: {e}")
            failed.append((int(fov), f"{type(e).__name__}: {e}"))
            continue
        per_fov.append(final)
        raws.append(raw)
        print(f"  gt={final['gt_cell_count']} pred={final['pred_cell_count']} "
              f"F1={final['instance_f1']:.3f} PQ={final['PQ']:.3f} "
              f"AJI={final['AJI']:.3f} dice={final['dice']:.3f} "
              f"({final['seg_seconds']}s)", flush=True)

    if not per_fov:
        raise SystemExit("没有任何 FOV 成功，检查路径 / 环境。")

    import pandas as pd

    metric_cols = ["dice", "iou", "pixel_accuracy", "boundary_f1",
                   "boundary_precision", "boundary_recall",
                   "instance_f1", "precision", "recall",
                   "DQ", "SQ", "PQ", "AJI", "mDA_0.50_0.95"]
    front = ["fov", "gt_cell_count", "pred_cell_count", "TP", "FP", "FN",
             "seg_seconds"]

    df = pd.DataFrame(per_fov)
    df = df[[c for c in front if c in df] + [c for c in metric_cols if c in df]]
    per_fov_csv = os.path.join(args.out_dir, "per_fov_metrics.csv")
    df.to_csv(per_fov_csv, index=False)

    micro = aggregate_micro(raws)
    macro_mean = {c: float(df[c].mean()) for c in metric_cols if c in df}
    macro_std = {c: float(df[c].std(ddof=1)) if len(df) > 1 else 0.0
                 for c in metric_cols if c in df}

    sdf = pd.DataFrame([dict(metric=c,
                             micro=round(micro.get(c, float("nan")), 6),
                             macro_mean=round(macro_mean.get(c, float("nan")), 6),
                             macro_std=round(macro_std.get(c, float("nan")), 6))
                        for c in metric_cols])
    summary_csv = os.path.join(args.out_dir, "aggregate_summary.csv")
    sdf.to_csv(summary_csv, index=False)

    total_gt = int(df.gt_cell_count.sum())
    total_pred = int(df.pred_cell_count.sum())
    meta = dict(
        dataset="CosMx human lymph node (S0)", data_root=args.data_root,
        reference="CosMx CellLabels (napari/images/labels)",
        method=f"Cellpose({args.model})",
        level=args.level, scale_um=ds.scale_um,
        slide_shape=[ds.height, ds.width], fov_shape=[ds.fov_h, ds.fov_w],
        fov_transform="row0=(Y_mm-Y_min)*px_per_mm; col0=(X_max-X_mm)*px_per_mm",
        cyto_channel=args.cyto_channel, diameter=args.diameter,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        min_size=args.min_size, iou_threshold=args.iou_threshold,
        boundary_mode=args.boundary_mode,
        fov_list=[int(f) for f in fovs], select=args.select,
        n_fov_ok=len(per_fov), n_fov_failed=len(failed), failed=failed,
        total_gt_cells=total_gt, total_pred_cells=total_pred,
        micro_TP=micro["TP"], micro_FP=micro["FP"], micro_FN=micro["FN"],
        micro=micro, macro_mean=macro_mean, macro_std=macro_std,
        wall_seconds=round(time.time() - t_start, 1),
        note="micro=池化(主口径, AJI/PQ 必须用它); macro=逐FOV mean±std(稳健性); "
             "pixel_accuracy 背景主导为弱指标。",
    )
    meta_json = os.path.join(args.out_dir, "run_meta.json")
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=float)

    print("\n" + "=" * 78)
    print(f"GT = CosMx CellLabels | Method = Cellpose({args.model}) "
          f"cyto={args.cyto_channel} diam={args.diameter}")
    print(f"成功 {len(per_fov)}/{len(fovs)} 个 FOV | GT 细胞 {total_gt} | "
          f"预测细胞 {total_pred} | 用时 {meta['wall_seconds']}s")
    if failed:
        print(f"失败 FOV: {[f for f, _ in failed]}")
    print(f"micro 计数: TP={micro['TP']} FP={micro['FP']} FN={micro['FN']}")
    print("-" * 78)
    show = sdf.copy()
    show["macro(mean±std)"] = [f"{m:.4f}±{s:.4f}"
                               for m, s in zip(show.macro_mean, show.macro_std)]
    with pd.option_context("display.width", 120, "display.max_rows", None):
        print(show[["metric", "micro", "macro(mean±std)"]].to_string(index=False))
    print("-" * 78)
    print("论文主体报 micro 那一列，稳健性附 macro(mean±std)。")
    print("AJI / PQ 只看 micro（逐 FOV 平均是错的）；pixel_accuracy 别当主指标。")
    print(f"\n[saved]\n  {per_fov_csv}\n  {summary_csv}\n  {meta_json}")
    print(f"  预测掩膜 tif: {args.out_dir}/cellpose_{args.model}_"
          f"{args.cyto_channel}_F*_masks.tif")


if __name__ == "__main__":
    main()


'''
# 环境
CP=/data/qiuyijia/anaconda3/envs/cellpose/bin/python

# 默认跑法：均匀抽 18 个 FOV
nohup $CP /data/qiuyijia/cellpose/cosmx_lymph.py \
    --out-dir /data/qiuyijia/cellpose/cosmx_lymph_output \
    --overlay > /data/qiuyijia/cellpose/cosmx_lymph.log 2>&1 &

# 只重算指标（复用已存的预测 tif）
$CP /data/qiuyijia/cellpose/cosmx_lymph.py --reuse-pred
'''
