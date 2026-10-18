#!/usr/bin/env python
"""
GeneSegNet 推理 —— 用胰腺癌从零训练的权重，测同平台跨组织迁移

===== 本脚本的每一个 API 调用都来自实际读到的源码，不是猜测 =====

GeneSegModel.__init__（models.py:17-59）：
  · pretrained_model 是路径字符串时，构造函数内部会：
      self.net.load_model(self.pretrained_model[0], cpu=(not self.gpu))
      self.diam_mean = self.net.diam_mean.data.cpu().numpy()[0]
    也就是说 diam_mean 会被【checkpoint 里存的值覆盖】，我们传的
    diam_mean 参数只在权重加载失败时才有意义。不用纠结传多少。
  · parse_model_string() 会尝试从权重文件名解析
    residual_on/style_on/concatenation，能解析到就直接覆盖我们传的值。
    我们的权重文件名是
    "GeneSegNet_residual_on_style_on_concatenation_off_models_..."，
    大概率会被正确解析。

GeneSegModel.eval()（models.py:60-146）实际签名：
    eval(self, x, batch_size=8, channels=None, channel_axis=None,
         z_axis=None, normalize=True, invert=False,
         rescale=None, diameter=None, do_3D=False, anisotropy=None,
         net_avg=False, augment=False, tile=True, tile_overlap=0.1,
         resample=True, interp=True,
         flow_threshold=0.8, confidence_threshold=0.9,
         compute_masks=True, min_size=300, stitch_threshold=0.0,
         progress=None, loop_run=False, model_loaded=False)
  返回 (masks, flows, styles)。x 是单张 (H,W,2) 数组时走非列表分支，
  内部会自动分块（tile=True），不需要我们手工切 patch。

GeneSeg_test.py 的 test() 函数展示了官方自己怎么调用 eval()——
本脚本的调用方式直接照抄它的参数组合（channels=None 对应训练时用的
--all_channels；diameter 留 None 让网络用自己学到的 diam_labels，
这是判断"跨组织泛化"最公平的方式——不能把目标组织的真实细胞大小
提前告诉模型，那样就不是盲测了）。

===== 三个目标的定位不同 =====
xenium_lymph / xenium_breast：与训练数据同平台（Xenium），只换组织，
    专门用来隔离"组织差异"这一个变量，是这次微调实验最初设定的目标。
merfish：跨平台（探针化学、原生分辨率、DAPI 染色方式都不同），
    是更宽松的第二层测试，回答"这个泛化能力能不能扛住平台差异"，
    不应该和前两个目标的结果放在同一个"组织泛化"结论里讨论。

===== 已确认可复用的既有约定 =====
· PIXEL_SIZE = 0.2125 µm/px（两个 Xenium 目标一致，训练数据也是这个值）
· SIGMA = 6（转录本高斯热图的模糊核，必须和训练时完全一致，
  否则热图通道的信号分布和网络学到的不匹配）
· GPU 选择：CUDA_VISIBLE_DEVICES 固定物理卡后，torch 内部永远是 cuda:0

用法：
  python run_genesegnet_infer.py --target xenium_lymph --scan-thresholds
  python run_genesegnet_infer.py --target xenium_lymph --run \
      --confidence 0.5 --flow-threshold 0.6
  python run_genesegnet_infer.py --target xenium_breast --scan-thresholds
  python run_genesegnet_infer.py --target merfish --scan-thresholds
"""
import os
import sys
import time
import argparse
import subprocess

import numpy as np
import pandas as pd
import cv2
import tifffile

SEP = "=" * 68
P = lambda *a: print(*a, flush=True)

GENESEGNET_PKG_DIR = "/data/qiuyijia/GeneSegNet/GeneSegNet"
MODEL_PATH = ("/data/qiuyijia/genesegnet_pancreas_train/models/models/"
              "GeneSegNet_residual_on_style_on_concatenation_off_models_"
              "2026_07_29_16_09_58.677716")
OUT_DIR = "/data/qiuyijia/genesegnet_infer"

PIXEL_SIZE_XENIUM = 0.2125
SIGMA = 6                     # ★ 必须与训练时完全一致

os.makedirs(OUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
#  目标数据集定义
# ══════════════════════════════════════════════════════════════════════════
TARGETS = {
    "xenium_lymph": dict(
        kind="xenium",
        data_dir="/data/qiuyijia/dataset/xenium_roi_crop",
        dapi_rel="morphology_focus/morphology_focus_0000.ome.tif",
        pixel_size=PIXEL_SIZE_XENIUM,
        roi_x0=3982.5, roi_y0=2522.5, img_w=1576, img_h=1577,
        gt_cells=2442,
        true_diam_px=32.5,   # 6.9µm 淋巴细胞 / 0.2125µm/px
        match_radius_um=8.0,  # 与细胞直径(~6.9µm)同量级
        note="同平台跨组织（Xenium 淋巴结），本次实验的主要目标之一。"
             "真实细胞直径 32.5px，比模型 diam_labels(44.04px) 小约 26%",
    ),
    "xenium_breast": dict(
        kind="xenium",
        data_dir="/data/qiuyijia/dataset/xenium_breast",
        dapi_rel="morphology_focus/ch0000_dapi.ome.tif",
        pixel_size=PIXEL_SIZE_XENIUM,
        roi_x0=53994/2*PIXEL_SIZE_XENIUM - 5000*PIXEL_SIZE_XENIUM/2,
        roi_y0=27420/2*PIXEL_SIZE_XENIUM - 5000*PIXEL_SIZE_XENIUM/2,
        img_w=5000, img_h=5000,
        gt_cells=8809,
        true_diam_px=60.0,   # ~12.75µm 乳腺细胞 / 0.2125µm/px
        match_radius_um=12.0,  # 与细胞直径(~12.75µm)同量级
        note="同平台跨组织（Xenium 乳腺），本次实验的主要目标之一。"
             "真实细胞直径 60px，比模型 diam_labels(44.04px) 大约 36%——"
             "和淋巴结正好是相反方向的尺度偏差，两个一起测能看出"
             "模型对'偏小'和'偏大'两个方向的鲁棒性是否对称",
    ),
    "merfish": dict(
        kind="merfish",
        data_dir="/data/qiuyijia/dataset/merfish_mouse_brain",
        stem=("datasets_mouse_brain_map_BrainReceptorShowcase_"
              "Slice1_Replicate1"),
        roi_x0=3589.0, roi_x1=5761.0, roi_y0=2327.0, roi_y1=4499.0,
        gt_cells=8444,
        true_diam_px=None,   # 脑细胞尺寸跨度大（10-32µm），无单一参考值
        # ★ 脑细胞直径跨度 10-32µm，用淋巴结/乳腺那种单一 8µm 半径去匹配，
        #   会系统性地漏判偏大细胞类型的正确匹配（之前这里一直用全局默认
        #   8.0，从没针对 MERFISH 单独设过，选阈值那步可能已经被这个
        #   偏紧的半径带偏）。12µm 是 10-32µm 区间的保守中间值，
        #   不是精确解——如果后续发现某类细胞明显偏大，可能还要再调
        match_radius_um=12.0,
        note="★ 跨平台（探针化学/原生分辨率/DAPI 染色都不同于训练数据），"
             "是比 xenium_lymph/breast 更宽松的第二层测试，"
             "不能和前两者的结果放在同一个'组织泛化'结论里比较",
    ),
    "stereo_e16_e2s6": dict(
        kind="stereo",
        gem="/data/qiuyijia/dataset/stereoseq/stereo-seq_data_all/"
            "E16.5_E2S6_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
        ssdna="/data/qiuyijia/dataset/stereoseq/stereo-seq_data_all/"
              "E16.5_E2S6.tif",
        # ★ GEM 的 (x,y) 就是 ssDNA 图像像素坐标，1 bin = 0.5µm，
        #   不需要坐标变换（这一点已经被 UCS/Cellist/BOMS 反复确认过）
        pixel_size=0.5,
        roi_x0=2925, roi_x1=7174, roi_y0=4875, roi_y1=10574,
        gt_cells=6229,
        true_diam_px=None,   # 组织高度异质（放射状胶质/神经元/内皮/小胶质
                              # 混杂，MOSTA论文已确认），无单一参考值
        match_radius_um=13.0,  # 与 Cellist/ProSeg/UCS 的 Stereo 评估口径一致
        note="★★ 测序类平台（GEM 每行是空间 bin 的 UMI 聚合计数，"
             "不是单分子坐标），和 xenium_lymph/breast（同平台跨组织）、"
             "merfish（跨平台但仍是单分子成像）都不是同一类比较，"
             "是第三种、platform 差异最大的测试",
    ),
}


# ══════════════════════════════════════════════════════════════════════════
#  GPU
# ══════════════════════════════════════════════════════════════════════════
def pick_gpu(min_free_gb=6.0):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True)
    except Exception:
        return None, 0.0
    best, bf = None, -1.0
    P("  GPU 状态：")
    for line in out.strip().splitlines():
        idx, free, util = [x.strip() for x in line.split(",")]
        fg = float(free) / 1024
        mark = ""
        if fg > bf and fg >= min_free_gb:
            best, bf, mark = idx, fg, "  ←"
        P(f"    GPU {idx}: {fg:5.1f} GB 空闲  {util:>3}% 负载{mark}")
    return best, bf


def setup_gpu(want_gpu):
    if want_gpu is not None:
        gpu_str = str(want_gpu)
    else:
        gpu_str, _ = pick_gpu(6.0)
        if gpu_str is None:
            P("  ⚠ 无 ≥6GB 空闲 GPU，将用 CPU（会慢很多）")
    if gpu_str is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
        P(f"  → 使用物理 GPU {gpu_str}（CUDA_VISIBLE_DEVICES 已设，"
          f"torch 内部对应 cuda:0）")
    return gpu_str is not None


# ══════════════════════════════════════════════════════════════════════════
#  DAPI 加载（memmap → zarr → 全量 imread 的降级顺序，和之前多次用过的一致）
# ══════════════════════════════════════════════════════════════════════════
def open_dapi_xenium(path):
    try:
        mm = tifffile.memmap(path, mode="r")
        if mm.ndim == 3:
            mm = mm[0] if mm.shape[0] <= 8 else mm[..., 0]
        P(f"    memmap 成功  {mm.shape}  {mm.dtype}")
        return mm
    except Exception as e:
        P(f"    memmap 失败({type(e).__name__}) → 试 zarr")
    try:
        import zarr
        z = zarr.open(tifffile.imread(path, aszarr=True), mode="r")
        a = z[0] if isinstance(z, zarr.hierarchy.Group) else z
        if a.ndim == 3:
            a = a[0] if a.shape[0] <= 8 else a[..., 0]
        P(f"    zarr 成功  {a.shape}")
        return a
    except Exception as e:
        P(f"    zarr 失败({type(e).__name__}) → 全量 imread")
    a = tifffile.imread(path)
    if a.ndim == 3:
        a = a[0] if a.shape[0] <= 8 else a[..., 0]
    P(f"    imread 成功  {a.shape}")
    return a


def to_u8(img, lo_pct=1, hi_pct=99.5):
    """与训练 patch 完全一致的归一化方式：百分位拉伸到 uint8"""
    if img.dtype == np.uint8:
        return img
    nz = img[img > 0]
    if len(nz) == 0:
        return np.zeros_like(img, np.uint8)
    lo, hi = np.percentile(nz, [lo_pct, hi_pct])
    v = np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    return (v * 255).astype(np.uint8)


def gaussian_heatmap(px, py, h, w, sigma=SIGMA):
    """与训练 patch 完全一致的转录本热图生成方式"""
    hm = np.zeros((h, w), np.float32)
    ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    if ok.any():
        np.add.at(hm, (py[ok], px[ok]), 1.0)
    k = int(sigma * 4) | 1
    hm = cv2.GaussianBlur(hm, (k, k), sigma)
    if hm.max() > 0:
        hm = hm / hm.max()
    return (hm * 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════
#  按目标加载 (image_2ch, gt_centroids)
# ══════════════════════════════════════════════════════════════════════════
def load_xenium(cfg):
    x0 = cfg["roi_x0"]; y0 = cfg["roi_y0"]
    w, h, ps = cfg["img_w"], cfg["img_h"], cfg["pixel_size"]
    x1, y1 = x0 + w * ps, y0 + h * ps
    dapi_path = f"{cfg['data_dir']}/{cfg['dapi_rel']}"

    P(f"  ROI µm x[{x0:.1f},{x1:.1f}] y[{y0:.1f},{y1:.1f}]  "
      f"→ {w}×{h} px @ {ps} µm/px")
    P(f"  读 DAPI {dapi_path}")
    dapi = open_dapi_xenium(dapi_path)
    FH, FW = dapi.shape
    px0, py0 = int(round(x0 / ps)), int(round(y0 / ps))
    crop = np.asarray(dapi[max(0, py0):min(FH, py0 + h),
                           max(0, px0):min(FW, px0 + w)])
    if crop.shape != (h, w):
        pad = np.zeros((h, w), crop.dtype)
        pad[:crop.shape[0], :crop.shape[1]] = crop
        crop = pad
    dapi_u8 = to_u8(crop)
    P(f"    DAPI crop {dapi_u8.shape}  均值 {dapi_u8.mean():.1f}")

    P(f"  读转录本 + GT 质心 ...")
    tx = pd.read_parquet(f"{cfg['data_dir']}/transcripts.parquet",
                         columns=["feature_name", "x_location",
                                  "y_location", "qv"])
    tx = tx[(tx.qv >= 20) &
            (tx.x_location >= x0) & (tx.x_location < x1) &
            (tx.y_location >= y0) & (tx.y_location < y1)]
    tx = tx[~tx.feature_name.str.startswith(
        ("NegControl", "Unassigned", "BLANK", "antisense"), na=False)]
    tpx = ((tx.x_location.values - x0) / ps).astype(np.int32).clip(0, w - 1)
    tpy = ((tx.y_location.values - y0) / ps).astype(np.int32).clip(0, h - 1)
    P(f"    转录本 {len(tx):,} 条")
    hm_u8 = gaussian_heatmap(tpx, tpy, h, w)

    gt = pd.read_parquet(f"{cfg['data_dir']}/cells.parquet",
                         columns=["cell_id", "x_centroid", "y_centroid"])
    gt = gt[(gt.x_centroid >= x0) & (gt.x_centroid < x1) &
            (gt.y_centroid >= y0) & (gt.y_centroid < y1)]
    P(f"    GT {len(gt):,} 细胞（参考 {cfg['gt_cells']:,}）")

    image_2ch = np.stack([dapi_u8, hm_u8], axis=-1)
    gt_xy_px = np.stack([(gt.x_centroid.values - x0) / ps,
                         (gt.y_centroid.values - y0) / ps], axis=1)
    return image_2ch, gt_xy_px, len(gt)


def load_merfish(cfg, resolution):
    d = cfg["data_dir"]; stem = cfg["stem"]
    x0, x1 = cfg["roi_x0"], cfg["roi_x1"]
    y0, y1 = cfg["roi_y0"], cfg["roi_y1"]
    w = int(round((x1 - x0) / resolution))
    h = int(round((y1 - y0) / resolution))
    P(f"  ROI µm x[{x0},{x1}] y[{y0},{y1}] → {w}×{h} px @ {resolution} µm/px")
    P(f"  ★ 跨平台目标：DAPI 染色特性、原生分辨率都与训练数据不同")

    tform = pd.read_csv(
        f"{d}/images/{stem}_images_micron_to_mosaic_pixel_transform.csv",
        sep=r"\s+", header=None).values
    sx, sy = float(tform[0, 0]), float(tform[1, 1])
    ox, oy = float(tform[0, 2]), float(tform[1, 2])
    mx0, my0 = int(np.floor(sx * x0 + ox)), int(np.floor(sy * y0 + oy))
    mx1, my1 = int(np.ceil(sx * x1 + ox)), int(np.ceil(sy * y1 + oy))

    dapi_path = f"{d}/images/{stem}_images_mosaic_DAPI_z3.tif"
    P(f"  读 DAPI {dapi_path}")
    try:
        mm = tifffile.memmap(dapi_path, mode="r")
        FH, FW = mm.shape
        crop = np.ascontiguousarray(
            mm[max(0, my0):min(FH, my1), max(0, mx0):min(FW, mx1)])
    except Exception:
        import zarr
        z = zarr.open(tifffile.imread(dapi_path, aszarr=True), mode="r")
        a = z[0] if isinstance(z, zarr.hierarchy.Group) else z
        FH, FW = a.shape
        crop = np.ascontiguousarray(
            a[max(0, my0):min(FH, my1), max(0, mx0):min(FW, mx1)])
    crop_u8 = to_u8(crop)
    crop_u8 = cv2.resize(crop_u8, (w, h), interpolation=cv2.INTER_AREA)
    P(f"    DAPI crop → {crop_u8.shape}  均值 {crop_u8.mean():.1f}")

    P(f"  读转录本 + GT 质心 ...")
    chunks = []
    for ck in pd.read_csv(
            f"{d}/{stem}_detected_transcripts_S1R1.csv",
            chunksize=2_000_000,
            usecols=["global_x", "global_y", "gene"]):
        sub = ck[(ck.global_x >= x0) & (ck.global_x < x1) &
                 (ck.global_y >= y0) & (ck.global_y < y1)]
        if len(sub):
            chunks.append(sub)
    tx = pd.concat(chunks, ignore_index=True)
    tx = tx[~tx.gene.astype(str).str.startswith(("Blank", "blank", "BLANK"))]
    tpx = ((tx.global_x.values - x0) / resolution).astype(np.int32).clip(0, w-1)
    tpy = ((tx.global_y.values - y0) / resolution).astype(np.int32).clip(0, h-1)
    P(f"    转录本 {len(tx):,} 条")
    hm_u8 = gaussian_heatmap(tpx, tpy, h, w)

    meta = pd.read_csv(
        f"{d}/{stem}_cell_metadata_S1R1.csv").rename(
        columns={"Unnamed: 0": "cell_id"})
    gt = meta[(meta.center_x >= x0) & (meta.center_x < x1) &
              (meta.center_y >= y0) & (meta.center_y < y1)]
    P(f"    GT {len(gt):,} 细胞（参考 {cfg['gt_cells']:,}）")

    image_2ch = np.stack([crop_u8, hm_u8], axis=-1)
    gt_xy_px = np.stack([(gt.center_x.values - x0) / resolution,
                         (gt.center_y.values - y0) / resolution], axis=1)
    return image_2ch, gt_xy_px, len(gt)


def load_stereo(cfg):
    """
    Stereo-seq —— 测序类平台，GEM 每行是空间 bin 的 UMI 聚合计数，
    不是单分子坐标。GEM 的 (x,y) 就是 ssDNA 图像的像素坐标，
    这一点已被 UCS/Cellist/BOMS 反复确认过，不需要额外坐标变换。
    """
    import gzip
    gem_path, ssdna_path = cfg["gem"], cfg["ssdna"]
    x0, x1 = cfg["roi_x0"], cfg["roi_x1"]
    y0, y1 = cfg["roi_y0"], cfg["roi_y1"]
    w, h = x1 - x0, y1 - y0

    P(f"  读 GEM（约几百万行，稍等）...")
    opener = gzip.open if gem_path.endswith(".gz") else open
    skip = 0
    with opener(gem_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                skip += 1
            else:
                break
    gem = pd.read_csv(gem_path, sep="\t", skiprows=skip)
    ren = {}
    for c in gem.columns:
        cl = c.lower()
        if cl in ("geneid", "gene"):
            ren[c] = "gene"
        elif cl == "x":
            ren[c] = "x"
        elif cl == "y":
            ren[c] = "y"
        elif cl in ("midcounts", "midcount", "count"):
            ren[c] = "count"
        elif cl in ("cell", "cellid", "label"):
            ren[c] = "cell"
    gem = gem.rename(columns=ren)
    if "count" not in gem.columns:
        gem["count"] = 1
    if "cell" not in gem.columns:
        gem["cell"] = 0
    gem = gem[(gem.x >= x0) & (gem.x < x1) & (gem.y >= y0) & (gem.y < y1)]
    P(f"    ROI 内 {len(gem):,} 行")

    P(f"  读 ssDNA（GEM (x,y) 就是图像像素坐标，无需变换）...")
    ssdna = tifffile.imread(ssdna_path)
    if ssdna.ndim == 3:
        ssdna = ssdna[0] if ssdna.shape[0] <= 8 else ssdna[..., 0]
    crop = np.asarray(ssdna[y0:y1, x0:x1])
    if crop.dtype != np.uint8:
        nz = crop[crop > 0]
        lo, hi = (np.percentile(nz, [1, 99.5]) if len(nz) else (0, 1))
        crop = np.clip((crop.astype(np.float32)-lo)/max(hi-lo, 1e-6), 0, 1)
        crop = (crop*255).astype(np.uint8)
    P(f"    ssDNA crop {crop.shape}  均值 {crop.mean():.1f}")

    P(f"  建热图（★ 按 MIDCounts 加权，不是每行+1）...")
    px = (gem.x.values - x0).astype(np.int32).clip(0, w-1)
    py = (gem.y.values - y0).astype(np.int32).clip(0, h-1)
    cnt = gem["count"].values.astype(np.float32)
    hm = np.zeros((h, w), np.float32)
    np.add.at(hm, (py, px), cnt)
    k = int(6*4) | 1   # sigma=6，与训练时一致
    hm = cv2.GaussianBlur(hm, (k, k), 6)
    if hm.max() > 0:
        hm = hm / hm.max()
    hm_u8 = (hm * 255).astype(np.uint8)

    P(f"  GT 质心（GEM 的 cell 列，CellBin 官方）...")
    sub = gem[gem.cell > 0]
    gt_grp = sub.groupby("cell")[["x", "y"]].mean()
    P(f"    GT {len(gt_grp):,} 细胞（参考 {cfg['gt_cells']:,}）")

    image_2ch = np.stack([crop, hm_u8], axis=-1)
    gt_xy_px = np.stack([gt_grp.x.values - x0, gt_grp.y.values - y0], axis=1)
    return image_2ch, gt_xy_px, len(gt_grp)


def load_target(name, resolution):
    cfg = TARGETS[name]
    P(f"  {cfg['note']}")
    if cfg["kind"] == "xenium":
        return load_xenium(cfg)
    if cfg["kind"] == "stereo":
        return load_stereo(cfg)
    return load_merfish(cfg, resolution)


# ══════════════════════════════════════════════════════════════════════════
#  模型
# ══════════════════════════════════════════════════════════════════════════
def build_model():
    """
    ★ sys.path 必须同时包含内层和外层目录，原因：

    训练时用的是 PYTHONPATH=/data/qiuyijia/GeneSegNet（外层）去跑
    GeneSeg_train.py，同时 Python 自己又会把脚本所在目录（内层，
    .../GeneSegNet/GeneSegNet）自动加进 sys.path[0]——两个目录当时是
    同时在 sys.path 上的。dynamics.py 里有个 numba 编译并缓存的函数
    （map_coordinates），缓存里记录的模块环境跟"GeneSegNet"这个包名
    绑在一起；如果只加内层目录（之前的写法），"GeneSegNet"作为包名
    就解析不到，numba 想要重建缓存环境时会 importlib.import_module
    ("GeneSegNet") 直接失败：ModuleNotFoundError: No module named
    'GeneSegNet'。同时加上外层目录能让两种导入方式都解析得到，
    numba 的缓存重建就不会失败。
    """
    for p in (GENESEGNET_PKG_DIR, os.path.dirname(GENESEGNET_PKG_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import models

    if not os.path.exists(MODEL_PATH):
        sys.exit(f"✘ 权重不存在: {MODEL_PATH}")
    P(f"  权重 {os.path.basename(MODEL_PATH)}  "
      f"({os.path.getsize(MODEL_PATH)/1e6:.1f} MB)")

    device, gpu = models.assign_device(use_torch=True, gpu=True, device=0)
    P(f"  device={device}  gpu={gpu}")

    t0 = time.time()
    model = models.GeneSegModel(
        gpu=gpu, device=device,
        pretrained_model=MODEL_PATH,
        model_type=None,
        diam_mean=42.0,          # 会被 checkpoint 里存的真实值覆盖，写多少不重要
        residual_on=True, style_on=True, concatenation=False,
        net_avg=False,
        nchan=2,                 # DAPI + 转录本热图，两通道
    )
    P(f"  模型加载完成 ({time.time()-t0:.1f}s)")
    P(f"    实际 diam_mean（从 checkpoint 读出）: {model.diam_mean:.2f}")
    P(f"    实际 diam_labels（训练集经验直径）: {model.diam_labels:.2f}")
    return model


def run_eval(model, image_2ch, confidence, flow_threshold, min_size,
            diameter=None):
    masks, flows, styles = model.eval(
        image_2ch,
        channels=None,            # 对应训练时用的 --all_channels
        diameter=diameter,        # None → 用 model.diam_labels，公平盲测
        do_3D=False,
        net_avg=False,
        augment=False,
        tile=True,                # 大图内部自动分块，不需要手工切 patch
        resample=True,
        interp=True,
        flow_threshold=flow_threshold,
        confidence_threshold=confidence,
        min_size=min_size,
        stitch_threshold=0.0,
        invert=False,
        batch_size=8,
        normalize=True,
        model_loaded=True,        # 权重已在构造函数里加载过，不用重复加载
    )
    return masks


def match_to_gt(masks, gt_xy_px, match_radius_px):
    """粗略的一对一匹配，只为诊断用，不是完整评估"""
    from scipy.spatial import cKDTree
    import skimage.measure
    props = skimage.measure.regionprops(masks)
    if not props:
        return 0, 0
    pred_xy = np.array([[p.centroid[1], p.centroid[0]] for p in props])
    n_pred, n_gt = len(pred_xy), len(gt_xy_px)
    if n_gt == 0:
        return n_pred, 0
    tree = cKDTree(gt_xy_px)
    d, idx = tree.query(pred_xy, k=1, distance_upper_bound=match_radius_px)
    best = {}
    for i in range(n_pred):
        if not np.isfinite(d[i]) or d[i] >= match_radius_px:
            continue
        g = int(idx[i])
        if g not in best or d[i] < best[g]:
            best[g] = d[i]
    return n_pred, len(best)


# ══════════════════════════════════════════════════════════════════════════
def scan_thresholds(model, image_2ch, gt_xy_px, n_gt, match_radius_px,
                    confidences, flow_thresholds, min_size, diameter=None):
    P(f"\n{SEP}")
    P(f"阈值扫描（GT {n_gt:,}）")
    P(f"  diameter={'盲测(None，用 model.diam_labels)' if diameter is None else diameter}")
    P(SEP)
    P(f"  {'confidence':>10} {'flow_th':>8} {'原始细胞数':>10} "
      f"{'匹配':>6} {'vs GT':>9}")
    P("  " + "-" * 52)
    rows = []
    for conf in confidences:
        for ft in flow_thresholds:
            t0 = time.time()
            masks = run_eval(model, image_2ch, conf, ft, min_size, diameter)
            n_pred = int(masks.max())
            _, n_match = match_to_gt(masks, gt_xy_px, match_radius_px)
            el = time.time() - t0
            flag = "  ←" if 0.5 * n_gt < n_pred < 1.5 * n_gt else ""
            P(f"  {conf:>10.2f} {ft:>8.2f} {n_pred:>10,} {n_match:>6,} "
              f"{n_pred-n_gt:>+9,}{flag}  ({el:.0f}s)")
            rows.append(dict(confidence=conf, flow_threshold=ft,
                             diameter=diameter, n_pred=n_pred,
                             n_match=n_match))
    df = pd.DataFrame(rows)
    if df.n_pred.max() == 0:
        P(f"\n  ✘ 所有阈值组合下原始细胞数都是 0")
        P(f"     说明 flow field 对这个目标没有有效信号，"
          f"和当初 NSCLC 权重在淋巴结上的失败模式一致")
        P(f"     （那次扫 confidence_threshold 0.0~0.6，细胞数恒为 99，"
          f"问题不在阈值而在模型本身没学到这类细胞的特征）")
    else:
        df["gap"] = (df.n_pred - n_gt).abs()
        b = df.sort_values("gap").iloc[0]
        P(f"\n  最接近 GT: confidence={b.confidence} "
          f"flow_threshold={b.flow_threshold} → {int(b.n_pred):,} 细胞")
        P(f"  正式跑: --run --confidence {b.confidence} "
          f"--flow-threshold {b.flow_threshold}"
          f"{f' --diameter {diameter}' if diameter is not None else ''}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=list(TARGETS))
    ap.add_argument("--scan-thresholds", action="store_true")
    ap.add_argument("--confidences", type=float, nargs="+",
                    default=[0.0, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--flow-thresholds", type=float, nargs="+",
                    default=[0.4, 0.6, 0.8])
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--confidence", type=float, default=0.5)
    ap.add_argument("--flow-threshold", type=float, default=0.6)
    ap.add_argument("--min-size", type=int, default=300)
    ap.add_argument("--diameter", type=float, default=None,
                    help="留空=盲测（用模型自己的 diam_labels）；"
                         "指定数值=告诉模型目标组织的真实细胞直径(px)，"
                         "这样做更容易得高分但不是公平的泛化测试")
    ap.add_argument("--match-radius-um", type=float, default=None,
                    help="留空则用该 target 配置的默认值"
                         "（lymph=8.0 breast=12.0 merfish=12.0）")
    ap.add_argument("--resolution", type=float, default=0.5,
                    help="仅 merfish 用：工作分辨率 µm/px")
    ap.add_argument("--gpu", type=int, default=None)
    cfg = ap.parse_args()

    P(SEP); P(f"GeneSegNet 推理 —— {cfg.target}"); P(SEP)
    tgt = TARGETS[cfg.target]
    P(f"  {tgt['note']}")
    if tgt.get("true_diam_px") is not None and cfg.diameter is None:
        P(f"  ⓘ 该组织真实细胞直径参考值 {tgt['true_diam_px']}px"
          f"（模型自己的 diam_labels 约 44px，方向和幅度差异见上）")
        P(f"    当前跑的是盲测（--diameter 未指定）；"
          f"想测'告知真实尺度后是否改善'可加 "
          f"--diameter {tgt['true_diam_px']}")
    setup_gpu(cfg.gpu)

    image_2ch, gt_xy_px, n_gt = load_target(cfg.target, cfg.resolution)
    ps = tgt.get("pixel_size", cfg.resolution)
    match_radius_um = (cfg.match_radius_um if cfg.match_radius_um is not None
                       else tgt.get("match_radius_um", 8.0))
    P(f"  匹配半径 {match_radius_um}µm"
      f"{'（用户指定）' if cfg.match_radius_um is not None else '（该target默认值）'}")
    match_radius_px = match_radius_um / ps

    model = build_model()

    if cfg.scan_thresholds:
        df = scan_thresholds(model, image_2ch, gt_xy_px, n_gt,
                             match_radius_px, cfg.confidences,
                             cfg.flow_thresholds, cfg.min_size,
                             cfg.diameter)
        df.to_csv(f"{OUT_DIR}/{cfg.target}_scan.csv", index=False)
        P(f"\n  扫描表 → {OUT_DIR}/{cfg.target}_scan.csv")
        return

    if not cfg.run:
        P(f"\n  加 --scan-thresholds 先扫参数，或 --run 直接指定阈值跑一次")
        return

    P(f"\n  confidence={cfg.confidence}  flow_threshold={cfg.flow_threshold}"
      f"  diameter={'盲测(None)' if cfg.diameter is None else cfg.diameter}")
    masks = run_eval(model, image_2ch, cfg.confidence, cfg.flow_threshold,
                     cfg.min_size, cfg.diameter)
    n_pred = int(masks.max())
    n_pred_total, n_match = match_to_gt(masks, gt_xy_px, match_radius_px)
    P(f"\n{SEP}")
    P(f"  细胞数 {n_pred:,}  (GT {n_gt:,}, ratio {n_pred/max(n_gt,1):.3f})")
    P(f"  粗略匹配（仅诊断用，非完整评估）: {n_match:,}")
    out_mask = f"{OUT_DIR}/{cfg.target}_mask.tif"
    tifffile.imwrite(out_mask, masks.astype(np.uint32))
    P(f"  mask → {out_mask}")
    P(f"\n  完整 19 项指标评估：把这个 mask 接入本次对话中已经建立的")
    P(f"  eval_cellist_xenium.py 同款评估框架（质心匹配 + 计数相关 +")
    P(f"  基因向量余弦 + 分配准确率），而不是只看这里的粗略计数")


if __name__ == "__main__":
    main()