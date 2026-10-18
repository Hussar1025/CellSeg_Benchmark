#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmx_lymph.py  (CellSAM)
=========================
在 CosMx 人淋巴结数据集（/data/qiuyijia/dataset/cosmx_lymph_node）上跑 CellSAM
（通用基础模型，零样本，不训练）做整细胞分割，以数据集自带的 CosMx CellLabels 为 GT
做 benchmark。数据读取 / FOV 选择 / 指标 / 聚合全部走
/data/qiuyijia/eval/cosmx_lymph_common.py，与 cellpose / cellotype / mesmer /
stardist 四个版本逐行同源 —— 方法之间唯一差别就是推理这一步。

数据集事实（已实测）
--------------------
* 数据集**自带 FOV 划分**：共 **400 个 FOV**，每个 4256x4256 @0.1203um/px。
  napari OME-Zarr 是拼好的全片(93618x102128)，本脚本按 .zattrs 的 fov_offsets 裁回单 FOV。
* metadata 共 1,852,946 个细胞，中位 **4897 cells/FOV** → **凑 1w 细胞只要 2 个 FOV**。
* 默认 FOV = (48, 261)：密度最接近全片中位数、且分处切片前后两半，GT 4898+4896=9794。

关于 CellSAM
------------
CellSAM = CellFinder(检测框) + SAM(mask decoder) 的通用基础模型，零样本即可跨模态分割，
原生做整细胞分割 → 直接对整细胞 CellLabels 比，无需 StarDist 那种膨胀超参，
与 Cellpose / CelloType / Mesmer 同口径可横向比。

两个必须写对的点
----------------
1) **动态范围**：zarr 存的是 uint16，而 CellSAM 权重是在 8-bit 图上训的。
   直接喂 uint16 等于给了它没见过的量纲。所以按 1~99.5 百分位拉伸到 uint8
   （--norm pct，默认）。这是"把数据放回方法期望的量纲"，不是调参。
   --norm raw 可关掉做对照。
2) **切块**：4256² 一次进不去，用官方 cellsam_pipeline(use_wsi=True) 的 whole-slide
   切块模式（相当于 cellpose 的 n_tiles / cellotype 的 SAHI）。

通道排布沿用本仓库 CosMx_cellsam.py 的默认 blank_nuc_mem = [0, 核, 膜]。
核=DNA，膜=Membrane，与其它四个方法完全同源。

依赖：/data/qiuyijia/anaconda3/envs/cellsam（cellSAM，已带 zarr）
"""

import os
import sys
import argparse

import numpy as np

sys.path.insert(0, "/data/qiuyijia/eval")
import cosmx_lymph_common as C

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# CellSAM 输入
# --------------------------------------------------------------------------- #
def build_cellsam_input(chans, mem_channel, channel_order="blank_nuc_mem", norm="pct"):
    """
    核 = DNA，膜 = Membrane（或 memmax）。
    norm=pct  -> 1~99.5 百分位拉伸到 uint8（默认，匹配 CellSAM 预训练的 8-bit 量纲）
    norm=raw  -> 原样 float32（对照用）
    channel_order:
      blank_nuc_mem -> (H,W,3) = [0, 核, 膜]   (默认，与本仓库 CosMx 版一致)
      nuc_mem       -> (H,W,2) = [核, 膜]
      mem_nuc       -> (H,W,2) = [膜, 核]
      nuc_only      -> (H,W)   仅核（保守回退）
    """
    nuclear, membrane = C.nuc_mem(chans, mem_channel)
    if norm == "pct":
        nuclear = C.pct_norm_u8(nuclear)
        membrane = C.pct_norm_u8(membrane) if membrane is not None else None
    if membrane is None:
        membrane = np.zeros_like(nuclear)
    zero = np.zeros_like(nuclear)

    if channel_order == "nuc_mem":
        return np.stack([nuclear, membrane], axis=-1)
    if channel_order == "mem_nuc":
        return np.stack([membrane, nuclear], axis=-1)
    if channel_order == "nuc_only":
        return nuclear
    return np.stack([zero, nuclear, membrane], axis=-1)     # blank_nuc_mem


# --------------------------------------------------------------------------- #
# CellSAM 运行器（权重只加载一次）
# --------------------------------------------------------------------------- #
class CellSAMRunner:
    def __init__(self, device="cuda", use_wsi=True, low_contrast_enhancement=False,
                 gauge_cell_size=False, block_size=None):
        self.device = device
        self.use_wsi = use_wsi
        self.low_contrast_enhancement = low_contrast_enhancement
        self.gauge_cell_size = gauge_cell_size
        self.block_size = block_size
        self._pipeline = None
        self._seg_fn = None

    def _ensure_loaded(self):
        if self._pipeline is None and self._seg_fn is None:
            try:
                from cellSAM import cellsam_pipeline
                self._pipeline = cellsam_pipeline
                print(f"[cellsam] cellsam_pipeline(use_wsi={self.use_wsi}) "
                      f"| device={self.device} | block_size={self.block_size}")
            except Exception:
                from cellSAM import segment_cellular_image
                self._seg_fn = segment_cellular_image
                print("[cellsam] 旧版本：回退 segment_cellular_image（不切块，4256² 可能 OOM）")

    def segment(self, img):
        self._ensure_loaded()
        if self._pipeline is not None:
            kwargs = dict(use_wsi=self.use_wsi,
                          low_contrast_enhancement=self.low_contrast_enhancement,
                          gauge_cell_size=self.gauge_cell_size, device=self.device)
            if self.block_size:
                kwargs["block_size"] = self.block_size
            try:
                mask = self._pipeline(img, **kwargs)
            except TypeError:
                # 某些版本参数名不同 / device 内部自动检测
                try:
                    mask = self._pipeline(img, use_wsi=self.use_wsi)
                except TypeError:
                    mask = self._pipeline(img)
        else:
            out = self._seg_fn(img, device=self.device)
            mask = out[0] if isinstance(out, (tuple, list)) else out
        mask = np.asarray(mask)
        if mask.ndim > 2:
            mask = np.squeeze(mask)
        return mask.astype(np.int32, copy=False)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="CosMx 淋巴结：CellSAM(零样本) + 对 CosMx CellLabels 的 benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    C.add_common_args(ap, out_dir_default=os.path.join(HERE, "cosmx_lymph_output"))

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--use-wsi", dest="use_wsi", action="store_true", default=True,
                    help="whole-slide 切块模式（4256² 大 FOV 必开）")
    ap.add_argument("--no-use-wsi", dest="use_wsi", action="store_false")
    ap.add_argument("--block-size", type=int, default=None,
                    help="use_wsi 的分块边长；不给用 CellSAM 默认")
    ap.add_argument("--low-contrast-enhancement", action="store_true")
    ap.add_argument("--gauge-cell-size", action="store_true")
    ap.add_argument("--channel-order", default="blank_nuc_mem",
                    choices=["blank_nuc_mem", "nuc_mem", "mem_nuc", "nuc_only"],
                    help="核/膜通道排布（与本仓库 CosMx_cellsam.py 默认一致）")
    ap.add_argument("--norm", default="pct", choices=["pct", "raw"],
                    help="pct=1~99.5百分位拉到uint8(默认,匹配预训练8-bit量纲); raw=原始uint16")
    args = ap.parse_args()

    runner = None if args.reuse_pred else CellSAMRunner(
        device=args.device, use_wsi=args.use_wsi,
        low_contrast_enhancement=args.low_contrast_enhancement,
        gauge_cell_size=args.gauge_cell_size, block_size=args.block_size)

    def segment_fn(chans, mem_channel):
        img = build_cellsam_input(chans, mem_channel,
                                  channel_order=args.channel_order, norm=args.norm)
        return runner.segment(img)

    meta_extra = dict(
        cellsam_model="generalist_pretrained", finetuned=False,
        use_wsi=args.use_wsi, block_size=args.block_size,
        channel_order=args.channel_order, norm=args.norm,
        low_contrast_enhancement=args.low_contrast_enhancement,
        gauge_cell_size=args.gauge_cell_size,
        note_cellsam="CellSAM 原生整细胞、零样本；uint16 需按百分位拉到 8-bit "
                     "以匹配预训练量纲；4256² 用 use_wsi 切块。")

    C.run_benchmark(args, "CellSAM", segment_fn, meta_extra, "cellsam")


if __name__ == "__main__":
    main()


'''
# cellsam env 自带 zarr，可以直接跑
CS=/data/qiuyijia/anaconda3/envs/cellsam/bin/python

nohup $CS /data/qiuyijia/CellSam/cosmx_lymph.py \
    --use-default-fovs --use-wsi --overlay \
    --out-dir /data/qiuyijia/CellSam/cosmx_lymph_output \
    > /data/qiuyijia/CellSam/cosmx_lymph.log 2>&1 &
'''
