#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmx_lymph.py  (Mesmer / DeepCell)
===================================
在 CosMx 人淋巴结数据集（/data/qiuyijia/dataset/cosmx_lymph_node）上跑 Mesmer
（DeepCell，TissueNet 预训练，零样本，不训练）做整细胞分割，以数据集自带的
CosMx CellLabels 为 GT 做 benchmark。数据读取 / FOV 选择 / 指标 / 聚合全部走
/data/qiuyijia/eval/cosmx_lymph_common.py，与 cellpose / cellotype / cellsam /
stardist 四个版本逐行同源 —— 方法之间唯一差别就是推理这一步。

数据集事实（已实测）
--------------------
* 数据集**自带 FOV 划分**：共 **400 个 FOV**，每个 4256x4256 @0.1203um/px。
  napari OME-Zarr 是拼好的全片(93618x102128)，本脚本按 .zattrs 的 fov_offsets 裁回单 FOV。
* metadata 共 1,852,946 个细胞，中位 **4897 cells/FOV** → **凑 1w 细胞只要 2 个 FOV**。
* 默认 FOV = (48, 261)：密度最接近全片中位数、且分处切片前后两半，GT 4898+4896=9794。

Mesmer 两个必须写对的点
-----------------------
1) 输入是【2 通道】[核, 膜]，核(DAPI)在前。本数据集：
     核 = DNA 通道；膜 = Membrane 通道（--mem-channel memmax 可换成 4 marker 取 max）
   这与 cellpose 版的通道构造完全一致（那边实测 Membrane 单通道已经吃满，
   堆其它 marker 只有千分位差别），所以五个方法拿到的是同一份核/膜信息。
2) **image_mpp 必须是数据真实分辨率**。Mesmer 训练分辨率 model_mpp=0.5um/px，
   predict 会按 image_mpp 把图缩放到训练分辨率。本数据集 scale_um=0.1203 →
   必须传 0.1203（脚本默认直接从 .zattrs 读，不写死，避免抄错）。
   0.1203/0.5 = 0.24，4256px 会被降采样到 ~1022px，正好落在 TissueNet 的尺度上。
   **别照抄别人给 TissueNet 用的 image_mpp=0.5**，那会让 Mesmer 以为细胞大了 ~4 倍而严重欠分割。

compartment='whole-cell' → 整细胞，直接对 CellLabels 比，与其它四方法同口径。
零样本、不训练（公平性 + 避免同片 FOV 泄漏）。

依赖：/data/qiuyijia/anaconda3/envs/deepcell（deepcell-tf + tensorflow 2.8.4，已带 zarr）
"""

import os
import sys
import argparse

import numpy as np

sys.path.insert(0, "/data/qiuyijia/eval")
import cosmx_lymph_common as C

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# Mesmer 输入：[1, H, W, 2] = [核, 膜]
# --------------------------------------------------------------------------- #
def build_mesmer_input(chans, mem_channel, channel_order="nuc_mem"):
    nuclear, membrane = C.nuc_mem(chans, mem_channel)
    if membrane is None:                       # dna_only：膜通道置零（Mesmer 仍要 2 通道）
        membrane = np.zeros_like(nuclear)
    if channel_order == "mem_nuc":
        two = np.stack([membrane, nuclear], axis=-1)
    else:                                      # nuc_mem（Mesmer 约定：核在前）
        two = np.stack([nuclear, membrane], axis=-1)
    return two[np.newaxis, ...].astype(np.float32)


# --------------------------------------------------------------------------- #
# Mesmer 运行器（权重只加载一次）
# --------------------------------------------------------------------------- #
class MesmerRunner:
    def __init__(self, image_mpp, compartment="whole-cell", batch_size=4,
                 pixel_expansion=0):
        self.image_mpp = float(image_mpp)
        self.compartment = compartment
        self.batch_size = batch_size
        self.pixel_expansion = pixel_expansion
        self._app = None

    def app(self):
        if self._app is None:
            from deepcell.applications import Mesmer
            self._app = Mesmer()
            print(f"[mesmer] 已加载 | model_mpp(训练分辨率)={self._app.model_mpp} "
                  f"| image_mpp(你的数据)={self.image_mpp} | compartment={self.compartment}")
            if abs(self.image_mpp - 0.5) < 1e-6:
                print("  [warn] image_mpp=0.5 对 CosMx 是错的！本数据集应为 0.1203，"
                      "填 0.5 会严重欠分割。")
        return self._app

    def segment(self, X):
        kwargs = dict(image_mpp=self.image_mpp, compartment=self.compartment,
                      batch_size=self.batch_size)
        if self.pixel_expansion and self.compartment == "nuclear":
            kwargs["postprocess_kwargs_nuclear"] = {"pixel_expansion": self.pixel_expansion}
        y = np.squeeze(np.asarray(self.app().predict(X, **kwargs)))
        if y.ndim > 2:                # compartment='both' -> [H,W,2]，取整细胞通道 0
            y = y[..., 0]
        return y.astype(np.int32, copy=False)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="CosMx 淋巴结：Mesmer(TissueNet预训练,零样本) + 对 CosMx CellLabels 的 benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    C.add_common_args(ap, out_dir_default=os.path.join(HERE, "cosmx_lymph_output"))

    ap.add_argument("--image-mpp", type=float, default=None,
                    help="数据真实分辨率(um/px)。默认从 .zattrs 的 scale_um 读(=0.1203)。别填 0.5！")
    ap.add_argument("--compartment", default="whole-cell",
                    choices=["whole-cell", "nuclear", "both"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--pixel-expansion", type=int, default=0,
                    help="仅 compartment=nuclear 时生效，核后再膨胀 N 像素")
    ap.add_argument("--channel-order", default="nuc_mem", choices=["nuc_mem", "mem_nuc"],
                    help="Mesmer 约定核在前(nuc_mem)。除非确认数据反了，别改")
    args = ap.parse_args()

    # image_mpp 从数据集自己的 scale_um 读，避免手抄出错
    if args.image_mpp is None:
        args.image_mpp = C.read_fov_meta(args.data_root)["scale_um"]
        print(f"[mesmer] image_mpp 未指定，从 .zattrs 读到 scale_um={args.image_mpp}")

    runner = None if args.reuse_pred else MesmerRunner(
        image_mpp=args.image_mpp, compartment=args.compartment,
        batch_size=args.batch_size, pixel_expansion=args.pixel_expansion)

    def segment_fn(chans, mem_channel):
        X = build_mesmer_input(chans, mem_channel, channel_order=args.channel_order)
        return runner.segment(X)

    meta_extra = dict(
        mesmer_model="TissueNet_pretrained", finetuned=False,
        image_mpp=args.image_mpp, compartment=args.compartment,
        channel_order=args.channel_order,
        note_mesmer="Mesmer 原生整细胞、零样本；image_mpp 取数据真实分辨率(0.1203)，"
                    "Mesmer 内部按 model_mpp=0.5 降采样 0.24x 到训练尺度。")

    C.run_benchmark(args, f"Mesmer({args.compartment})", segment_fn,
                    meta_extra, f"mesmer_{args.compartment}")


if __name__ == "__main__":
    main()


'''
# deepcell env 自带 zarr，可以直接跑（顺手把缓存导出来给 stardist/cellotype 用）
DC=/data/qiuyijia/anaconda3/envs/deepcell/bin/python

$DC /data/qiuyijia/Mesmer/cosmx_lymph.py --use-default-fovs --export-cache

nohup $DC /data/qiuyijia/Mesmer/cosmx_lymph.py \
    --use-default-fovs --overlay \
    --out-dir /data/qiuyijia/Mesmer/cosmx_lymph_output \
    > /data/qiuyijia/Mesmer/cosmx_lymph.log 2>&1 &
'''
