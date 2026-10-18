#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmx_lymph.py  (CelloType)
===========================
在 CosMx 人淋巴结数据集（/data/qiuyijia/dataset/cosmx_lymph_node）上跑 CelloType
（TissueNet 预训练，零样本，不微调）做整细胞分割，以数据集自带的 CosMx CellLabels 为 GT
做 benchmark。数据读取 / FOV 选择 / 指标 / 聚合全部走
/data/qiuyijia/eval/cosmx_lymph_common.py，与 cellpose / cellsam / mesmer /
stardist 四个版本逐行同源 —— 方法之间唯一差别就是推理这一步。

数据集事实（已实测）
--------------------
* 数据集**自带 FOV 划分**：共 **400 个 FOV**，每个 4256x4256 @0.1203um/px。
  napari OME-Zarr 是拼好的全片(93618x102128)，本脚本按 .zattrs 的 fov_offsets 裁回单 FOV。
* metadata 共 1,852,946 个细胞，中位 **4897 cells/FOV** → **凑 1w 细胞只要 2 个 FOV**。
* 默认 FOV = (48, 261)：密度最接近全片中位数、且分处切片前后两半，GT 4898+4896=9794。

为什么不微调（公平性）
----------------------
Cellpose(cpsam) / StarDist / CellSAM / Mesmer 在这套 benchmark 里全是零样本，
CelloType 也零样本，五方法才在同一起跑线。而且只有同一张切片的 FOV，
切它自己训自己测会系统性高估；GT 本身是 NanoString 算法输出（银标准），
在它上面训 CelloType 只是"模仿厂商"。

三个必须写对的点
----------------
1) **动态范围**：zarr 是 uint16，CelloType 读 PNG 即 uint8，预训练权重也在 8-bit 上训的。
   所以按 1~99.5 百分位拉伸到 uint8（--norm pct，默认）。这是放回方法期望的量纲。
2) **通道排布**：CelloType/TissueNet 约定 [R=可选, G=膜, B=核]，默认 opt_mem_nuc。
   核=DNA，膜=Membrane，与其它四个方法完全同源。
3) **切块**：4256² 整图进 Detectron2 会被按短边缩放而掉小细胞，所以默认 tile 模式，
   512 窗 + 128 重叠自己切块拼接。本数据集细胞等效直径中位 ~65px，
   overlap=128 > 2x 细胞直径，骑缝细胞不会被切两半。

依赖：/data/qiuyijia/anaconda3/envs/cellotype（CelloType + Detectron2 + MaskDINO）
      需要 models/tissuenet_model_0019999.pth 与 configs/maskdino_R50_..._2048.yaml。
      该 env **没装 zarr**，所以要先在带 zarr 的 env 里导一次缓存（见文件末尾用法）。
      注意：本机该 env 目前 import MultiScaleDeformableAttention 报 GLIBC_2.34 缺失，
      需在能跑 Detectron2 的机器上执行（脚本本身与环境无关）。
"""

import os
import sys
import argparse

import numpy as np

sys.path.insert(0, "/data/qiuyijia/eval")
import cosmx_lymph_common as C

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MODEL = "/data/qiuyijia/cellotype/CelloType/models/tissuenet_model_0019999.pth"
DEFAULT_CONFIG = ("/data/qiuyijia/cellotype/CelloType/configs/"
                  "maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml")


# --------------------------------------------------------------------------- #
# CelloType 输入（TissueNet 约定：R=可选, G=膜, B=核）
# --------------------------------------------------------------------------- #
def build_cellotype_input(chans, mem_channel, channel_order="opt_mem_nuc", norm="pct"):
    nuclear, membrane = C.nuc_mem(chans, mem_channel)
    if norm == "pct":
        nuclear = C.pct_norm_u8(nuclear)
        membrane = C.pct_norm_u8(membrane) if membrane is not None else None
    else:
        nuclear = np.clip(nuclear, 0, 255).astype(np.uint8)
        membrane = (np.clip(membrane, 0, 255).astype(np.uint8)
                    if membrane is not None else None)
    if membrane is None:
        membrane = np.zeros_like(nuclear)
    zero = np.zeros_like(nuclear)
    order = {
        "opt_mem_nuc": (zero, membrane, nuclear),
        "nuc_mem_opt": (nuclear, membrane, zero),
        "mem_nuc_opt": (membrane, nuclear, zero),
    }[channel_order]
    return np.stack(order, axis=-1).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 切块 + 实例拼接（只依赖稳定的 predict()；沿用本仓库 CosMx_cellotype.py 那套）
# --------------------------------------------------------------------------- #
def _starts(size, tile, step):
    if size <= tile:
        return [0]
    s = list(range(0, size - tile + 1, step))
    if s[-1] != size - tile:
        s.append(size - tile)
    return s


def tile_predict_stitch(img_rgb, predict_fn, tile=512, overlap=128, merge_frac=0.5,
                        verbose=True):
    """
    把大图切成 tile×tile（overlap 重叠）逐块 predict，再贪心拼成一张全局实例掩膜。
    贪心策略：先到先得；后续块若与已有实例重叠 >= merge_frac 则并入，否则新 id 填空白像素。
    overlap 建议 >= 2x 细胞直径（本数据集细胞 ~65px，128 足够），避免骑缝细胞被切两半。
    注：这是对稳定 API 的便捷封装；更严谨的大图推理可用官方 SAHI(NMM 合并)。
    """
    H, W = img_rgb.shape[:2]
    gmask = np.zeros((H, W), np.int32)
    nxt = 1
    step = max(1, tile - overlap)
    ys, xs = _starts(H, tile, step), _starts(W, tile, step)
    total = len(ys) * len(xs)
    done = 0
    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            local = np.asarray(predict_fn(img_rgb[y0:y1, x0:x1]))
            if local.ndim > 2:
                local = np.squeeze(local)
            g_sub = gmask[y0:y1, x0:x1]           # 视图，原地改即改 gmask
            for lid in np.unique(local):
                if lid == 0:
                    continue
                region = local == lid
                nz = g_sub[region]
                nz = nz[nz > 0]
                assigned = False
                if nz.size:
                    vals, counts = np.unique(nz, return_counts=True)
                    if counts.max() / region.sum() >= merge_frac:
                        g_sub[region & (g_sub == 0)] = int(vals[counts.argmax()])
                        assigned = True
                if not assigned:
                    fill = region & (g_sub == 0)
                    if fill.sum() > 0:
                        g_sub[fill] = nxt
                        nxt += 1
            done += 1
            if verbose and done % 20 == 0:
                print(f"    [tile] {done}/{total} 块, 已有 {nxt-1} 个实例", flush=True)
    return gmask


# --------------------------------------------------------------------------- #
# CelloType 运行器（模型只加载一次）
# --------------------------------------------------------------------------- #
class CelloTypeRunner:
    def __init__(self, model_path, config_path, confidence, max_det, device,
                 infer="tile", tile=512, overlap=128):
        self.model_path = model_path
        self.config_path = config_path
        self.confidence = confidence
        self.max_det = max_det
        self.device = device
        self.infer = infer
        self.tile = tile
        self.overlap = overlap
        self._predictor = None

    def predictor(self):
        if self._predictor is None:
            from cellotype.predict import CelloTypePredictor
            if not (self.model_path and os.path.exists(self.model_path)):
                raise SystemExit(f"找不到 CelloType 权重：{self.model_path}（用 --model-path 指定）")
            if not (self.config_path and os.path.exists(self.config_path)):
                raise SystemExit(f"找不到 CelloType 配置：{self.config_path}（用 --config-path 指定）")
            print(f"[cellotype] 加载 {os.path.basename(self.model_path)} "
                  f"| conf={self.confidence} max_det={self.max_det} device={self.device}")
            self._predictor = CelloTypePredictor(
                model_path=self.model_path, confidence_thresh=self.confidence,
                max_det=self.max_det, device=self.device, config_path=self.config_path)
        return self._predictor

    def segment(self, img_rgb):
        p = self.predictor()
        if self.infer == "whole":
            m = np.asarray(p.predict(img_rgb))
            return (np.squeeze(m) if m.ndim > 2 else m).astype(np.int32)
        return tile_predict_stitch(img_rgb, lambda s: p.predict(s),
                                   tile=self.tile, overlap=self.overlap).astype(np.int32)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="CosMx 淋巴结：CelloType(TissueNet预训练,零样本) + 对 CosMx CellLabels 的 benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    C.add_common_args(ap, out_dir_default=os.path.join(HERE, "cosmx_lymph_output"))

    ap.add_argument("--model-path", default=DEFAULT_MODEL)
    ap.add_argument("--config-path", default=DEFAULT_CONFIG)
    ap.add_argument("--confidence", type=float, default=0.4)
    ap.add_argument("--max-det", type=int, default=1000,
                    help="每块最大检测数。512² 块内本数据集约 70~90 个细胞，1000 足够")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu-id", default=None,
                    help="设 CUDA_VISIBLE_DEVICES（原 CosMx 版写死了 6，这里改成可选）")
    ap.add_argument("--channel-order", default="opt_mem_nuc",
                    choices=["opt_mem_nuc", "nuc_mem_opt", "mem_nuc_opt"],
                    help="三通道排布，需与 CelloType 预训练一致（拿 example_tissuenet.png 核对）")
    ap.add_argument("--norm", default="pct", choices=["pct", "raw"],
                    help="pct=1~99.5百分位拉到uint8(默认,匹配预训练8-bit量纲)")
    ap.add_argument("--infer", choices=["tile", "whole"], default="tile")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=128,
                    help="切块重叠，建议 >= 2x 细胞直径（本数据集细胞 ~65px）")
    args = ap.parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    runner = None if args.reuse_pred else CelloTypeRunner(
        model_path=args.model_path, config_path=args.config_path,
        confidence=args.confidence, max_det=args.max_det, device=args.device,
        infer=args.infer, tile=args.tile, overlap=args.overlap)

    def segment_fn(chans, mem_channel):
        img = build_cellotype_input(chans, mem_channel,
                                    channel_order=args.channel_order, norm=args.norm)
        return runner.segment(img)

    meta_extra = dict(
        cellotype_model="TissueNet_pretrained", finetuned=False,
        model_path=args.model_path, config_path=args.config_path,
        confidence=args.confidence, max_det=args.max_det,
        infer=args.infer, tile=args.tile, overlap=args.overlap,
        channel_order=args.channel_order, norm=args.norm,
        note_cellotype="CelloType 原生整细胞、零样本；uint16 需按百分位拉到 8-bit；"
                       "4256² 用 512/128 切块拼接（whole 模式会被 Detectron2 缩放而掉小细胞）。")

    C.run_benchmark(args, "CelloType(TissueNet)", segment_fn, meta_extra, "cellotype")


if __name__ == "__main__":
    main()


'''
# 1) 先在有 zarr 的 env（cellpose / cellsam / deepcell 任一）导出缓存
/data/qiuyijia/anaconda3/envs/cellpose/bin/python \
    /data/qiuyijia/cellotype/CelloType/code/cosmx_lymph.py --use-default-fovs --export-cache

# 2) 再在 cellotype env（没 zarr）里跑
CT=/data/qiuyijia/anaconda3/envs/cellotype/bin/python
nohup $CT /data/qiuyijia/cellotype/CelloType/code/cosmx_lymph.py \
    --use-default-fovs --infer tile --overlay --gpu-id 0 \
    --out-dir /data/qiuyijia/cellotype/CelloType/code/cosmx_lymph_output \
    > /data/qiuyijia/cellotype/CelloType/code/cosmx_lymph.log 2>&1 &
'''
