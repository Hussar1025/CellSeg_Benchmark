#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmx_lymph.py  (StarDist)
==========================
在 CosMx 人淋巴结数据集（/data/qiuyijia/dataset/cosmx_lymph_node）上跑 StarDist
（2D_versatile_fluo 预训练，零样本，不微调），以数据集自带的 CosMx CellLabels 为 GT
做 benchmark。数据读取 / FOV 选择 / 指标 / 聚合全部走 /data/qiuyijia/eval/cosmx_lymph_common.py，
与 cellpose / cellotype / cellsam / mesmer 四个版本逐行同源 —— 方法之间唯一差别就是推理。

数据集事实（已实测）
--------------------
* 数据集**自带 FOV 划分**：共 **400 个 FOV**，每个 4256x4256 @0.1203um/px。
  napari OME-Zarr 是拼好的全片(93618x102128)，本脚本按 .zattrs 的 fov_offsets 裁回单 FOV。
* metadata 共 1,852,946 个细胞，中位 **4897 cells/FOV** → **凑 1w 细胞只要 2 个 FOV**。
* 默认 FOV = (48, 261)：密度最接近全片中位数、且分处切片前后两半，GT 4898+4896=9794。

StarDist 的口径问题（必须写进论文）
----------------------------------
StarDist 预训练是**细胞核**模型，而 CellLabels 是**整细胞**。核比细胞小，直接比会
系统性偏低 —— 那是"分割对象不同"，不是 StarDist 差。所以给两种口径：

  --target cell   （默认，与 Cellpose/CelloType/CellSAM/Mesmer 同 GT，可横向比）
      核 → skimage.expand_labels 膨胀 --expand-px 近似整细胞 → 对 CellLabels 比。
      **本数据集的 expand-px 是从 GT 几何标定出来的，不是拍脑袋**：
      metadata 的 Area/NucArea 在默认 2 个 FOV 上（n=9185）给出
        细胞等效半径中位 32.9 px，核等效半径中位 25.2 px，逐细胞差值中位 **7.4 px**
        （NucArea/Area 中位 0.60）→ 默认 --expand-px 7（≈0.84um）。
      注意这是"StarDist + 膨胀"的联合表现，膨胀这一步必须如实报告。
  --target nuclei （对 StarDist 最公平、无膨胀超参）
      需要核 GT。但**本淋巴结数据集只有 labels(整细胞)，没有 CompartmentLabels**，
      所以本脚本用 metadata 的 NucArea 做**面积等价核 GT**：把每个 GT 细胞按距边界的
      距离从外往内剥，剥到剩余面积 ≈ 该细胞 NucArea 为止（--nuclei-from area_shrink）。
      这是近似核 GT，只用于"体现 StarDist 核分割能力"，不能和整细胞口径混着比。

依赖：/data/qiuyijia/anaconda3/envs/stardist（stardist 0.9.2 + tensorflow）
      该 env **没装 zarr**，所以要先在带 zarr 的 env 里导一次缓存（见文件末尾用法）。
"""

import os
import sys
import argparse

import numpy as np

sys.path.insert(0, "/data/qiuyijia/eval")
import cosmx_lymph_common as C

HERE = os.path.dirname(os.path.abspath(__file__))

# 由 GT 几何（Area vs NucArea, n=9185, FOV 48/261）标定：逐细胞等效半径差中位 7.4px
DEFAULT_EXPAND_PX = 7


# --------------------------------------------------------------------------- #
# StarDist 推理
# --------------------------------------------------------------------------- #
class StarDistRunner:
    """模型只加载一次（多 FOV 循环里反复 from_pretrained 很浪费）。"""

    def __init__(self, model_name="2D_versatile_fluo", prob_thresh=None,
                 nms_thresh=None, n_tiles=(8, 8)):
        self.model_name = model_name
        self.prob_thresh = prob_thresh
        self.nms_thresh = nms_thresh
        self.n_tiles = n_tiles
        self._model = None

    def model(self):
        if self._model is None:
            from stardist.models import StarDist2D
            self._model = StarDist2D.from_pretrained(self.model_name)
            print(f"[stardist] model={self.model_name} prob={self.prob_thresh} "
                  f"nms={self.nms_thresh} n_tiles={self.n_tiles}")
        return self._model

    def run(self, dna2d):
        from csbdeep.utils import normalize
        m = self.model()
        # StarDist 官方推荐的 1~99.8 百分位归一化（这是方法自带的预处理，不是我们加的）
        x = normalize(np.asarray(dna2d, dtype=np.float32), 1, 99.8, axis=(0, 1))
        n_tiles = self.n_tiles
        if n_tiles is None:
            try:
                n_tiles = m._guess_n_tiles(x)
            except Exception:
                n_tiles = None
        labels, _ = m.predict_instances(x, prob_thresh=self.prob_thresh,
                                        nms_thresh=self.nms_thresh, n_tiles=n_tiles)
        labels = np.asarray(labels)
        if labels.ndim > 2:
            labels = np.squeeze(labels)
        return labels.astype(np.int32, copy=False)


def nuclei_to_cells(nuclei, expand_px):
    """核 → 近似整细胞：各向外膨胀 expand_px，遇到相邻 label 即停（不会互相吃）。"""
    if not expand_px or expand_px <= 0:
        return nuclei
    from skimage.segmentation import expand_labels
    return expand_labels(nuclei, distance=expand_px).astype(nuclei.dtype)


# --------------------------------------------------------------------------- #
# 核 GT 近似构造（淋巴结数据集没有 CompartmentLabels，用 metadata NucArea 回推）
# --------------------------------------------------------------------------- #
def build_nuclei_gt_from_area(gt_cell, fov, data_root):
    """
    用 metadata 的 NucArea 做**面积等价核 GT**：每个 GT 细胞按"距边界距离"从外往内剥，
    剥到剩余面积 == 该细胞 NucArea 为止（保留细胞内最"深"的那 NucArea 个像素）。
    配对方式：像素面积在 FOV 内唯一的细胞可与 metadata 的 Area 列无歧义对上
    （cellpose/cosmx_lymph.py 的 verify_alignment 已验证两者逐个精确相等）。
    对不上的细胞按全局 NucArea/Area 中位比例(0.60)回退，保证不丢实例。
    这是**近似**核 GT，只用于体现 StarDist 核分割能力，不能和整细胞口径混着比。
    """
    import pandas as pd
    from scipy.ndimage import distance_transform_edt, find_objects

    md_path = os.path.join(data_root, "flat_files", "S0_metadata_file.csv.gz")
    if not os.path.exists(md_path):
        raise FileNotFoundError(f"需要 metadata 来构造核 GT：{md_path}")
    md = pd.read_csv(md_path, usecols=["fov", "Area", "NucArea"])
    md = md[(md.fov == fov) & (md.Area > 0) & (md.NucArea > 0)]
    if md.empty:
        raise ValueError(f"FOV {fov} 在 metadata 中无可用数据")

    # 每个 GT label 的像素面积
    n_lab = int(gt_cell.max())
    area_px = np.bincount(gt_cell.ravel(), minlength=n_lab + 1)

    # 面积 -> NucArea（只用面积在两侧都唯一的，避免歧义配对）
    ma = md.Area.value_counts()
    lab_area_counts = np.bincount(area_px[1:], minlength=int(area_px.max()) + 1) \
        if n_lab > 0 else np.zeros(1, dtype=np.int64)
    area_to_nuc = {int(r.Area): int(r.NucArea) for _, r in md.iterrows()
                   if ma[r.Area] == 1
                   and r.Area < len(lab_area_counts) and lab_area_counts[int(r.Area)] == 1}

    frac_fallback = float(np.median(md.NucArea / md.Area))   # ≈0.60
    objs = find_objects(gt_cell)
    nuc_gt = np.zeros_like(gt_cell)
    matched = fallback = 0

    for lab in range(1, n_lab + 1):
        sl = objs[lab - 1]
        if sl is None:
            continue
        cell_area = int(area_px[lab])
        if cell_area <= 0:
            continue
        nuc_area = area_to_nuc.get(cell_area)
        if nuc_area is None:
            nuc_area = max(1, int(round(cell_area * frac_fallback)))
            fallback += 1
        else:
            matched += 1
        if nuc_area >= cell_area:
            nuc_gt[sl][gt_cell[sl] == lab] = lab
            continue
        sub = gt_cell[sl] == lab
        dist = distance_transform_edt(sub)
        d = dist[sub]
        thresh = np.partition(d, d.size - nuc_area)[d.size - nuc_area]   # 第 nuc_area 大
        pick = sub & (dist >= thresh)
        nuc_gt[sl][pick] = lab

    print(f"  [nuclei_gt] 面积唯一配对={matched} 比例回退={fallback} "
          f"(回退用 NucArea/Area 中位={frac_fallback:.2f})")
    return nuc_gt


# --------------------------------------------------------------------------- #
# segment_fn：chans + mem_channel → pred
# --------------------------------------------------------------------------- #
def make_segment_fn(runner, args):
    def fn(chans, mem_channel):
        nuc, _ = C.nuc_mem(chans, mem_channel)
        nuclei = runner.run(nuc)
        if args.target == "nuclei":
            return nuclei
        return nuclei_to_cells(nuclei, args.expand_px)
    return fn


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="CosMx 淋巴结：StarDist 分割 + 对 CosMx CellLabels 的 benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    C.add_common_args(ap, out_dir_default=os.path.join(HERE, "cosmx_lymph_output"),
                      mem_default="dna_only")

    ap.add_argument("--model", default="2D_versatile_fluo",
                    help="StarDist 预训练模型（荧光核）")
    ap.add_argument("--prob-thresh", type=float, default=None, help="None=用模型默认")
    ap.add_argument("--nms-thresh", type=float, default=None)
    ap.add_argument("--n-tiles", type=int, nargs=2, default=[8, 8],
                    help="大图分块；不想分块就给 None")

    ap.add_argument("--target", choices=["cell", "nuclei"], default="cell",
                    help="cell=核膨胀成整细胞对 CellLabels 比（可与 Cellpose 横向比）; "
                         "nuclei=对近似核 GT 比（对 StarDist 最公平，不含膨胀超参）")
    ap.add_argument("--expand-px", type=int, default=DEFAULT_EXPAND_PX,
                    help=f"target=cell 时核→细胞的膨胀像素。本数据集从 GT 几何标定得 {DEFAULT_EXPAND_PX}px "
                         f"（≈{DEFAULT_EXPAND_PX*0.1203:.2f}um），是逐细胞等效半径差中位值（metadata n=9185）")
    ap.add_argument("--nuclei-from", default="area_shrink",
                    choices=["area_shrink"],
                    help="target=nuclei 时构造核 GT 的方法（数据集没有 CompartmentLabels）")

    args = ap.parse_args()

    if args.n_tiles and (args.n_tiles[0] <= 0 or args.n_tiles[1] <= 0):
        args.n_tiles = None

    runner = None if args.reuse_pred else StarDistRunner(
        model_name=args.model, prob_thresh=args.prob_thresh,
        nms_thresh=args.nms_thresh, n_tiles=tuple(args.n_tiles) if args.n_tiles else None)

    tag = f"stardist_{args.target}" + (f"_exp{args.expand_px}" if args.target == "cell" else "")
    meta_extra = dict(
        stardist_model=args.model, target=args.target,
        expand_px=(args.expand_px if args.target == "cell" else None),
        nuclei_gt_method=(args.nuclei_from if args.target == "nuclei" else None),
        note_stardist="StarDist 预训练是核模型，零样本。cell 口径含膨胀超参（需报告）；"
                      "nuclei 口径用 metadata NucArea 做面积等价核 GT（近似）。"
    )

    # nuclei 口径才需要把整细胞 GT 换成近似核 GT；cell 口径 gt_transform=None
    gt_transform = None
    if args.target == "nuclei":
        def gt_transform(gt_cell, fov):
            return build_nuclei_gt_from_area(gt_cell, fov, args.data_root)

    C.run_benchmark(args, f"StarDist({args.model}, {args.target}口径)",
                    make_segment_fn(runner, args), meta_extra, tag,
                    gt_transform=gt_transform)


if __name__ == "__main__":
    main()


'''
# 先在有 zarr 的 env（cellpose）里导出缓存
/data/qiuyijia/anaconda3/envs/cellpose/bin/python /data/qiuyijia/stardist/cosmx_lymph.py \
    --use-default-fovs --export-cache

# 再在 stardist env（没 zarr）里跑
/data/qiuyijia/anaconda3/envs/stardist/bin/python /data/qiuyijia/stardist/cosmx_lymph.py \
    --use-default-fovs --target cell --expand-px 7 \
    --out-dir /data/qiuyijia/stardist/cosmx_lymph_output --overlay
'''

