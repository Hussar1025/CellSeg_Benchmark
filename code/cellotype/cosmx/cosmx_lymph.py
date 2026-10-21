#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmx_lymph.py (CelloType)
===========================
Run CelloType (TissueNet pretrained, zero-shot, no fine-tuning) whole-cell
segmentation on the CosMx human lymph node dataset and benchmark against the
dataset's own CosMx CellLabels as GT. Data loading / FOV selection / metrics /
aggregation all go through cosmx_lymph_common.py, line-for-line identical to
the cellpose / cellsam / mesmer / stardist versions -- the only difference
between methods is the inference step.

Dataset facts (measured)
------------------------
* The dataset ships its own FOV split: 400 FOVs, each 4256x4256 @0.1203um/px.
  The napari OME-Zarr is a stitched whole slide (93618x102128); this script
  crops single FOVs back out using fov_offsets from .zattrs.
* metadata has 1,852,946 cells, median 4897 cells/FOV -> 2 FOVs give ~1w cells.
* Default FOVs = (48, 261): densities closest to the whole-slide median and
  one in each half of the slide, GT 4898+4896=9794.

Why no fine-tuning (fairness)
-----------------------------
Cellpose(cpsam) / StarDist / CellSAM / Mesmer are all zero-shot in this
benchmark; CelloType also zero-shot keeps all five methods on the same
footing. Only FOVs from the same slide are available; train/test on the same
data systematically overestimates. GT itself is NanoString's algorithm output
(a silver standard); fine-tuning CelloType on it only imitates the vendor.

Three things that must be right
-------------------------------
1) Dynamic range: zarr is uint16; CelloType reads PNG as uint8, and the
   pretrained weights were trained on 8-bit. So stretch to uint8 by the
   1~99.5 percentile (--norm pct, default). This puts data back into the
   expected units.
2) Channel layout: CelloType/TissueNet convention is [R=optional, G=membrane,
   B=nuclear], default opt_mem_nuc. Nuclear=DNA, membrane=Membrane, identical
   to the other four methods.
3) Tiling: 4256^2 fed whole into Detectron2 gets rescaled by short edge and
   drops small cells, so tile mode is the default: 512 windows + 128 overlap,
   stitched here. Cell equivalent diameter median is ~65px, so overlap=128 >
   2x diameter and seam cells are not split.

Dependencies: cellotype env (CelloType + Detectron2 + MaskDINO); needs
models/tissuenet_model_0019999.pth and configs/maskdino_R50_..._2048.yaml.
That env has no zarr, so export a cache once from an env that has zarr (see
usage at the end of the file). Note: on this machine that env currently fails
to import MultiScaleDeformableAttention with a missing GLIBC_2.34; run on a
machine where Detectron2 works (the script itself is environment-agnostic).
"""

import os
import sys
import argparse

import numpy as np

sys.path.insert(0, "yourpath")
import cosmx_lymph_common as C

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MODEL = "yourpath"
DEFAULT_CONFIG = ("yourpath")


# CelloType input (TissueNet convention: R=optional, G=membrane, B=nuclear)
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


# Tiling + instance stitching (depends only on the stable predict(); same recipe
# as the repo's CosMx_cellotype.py)
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
    Cut a large image into tile x tile windows (with overlap), predict each, then
    greedily stitch into a global instance mask.
    Greedy policy: first come first served; later tiles that overlap an existing
    instance by >= merge_frac are merged into it; otherwise a new id fills the
    empty pixels.
    overlap should be >= 2x cell diameter (~65px here; 128 is enough) to avoid
    splitting cells on seams.
    Note: convenient wrapper around the stable API; for stricter large-image
    inference use the official SAHI (NMM merge).
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
            g_sub = gmask[y0:y1, x0:x1]           # view; in-place writes update gmask
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
                print(f"    [tile] {done}/{total} tiles, {nxt-1} instances so far", flush=True)
    return gmask


# CelloType runner (model loaded once)
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
                raise SystemExit(f"CelloType weights not found: {self.model_path} (pass --model-path)")
            if not (self.config_path and os.path.exists(self.config_path)):
                raise SystemExit(f"CelloType config not found: {self.config_path} (pass --config-path)")
            print(f"[cellotype] loading {os.path.basename(self.model_path)} "
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


# main
def main():
    ap = argparse.ArgumentParser(
        description="CosMx lymph node: CelloType (TissueNet pretrained, zero-shot) + benchmark vs CosMx CellLabels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    C.add_common_args(ap, out_dir_default=os.path.join(HERE, "cosmx_lymph_output"))

    ap.add_argument("--model-path", default=DEFAULT_MODEL)
    ap.add_argument("--config-path", default=DEFAULT_CONFIG)
    ap.add_argument("--confidence", type=float, default=0.4)
    ap.add_argument("--max-det", type=int, default=1000,
                    help="max detections per tile. A 512^2 tile holds ~70-90 cells here; 1000 is enough")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu-id", default=None,
                    help="set CUDA_VISIBLE_DEVICES (the original CosMx version hard-coded 6; optional here)")
    ap.add_argument("--channel-order", default="opt_mem_nuc",
                    choices=["opt_mem_nuc", "nuc_mem_opt", "mem_nuc_opt"],
                    help="3-channel layout; must match CelloType pretraining (verify against example_tissuenet.png)")
    ap.add_argument("--norm", default="pct", choices=["pct", "raw"],
                    help="pct=1~99.5 percentile stretched to uint8 (default, matches pretraining 8-bit units)")
    ap.add_argument("--infer", choices=["tile", "whole"], default="tile")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=128,
                    help="tile overlap; >= 2x cell diameter (~65px here)")
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
        note_cellotype="CelloType is natively whole-cell and zero-shot; uint16 must be "
                       "stretched to 8-bit by percentile; 4256^2 uses 512/128 tiling "
                       "(whole mode gets rescaled by Detectron2 and drops small cells).")

    C.run_benchmark(args, "CelloType(TissueNet)", segment_fn, meta_extra, "cellotype")


if __name__ == "__main__":
    main()


'''
# 1) first export the cache in an env that has zarr (cellpose / cellsam / deepcell)
yourpath \
    yourpath --use-default-fovs --export-cache

# 2) then run in the cellotype env (no zarr)
CT=yourpath
nohup $CT yourpath \
    --use-default-fovs --infer tile --overlay --gpu-id 0 \
    --out-dir yourpath \
    > yourpath 2>&1 &
'''