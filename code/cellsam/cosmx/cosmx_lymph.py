#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmx_lymph.py  (CellSAM)
=========================
Run CellSAM (generalist foundation model, zero-shot, no training) for whole-cell segmentation
on the CosMx human lymph node dataset (yourpath/dataset/cosmx_lymph_node), benchmarked against
the dataset's own CosMx CellLabels as GT. Data loading / FOV selection / metrics / aggregation
all go through yourpath/eval/cosmx_lymph_common.py, line-for-line shared with the cellpose /
cellotype / mesmer / stardist versions -- the only per-method difference is the inference step.

Dataset facts (measured)
------------------------
* The dataset **ships its own FOV grid**: **400 FOVs** in total, each 4256x4256 @0.1203um/px.
  napari OME-Zarr holds the stitched whole slide (93618x102128); this script crops single FOVs
  back out via fov_offsets in .zattrs.
* metadata lists 1,852,946 cells, median **4897 cells/FOV** -> **2 FOVs already give ~10k cells**.
* Default FOVs = (48, 261): density closest to the whole-slide median, one in each half of the
  section, GT 4898+4896=9794.

About CellSAM
-------------
CellSAM = CellFinder (detection boxes) + SAM (mask decoder), a generalist foundation model that
segments across modalities zero-shot and natively does whole-cell segmentation -> it is compared
directly against whole-cell CellLabels, with no StarDist-style dilation hyperparameter, and is
horizontally comparable to Cellpose / CelloType / Mesmer under the same scope.

Two things that must be right
-----------------------------
1) **Dynamic range**: zarr stores uint16, but the CellSAM weights were trained on 8-bit images.
   Feeding uint16 raw hands it a scale it has never seen, so stretch the 1~99.5 percentile to
   uint8 (--norm pct, the default). This is "putting the data back on the scale the method
   expects", not parameter tuning. --norm raw disables it for a control run.
2) **Tiling**: 4256^2 does not fit in one pass, so use the whole-slide tiling mode of the
   official cellsam_pipeline(use_wsi=True) (equivalent to cellpose n_tiles / cellotype SAHI).

The channel layout follows the default blank_nuc_mem = [0, nucleus, membrane] of this repo's
CosMx_cellsam.py. nucleus=DNA, membrane=Membrane, fully identical to the other four methods.

Dependency: yourpath/anaconda3/envs/cellsam (cellSAM, already ships zarr)
"""

import os
import sys
import argparse

import numpy as np

sys.path.insert(0, "yourpath/eval")
import cosmx_lymph_common as C

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# CellSAM input
# --------------------------------------------------------------------------- #
def build_cellsam_input(chans, mem_channel, channel_order="blank_nuc_mem", norm="pct"):
    """
    nucleus = DNA, membrane = Membrane (or memmax).
    norm=pct  -> stretch 1~99.5 percentile to uint8 (default; matches CellSAM's 8-bit pretraining scale)
    norm=raw  -> as-is float32 (for control runs)
    channel_order:
      blank_nuc_mem -> (H,W,3) = [0, nucleus, membrane]  (default; same as this repo's CosMx version)
      nuc_mem       -> (H,W,2) = [nucleus, membrane]
      mem_nuc       -> (H,W,2) = [membrane, nucleus]
      nuc_only      -> (H,W)   nucleus only (conservative fallback)
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
# CellSAM runner (weights loaded once)
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
                print("[cellsam] old version: falling back to segment_cellular_image (no tiling; 4256^2 may OOM)")

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
                # some versions use different arg names / auto-detect device internally
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
        description="CosMx lymph node: CellSAM (zero-shot) + benchmark against CosMx CellLabels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    C.add_common_args(ap, out_dir_default=os.path.join(HERE, "cosmx_lymph_output"))

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--use-wsi", dest="use_wsi", action="store_true", default=True,
                    help="whole-slide tiling mode (required for large 4256^2 FOVs)")
    ap.add_argument("--no-use-wsi", dest="use_wsi", action="store_false")
    ap.add_argument("--block-size", type=int, default=None,
                    help="block side length for use_wsi; omit to use the CellSAM default")
    ap.add_argument("--low-contrast-enhancement", action="store_true")
    ap.add_argument("--gauge-cell-size", action="store_true")
    ap.add_argument("--channel-order", default="blank_nuc_mem",
                    choices=["blank_nuc_mem", "nuc_mem", "mem_nuc", "nuc_only"],
                    help="nucleus/membrane channel layout (same default as this repo's CosMx_cellsam.py)")
    ap.add_argument("--norm", default="pct", choices=["pct", "raw"],
                    help="pct=stretch 1~99.5 percentile to uint8 (default; matches the 8-bit pretraining scale); raw=raw uint16")
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
        note_cellsam="CellSAM is natively whole-cell and zero-shot; uint16 must be percentile-stretched to 8-bit "
                     "to match the pretraining scale; 4256^2 uses use_wsi tiling.")

    C.run_benchmark(args, "CellSAM", segment_fn, meta_extra, "cellsam")


if __name__ == "__main__":
    main()


'''
# the cellsam env ships zarr, so this can run directly
CS=yourpath/anaconda3/envs/cellsam/bin/python

nohup $CS yourpath/CellSam/cosmx_lymph.py \
    --use-default-fovs --use-wsi --overlay \
    --out-dir yourpath/CellSam/cosmx_lymph_output \
    > yourpath/CellSam/cosmx_lymph.log 2>&1 &
'''
