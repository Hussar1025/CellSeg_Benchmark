#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmx_lymph.py  (Mesmer / DeepCell)
===================================
Run Mesmer (DeepCell, TissueNet pretrained, zero-shot, no training) for
whole-cell segmentation on the CosMx human lymph node dataset
(yourpath/dataset/cosmx_lymph_node), benchmarked against the dataset's own CosMx
CellLabels as GT. Data loading / FOV selection / metrics / aggregation all go
through yourpath/eval/cosmx_lymph_common.py, mirroring the cellpose / cellotype /
cellsam / stardist versions line for line - the only difference between methods
is the inference step.

Dataset facts (measured)
------------------------
* The dataset **ships its own FOV split**: **400 FOVs**, each 4256x4256 @0.1203um/px.
  The napari OME-Zarr is the stitched whole slide (93618x102128); this script
  crops single FOVs back out using fov_offsets from .zattrs.
* Metadata holds 1,852,946 cells, median **4897 cells/FOV** -> **2 FOVs are
  enough for ~10k cells**.
* Default FOV = (48, 261): density closest to the whole-slide median, and they
  fall in the first and second half of the section, GT 4898+4896=9794.

Two things that must be right for Mesmer
----------------------------------------
1) Input is 2-channel [nucleus, membrane], nucleus (DAPI) first. For this dataset:
     nucleus  = DNA channel; membrane = Membrane channel (--mem-channel memmax
     switches to the max of the 4 markers)
   This is exactly the channel construction of the cellpose version (there the
   Membrane single channel already saturates; stacking other markers differs only
   in the third decimal), so all five methods get the same nucleus/membrane info.
2) **image_mpp must be the real data resolution**. Mesmer's training resolution
   is model_mpp=0.5um/px, and predict rescales the image using image_mpp. This
   dataset has scale_um=0.1203 -> pass 0.1203 (the script reads it from .zattrs
   by default instead of hard-coding it, to avoid transcription errors).
   0.1203/0.5 = 0.24, so 4256px is downsampled to ~1022px, right at TissueNet's
   scale. **Do not copy the image_mpp=0.5 used elsewhere for TissueNet**: that
   makes Mesmer think cells are ~4x larger and under-segment badly.

compartment='whole-cell' -> whole-cell, compared directly against CellLabels, the
same scope as the other four methods.
Zero-shot, no training (fairness + avoids same-slide FOV leakage).

Dependencies: yourpath/anaconda3/envs/deepcell (deepcell-tf + tensorflow 2.8.4,
zarr included)
"""

import os
import sys
import argparse

import numpy as np

sys.path.insert(0, "yourpath/eval")
import cosmx_lymph_common as C

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# Mesmer input: [1, H, W, 2] = [nucleus, membrane]
# --------------------------------------------------------------------------- #
def build_mesmer_input(chans, mem_channel, channel_order="nuc_mem"):
    nuclear, membrane = C.nuc_mem(chans, mem_channel)
    if membrane is None:                       # dna_only: zero the membrane channel (Mesmer still needs 2)
        membrane = np.zeros_like(nuclear)
    if channel_order == "mem_nuc":
        two = np.stack([membrane, nuclear], axis=-1)
    else:                                      # nuc_mem (Mesmer convention: nucleus first)
        two = np.stack([nuclear, membrane], axis=-1)
    return two[np.newaxis, ...].astype(np.float32)


# --------------------------------------------------------------------------- #
# Mesmer runner (weights loaded once)
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
            print(f"[mesmer] Loaded | model_mpp(training resolution)={self._app.model_mpp} "
                  f"| image_mpp(your data)={self.image_mpp} | compartment={self.compartment}")
            if abs(self.image_mpp - 0.5) < 1e-6:
                print("  [warn] image_mpp=0.5 is wrong for CosMx! This dataset is 0.1203; "
                      "0.5 causes severe under-segmentation.")
        return self._app

    def segment(self, X):
        kwargs = dict(image_mpp=self.image_mpp, compartment=self.compartment,
                      batch_size=self.batch_size)
        if self.pixel_expansion and self.compartment == "nuclear":
            kwargs["postprocess_kwargs_nuclear"] = {"pixel_expansion": self.pixel_expansion}
        y = np.squeeze(np.asarray(self.app().predict(X, **kwargs)))
        if y.ndim > 2:                # compartment='both' -> [H,W,2]; take whole-cell channel 0
            y = y[..., 0]
        return y.astype(np.int32, copy=False)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="CosMx lymph node: Mesmer (TissueNet pretrained, zero-shot) benchmarked against CosMx CellLabels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    C.add_common_args(ap, out_dir_default=os.path.join(HERE, "cosmx_lymph_output"))

    ap.add_argument("--image-mpp", type=float, default=None,
                    help="Real data resolution (um/px). Defaults to scale_um from .zattrs (=0.1203). Do not use 0.5!")
    ap.add_argument("--compartment", default="whole-cell",
                    choices=["whole-cell", "nuclear", "both"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--pixel-expansion", type=int, default=0,
                    help="Only used with compartment=nuclear: expand nuclei by N pixels")
    ap.add_argument("--channel-order", default="nuc_mem", choices=["nuc_mem", "mem_nuc"],
                    help="Mesmer convention is nucleus first (nuc_mem). Do not change unless the data is really reversed")
    args = ap.parse_args()

    # Read image_mpp from the dataset's own scale_um to avoid transcription errors
    if args.image_mpp is None:
        args.image_mpp = C.read_fov_meta(args.data_root)["scale_um"]
        print(f"[mesmer] image_mpp not given; .zattrs scale_um={args.image_mpp}")

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
        note_mesmer="Mesmer is natively whole-cell and zero-shot; image_mpp is the real data "
                    "resolution (0.1203); Mesmer internally downsamples 0.24x to model_mpp=0.5, its training scale.")

    C.run_benchmark(args, f"Mesmer({args.compartment})", segment_fn,
                    meta_extra, f"mesmer_{args.compartment}")


if __name__ == "__main__":
    main()


'''
# The deepcell env ships zarr, so this runs directly (also export the cache for stardist/cellotype)
DC=yourpath/anaconda3/envs/deepcell/bin/python

$DC yourpath/Mesmer/cosmx_lymph.py --use-default-fovs --export-cache

nohup $DC yourpath/Mesmer/cosmx_lymph.py \
    --use-default-fovs --overlay \
    --out-dir yourpath/Mesmer/cosmx_lymph_output \
    > yourpath/Mesmer/cosmx_lymph.log 2>&1 &
'''
