#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmx_lymph.py  (StarDist)
==========================
Runs StarDist (2D_versatile_fluo pretrained, zero-shot, no fine-tuning) on the
CosMx human lymph node dataset (yourpath/dataset/cosmx_lymph_node), benchmarking
against the dataset's built-in CosMx CellLabels as GT. Data loading / FOV selection /
metrics / aggregation all go through yourpath/eval/cosmx_lymph_common.py, line-by-line
identical to the cellpose / cellotype / cellsam / mesmer versions - the only difference
between methods is inference.

Dataset facts (measured)
--------------------
* The dataset ships with its own FOV split: 400 FOVs total, each 4256x4256 @0.1203um/px.
  The napari OME-Zarr is the stitched full slide (93618x102128); this script crops
  single FOVs back via the .zattrs fov_offsets.
* metadata has 1,852,946 cells total, median 4897 cells/FOV -> just 2 FOVs reach ~10k cells.
* default FOV = (48, 261): density closest to the full-slide median, split across the
  first/second halves of the slide; GT 4898+4896=9794.

StarDist target issue (must be documented in the paper)
----------------------------------
StarDist's pretrained model is a nucleus model, while CellLabels is whole-cell.
Nuclei are smaller than cells, so a direct comparison is systematically lower - that
is "different segmentation objects", not StarDist being worse. So two targets exist:

  --target cell   (default, same GT as Cellpose/CelloType/CellSAM/Mesmer, cross-comparable)
      nuclei -> skimage.expand_labels expansion by --expand-px to approximate whole
      cells -> compare to CellLabels.
      **this dataset's expand-px is calibrated from GT geometry, not guessed**:
      metadata Area/NucArea over the default 2 FOVs (n=9185) give
        median cell equivalent radius 32.9 px, median nucleus equivalent radius 25.2 px,
        per-cell difference median 7.4 px (median NucArea/Area 0.60) -> default
        --expand-px 7 (~0.84um).
      Note this is the joint "StarDist + expansion" behavior; the expansion step must
      be reported honestly.
  --target nuclei (fairest to StarDist, no expansion hyperparameter)
      needs nucleus GT. But this lymph node dataset only has labels (whole-cell), no
      CompartmentLabels, so this script builds an area-equivalent nucleus GT from
      metadata NucArea: shrink each GT cell inward from the boundary by distance until
      the remaining area ~= that cell's NucArea (--nuclei-from area_shrink).
      This is an approximate nucleus GT, only to "reflect StarDist's nuclear
      segmentation ability"; it cannot be compared against the whole-cell target.

Dependency: yourpath/anaconda3/envs/stardist (stardist 0.9.2 + tensorflow)
      That env does NOT have zarr installed, so first export the cache once in a
      zarr-enabled env (see usage at the end of this file).
"""

import os
import sys
import argparse

import numpy as np

sys.path.insert(0, "yourpath/eval")
import cosmx_lymph_common as C

HERE = os.path.dirname(os.path.abspath(__file__))

# calibrated from GT geometry (Area vs NucArea, n=9185, FOV 48/261): per-cell
# equivalent radius difference median 7.4px
DEFAULT_EXPAND_PX = 7


# --------------------------------------------------------------------------- #
# StarDist inference
# --------------------------------------------------------------------------- #
class StarDistRunner:
    """Load the model only once (repeated from_pretrained in the multi-FOV loop is wasteful)."""

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
        # StarDist's recommended 1~99.8 percentile normalization (part of the method's
        # own preprocessing, not added by us)
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
    """Nuclei -> approximate whole cells: dilate each outward by expand_px, stopping at neighboring labels (no mutual encroachment)."""
    if not expand_px or expand_px <= 0:
        return nuclei
    from skimage.segmentation import expand_labels
    return expand_labels(nuclei, distance=expand_px).astype(nuclei.dtype)


# --------------------------------------------------------------------------- #
# approximate nucleus GT construction (lymph node dataset has no CompartmentLabels;
# back out via metadata NucArea)
# --------------------------------------------------------------------------- #
def build_nuclei_gt_from_area(gt_cell, fov, data_root):
    """
    Build an area-equivalent nucleus GT from metadata NucArea: shrink each GT cell
    inward by "distance to boundary" until the remaining area == that cell's NucArea
    (keep the deepest NucArea pixels inside the cell).
    Matching: cells whose pixel area is unique within the FOV pair unambiguously to
    the metadata Area column (cellpose/cosmx_lymph.py's verify_alignment has verified
    the two are exactly equal cell by cell).
    Unmatched cells fall back to the global NucArea/Area median ratio (0.60) so no
    instance is lost.
    This is an approximate nucleus GT, only to reflect StarDist's nuclear segmentation
    ability; it cannot be compared against the whole-cell target.
    """
    import pandas as pd
    from scipy.ndimage import distance_transform_edt, find_objects

    md_path = os.path.join(data_root, "flat_files", "S0_metadata_file.csv.gz")
    if not os.path.exists(md_path):
        md_path = os.path.join(data_root, "S0_metadata_file.csv.gz")
    if not os.path.exists(md_path):
        raise FileNotFoundError(f"metadata is needed to build nucleus GT: {md_path}")
    md = pd.read_csv(md_path, usecols=["fov", "Area", "NucArea"])
    md = md[(md.fov == fov) & (md.Area > 0) & (md.NucArea > 0)]
    if md.empty:
        raise ValueError(f"FOV {fov} has no usable data in metadata")

    # pixel area of each GT label
    n_lab = int(gt_cell.max())
    area_px = np.bincount(gt_cell.ravel(), minlength=n_lab + 1)

    # area -> NucArea (only where area is unique on both sides, avoiding ambiguous pairing)
    ma = md.Area.value_counts()
    lab_area_counts = np.bincount(area_px[1:], minlength=int(area_px.max()) + 1) \
        if n_lab > 0 else np.zeros(1, dtype=np.int64)
    area_to_nuc = {int(r.Area): int(r.NucArea) for _, r in md.iterrows()
                   if ma[r.Area] == 1
                   and r.Area < len(lab_area_counts) and lab_area_counts[int(r.Area)] == 1}

    frac_fallback = float(np.median(md.NucArea / md.Area))   # ~0.60
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
        thresh = np.partition(d, d.size - nuc_area)[d.size - nuc_area]   # nuc_area-th largest
        pick = sub & (dist >= thresh)
        nuc_gt[sl][pick] = lab

    print(f"  [nuclei_gt] area-unique matched={matched} ratio-fallback={fallback} "
          f"(fallback uses NucArea/Area median={frac_fallback:.2f})")
    return nuc_gt


# --------------------------------------------------------------------------- #
# segment_fn: chans + mem_channel -> pred
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
        description="CosMx lymph node: StarDist segmentation + benchmark against CosMx CellLabels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    C.add_common_args(ap, out_dir_default=os.path.join(HERE, "cosmx_lymph_output"),
                      mem_default="dna_only")

    ap.add_argument("--model", default="2D_versatile_fluo",
                    help="StarDist pretrained model (fluorescence nucleus)")
    ap.add_argument("--prob-thresh", type=float, default=None, help="None=use model default")
    ap.add_argument("--nms-thresh", type=float, default=None)
    ap.add_argument("--n-tiles", type=int, nargs=2, default=[8, 8],
                    help="tiling for large images; pass None to disable")

    ap.add_argument("--target", choices=["cell", "nuclei"], default="cell",
                    help="cell=expand nuclei to whole cells and compare to CellLabels "
                         "(cross-comparable with Cellpose); nuclei=compare to approximate "
                         "nucleus GT (fairest to StarDist, no expansion hyperparameter)")
    ap.add_argument("--expand-px", type=int, default=DEFAULT_EXPAND_PX,
                    help=f"nucleus->cell expansion pixels for target=cell. Calibrated from this "
                         f"dataset's GT geometry to {DEFAULT_EXPAND_PX}px "
                         f"(~{DEFAULT_EXPAND_PX*0.1203:.2f}um), the per-cell equivalent radius "
                         f"difference median (metadata n=9185)")
    ap.add_argument("--nuclei-from", default="area_shrink",
                    choices=["area_shrink"],
                    help="method to build nucleus GT for target=nuclei (dataset has no CompartmentLabels)")

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
        note_stardist="StarDist pretrained is a nucleus model, zero-shot. The cell target "
                      "includes the expansion hyperparameter (report it); the nuclei target "
                      "uses metadata NucArea as an area-equivalent nucleus GT (approximate)."
    )

    # only the nuclei target replaces whole-cell GT with approximate nucleus GT;
    # cell target keeps gt_transform=None
    gt_transform = None
    if args.target == "nuclei":
        def gt_transform(gt_cell, fov):
            return build_nuclei_gt_from_area(gt_cell, fov, args.data_root)

    C.run_benchmark(args, f"StarDist({args.model}, target={args.target})",
                    make_segment_fn(runner, args), meta_extra, tag,
                    gt_transform=gt_transform)


if __name__ == "__main__":
    main()


'''
# first export the cache in the zarr-enabled env (cellpose)
yourpath/anaconda3/envs/cellpose/bin/python yourpath/stardist/cosmx_lymph.py \
    --use-default-fovs --export-cache

# then run in the stardist env (no zarr)
yourpath/anaconda3/envs/stardist/bin/python yourpath/stardist/cosmx_lymph.py \
    --use-default-fovs --target cell --expand-px 7 \
    --out-dir yourpath/stardist/cosmx_lymph_output --overlay
'''
