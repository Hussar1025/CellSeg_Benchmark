#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
debug_train_labels.py   v2026.08.14

Find the exact call that empties the training labels, in seconds instead of six
hours.

What is already known
---------------------
    load_train_test_data      label (1024,1024) uint16 max=285 instances=285   OK
    dynamics.labels_to_flows  out channel0 max=285.0 instances=285             OK
    core.py:807  diam_train = [utils.diameters(train_labels[k][0])[0] ...]      all zero
    models.py:264 nmasks = [label[0].max() for label in train_flows]            all zero

So the masks are intact when loaded and intact when labels_to_flows is called
directly, yet empty by the time models.train filters on them.  Between those two
points models.train does exactly two things:

    train_data, train_labels, ... = transforms.reshape_train_test(...)
    train_flows = dynamics.labels_to_flows(train_labels, files=train_files, ...)

This script walks those two steps in order and prints the label statistics after
each, so the culprit is named rather than guessed at.  It loads the same data the
trainer does, touches nothing on disk, and trains nothing.

    conda activate genesegnet_env
    cd yourpath/GeneSegNet/GeneSegNet
    python debug_train_labels.py \
        --train-dir yourpath/genesegnet_xenium/pancreas_cancer/split/train/ \
        --val-dir   yourpath/genesegnet_xenium/pancreas_cancer/split/val/ \
        --chan 2 --n 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def stat(tag, labels, n=6):
    print(f"  {tag}")
    if labels is None:
        print("      None")
        return
    print(f"      list of {len(labels)}")
    for i, lb in enumerate(labels[:n]):
        a = np.asarray(lb)
        ch0 = a[0] if a.ndim == 3 else a
        u = np.unique(ch0)
        u = u[u > 0]
        print(f"      [{i}] shape {a.shape} dtype {a.dtype}  "
              f"ch0.max={float(np.max(ch0)):.1f}  instances={len(u)}")
    if len(labels) > n:
        mx = [float(np.max(np.asarray(lb)[0] if np.asarray(lb).ndim == 3
                          else np.asarray(lb))) for lb in labels]
        z = sum(1 for v in mx if v == 0)
        print(f"      over all {len(labels)}: min {min(mx):.1f}  max {max(mx):.1f}"
              f"  zeros {z}/{len(labels)}"
              + ("   <-- THIS is where they die" if z == len(labels) else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--val-dir", default=None)
    ap.add_argument("--chan", type=int, default=2)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--img-filter", default="_image")
    ap.add_argument("--mask-filter", default="_label")
    ap.add_argument("--heatmap-filter", default="_gaumap_all")
    ap.add_argument("--output-filename", default="newlabels")
    ap.add_argument("--normalize", type=int, default=1)
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.repo).resolve()))
    import Gseg_io, dynamics, transforms  # noqa: E402
    print("=" * 78)
    print("STEP 0  load_train_test_data, the same call the trainer makes")
    print("=" * 78)

    class A:
        pass
    a = A()
    a.img_filter = args.img_filter
    a.mask_filter = args.mask_filter
    a.heatmap_filter = args.heatmap_filter
    a.output_filename = args.output_filename
    a.chan = args.chan
    a.all_channels = True
    a.look_one_level_down = False

    out = Gseg_io.load_train_test_data(
        args.train_dir, 1, a, image_filter=args.img_filter,
        mask_filter=args.mask_filter, heatmap_filter=args.heatmap_filter,
        foldername=args.output_filename)
    images, labels, heatmaps, spots, label_names = out[0], out[1], out[2], out[3], out[4]
    print(f"  images {len(images)}  labels {len(labels)}  heatmaps {len(heatmaps)}")
    print(f"  first image {np.asarray(images[0]).shape} "
          f"{np.asarray(images[0]).dtype}")
    stat("labels straight from load_train_test_data", labels, args.n)

    data = list(np.concatenate((np.array(images), np.array(heatmaps)), axis=3))
    print(f"  concat(images, heatmaps, axis=3) -> first {np.asarray(data[0]).shape}")

    print()
    print("=" * 78)
    print("STEP 1  transforms.reshape_train_test")
    print("=" * 78)
    try:
        import inspect
        print(f"  signature{inspect.signature(transforms.reshape_train_test)}")
        r = transforms.reshape_train_test(data, labels, None, None, None,
                                         bool(args.normalize))
        td, tl = r[0], r[1]
        print(f"  train_data {len(td)}  first {np.asarray(td[0]).shape} "
              f"{np.asarray(td[0]).dtype}")
        stat("labels AFTER reshape_train_test", tl, args.n)
    except Exception as e:
        import traceback
        print(f"  reshape_train_test raised {e.__class__.__name__}: {e}")
        traceback.print_exc()
        td, tl = data, labels

    print()
    print("=" * 78)
    print("STEP 2  dynamics.labels_to_flows, with files= as the trainer passes it")
    print("=" * 78)
    import inspect
    print(f"  signature{inspect.signature(dynamics.labels_to_flows)}")
    # with files=, this WRITES <name>_flows.tif next to the labels; do it on a
    # copy of the list only, and report whether any such file already exists
    pre = sorted(Path(args.train_dir).rglob("*_flows.tif"))
    print(f"  {len(pre)} *_flows.tif already on disk before the call")
    if pre:
        import tifffile
        for p in pre[:3]:
            f = tifffile.imread(p)
            ch0 = f[0] if f.ndim == 3 else f
            u = np.unique(ch0); u = u[u > 0]
            print(f"      {p.name}: {f.shape} ch0.max={float(ch0.max()):.1f} "
                  f"instances={len(u)}")
        print("      !! labels_to_flows may reuse these. If their ch0 is empty, "
              "every run inherits the emptiness until they are deleted.")

    try:
        fl = dynamics.labels_to_flows(tl, files=label_names, use_gpu=False,
                                      device=None)
        stat("flows from labels_to_flows(files=label_names)", fl, args.n)
    except Exception as e:
        import traceback
        print(f"  raised {e.__class__.__name__}: {e}")
        traceback.print_exc()
        fl = None

    print()
    print("=" * 78)
    print("STEP 3  the same call WITHOUT files=, to separate computation from I/O")
    print("=" * 78)
    try:
        fl2 = dynamics.labels_to_flows(tl, use_gpu=False, device=None)
        stat("flows from labels_to_flows(no files=)", fl2, args.n)
    except Exception as e:
        print(f"  raised {e.__class__.__name__}: {e}")
        fl2 = None

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)

    def zeros(x):
        if x is None:
            return None
        return sum(1 for lb in x
                   if float(np.max(np.asarray(lb)[0]
                                   if np.asarray(lb).ndim == 3
                                   else np.asarray(lb))) == 0)

    z0, z1, z2, z3 = zeros(labels), zeros(tl), zeros(fl), zeros(fl2)
    n = len(labels)
    print(f"  loaded              zeros {z0}/{n}")
    print(f"  after reshape       zeros {z1}/{n}")
    print(f"  labels_to_flows     zeros {z2}/{n}" if z2 is not None else
          "  labels_to_flows     failed")
    print(f"  same, no files=     zeros {z3}/{n}" if z3 is not None else
          "  same, no files=     failed")
    print()
    if z0 == 0 and z1 and z1 == n:
        print("  reshape_train_test is emptying the labels.")
    elif (z1 or 0) == 0 and z2 == n and (z3 or 0) == 0:
        print("  labels_to_flows only fails when files= is passed, so it is "
              "reusing stale *_flows.tif. Delete them and rerun; the trainer "
              "should then work.")
    elif (z1 or 0) == 0 and z2 == n and z3 == n:
        print("  labels_to_flows itself empties them regardless of files=. "
              "Check what it does with a label whose dtype/ndim is as printed "
              "in STEP 1.")
    elif z0 == 0 and (z1 or 0) == 0 and (z2 or 0) == 0:
        print("  nothing is empty here, so the trainer's failure comes from "
              "arguments this script does not reproduce. Print min_train_masks "
              "and the label_names it receives.")
    else:
        print("  see the per-step counts above.")


if __name__ == "__main__":
    main()