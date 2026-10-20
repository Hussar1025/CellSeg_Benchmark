#!/usr/bin/env python
"""
GeneSegNet x MERFISH input channel diagnostics -- rule out code bugs before concluding "cross-platform failure"

Usage:
  python diag_merfish_input.py
"""
import numpy as np
import tifffile

import sys
sys.path.insert(0, "yourpath/genesegnet")
from run_genesegnet_infer import load_merfish, TARGETS

SEP = "=" * 68
P = lambda *a: print(*a, flush=True)


def main():
    P(SEP); P("MERFISH input channel diagnostics"); P(SEP)
    cfg = TARGETS["merfish"]
    image_2ch, gt_xy_px, n_gt = load_merfish(cfg, 0.5)

    dapi = image_2ch[..., 0]
    hm = image_2ch[..., 1]
    H, W = dapi.shape

    P(f"\n[1] DAPI channel stats")
    P(f"    shape={dapi.shape}  dtype={dapi.dtype}")
    P(f"    mean={dapi.mean():.2f}  nonzero ratio={((dapi>0).mean()*100):.1f}%")
    P(f"    p50={np.percentile(dapi,50):.0f} p95={np.percentile(dapi,95):.0f} "
      f"max={dapi.max()}")

    P(f"\n[2] Heatmap channel stats")
    P(f"    mean={hm.mean():.2f}  nonzero ratio={((hm>0).mean()*100):.1f}%")
    P(f"    p50={np.percentile(hm,50):.0f} p95={np.percentile(hm,95):.0f} "
      f"max={hm.max()}")

    P(f"\n[3] ★ Key check: DAPI brightness at GT centroids vs random positions")
    P(f"    (breast was 3.47x; >2 means truly aligned)")
    gx = gt_xy_px[:, 0].astype(int).clip(0, W-1)
    gy = gt_xy_px[:, 1].astype(int).clip(0, H-1)
    v_gt = dapi[gy, gx].astype(float)
    rng = np.random.default_rng(0)
    v_rand = dapi[rng.integers(0, H, 20000),
                  rng.integers(0, W, 20000)].astype(float)
    ratio = v_gt.mean() / max(v_rand.mean(), 1e-6)
    P(f"    mean at GT centroids={v_gt.mean():.2f}  mean at random positions={v_rand.mean():.2f}")
    P(f"    brightness ratio = {ratio:.2f}x  "
      f"{'✓ aligned, channel itself is fine' if ratio > 2 else '⚠ weak, possible alignment issue' if ratio > 1.3 else '✘ barely above background, channel construction likely buggy'}")

    P(f"\n[4] Same check for the heatmap channel (transcript density should be higher near GT centroids)")
    v_gt_hm = hm[gy, gx].astype(float)
    v_rand_hm = hm[rng.integers(0, H, 20000),
                   rng.integers(0, W, 20000)].astype(float)
    ratio_hm = v_gt_hm.mean() / max(v_rand_hm.mean(), 1e-6)
    P(f"    brightness ratio = {ratio_hm:.2f}x")

    out_dapi = "yourpath/genesegnet_infer/merfish_dapi_check.tif"
    out_hm = "yourpath/genesegnet_infer/merfish_heatmap_check.tif"
    tifffile.imwrite(out_dapi, dapi)
    tifffile.imwrite(out_hm, hm)
    P(f"\n  Both channels saved separately; open them for a visual check:")
    P(f"    {out_dapi}")
    P(f"    {out_hm}")

    P(f"\n{SEP}\nConclusion\n{SEP}")
    if ratio > 2 and ratio_hm > 2:
        P("  ✓ Both channels are aligned normally; the very low detection count is more likely a genuine cross-platform transfer failure")
    else:
        P("  ✘ At least one channel looks misaligned; fix that before re-assessing transfer ability")


if __name__ == "__main__":
    main()