#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cellist_fix_origin.py - Fix the issue where detect_swap_scale does not
subtract the coordinate origin

Symptom:
  E14.5_E1S3 full image 22239x15359, GEM x max=32999 > 22239
  -> the script assumes a 1.48x upscale, direction mismatch 106%/43.5%, exits

Cause:
  Stereo-seq GEM coordinates carry a global offset (E14.5_E1S3 origin
  28701,13125); detect_swap_scale only uses sx.max()/sy.max() without
  subtracting min. Recomputing from span: with swap=True, W 0.250 / H 0.280,
  11% difference; scale~0.26 means ssDNA is 0.125um/px while the bin is
  0.5um -- exactly 4x.

Changes:
  1. detect_swap_scale takes spans instead of maxima
  2. load_crop passes (max-min) and records min as the GEM origin in the cache
  3. All bin->px conversions add the GEM origin offset

Usage:
  python cellist_fix_origin.py                    # fix stereo_2 and 3
  python cellist_fix_origin.py --files a.py b.py
"""
import argparse, re, shutil, sys
from pathlib import Path

DEFAULT = ["yourpath",
           "yourpath"]

OLD_DETECT_CALL = """    transposed, scale = detect_swap_scale(
        float(sx.max()), float(sy.max()), H, W)"""

NEW_DETECT_CALL = """    # Stereo-seq GEM coordinates carry a global offset; subtract the origin
    # before comparing spans
    gx0, gy0 = float(sx.min()), float(sy.min())
    span_x, span_y = float(sx.max()) - gx0, float(sy.max()) - gy0
    P(f"    GEM origin ({gx0:.0f},{gy0:.0f})  span {span_x:.0f}x{span_y:.0f} bin")
    transposed, scale = detect_swap_scale(span_x, span_y, H, W)"""

OLD_CROP = """    sx_px = sx / scale
    sy_px = sy / scale"""

NEW_CROP = """    sx_px = (sx - gx0) / scale
    sy_px = (sy - gy0) / scale"""

OLD_SAVE = """    ox_bin = cx0 * scale
    oy_bin = cy0 * scale
    np.savez_compressed(f, crop=crop, ox=ox_bin, oy=oy_bin,
                        transposed=transposed, bin_scale=scale)
    return crop, ox_bin, oy_bin, scale"""

NEW_SAVE = """    # ox/oy back to absolute GEM bin coordinates: crop offset*scale + GEM origin
    ox_bin = cx0 * scale + gx0
    oy_bin = cy0 * scale + gy0
    np.savez_compressed(f, crop=crop, ox=ox_bin, oy=oy_bin,
                        transposed=transposed, bin_scale=scale,
                        gem_x0=gx0, gem_y0=gy0)
    return crop, ox_bin, oy_bin, scale"""

OLD_DOC = '''def detect_swap_scale(gx_max, gy_max, H, W, tol=CONSISTENCY_TOL, label=""):
    """For each of the standard/transposed orientations, invert the required
    scale for the W/H directions and pick the one where both directions agree.
    No assumption that scale is an integer multiple; use the real ratio."""
    cands = []
    for swap in (False, True):
        need_w_raw, need_h_raw = (gy_max, gx_max) if swap else (gx_max, gy_max)'''

NEW_DOC = '''def detect_swap_scale(gx_span, gy_span, H, W, tol=CONSISTENCY_TOL, label=""):
    """For each of the standard/transposed orientations, invert the required
    scale for the W/H directions and pick the one where both directions agree.
    Input must be spans (max-min), not maxima -- Stereo-seq GEM coordinates
    carry a global offset; using maxima gives completely wrong ratios."""
    cands = []
    for swap in (False, True):
        need_w_raw, need_h_raw = (gy_span, gx_span) if swap else (gx_span, gy_span)'''


def patch(path):
    p = Path(path)
    if not p.exists():
        print(f"  [skipped] does not exist: {path}")
        return False
    s = p.read_text()
    orig = s
    hits = []
    for name, old, new in (("detect_swap_scale signature", OLD_DOC, NEW_DOC),
                           ("detection call", OLD_DETECT_CALL, NEW_DETECT_CALL),
                           ("crop coordinates", OLD_CROP, NEW_CROP),
                           ("cache write", OLD_SAVE, NEW_SAVE)):
        if old in s:
            s = s.replace(old, new)
            hits.append(name)
        else:
            print(f"  [no match] {name}")
    # The cache-reuse branch must also read back the gem origin
    s = s.replace(
        '        scale = float(d.get("bin_scale", 1.0))',
        '        scale = float(d.get("bin_scale", 1.0))\n'
        '        gx0 = float(d.get("gem_x0", 0.0))\n'
        '        gy0 = float(d.get("gem_y0", 0.0))')
    if s == orig:
        print(f"  [unchanged] {path}")
        return False
    bak = p.with_suffix(".py.bak_origin")
    shutil.copy2(p, bak)
    p.write_text(s)
    print(f"  OK {path}")
    print(f"    changed: {', '.join(hits)}")
    print(f"    backup: {bak.name}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=DEFAULT)
    a = ap.parse_args()
    print("Fix the coordinate-origin issue in detect_swap_scale")
    print("=" * 60)
    n = sum(patch(f) for f in a.files)
    print("=" * 60)
    print(f"{n} file(s) modified")
    if n:
        print("\nMUST delete the old crop cache, otherwise the wrong offset is read back:")
        print("  rm -f yourpath/cellist/stereo_*/ckpt_scan/crop.npz")
        print("\nThen rerun the scan. Expected:")
        print("  E14.5_E1S3 origin(28701,13125) span 4298x5549")
        print("  swap=True  W 0.250 / H 0.280  difference 11%")
        print("  scale~0.26 -> ssDNA 0.125um/px while bin 0.5um, exactly 4x")
        print("  If it still reports an over-limit difference, widen CONSISTENCY_TOL from 0.10 to 0.15")


if __name__ == "__main__":
    main()