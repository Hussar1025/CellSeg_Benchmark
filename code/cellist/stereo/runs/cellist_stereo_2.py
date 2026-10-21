#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cellist_align_stereo.py v3 - geometric feasibility check + visual verification

Why v2 was discarded:
  Benchmarked on E16.5_E2S6, whose known-correct answer is scale=1.0 / no
  transpose / no flip (that is how stereo_1's historical result was produced).
  But v2's search chose scale=0.164 / flip y, and the top-10 r values all
  clustered at 0.159~0.180, a spread of 0.003.
  Root cause: letting FFT freely search translation over the full image gives
  too many degrees of freedom -- any scale can find a "looks OK" position, and
  a half-image offset like (12832,3072) is evidence of random searching. If
  the scoring function fails on the correct answer, it cannot be used to judge
  E14.5.

v3 approach:
  Keep the old cellist_stereo_run.py assumptions: scale=1, zero translation,
  GEM coordinates used directly as pixel coordinates. Judge only one thing --
  which orientation puts spots inside the image. That is pure geometry, no
  scoring needed. Final decision goes to visual overlay inspection.

Geometric facts:
  E16.5_E2S6  x[2925,7174]  y[4875,10574]   image 27200x16385  -> no transpose needed
  E14.5_E1S3  x[19875,22697] y[4475,10074]  image 22239x15359
              no transpose: x upper bound 22697 > W=15359, 0% coverage -> must transpose
              after transpose: W=22239, x overflows by 458px (1.6%); the old script
              handles this automatically via min(W,..)

Usage:
  python cellist_align_stereo.py --dataset E16.5_E2S6 --preview
  python cellist_align_stereo.py --dataset E14.5_E1S3 --preview
  python cellist_align_stereo.py --dataset E14.5_E1S3 --orient T00 --write --preview
"""
import argparse
import os
import sys

import numpy as np
import tifffile

DATA = "yourpath"
DATASETS = {
    "E14.5_E1S3": "yourpath",
    "E16.5_E2S6": "yourpath",
    "E16.5_E2S7": "yourpath",
}
MARGIN = 50
BIN_UM = 0.5


def P(*a):
    print(*a, flush=True)


def load_spots(ckpt):
    f = os.path.join(ckpt, "gem_full.npz")
    if not os.path.exists(f):
        sys.exit(f"FAIL: missing {f}, run the main script once to generate the GEM cache")
    d = np.load(f, allow_pickle=True)
    sx = d["sx"].astype(np.int64)
    sy = d["sy"].astype(np.int64)
    P(f"  GEM spots {len(sx):,}")
    P(f"    x[{sx.min()},{sx.max()}]  y[{sy.min()},{sy.max()}]")
    P(f"    span {sx.max()-sx.min()} x {sy.max()-sy.min()} bin"
      f" = {(sx.max()-sx.min())*BIN_UM:.0f} x "
      f"{(sy.max()-sy.min())*BIN_UM:.0f} um")
    return sx, sy


def load_img(path):
    img = tifffile.imread(path)
    if img.ndim == 3:
        img = img[0] if img.shape[0] <= 8 else img[..., 0]
    P(f"  ssDNA H={img.shape[0]} W={img.shape[1]} "
      f"{img.dtype} {img.nbytes/1e9:.2f} GB")
    return img


def orient(img, t, fx, fy):
    """Build A such that A[sy, sx] is the pixel where a spot should land. Flips are baked into the array."""
    a = img.T if t else img
    if fy:
        a = a[::-1, :]
    if fx:
        a = a[:, ::-1]
    return a


def geom(sx, sy, H, W):
    """In-image statistics under scale=1, zero translation."""
    ok = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
    over_x = max(0, int(sx.max()) - (W - 1))
    over_y = max(0, int(sy.max()) - (H - 1))
    return float(ok.mean()), over_x, over_y


def contrast(A, sx, sy, n=200_000):
    H, W = A.shape
    ok = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
    if ok.sum() < 100:
        return np.nan
    qx, qy = sx[ok], sy[ok]
    k = min(len(qx), n)
    i = np.random.default_rng(0).choice(len(qx), k, replace=False)
    at = float(A[qy[i], qx[i]].astype(np.float64).mean())
    gl = float(A[::9, ::9].astype(np.float64).mean())
    return at / max(gl, 1e-9)


def to_uint8(a):
    if a.dtype == np.uint8:
        return a
    lo, hi = np.percentile(a, [1, 99.5])
    b = np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    return (b * 255).astype(np.uint8)


def crop_of(A, sx, sy):
    H, W = A.shape
    cy0 = max(0, int(sy.min()) - MARGIN)
    cy1 = min(H, int(sy.max()) + MARGIN + 1)
    cx0 = max(0, int(sx.min()) - MARGIN)
    cx1 = min(W, int(sx.max()) + MARGIN + 1)
    return cx0, cx1, cy0, cy1


def preview(crop, px, py, out, n=60_000):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        P(f"  (skipping preview: {e})")
        return
    h, w = crop.shape
    ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    px, py = px[ok], py[ok]
    k = min(len(px), n)
    i = np.random.default_rng(0).choice(len(px), k, replace=False)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10 * h / max(w, 1) + 1))
    ax[0].imshow(crop, cmap="gray")
    ax[0].set_title("ssDNA")
    ax[1].imshow(crop, cmap="gray")
    ax[1].scatter(px[i], py[i], s=.06, c="red", alpha=.28, linewidths=0)
    ax[1].set_title("ssDNA + GEM spots")
    for a_ in ax:
        a_.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight", pad_inches=.02)
    plt.close(fig)
    P(f"  OK overlay preview {out}")
    P("    Compare side by side: do the red dots trace the same tissue outline as the left image?")
    P("    If red dots fill the whole frame or land on black background -> not aligned")


def write_crop(A, sx, sy, t, ckpt):
    cx0, cx1, cy0, cy1 = crop_of(A, sx, sy)
    if cy1 <= cy0 or cx1 <= cx0:
        sys.exit("FAIL: empty crop box")
    crop = to_uint8(np.ascontiguousarray(A[cy0:cy1, cx0:cx1]))
    inside = float(((sx >= cx0) & (sx < cx1) &
                    (sy >= cy0) & (sy < cy1)).mean())
    H, W = A.shape
    P(f"  crop {crop.shape}  "
      f"{(cy1-cy0)*(cx1-cx0)/(H*W)*100:.1f}% of full image")
    P(f"  spots inside the box {inside*100:.1f}%")
    if inside < 0.95:
        P(f"  WARNING: {(1-inside)*100:.1f}% of spots are outside the box; those nuclei "
          f"cannot be segmented, which lowers recall during evaluation; note this down")
    f = os.path.join(ckpt, "crop.npz")
    np.savez_compressed(f, crop=crop, ox=float(cx0), oy=float(cy0),
                        transposed=bool(t), bin_scale=1.0,
                        gem_x0=0.0, gem_y0=0.0)
    P(f"  OK {f}   ox={cx0} oy={cy0} scale=1.0 swap={t}")
    P("  The main script will take the 'reuse crop cache' branch and skip detect_swap_scale.")
    P("    Do not pass --force to the main script: it would regenerate crop.npz and discard this alignment.")
    return crop, sx - cx0, sy - cy0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--orient", default="",
                    help="specify orientation, e.g. N00 (no transpose, no flip) / T00 (transpose) / T01; "
                         "empty means auto-select the highest coverage")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args()

    ckpt = "yourpath"
    os.makedirs(ckpt, exist_ok=True)
    P("=" * 74)
    P(f"GEM <-> ssDNA geometric check (scale=1, zero translation) -- {a.dataset}")
    P("=" * 74)
    sx, sy = load_spots(ckpt)
    img = load_img(DATASETS[a.dataset])

    P(f"\n  {'orient':>6} {'transpose':>9} {'flipx':>5} {'flipy':>5} "
      f"{'coverage':>8} {'xover':>8} {'yover':>8} {'contrast':>8}")
    P("  " + "-" * 62)
    rows = []
    for t in (False, True):
        for fx in (False, True):
            for fy in (False, True):
                A = orient(img, t, fx, fy)
                H, W = A.shape
                cov, ox_, oy_ = geom(sx, sy, H, W)
                ct = contrast(A, sx, sy) if cov > .05 else np.nan
                tag = ("T" if t else "N") + str(int(fx)) + str(int(fy))
                rows.append(dict(tag=tag, t=t, fx=fx, fy=fy,
                                 cov=cov, ox=ox_, oy=oy_, ct=ct))
                cs = f"{ct:.3f}" if np.isfinite(ct) else "-"
                P(f"  {tag:>6} {str(t):>9} {str(fx):>5} {str(fy):>5} "
                  f"{cov*100:>7.1f}% {ox_:>8} {oy_:>8} {cs:>8}")
                del A

    if a.orient:
        s = a.orient.upper()
        if len(s) != 3:
            sys.exit("FAIL: --orient needs 3 characters, e.g. N00 / T00 / T01")
        b = dict(tag=s, t=s[0] == "T", fx=s[1] == "1", fy=s[2] == "1")
        P(f"\n  Manually specified orientation {s}")
    else:
        cands = [r for r in rows if r["cov"] > .5]
        if not cands:
            sys.exit("\nFAIL: no orientation holds more than half the spots; check this dataset manually")
        # Prefer coverage, then no flip (the old script's default assumption)
        b = max(cands, key=lambda r: (round(r["cov"], 3),
                                      -(int(r["fx"]) + int(r["fy"]))))
        P(f"\n  -> auto-selected {b['tag']} (coverage {b['cov']*100:.1f}%, "
          f"preferring no flip at equal coverage)")

    P("\n  NOTE: the automatic score is not sufficient to decide -- on the known-correct")
    P("    E16.5_E2S6, a correlation-based auto search also picked wrong. Use the")
    P("    overlay image for visual confirmation.")
    P("    Reference: E16.5_E2S6's historically correct configuration is N00 (no transpose, no flip)")

    A = np.ascontiguousarray(orient(img, b["t"], b["fx"], b["fy"]))
    if a.write:
        P("")
        crop, px, py = write_crop(A, sx, sy, b["t"], ckpt)
    else:
        cx0, cx1, cy0, cy1 = crop_of(A, sx, sy)
        crop = to_uint8(np.ascontiguousarray(A[cy0:cy1, cx0:cx1]))
        px, py = sx - cx0, sy - cy0
    if a.preview:
        preview(crop, px, py,
                f"yourpath")
    if not a.write:
        P("\n  After visual confirmation, add --write to save crop.npz")


if __name__ == "__main__":
    main()