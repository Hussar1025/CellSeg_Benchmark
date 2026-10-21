#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DISSECT x cosmx_3 (CosMx 6K human brain, 29 FOVs) - inspect / prep / run / merge in one script

Same approach as dissect_merfish_*.py: DISSECT's xenium branch hard-codes x_location/0.2125
-> image pixels, and detector weights are trained at 0.2125 um/px, so CosMx is posed as xenium.
Differences between CosMx and MERFISH:
  * The analysis region is a set of FOVs, not the central ROI of a full mosaic, so run
    prep / run per FOV and merge the results at the end;
  * Coordinates use FOV-local pixels (x_local_px, y_local_px, about 0.12028 um/px);
  * Per-FOV morphology image: prefer FOV-level images (Morphology2D / CellComposite);
    otherwise use the whole-slide image (DAPI.tif or napari/images). Run --stage register first
    to register cell centers (global px) to the whole-slide image (orientation / scale / offset,
    see wsi_register.py), then sample it onto each FOV's local grid using the local->global affine.
  * wsi_register.py must be in the same directory as this script.

Usage:
  python dissect_cosmx_3.py --stage inspect                 # inspect found files, FOVs, images and channel scores
  python dissect_cosmx_3.py --stage register                # WSI registration (needed without FOV-level images), check QC
  python dissect_cosmx_3.py --stage all  --n-jobs 4
  python dissect_cosmx_3.py --stage prep
  python dissect_cosmx_3.py --stage run  --n-jobs 4
  python dissect_cosmx_3.py --stage merge
Optional: --fovs 51 52 ...  --channel 4  --flip-y no  --image-template 'yourpath/F{fov:03d}.TIF'
"""
import argparse, glob, json, re, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))   # find wsi_register.py in the same directory
import numpy as np, pandas as pd, tifffile

# ===== hard-coded dataset constants =====
TAG       = "cosmx_3"
ROOT_CANDIDATES = ["yourpath/dataset/cosmx_human_brain"]
OUT       = Path("yourpath/dissect_cosmx_human_brain")
# Analysis FOVs (Bering rule reproduction: 29 FOVs, 10,359 cells); None = --fovs required
FOVS      = [4, 18, 51, 52, 56, 65, 74, 87, 90, 101, 104, 110, 124, 131, 147, 173, 187,
             220, 224, 234, 242, 250, 255, 256, 264, 296, 298, 307, 339]
GT_CELLS  = 10359              # reference cells in the analysis region (Tables S1 / S3)
UM_PER_PX = 0.12028             # CosMx FOV pixel size (um/px)
FOV_PX    = 4256                # FOV width/height (local px); checked against transcript extent
TARGET_UM = 0.2125              # DISSECT/xenium detector training scale
ENGINE    = Path("yourpath/DISSECT/code/run_xenium_pancreas_dissect.py")
RUNNER    = Path(__file__).resolve().parent / "dissect_native_runner.py"   # native nanostring entry
CONTROL_PREFIX = ("Negative", "NegPrb", "SystemControl", "FalseCode", "Custom")
WSI_CANDIDATES = ["yourpath/dataset/cosmx_human_brain/Run_*/*/Napari/images/UV",
                  "yourpath/dataset/cosmx_human_brain/Run_*/*/Napari/images",
                  "yourpath/dataset/cosmx_human_brain/**/[Nn]apari/images"]        # WSI (used when no FOV-level images)
FLAT_HINT = None        # if flat files are missing under root, search this keyword under yourpath/dataset

P = lambda *a: print(*a, flush=True)
SEP = "=" * 78


# ---------------------------------------------------------------- file discovery
def find_root():
    for r in ROOT_CANDIDATES:
        if Path(r).exists():
            return Path(r)
    raise SystemExit(f"Data directories not found: {ROOT_CANDIDATES}")


def flat_files(root):
    out = _flat_in(root)
    if FLAT_HINT and ("tx" not in out or "meta" not in out):
        base = Path("yourpath/dataset")
        for key, pat in (("tx", f"*{FLAT_HINT}*tx_file*"), ("meta", f"*{FLAT_HINT}*metadata*"),
                         ("fov", f"*{FLAT_HINT}*fov_positions*")):
            if key in out:
                continue
            for depth in ("*/", "*/*/", "*/*/*/"):
                h = sorted(p for p in base.glob(depth + pat) if p.is_file())
                if h:
                    out[key] = h[0]; break
    return out


def _flat_in(root):
    out = {}
    for key, pats in (("tx", ["*tx_file*.csv*", "*tx_file*"]),
                      ("meta", ["*metadata_file*.csv*", "*metadata_file*"]),
                      ("fov", ["*fov_positions*.csv*", "*fov_positions*"])):
        for pat in pats:
            h = sorted(root.rglob(pat))
            h = [p for p in h if p.is_file() and "_inspect_cache" not in str(p)]
            if h:
                out[key] = h[0]
                break
    return out


def read_meta(root, fovs):
    ff = flat_files(root)
    if "meta" not in ff:
        raise SystemExit("Metadata file not found: *metadata_file*")
    md = pd.read_csv(ff["meta"])
    fcol = "fov" if "fov" in md.columns else "FOV"
    md = md[md[fcol].isin(fovs)].copy()
    md["fov"] = md[fcol]
    return md


def load_tx(root, fovs, cache_dir):
    """Reuse ComSeg's _inspect_cache first, then our cache, then a chunked scan."""
    fovs = sorted(int(f) for f in fovs)
    cands = sorted((root / "_inspect_cache").glob("tx_fov_*.parquet")) + \
        sorted((root / "_inspect_cache").glob("tx_fov_*.csv.gz")) + \
        sorted(cache_dir.glob("tx_fov_*.parquet"))
    for c in cands:
        try:
            df = pd.read_parquet(c) if c.suffix == ".parquet" else pd.read_csv(c)
        except Exception:
            continue
        if "fov" in df.columns and set(fovs) <= set(df["fov"].unique()):
            P(f"  transcript cache: {c}")
            return df[df["fov"].isin(fovs)].reset_index(drop=True)
    ff = flat_files(root)
    if "tx" not in ff:
        raise SystemExit("Transcript file not found: *tx_file*")
    P(f"  chunked scan of {ff['tx']} (scan once, cache afterwards)")
    keep_cols = {"fov", "x_local_px", "y_local_px", "x_global_px", "y_global_px",
                 "target", "gene", "cell_ID", "cell_id", "CellComp"}
    parts, n = [], 0
    for ch in pd.read_csv(ff["tx"], chunksize=5_000_000,
                          usecols=lambda c: c in keep_cols):
        n += len(ch)
        s = ch[ch["fov"].isin(fovs)]
        if len(s):
            parts.append(s)
        if n % 100_000_000 < 5_000_000:
            P(f"    scanned {n:,}")
    df = pd.concat(parts, ignore_index=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    c = cache_dir / f"tx_fov_{'_'.join(map(str, fovs))[:120]}.parquet"
    try:
        df.to_parquet(c, index=False)
    except Exception:
        c = c.with_suffix(".csv.gz"); df.to_csv(c, index=False)
    P(f"  scanned {n:,} rows, kept {len(df):,} -> {c}")
    return df


_FOV_RE = re.compile(r"F0*(\d+)(?=[_.\-])", re.I)


def index_images(root, template=None, fovs=()):
    """Return {fov: path}. Morphology2D TIFs take priority over CellComposite images."""
    if template:
        return {f: Path(template.format(fov=f)) for f in fovs
                if Path(template.format(fov=f)).exists()}
    idx = {}
    pats = ["**/Morphology2D/*.TIF", "**/Morphology2D/*.tif", "**/*Morphology*F*.TIF",
            "**/*Morphology*F*.tif", "**/CellComposite/*.jpg", "**/CellComposite/*.png",
            "**/CellComposite/*.tif", "**/*CellComposite*F*.*"]
    for rank, pat in enumerate(pats):
        for p in glob.glob(str(root / pat), recursive=True):
            m = _FOV_RE.search(Path(p).name)
            if not m:
                continue
            f = int(m.group(1))
            if f not in idx or rank < idx[f][0]:
                idx[f] = (rank, Path(p))
    return {f: v[1] for f, v in idx.items()}


def load_channels(path):
    """Return 2D channels. Supports (C,H,W), (H,W,C), (Z,C,H,W) and RGB images."""
    s = str(path).lower()
    if s.endswith((".jpg", ".jpeg", ".png")):
        try:
            import imageio.v3 as iio
            a = iio.imread(path)
        except Exception:
            from PIL import Image
            a = np.asarray(Image.open(path))
    else:
        a = tifffile.imread(path)
    a = np.squeeze(np.asarray(a))
    if a.ndim == 2:
        return [a]
    if a.ndim == 4:                      # (Z,C,H,W) -> max projection
        a = a.max(axis=0)
    if a.ndim == 3:
        if a.shape[-1] <= 8 and a.shape[0] > 8:
            return [a[..., i] for i in range(a.shape[-1])]
        return [a[i] for i in range(a.shape[0])]
    raise SystemExit(f"Cannot parse image dimensions {a.shape}: {path}")


def score_channels(chans, md_fov, fov_w, fov_h):
    """Signal at platform cell centers / image mean: highest for DAPI with correct orientation."""
    if md_fov is None or not len(md_fov):
        return []
    cx = md_fov["CenterX_local_px"].to_numpy(float)
    cy = md_fov["CenterY_local_px"].to_numpy(float)
    rows = []
    for ci, ch in enumerate(chans):
        h, w = ch.shape
        img = ch.astype(np.float32)
        base = float(img.mean()) + 1e-6
        for flip in (False, True):
            yy = (fov_h - 1 - cy) if flip else cy
            r = np.clip((yy * h / fov_h).astype(int), 0, h - 1)
            c = np.clip((cx * w / fov_w).astype(int), 0, w - 1)
            rows.append(dict(channel=ci, flip_y=flip,
                             enrichment=float(img[r, c].mean()) / base))
    return sorted(rows, key=lambda d: -d["enrichment"])


def fov_extent(tx_f):
    w = max(FOV_PX, int(np.ceil(tx_f["x_local_px"].max() + 1)))
    h = max(FOV_PX, int(np.ceil(tx_f["y_local_px"].max() + 1)))
    return w, h


def resize_to(img, out_hw):
    oh, ow = out_hw
    try:
        import cv2
        return cv2.resize(img.astype(np.float32), (ow, oh), interpolation=cv2.INTER_AREA)
    except Exception:
        from skimage.transform import resize as sk
        return sk(img.astype(np.float32), (oh, ow), preserve_range=True, anti_aliasing=True)


def area_limits(a, um_per_px):
    """Detection area threshold (px^2). Defaults match other DISSECT benchmark runs: 350-1600;
    --area-rule physical converts to the physical equivalent of Xenium (0.2125 um/px). Explicit --minarea/--maxarea win."""
    f = (0.2125 / um_per_px) ** 2 if getattr(a, "area_rule", "fixed") == "physical" else 1.0
    lo = a.minarea if a.minarea is not None else int(round(350 * f))
    hi = a.maxarea if a.maxarea is not None else int(round(1600 * f))
    return lo, hi


def report_box_coverage(areas_px, lo, hi, tag="", um_per_px=None):
    areas_px = np.asarray(areas_px, float)
    areas_px = areas_px[np.isfinite(areas_px) & (areas_px > 0)]
    if not len(areas_px):
        return
    q = np.percentile(areas_px, [5, 25, 50, 75, 95])
    P(f"  Reference cell bounding box area{tag} (px^2): 5%={q[0]:.0f} 25%={q[1]:.0f} median={q[2]:.0f} 75%={q[3]:.0f} 95%={q[4]:.0f}")
    cands = [("current", lo, hi), ("numeric default 350-1600", 350, 1600), ("library default 600-1200", 600, 1200)]
    if um_per_px:
        f = (0.2125 / um_per_px) ** 2
        cands.append(("physical-equivalent Xenium", int(round(350 * f)), int(round(1600 * f))))
    cands.append(("reference 5-95% quantile", int(q[0]), int(q[4])))
    for name, l, h in cands:
        inside = float(((areas_px >= l) & (areas_px <= h)).mean())
        P(f"    {name:<16} [{l:>6},{h:>6}]  coverage {inside:6.1%}")
    inside = float(((areas_px >= lo) & (areas_px <= hi)).mean())
    if inside < 0.5:
        P("  [Warning] Most reference cell bounding boxes are outside the detection area under the current settings; consider --minarea/--maxarea")


def SFX(a):
    return "_native" if a.mode == "native" else ""


def get_fovs(a):
    f = a.fovs or FOVS
    if not f:
        raise SystemExit("No FOVs specified: use --fovs to give the analysis region's FOV IDs")
    return [int(x) for x in f]


# ---------------------------------------------------------------- whole-slide image
def find_wsi():
    """WSI_CANDIDATES supports wildcards (e.g. Run_*/images); return the first existing path."""
    for p in WSI_CANDIDATES:
        hits = sorted(glob.glob(str(p), recursive=True)) if any(ch in str(p) for ch in "*?[") else [str(p)]
        for h in hits:
            if Path(h).exists():
                return Path(h)
    return None


def all_cell_centers(md):
    if {"CenterX_global_px", "CenterY_global_px"} <= set(md.columns):
        return md[["CenterX_global_px", "CenterY_global_px"]].to_numpy(float)
    return None


def fov_affine(t):
    """Local-pixel -> global-pixel affine (least squares); return the 2x3 matrix and residual."""
    A = np.column_stack([t["x_local_px"], t["y_local_px"], np.ones(len(t))]).astype(float)
    B = t[["x_global_px", "y_global_px"]].to_numpy(float)
    M, *_ = np.linalg.lstsq(A, B, rcond=None)
    res = float(np.sqrt(((A @ M - B) ** 2).sum(1)).mean())
    return M.T, res


def _refine_by_fov(wr, pts, lab, w, T, cell_size, log):
    """Refine translation at finer levels on the cell-richest FOVs (rigid) and take the median."""
    fovs, cnt = np.unique(lab, return_counts=True)
    top = fovs[np.argsort(-cnt)][:5]
    for target_px in (1500, 4000):
        offs = []
        uv_all = np.asarray(T["uv_min"], float)
        for f in top:
            p = pts[lab == f]
            # Use each FOV's own uv minimum, shift the offset, refine, then restore the global origin
            uv_f = wr.orient(p, T["orientation"]).min(0)
            d = (uv_f - uv_all) * T["scale"]
            Ts = dict(T, uv_min=uv_f.tolist(), offset_col=T["offset_col"] + d[0], offset_row=T["offset_row"] + d[1])
            Tf = wr.refine(p, w, Ts, cell_size, log=lambda *x: None, rel=0.0, steps=1, target_px=target_px)
            if Tf is not Ts and np.isfinite(Tf.get("z_fine", np.nan)):
                offs.append((Tf["offset_col"] - d[0], Tf["offset_row"] - d[1], Tf["z_fine"]))
        if not offs:
            continue
        o = np.array(offs)
        dc, dr = float(np.median(o[:, 0])), float(np.median(o[:, 1]))
        spread = float(np.max(np.hypot(o[:, 0] - dc, o[:, 1] - dr)))
        log(f"  [register] FOV local refinement (target {target_px} px): {len(o)} FOVs  median offset ({dc:.1f},{dr:.1f})"
            f"  spread {spread:.1f} px  median z {np.median(o[:, 2]):.1f}")
        T = dict(T, offset_col=dc, offset_row=dr, z_fine=float(np.median(o[:, 2])))
    return T


def stage_register(a):
    """The WSI covers the full section; the analysis region is only some FOVs. Keep scale = 1
    (WSI px = global px), search 8 orientations x whole-image offsets on analysis-FOV cell centers, then refine per FOV."""
    import wsi_register as wr
    from scipy.spatial import cKDTree
    P(f"\n{SEP}\nREGISTER (analysis FOV cell centers -> whole-slide image, rigid)  {TAG}\n{SEP}")
    wsi_p = Path(a.wsi) if a.wsi else find_wsi()
    if wsi_p is None:
        raise SystemExit(f"Whole-slide image not found: {WSI_CANDIDATES}")
    root = find_root()
    fovs = get_fovs(a)
    md = read_meta(root, fovs)
    if {"CenterX_global_px", "CenterY_global_px"} <= set(md.columns):
        pts = md[["CenterX_global_px", "CenterY_global_px"]].to_numpy(float)
        lab = md["fov"].to_numpy()
        src = "analysis FOV cells' CenterX/Y_global_px"
    else:
        tx = load_tx(root, fovs, OUT / "cache")
        rows, labs = [], []
        for f in fovs:
            M, res = fov_affine(tx[tx.fov == f])
            m = md[md.fov == f]
            rows.append(np.column_stack([m.CenterX_local_px, m.CenterY_local_px, np.ones(len(m))]) @ M.T)
            labs.append(np.full(len(m), f))
        pts, lab = np.vstack(rows), np.concatenate(labs)
        src = "analysis FOV cells (local->global affine)"
    ok = np.isfinite(pts).all(1); pts, lab = pts[ok], lab[ok]
    P(f"  points: {len(pts):,} ({src}, {len(np.unique(lab))} FOVs)  x range {pts[:,0].min():.0f}-{pts[:,0].max():.0f}"
      f"  y {pts[:,1].min():.0f}-{pts[:,1].max():.0f}")
    sub = pts[np.random.default_rng(1).choice(len(pts), min(len(pts), 20000), replace=False)]
    cell_size = float(np.median(cKDTree(sub).query(sub, k=2)[0][:, 1]))
    w0 = wr.WSI(wsi_p, a.wsi_channel)
    P("  whole-slide image: " + w0.describe())
    if a.scale_search:
        H, W = w0.shape
        xr, yr = np.ptp(pts[:, 0]), np.ptp(pts[:, 1])
        est = [W / xr, W / yr, H / xr, H / yr]
        cands = sorted(set([round(e * k, 5) for e in est for k in np.linspace(0.5, 1.05, 12)] +
                           [1.0, 0.5, 0.25, 0.125, 0.0625]))
    else:
        cands = [a.wsi_scale if a.wsi_scale else 1.0]
    P(f"  scale candidates: {cands if len(cands) <= 4 else f'{len(cands)} total'} ({'search' if a.scale_search else 'fixed'})")
    chans = [w0.channel] if (a.wsi_channel is not None or w0.n_channels == 1 or
                             any(wr.DAPI_PAT.search(str(n)) for n in w0.channel_names)) \
        else list(range(min(w0.n_channels, 6)))
    best = None
    for c in chans:
        w = wr.WSI(wsi_p, c)
        T = wr.register(pts, w, cands, coarse_max=a.coarse_max, cell_size=cell_size, log=P)
        T = _refine_by_fov(wr, pts, lab, w, T, cell_size, P)
        T["enrichment"] = wr.enrichment(pts, w, T)
        per = [wr.enrichment(pts[lab == f], w, T) for f in np.unique(lab)]
        T["enrichment_per_fov"] = {int(f): float(e) for f, e in zip(np.unique(lab), per)}
        T["channel"] = int(c)
        P(f"  channel {c}: enrichment {T['enrichment']:.3f}   per-FOV enrichment min {np.nanmin(per):.3f} median {np.nanmedian(per):.3f}")
        if best is None or T["enrichment"] > best["enrichment"]:
            best = T
    best["wsi"] = str(wsi_p)
    wr.save(best, OUT / "wsi_transform.json")
    wr.qc_png(wr.WSI(wsi_p, best["channel"]), best, pts, OUT / "wsi_register_qc.png")
    P(f"  result: scale={best['scale']:.4f} orientation={best['orientation']} offset=({best['offset_col']:.1f},{best['offset_row']:.1f})"
      f"  enrichment {best['enrichment']:.3f}")
    P(f"  -> {OUT/'wsi_transform.json'}  QC image {OUT/'wsi_register_qc.png'}")
    low = [f for f, e in best["enrichment_per_fov"].items() if not (e >= 1.2)]
    if best["enrichment"] < 1.2 or low:
        P(f"  [Warning] FOVs with enrichment < 1.2: {low[:20]}; registration may have failed, see the QC image (try --scale-search or --wsi-channel)")


def wsi_fov_image(t, md_f, fov_w, fov_h, out_hw):
    """Sample the WSI in the same orientation as this FOV's local coordinates (out_hw is the 0.2125 um/px grid)."""
    import wsi_register as wr
    T = wr.load(OUT / "wsi_transform.json")
    w = wr.WSI(T["wsi"], T["channel"])
    M, res = fov_affine(t)
    oh, ow = out_hw
    sx, sy = fov_w / ow, fov_h / oh          # local pixels per output pixel

    def grid(rr, cc):
        xl, yl = cc * sx, rr * sy
        return M[0, 0] * xl + M[0, 1] * yl + M[0, 2], M[1, 0] * xl + M[1, 1] * yl + M[1, 2]
    img, info = wr.sample(w, T, grid, out_hw)
    return img, res, info


# ---------------------------------------------------------------- inspect
def stage_inspect(a):
    P(f"\n{SEP}\nINSPECT  {TAG}\n{SEP}")
    root = find_root(); P(f"  data dir: {root}")
    ff = flat_files(root)
    for k, v in ff.items():
        P(f"  {k:<5} {v}")
    if "meta" in ff:
        md_all = pd.read_csv(ff["meta"], nrows=200000)
        P(f"  metadata columns: {list(md_all.columns)[:30]}")
        fcol = "fov" if "fov" in md_all.columns else "FOV"
        P(f"  FOVs in the first 200k metadata rows: {md_all[fcol].nunique()}")
    if "tx" in ff:
        t = pd.read_csv(ff["tx"], nrows=5)
        P(f"  transcript columns: {list(t.columns)}")
        P(t.to_string())
    fovs = a.fovs or FOVS
    if not fovs:
        P("  [Note] no analysis FOVs specified; later stages need --fovs"); return
    md = read_meta(root, fovs)
    P(f"  analysis FOVs: {len(fovs)}, platform cells {len(md):,} (expected {GT_CELLS})")
    imgs = index_images(root, a.image_template, fovs)
    miss = [f for f in fovs if f not in imgs]
    P(f"  FOVs with FOV-level images: {len(fovs) - len(miss)}/{len(fovs)}  missing: {miss[:20]}")
    wp = find_wsi()
    P(f"  whole-slide image: {wp}")
    if wp is not None:
        try:
            import wsi_register as wr
            P("    " + wr.WSI(wp).describe())
        except SystemExit as e:
            P(f"    [open failed] {e}")
    if "tx" in ff:
        t = pd.read_csv(ff["tx"], nrows=200000)
        t = t[t.fov == t.fov.iloc[0]]
        if {"x_global_px", "y_global_px"} <= set(t.columns):
            M, res = fov_affine(t)
            P(f"  local->global affine (FOV {t.fov.iloc[0]}):\n    {np.round(M, 5).tolist()}  residual {res:.3f} px")
    for f in fovs[:3]:
        if f not in imgs:
            continue
        chans = load_channels(imgs[f])
        P(f"  FOV {f}: {imgs[f]}  channels {len(chans)}  size {chans[0].shape} {chans[0].dtype}")
        sc = score_channels(chans, md[md.fov == f], FOV_PX, FOV_PX)
        for d in sc[:6]:
            P(f"     channel {d['channel']}  flip_y={d['flip_y']!s:<5}  enrichment {d['enrichment']:.3f}")


# ---------------------------------------------------------------- prep
def stage_prep(a):
    P(f"\n{SEP}\nPREP  {TAG}\n{SEP}")
    root = find_root(); fovs = get_fovs(a)
    md = read_meta(root, fovs)
    tx = load_tx(root, fovs, OUT / "cache")
    gcol = "target" if "target" in tx.columns else "gene"
    imgs = index_images(root, a.image_template, fovs)
    done = []
    for f in fovs:
        native = a.mode == "native"
        fd = OUT / f"fov{f}" / ("input_native" if native else "input")
        img_f = fd / "dapi_roi.tif"
        tx_f = fd / ("genes.csv" if native else "transcripts.parquet")
        meta_f = fd / "meta.json"
        if img_f.exists() and tx_f.exists() and meta_f.exists() and not a.force_prep:
            P(f"  FOV {f}: reuse prep cache"); done.append(f); continue
        fd.mkdir(parents=True, exist_ok=True)
        t = tx[tx["fov"] == f]
        fov_w, fov_h = fov_extent(t)
        px_um = UM_PER_PX if native else TARGET_UM          # output image pixel size
        out_w = fov_w if native else int(round(fov_w * UM_PER_PX / TARGET_UM))
        out_h = fov_h if native else int(round(fov_h * UM_PER_PX / TARGET_UM))
        if f not in imgs or a.image_source == "wsi":
            if not (OUT / "wsi_transform.json").exists():
                raise SystemExit("No FOV-level images and no WSI registration: run --stage register first")
            img, res, info = wsi_fov_image(t, md[md.fov == f], fov_w, fov_h, (out_h, out_w))
            sc = score_channels([img], md[md.fov == f], fov_w, fov_h)
            best = next((d["enrichment"] for d in sc if not d["flip_y"]), np.nan)
            P(f"  FOV {f}: WSI sampling  local->global affine residual {res:.2f} px  coverage {info['inside']*100:.1f}%  enrichment={best:.3f}")
            if best == best and best < 1.05:
                P("     [Warning] little enrichment at cell centers; check wsi_register_qc.png")
            chans, ch, flip, src_img = [img], 0, False, "wsi"
            sm = np.clip(img, 0, None)
        else:
            chans = load_channels(imgs[f])
            src_img = str(imgs[f])
            sm = None
        sc = score_channels(chans, md[md.fov == f], fov_w, fov_h)
        if sm is None:
            sc = score_channels(chans, md[md.fov == f], fov_w, fov_h)
            ch = a.channel if a.channel is not None else (sc[0]["channel"] if sc else 0)
            if a.flip_y == "auto":
                flip = next((d["flip_y"] for d in sc if d["channel"] == ch), False)
            else:
                flip = a.flip_y == "yes"
            best = next((d["enrichment"] for d in sc if d["channel"] == ch and d["flip_y"] == flip), np.nan)
            P(f"  FOV {f}: image {imgs[f].name} {chans[ch].shape}  channel={ch}  flip_y={flip}  enrichment={best:.3f}")
            if best == best and best < 1.05:
                P("     [Warning] little enrichment at cell centers; channel or orientation may be wrong; run --stage inspect first")
            img = chans[ch]
            if flip:
                img = img[::-1]
            sm = np.clip(resize_to(img, (out_h, out_w)), 0, None)
        mx = float(sm.max())
        tifffile.imwrite(img_f, (sm / mx * 65535).astype(np.uint16) if mx > 0 else sm.astype(np.uint16))

        ctrl = t[gcol].astype(str).str.startswith(CONTROL_PREFIX)
        t = t[~ctrl]
        # Image is already flipped to match local coordinates; transcript coordinates stay unchanged
        yl = t["y_local_px"].to_numpy(float)
        if native:
            # DISSECT nanostring branch: x/y = x_local_px/y_local_px used directly as pixel indices (integers); MIDCount = 1
            xl = t["x_local_px"].to_numpy(float)
            inb = (xl >= 0) & (xl < out_w) & (yl >= 0) & (yl < out_h)
            df = pd.DataFrame({"fov": int(f), "x_local_px": xl[inb], "y_local_px": yl[inb],
                               "target": t[gcol].astype(str).values[inb]})
            df.to_csv(tx_f, index=False)
            if (~inb).any():
                P(f"     dropped (outside image bounds): {int((~inb).sum()):,}")
        else:
            df = pd.DataFrame({"x_location": t["x_local_px"].to_numpy(float) * UM_PER_PX,
                               "y_location": yl * UM_PER_PX,
                               "feature_name": t[gcol].astype(str).values})
            df["z_location"] = 0.0; df["qv"] = 40.0
            df["cell_id"] = "UNASSIGNED"; df["overlaps_nucleus"] = 0
            df["transcript_id"] = np.arange(len(df), dtype=np.int64)
            df.to_parquet(tx_f, index=False)

        off_x = off_y = np.nan
        if {"x_global_px", "y_global_px"} <= set(t.columns):
            off_x = float(np.median(t["x_global_px"] - t["x_local_px"]))
            off_y = float(np.median(t["y_global_px"] - t["y_local_px"]))
        meta = dict(tag=TAG, fov=f, image=src_img, channel=int(ch), flip_y=bool(flip),
                    enrichment=best, fov_w=fov_w, fov_h=fov_h, um_per_px=UM_PER_PX,
                    target_um=px_um, resampled_hw=[out_h, out_w], mode=a.mode,
                    global_offset_px=[off_x, off_y], n_transcripts=int(len(df)),
                    n_control_dropped=int(ctrl.sum()), platform_cells=int((md.fov == f).sum()))
        json.dump(meta, open(meta_f, "w"), indent=2)
        P(f"     transcripts {len(df):,} (control probes removed {int(ctrl.sum()):,})  resampled {out_h}x{out_w} px")
        done.append(f)
    lo, hi = area_limits(a, UM_PER_PX if a.mode == "native" else TARGET_UM)
    if {"Width", "Height"} <= set(md.columns):
        k2 = 1.0 if a.mode == "native" else (UM_PER_PX / TARGET_UM) ** 2
        report_box_coverage(md["Width"].to_numpy(float) * md["Height"].to_numpy(float) * k2, lo, hi,
                            f"({a.mode}, {'0.12' if a.mode == 'native' else '0.2125'} um/px)",
                            UM_PER_PX if a.mode == "native" else TARGET_UM)
    P(f"  PREP DONE: {len(done)}/{len(fovs)} FOVs   platform cells {len(md):,} (expected {GT_CELLS})")


# ---------------------------------------------------------------- run
def stage_run(a):
    P(f"\n{SEP}\nRUN (DISSECT {'native nanostring entry' if a.mode == 'native' else 'xenium entry'})  {TAG}\n{SEP}")
    if not (RUNNER if a.mode == "native" else ENGINE).exists():
        raise SystemExit(f"Missing {RUNNER if a.mode == 'native' else ENGINE}")
    fovs = get_fovs(a); n_ok = n_fail = 0
    for f in fovs:
        base = OUT / f"fov{f}"
        native = a.mode == "native"
        inp = base / ("input_native" if native else "input")
        od = base / ("dissect_out_native" if native else "dissect_out")
        if not (inp / "meta.json").exists():
            P(f"  FOV {f}: [Skip] no prep result"); continue
        if (od / "DONE.flag").exists() and not a.force_run:
            P(f"  FOV {f}: done, skip"); n_ok += 1; continue
        od.mkdir(parents=True, exist_ok=True)
        lo, hi = area_limits(a, UM_PER_PX if native else TARGET_UM)
        if native:
            cmd = [sys.executable, "-u", str(RUNNER), "--platform", "nanostring",
                   "--image", str(inp / "dapi_roi.tif"), "--genes", str(inp / "genes.csv"),
                   "--output", str(od), "--n-jobs", str(a.n_jobs),
                   "--minarea", str(lo), "--maxarea", str(hi),
                   "--threshold", str(a.threshold), "--stride", str(a.stride)]
        else:
            cmd = [sys.executable, "-u", str(ENGINE),
                   "--image", str(inp / "dapi_roi.tif"),
                   "--transcripts", str(inp / "transcripts.parquet"),
                   "--output", str(od),
                   "--n-jobs", str(a.n_jobs),
                   "--minarea", str(lo),
                   "--maxarea", str(hi),
                   "--refine-batch-size", str(a.refine_batch_size),
                   "--threshold", str(a.threshold),
                   "--stride", str(a.stride)]
        if a.engine_extra:
            cmd += a.engine_extra.split()
        P(f"  FOV {f}: " + " ".join(cmd))
        if a.dry:
            continue
        rc = subprocess.call(cmd)
        if rc == 0:
            (od / "DONE.flag").touch(); n_ok += 1
        else:
            (od / "FAILED.flag").write_text(f"exit code {rc}\n"); n_fail += 1
            P(f"     [Failed] engine exit code {rc}")
            if a.stop_on_first_failure:
                raise SystemExit(rc)
    P(f"  RUN DONE: ok {n_ok}  failed {n_fail}  total {len(fovs)}")


# ---------------------------------------------------------------- merge
def find_label(od):
    for pat in ("*mask*.tif", "*segmentation*.tif", "*label*.tif", "*mask*.npy", "*label*.npy"):
        h = sorted(glob.glob(str(od / "**" / pat), recursive=True))
        if h:
            lab = np.load(h[0]) if h[0].endswith(".npy") else tifffile.imread(h[0])
            return h[0], np.squeeze(lab)
    return None, None


def stage_merge(a):
    P(f"\n{SEP}\nMERGE / QC  {TAG}\n{SEP}")
    fovs = get_fovs(a)
    cells_all, tx_all, qc = [], [], []
    for f in fovs:
        base = OUT / f"fov{f}"
        native = a.mode == "native"
        inp = base / ("input_native" if native else "input")
        odir = base / ("dissect_out_native" if native else "dissect_out")
        mp = inp / "meta.json"
        if not mp.exists():
            continue
        meta = json.load(open(mp))
        src, lab = find_label(odir)
        if lab is None:
            P(f"  FOV {f}: [Missing] no label image; dissect_out contents: "
              f"{[Path(p).name for p in glob.glob(str(odir / '*'))][:15]}")
            continue
        lab = lab.astype(np.int64)
        um = meta["target_um"]
        s_loc = um / meta["um_per_px"]                 # local pixels per resampled pixel
        ids, cnt = np.unique(lab[lab > 0], return_counts=True)
        ys, xs = np.nonzero(lab)
        v = lab[ys, xs]; o = np.argsort(v, kind="stable")
        v, ys, xs = v[o], ys[o], xs[o]
        b = np.append(np.searchsorted(v, ids), len(v))
        cy = np.array([ys[b[i]:b[i + 1]].mean() for i in range(len(ids))]) + .5
        cx = np.array([xs[b[i]:b[i + 1]].mean() for i in range(len(ids))]) + .5
        lx = cx * s_loc                                # prep flipped the image to local-coordinate orientation
        ly = cy * s_loc
        gid = f * 1_000_000 + ids
        ox, oy = meta["global_offset_px"]
        cells_all.append(pd.DataFrame({
            "cell_id": gid, "fov": f, "dissect_label": ids,
            "center_x_local_px": lx, "center_y_local_px": ly,
            "center_x_global_px": lx + ox if ox == ox else np.nan,
            "center_y_global_px": ly + oy if oy == oy else np.nan,
            "area_um2": cnt * um * um}))
        tifffile.imwrite(base / f"dissect_segmentation{'_native' if native else ''}.tif", lab.astype(np.uint32), compression="zlib")

        if native:
            gt_ = pd.read_csv(inp / "genes.csv")
            t = pd.DataFrame({"x_location": gt_["x_local_px"].to_numpy(float) * meta["um_per_px"],
                              "y_location": gt_["y_local_px"].to_numpy(float) * meta["um_per_px"],
                              "feature_name": gt_["target"].values})
        else:
            t = pd.read_parquet(inp / "transcripts.parquet")
        r = np.clip((t["y_location"].to_numpy() / um).astype(int), 0, lab.shape[0] - 1)
        c = np.clip((t["x_location"].to_numpy() / um).astype(int), 0, lab.shape[1] - 1)
        hit = lab[r, c]
        yl = t["y_location"].to_numpy() / meta["um_per_px"]
        tx_all.append(pd.DataFrame({
            "fov": f, "x_local_px": t["x_location"].to_numpy() / meta["um_per_px"],
            "y_local_px": yl, "target": t["feature_name"].values,
            "dissect_cell": np.where(hit > 0, f * 1_000_000 + hit, 0)}))
        rate = float((hit > 0).mean())
        qc.append(dict(fov=f, cells=int(len(ids)), platform_cells=meta["platform_cells"],
                       ratio=len(ids) / max(meta["platform_cells"], 1),
                       assignment_rate=rate, enrichment=meta["enrichment"]))
        P(f"  FOV {f}: cells {len(ids):,}  platform {meta['platform_cells']:,}  "
          f"ratio {len(ids)/max(meta['platform_cells'],1):.3f}  assignment rate {rate:.3f}  ({Path(src).name})")

    if not cells_all:
        raise SystemExit("No FOV produced results")
    cells = pd.concat(cells_all, ignore_index=True)
    cells.to_csv(OUT / f"dissect_cells{SFX(a)}.csv", index=False)
    txa = pd.concat(tx_all, ignore_index=True)
    try:
        txa.to_parquet(OUT / f"dissect_transcript_assignment{SFX(a)}.parquet", index=False)
    except Exception:
        txa.to_csv(OUT / f"dissect_transcript_assignment{SFX(a)}.csv.gz", index=False)
    q = pd.DataFrame(qc); q.to_csv(OUT / f"dissect_qc_per_fov{SFX(a)}.csv", index=False)
    ratio = len(cells) / max(GT_CELLS, 1)
    med = float(cells.area_um2.median())
    P(f"\n  FOV {len(q)}/{len(fovs)}   cells {len(cells):,}   GT {GT_CELLS:,}   ratio {ratio:.4f}")
    P(f"  transcript assignment rate {float((txa.dissect_cell > 0).mean()):.4f}")
    P(f"  area median {med:.1f} um^2   equiv diam {2*np.sqrt(med/np.pi):.2f} um")
    json.dump(dict(tag=TAG, fovs=fovs, fovs_merged=q.fov.tolist(), cells=int(len(cells)),
                   gt_cells=GT_CELLS, cell_ratio=ratio,
                   mask_assignment_rate=float((txa.dissect_cell > 0).mean())),
              open(OUT / f"merge_summary{SFX(a)}.json", "w"), indent=2)
    P(f"  -> {OUT/f'dissect_cells{SFX(a)}.csv'}\n  MERGE DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["inspect", "register", "prep", "run", "merge", "all"], default="all")
    ap.add_argument("--image-source", choices=["auto", "wsi"], default="auto",
                    help="auto: use FOV-level images if available, else the WSI; wsi: force the WSI")
    ap.add_argument("--wsi", default=None, help="WSI path; defaults to the first existing WSI_CANDIDATES")
    ap.add_argument("--wsi-channel", type=int, default=None)
    ap.add_argument("--wsi-scale", type=float, default=None, help="WSI pixels / global pixels; default fixed at 1")
    ap.add_argument("--scale-search", action="store_true", help="also search scale (default: orientation and offset only)")
    ap.add_argument("--coarse-max", type=int, default=6000, help="max side of the coarse-search thumbnail (px)")
    ap.add_argument("--fovs", nargs="+", type=int, default=None)
    ap.add_argument("--image-template", default=None,
                    help="morphology image path template, e.g. 'yourpath/Morphology2D/img_F{fov:03d}.TIF'")
    ap.add_argument("--channel", type=int, default=None, help="DAPI channel index; auto by default")
    ap.add_argument("--flip-y", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--mode", choices=["native", "xenium"], default="native",
                    help="native = DISSECT native nanostring entry (0.12 um/px local pixels); xenium = resample to 0.2125 um/px and pose as xenium")
    ap.add_argument("--minarea", type=int, default=None, help="default 350; converted by physical size with --area-rule physical")
    ap.add_argument("--area-rule", choices=["fixed", "physical"], default="fixed",
                    help="fixed = numeric 350-1600 as in other DISSECT runs; physical = Xenium physical-size equivalent")
    ap.add_argument("--maxarea", type=int, default=None, help="default 1600; converted by physical size with --area-rule physical")
    ap.add_argument("--refine-batch-size", type=int, default=512)
    ap.add_argument("--threshold", type=float, default=-0.20)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--engine-extra", default="")
    ap.add_argument("--force-prep", action="store_true")
    ap.add_argument("--force-run", action="store_true")
    ap.add_argument("--stop-on-first-failure", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    P(f"{SEP}\nDISSECT x {TAG}   out {OUT}\n{SEP}")
    if a.stage == "inspect":
        stage_inspect(a); return
    if a.stage == "register" or (a.stage == "all" and not (OUT / "wsi_transform.json").exists()
                                 and find_wsi() is not None):
        stage_register(a)
        if a.stage == "register":
            return
    if a.stage in ("prep", "all"):
        stage_prep(a)
    if a.stage in ("run", "all"):
        stage_run(a)
    if a.stage in ("merge", "all"):
        stage_merge(a)


if __name__ == "__main__":
    main()
