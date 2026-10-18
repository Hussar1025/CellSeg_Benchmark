#!/usr/bin/env python3
"""DISSECT x MERFISH prostate1 —— prep / run / merge 一体

DISSECT 的 STReader 只认 ['stereoseq','xenium','nanostring','hd']，xenium 分支
固定 x_location/0.2125 -> 图像像素，detector 权重也在 0.2125 um/px 上训练。
所以本脚本把 merfish 伪装成 xenium：
  prep : DAPI 裁中心 ROI -> 重采样到 0.2125 um/px；转录本 global_x/y(um)
         -> mosaic px -> 相对 ROI 的 um 偏移，写 xenium schema parquet
         (DISSECT 内部 /0.2125 正好等于重采样图像素，minarea/maxarea 沿用 xenium 默认)
  run  : 调已验证的官方 wrapper run_xenium_pancreas_dissect.py
  merge: 汇总 DISSECT 输出 -> 全局 um 质心 csv + 标签 tif + QC

用法:
  python dissect_merfish_prostate1.py --stage all
  python dissect_merfish_prostate1.py --stage prep
  python dissect_merfish_prostate1.py --stage run --n-jobs 4
  python dissect_merfish_prostate1.py --stage merge
"""
import argparse, glob, json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, tifffile

# ===== 数据集写死常量 =====
ROOT      = "/data/qiuyijia/dataset/merfish_breast_cancer"
OUT       = Path("/data/qiuyijia/dissect_merfish_breast")
ROI_SIZE  = 10000          # native mosaic px, 居中
GT_CELLS  = 7524
TARGET_UM = 0.2125         # DISSECT/xenium detector 训练尺度
ENGINE    = Path("/data/qiuyijia/DISSECT/code/run_xenium_pancreas_dissect.py")
TAG       = "merfish_breast"

P = lambda *a: print(*a, flush=True)
SEP = "=" * 78


def find_one(root, pats):
    for p in pats:
        h = sorted(glob.glob(f"{root}/**/{p}", recursive=True))
        if h:
            return h[0]
    return None


def read_crop(dapi, y0, x0, size):
    try:
        a = tifffile.memmap(dapi, mode="r")
        return np.asarray(a[y0:y0 + size, x0:x0 + size])
    except Exception:
        import zarr
        z = zarr.open(tifffile.imread(dapi, aszarr=True), mode="r")
        a = z if hasattr(z, "shape") else z["0"]
        return np.asarray(a[y0:y0 + size, x0:x0 + size])


def resize_to(img, out):
    try:
        import cv2
        return cv2.resize(img.astype(np.float32), (out, out), interpolation=cv2.INTER_AREA)
    except Exception:
        from skimage.transform import resize as sk
        return sk(img.astype(np.float32), (out, out), preserve_range=True, anti_aliasing=True)


# ---------------------------------------------------------------- prep
def stage_prep(force=False):
    P(f"\n{SEP}\nPREP  {TAG}\n{SEP}")
    inp = OUT / "input"; inp.mkdir(parents=True, exist_ok=True)
    img_f, tx_f, meta_f = inp / "dapi_roi.tif", inp / "transcripts.parquet", inp / "meta.json"
    if img_f.exists() and tx_f.exists() and meta_f.exists() and not force:
        m = json.load(open(meta_f))
        P(f"  复用 prep 缓存: {m['resampled_px']}px  tx={m['n_transcripts']:,}  GT={m['gt_cells']}")
        return m

    dapi = find_one(ROOT, ["*DAPI*.tif*", "*dapi*.tif*"])
    tfm  = find_one(ROOT, ["*micron_to_mosaic*transform*.csv"])
    txf  = find_one(ROOT, ["detected_transcripts.csv*"])
    mdf  = find_one(ROOT, ["cell_metadata.csv*"])
    for n, v in (("dapi", dapi), ("transform", tfm), ("transcripts", txf), ("cell_metadata", mdf)):
        if v is None:
            raise SystemExit(f"缺 {n}")

    M = np.loadtxt(tfm)
    with tifffile.TiffFile(dapi) as t:
        H, W = t.series[0].shape[-2:]
    r = ROI_SIZE; y0 = (H - r) // 2; x0 = (W - r) // 2
    px_um  = 1.0 / float(np.hypot(M[0, 0], M[0, 1]))
    out_px = int(round(r * px_um / TARGET_UM))
    P(f"  图像 {H}x{W}  px_um={px_um:.5f}  ROI px y[{y0},{y0+r}) x[{x0},{x0+r})")
    P(f"  重采样 {r} -> {out_px} px @ {TARGET_UM} um/px")

    sm = np.clip(resize_to(read_crop(dapi, y0, x0, r), out_px), 0, None)
    mx = float(sm.max())
    tifffile.imwrite(img_f, (sm / mx * 65535).astype(np.uint16) if mx > 0 else sm.astype(np.uint16))
    P(f"  image -> {img_f}")

    keep = []
    for ch in pd.read_csv(txf, chunksize=5_000_000,
                          usecols=lambda c: c in ("global_x", "global_y", "gene")):
        px = M[0, 0] * ch.global_x + M[0, 1] * ch.global_y + M[0, 2]
        py = M[1, 0] * ch.global_x + M[1, 1] * ch.global_y + M[1, 2]
        m = (px >= x0) & (px < x0 + r) & (py >= y0) & (py < y0 + r)
        if m.any():
            keep.append(pd.DataFrame({
                "x_location": (px[m] - x0) * px_um,
                "y_location": (py[m] - y0) * px_um,
                "feature_name": ch.loc[m, "gene"].astype(str).values}))
    df = pd.concat(keep, ignore_index=True)
    n0 = len(df)
    df = df[~df.feature_name.str.lower().str.startswith("blank")].reset_index(drop=True)
    df["z_location"] = 0.0; df["qv"] = 40.0
    df["cell_id"] = "UNASSIGNED"; df["overlaps_nucleus"] = 0
    df["transcript_id"] = np.arange(len(df), dtype=np.int64)
    df.to_parquet(tx_f, index=False)
    P(f"  转录本 ROI内 {n0:,} -> 去Blank {len(df):,}  基因 {df.feature_name.nunique()}")
    P(f"  x_location -> px [{df.x_location.min()/TARGET_UM:.0f},"
      f"{df.x_location.max()/TARGET_UM:.0f}]  (图像 0..{out_px})")

    c = (np.array([[x0, y0, 1], [x0 + r, y0, 1], [x0, y0 + r, 1], [x0 + r, y0 + r, 1]], float)
         @ np.linalg.inv(M).T)
    X0, X1, Y0, Y1 = c[:, 0].min(), c[:, 0].max(), c[:, 1].min(), c[:, 1].max()
    d = pd.read_csv(mdf)
    gt = int(((d.center_x >= X0) & (d.center_x < X1) &
              (d.center_y >= Y0) & (d.center_y < Y1)).sum())
    P(f"  ROI um x[{X0:.1f},{X1:.1f}] y[{Y0:.1f},{Y1:.1f}]   ROI内 GT = {gt} (预期 {GT_CELLS})")
    meta = dict(tag=TAG, root=ROOT, dapi=dapi, roi_px=[int(y0), int(x0), r],
                roi_um=[X0, X1, Y0, Y1], px_um=px_um, target_um=TARGET_UM,
                resampled_px=out_px, gt_cells=gt, n_transcripts=len(df))
    json.dump(meta, open(meta_f, "w"), indent=2)
    P("  PREP DONE")
    return meta


# ---------------------------------------------------------------- run
def stage_run(a):
    P(f"\n{SEP}\nRUN (DISSECT engine)  {TAG}\n{SEP}")
    if not ENGINE.exists():
        raise SystemExit(f"缺 engine: {ENGINE}")
    inp, od = OUT / "input", OUT / "dissect_out"
    od.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", str(ENGINE),
           "--image", str(inp / "dapi_roi.tif"),
           "--transcripts", str(inp / "transcripts.parquet"),
           "--output", str(od),
           "--n-jobs", str(a.n_jobs),
           "--minarea", str(a.minarea),
           "--maxarea", str(a.maxarea),
           "--refine-batch-size", str(a.refine_batch_size),
           "--threshold", str(a.threshold),
           "--stride", str(a.stride)]
    if a.engine_extra:
        cmd += a.engine_extra.split()
    P("  " + " ".join(cmd))
    if a.dry:
        P("  DRY RUN, 不执行"); return
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"engine 退出码 {rc}")
    P("  RUN DONE")


# ---------------------------------------------------------------- merge
def stage_merge():
    P(f"\n{SEP}\nMERGE / QC  {TAG}\n{SEP}")
    meta = json.load(open(OUT / "input" / "meta.json"))
    od = OUT / "dissect_out"
    files = sorted(glob.glob(str(od / "**" / "*"), recursive=True))
    P("  DISSECT 输出文件:")
    for f in files[:25]:
        if Path(f).is_file():
            P(f"    {Path(f).name}  {Path(f).stat().st_size/1e6:.2f} MB")

    # 1) 标签图优先
    lab = None
    for pat in ("*mask*.tif", "*segmentation*.tif", "*label*.tif", "*mask*.npy", "*label*.npy"):
        h = sorted(glob.glob(str(od / "**" / pat), recursive=True))
        if h:
            lab = np.load(h[0]) if h[0].endswith(".npy") else np.squeeze(tifffile.imread(h[0]))
            P(f"  标签图: {h[0]}  {lab.shape}"); break

    # 2) 否则用 boxes/cells csv 的 cell_id + 坐标
    cells = None
    if lab is None:
        for pat in ("cells.csv", "boxes.csv", "*cell*.csv", "raw_boxes.csv"):
            h = sorted(glob.glob(str(od / "**" / pat), recursive=True))
            if h:
                d = pd.read_csv(h[0])
                P(f"  表格: {h[0]}  行={len(d):,}  列={list(d.columns)[:12]}")
                cells = (h[0], d); break
        if cells is None:
            raise SystemExit("未找到标签图或 cells/boxes csv —— 把上面文件清单发我，我按实际列名补 merge")

    X0, X1, Y0, Y1 = meta["roi_um"]; S = meta["resampled_px"]; um = meta["target_um"]
    to_gx = lambda xl: X0 + np.asarray(xl, float) * (X1 - X0) / (S * um)
    to_gy = lambda yl: Y0 + np.asarray(yl, float) * (Y1 - Y0) / (S * um)

    if lab is not None:
        ids, cnt = np.unique(lab[lab > 0], return_counts=True)
        area = cnt * um * um
        ys, xs = np.nonzero(lab)
        v = lab[ys, xs]; o = np.argsort(v, kind="stable")
        v, ys, xs = v[o], ys[o], xs[o]
        b = np.append(np.searchsorted(v, ids), len(v))
        cy = np.array([ys[b[i]:b[i+1]].mean() for i in range(len(ids))])
        cx = np.array([xs[b[i]:b[i+1]].mean() for i in range(len(ids))])
        out = pd.DataFrame({"cell_id": ids,
                            "center_x": to_gx((cx + .5) * um), "center_y": to_gy((cy + .5) * um),
                            "area_um2": area})
        tifffile.imwrite(OUT / "dissect_segmentation.tif", lab.astype(np.uint32), compression="zlib")
        P(f"  标签图 -> {OUT/'dissect_segmentation.tif'}")
    else:
        _, d = cells
        cid = next((c for c in d.columns if c.lower() in ("cell_id", "cell", "label", "id")), None)
        xc  = next((c for c in d.columns if c.lower() in ("center_x", "x_center", "cx", "x")), None)
        yc  = next((c for c in d.columns if c.lower() in ("center_y", "y_center", "cy", "y")), None)
        if xc is None and {"x1", "x2"} <= set(d.columns):
            d["_cx"] = (d.x1 + d.x2) / 2; d["_cy"] = (d.y1 + d.y2) / 2; xc, yc = "_cx", "_cy"
        if None in (xc, yc):
            raise SystemExit(f"csv 里认不出质心列: {list(d.columns)} —— 发我这些列名")
        if cid is None:
            d["cell_id"] = np.arange(1, len(d) + 1); cid = "cell_id"
        d = d[d[cid].astype(str).str.lower().ne("nan")]
        out = pd.DataFrame({"cell_id": d[cid].values,
                            "center_x": to_gx(d[xc].values * um), "center_y": to_gy(d[yc].values * um),
                            "area_um2": np.nan})

    out.to_csv(OUT / "dissect_cells.csv", index=False)
    ratio = len(out) / max(meta["gt_cells"], 1)
    P(f"\n  cells {len(out):,}   GT {meta['gt_cells']:,}   ratio {ratio:.4f}")
    if out.area_um2.notna().any():
        med = float(out.area_um2.median())
        P(f"  area median {med:.1f} um^2   equiv diam {2*np.sqrt(med/np.pi):.2f} um")
    P(f"  cells -> {OUT/'dissect_cells.csv'}")
    json.dump(dict(tag=TAG, cells=int(len(out)), gt_cells=meta["gt_cells"],
                   cell_ratio=ratio, roi_um=meta["roi_um"]),
              open(OUT / "merge_summary.json", "w"), indent=2)
    P("  MERGE DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prep", "run", "merge", "all"], default="all")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--minarea", type=int, default=350)
    ap.add_argument("--maxarea", type=int, default=1600)
    ap.add_argument("--refine-batch-size", type=int, default=512)
    ap.add_argument("--threshold", type=float, default=-0.20)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--engine-extra", default="")
    ap.add_argument("--force-prep", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    P(f"{SEP}\nDISSECT x {TAG}   out {OUT}\n{SEP}")
    if a.stage in ("prep", "all"):
        stage_prep(a.force_prep)
    if a.stage in ("run", "all"):
        stage_run(a)
    if a.stage in ("merge", "all"):
        stage_merge()


if __name__ == "__main__":
    main()
