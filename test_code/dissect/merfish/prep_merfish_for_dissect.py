#!/usr/bin/env python3
"""merfish -> DISSECT(xenium 伪装) 输入转换。

STReader 只认 ['stereoseq','xenium','nanostring','hd']，xenium 分支固定
x_location/0.2125 -> 图像像素。所以：
  1. DAPI mosaic 裁中心 ROI，重采样到 0.2125 um/px（对齐 detector 训练尺度）
  2. 转录本 global_x/y(um) -> mosaic px -> 相对 ROI 的 um 偏移，写 xenium schema parquet
  3. DISSECT 内部 /0.2125 正好等于重采样图的像素坐标，minarea/maxarea 沿用 xenium 默认
用法: python prep_merfish_for_dissect.py --root /data/qiuyijia/dataset/merfish_prostate1 \
        --out /data/qiuyijia/dissect_inputs/prostate1
"""
import argparse, glob, json
from pathlib import Path
import numpy as np, pandas as pd, tifffile

TARGET_UM = 0.2125

def find_one(root, pats):
    for p in pats:
        h = sorted(glob.glob(f"{root}/**/{p}", recursive=True))
        if h: return h[0]
    return None

def read_crop(dapi, y0, x0, size):
    try:
        a = tifffile.memmap(dapi, mode="r")
        return np.asarray(a[y0:y0+size, x0:x0+size])
    except Exception:
        import zarr
        z = zarr.open(tifffile.imread(dapi, aszarr=True), mode="r")
        a = z if hasattr(z, "shape") else z["0"]
        return np.asarray(a[y0:y0+size, x0:x0+size])

def resize_to(img, out):
    try:
        import cv2
        return cv2.resize(img.astype(np.float32), (out, out), interpolation=cv2.INTER_AREA)
    except Exception:
        from skimage.transform import resize as sk
        return sk(img.astype(np.float32), (out, out), preserve_range=True, anti_aliasing=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--roi-size", type=int, default=10000)
    a = ap.parse_args()

    root = Path(a.root); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dapi = find_one(root, ["*DAPI*.tif*", "*dapi*.tif*"])
    tfm  = find_one(root, ["*micron_to_mosaic*transform*.csv"])
    txf  = find_one(root, ["detected_transcripts.csv*"])
    mdf  = find_one(root, ["cell_metadata.csv*"])
    for n, v in (("dapi", dapi), ("transform", tfm), ("transcripts", txf), ("cell_metadata", mdf)):
        if v is None: raise SystemExit(f"缺 {n}")

    M = np.loadtxt(tfm)
    with tifffile.TiffFile(dapi) as t: H, W = t.series[0].shape[-2:]
    r = a.roi_size; y0 = (H - r) // 2; x0 = (W - r) // 2
    px_um = 1.0 / float(np.hypot(M[0, 0], M[0, 1]))
    out_px = int(round(r * px_um / TARGET_UM))
    print(f"图像 {H}x{W}  px_um={px_um:.5f}  ROI px y[{y0},{y0+r}) x[{x0},{x0+r})")
    print(f"重采样 {r} -> {out_px} px @ {TARGET_UM} um/px")

    img = read_crop(dapi, y0, x0, r)
    sm = np.clip(resize_to(img, out_px), 0, None)
    mx = float(sm.max())
    tifffile.imwrite(out / "dapi_roi.tif",
                     (sm / mx * 65535).astype(np.uint16) if mx > 0 else sm.astype(np.uint16))
    print(f"  -> {out/'dapi_roi.tif'}")

    keep = []
    for ch in pd.read_csv(txf, chunksize=5_000_000,
                          usecols=lambda c: c in ("global_x", "global_y", "gene")):
        px = M[0, 0] * ch.global_x + M[0, 1] * ch.global_y + M[0, 2]
        py = M[1, 0] * ch.global_x + M[1, 1] * ch.global_y + M[1, 2]
        m = (px >= x0) & (px < x0 + r) & (py >= y0) & (py < y0 + r)
        if not m.any(): continue
        s = pd.DataFrame({"x_location": (px[m] - x0) * px_um,
                          "y_location": (py[m] - y0) * px_um,
                          "feature_name": ch.loc[m, "gene"].astype(str).values})
        keep.append(s)
    df = pd.concat(keep, ignore_index=True)
    n0 = len(df)
    df = df[~df.feature_name.str.lower().str.startswith("blank")].reset_index(drop=True)
    df["z_location"] = 0.0; df["qv"] = 40.0
    df["cell_id"] = "UNASSIGNED"; df["overlaps_nucleus"] = 0
    df["transcript_id"] = np.arange(len(df), dtype=np.int64)
    df.to_parquet(out / "transcripts.parquet", index=False)
    print(f"转录本 ROI内 {n0:,} -> 去Blank {len(df):,}  基因 {df.feature_name.nunique()}")
    print(f"  x_location [{df.x_location.min():.1f},{df.x_location.max():.1f}] um"
          f"  -> px [{df.x_location.min()/TARGET_UM:.0f},{df.x_location.max()/TARGET_UM:.0f}]"
          f"  (图像 0..{out_px})")

    Mi = np.linalg.inv(M)
    c = np.array([[x0, y0, 1], [x0+r, y0, 1], [x0, y0+r, 1], [x0+r, y0+r, 1]], float) @ Mi.T
    X0, X1, Y0, Y1 = c[:, 0].min(), c[:, 0].max(), c[:, 1].min(), c[:, 1].max()
    d = pd.read_csv(mdf)
    gt = int(((d.center_x >= X0) & (d.center_x < X1) &
              (d.center_y >= Y0) & (d.center_y < Y1)).sum())
    print(f"ROI um x[{X0:.1f},{X1:.1f}] y[{Y0:.1f},{Y1:.1f}]   ROI内 GT = {gt}")
    json.dump(dict(root=str(root), dapi=dapi, roi_px=[int(y0), int(x0), r],
                   roi_um=[X0, X1, Y0, Y1], px_um=px_um, target_um=TARGET_UM,
                   resampled_px=out_px, gt_cells=gt, n_transcripts=len(df)),
              open(out / "meta.json", "w"), indent=2)

if __name__ == "__main__":
    main()
