#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xenium_8: 联合分布图 (邻域污染源密度 vs 细胞污染富集)
  - 每个方法一个 panel, 圆点按局部密度着色
  - 线性拟合(含截距), 浅灰虚线
  - 拟合解析式与相关系数统一放在整图右侧, 不压在散点上
斜率 = 对邻居的敏感性: 邻域源密度每升高一单位, 细胞污染富集升高多少
"""
import os
import numpy as np
import pandas as pd
import tifffile
import zarr
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.filters import threshold_otsu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300

X8 = "/data/qiuyijia/dataset/xenium_8"
MORPH = os.path.join(X8, "morphology_focus")
TXF = os.path.join(X8, "_roi_cache", "tx_roi_8664_22001_10000_qv20.parquet")
OUTDIR = os.path.join(X8, "_roi_cache", "case_knn")
os.makedirs(OUTDIR, exist_ok=True)

Y0, X0, SZ = 8664, 22001, 10000
PX = 0.2125
CASES = {"T": ("ch0011_cd3e", "T"), "Myeloid": ("ch0015_cd68", "Myeloid")}
LINEAGES = ["T", "NK", "B", "Plasma", "Myeloid"]
MARK = {
    "T":       ["CD3D", "CD3E", "CD3G", "TRAC", "CD247", "CD8A", "CCR7"],
    "NK":      ["NKG7", "GNLY", "KLRD1", "KLRC1", "PRF1", "FGFBP2", "NCAM1"],
    "B":       ["MS4A1", "CD19", "CD79A", "BANK1", "TCL1A", "TNFRSF13B", "GPR183"],
    "Plasma":  ["MZB1", "PRDM1", "DERL3", "SLAMF7", "TENT5C", "TNFRSF17",
                "SDC1", "FKBP11"],
    "Myeloid": ["AIF1", "SPI1", "CD14", "CD68", "CD163", "MS4A6A", "MS4A4A",
                "TREM2", "VSIG4", "MRC1", "LILRB2", "FCGR1A"],
}
METHOD_MASK = {
    "proseg":     "/data/qiuyijia/proseg_xenium_8/proseg_segmentation.tif",
    "ucs":        "/data/qiuyijia/ucs_xenium_8/ucs_segmentation.tif",
    "genesegnet": "/data/qiuyijia/genesegnet_xenium_8/genesegnet_xenium_8_mask.tif",
    "cellpose":   "/data/qiuyijia/cellpose/xenium_8_output/xenium_8_cellpose_pred.tif",
}
POINT_METHODS = ["boms", "baysor", "bering"]
DISPLAY = {"proseg": "ProSeg", "ucs": "UCS", "genesegnet": "GeneSegNet",
           "cellpose": "Cellpose", "boms": "BOMS", "baysor": "Baysor",
           "bering": "Bering"}
POS_AREA_FRAC = 0.15
MIN_TX = 10
MIN_OWN = 1
MAIN_R = 15
NCOL = 4


def log(*a):
    print(*a, flush=True)


def read_ch(name):
    st = tifffile.imread(os.path.join(MORPH, name + ".ome.tif"),
                         aszarr=True, key=0, is_ome=False)
    z = zarr.open(st, mode="r")
    a = z[Y0:Y0 + SZ, X0:X0 + SZ] if z.ndim == 2 else z[0, Y0:Y0 + SZ, X0:X0 + SZ]
    return np.asarray(a, dtype=np.float32)


def protein_pos(chname):
    img = read_ch(chname)
    thr = threshold_otsu(img[::4, ::4])
    pos = ndi.binary_opening(img > thr, structure=np.ones((3, 3)))
    lbl, n = ndi.label(pos)
    sizes = ndi.sum(pos, lbl, range(1, n + 1))
    keep = np.isin(lbl, np.nonzero(sizes >= 50)[0] + 1)
    log(f"  {chname}: otsu={thr:.0f} 阳性区={100*keep.mean():.2f}%")
    return keep


def down_frac(b, f):
    if f == 1:
        return b.astype(np.float32)
    h = b.shape[0] // f * f
    w = b.shape[1] // f * f
    x = b[:h, :w].astype(np.float32)
    return x.reshape(h // f, f, w // f, f).mean((1, 3))


def _points_to_mask(xu, yu, cellid, size):
    ix = np.clip(((xu / PX - X0) / SZ * size).astype(int), 0, size - 1)
    iy = np.clip(((yu / PX - Y0) / SZ * size).astype(int), 0, size - 1)
    lab = np.zeros((size, size), dtype=np.int32)
    lab[iy, ix] = cellid.astype(np.int32)
    return np.where(lab > 0, lab, ndi.grey_dilation(lab, size=3))


def boms_mask(size=2500):
    z = np.load("/data/qiuyijia/boms_xenium_8/boms_xenium_8.npz", allow_pickle=True)
    seg, xu, yu = z["seg"], z["x"], z["y"]
    k = seg > 0
    return _points_to_mask(xu[k], yu[k], seg[k], size)


def baysor_mask(size=2500):
    import json
    gj = "/data/qiuyijia/baysor/xenium_8_output/segmentation_polygons_2d.json"
    try:
        from matplotlib.path import Path as MplPath
        feats = json.load(open(gj))["features"]
        lab = np.zeros((size, size), dtype=np.int32)
        n_ok = 0
        for k, ft in enumerate(feats, start=1):
            try:
                coords = np.asarray(ft["geometry"]["coordinates"][0], dtype=float)
            except Exception:
                continue
            gx = (coords[:, 0] / PX - X0) / SZ * size
            gy = (coords[:, 1] / PX - Y0) / SZ * size
            x0i, x1i = int(max(gx.min(), 0)), int(min(gx.max() + 1, size))
            y0i, y1i = int(max(gy.min(), 0)), int(min(gy.max() + 1, size))
            if x1i <= x0i or y1i <= y0i:
                continue
            sy, sx = np.mgrid[y0i:y1i, x0i:x1i]
            inside = MplPath(np.c_[gx, gy]).contains_points(
                np.c_[sx.ravel(), sy.ravel()])
            if inside.any():
                blk = lab[y0i:y1i, x0i:x1i]
                blk.flat[inside] = k
                n_ok += 1
        log(f"    baysor 多边形栅格化: {n_ok} 个细胞")
        if n_ok > 100:
            return lab
    except Exception as e:
        log(f"    baysor 多边形失败 ({e}), 退回点栅格化")
    b = pd.read_csv("/data/qiuyijia/baysor/xenium_8_output/segmentation.csv",
                    usecols=["cell", "x", "y"])
    b = b[b["cell"] > 0]
    return _points_to_mask(b["x"].values, b["y"].values, b["cell"].values, size)


def bering_mask(size=2500):
    t = pd.read_parquet(
        "/data/qiuyijia/bering/xenium_8_center10000_output/transcript_assignments.parquet",
        columns=["cell_id", "x", "y"])
    sid = t["cell_id"].astype(str)
    num = pd.to_numeric(sid.str.extract(r"(\d+)", expand=False), errors="coerce")
    ok = num.notna().values & ~sid.isin(["", "nan", "None", "background",
                                         "UNASSIGNED", "-1"]).values
    cid = num.fillna(-1).values + 1.0
    log(f"    bering: {int(ok.sum())} 条转录本有归属, "
        f"{int(np.unique(cid[ok]).size)} 个细胞")
    return _points_to_mask(t["x"].values[ok], t["y"].values[ok], cid[ok], size)


def get_mask(name):
    if name == "boms":
        return boms_mask(2500)
    if name == "baysor":
        return baysor_mask(2500)
    if name == "bering":
        return bering_mask(2500)
    return tifffile.imread(METHOD_MASK[name])


def main():
    log("== 蛋白阳性区 ==")
    posmap = {c: protein_pos(ch) for c, (ch, _) in CASES.items()}

    log("\n== 转录本 ==")
    tx = pd.read_parquet(TXF)
    g = tx["gene"].astype(str).values
    tix = tx["x_um"].values / PX - X0
    tiy = tx["y_um"].values / PX - Y0
    k = (tix >= 0) & (tix < SZ) & (tiy >= 0) & (tiy < SZ)
    g, tix, tiy = g[k], tix[k], tiy[k]
    gm = {a: np.isin(g, b) for a, b in MARK.items()}
    lin_total = sum(gm[a].sum() for a in LINEAGES)
    expect = {a: gm[a].sum() / lin_total for a in LINEAGES}
    trees = {a: cKDTree(np.c_[tix[gm[a]], tiy[gm[a]]]) for a in LINEAGES}

    methods = list(METHOD_MASK) + POINT_METHODS
    store = {}
    for name in methods:
        log(f"\n#### {name} ####")
        mask = get_mask(name)
        f = SZ // mask.shape[0]
        flat = mask.ravel()
        ok = flat > 0
        labels = np.unique(flat[ok])
        nl = len(labels)
        idx_map = np.zeros(int(mask.max()) + 1, dtype=np.int64)
        idx_map[labels] = np.arange(nl)
        cnt_px = np.bincount(idx_map[flat[ok]], minlength=nl).astype(np.float32)
        ys, xs = np.nonzero(mask)
        cj = idx_map[mask[ys, xs]]
        cy = np.bincount(cj, weights=ys, minlength=nl) / np.clip(cnt_px, 1, None)
        cx = np.bincount(cj, weights=xs, minlength=nl) / np.clip(cnt_px, 1, None)
        scale = SZ / mask.shape[0]
        cen_px = np.c_[cx * scale, cy * scale]
        mx = np.clip((tix / scale).astype(int), 0, mask.shape[1] - 1)
        my = np.clip((tiy / scale).astype(int), 0, mask.shape[0] - 1)
        cid = mask[my, mx]
        ci = np.where(cid > 0, idx_map[np.clip(cid, 0, idx_map.size - 1)], -1)
        good = ci >= 0
        cnt = {a: np.bincount(ci[good & gm[a]], minlength=nl).astype(np.float32)
               for a in LINEAGES}
        for cname, (ch, self_lin) in CASES.items():
            pf = down_frac(posmap[cname], f)
            h = min(pf.shape[0], mask.shape[0])
            w = min(pf.shape[1], mask.shape[1])
            sub = mask[:h, :w].ravel()
            sel = sub > 0
            pfr = np.bincount(idx_map[sub[sel]], weights=pf[:h, :w].ravel()[sel],
                              minlength=nl)
            cpx = np.bincount(idx_map[sub[sel]], minlength=nl).astype(np.float32)
            is_case = (pfr / np.clip(cpx, 1, None)) >= POS_AREA_FRAC
            own = cnt[self_lin]
            tot_lin = np.sum([cnt[a] for a in LINEAGES], 0)
            frn = tot_lin - own
            exp_frn = 1.0 - expect[self_lin]
            enrich = np.where(tot_lin > 0,
                              (frn / np.clip(tot_lin, 1, None)) / exp_frn, np.nan)
            ev = is_case & (tot_lin >= MIN_TX) & (own >= MIN_OWN)
            pts = cen_px[ev]
            rpx = MAIN_R / PX
            tot = np.zeros(len(pts))
            for a in LINEAGES:
                if a == self_lin:
                    continue
                tot += np.array([len(q) for q in
                                 trees[a].query_ball_point(pts, rpx)], dtype=float)
            dens = tot / (np.pi * MAIN_R * MAIN_R)
            log(f"  {cname}: eval={int(ev.sum())} "
                f"中位富集={np.nanmedian(enrich[ev]):.2f}x")
            store[(name, cname)] = dict(dens=dens, enrich=enrich[ev])

    # ---------- 统一坐标范围 (两张图、所有 panel 共用) ----------
    all_x, all_y = [], []
    for key, d in store.items():
        x, y = d["dens"], d["enrich"]
        m = np.isfinite(x) & np.isfinite(y)
        all_x.append(x[m]); all_y.append(y[m])
    all_x = np.concatenate(all_x); all_y = np.concatenate(all_y)
    XMAX = float(np.percentile(all_x, 99.0))
    YMAX = float(max(np.percentile(all_y, 99.0), 1.2))
    # 刻度取整到好看的间隔
    def nice_ticks(vmax, n=5):
        raw = vmax / n
        mag = 10 ** np.floor(np.log10(max(raw, 1e-9)))
        for mult in (1, 2, 2.5, 5, 10):
            step = mult * mag
            if step >= raw:
                break
        return np.arange(0, vmax + step * 0.5, step)
    XTICKS = nice_ticks(XMAX)
    YTICKS = nice_ticks(YMAX)
    XMAX = float(XTICKS[-1]); YMAX = float(YTICKS[-1])
    ASPECT = XMAX / YMAX      # 让数据单位在 x/y 上等价, 斜率倾角可直接比较
    log(f"\n统一坐标: x[0,{XMAX:.3f}] y[0,{YMAX:.2f}]  "
        f"斜率 1.0 对应的视觉倾角固定")

    # ---------- 绘图 ----------
    for cname in CASES:
        n = len(methods)
        nrow = int(np.ceil(n / NCOL))
        fig, axes = plt.subplots(nrow, NCOL, figsize=(4.5 * NCOL, 4.3 * nrow),
                                 constrained_layout=True,
                                 sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel()
        for a_ in axes[n:]:
            a_.axis("off")
        sc = None
        lines = []
        for ax, name in zip(axes, methods):
            d = store[(name, cname)]
            x, y = d["dens"], d["enrich"]
            m = np.isfinite(x) & np.isfinite(y)
            x, y = x[m], y[m]
            ax.set_title(DISPLAY.get(name, name), fontsize=11)
            ax.grid(alpha=0.15)
            ax.set_xlim(0, XMAX); ax.set_ylim(0, YMAX)
            ax.set_xticks(XTICKS); ax.set_yticks(YTICKS)
            ax.set_box_aspect(1.0)          # 所有 panel 同样的绘图框形状
            if len(x) < 20:
                lines.append(f"{DISPLAY.get(name, name):<12s}  n/a")
                continue
            k2 = (x <= XMAX) & (y <= YMAX)
            xs_, ys_ = x[k2], y[k2]
            nb = 60
            H, xe, ye = np.histogram2d(xs_, ys_, bins=nb,
                                       range=[[0, XMAX], [0, YMAX]])
            bx = np.clip(np.digitize(xs_, xe) - 1, 0, nb - 1)
            by = np.clip(np.digitize(ys_, ye) - 1, 0, nb - 1)
            dc = H[bx, by]
            order = np.argsort(dc)
            sc = ax.scatter(xs_[order], ys_[order], c=dc[order], cmap="Blues",
                            s=7, linewidths=0, alpha=0.85)
            slope, intercept = np.polyfit(xs_, ys_, 1)
            r = np.corrcoef(xs_, ys_)[0, 1]
            xl = np.linspace(0, XMAX, 50)
            ax.plot(xl, slope * xl + intercept, "--", color="#7F7F7F", lw=2.2,
                    zorder=4)
            ax.axhline(1.0, color="#888888", ls=":", lw=1)
            sign = "+" if intercept >= 0 else "-"
            lines.append(f"{DISPLAY.get(name, name):<12s}  "
                         f"y = {slope:.2f}x {sign} {abs(intercept):.2f}   "
                         f"r = {r:.2f}   n = {len(xs_)}")
        for i, ax in enumerate(axes[:n]):
            if i % NCOL == 0:
                ax.set_ylabel("contamination enrichment (x)")
            if i >= n - NCOL:
                ax.set_xlabel("neighbourhood source density")
        if sc is not None:
            fig.colorbar(sc, ax=axes[n - 1], shrink=0.85, label="local density")
        fig.text(1.005, 0.5, "Linear fit\n" + "\n".join(lines),
                 transform=fig.transFigure, ha="left", va="center",
                 fontsize=9.5, family="monospace", color="#333333")
        fig.suptitle(f"{cname} cells: contamination vs neighbourhood source "
                     f"density (r = {MAIN_R} um; identical axes across all panels)",
                     fontsize=12)
        fp = os.path.join(OUTDIR, f"joint_{cname}.pdf")
        fig.savefig(fp, bbox_inches="tight", dpi=600)
        fig.savefig(fp.replace(".pdf", ".png"), bbox_inches="tight", dpi=400)
        plt.close(fig)
        log(f">>> {fp}")


if __name__ == "__main__":
    main()