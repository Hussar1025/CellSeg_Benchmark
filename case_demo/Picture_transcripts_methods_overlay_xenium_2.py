# -*- coding: utf-8 -*-
"""
脚本 1/4 —— xenium_2 (乳腺) 图像方法掩膜可视化
================================================
五个图像分割方法 + GT，每个单独输出一个 PDF，便于在 Adobe 里逐个替换。
命名沿用 xenium_3 那版的约定：xenium_overlay_{方法}_{x0}_{y0}_{side}.pdf
其中 x0/y0 是裁剪块在掩膜里的起始像素。

坐标参数（已用 GT 细胞质心命中率标定，A 系 = cells.parquet 全局坐标）：
    ROI 原点 (5205.6125, 2382.125) µm = 像素 (24497, 11210) × 0.2125
    所有掩膜 5000×5000、0.2125 µm/px、原点即 ROI 原点
    标定命中率：GT 99.9% / CellSAM 95.4% / GeneSegNet 93.8% / Mesmer 93.5%
                Cellpose 90.8% / StarDist 81.8% / CelloType 69.4%

裁剪位置默认自动选：在 GT 掩膜上按滑窗找细胞覆盖率最高的一块。
乳腺切片细胞分布极不均匀，居中裁常落在间质空白区，所以默认自动选。
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tifffile

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# ==========================================================================
R = "/data/qiuyijia"
PX = 0.2125
X0_PX, Y0_PX = 24497, 11210                  # ROI 原点（全片像素）
X0, Y0 = X0_PX * PX, Y0_PX * PX
ROI_SIDE_UM = 1062.5

SIDE_UM = 300.0
AUTO_CENTER = True
AUTO_STRIDE_UM = 25.0           # 滑窗步长，越小越准但越慢

GT_MASK = f"{R}/CellSam/xenium_breast/xenium_breast_gt.tif"
METHODS = {
    "Cellpose":  f"{R}/cellpose/xenium_breast_out/xenium_breast_cellpose_pred.tif",
    "CellSAM":   f"{R}/CellSam/xenium_breast/xenium_breast_cellsam_pred.tif",
    "Mesmer":    f"{R}/Mesmer/xenium_breast_output/xenium_breast_mesmer_pred.tif",
    "StarDist":  f"{R}/stardist/xenium_breast/xenium_breast_stardist_pred.tif",
    "CelloType": f"{R}/cellotype/xenium_breast_output/xenium_breast_cellotype_pred.tif",
}

MORPH = f"{R}/dataset/xenium_breast/morphology_focus/ch0000_dapi.ome.tif"
USE_MORPH = True

OUT_DIR = f"{R}/viz"
OUT_PREFIX = "xenium_overlay"   # 与 xenium_3 那版一致
FILE_FORMAT = "pdf"
RASTER_DPI = 300

C_BORDER = (68 / 255, 144 / 255, 169 / 255, 0.95)     # 边界 #4490A9
C_OK = (91 / 255, 155 / 255, 213 / 255, 0.50)         # 归属正确 #5B9BD5
C_BAD = (237 / 255, 125 / 255, 49 / 255, 0.55)        # 错分 #ED7D31

PANEL_IN = 3.8                  # 单张图的边长（英寸）
FS = 12
SHOW_TITLE = True               # 关掉则只出纯图，便于自己在 AI 里加标题


# ==========================================================================
def boundaries(lab):
    """实例边界：与右/下邻居标签不同，且自身或邻居属于某个细胞。"""
    b = np.zeros(lab.shape, bool)
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    b[:-1, :] |= lab[:-1, :] != lab[1:, :]
    cell = lab > 0
    g = np.zeros_like(b)
    g[:, :-1] |= cell[:, 1:]; g[:, 1:] |= cell[:, :-1]
    g[:-1, :] |= cell[1:, :]; g[1:, :] |= cell[:-1, :]
    return b & (cell | g)


def classify(m, gt):
    """逐像素判断归属正误：预测细胞与哪个 GT 细胞重叠最多即主匹配。
    一次 bincount 求全部配对计数，比逐细胞做掩膜快两个数量级。"""
    green = np.zeros(m.shape, bool); red = np.zeros(m.shape, bool)
    sel = (m > 0) & (gt > 0)
    if not sel.any():
        return green, red
    mm = m[sel].astype(np.int64); gg = gt[sel].astype(np.int64)
    gmax = int(gg.max()) + 1
    codes, counts = np.unique(mm * gmax + gg, return_counts=True)
    pid, gid = codes // gmax, codes % gmax
    order = np.lexsort((-counts, pid))
    ps, gs = pid[order], gid[order]
    first = np.ones(len(ps), bool); first[1:] = ps[1:] != ps[:-1]
    main = np.zeros(int(m.max()) + 1, dtype=np.int64)
    main[ps[first]] = gs[first]
    same = gt == main[m]
    return sel & same, sel & ~same


def pick_dense_window(gt_full, side_px, stride_px):
    """滑窗找细胞覆盖率最高的一块。用积分图求任意窗口的前景像素数，O(1) 取值。"""
    fg = (gt_full > 0).astype(np.int32)
    ii = np.pad(fg.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    H, W = fg.shape
    rs = np.arange(0, H - side_px + 1, stride_px)
    cs = np.arange(0, W - side_px + 1, stride_px)
    best, br, bc = -1, 0, 0
    for r in rs:
        top, bot = ii[r], ii[r + side_px]
        v = bot[cs + side_px] - bot[cs] - top[cs + side_px] + top[cs]
        j = int(np.argmax(v))
        if v[j] > best:
            best, br, bc = int(v[j]), int(r), int(cs[j])
    cov = best / (side_px * side_px)
    n_cell = len(np.unique(gt_full[br:br + side_px, bc:bc + side_px])) - 1
    print(f"  自动选窗：row[{br}:{br+side_px}] col[{bc}:{bc+side_px}]  "
          f"细胞覆盖率 {cov:.1%}，含 {n_cell} 个细胞")
    return br, bc


def load_morph_crop(path, r0, r1, c0, c1):
    """全片金字塔 OME-TIFF 惰性裁剪。整幅 27420×53994、12 GB，不能 imread。"""
    if not (USE_MORPH and os.path.exists(path)):
        return None
    try:
        import zarr
        with tifffile.TiffFile(path) as tf:
            ser = tf.series[0]
            lv = ser.levels[0] if getattr(ser, "levels", None) else ser
            z = zarr.open(lv.aszarr(), mode="r")
            if z.ndim == 2:
                a = np.asarray(z[r0:r1, c0:c1])
            elif z.ndim == 3 and z.shape[0] <= 8:      # 金字塔层被当成一个维度
                a = np.asarray(z[0, r0:r1, c0:c1])
            else:
                a = np.asarray(z[r0:r1, c0:c1, 0])
        print(f"  形态底图裁剪 {a.shape}（全片 {tuple(z.shape)}）")
        return a
    except Exception as e:
        print(f"  [提示] 形态图读取失败（{type(e).__name__}: {e}），改用纯白底")
        return None


def norm(a, lo=1, hi=99.5, invert=True):
    a = a.astype(np.float32)
    l, h = np.percentile(a, [lo, hi]); h = h if h > l else l + 1
    v = np.clip((a - l) / (h - l), 0, 1)
    return 1.0 - v if invert else v


def save_panel(name, lab_c, gtc, bg, out_path):
    """单个方法一张图。"""
    fig = plt.figure(figsize=(PANEL_IN, PANEL_IN), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])              # 铺满，无白边
    ax.imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="nearest",
              rasterized=True)
    if name != "GT":
        green, red = classify(lab_c, gtc)
        rgba = np.zeros((*gtc.shape, 4), np.float32)
        rgba[green] = C_OK; rgba[red] = C_BAD
        ax.imshow(rgba, interpolation="nearest", rasterized=True)
        tot = int(green.sum() + red.sum())
        rate = red.sum() / max(tot, 1)
        print(f"    {name:<11} 错分像素 {int(red.sum()):>9,} / {tot:,}  ({rate:.1%})")
        title = f"{name}   misassign = {rate:.1%}"
    else:
        title = f"Ground truth   {len(np.unique(lab_c))-1} cells"
    bm = boundaries(lab_c)
    ov = np.zeros((*bm.shape, 4), np.float32); ov[bm] = C_BORDER
    ax.imshow(ov, interpolation="nearest", rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    if SHOW_TITLE:
        ax.set_title(title, fontsize=FS, pad=6)
    fig.savefig(out_path, dpi=RASTER_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"      -> {os.path.basename(out_path)}  "
          f"({os.path.getsize(out_path)/1e6:.2f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cx", type=float, default=None, help="裁剪中心 x（µm），不给则自动选")
    ap.add_argument("--cy", type=float, default=None)
    ap.add_argument("--side", type=float, default=SIDE_UM)
    ap.add_argument("--no-auto", action="store_true")
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()

    gt_full = tifffile.imread(GT_MASK)
    side_px = int(round(a.side / PX))

    if a.cx is not None and a.cy is not None:
        c0 = int(round((a.cx - a.side / 2 - X0) / PX))
        r0 = int(round((a.cy - a.side / 2 - Y0) / PX))
    elif AUTO_CENTER and not a.no_auto:
        r0, c0 = pick_dense_window(gt_full, side_px,
                                   max(1, int(round(AUTO_STRIDE_UM / PX))))
    else:
        c0 = r0 = int(round((ROI_SIDE_UM / 2 - a.side / 2) / PX))
    r0 = max(0, min(r0, gt_full.shape[0] - side_px))
    c0 = max(0, min(c0, gt_full.shape[1] - side_px))
    r1, c1 = r0 + side_px, c0 + side_px
    cx = X0 + (c0 + side_px / 2) * PX
    cy = Y0 + (r0 + side_px / 2) * PX
    print(f"展示中心 ({cx:.1f}, {cy:.1f}) µm，边长 {a.side:.0f} µm")
    print(f"  掩膜像素 row[{r0}:{r1}] col[{c0}:{c1}]")

    gtc = gt_full[r0:r1, c0:c1]
    print(f"  GT 裁剪块 {gtc.shape}，{len(np.unique(gtc))-1} 个细胞，"
          f"前景占 {(gtc>0).mean():.1%}")

    mo = load_morph_crop(MORPH, r0 + Y0_PX, r1 + Y0_PX, c0 + X0_PX, c1 + X0_PX)
    bg = norm(mo) if (mo is not None and mo.shape == gtc.shape) \
        else np.ones(gtc.shape, np.float32)
    if mo is not None and mo.shape != gtc.shape:
        print(f"  [提示] 形态裁剪 {mo.shape} 与掩膜 {gtc.shape} 不一致，改用纯白底")

    os.makedirs(a.out, exist_ok=True)
    tag = f"{c0}_{r0}_{int(a.side)}"
    save_panel("GT", gtc, gtc, bg,
               os.path.join(a.out, f"{OUT_PREFIX}_GT_{tag}.{FILE_FORMAT}"))
    for name, p in METHODS.items():
        if not os.path.exists(p):
            print(f"    [跳过] {name}: 找不到 {p}"); continue
        lab = tifffile.imread(p)
        if lab.shape != gt_full.shape:
            print(f"    [跳过] {name}: 尺寸 {lab.shape} 与 GT {gt_full.shape} 不一致")
            continue
        save_panel(name, lab[r0:r1, c0:c1], gtc, bg,
                   os.path.join(a.out, f"{OUT_PREFIX}_{name}_{tag}.{FILE_FORMAT}"))
    print(f">>> 全部保存在 {a.out}")


if __name__ == "__main__":
    main()