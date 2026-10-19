#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xenium_8 (ccRCC): 分割掩膜 + 转录本点 放大对比图  (T vs 肿瘤)
=============================================================
在同一个小 ROI 上并排展示 ProSeg 与 Cellpose 的分割, 直观呈现边界差异
如何影响转录本纯净性。

细胞身份
  T 细胞      : CD3e 蛋白阳性 (独立于转录本)
  肿瘤细胞    : 优先用 PanCK 蛋白; 若阳性细胞过少则退回转录本判定
                (Epi/Tumor marker 占优), 日志与图例会标明实际所用方式

绘制层次 (由下至上)
  1. DAPI 核背景 (灰度)
  2. 细胞填充色: T = 浅橙, 肿瘤 = 浅紫, 其他 = 浅灰 (低透明度, 不遮核)
  3. 细胞边界: 用 skimage.find_contours 提取轮廓后以固定线宽绘制,
               因此两个方法的线条粗细完全一致, 不受 mask 分辨率影响
  4. 转录本点: 蓝 = 该细胞身份对应谱系的特异基因 (自身信号)
               橙 = 其他谱系的特异基因 (外来 -> 污染)
               浅灰 = 非谱系特异基因 (管家等), 可关闭
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
from scipy.spatial import cKDTree

from panel import (OUTDIR, MARK, LINEAGES, PX, X0, Y0, SZ,
                   ORANGE, BLUE, DISPLAY, POS_AREA_FRAC)
from common import (log, read_ch, protein_pos, get_mask, load_transcripts,
                    cell_table, down_frac)

# ================= 可调参数 =================
METHODS = ["proseg", "cellpose"]
ROI_UM = 40.0                  # 搜索用视野边长 (um)
N_CELL_TARGET = (4, 12)        # 视野内细胞数期望区间
MAX_UNASSIGNED_TRY = [0, 1, 2, 3, 99]   # 允许的 unassigned 细胞数, 逐级放宽
SUB_FRAC = 0.75                # 最终视野 = 40 * 0.75 = 30 um
SUB_CORNER = "SW"              # SW=左下 / SE=右下 / NW=左上 / NE=右上
SEARCH_TOP = 800
NEIGHBOR_UM = 12.0             # 判断"被包围"时的邻域半径
MIN_TUMOR_AROUND = 2           # 视野内至少要有几个肿瘤细胞
SHOW_OTHER_TX = True           # 是否画非谱系特异转录本(浅灰)
PT_SIZE = 26
LINE_W = 1.2                   # 细胞轮廓线宽(两方法一致)
FILL_ALPHA = 0.42              # 掩膜填充透明度
BG_ALPHA = 1.0                 # DAPI 背景不透明度
BG_GAMMA = 0.65                # <1 提亮暗部, 让核更醒目
# 身份完全由蛋白决定, 不使用任何转录本信息 -> 与污染判定完全正交

CH_T = "ch0011_cd3e"        # T 细胞
CH_MYE = "ch0015_cd68"      # 髓系
CH_TUMOR = "ch0030_panck"   # 上皮/肿瘤
CH_CD45 = "ch0033_cd45"     # 泛白细胞(用于"其他免疫细胞")
LIN_T = "T"
LIN_MYE = "Myeloid"
LIN_TUMOR = "Epi/Tumor"
LIN_IMM = "other immune"

# 仅蛋白可判定的身份(证据纯净): 其余一律 unassigned
# 指定配色(仅此四色): (68,144,169) (91,155,213) (255,192,0) (237,125,49)
# 其中橙色 (237,125,49) 专门留给"外来转录本", 掩膜不使用。
# 无蛋白身份者无可用色, 以中性浅灰占位。
EDGE = {
    LIN_T:        "#4490A9",   # (68,144,169)  CD3e+ 蛋白确证
    LIN_MYE:      "#5B9BD5",   # (91,155,213)  CD68+ 蛋白确证
    LIN_TUMOR:    "#FFC000",   # (255,192,0)   PanCK+ 或 无蛋白身份者(见下)
}
def _light(hexc, f=0.72):
    from matplotlib.colors import to_rgb
    r, g_, b = to_rgb(hexc)
    return (r + (1 - r) * f, g_ + (1 - g_) * f, b + (1 - b) * f)


def _dark(hexc, f=0.35):
    """加深: 用于自身转录本点, 与同色的半透明掩膜底色拉开对比"""
    from matplotlib.colors import to_rgb
    r, g_, b = to_rgb(hexc)
    return (r * (1 - f), g_ * (1 - f), b * (1 - f))


FILL = {k: _light(v) for k, v in EDGE.items()}
DOT = {k: _dark(v) for k, v in EDGE.items()}      # 自身转录本点用色
# 身份 -> 用于判定转录本"自身/外来"的谱系
IDENT2LIN = {LIN_T: "T", LIN_MYE: "Myeloid", LIN_TUMOR: "Epi/Tumor"}
TX_FOREIGN = "#ED7D31"        # (237,125,49) 外来转录本专用
TX_OTHER = "#D9D9D9"          # 非谱系特异转录本(灰)


def identity_protein(tab, posmap):
    """身份完全由蛋白通道决定, 不使用任何转录本信息。
    优先级: CD3e+ -> T;  CD68+ -> Myeloid;  PanCK+ -> Epi/Tumor;
            余下 CD45+ -> other immune;  其他 -> unassigned
    多通道同时阳性时, 取阳性面积占比最高者。"""
    nl = tab["nl"]
    f = tab["f"]
    mask = tab["mask"]

    def pos_frac(binary):
        pf = down_frac(binary, f)
        h = min(pf.shape[0], mask.shape[0])
        w = min(pf.shape[1], mask.shape[1])
        sub = mask[:h, :w].ravel()
        sel = sub > 0
        num = np.bincount(tab["idx_map"][sub[sel]],
                          weights=pf[:h, :w].ravel()[sel], minlength=nl)
        den = np.bincount(tab["idx_map"][sub[sel]], minlength=nl).astype(np.float32)
        return num / np.clip(den, 1, None)

    fr = {k: pos_frac(v) for k, v in posmap.items()}
    keys = [LIN_T, LIN_MYE, LIN_TUMOR]
    M = np.vstack([fr[k] for k in keys])            # 3 x cell
    best_i = M.argmax(0)
    best_v = M.max(0)
    ident = np.where(best_v >= POS_AREA_FRAC,
                     np.array(keys, dtype=object)[best_i], "unassigned")
    ident = ident.astype(object)
    n_prot = {k: int((ident == k).sum()) for k in keys}
    # 无蛋白身份者统一归为上皮/肿瘤:
    #   本样本为 ccRCC, 肿瘤细胞占主体, 而 PanCK 染色偏弱(阳性像素仅 1.26%),
    #   未被任何通道判定者多为 PanCK 染色不足的肿瘤细胞, 假阴性代价小。
    #   注意: 这一步引入先验假设, 非纯蛋白证据, 解读时需区别对待。
    confirmed = ident != "unassigned"        # 是否由蛋白通道确证
    n_fallback = int((~confirmed).sum())
    ident[ident == "unassigned"] = LIN_TUMOR
    from collections import Counter
    log("    身份分布: " + str(dict(Counter(ident).most_common())))
    for k in keys:
        log(f"      {k:10s} 蛋白确证 {n_prot[k]} cells")
    log(f"      其中 {n_fallback} 个无蛋白身份者按假设归入 {LIN_TUMOR}")
    return ident, confirmed, ("CD3e/CD68/PanCK protein; cells without any "
                              "protein marker assigned to Epi/Tumor")


def pick_roi(tab, ident, gm, tix, tiy):
    """选 ROI: 找一个被上皮/肿瘤细胞(PanCK+)包围的 T 细胞(CD3e+)。
    评分 = 邻域内 PanCK+ 细胞数(主) + 邻域肿瘤转录本密度(次),
    并要求视野内细胞数适中。"""
    t_idx = np.nonzero(ident == LIN_T)[0]
    tum_idx = np.nonzero(ident == LIN_TUMOR)[0]
    if len(t_idx) == 0:
        raise RuntimeError("没有蛋白确认的 T 细胞")
    if len(tum_idx) == 0:
        log("  警告: 没有 PanCK+ 细胞, 退回按肿瘤转录本密度选 ROI")
    cen_t = tab["cen_px"][t_idx]
    allc = tab["cen_px"]
    half = ROI_UM / 2 / PX
    r_near = NEIGHBOR_UM / PX

    # 邻域内的肿瘤细胞数
    if len(tum_idx):
        tree_c = cKDTree(tab["cen_px"][tum_idx])
        n_tum_cell = np.array([len(q) for q in
                               tree_c.query_ball_point(cen_t, r_near)])
    else:
        n_tum_cell = np.zeros(len(cen_t))
    # 邻域内的肿瘤转录本数
    tree_x = cKDTree(np.c_[tix[gm["Epi/Tumor"]], tiy[gm["Epi/Tumor"]]])
    n_tum_tx = np.array([len(q) for q in
                         tree_x.query_ball_point(cen_t, r_near)])
    score = n_tum_cell * 1000 + n_tum_tx          # 肿瘤细胞数优先
    order = np.argsort(score)[::-1][:SEARCH_TOP]

    # 分级搜索: 先要求视野内没有 unassigned 细胞, 找不到再逐步放宽
    for max_un in MAX_UNASSIGNED_TRY:
        for k in order:
            cx, cy = cen_t[k]
            inb = ((np.abs(allc[:, 0] - cx) < half) &
                   (np.abs(allc[:, 1] - cy) < half))
            n = int(inb.sum())
            if not (N_CELL_TARGET[0] <= n <= N_CELL_TARGET[1]):
                continue
            sub_id = ident[np.nonzero(inb)[0]]
            n_un = 0          # 已无 unassigned(统一归入 Epi/Tumor)
            n_tum_in = int((sub_id == LIN_TUMOR).sum())
            if n_tum_in < MIN_TUMOR_AROUND:
                continue
            from collections import Counter
            log(f"  ROI 中心=({cx:.0f},{cy:.0f})px  "
                f"邻域肿瘤细胞={int(n_tum_cell[k])}  "
                f"邻域肿瘤转录本={int(n_tum_tx[k])}")
            log(f"    视野内细胞={n}  构成: {dict(Counter(sub_id).most_common())}")
            log(f"    (本次允许的 unassigned 上限 = "
                f"{'不限' if max_un >= 99 else max_un}, 实际 {n_un})")
            return float(cx), float(cy)
        if max_un < 99:
            log(f"  无满足 unassigned<={max_un} 的位置, 放宽条件重试")

    k = order[0]
    cx, cy = cen_t[k]
    log(f"  未找到完全符合条件者, 用评分最高的 T 细胞: ({cx:.0f},{cy:.0f})  "
        f"邻域肿瘤细胞={int(n_tum_cell[k])}")
    return float(cx), float(cy)


def draw_panel(ax, name, tab, ident, confirmed, bg_dapi, bg_ext,
               gm, tix, tiy, cx, cy, roi_um):
    """分层: DAPI 背景 -> 灰色非谱系转录本 -> 掩膜着色 -> 谱系特异转录本"""
    mask = tab["mask"]
    scale = tab["scale"]
    half = roi_um / 2 / PX
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half
    ext = [(x0 + X0) * PX, (x1 + X0) * PX, (y1 + Y0) * PX, (y0 + Y0) * PX]

    # --- L0: DAPI 背景 ---
    ax.imshow(bg_dapi, extent=bg_ext, cmap="gray_r", vmin=0, vmax=1,
              alpha=BG_ALPHA, interpolation="bilinear", zorder=0)

    # 视野内转录本及其宿主/谱系
    inroi = (tix >= x0) & (tix < x1) & (tiy >= y0) & (tiy < y1)
    idx = np.nonzero(inroi)[0]
    px_ = (tix[idx] + X0) * PX
    py_ = (tiy[idx] + Y0) * PX
    cl = mask[np.clip((tiy[idx] / scale).astype(int), 0, mask.shape[0] - 1),
              np.clip((tix[idx] / scale).astype(int), 0, mask.shape[1] - 1)]
    host = np.where(cl > 0,
                    tab["idx_map"][np.clip(cl, 0, tab["idx_map"].size - 1)], -1)
    host_id = np.array(["none"] * len(idx), dtype=object)
    ok = host >= 0
    host_id[ok] = ident[host[ok]]
    lin_of = np.array(["other"] * len(idx), dtype=object)
    for a in LINEAGES:
        lin_of[gm[a][idx]] = a
    is_lin = lin_of != "other"
    host_lin = np.array([IDENT2LIN.get(h, "") for h in host_id], dtype=object)
    # 只有"蛋白确证身份"的细胞才参与污染判定;
    # 按假设归入 Epi/Tumor 的细胞不标橙点, 其转录本并入灰色背景层
    host_conf = np.zeros(len(idx), bool)
    host_conf[ok] = confirmed[host[ok]]
    judgeable = (host_lin != "") & host_conf
    is_self = is_lin & judgeable & (lin_of == host_lin)
    is_frn = is_lin & judgeable & (lin_of != host_lin)

    # --- L1: 灰色非谱系转录本(位于掩膜之下) ---
    if SHOW_OTHER_TX:
        m = (~is_lin) | (is_lin & ~judgeable)   # 非谱系 + 身份未确证细胞内的
        ax.scatter(px_[m], py_[m], s=PT_SIZE * 0.42, c=TX_OTHER, linewidths=0,
                   zorder=1, alpha=0.85)

    # --- L2 + L3: 用轮廓多边形同时填充与描边 ---
    # 填充与边界线共用同一组顶点, 因此严格贴合, 不受 mask 分辨率影响
    mx0, my0 = max(int(x0 / scale), 0), max(int(y0 / scale), 0)
    mx1 = min(int(np.ceil(x1 / scale)), mask.shape[1])
    my1 = min(int(np.ceil(y1 / scale)), mask.shape[0])
    sub = mask[my0:my1, mx0:mx1]
    labs = np.unique(sub)
    labs = labs[labs > 0]
    try:
        from skimage.measure import find_contours
        has_sk = True
    except ImportError:
        has_sk = False
        log("    (无 skimage, 退回像素填充)")

    if has_sk:
        for lb in labs:
            i = tab["idx_map"][lb]
            idt = ident[i] if i < len(ident) else LIN_TUMOR
            col = EDGE[idt if idt in EDGE else LIN_TUMOR]
            m = (sub == lb).astype(float)
            # 边缘补零, 保证贴边细胞也能闭合
            mp = np.pad(m, 1, mode="constant")
            for c in find_contours(mp, 0.5):
                gx = ((c[:, 1] - 1 + mx0 + 0.5) * scale + X0) * PX
                gy = ((c[:, 0] - 1 + my0 + 0.5) * scale + Y0) * PX
                ax.fill(gx, gy, facecolor=col, alpha=FILL_ALPHA,
                        edgecolor="none", zorder=2)
                ax.plot(gx, gy, color=col, lw=LINE_W, zorder=3,
                        solid_joinstyle="round")
    else:
        sub_ext = [(mx0 * scale + X0) * PX, (mx1 * scale + X0) * PX,
                   (my1 * scale + Y0) * PX, (my0 * scale + Y0) * PX]
        rgba = np.zeros((*sub.shape, 4), np.float32)
        for lb in labs:
            i = tab["idx_map"][lb]
            idt = ident[i] if i < len(ident) else LIN_TUMOR
            rgba[sub == lb] = to_rgba(EDGE[idt if idt in EDGE else LIN_TUMOR],
                                      alpha=FILL_ALPHA)
        ax.imshow(rgba, extent=sub_ext, interpolation="nearest", zorder=2)

    # --- L4: 谱系特异转录本(最上层) ---
    #   自身信号: 用宿主细胞同色;  外来信号: 统一橙色
    for idt in set(host_id[is_self]):
        m = is_self & (host_id == idt)
        if m.any():
            ax.scatter(px_[m], py_[m], s=PT_SIZE * 1.1,
                       c=[DOT.get(idt, "#222222")], linewidths=0.5,
                       edgecolors="#FFFFFFCC", zorder=6)
    ax.scatter(px_[is_frn], py_[is_frn], s=PT_SIZE * 1.1, c=TX_FOREIGN,
               linewidths=0.5, edgecolors="#FFFFFFCC", zorder=7)

    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.set_box_aspect(1.0)          # 强制绘图框为正方形
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{DISPLAY.get(name, name)}   cells: {len(labs)}\n"
                 f"own {int(is_self.sum())}   foreign {int(is_frn.sum())}",
                 fontsize=11)
    return int(is_self.sum()), int(is_frn.sum())


def main():
    log("== 蛋白阳性区 ==")
    posmap = {LIN_T: protein_pos(CH_T),
              LIN_MYE: protein_pos(CH_MYE),
              LIN_TUMOR: protein_pos(CH_TUMOR)}

    log("\n== 转录本 ==")
    g, tix, tiy, gm, expect = load_transcripts()

    log("\n== 各方法细胞表与身份 ==")
    tabs, idents, confs = {}, {}, {}
    mode = ""
    for name in METHODS:
        log(f"  {name}")
        tab = cell_table(get_mask(name), tix, tiy, gm, {})
        ident, confirmed, mode = identity_protein(tab, posmap)
        tabs[name], idents[name] = tab, ident
        confs[name] = confirmed
        log(f"    cells={tab['nl']}")

    log("\n== 选取 ROI (以 %s 为准, 两图共用) ==" % METHODS[0])
    cx, cy = pick_roi(tabs[METHODS[0]], idents[METHODS[0]], gm, tix, tiy)

    # 取选中 ROI 的一角作为最终视野
    half_full = ROI_UM / 2 / PX
    roi_um = ROI_UM * SUB_FRAC
    half = roi_um / 2 / PX
    dxs = -1 if SUB_CORNER in ("SW", "NW") else 1      # 左/右
    dys = 1 if SUB_CORNER in ("SW", "SE") else -1      # 下/上 (y 向下为正)
    cx = cx + dxs * (half_full - half)
    cy = cy + dys * (half_full - half)
    # 钳制中心, 保证整个视野落在图像范围内(否则背景被截断会出现留白)
    cx = float(np.clip(cx, half, SZ - half))
    cy = float(np.clip(cy, half, SZ - half))
    log(f"  取 {SUB_CORNER} 角, 视野 {roi_um:.0f} um, 中心=({cx:.0f},{cy:.0f})px")

    log("\n== 读取 DAPI 背景 ==")
    dx0, dy0 = int(max(cx - half, 0)), int(max(cy - half, 0))
    dx1, dy1 = int(min(cx + half, SZ)), int(min(cy + half, SZ))
    dapi_full = read_ch("ch0000_dapi")
    d = dapi_full[dy0:dy1, dx0:dx1].astype(np.float32)
    lo, hi = np.percentile(d, [1, 99.8])
    bg = np.clip((d - lo) / max(hi - lo, 1), 0, 1) ** BG_GAMMA
    bg_ext = [(dx0 + X0) * PX, (dx1 + X0) * PX,
              (dy1 + Y0) * PX, (dy0 + Y0) * PX]

    log("\n== 绘图 ==")
    fig, axes = plt.subplots(1, len(METHODS), figsize=(6.0 * len(METHODS), 6.4),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, METHODS):
        ns, nf = draw_panel(ax, name, tabs[name], idents[name], confs[name],
                            bg, bg_ext, gm, tix, tiy, cx, cy, roi_um)
        log(f"  {name}: own={ns} foreign={nf}")

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=DOT[LIN_T],
               markeredgecolor="#FFFFFF", markersize=9,
               label="own-lineage transcript (darker host colour)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TX_FOREIGN,
               markersize=9, label="foreign-lineage transcript"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TX_OTHER,
               markersize=7, label="non-lineage transcript"),
    ]
    handles.append(Line2D([0], [0], color="none", label=""))
    for L, lab in [(LIN_T, "T cell (CD3e+)"),
                   (LIN_MYE, "myeloid (CD68+)"),
                   (LIN_TUMOR, "epithelial / tumour (PanCK+ or no marker)")]:
        handles.append(Line2D([0], [0], marker="s", color=EDGE[L],
                              markerfacecolor=FILL[L], markersize=11,
                              linestyle="none", markeredgewidth=1.4, label=lab))
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
               frameon=False, fontsize=10)
    fig.suptitle(f"Segmentation boundaries and transcript purity\n"
                 f"same {int(ROI_UM*SUB_FRAC)} um field;  cell identity from {mode}\n"
                 f"(orange shown only for protein-confirmed cells)",
                 fontsize=12)
    fp = os.path.join(OUTDIR, "zoom_T_tumor.pdf")
    fig.savefig(fp, bbox_inches="tight", dpi=600)
    fig.savefig(fp.replace(".pdf", ".png"), bbox_inches="tight", dpi=400)
    plt.close(fig)
    log(f">>> {fp}")


if __name__ == "__main__":
    main()