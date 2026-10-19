#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xenium_8 (ccRCC) 污染分析 v4
主 case: 蛋白定义的 T 细胞 (CD3e+) 被肿瘤转录本 (KRT8/18/KRT7/MUC1/CDH1/EPCAM) 污染
对照   : 蛋白定义的髓系细胞 (CD68+)、上皮/肿瘤细胞 (PanCK+)
指标   : 胞内 外来谱系 / 自身谱系 转录本比值 (无参照系, 直接可读)
输出   : 空间图 / 气泡矩阵 / 联合分布 / 累积剖面(边界原点 + 归一化相对距离)

v4 相对 v3 的修改（计算逻辑不变）：
  · 方法列表来自 common.all_methods()（16 个方法中找到结果的那些），某个方法读取失败
    或与 GT 不对齐时跳过并记录，不影响其他方法
  · 气泡矩阵的配色改用 panel.CASE_COLORS（v3 本地只定义了 T / Myeloid，第三个 case
    Epi/Tumor 会报 KeyError）
  · 气泡描边 1.4 -> 0.8，与 Fig. 2 的气泡图保持一致
  · 方法显示名统一用 panel.DISPLAY（Cellpose 显示为 Cellpose-SAM）
先运行 python common.py --check 确认各方法的输入。
"""
import os
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Polygon, Rectangle

from panel import (OUTDIR, MARK, LINEAGES, CASES, PX, X0, Y0, SZ,
                   ORANGE, BLUE, GREY, MCOL, DISPLAY, CASE_COLORS as _PANEL_CASE_COLORS,
                   MIN_TX, MIN_OWN, ENV_RING)

# case 配色：T / 髓系沿用 v3 的两色（橙 / 蓝），其余 case 用 panel 中的颜色
CASE_COLORS = dict(_PANEL_CASE_COLORS, T=ORANGE, Myeloid=BLUE)
from common import (log, read_ch, protein_pos, get_mask, all_methods,
                    load_transcripts, cell_table, protein_identity)

CMAP_C = LinearSegmentedColormap.from_list("bo", [BLUE, "#FFFFFF", ORANGE])
CLEAN_BLUE = BLUE
NCOL = 4

_IN_UM = 10.0
_OUT_UM = _IN_UM / 3.0
BIN_B = np.concatenate([np.arange(-_IN_UM, 0.0, 0.5),
                        np.arange(0.0, _OUT_UM + 0.25, 1.0 / 3.0)])
_REL_OUT = 1.0 / 3.0
BIN_R = np.concatenate([np.arange(0.0, 1.0, 0.05),
                        np.arange(1.0, 1.0 + _REL_OUT + 0.01, 0.05 / 3)])

# 气泡矩阵外观（配色来自 panel.CASE_COLORS）
ROW_BG_A, ROW_BG_B = "#EDEDED", "#FFFFFF"
EDGE_C, EDGE_W = "#1A1A1A", 0.8          # 气泡描边（与 bubble_xenium8.py 一致）
LIGHT_MAX, LIGHT_GAMMA = 0.97, 0.65
HALF_MIN, HALF_MAX = 0.077, 0.300
GAP_EPS = 0.004                           # 气泡间额外留白（英寸）
MIN_INDENT = EDGE_W / 72.0 / 2.0 + GAP_EPS
SQ_RANGE, SQ_POWER = (2.0, 8.0), 6.0
CAT_GAP = 0.5
LABEL_ROT, LEADER_IN = 45, 0.13
FS = 12
SCORE_LO, SCORE_SPAN = 0.05, 0.90
_SCRATCH = None

# 每个 (方法, case) 至少需要的可评估细胞数；低于则跳过
MIN_CELLS = 30
# 强制放行：即使可评估细胞少于 MIN_CELLS 也保留并画全所有图（细胞数为 0 时仍跳过）。
# DISSECT 只检出约 1/4 的细胞，蛋白定义的髓系细胞仅剩十余个；结果仅作展示，图注需注明 n。
FORCE_KEEP = {("dissect", "Myeloid")}
MIN_FIT_FORCED = 3          # 强制放行的组合在联合分布图中做线性拟合所需的最少点数（常规为 20）


def text_w(t, fs):
    global _SCRATCH
    if _SCRATCH is None:
        _SCRATCH = plt.figure(figsize=(1, 1), dpi=100)
    o = _SCRATCH.text(0, 0, str(t), fontsize=fs)
    _SCRATCH.canvas.draw()
    bb = o.get_window_extent(renderer=_SCRATCH.canvas.get_renderer())
    o.remove()
    return bb.width / _SCRATCH.dpi


def superellipse(cx, cy, half, n, k=180):
    t = np.linspace(0, 2 * np.pi, k)
    ct, st = np.cos(t), np.sin(t)
    return np.column_stack([cx + half * np.sign(ct) * np.abs(ct) ** (2.0 / n),
                            cy + half * np.sign(st) * np.abs(st) ** (2.0 / n)])


def shade(base, v):
    rgb = np.asarray(to_rgb(base), float)
    t = float(np.clip(v, 0, 1)) ** LIGHT_GAMMA * LIGHT_MAX
    return tuple(rgb + (1.0 - rgb) * t)


def shape_of(v):
    v = float(np.clip(v, 0, 1))
    lo, hi = SQ_RANGE
    return HALF_MIN + v * (HALF_MAX - HALF_MIN), lo + (v ** SQ_POWER) * (hi - lo)


def signed_distance(binary, px_um):
    d_out = ndi.distance_transform_edt(~binary) * px_um
    d_in = ndi.distance_transform_edt(binary) * px_um
    return np.where(binary, -d_in, d_out)


# =====================================================================
def compute():
    log("== 蛋白阳性区 ==")
    posmap = {c: protein_pos(ch) for c, (ch, _) in CASES.items()}
    log("\n== 转录本 ==")
    g, tix, tiy, gm, expect = load_transcripts()

    methods, skipped = [], {}
    store = {}
    for name in all_methods():
        log(f"\n#### {DISPLAY.get(name, name)} ####")
        try:
            mask = get_mask(name)
        except Exception as e:
            log(f"  [跳过] {type(e).__name__}: {e}")
            skipped[name] = str(e)
            continue
        methods.append(name)
        tab = cell_table(mask, tix, tiy, gm, posmap)
        ident = protein_identity(tab)
        cnt = tab["cnt"]
        tot_lin = np.sum([cnt[a] for a in LINEAGES], 0)

        for cname, (_ch, self_lin) in CASES.items():
            sel = ident == cname
            own = cnt[self_lin]
            frn = tot_lin - own
            ratio = np.where(own > 0, frn / np.clip(own, 1, None), np.nan)
            ev = sel & (own >= MIN_OWN) & (tot_lin >= MIN_TX)
            n_cell = int(ev.sum())
            forced = (name, cname) in FORCE_KEEP
            if n_cell == 0 or (n_cell < MIN_CELLS and not forced):
                log(f"  {cname}: 细胞过少({n_cell}), 跳过")
                continue
            if n_cell < MIN_CELLS:
                log(f"  {cname}: 细胞仅 {n_cell} 个（< {MIN_CELLS}），按 FORCE_KEEP 强制保留")

            src = {a: float(cnt[a][ev].sum()) for a in LINEAGES}
            fsum = sum(src[a] for a in LINEAGES if a != self_lin)
            share = {a: (src[a] / fsum if fsum and a != self_lin else np.nan)
                     for a in LINEAGES}

            keep_lab = tab["labels"][ev]
            binary = np.isin(mask, keep_lab)
            dist, (iy_n, ix_n) = ndi.distance_transform_edt(~binary,
                                                            return_indices=True)
            d_um = dist * tab["px_um"]
            ring = (~binary) & (d_um > 0) & (d_um <= ENV_RING)
            owner = np.zeros(mask.shape, dtype=np.int64)
            owner[ring] = mask[iy_n[ring], ix_n[ring]]
            ring_area = np.bincount(
                tab["idx_map"][np.clip(owner[ring], 0, tab["idx_map"].size - 1)],
                minlength=tab["nl"]).astype(np.float32) * (tab["px_um"] ** 2)
            frn_tx = np.zeros(len(g), bool)
            for a in LINEAGES:
                if a != self_lin:
                    frn_tx |= gm[a]
            in_ring = ring[tab["my"], tab["mx"]] & frn_tx
            own_of = owner[tab["my"][in_ring], tab["mx"][in_ring]]
            env_c = np.bincount(
                tab["idx_map"][np.clip(own_of, 0, tab["idx_map"].size - 1)],
                minlength=tab["nl"]).astype(np.float32)
            env_dens = np.where(ring_area > 0,
                                env_c / np.clip(ring_area, 1e-9, None), np.nan)

            sd = signed_distance(binary, tab["px_um"])
            d_tx = sd[tab["my"], tab["mx"]]
            own_m = gm[self_lin]
            hb_o, _ = np.histogram(d_tx[own_m], bins=BIN_B)
            hb_f, _ = np.histogram(d_tx[frn_tx], bins=BIN_B)
            mid_b = 0.5 * (BIN_B[1:] + BIN_B[:-1])
            i0 = int(np.clip(np.searchsorted(BIN_B, 0.0) - 1, 0, len(mid_b) - 1))
            area_px_cell = np.bincount(
                tab["idx_map"][mask[np.nonzero(binary)]],
                minlength=tab["nl"]).astype(np.float32)
            R_cell = np.sqrt(np.clip(area_px_cell * tab["px_um"] ** 2, 1e-9, None)
                             / np.pi)
            owner_all = np.where(binary, mask, mask[iy_n, ix_n])
            R_map = R_cell[tab["idx_map"][np.clip(owner_all, 0,
                                                  tab["idx_map"].size - 1)]]
            rel = 1.0 + d_tx / np.clip(R_map[tab["my"], tab["mx"]], 1e-6, None)
            hr_o, _ = np.histogram(rel[own_m], bins=BIN_R)
            hr_f, _ = np.histogram(rel[frn_tx], bins=BIN_R)
            mid_r = 0.5 * (BIN_R[1:] + BIN_R[:-1])
            i1 = int(np.clip(np.searchsorted(BIN_R, 1.0) - 1, 0, len(mid_r) - 1))

            log(f"  {cname}: 细胞={n_cell} 中位比值={np.nanmedian(ratio[ev]):.2f}"
                f"  来源=" + str({a: (round(share[a], 3)
                                    if np.isfinite(share[a]) else 'self')
                                  for a in LINEAGES}))
            store[(name, cname)] = dict(
                ev=ev, ratio=ratio, cen=tab["cen_um"], frn_n=frn,
                share=share, self_lin=self_lin, n_cell=n_cell,
                env=env_dens[ev],
                mid_b=mid_b, cum_b_own=np.cumsum(hb_o) / n_cell,
                cum_b_frn=np.cumsum(hb_f) / n_cell, i0=i0,
                mid_r=mid_r, cum_r_own=np.cumsum(hr_o) / n_cell,
                cum_r_frn=np.cumsum(hr_f) / n_cell, i1=i1,
                R_med=float(np.median(R_cell[ev])))
    if skipped:
        log("\n== 跳过的方法 ==")
        for k, v in skipped.items():
            log(f"  {DISPLAY.get(k, k)}: {v}")
    return methods, store


# =====================================================================
def plot_spatial(methods, store):
    dapi = read_ch("ch0000_dapi", step=4)
    lo, hi = np.percentile(dapi, [5, 99])
    bg = np.clip((dapi - lo) / max(hi - lo, 1), 0, 1)
    extent = [X0 * PX, (X0 + SZ) * PX, (Y0 + SZ) * PX, Y0 * PX]
    for cname in CASES:
        have = [m for m in methods if (m, cname) in store]
        if not have:
            continue
        nrow = int(np.ceil(len(have) / NCOL))
        fig, axes = plt.subplots(nrow, NCOL, figsize=(5.0 * NCOL, 5.3 * nrow),
                                 constrained_layout=True)
        axes = np.atleast_1d(axes).ravel()
        for a_ in axes[len(have):]:
            a_.axis("off")
        sc = None
        for ax, name in zip(axes, have):
            d = store[(name, cname)]
            ax.imshow(bg, cmap="gray_r", extent=extent, vmin=0, vmax=1,
                      alpha=0.5, interpolation="bilinear", rasterized=True)
            ev = d["ev"]
            cen = d["cen"][ev]
            r = d["ratio"][ev]
            nf = d["frn_n"][ev]
            clean = nf == 0
            ax.scatter(cen[clean, 0], cen[clean, 1], c=CLEAN_BLUE, s=8,
                       linewidths=0.2, edgecolors="#00000025",
                       label="no foreign transcripts")
            dirty = ~clean
            if dirty.sum():
                vmax = max(np.nanpercentile(r[dirty], 95), 0.2)
                sc = ax.scatter(cen[dirty, 0], cen[dirty, 1], c=r[dirty],
                                cmap=CMAP_C, vmin=0, vmax=vmax, s=9,
                                linewidths=0.2, edgecolors="#00000030")
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{DISPLAY.get(name, name)}\nn={ev.sum()}  "
                         f"clean={int(clean.sum())} "
                         f"({100*clean.sum()/max(ev.sum(),1):.0f}%)", fontsize=10)
        if sc is not None:
            fig.colorbar(sc, ax=axes[len(have) - 1], shrink=0.8,
                         label="foreign / own transcripts")
        axes[0].legend(loc="lower left", fontsize=8, markerscale=1.6)
        fig.suptitle(f"Protein-defined {cname} cells: in-cell contamination "
                     f"(blue = none, gradient = foreign/own ratio)", fontsize=12)
        fp = os.path.join(OUTDIR, f"spatial_{cname.replace('/', '_')}.pdf")
        fig.savefig(fp, bbox_inches="tight", dpi=600)
        fig.savefig(fp.replace(".pdf", ".png"), bbox_inches="tight", dpi=400)
        plt.close(fig)
        log(f">>> {fp}")


OMICS_METHODS = ["proseg", "ucs", "genesegnet", "comseg", "cellist", "boms", "baysor", "bering", "dissect"]
IMAGE_METHODS = ["cellpose", "cellpose3", "cellsam", "mesmer", "stardist", "cellotype", "unseg"]


def plot_bubble(methods, store):
    """三张气泡矩阵：全部方法 / 9 种转录本方法 / 7 种图像方法。
    每张图内按列（外来谱系）重新归一化。某方法在某个 case 中细胞不足而没有结果时，
    该方法仍保留一行，对应格子画灰色短横线，保证所有方法都出现在图中。"""
    order = [m for m in OMICS_METHODS + IMAGE_METHODS if m in methods] + \
            [m for m in methods if m not in OMICS_METHODS + IMAGE_METHODS]
    missing = [DISPLAY.get(m, m) for m in OMICS_METHODS + IMAGE_METHODS if m not in methods]
    if missing:
        log(f"  [注意] 这些方法没有读取成功，无法进入气泡图：{missing}")
    subsets = [("all", "All methods", order),
               ("omics", "Transcript-based methods", [m for m in order if m in OMICS_METHODS]),
               ("image", "Image-based methods", [m for m in order if m in IMAGE_METHODS])]
    for tag, label, rows in subsets:
        if rows:
            _bubble_one(rows, store, label, os.path.join(OUTDIR, f"bubble_source_{tag}.pdf"))


def _bubble_one(rows, store, label, fp):
    case_list = [c for c in CASES if any((m, c) in store for m in rows)]
    blocks = []
    for cname in case_list:
        self_lin = CASES[cname][1]
        cols = [a for a in LINEAGES if a != self_lin]
        Mx = np.full((len(rows), len(cols)), np.nan)
        for i, m in enumerate(rows):
            if (m, cname) in store:
                Mx[i] = np.nan_to_num([store[(m, cname)]["share"][a] for a in cols], nan=0.0)
        blocks.append((cname, cols, Mx))
    if not blocks:
        return
    for cname, cols, Mx in blocks:
        gap = [DISPLAY.get(rows[i], rows[i]) for i in np.nonzero(np.isnan(Mx).all(1))[0]]
        if gap:
            log(f"  [{label}] {cname}: 无结果（细胞不足），画灰色短横线：{gap}")
    scores = []
    for cname, cols, Mx in blocks:
        S = np.full_like(Mx, np.nan)
        for j in range(Mx.shape[1]):
            col = Mx[:, j]
            ok = np.isfinite(col)
            if not ok.any():
                continue
            lo, hi = np.nanmin(col), np.nanmax(col)
            if hi - lo < 1e-12:
                S[ok, j] = SCORE_LO + SCORE_SPAN * 0.5
            else:
                S[ok, j] = SCORE_LO + SCORE_SPAN * (1.0 - (col[ok] - lo) / (hi - lo))
        scores.append(S)

    vmax_s = max(np.nanmax(s) for s in scores if np.isfinite(s).any())
    half_top = HALF_MIN + float(np.clip(vmax_s, 0, 1)) * (HALF_MAX - HALF_MIN)
    cell = 2 * (half_top + MIN_INDENT)
    col_x, xlab, span, cur = [], [], [], 0.0
    for bi, (cname, cols, _M) in enumerate(blocks):
        if bi:
            cur += cell * CAT_GAP
        s0 = cur
        for c in cols:
            col_x.append(cur); xlab.append(c); cur += cell
        span.append((cname, s0, cur - cell))
    grid_w, nR = cur, len(rows)
    grid_h = nR * cell
    rot = np.deg2rad(LABEL_ROT)
    bar_gap, bar_h = 0.06, 0.14
    cat_h = FS * 1.35 / 72.0
    y_bar = grid_h + bar_gap
    y_cat = y_bar + bar_h + 0.05
    y_lead = y_cat + cat_h + 0.06
    lab_len = max(text_w(a, FS) for a in xlab)
    head_h = y_lead + LEADER_IN * np.sin(rot) + lab_len * np.sin(rot) + 0.10 - grid_h
    left_in = max(text_w(DISPLAY.get(m, m), FS) for m in rows) + 0.22
    plot_w = grid_w + lab_len * np.cos(rot) * 0.6 + 0.3
    fig_w = left_in + plot_w + 0.3
    fig_h = 0.35 + grid_h + head_h + FS * 1.6 / 72.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    ax = fig.add_axes([left_in / fig_w, 0.35 / fig_h,
                       plot_w / fig_w, (grid_h + head_h) / fig_h])
    ax.set_xlim(0, plot_w); ax.set_ylim(0, grid_h + head_h)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(nR):
        ax.add_patch(Rectangle((0, (nR - 1 - i) * cell), grid_w, cell, zorder=0,
                               edgecolor="none",
                               facecolor=ROW_BG_A if i % 2 == 0 else ROW_BG_B))
    k = 0
    for bi, (cname, cols, Mx) in enumerate(blocks):
        S = scores[bi]
        for j in range(len(cols)):
            X = col_x[k] + cell / 2
            k += 1
            for i in range(nR):
                yc = (nR - 1 - i) * cell + cell / 2
                v = S[i, j]
                if not np.isfinite(v):                       # 该方法在此 case 无结果
                    ax.plot([X - cell * 0.18, X + cell * 0.18], [yc, yc], color="#9E9E9E",
                            lw=1.1, zorder=3)
                    continue
                half, nexp = shape_of(v)
                ax.add_patch(Polygon(
                    superellipse(X, yc, half, nexp),
                    closed=True, facecolor=shade(CASE_COLORS[cname], v),
                    edgecolor=EDGE_C, linewidth=EDGE_W, zorder=3))
    for i, m in enumerate(rows):
        ax.text(-0.10, (nR - 1 - i) * cell + cell / 2, DISPLAY.get(m, m),
                ha="right", va="center", fontsize=FS, color="#1A1A1A")
    for cname, a, b in span:
        ax.add_patch(Rectangle((a + 0.006, y_bar), (b + cell - a) - 0.012, bar_h,
                               facecolor=CASE_COLORS[cname], edgecolor="none",
                               zorder=2))
        ax.text((a + b + cell) / 2, y_cat, f"{cname} cells", ha="center",
                va="bottom", fontsize=FS, color="#1A1A1A")
    for j, a in enumerate(xlab):
        xc = col_x[j] + cell / 2
        xe = xc + LEADER_IN * np.cos(rot)
        ye = y_lead + LEADER_IN * np.sin(rot)
        ax.plot([xc, xe], [y_lead, ye], color="#6B7280", lw=0.7, zorder=2)
        ax.text(xe + 0.015 * np.cos(rot), ye + 0.015 * np.sin(rot), a,
                rotation=LABEL_ROT, rotation_mode="anchor", ha="left",
                va="center", fontsize=FS, color="#1A1A1A")
    fig.text(left_in / fig_w - 0.004, 1 - 0.06 / fig_h,
             f"Contamination source composition — {label}   "
             "(larger & squarer & lighter = smaller share)",
             ha="left", va="top", fontsize=FS, color="#1A1A1A")
    fig.savefig(fp, bbox_inches="tight", facecolor="white")
    fig.savefig(fp.replace(".pdf", ".png"), bbox_inches="tight",
                facecolor="white", dpi=400)
    plt.close(fig)
    log(f">>> {fp}")


def plot_joint(methods, store):
    ax_all = np.concatenate([d["env"][np.isfinite(d["env"])]
                             for d in store.values()])
    ay_all = np.concatenate([d["ratio"][d["ev"]][
        np.isfinite(d["ratio"][d["ev"]])] for d in store.values()])
    XFIT = float(np.percentile(ax_all, 99.0))
    YFIT = float(np.percentile(ay_all, 99.0))

    def ticks(vmax, n=5):
        raw = vmax / n
        mag = 10 ** np.floor(np.log10(max(raw, 1e-9)))
        for mult in (1, 2, 2.5, 5, 10):
            step = mult * mag
            if step >= raw:
                break
        return np.arange(0, vmax + step * 0.5, step)

    XT = ticks(float(np.percentile(ax_all, 98.0)))
    XMAX = float(XT[-1])
    YSCALE = {"T": 1.0, "Myeloid": 1.0 / 3.0}
    for cname in CASES:
        have = [m for m in methods if (m, cname) in store]
        if not have:
            continue
        YT = ticks(float(np.percentile(ay_all, 85.0)) * 3.0 * YSCALE.get(cname, 1.0))
        YMAX = float(YT[-1])
        nrow = int(np.ceil(len(have) / NCOL))
        fig, axes = plt.subplots(nrow, NCOL, figsize=(4.5 * NCOL, 4.4 * nrow),
                                 constrained_layout=True, sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel()
        for a_ in axes[len(have):]:
            a_.axis("off")
        sc, lines = None, []
        for ax, name in zip(axes, have):
            d = store[(name, cname)]
            x = d["env"]
            y = d["ratio"][d["ev"]]
            m = np.isfinite(x) & np.isfinite(y)
            x, y = x[m], y[m]
            ax.set_title(DISPLAY.get(name, name), fontsize=11)
            ax.grid(alpha=0.15)
            ax.set_xlim(0, XMAX); ax.set_ylim(0, YMAX)
            ax.set_xticks(XT); ax.set_yticks(YT)
            ax.set_box_aspect(1.0)
            kf = (x <= XFIT) & (y <= YFIT)
            forced = (name, cname) in FORCE_KEEP
            min_fit = MIN_FIT_FORCED if forced else 20
            if kf.sum() < min_fit:
                lines.append(f"{DISPLAY.get(name, name):<12s}  n/a")
                if forced and len(x):                       # 强制放行：点太少也照样画散点，不留空白
                    ax.scatter(x, y, c="#5B9BD5", s=12, linewidths=0, alpha=0.9)
                continue
            slope, itc = np.polyfit(x[kf], y[kf], 1)
            r = np.corrcoef(x[kf], y[kf])[0, 1]
            kv = (x <= XMAX) & (y <= YMAX)
            xs_, ys_ = x[kv], y[kv]
            nb = 60
            H, xe, ye = np.histogram2d(xs_, ys_, bins=nb,
                                       range=[[0, XMAX], [0, YMAX]])
            bx = np.clip(np.digitize(xs_, xe) - 1, 0, nb - 1)
            by = np.clip(np.digitize(ys_, ye) - 1, 0, nb - 1)
            dc = H[bx, by]
            o = np.argsort(dc)
            sc = ax.scatter(xs_[o], ys_[o], c=dc[o], cmap="Blues", s=7,
                            linewidths=0, alpha=0.85)
            xl = np.linspace(0, XMAX, 50)
            ax.plot(xl, slope * xl + itc, "--", color="#7F7F7F", lw=2.2,
                    zorder=4, clip_on=True)
            sg = "+" if itc >= 0 else "-"
            lines.append(f"{DISPLAY.get(name, name):<12s}  "
                         f"y = {slope:.3f}x {sg} {abs(itc):.2f}   "
                         f"r = {r:.2f}   n = {int(kf.sum())}")
        for i, ax in enumerate(axes[:len(have)]):
            if i % NCOL == 0:
                ax.set_ylabel("in-cell foreign / own")
            if i >= len(have) - NCOL:
                ax.set_xlabel(f"foreign density outside cell (0-{int(ENV_RING)} um)")
        if sc is not None:
            fig.colorbar(sc, ax=axes[len(have) - 1], shrink=0.85,
                         label="local density")
        fig.text(1.005, 0.5, "Linear fit\n" + "\n".join(lines),
                 transform=fig.transFigure, ha="left", va="center",
                 fontsize=9.5, family="monospace", color="#333333")
        fig.suptitle(f"{cname} cells: in-cell contamination vs surrounding "
                     f"environment", fontsize=12)
        fp = os.path.join(OUTDIR, f"joint_{cname.replace('/', '_')}.pdf")
        fig.savefig(fp, bbox_inches="tight", dpi=600)
        fig.savefig(fp.replace(".pdf", ".png"), bbox_inches="tight", dpi=400)
        plt.close(fig)
        log(f">>> {fp}")


def plot_cumulative(methods, store):
    for mode in ("boundary", "relative"):
        for cname in CASES:
            have = [m for m in methods if (m, cname) in store]
            if not have:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.2),
                                     constrained_layout=True)
            lines = []
            for name in have:
                d = store[(name, cname)]
                c = MCOL.get(name, "#333333")
                if mode == "boundary":
                    mid, co, cf, i = d["mid_b"], d["cum_b_own"], d["cum_b_frn"], d["i0"]
                    x0 = 0.0
                else:
                    mid, co, cf, i = d["mid_r"], d["cum_r_own"], d["cum_r_frn"], d["i1"]
                    x0 = 1.0
                axes[0].plot(mid, cf, "-o", ms=3, lw=1.9, color=c,
                             label=DISPLAY.get(name, name))
                axes[1].plot(mid, co, "-o", ms=3, lw=1.9, color=c,
                             label=DISPLAY.get(name, name))
                axes[0].plot([x0], [cf[i]], "o", ms=8, mfc="white", mec=c, mew=2,
                             zorder=5)
                axes[1].plot([x0], [co[i]], "o", ms=8, mfc="white", mec=c, mew=2,
                             zorder=5)
                lines.append(f"{DISPLAY.get(name, name):<12s}  "
                             f"own {co[i]:6.1f}   foreign {cf[i]:6.1f}   "
                             f"ratio {cf[i]/max(co[i],1e-9):5.2f}   "
                             f"R {d['R_med']:4.1f}um   n={d['n_cell']}")
            self_lin = CASES[cname][1]
            for ax, ttl in [(axes[0], "foreign-lineage transcripts"),
                            (axes[1], f"own ({self_lin}) transcripts")]:
                ax.axvline(0.0 if mode == "boundary" else 1.0, color="#333333",
                           ls="-", lw=1.4, alpha=0.7)
                if mode == "boundary":
                    ax.set_xlabel("signed distance to cell boundary (um)\n"
                                  "negative = inside,  positive = outside")
                    ax.set_xlim(BIN_B[0], BIN_B[-1])
                    ax.set_xticks(np.arange(-_IN_UM, _OUT_UM + 0.1, 2.0))
                else:
                    ax.set_xlabel("normalised distance  "
                                  "(0 = centre, 1 = boundary)")
                    ax.set_xlim(0, 1.0 + _REL_OUT)
                    ax.set_xticks(np.round(np.arange(0, 1.0 + _REL_OUT + 0.01,
                                                     0.25), 2))
                ax.set_ylabel("cumulative transcripts per cell")
                ax.set_title(ttl, fontsize=11)
                ax.grid(alpha=0.25)
                ax.set_ylim(bottom=0)
            # 图例与解析式信息横向并列：图例紧贴图右侧，解析式放在图例右边，两者垂直居中对齐
            h, l = axes[1].get_legend_handles_labels()
            leg = fig.legend(h, l, loc="center left", bbox_to_anchor=(1.0, 0.5),
                             frameon=False, fontsize=10)
            fig.canvas.draw()
            lb = leg.get_window_extent(renderer=fig.canvas.get_renderer()) \
                .transformed(fig.transFigure.inverted())
            fig.text(lb.x1 + 0.012, 0.5, "At boundary\n" + "\n".join(lines),
                     transform=fig.transFigure, ha="left", va="center",
                     fontsize=9, family="monospace", color="#333333")
            fig.suptitle(f"{cname} cells: cumulative transcripts "
                         f"({'origin = boundary' if mode=='boundary' else 'normalised distance'})",
                         fontsize=12)
            fp = os.path.join(OUTDIR, f"cum_{mode}_{cname.replace('/', '_')}.pdf")
            fig.savefig(fp, bbox_inches="tight", dpi=600)
            fig.savefig(fp.replace(".pdf", ".png"), bbox_inches="tight", dpi=400)
            plt.close(fig)
            log(f">>> {fp}")


def save_bubble_data(methods, store):
    """把气泡矩阵所需的数据（各方法、各 case 的外来转录本来源构成）存成 CSV，供独立的气泡图脚本使用。"""
    rows = []
    for (name, cname), d in store.items():
        for a in LINEAGES:
            if a == d["self_lin"]:
                continue
            rows.append(dict(method=name, display=DISPLAY.get(name, name), case=cname,
                             self_lineage=d["self_lin"], source_lineage=a,
                             share=d["share"][a], n_cell=d["n_cell"]))
    fp = os.path.join(OUTDIR, "bubble_source_data.csv")
    pd.DataFrame(rows).to_csv(fp, index=False)
    log(f">>> {fp}")


def main():
    methods, store = compute()
    if not store:
        raise SystemExit("没有任何方法产出结果，请先运行 python common.py --check")
    save_bubble_data(methods, store)
    log("\n== 绘图 ==")
    plot_spatial(methods, store)
    plot_bubble(methods, store)
    plot_joint(methods, store)
    plot_cumulative(methods, store)


if __name__ == "__main__":
    main()