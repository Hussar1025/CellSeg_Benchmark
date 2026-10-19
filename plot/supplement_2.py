# -*- coding: utf-8 -*-
"""
补充图 B —— 数据集之间"方法排名"的一致性
  a  转录本方法（9 种）：数据集 × 数据集矩阵，每格 = 两个数据集上各方法 F1 排名的 Spearman 相关
  b  图像方法（7 种）：同上
  c  平台内 vs 平台间：a、b 中所有数据集对的相关，按"同一平台 / 不同平台"分组的箱线图
行列按平台分组（Xenium -> STARmap -> Stereo-seq -> MERFISH -> CosMx，组内自然序），两侧为平台色条。
两个数据集共同评估的方法少于 MIN_COMMON 个时无法计算相关；这样的数据集直接从矩阵中去掉（不留灰格）：
每次去掉"无法计算的格子最多"的那个数据集，直到矩阵中没有空格为止，去掉的数据集打印在终端并写进 _data.xlsx。
读取与清洗沿用 robustness_bars.py（Fig. 2d），保证与正文口径一致：请把本脚本与 robustness_bars.py 放在同一目录。
本地运行：python supp_figB_rank_consistency.py
"""
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import supplement_4 as RB          # noqa: E402

# ======================== 参数 ========================
PATH_OMICS = RB.PATH_OMICS            # 默认与 Fig. 2d 相同：C:\Users\阿飞\Desktop\benchmark_result_omics.xlsx
PATH_IMAGE = RB.PATH_IMAGE
OUT_DIR = r"E:\supp_figures"
OUT_NAME = "supp_fig_rank_consistency"
METRIC = "f1"                         # 用于排名的指标（robustness_bars 读入的列：f1 / precision / recall）
MIN_COMMON = 4                        # 两个数据集至少有几个共同方法才计算相关
FIG_W, FONT = 7.2, 6
PLAT_ORDER = ["xenium", "starmap", "stereo", "merfish", "cosmx"]
PLAT_NAME = {"xenium": "Xenium", "starmap": "STARmap", "stereo": "Stereo-seq", "merfish": "MERFISH", "cosmx": "CosMx"}
PLAT_COLOR = {"xenium": "#8FCFC4", "starmap": "#F3DF7A", "stereo": "#E8837A", "merfish": "#8DB3D6", "cosmx": "#F2B26B"}
CMAP = LinearSegmentedColormap.from_list("rb", ["#2166AC", "#92C5DE", "#F7F7F7", "#F4A582", "#B2182B"])
NAN_C = "#D9D9D9"

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": FONT,
    "axes.linewidth": 0.5, "xtick.major.size": 0, "ytick.major.size": 0,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})


def nat_key(d):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", d)]


def corr_matrix(agg, kind):
    sub = agg[agg["_type"] == kind]
    P = sub.pivot_table(index="ds", columns="method", values=METRIC, aggfunc="mean")
    plat = sub.groupby("ds")["platform"].first()
    ds = sorted(P.index, key=lambda d: (PLAT_ORDER.index(plat[d]) if plat[d] in PLAT_ORDER else 99, nat_key(d)))
    P = P.loc[ds]
    n = len(ds)
    C = np.full((n, n), np.nan)
    ncom = np.zeros((n, n), int)
    for i in range(n):
        for j in range(n):
            ok = P.iloc[i].notna() & P.iloc[j].notna()
            ncom[i, j] = int(ok.sum())
            if i == j:
                C[i, j] = 1.0
            elif ok.sum() >= MIN_COMMON:
                C[i, j] = spearmanr(P.iloc[i][ok], P.iloc[j][ok])[0]
    return pd.DataFrame(C, index=ds, columns=ds), plat.loc[ds], P.shape[1]


def drop_incomplete(C, plat):
    """反复去掉空格最多的数据集，直到矩阵没有 NaN；返回 (C, plat, 去掉的数据集及原因)。"""
    dropped = []
    while True:
        nan_n = C.isna().sum(axis=1)
        if nan_n.max() == 0 or len(C) <= 2:
            break
        worst = nan_n.idxmax()
        dropped.append((worst, plat[worst], int(nan_n[worst]), len(C) - 1))
        C = C.drop(index=worst, columns=worst)
    return C, plat.loc[C.index], dropped


def pair_table(C, plat):
    rows = []
    ds = list(C.index)
    for i in range(len(ds)):
        for j in range(i + 1, len(ds)):
            v = C.iat[i, j]
            if np.isfinite(v):
                rows.append(dict(a=ds[i], b=ds[j], rho=v, same=plat[ds[i]] == plat[ds[j]]))
    return pd.DataFrame(rows)


def draw_heat(ax, C, plat, title, norm):
    n = len(C)
    M = np.ma.masked_invalid(C.to_numpy())
    cm = CMAP.copy(); cm.set_bad(NAN_C)
    im = ax.imshow(M, cmap=cm, norm=norm, interpolation="nearest")
    # 平台分组框
    p = list(plat)
    edges = [0] + [k for k in range(1, n) if p[k] != p[k - 1]] + [n]
    for a, b in zip(edges[:-1], edges[1:]):
        ax.add_patch(Rectangle((a - 0.5, a - 0.5), b - a, b - a, fill=False, ec="black", lw=0.8, zorder=3))
    # 平台色条（上、左）
    for k, d in enumerate(C.index):
        c = PLAT_COLOR.get(plat[d], "#BFBFBF")
        ax.add_patch(Rectangle((k - 0.5, -1.9), 1, 1.0, color=c, clip_on=False, lw=0))
        ax.add_patch(Rectangle((-1.9, k - 0.5), 1.0, 1, color=c, clip_on=False, lw=0))
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(range(n)); ax.set_xticklabels(C.columns, rotation=90, fontsize=FONT - 1)
    ax.set_yticks(range(n)); ax.set_yticklabels(C.index, fontsize=FONT - 1)
    ax.tick_params(axis="y", pad=12); ax.tick_params(axis="x", pad=2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=FONT + 1, pad=14)
    return im


DROPPED = {}


def main():
    agg, _gt = RB.load_both(PATH_OMICS, PATH_IMAGE)
    res = {}
    for kind, label in [("rna", "Transcript-based methods"), ("image", "Image-based methods")]:
        C, plat, nm = corr_matrix(agg, kind)
        C, plat, dropped = drop_incomplete(C, plat)
        if dropped:
            print(f"[{label}] 去掉 {len(dropped)} 个数据集（与其他数据集共同评估的方法少于 {MIN_COMMON} 个）：")
            for d, p, k, tot in dropped:
                print(f"      {d:<16} {PLAT_NAME.get(p, p):<10} 无法计算 {k}/{tot} 格")
        pt = pair_table(C, plat)
        res[kind] = (C, plat, nm, pt, label)
        DROPPED[kind] = dropped
        w, b = pt[pt.same].rho, pt[~pt.same].rho
        print(f"[{label}] 方法 {nm} 个，数据集 {len(C)} 个；数据集对：平台内 {len(w)} 对 中位 ρ = {w.median():.2f}，"
              f"平台间 {len(b)} 对 中位 ρ = {b.median():.2f}")
        for p in PLAT_ORDER:
            x = pt[pt.same & pt.a.map(plat).eq(p)].rho
            if len(x):
                print(f"      {PLAT_NAME[p]:<10} 平台内 {len(x):>3} 对 中位 ρ = {x.median():.2f}")

    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    nA, nB = len(res["rna"][0]), len(res["image"][0])
    L, GAP, R_ = 0.80, 0.95, 0.15                     # 左边距（含行名）、两热图间距、右边距（inch）
    cell = (FIG_W - L - GAP - R_) / (nA + nB)
    hA, hB = nA * cell, nB * cell
    row2 = 2.0                                         # 第二行（c + 图例 + 色标）高度
    fig_h = 0.45 + max(hA, hB) + 0.95 + row2 + 0.25
    fig = plt.figure(figsize=(FIG_W, fig_h))
    top = fig_h - 0.45
    axA = fig.add_axes([L / FIG_W, (top - hA) / fig_h, hA / FIG_W, hA / fig_h])
    axB = fig.add_axes([(L + hA + GAP) / FIG_W, (top - hB) / fig_h, hB / FIG_W, hB / fig_h])
    for ax, kind, lab in [(axA, "rna", "a"), (axB, "image", "b")]:
        C, plat, nm, pt, label = res[kind]
        im = draw_heat(ax, C, plat, f"{label} ({nm} methods, {len(C)} datasets)", norm)
        ax.text(-0.22, 1.06, lab, transform=ax.transAxes, fontsize=FONT + 3, fontweight="bold")
    y2 = top - max(hA, hB) - 0.95 - row2               # 第二行底边
    # c：平台内 vs 平台间
    axC = fig.add_axes([L / FIG_W, y2 / fig_h, 2.6 / FIG_W, row2 / fig_h])
    data, pos, cols = [], [], []
    for k, kind in enumerate(["rna", "image"]):
        pt = res[kind][3]
        for j_, same in enumerate([True, False]):
            data.append(pt[pt.same == same].rho.to_numpy()); pos.append(k * 2.6 + j_)
            cols.append("#4D4D4D" if same else "#BFBFBF")
    bp = axC.boxplot(data, positions=pos, widths=0.7, patch_artist=True, showfliers=False,
                     medianprops=dict(color="black", lw=0.9), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6),
                     boxprops=dict(lw=0.6))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.55)
    rng = np.random.default_rng(0)
    for d, p_ in zip(data, pos):
        axC.scatter(p_ + rng.uniform(-0.22, 0.22, len(d)), d, s=2.5, color="black", alpha=0.35, lw=0, zorder=3)
    for d, p_ in zip(data, pos):
        axC.text(p_, 1.08, f"{np.median(d):.2f}", ha="center", va="bottom", fontsize=FONT - 0.5)
    axC.axhline(0, color="#9E9E9E", lw=0.5, ls="--")
    axC.set_xticks([0.5, 3.1]); axC.set_xticklabels(["Transcript-based", "Image-based"])
    axC.set_ylabel("Spearman ρ between datasets"); axC.set_ylim(-1.05, 1.25); axC.set_yticks([-1, -0.5, 0, 0.5, 1])
    for sp in axC.spines.values():               # 全框线
        sp.set_visible(True); sp.set_linewidth(0.6); sp.set_color("black")
    axC.tick_params(axis="y", length=2)
    from matplotlib.patches import Patch
    axC.legend(handles=[Patch(fc="#4D4D4D", alpha=0.55, label="same platform"),
                        Patch(fc="#BFBFBF", alpha=0.55, label="different platforms")],
               loc="lower left", frameon=False, fontsize=FONT - 0.5)
    axC.text(-0.22, 1.02, "c", transform=axC.transAxes, fontsize=FONT + 3, fontweight="bold")
    # 色标与平台图例（第二行右侧）
    cx = L + 2.6 + 0.9
    cax = fig.add_axes([cx / FIG_W, (y2 + row2 - 0.25) / fig_h, 2.2 / FIG_W, 0.07 / fig_h])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False); cb.set_ticks([-1, -0.5, 0, 0.5, 1])
    cax.set_title(f"Spearman ρ of method ranks ({METRIC.upper()}) between two datasets", fontsize=FONT, pad=2)
    cax.tick_params(length=2, width=0.4)
    shown = [p for p in PLAT_ORDER if any((res[k][1] == p).any() for k in res)]
    fig.legend(handles=[Patch(fc=PLAT_COLOR[p], label=PLAT_NAME[p]) for p in shown],
               loc="upper left", bbox_to_anchor=(cx / FIG_W, (y2 + row2 - 0.55) / fig_h),
               frameon=False, fontsize=FONT, ncol=2, title="Platform", title_fontsize=FONT,
               alignment="left")

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext, kw in [(".pdf", {}), (".svg", {}), (".png", {"dpi": 600})]:
        p = os.path.join(OUT_DIR, OUT_NAME + ext)
        fig.savefig(p, bbox_inches="tight", facecolor="white", **kw)
        print("已保存:", p)
    plt.close(fig)
    # 数值导出，便于核对与写进附表
    with pd.ExcelWriter(os.path.join(OUT_DIR, OUT_NAME + "_data.xlsx")) as xw:
        for kind in ["rna", "image"]:
            C, plat, nm, pt, label = res[kind]
            C.round(3).to_excel(xw, sheet_name=f"{kind}_matrix")
            pt.assign(platform_a=pt.a.map(plat), platform_b=pt.b.map(plat)).round(3).to_excel(
                xw, sheet_name=f"{kind}_pairs", index=False)
            pd.DataFrame(DROPPED.get(kind, []), columns=["dataset", "platform", "pairs_not_computable",
                                                          "pairs_total"]).to_excel(
                xw, sheet_name=f"{kind}_dropped", index=False)
    print("已保存:", os.path.join(OUT_DIR, OUT_NAME + "_data.xlsx"))


if __name__ == "__main__":
    main()