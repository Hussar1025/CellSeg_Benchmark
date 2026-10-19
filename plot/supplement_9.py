# -*- coding: utf-8 -*-
"""
附图 —— 按数据类别分层的分割表现（各分层作为子图并排拼成一张图 supp_fig_stratified）
  平台（Xenium / MERFISH / CosMx / Stereo-seq / STARmap）、基因面板（靶向 ≤600 / 高通量 1,000–6,600 / 全转录组）、
  图像输入（DAPI / 核加膜 / H&E / ssDNA）、物种（人 / 小鼠）；要去掉某个分层，删掉 GROUPS 中对应的一行即可
每格为该方法在该类数据集上的 F1 中位数（各格的数据集数 n 写在 _data.csv 中）；只有一个数据集的格子数值后加 †；
没有数据集的格子留白并画一条细黑虚线对角线（实 2 pt、空 1 pt）；
某一列在该分层中全为空则删除该列；某一行在所有分层中都为空则删除该行（各子图共用同一组行，保证对齐）。
数据集的物种与图像输入已按附表 S1 校正（见 supp_common.py）。
本地运行：python supp_fig_stratified.py（与 supp_common.py、robustness_bars.py 放在同一目录）
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

import supplement_6 as C

OUT_NAME = "supp_fig_stratified"
FONT = 6
C.style_rc(mpl, FONT)
CMAP = LinearSegmentedColormap.from_list("f1", ["#FFF5EB", "#FDD0A2", "#FD8D3C", "#D94801", "#7F2704"])
GROUPS = [   # (文件后缀, 标题, 列名, 取值顺序, 显示名)
    ("platform", "Platform", "platform", ["xenium", "merfish", "cosmx", "stereo", "starmap"], C.PLAT_NAME),
    ("panel", "Gene panel", "panel", ["Targeted (≤600)", "High-plex (1,000–6,600)", "Whole transcriptome"], None),
    ("image", "Image input", "stain", ["DAPI", "Nuclear + membrane", "H&E", "ssDNA"], None),
    ("species", "Species", "species", ["Human", "Mouse"], None),
]
CELL_W, CELL_H = 0.40, 0.205      # 每格宽、高（inch）
GAP_IN = 0.28                      # 子图之间的间距（inch）
DIAG_LW = 0.4                      # 空白格对角线线宽（pt）
DIAG_DASH = (2.0, 1.0)             # 虚线：实 2 pt、空 1 pt


def blank_diag(ax, i, j):
    """空白格画一条细黑虚线对角线；matplotlib 默认按线宽缩放虚线，这里除以线宽，使实际长度为 2 pt / 1 pt。"""
    ax.plot([j - 0.5, j + 0.5], [i + 0.5, i - 0.5], color="black", lw=DIAG_LW, zorder=4,
            ls=(0, (DIAG_DASH[0] / DIAG_LW, DIAG_DASH[1] / DIAG_LW)), solid_capstyle="butt",
            dash_capstyle="butt")


def stats(d, key, vals):
    med = np.full((len(C.ORDER), len(vals)), np.nan)
    num = np.zeros_like(med, dtype=int)
    for i, m in enumerate(C.ORDER):
        g = d[d.method == m]
        for j, v in enumerate(vals):
            x = g[g[key] == v].f1.dropna()
            num[i, j] = len(x)
            if len(x):
                med[i, j] = x.median()
    return med, num


def main():
    acc = C.load_accuracy()
    meta = C.dataset_table(acc)
    d = acc.join(meta, on="ds")

    blocks, rows_any, csv = [], np.zeros(len(C.ORDER), bool), []
    for suffix, title, key, vals, names in GROUPS:
        labels = [(names or {}).get(v, v) for v in vals]
        med, num = stats(d, key, vals)
        for i, m in enumerate(C.ORDER):
            for j, lab in enumerate(labels):
                csv.append(dict(group=title, level=lab, method=m, n=int(num[i, j]),
                                median_f1=med[i, j]))
        keep_c = num.sum(axis=0) > 0
        if (~keep_c).any():
            print(f"[{title}] 删除的空列:", [l for l, k in zip(labels, keep_c) if not k])
        rows_any |= num[:, keep_c].sum(axis=1) > 0
        blocks.append((title, [l for l, k in zip(labels, keep_c) if k], med[:, keep_c], num[:, keep_c]))
    if (~rows_any).any():
        print("删除的空行:", [m for m, k in zip(C.ORDER, rows_any) if not k])
    methods = [m for m, k in zip(C.ORDER, rows_any) if k]
    blocks = [(t, l, me[rows_any], nu[rows_any]) for t, l, me, nu in blocks if len(l)]

    L, B, Rm, Tm = 1.0, 1.05, 1.6, 0.45
    grid_w = sum(len(l) for _, l, _, _ in blocks) * CELL_W + GAP_IN * (len(blocks) - 1)
    fig_w, fig_h = L + grid_w + Rm, B + len(methods) * CELL_H + Tm
    fig = plt.figure(figsize=(fig_w, fig_h))
    norm = Normalize(0, 1)
    cm = CMAP.copy(); cm.set_bad("white")
    x = L
    axes = []
    for k, (title, labels, med, num) in enumerate(blocks):
        w = len(labels) * CELL_W
        ax = fig.add_axes([x / fig_w, B / fig_h, w / fig_w, len(methods) * CELL_H / fig_h])
        axes.append(ax)
        ax.imshow(np.ma.masked_invalid(med), cmap=cm, norm=norm, aspect="auto")
        for i in range(med.shape[0]):
            for j in range(med.shape[1]):
                if np.isfinite(med[i, j]):
                    ax.text(j, i, f"{med[i, j]:.2f}" + ("$^{\\dagger}$" if num[i, j] == 1 else ""),
                            ha="center", va="center", fontsize=FONT - 1.2,
                            color="white" if med[i, j] > 0.6 else "black")
                else:
                    blank_diag(ax, i, j)
        ax.set_xlim(-0.5, len(labels) - 0.5); ax.set_ylim(len(methods) - 0.5, -0.5)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=FONT - 0.5)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods if k == 0 else [])
        ax.tick_params(length=0)
        ax.set_title(title, fontsize=FONT + 0.5, pad=4)
        n_tx = sum(C.CAT_OF[m] != "Image-based" for m in methods)
        if 0 < n_tx < len(methods):
            ax.axhline(n_tx - 0.5, color="black", lw=0.6)
        C.full_frame(ax)
        x += w + GAP_IN
    C.category_strip(fig, axes[0], methods)
    x_r = (L + grid_w + 0.18) / fig_w
    cax = fig.add_axes([x_r, B / fig_h, 0.08 / fig_w, len(methods) * CELL_H * 0.40 / fig_h])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
    cb.set_label("Median F1", fontsize=FONT); cb.outline.set_linewidth(0.4); cax.tick_params(labelsize=FONT - 1)
    fig.legend(handles=[Line2D([], [], ls="", marker="s", ms=5, mfc=v, mec="none", label=k) for k, v in C.CAT_COLOR.items()],
               loc="upper left", bbox_to_anchor=(x_r - 0.01, (B + len(methods) * CELL_H) / fig_h), frameon=False,
               fontsize=FONT, title="Category", title_fontsize=FONT)
    C.save(fig, OUT_NAME)
    plt.close(fig)
    pd.DataFrame(csv).round(4).to_csv(f"{C.OUT_DIR}/{OUT_NAME}_data.csv", index=False)
    print("已保存:", f"{C.OUT_DIR}/{OUT_NAME}_data.csv")


if __name__ == "__main__":
    main()