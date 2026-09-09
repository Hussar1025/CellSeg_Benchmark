import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import os


def pick_font():
    candidates = ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
    avail = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in avail:
            return c
    for path in fm.findSystemFonts():
        if "yahei" in os.path.basename(path).lower():
            return fm.FontProperties(fname=path).get_name()
    return "DejaVu Sans"


FONT = pick_font()
print(f"字体: {FONT}")
plt.rcParams.update({
    "font.family":        FONT,
    "font.weight":        "bold",
    "axes.unicode_minus": False,
    "pdf.fonttype":       42,      # TrueType 嵌入，便于 Illustrator 编辑
    "ps.fonttype":        42,
})

# ── 数据：20 个数据集（真实统计）──────────────────────────────
DATA = [
    (1,  "Xenium",     "Human", "Breast Cancer",       "≤1 cell",    209467,   541,    64_581_006),
    (2,  "Xenium",     "Human", "Colon",               "≤1 cell",    219797,   541,    82_549_008),
    (3,  "Xenium",     "Human", "Liver",               "≤1 cell",    239271,   541,    25_424_111),
    (4,  "Xenium",     "Mouse", "Brain",               "≤1 cell",     63173, 13780,   188_540_362),
    (5,  "Xenium",     "Human", "Pancreatic Cancer",   "≤1 cell",    190965,   538,    28_455_659),
    (6,  "Xenium",     "Human", "Lymph Node",          "≤1 cell",      2442,  4509,       838_883),
    (7,  "CosMx",      "Human", "Frontal Cortex",      "≤1 cell",    188686,  6622,   163_339_286),
    (8,  "CosMx",      "Human", "Lymph Node",          "≤1 cell",   1852946,  6175, 1_413_412_378),
    (9,  "CosMx",      "Human", "Pancreas (WTx)",      "≤1 cell",     48944, 18946,    65_794_237),
    (10, "MERFISH",    "Human", "Liver (Cancer)",      "≤1 cell",    568355,   500,   272_021_991),
    (11, "MERFISH",    "Human", "Liver",               "≤1 cell",    598141,   500,   283_068_068),
    (12, "MERFISH",    "Human", "Lung Cancer",         "≤1 cell",    353762,   500,   144_388_044),
    (13, "MERFISH",    "Mouse", "Brain",               "≤1 cell",     78329,   483,    54_712_414),
    (14, "Stereo-seq", "Mouse", "Embryo",              "0.5 µm/bin", 1191859, 29082, 1_092_062_636),
    (15, "Stereo-seq", "Mouse", "Midbrain (E12.5)",    "0.5 µm/bin",    4271, 18087,     3_655_318),
    (16, "Stereo-seq", "Mouse", "Midbrain (E14.5)",    "0.5 µm/bin",    4872, 18698,     3_362_914),
    (17, "Stereo-seq", "Mouse", "Midbrain (E16.5 S3)", "0.5 µm/bin",    6892, 21202,     8_031_385),
    (18, "Stereo-seq", "Mouse", "Midbrain (E16.5 S6)", "0.5 µm/bin",    6229, 20654,     5_292_816),
    (19, "Stereo-seq", "Mouse", "Midbrain (E16.5 S7)", "0.5 µm/bin",    7960, 21370,     7_602_540),
    (20, "STARmap",    "Mouse", "Brain",               "≤1 cell",      1424,  1019,       471_295),
]

HEADERS = ["No.", "Platform", "Species", "# Datasets", "Resolution",
           "# Cells", "# Genes", "# Transcripts"]


# ── 平台技术分层：按基因 panel 规模切分 ───────────────────────
# 同一平台的不同 panel 在基因数上差一个数量级，混在一起求均值会误导，
# 所以先按基因数拆成技术层级。阈值可调，分组结果会打印出来核对。
def infer_tier(platform, n_genes):
    if platform == "Xenium":
        return "Xenium Prime 5K" if n_genes >= 3000 else "Xenium (500-plex)"
    if platform == "CosMx":
        return "CosMx WTx" if n_genes >= 15000 else "CosMx 6K"
    if platform == "MERFISH":
        return "MERFISH (500-plex)"
    if platform == "Stereo-seq":
        return "Stereo-seq (WT)"
    if platform == "STARmap":
        return "STARmap 1K"
    return platform


TIER_ORDER = [
    "Xenium (500-plex)", "Xenium Prime 5K",
    "CosMx 6K", "CosMx WTx",
    "MERFISH (500-plex)",
    "Stereo-seq (WT)",
    "STARmap 1K",
]

# 柱长代表的统计量。数值跨好几个数量级时，算术均值会被最大的数据集拉走，
# 几何均值更能代表"典型水平"。默认算术均值，需要时改成 "geometric"。
MEAN_MODE = "arithmetic"      # "arithmetic" / "geometric"


def fmt_num(n):
    return f"{int(round(n)):,}"


# 列宽（和为 1）
W = [0.040, 0.140, 0.078, 0.084, 0.082, 0.186, 0.180, 0.210]

BAR_AREA_FRAC = 0.54

# ── 三个统计列的固定颜色 ─────────────────────────────────────
# # Cells、# Genes、# Transcripts 分别使用蓝、橙、灰。
BAR_COLORS = {
    5: "#5B9BD5",
    6: "#ED7D31",
    7: "#A5A5A5",
}


# ── 分组与聚合 ────────────────────────────────────────────────
tier_of = {row[0]: infer_tier(row[1], row[6]) for row in DATA}
tiers = [t for t in TIER_ORDER if t in set(tier_of.values())]
for row in DATA:
    if tier_of[row[0]] not in tiers:
        tiers.append(tier_of[row[0]])

BAR_COLS = [5, 6, 7]
GROUPS = []
for i, t in enumerate(tiers, start=1):
    rows = [r for r in DATA if tier_of[r[0]] == t]
    sp = sorted({r[2] for r in rows})
    res = sorted({r[4] for r in rows})
    g = dict(
        no=i, tier=t,
        species="/".join(sp),
        n=len(rows),
        resolution=res[0] if len(res) == 1 else "mixed",
        vals={c: np.array([r[c] for r in rows], float) for c in BAR_COLS},
    )
    for c in BAR_COLS:
        v = g["vals"][c]
        g[c] = (float(np.exp(np.log(v).mean())) if MEAN_MODE == "geometric"
                else float(v.mean()))
    GROUPS.append(g)

print(f"\n分层聚合（柱 = {MEAN_MODE} mean，点 = 各数据集）：")
for g in GROUPS:
    print(f"  {g['tier']:<22} n={g['n']}  cells均值 {g[5]:>12,.0f}  "
          f"基因 {g['vals'][6].min():>6,.0f}~{g['vals'][6].max():<6,.0f}  "
          f"转录本均值 {g[7]:>15,.0f}")

# ── 归一化：柱与点共用同一套尺度 ──────────────────────────────
# lo/hi 取全部单个数据集的对数范围。任何均值都落在 min~max 之间，
# 所以柱和点都不会越界，两者位置可直接对照。
NORM_MODE = "stretch"
FLOOR     = 0.08

scale = {}
for c in BAR_COLS:
    allv = np.log10(np.array([r[c] for r in DATA], float))
    lo, hi = allv.min(), allv.max()
    scale[c] = (lo, hi)


def to_len(c, value):
    """原始值 -> 柱长/点位（0~1），柱与点共用。"""
    lo, hi = scale[c]
    lv = np.log10(np.asarray(value, float))
    if NORM_MODE == "ratio":
        return lv / hi
    n = (lv - lo) / (hi - lo) if hi > lo else np.ones_like(lv)
    return FLOOR + (1.0 - FLOOR) * n


# ── 点图叠加参数 ──────────────────────────────────────────────
SHOW_DOTS   = True
DOT_MS      = 8.5          # 点直径（磅）
DOT_EDGE_W  = 1.3          # 白色描边，压在柱上才看得清
DOT_DARKEN  = 0.62
DOT_JITTER  = 0.30         # 纵向错开幅度（占柱高的比例），0 = 全部落在中线
DOT_ALPHA   = 0.95

NR = len(GROUPS)
ROW_IN = 0.62              # 一行装一个平台，比原来高一些
HDR_IN = 0.60
PAD_IN = 0.10
FIG_W  = 20
FIG_H  = HDR_IN + NR * ROW_IN + 2 * PAD_IN
ROW_H = ROW_IN / FIG_H
HDR_H = HDR_IN / FIG_H
PAD_T = PAD_IN / FIG_H

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

cx = [0.0]
for w in W:
    cx.append(cx[-1] + w)
col_x = cx[:-1]

ROW_WHITE = "#FFFFFF"
ROW_GRAY  = "#F0F0F0"
HDR_FG    = "#111111"
TEXT_FG   = "#1A1A1A"

for ri in range(NR):
    y0 = 1.0 - PAD_T - HDR_H - ri * ROW_H
    bg = ROW_WHITE if ri % 2 == 0 else ROW_GRAY
    ax.add_patch(mpatches.Rectangle((0, y0 - ROW_H), 1, ROW_H,
                                    fc=bg, ec="none",
                                    transform=ax.transAxes, zorder=1))

ax.axhline(1.0 - PAD_T - HDR_H, color="#AAAAAA", lw=1.6, zorder=2)

yh = 1.0 - PAD_T - HDR_H / 2
for hdr, w, x0 in zip(HEADERS, W, col_x):
    ax.text(x0 + w / 2, yh, hdr,
            ha="center", va="center",
            fontsize=14, fontweight="bold",
            color=HDR_FG, transform=ax.transAxes, zorder=5)

BAR_PAD  = 0.003
BAR_HFRC = 0.40

for ri, g in enumerate(GROUPS):
    yc = 1.0 - PAD_T - HDR_H - ri * ROW_H - ROW_H / 2
    txt = [g["no"], g["tier"], g["species"], g["n"], g["resolution"]]

    for ci, (w, x0) in enumerate(zip(W, col_x)):
        if ci <= 4:
            ax.text(x0 + w / 2, yc, str(txt[ci]),
                    ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color=TEXT_FG, transform=ax.transAxes, zorder=5)
            continue

        mean_v = g[ci]
        t = float(to_len(ci, mean_v))
        bcolor = BAR_COLORS[ci]

        bar_zone_w = w * BAR_AREA_FRAC
        num_zone_r = x0 + w - 0.004
        bx = x0 + BAR_PAD
        bw = (bar_zone_w - BAR_PAD) * t
        bh = ROW_H * BAR_HFRC
        by = yc - bh / 2

        # 底槽统一白色，灰行上也能看清柱长
        ax.add_patch(mpatches.FancyBboxPatch(
            (bx, by), bar_zone_w - BAR_PAD, bh,
            boxstyle="round,pad=0.001",
            fc=ROW_WHITE, ec="none", zorder=3))

        # 柱体 = 该层级的均值
        if bw > 0.002:
            ax.add_patch(mpatches.FancyBboxPatch(
                (bx, by), bw, bh,
                boxstyle="round,pad=0.001",
                fc=bcolor, ec="none", zorder=4))

        # 点 = 层级内各数据集的实际值，与柱共用同一套尺度
        if SHOW_DOTS:
            vals = g["vals"][ci]
            ts = to_len(ci, vals)
            order = np.argsort(ts)
            m = len(vals)
            for rank, j in enumerate(order):
                # 纵向按次序错开，避免数值接近时圆点完全重叠
                off = 0.0 if m == 1 else (rank / (m - 1) - 0.5) * DOT_JITTER * bh * 2
                ax.plot([bx + (bar_zone_w - BAR_PAD) * ts[j]], [yc + off],
                        marker="o", markersize=DOT_MS,
                        markerfacecolor=BAR_COLORS[ci],
                        markeredgecolor="white", markeredgewidth=DOT_EDGE_W,
                        alpha=DOT_ALPHA, transform=ax.transAxes,
                        zorder=6, clip_on=False, linestyle="none")

        ax.text(num_zone_r, yc, fmt_num(mean_v),
                ha="right", va="center",
                fontsize=11, fontweight="bold",
                color=TEXT_FG, transform=ax.transAxes, zorder=7)

plt.savefig("dataset_stats_table.png", dpi=250, bbox_inches="tight",
            facecolor="white", pad_inches=0.06)
plt.savefig("dataset_stats_table.pdf", bbox_inches="tight",
            facecolor="white", pad_inches=0.06)
print("\n✓ dataset_stats_table.png / .pdf")
