#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xenium_8 (ccRCC): segmentation mask + transcript-point zoom-in comparison (T vs tumor)
=============================================================
Show ProSeg and Cellpose segmentations side by side on the same small ROI, making visible
how the boundary difference affects transcript purity.

Cell identity
  T cells      : CD3e protein-positive (independent of transcripts)
  tumor cells  : prefer PanCK protein; if too few positive cells, fall back to
                 transcript calls (Epi/Tumor marker dominant); logs and legend note
                 the method actually used

Drawing layers (bottom to top)
  1. DAPI nuclear background (grayscale)
  2. cell fill: T = light orange, tumor = light purple, other = light gray
     (low alpha, does not hide nuclei)
  3. cell boundary: contours extracted with skimage.find_contours, drawn at a fixed
                 width so both methods share identical line weight regardless of mask
                 resolution
  4. transcript dots: blue = lineage-specific genes of the cell's own identity (self)
                orange = lineage-specific genes of other lineages (foreign -> contamination)
                light gray = non-lineage-specific genes (housekeeping etc.), optional
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

# ================= tunable parameters =================
METHODS = ["proseg", "cellpose"]
ROI_UM = 40.0                  # search field side length (um)
N_CELL_TARGET = (4, 12)        # expected cell-count range in the field of view
MAX_UNASSIGNED_TRY = [0, 1, 2, 3, 99]   # allowed unassigned cell counts, relaxed step by step
SUB_FRAC = 0.75                # final field = 40 * 0.75 = 30 um
SUB_CORNER = "SW"              # SW=lower-left / SE=lower-right / NW=upper-left / NE=upper-right
SEARCH_TOP = 800
NEIGHBOR_UM = 12.0             # neighbourhood radius used to judge "surrounded"
MIN_TUMOR_AROUND = 2           # minimum tumor cells required in the field
SHOW_OTHER_TX = True           # whether to draw non-lineage transcripts (light gray)
PT_SIZE = 26
LINE_W = 1.2                   # cell contour line width (same for both methods)
FILL_ALPHA = 0.42              # mask fill alpha
BG_ALPHA = 1.0                 # DAPI background opacity
BG_GAMMA = 0.65                # <1 brightens shadows so nuclei stand out
# identity is decided purely by protein, with no transcript info -> fully orthogonal to contamination calls

CH_T = "ch0011_cd3e"        # T cells
CH_MYE = "ch0015_cd68"      # myeloid
CH_TUMOR = "ch0030_panck"   # epithelial/tumor
CH_CD45 = "ch0033_cd45"     # pan-leukocyte (used for "other immune" cells)
LIN_T = "T"
LIN_MYE = "Myeloid"
LIN_TUMOR = "Epi/Tumor"
LIN_IMM = "other immune"

# only identities decidable by protein (pure evidence); everything else is unassigned
# fixed color scheme (only these four): (68,144,169) (91,155,213) (255,192,0) (237,125,49)
# orange (237,125,49) is reserved for "foreign transcripts", not used for masks.
# cells without a protein identity have no color and use neutral light gray as a placeholder.
EDGE = {
    LIN_T:        "#4490A9",   # (68,144,169)  CD3e+ protein-confirmed
    LIN_MYE:      "#5B9BD5",   # (91,155,213)  CD68+ protein-confirmed
    LIN_TUMOR:    "#FFC000",   # (255,192,0)   PanCK+ or no protein identity (see below)
}
def _light(hexc, f=0.72):
    from matplotlib.colors import to_rgb
    r, g_, b = to_rgb(hexc)
    return (r + (1 - r) * f, g_ + (1 - g_) * f, b + (1 - b) * f)


def _dark(hexc, f=0.35):
    """Darken: used for self-transcript dots to stand out against the same-color translucent mask fill."""
    from matplotlib.colors import to_rgb
    r, g_, b = to_rgb(hexc)
    return (r * (1 - f), g_ * (1 - f), b * (1 - f))


FILL = {k: _light(v) for k, v in EDGE.items()}
DOT = {k: _dark(v) for k, v in EDGE.items()}      # self-transcript dot colors
# identity -> lineage used to judge "self/foreign" transcripts
IDENT2LIN = {LIN_T: "T", LIN_MYE: "Myeloid", LIN_TUMOR: "Epi/Tumor"}
TX_FOREIGN = "#ED7D31"        # (237,125,49) reserved for foreign transcripts
TX_OTHER = "#D9D9D9"          # non-lineage transcripts (gray)


def identity_protein(tab, posmap):
    """Identity decided purely by protein channels, with no transcript info.
    Priority: CD3e+ -> T;  CD68+ -> Myeloid;  PanCK+ -> Epi/Tumor;
              remaining CD45+ -> other immune;  otherwise -> unassigned.
    When multiple channels are positive, take the one with the highest positive-area fraction."""
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
    # Cells without a protein identity are all assigned to epithelial/tumor:
    #   this sample is ccRCC, so tumor cells dominate, while PanCK staining is weak
    #   (positive pixels only 1.26%); unassigned cells are mostly tumor cells with
    #   insufficient PanCK staining, so the false-negative cost is small.
    #   Note: this step introduces a prior assumption, not pure protein evidence, and
    #   must be treated separately when interpreting.
    confirmed = ident != "unassigned"        # whether confirmed by a protein channel
    n_fallback = int((~confirmed).sum())
    ident[ident == "unassigned"] = LIN_TUMOR
    from collections import Counter
    log("    identity distribution: " + str(dict(Counter(ident).most_common())))
    for k in keys:
        log(f"      {k:10s} protein-confirmed {n_prot[k]} cells")
    log(f"      {n_fallback} unassigned cells fall back to {LIN_TUMOR}")
    return ident, confirmed, ("CD3e/CD68/PanCK protein; cells without any "
                              "protein marker assigned to Epi/Tumor")


def pick_roi(tab, ident, gm, tix, tiy):
    """Pick an ROI: find a T cell (CD3e+) surrounded by epithelial/tumor cells (PanCK+).
    Score = neighborhood PanCK+ cell count (primary) + neighborhood tumor transcript
    density (secondary), and require a moderate cell count in the field."""
    t_idx = np.nonzero(ident == LIN_T)[0]
    tum_idx = np.nonzero(ident == LIN_TUMOR)[0]
    if len(t_idx) == 0:
        raise RuntimeError("no protein-confirmed T cells")
    if len(tum_idx) == 0:
        log("  Warning: no PanCK+ cells, falling back to tumor transcript density for ROI")
    cen_t = tab["cen_px"][t_idx]
    allc = tab["cen_px"]
    half = ROI_UM / 2 / PX
    r_near = NEIGHBOR_UM / PX

    # tumor cell count in the neighborhood
    if len(tum_idx):
        tree_c = cKDTree(tab["cen_px"][tum_idx])
        n_tum_cell = np.array([len(q) for q in
                               tree_c.query_ball_point(cen_t, r_near)])
    else:
        n_tum_cell = np.zeros(len(cen_t))
    # tumor transcript count in the neighborhood
    tree_x = cKDTree(np.c_[tix[gm["Epi/Tumor"]], tiy[gm["Epi/Tumor"]]])
    n_tum_tx = np.array([len(q) for q in
                         tree_x.query_ball_point(cen_t, r_near)])
    score = n_tum_cell * 1000 + n_tum_tx          # tumor cell count takes priority
    order = np.argsort(score)[::-1][:SEARCH_TOP]

    # tiered search: first require no unassigned cells in the field, then relax step by step
    for max_un in MAX_UNASSIGNED_TRY:
        for k in order:
            cx, cy = cen_t[k]
            inb = ((np.abs(allc[:, 0] - cx) < half) &
                   (np.abs(allc[:, 1] - cy) < half))
            n = int(inb.sum())
            if not (N_CELL_TARGET[0] <= n <= N_CELL_TARGET[1]):
                continue
            sub_id = ident[np.nonzero(inb)[0]]
            n_un = 0          # no unassigned left (all folded into Epi/Tumor)
            n_tum_in = int((sub_id == LIN_TUMOR).sum())
            if n_tum_in < MIN_TUMOR_AROUND:
                continue
            from collections import Counter
            log(f"  ROI center=({cx:.0f},{cy:.0f})px  "
                f"neighborhood tumor cells={int(n_tum_cell[k])}  "
                f"neighborhood tumor transcripts={int(n_tum_tx[k])}")
            log(f"    cells in view={n}  composition: {dict(Counter(sub_id).most_common())}")
            log(f"    (allowed unassigned upper bound this round = "
                f"{'unlimited' if max_un >= 99 else max_un}, actual {n_un})")
            return float(cx), float(cy)
        if max_un < 99:
            log(f"  no position meeting unassigned<={max_un}, relaxing and retrying")

    k = order[0]
    cx, cy = cen_t[k]
    log(f"  no fully matching candidate, using highest-scoring T cell: ({cx:.0f},{cy:.0f})  "
        f"neighborhood tumor cells={int(n_tum_cell[k])}")
    return float(cx), float(cy)


def draw_panel(ax, name, tab, ident, confirmed, bg_dapi, bg_ext,
               gm, tix, tiy, cx, cy, roi_um):
    """Layers: DAPI background -> gray non-lineage transcripts -> mask fill -> lineage-specific transcripts"""
    mask = tab["mask"]
    scale = tab["scale"]
    half = roi_um / 2 / PX
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half
    ext = [(x0 + X0) * PX, (x1 + X0) * PX, (y1 + Y0) * PX, (y0 + Y0) * PX]

    # --- L0: DAPI background ---
    ax.imshow(bg_dapi, extent=bg_ext, cmap="gray_r", vmin=0, vmax=1,
              alpha=BG_ALPHA, interpolation="bilinear", zorder=0)

    # transcripts in the field and their host/lineage
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
    # only protein-confirmed cells participate in contamination calls;
    # cells folded into Epi/Tumor by assumption get no orange dots; their transcripts
    # join the gray background layer
    host_conf = np.zeros(len(idx), bool)
    host_conf[ok] = confirmed[host[ok]]
    judgeable = (host_lin != "") & host_conf
    is_self = is_lin & judgeable & (lin_of == host_lin)
    is_frn = is_lin & judgeable & (lin_of != host_lin)

    # --- L1: gray non-lineage transcripts (beneath the mask) ---
    if SHOW_OTHER_TX:
        m = (~is_lin) | (is_lin & ~judgeable)   # non-lineage + those in identity-unconfirmed cells
        ax.scatter(px_[m], py_[m], s=PT_SIZE * 0.42, c=TX_OTHER, linewidths=0,
                   zorder=1, alpha=0.85)

    # --- L2 + L3: fill and stroke with the same contour polygons ---
    # fill and boundary share the same vertices, so they match exactly regardless of mask resolution
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
        log("    (no skimage, falling back to pixel fill)")

    if has_sk:
        for lb in labs:
            i = tab["idx_map"][lb]
            idt = ident[i] if i < len(ident) else LIN_TUMOR
            col = EDGE[idt if idt in EDGE else LIN_TUMOR]
            m = (sub == lb).astype(float)
            # zero-pad edges so border-touching cells still close
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

    # --- L4: lineage-specific transcripts (top layer) ---
    #   self signal: same color as host cell;  foreign signal: uniform orange
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
    ax.set_box_aspect(1.0)          # force a square plot box
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{DISPLAY.get(name, name)}   cells: {len(labs)}\n"
                 f"own {int(is_self.sum())}   foreign {int(is_frn.sum())}",
                 fontsize=11)
    return int(is_self.sum()), int(is_frn.sum())


def main():
    log("== protein-positive region ==")
    posmap = {LIN_T: protein_pos(CH_T),
              LIN_MYE: protein_pos(CH_MYE),
              LIN_TUMOR: protein_pos(CH_TUMOR)}

    log("\n== transcripts ==")
    g, tix, tiy, gm, expect = load_transcripts()

    log("\n== per-method cell table and identity ==")
    tabs, idents, confs = {}, {}, {}
    mode = ""
    for name in METHODS:
        log(f"  {name}")
        tab = cell_table(get_mask(name), tix, tiy, gm, {})
        ident, confirmed, mode = identity_protein(tab, posmap)
        tabs[name], idents[name] = tab, ident
        confs[name] = confirmed
        log(f"    cells={tab['nl']}")

    log("\n== selecting ROI (based on %s, shared by both plots) ==" % METHODS[0])
    cx, cy = pick_roi(tabs[METHODS[0]], idents[METHODS[0]], gm, tix, tiy)

    # take one corner of the selected ROI as the final field
    half_full = ROI_UM / 2 / PX
    roi_um = ROI_UM * SUB_FRAC
    half = roi_um / 2 / PX
    dxs = -1 if SUB_CORNER in ("SW", "NW") else 1      # left/right
    dys = 1 if SUB_CORNER in ("SW", "SE") else -1      # down/up (y grows downward)
    cx = cx + dxs * (half_full - half)
    cy = cy + dys * (half_full - half)
    # clamp the center so the whole field stays inside the image (otherwise the background is truncated, leaving whitespace)
    cx = float(np.clip(cx, half, SZ - half))
    cy = float(np.clip(cy, half, SZ - half))
    log(f"  take {SUB_CORNER} corner, field {roi_um:.0f} um, center=({cx:.0f},{cy:.0f})px")

    log("\n== loading DAPI background ==")
    dx0, dy0 = int(max(cx - half, 0)), int(max(cy - half, 0))
    dx1, dy1 = int(min(cx + half, SZ)), int(min(cy + half, SZ))
    dapi_full = read_ch("ch0000_dapi")
    d = dapi_full[dy0:dy1, dx0:dx1].astype(np.float32)
    lo, hi = np.percentile(d, [1, 99.8])
    bg = np.clip((d - lo) / max(hi - lo, 1), 0, 1) ** BG_GAMMA
    bg_ext = [(dx0 + X0) * PX, (dx1 + X0) * PX,
              (dy1 + Y0) * PX, (dy0 + Y0) * PX]

    log("\n== plotting ==")
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