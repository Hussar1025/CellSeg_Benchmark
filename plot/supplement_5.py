# -*- coding: utf-8 -*-
"""
Supplementary figure: running-status matrix (31 datasets × 16 methods = 496 tasks).
Colour = platform (five colours: Xenium light green; the other four platforms use the four agreed
colours); cells of the same platform share one colour.
Shape = task status (determined from three local tables).
Decision order (first match wins):
  – light-grey short dash  Not applicable: 7 image methods × 4 Stereo-seq datasets and 9 transcript/RNA
                methods × STARmap datasets with no result-table record (this rule overrides everything
                and ignores the resource table)
  ◩ half-filled square  Completed on a subregion: only the tasks listed in SUBREGION_TASKS
  ■ solid square        Completed and evaluated: record in the result table; other cells of methods in
                FORCE_SUCCESS (GeneSegNet) also count as success
  □ hollow square       Completed but not evaluated: run in the resource table, no result-table record
  × dark-grey cross     Failed: none of the above
Row = dataset (grouped by platform, left platform color bar); column = method (method name in black).
Top bars = completed-task count per method (stacked by platform, same colour); right bars = completed-method
count per dataset (platform colour); the number beside each bar = completed / applicable.
Loading and name normalization follow robustness_bars.py (same as Fig. 2d); keep this script in the same
directory as robustness_bars.py.
Local run: python supp_fig_run_status.py
"""
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import supplement_4 as RB          # noqa: E402

# ======================== Parameters ========================
PATH_OMICS = RB.PATH_OMICS
PATH_IMAGE = RB.PATH_IMAGE
PATH_RES = "yourpath/segmentation_resources_all_methods_summary.csv"
OUT_DIR = "yourpath/supp_figures"
OUT_NAME = "supp_fig_run_status"
FIG_W, FONT = 7.2, 6
CELL = 0.21                           # cell side length (inch)
GLYPH = 0.34                          # square half-side (cell = 1); smaller = larger gap between cells
EDGE_LW = 0.15                        # black square stroke width (pt)
FRAME_LW = 0.4                        # frame and platform divider line width (pt)
SHOW_LEGEND = False                   # matrix only: legend off by default (status/colour meanings go in the caption)
# Only these run on a subregion (other reference-cell mismatches are GT-definition differences, treated as full runs):
# ComSeg on xenium_1 (central 6000 px ROI), cosmx_1, cosmx_2; and Cellist on xenium_6
SUBREGION_TASKS = {("ComSeg", "xenium_1"), ("ComSeg", "cosmx_1"), ("ComSeg", "cosmx_2"), ("Cellist", "xenium_6")}
# Mark every cell of these methods as success except "not applicable" ones (regardless of resource/result-table records)
FORCE_SUCCESS = {"GeneSegNet"}

IMAGE_METHODS = ["Cellpose-SAM", "Cellpose-3", "CellSAM", "CelloType", "Mesmer", "StarDist", "UNSEG"]
OMICS_METHODS = ["ProSeg", "Baysor", "BOMS", "Bering", "ComSeg", "UCS", "Cellist", "GeneSegNet", "DISSECT"]
ORDER = OMICS_METHODS + IMAGE_METHODS
NA_RULES = [(IMAGE_METHODS, "stereo"), (OMICS_METHODS, "starmap")]     # (method set, platform prefix) -> not applicable

STATUS = [   # (key, legend text)
    ("eval", "Completed"),
    ("sub", "Completed on a sub-region"),
    ("noeval", "Completed, not evaluated"),
    ("fail", "Failed"),
]
PLAT_COLOR = {               # the five platform colours shared with the resource matrix (fig_resources_matrix.py)
    "xenium": "#8FCFC4",
    "starmap": "#F3DF7A",
    "stereo": "#E8837A",
    "merfish": "#8DB3D6",
    "cosmx": "#F2B26B",
}
FAIL_C = "#595959"
NA_C = "#BDBDBD"
PLAT_ORDER = ["xenium", "starmap", "stereo", "merfish", "cosmx"]
PLAT_NAME = {"xenium": "Xenium", "starmap": "STARmap", "stereo": "Stereo-seq", "merfish": "MERFISH", "cosmx": "CosMx"}

# resource-table name -> normalized name (matching the result table, lowercased)
RES_METHOD = {"proseg": "ProSeg", "cellposesam": "Cellpose-SAM", "cellpose-sam": "Cellpose-SAM",
              "cellpose3-cyto3": "Cellpose-3", "cellpose3": "Cellpose-3", "cellpose-3": "Cellpose-3",
              "baysor": "Baysor", "bering": "Bering", "boms": "BOMS", "comseg": "ComSeg", "dissect": "DISSECT",
              "genesegnet": "GeneSegNet", "ucs": "UCS", "cellist": "Cellist", "cellotype": "CelloType",
              "cellsam": "CellSAM", "mesmer": "Mesmer", "stardist": "StarDist", "unseg": "UNSEG"}
RES_DATASET = {
    "cosmx_pancreas": "cosmx_1", "pancreas-cosmx-wtx-flatfiles": "cosmx_1", "cosmx_lymph_node": "cosmx_2",
    "cosmx_human_brain": "cosmx_3", "merfish_mouse_brain": "merfish_1", "merfish_liver1": "merfish_2",
    "merfish_liver2": "merfish_3", "merfish_lung": "merfish_4", "merfish_lung_cancer": "merfish_4",
    "merfish_breast_cancer": "merfish_breast", "merfish_uterine": "merfish_uterian",
    "stereo_e14.5_e1s3": "stereo_1", "stereo_e16.5_e1s3": "stereo_2", "stereo_e16.5_e2s6": "stereo_3",
    "stereo_e16.5_e2s7": "stereo_3", "stereo_mouse_brain": "stereo_4",
    "xenium_pancreas_cancer": "xenium_1", "xenium_breast": "xenium_2", "xenium_roi_crop": "xenium_3",
    "xenium_liver": "xenium_4", "xenium_mouse_brain": "xenium_5", "xenium_colon": "xenium_6",
}

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": FONT,
    "axes.linewidth": 0.6, "xtick.major.size": 0, "ytick.major.size": 0,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})


def nat_key(d):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", d)]


def norm_res_ds(s):
    k = str(s).strip().split()[0].lower()
    k = RES_DATASET.get(k, k)
    m = re.fullmatch(r"xenium(\d+)", k)
    return f"xenium_{m.group(1)}" if m else k


def load_status():
    agg, gt = RB.load_both(PATH_OMICS, PATH_IMAGE)
    ev = agg.groupby(["method", "ds", "_type"], as_index=False)["_gt"].mean().rename(columns={"_gt": "gtc"})
    # standard reference cell count per dataset (mode), only for printing the subregion-task comparison
    ref = ev.groupby(["ds", "_type"])["gtc"].agg(lambda v: v.round().mode().iloc[0])
    ev["ref"] = [ref[(d, t)] for d, t in zip(ev["ds"], ev["_type"])]
    res = pd.read_csv(PATH_RES, encoding="utf-8-sig").iloc[:, :2]
    res.columns = ["method", "dataset"]
    res["method"] = res["method"].astype(str).str.strip().map(lambda m: RES_METHOD.get(m.lower(), m))
    res["ds"] = res["dataset"].map(norm_res_ds)
    datasets = sorted(set(ev["ds"]) | set(res["ds"]),
                      key=lambda d: (PLAT_ORDER.index(d.split("_")[0]) if d.split("_")[0] in PLAT_ORDER else 99,
                                     nat_key(d)))
    unknown_m = sorted((set(ev["method"]) | set(res["method"])) - set(ORDER))
    if unknown_m:
        print("[Warning] method names not in ORDER (not plotted):", unknown_m)
    ran = set(zip(res["method"], res["ds"]))
    evaluated = set(zip(ev["method"], ev["ds"]))
    S = pd.DataFrame(index=datasets, columns=ORDER, dtype=object)
    for d in datasets:
        plat = d.split("_")[0]
        for m in ORDER:
            na_cell = any(m in ms and plat == p for ms, p in NA_RULES)
            if na_cell and (m, d) not in evaluated:
                S.at[d, m] = "na"                      # rule takes priority over everything
            elif (m, d) in SUBREGION_TASKS:
                S.at[d, m] = "sub"
            elif (m, d) in evaluated or m in FORCE_SUCCESS:
                S.at[d, m] = "eval"
            elif (m, d) in ran:
                S.at[d, m] = "noeval"
            else:
                S.at[d, m] = "fail"
    miss_sub = [t for t in SUBREGION_TASKS if t[1] in S.index and t[0] in S.columns and S.at[t[1], t[0]] != "sub"]
    if miss_sub:
        print("[Warning] these SUBREGION_TASKS were not marked as subregion:", miss_sub)
    sub_info = ev[[(m, d) in SUBREGION_TASKS for m, d in zip(ev["method"], ev["ds"])]]
    return S, sub_info


def draw_glyph(ax, x, y, st, pc, h=None):
    """Draw the glyph for one task at (x, y); pc = platform colour."""
    from matplotlib.patches import Polygon
    h = GLYPH if h is None else h
    if st == "na":
        ax.plot([x - 0.22, x + 0.22], [y, y], color=NA_C, lw=0.9, solid_capstyle="round")
    elif st == "eval":
        ax.add_patch(Rectangle((x - h, y - h), 2 * h, 2 * h, facecolor=pc, edgecolor="black", lw=EDGE_LW))
    elif st == "sub":                                   # lower-left half filled
        ax.add_patch(Rectangle((x - h, y - h), 2 * h, 2 * h, facecolor="white", edgecolor="black", lw=EDGE_LW))
        ax.add_patch(Polygon([(x - h, y - h), (x - h, y + h), (x + h, y + h)], closed=True, facecolor=pc,
                             edgecolor="none"))
        ax.add_patch(Rectangle((x - h, y - h), 2 * h, 2 * h, facecolor="none", edgecolor="black", lw=EDGE_LW))
    elif st == "noeval":                                # hollow: thick platform-colour stroke
        ax.add_patch(Rectangle((x - h * 0.86, y - h * 0.86), 2 * h * 0.86, 2 * h * 0.86, facecolor="white",
                               edgecolor=pc, lw=0.8))
    elif st == "fail":
        d_ = h * 0.62
        ax.plot([x - d_, x + d_], [y - d_, y + d_], color=FAIL_C, lw=0.7, solid_capstyle="round")
        ax.plot([x - d_, x + d_], [y + d_, y - d_], color=FAIL_C, lw=0.7, solid_capstyle="round")


def main():
    S, subs = load_status()
    nD, nM = S.shape
    keys = [k for k, _ in STATUS]
    cnt = {k: int((S == k).sum().sum()) for k in keys + ["na"]}
    applicable = nD * nM - cnt["na"]
    done = cnt["eval"] + cnt["sub"] + cnt["noeval"]
    print(f"matrix {nD} datasets × {nM} methods = {nD * nM} tasks; not applicable {cnt['na']}, applicable {applicable}")
    for k, lab in STATUS:
        print(f"  {lab:<62} {cnt[k]:>4}")
    print(f"  completed (eval+sub+noeval) {done} / {applicable} applicable tasks")
    if len(subs):
        print("  tasks run on a subregion:")
        for r in subs.itertuples():
            print(f"      {r.method:<12} {r.ds:<16} reference cells {r.gtc:,.0f} (dataset standard {r.ref:,.0f})")

    # ---------- Layout (inch): matrix only ----------
    L, T_, R_, B_ = 1.05, 0.75, 0.1, 0.1
    grid_w, grid_h = nM * CELL, nD * CELL
    fig_w, fig_h = L + grid_w + R_, T_ + grid_h + B_
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([L / fig_w, B_ / fig_h, grid_w / fig_w, grid_h / fig_h])
    for i, d in enumerate(S.index):
        pc = PLAT_COLOR.get(d.split("_")[0], "#9E9E9E")
        for j, m in enumerate(S.columns):
            draw_glyph(ax, j, i, S.at[d, m], pc)
    ax.set_xlim(-0.5, nM - 0.5); ax.set_ylim(nD - 0.5, -0.5)
    ax.set_xticks(range(nM)); ax.set_xticklabels(S.columns, rotation=90, color="black")
    ax.xaxis.tick_top(); ax.tick_params(axis="x", pad=2)
    ax.set_yticks(range(nD)); ax.set_yticklabels(S.index, color="black")
    ax.tick_params(axis="y", pad=2)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(FRAME_LW)
    plats = [d.split("_")[0] for d in S.index]
    for k in range(1, nD):                                  # divider line between platforms
        if plats[k] != plats[k - 1]:
            ax.axhline(k - 0.5, color="black", lw=FRAME_LW)
    ax.axvline(len(OMICS_METHODS) - 0.5, color="black", lw=FRAME_LW)   # between transcript/RNA and image methods

    if SHOW_LEGEND:
        from matplotlib.patches import Patch as _P
        fig.legend(handles=[_P(facecolor=PLAT_COLOR[p], edgecolor="black", lw=EDGE_LW, label=PLAT_NAME[p])
                            for p in PLAT_ORDER if p in plats], loc="upper left",
                   bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=FONT - 0.3)

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext, kw in [(".pdf", {}), (".svg", {}), (".png", {"dpi": 600})]:
        p = os.path.join(OUT_DIR, OUT_NAME + ext)
        fig.savefig(p, bbox_inches="tight", facecolor="white", **kw)
        print("Saved:", p)
    plt.close(fig)
    lab = {k: l for k, l in STATUS}; lab["na"] = "Not applicable"
    S.replace(lab).to_excel(os.path.join(OUT_DIR, OUT_NAME + "_status.xlsx"))
    print("Saved:", os.path.join(OUT_DIR, OUT_NAME + "_status.xlsx"))


if __name__ == "__main__":
    main()