#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DISSECT x stereo_2 (Stereo-seq E14.5_E1S3, GT 4872) —— inspect / register / prep / run / merge

默认 --mode native：走 DISSECT 原生 stereoseq 入口（dissect_native_runner.py），
  ssDNA 在 DNB 网格上采样（1 px = 1 DNB = 0.5 um），GEM 平移到图像原点、保留 MIDCount 权重，不展开；
  源码中 gene_dense 用赋值而非累加，展开出的重复分子会互相覆盖，因此原生入口才能正确保留计数。
--mode xenium：下面描述的伪装 xenium 流程，仅作对照。
dissect_native_runner.py、wsi_register.py 需与本脚本放在同一目录。

与 dissect_merfish_*.py / dissect_cosmx_*.py 同一思路，把 Stereo-seq 伪装成 xenium：
  register : GEM 里 CellBin 细胞的中心（DNB 坐标）自动配准到 ssDNA 图
             （方向 / 缩放 / 偏移；E14.5、E2S7 的 GEM 坐标落在图像范围外，必须先配准）
  prep     : 在 GEM 包围盒上按 0.2125 um/px 建网格，从 ssDNA 图采样出与 GEM 同向的图像；
             每个 bin 的 MIDCount 展开成等量分子，坐标取 bin 中心，写 xenium schema parquet
             （与 BOMS × Stereo-seq 的处理一致；DISSECT 的梯度场按分子计数，展开是近似）
  run      : 调已验证的官方 wrapper run_xenium_pancreas_dissect.py
  merge    : 标签图 -> 细胞表（DNB / um 坐标）+ 每个 GEM bin 的细胞归属 + QC
wsi_register.py 需与本脚本放在同一目录。

用法:
  python dissect_stereo_2.py --stage inspect
  python dissect_stereo_2.py --stage register            # 看 QC 图 register_qc.png
  python dissect_stereo_2.py --stage all --n-jobs 4
"""
import argparse, glob, gzip, json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, tifffile
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ===== 数据集写死常量 =====
TAG       = "stereo_2"
SAMPLE    = "E14.5_E1S3"
DATA_DIR  = Path("/data/qiuyijia/dataset/stereoseq/stereo-seq_data_all")
GEM       = DATA_DIR / f"{SAMPLE}_Dorsal_Midbrain_GEM_CellBin.tsv.gz"
IMAGE     = DATA_DIR / f"{SAMPLE}.tif"
OUT       = Path("/data/qiuyijia/dissect_stereo_2_" + SAMPLE)
GT_CELLS  = 4872
BIN_UM    = 0.5               # 1 DNB = 0.5 um
TARGET_UM = 0.2125            # DISSECT/xenium detector 训练尺度
MARGIN_DNB = 20               # 网格在 GEM 包围盒外多留的 DNB
ENGINE    = Path("/data/qiuyijia/DISSECT/code/run_xenium_pancreas_dissect.py")

P = lambda *a: print(*a, flush=True)
SEP = "=" * 78


def read_gem():
    skip = 0
    with gzip.open(GEM, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                skip += 1
            else:
                break
    g = pd.read_csv(GEM, sep="\t", skiprows=skip)
    ren = {}
    for c in g.columns:
        cl = c.lower()
        if cl in ("geneid", "gene"):
            ren[c] = "gene"
        elif cl in ("midcounts", "midcount", "umicount", "count"):
            ren[c] = "count"
        elif cl in ("cell", "cellid", "cell_id", "label"):
            ren[c] = "cell"
    g = g.rename(columns=ren)
    if "count" not in g.columns:
        g["count"] = 1
    return g


def cell_centers(g):
    c = g[g.cell != 0].groupby("cell")[["x", "y"]].mean()
    return c.to_numpy(float) + 0.5          # DNB 连续坐标（bin k 覆盖 [k, k+1)）


def grid_origin(g):
    return float(g.x.min() - MARGIN_DNB), float(g.y.min() - MARGIN_DNB), \
        float(g.x.max() + 1 + MARGIN_DNB), float(g.y.max() + 1 + MARGIN_DNB)



def area_limits(a, um_per_px):
    """检测框面积阈值（px²）。默认与 benchmark 其他 DISSECT 运行相同的数值 350–1600；
    --area-rule physical 时按物理尺寸与 Xenium（0.2125 um/px）等价换算。--minarea/--maxarea 显式给出时优先。"""
    f = (0.2125 / um_per_px) ** 2 if getattr(a, "area_rule", "fixed") == "physical" else 1.0
    lo = a.minarea if a.minarea is not None else int(round(350 * f))
    hi = a.maxarea if a.maxarea is not None else int(round(1600 * f))
    return lo, hi


def report_box_coverage(areas_px, lo, hi, tag="", um_per_px=None):
    areas_px = np.asarray(areas_px, float)
    areas_px = areas_px[np.isfinite(areas_px) & (areas_px > 0)]
    if not len(areas_px):
        return
    q = np.percentile(areas_px, [5, 25, 50, 75, 95])
    P(f"  参考细胞外接框面积{tag}（px²）: 5%={q[0]:.0f} 25%={q[1]:.0f} 中位={q[2]:.0f} 75%={q[3]:.0f} 95%={q[4]:.0f}")
    cands = [("当前设置", lo, hi), ("数值默认 350–1600", 350, 1600), ("库函数默认 600–1200", 600, 1200)]
    if um_per_px:
        f = (0.2125 / um_per_px) ** 2
        cands.append(("物理等价 Xenium", int(round(350 * f)), int(round(1600 * f))))
    cands.append(("参考 5–95% 分位", int(q[0]), int(q[4])))
    for name, l, h in cands:
        inside = float(((areas_px >= l) & (areas_px <= h)).mean())
        P(f"    {name:<16} [{l:>6},{h:>6}]  覆盖 {inside:6.1%}")
    inside = float(((areas_px >= lo) & (areas_px <= hi)).mean())
    if inside < 0.5:
        P("  [警告] 当前设置下多数参考细胞的外接框不在检测面积范围内；考虑调整 --minarea/--maxarea")


# ---------------------------------------------------------------- 原生 stereoseq 入口
RUNNER = Path(__file__).resolve().parent / "dissect_native_runner.py"


def dirs(a):
    if a.mode == "native":
        return OUT / "input_native", OUT / "dissect_out_native", "_native"
    return OUT / "input", OUT / "dissect_out", ""


def stage_prep_native(a):
    """DNB 网格（1 px = 1 DNB = 0.5 um）上的对齐 ssDNA + 平移后的 GEM（geneID/x/y/MIDCount）。
    DISSECT stereoseq 分支直接把 x、y 当像素下标，MIDCount 作为权重写入 gene_dense，不展开。"""
    import wsi_register as wr
    P(f"\n{SEP}\nPREP (native stereoseq)  {TAG} {SAMPLE}\n{SEP}")
    inp, _, _ = dirs(a); inp.mkdir(parents=True, exist_ok=True)
    img_f, gem_f, meta_f = inp / "ssdna_dnb.tif", inp / "gem_native.tsv.gz", inp / "meta.json"
    if img_f.exists() and gem_f.exists() and meta_f.exists() and not a.force_prep:
        P("  复用 prep 缓存"); return
    if not (OUT / "register.json").exists():
        raise SystemExit("缺 register.json：请先运行 --stage register")
    T = wr.load(OUT / "register.json")
    g = read_gem()
    x0, y0, x1, y1 = grid_origin(g)
    ow, oh = int(x1 - x0), int(y1 - y0)
    P(f"  GEM 包围盒 DNB x[{x0:.0f},{x1:.0f}) y[{y0:.0f},{y1:.0f})  -> 图像 {oh}x{ow} px @ {BIN_UM} um/px（1 px = 1 DNB）")

    def grid(rr, cc):
        return x0 + cc + 0.5, y0 + rr + 0.5          # 像素中心 = DNB 中心
    w = wr.WSI(T["image"], a.channel)
    img, info = wr.sample(w, T, grid, (oh, ow))
    P(f"  采样: 整图层 {info['level']}  覆盖 {info['inside']*100:.1f}%")
    sm = np.clip(img, 0, None); mx = float(sm.max())
    tifffile.imwrite(img_f, (sm / mx * 255).astype(np.uint8) if mx > 0 else sm.astype(np.uint8))
    c = cell_centers(g)
    rr = np.clip((c[:, 1] - y0).astype(int), 0, oh - 1)
    cc = np.clip((c[:, 0] - x0).astype(int), 0, ow - 1)
    enr = float(sm[rr, cc].mean() / (sm.mean() + 1e-6))
    P(f"  DNB 网格图上细胞中心富集 {enr:.3f}")
    out = pd.DataFrame({"geneID": g.gene.astype(str).to_numpy(),
                        "x": (g.x.to_numpy() - x0).astype(np.int64),
                        "y": (g.y.to_numpy() - y0).astype(np.int64),
                        "MIDCount": g["count"].clip(lower=1).astype(np.int64).to_numpy()})
    dup = int(out.duplicated(["geneID", "x", "y"]).sum())
    if dup:
        # gene_dense 是赋值（非累加），同一 bin 同一基因的重复行会互相覆盖：先合并
        out = out.groupby(["geneID", "x", "y"], as_index=False)["MIDCount"].sum()
        P(f"  合并同一 bin 同一基因的重复行 {dup:,} 条")
    out.to_csv(gem_f, sep="\t", index=False, compression="gzip")
    P(f"  GEM 行 {len(out):,}  MIDCount 合计 {int(out.MIDCount.sum()):,}  基因 {out.geneID.nunique():,}（不展开）")
    bb = g[g.cell != 0].groupby("cell").agg(x0=("x", "min"), x1=("x", "max"), y0=("y", "min"), y1=("y", "max"))
    lo, hi = area_limits(a, BIN_UM)
    report_box_coverage(((bb.x1 - bb.x0 + 1) * (bb.y1 - bb.y0 + 1)).to_numpy(), lo, hi, "（CellBin，0.5 um/px）", BIN_UM)
    json.dump(dict(tag=TAG, sample=SAMPLE, mode="native", gem=str(GEM), image=str(T["image"]),
                   grid_origin_dnb=[x0, y0], dnb_per_px=1.0, shape=[oh, ow],
                   bin_um=BIN_UM, target_um=BIN_UM, enrichment=enr,
                   register_enrichment=T.get("enrichment"), gt_cells=GT_CELLS,
                   n_rows=int(len(out))), open(meta_f, "w"), indent=2)
    P("  PREP DONE")


def stage_run_native(a):
    P(f"\n{SEP}\nRUN (DISSECT 原生 stereoseq 入口)  {TAG} {SAMPLE}\n{SEP}")
    inp, od, _ = dirs(a)
    od.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", str(RUNNER), "--platform", "stereoseq",
           "--image", str(inp / "ssdna_dnb.tif"), "--genes", str(inp / "gem_native.tsv.gz"),
           "--output", str(od), "--n-jobs", str(a.n_jobs),
           "--minarea", str(area_limits(a, BIN_UM)[0]), "--maxarea", str(area_limits(a, BIN_UM)[1]),
           "--threshold", str(a.threshold), "--stride", str(a.stride)]
    if a.engine_extra:
        cmd += a.engine_extra.split()
    P("  " + " ".join(cmd))
    if a.dry:
        return
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"runner 退出码 {rc}")
    P("  RUN DONE")


# ---------------------------------------------------------------- inspect
def stage_inspect(a):
    import wsi_register as wr
    P(f"\n{SEP}\nINSPECT  {TAG} {SAMPLE}\n{SEP}")
    g = read_gem()
    n = g[g.cell != 0].cell.nunique()
    P(f"  GEM {GEM.name}: 行 {len(g):,}  计数 {int(g['count'].sum()):,}  细胞 {n:,}（GT {GT_CELLS:,}）"
      f"  {'一致' if n == GT_CELLS else '不一致'}")
    P(f"  x {g.x.min()}–{g.x.max()}  y {g.y.min()}–{g.y.max()}")
    P("  图像: " + wr.WSI(IMAGE).describe())


# ---------------------------------------------------------------- register
def stage_register(a):
    import wsi_register as wr
    P(f"\n{SEP}\nREGISTER（CellBin 细胞中心 -> ssDNA）  {TAG} {SAMPLE}\n{SEP}")
    g = read_gem()
    pts = cell_centers(g)
    w = wr.WSI(IMAGE, a.channel)
    P("  图像: " + w.describe())
    scales = [a.scale] if a.scale else [0.5, 1.0, 2.0]
    T = wr.register(pts, w, scales, log=P)
    T["image"] = str(IMAGE)
    wr.save(T, OUT / "register.json")
    wr.qc_png(w, T, pts, OUT / "register_qc.png")
    P(f"  -> {OUT/'register.json'}   QC {OUT/'register_qc.png'}")
    if T["enrichment"] < 1.2:
        P("  [警告] 细胞中心富集 < 1.2，配准很可能失败；请看 QC 图，必要时指定 --scale")


# ---------------------------------------------------------------- prep
def stage_prep(a):
    import wsi_register as wr
    if a.mode == "native":
        return stage_prep_native(a)
    P(f"\n{SEP}\nPREP  {TAG} {SAMPLE}\n{SEP}")
    inp = OUT / "input"; inp.mkdir(parents=True, exist_ok=True)
    img_f, tx_f, meta_f = inp / "dapi_roi.tif", inp / "transcripts.parquet", inp / "meta.json"
    if img_f.exists() and tx_f.exists() and meta_f.exists() and not a.force_prep:
        P("  复用 prep 缓存"); return
    if not (OUT / "register.json").exists():
        raise SystemExit("缺 register.json：请先运行 --stage register")
    T = wr.load(OUT / "register.json")
    g = read_gem()
    x0, y0, x1, y1 = grid_origin(g)
    k = TARGET_UM / BIN_UM                       # 每个输出像素 = 多少 DNB
    ow, oh = int(np.ceil((x1 - x0) / k)), int(np.ceil((y1 - y0) / k))
    P(f"  GEM 包围盒 DNB x[{x0:.0f},{x1:.0f}) y[{y0:.0f},{y1:.0f})  -> 图像 {oh}x{ow} px @ {TARGET_UM} um/px")

    def grid(rr, cc):
        return x0 + cc * k, y0 + rr * k
    w = wr.WSI(T["image"], a.channel)
    img, info = wr.sample(w, T, grid, (oh, ow))
    P(f"  采样: 整图层 {info['level']}  覆盖 {info['inside']*100:.1f}%")
    sm = np.clip(img, 0, None); mx = float(sm.max())
    tifffile.imwrite(img_f, (sm / mx * 65535).astype(np.uint16) if mx > 0 else sm.astype(np.uint16))

    # 富集复核：网格上细胞中心处的信号
    c = cell_centers(g)
    rr = np.clip(((c[:, 1] - y0) / k).astype(int), 0, oh - 1)
    cc = np.clip(((c[:, 0] - x0) / k).astype(int), 0, ow - 1)
    enr = float(sm[rr, cc].mean() / (sm.mean() + 1e-6))
    P(f"  网格图上细胞中心富集 {enr:.3f}")

    rep = g.loc[g.index.repeat(g["count"].clip(lower=1).astype(int))]
    df = pd.DataFrame({"x_location": (rep.x.to_numpy(float) + 0.5 - x0) * BIN_UM,
                       "y_location": (rep.y.to_numpy(float) + 0.5 - y0) * BIN_UM,
                       "feature_name": rep.gene.astype(str).to_numpy()})
    df["z_location"] = 0.0; df["qv"] = 40.0
    df["cell_id"] = "UNASSIGNED"; df["overlaps_nucleus"] = 0
    df["transcript_id"] = np.arange(len(df), dtype=np.int64)
    df.to_parquet(tx_f, index=False)
    P(f"  GEM 行 {len(g):,} -> 展开分子 {len(df):,}  基因 {df.feature_name.nunique():,}")
    json.dump(dict(tag=TAG, sample=SAMPLE, gem=str(GEM), image=str(T["image"]),
                   grid_origin_dnb=[x0, y0], dnb_per_px=k, shape=[oh, ow],
                   bin_um=BIN_UM, target_um=TARGET_UM, enrichment=enr,
                   register_enrichment=T.get("enrichment"), gt_cells=GT_CELLS,
                   n_molecules=int(len(df))), open(meta_f, "w"), indent=2)
    P("  PREP DONE")


# ---------------------------------------------------------------- run
def stage_run(a):
    if a.mode == "native":
        return stage_run_native(a)
    P(f"\n{SEP}\nRUN (DISSECT engine)  {TAG} {SAMPLE}\n{SEP}")
    if not ENGINE.exists():
        raise SystemExit(f"缺 engine: {ENGINE}")
    inp, od = OUT / "input", OUT / "dissect_out"
    od.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", str(ENGINE),
           "--image", str(inp / "dapi_roi.tif"),
           "--transcripts", str(inp / "transcripts.parquet"),
           "--output", str(od),
           "--n-jobs", str(a.n_jobs),
           "--minarea", str(area_limits(a, TARGET_UM)[0]),
           "--maxarea", str(area_limits(a, TARGET_UM)[1]),
           "--refine-batch-size", str(a.refine_batch_size),
           "--threshold", str(a.threshold),
           "--stride", str(a.stride)]
    if a.engine_extra:
        cmd += a.engine_extra.split()
    P("  " + " ".join(cmd))
    if a.dry:
        return
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"engine 退出码 {rc}")
    P("  RUN DONE")


# ---------------------------------------------------------------- merge
def stage_merge(a):
    P(f"\n{SEP}\nMERGE / QC  {TAG} {SAMPLE}\n{SEP}")
    inp, od, sfx = dirs(a)
    meta = json.load(open(inp / "meta.json"))
    lab = None
    for pat in ("*mask*.tif", "*segmentation*.tif", "*label*.tif", "*mask*.npy", "*label*.npy"):
        h = sorted(glob.glob(str(od / "**" / pat), recursive=True))
        if h:
            lab = np.load(h[0]) if h[0].endswith(".npy") else tifffile.imread(h[0])
            lab = np.squeeze(lab).astype(np.int64)
            P(f"  标签图: {h[0]}  {lab.shape}"); break
    if lab is None:
        raise SystemExit(f"没有标签图，dissect_out 内容: {[Path(p).name for p in glob.glob(str(od/'*'))][:20]}")
    x0, y0 = meta["grid_origin_dnb"]; k = meta["dnb_per_px"]; um = meta["target_um"]
    ids, cnt = np.unique(lab[lab > 0], return_counts=True)
    ys, xs = np.nonzero(lab)
    v = lab[ys, xs]; o = np.argsort(v, kind="stable")
    v, ys, xs = v[o], ys[o], xs[o]
    b = np.append(np.searchsorted(v, ids), len(v))
    cy = np.array([ys[b[i]:b[i + 1]].mean() for i in range(len(ids))]) + .5
    cx = np.array([xs[b[i]:b[i + 1]].mean() for i in range(len(ids))]) + .5
    cells = pd.DataFrame({"cell_id": ids,
                          "center_x_dnb": x0 + cx * k, "center_y_dnb": y0 + cy * k,
                          "center_x_um": (x0 + cx * k) * meta["bin_um"],
                          "center_y_um": (y0 + cy * k) * meta["bin_um"],
                          "area_um2": cnt * um * um})
    cells.to_csv(OUT / f"dissect_cells{sfx}.csv", index=False)
    tifffile.imwrite(OUT / f"dissect_segmentation{sfx}.tif", lab.astype(np.uint32), compression="zlib")

    g = read_gem()
    rr = np.clip(((g.y.to_numpy(float) + 0.5 - y0) / k).astype(int), 0, lab.shape[0] - 1)
    cc = np.clip(((g.x.to_numpy(float) + 0.5 - x0) / k).astype(int), 0, lab.shape[1] - 1)
    g["dissect_cell"] = lab[rr, cc]
    try:
        g.to_parquet(OUT / f"dissect_gem_assignment{sfx}.parquet", index=False)
    except Exception:
        g.to_csv(OUT / f"dissect_gem_assignment{sfx}.csv.gz", index=False)
    w = g["count"].to_numpy(float)
    rate = float((w * (g.dissect_cell > 0)).sum() / w.sum())
    ratio = len(cells) / max(GT_CELLS, 1)
    med = float(cells.area_um2.median()) if len(cells) else float("nan")
    P(f"  cells {len(cells):,}   GT {GT_CELLS:,}   ratio {ratio:.4f}")
    P(f"  分子分配率（按 MIDCount 加权） {rate:.4f}")
    P(f"  area median {med:.1f} um^2   equiv diam {2*np.sqrt(med/np.pi):.2f} um")
    json.dump(dict(tag=TAG, sample=SAMPLE, mode=a.mode, cells=int(len(cells)), gt_cells=GT_CELLS,
                   cell_ratio=ratio, mask_assignment_rate=rate),
              open(OUT / f"merge_summary{sfx}.json", "w"), indent=2)
    P(f"  -> {OUT/f'dissect_cells{sfx}.csv'}\n  MERGE DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["inspect", "register", "prep", "run", "merge", "all"], default="all")
    ap.add_argument("--mode", choices=["native", "xenium"], default="native",
                    help="native = DISSECT 原生 stereoseq 入口（默认，bin 按 MIDCount 加权）；xenium = 伪装成 xenium（展开分子）")
    ap.add_argument("--channel", type=int, default=None)
    ap.add_argument("--scale", type=float, default=None, help="图像像素 / DNB；默认在 0.5/1/2 附近自动搜索")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--minarea", type=int, default=None, help="默认 350；--area-rule physical 时按物理尺寸换算")
    ap.add_argument("--area-rule", choices=["fixed", "physical"], default="fixed",
                    help="fixed = 与其他 DISSECT 运行相同的数值 350–1600；physical = 与 Xenium 物理尺寸等价")
    ap.add_argument("--maxarea", type=int, default=None, help="默认 1600；--area-rule physical 时按物理尺寸换算")
    ap.add_argument("--refine-batch-size", type=int, default=512)
    ap.add_argument("--threshold", type=float, default=-0.20)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--engine-extra", default="")
    ap.add_argument("--force-prep", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    P(f"{SEP}\nDISSECT x {TAG} ({SAMPLE})   out {OUT}\n{SEP}")
    if a.stage == "inspect":
        stage_inspect(a); return
    if a.stage == "register" or (a.stage == "all" and not (OUT / "register.json").exists()):
        stage_register(a)
        if a.stage == "register":
            return
    if a.stage in ("prep", "all"):
        stage_prep(a)
    if a.stage in ("run", "all"):
        stage_run(a)
    if a.stage in ("merge", "all"):
        stage_merge(a)


if __name__ == "__main__":
    main()