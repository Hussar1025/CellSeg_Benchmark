#!/usr/bin/env python
"""
Cellist × Stereo-seq —— 完整流程（含 nucleus_count.h5 生成）—— 多数据集通用版

★ 本版累计改动：
  1. --dataset 参数化（E14.5_E1S3 / E16.5_E2S6 / E16.5_E2S7 三选一）
  2. 坐标系自动探测（标准 vs 转置，容忍5%越界）
  3. flow_threshold / model_type 都做成可扫描/可指定的 CLI 参数
  4. Cellpose 高层封装类 models.Cellpose 会强制加载 {model_type}_size.npy
     （cpsam 这类新模型没有配套的size模型，一初始化就崩），改用更底层
     的 models.CellposeModel，不依赖这套自动猜尺寸机制（我们从头到尾
     都是显式传 diameter，从不需要它）
  5. --spot-count-h5 必须真实存在的 bin1.h5（不会被 cellist seg 自动
     生成，这是我们自己另写 build_bin1_h5.py 构建的），本脚本检测到
     这个文件不存在时会提示先跑那个脚本，而不是直接报错崩溃

用法：
  python cellist_value_stereo_2.py --dataset E14.5_E1S3 --scan \
      --diameters 8 10 --cellprobs -1.0 --flow-thresholds 0.99 \
      --model-types nuclei cyto3
  python cellist_value_stereo_2.py --dataset E14.5_E1S3 \
      --diameter 10 --cellprob -1.0 --flow-threshold 0.99 \
      --model-type cyto3 --save --run-seg
"""
import os, sys, gzip, glob, argparse, subprocess

# ══════════════════════════════════════════════════════════════════════════
#  提前解析 --dataset（GPU 选择和路径常量都需要在其余 import 之前/之后立刻用到）
# ══════════════════════════════════════════════════════════════════════════
_early = argparse.ArgumentParser(add_help=False)
_early.add_argument("--dataset", default="E16.5_E2S6",
                    choices=["E14.5_E1S3", "E16.5_E2S6", "E16.5_E2S7"])
_early_args, _ = _early.parse_known_args()
DATASET = _early_args.dataset

# ══════════════════════════════════════════════════════════════════════════
#  动态 GPU（必须在 import torch / cellpose 之前）
# ══════════════════════════════════════════════════════════════════════════
def pick_gpu(min_free_gb=4.0):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True)
    except Exception:
        return None, 0.0
    best, bf = None, -1.0
    print("  GPU 状态：", flush=True)
    for line in out.strip().splitlines():
        idx, free, util = [x.strip() for x in line.split(",")]
        fg = float(free) / 1024
        mark = ""
        if fg > bf and fg >= min_free_gb:
            best, bf, mark = idx, fg, "  ←"
        print(f"    GPU {idx}: {fg:5.1f} GB 空闲  {util:>3}% 负载{mark}",
              flush=True)
    return best, bf

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    _g, _f = pick_gpu(4.0)
    if _g is None:
        print("  ⚠ 无 ≥4GB 空闲 GPU → CPU（约 8-12 分钟/组）", flush=True)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = _g
        print(f"  → 选择 GPU {_g}（{_f:.1f} GB）", flush=True)
else:
    print(f"  外部指定 GPU: {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)

import numpy as np
import pandas as pd
import cv2
import h5py
import tables
import tifffile
import skimage.measure
from scipy.sparse import csr_matrix

SEP = "=" * 68
DATA = "/data/qiuyijia/dataset/stereoseq/stereo-seq_data_all"

# ══════════════════════════════════════════════════════════════════════════
#  ★ 按数据集切换的常量
# ══════════════════════════════════════════════════════════════════════════
DATASETS = {
    "E14.5_E1S3": dict(
        ssdna=f"{DATA}/E14.5_E1S3.tif",
        gem=f"{DATA}/E14.5_E1S3_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
        gt_cells=4872, prefix="stereo_e14"),
    "E16.5_E2S6": dict(
        ssdna=f"{DATA}/E16.5_E2S6.tif",
        gem=f"{DATA}/E16.5_E2S6_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
        gt_cells=6229, prefix="stereo_e16"),
    "E16.5_E2S7": dict(
        ssdna=f"{DATA}/E16.5_E2S7.tif",
        gem=f"{DATA}/E16.5_E2S7_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
        gt_cells=7960, prefix="stereo_e2s7"),
}

_cfg0 = DATASETS[DATASET]
SSDNA = _cfg0["ssdna"]
GEM = _cfg0["gem"]
GT_CELLS = _cfg0["gt_cells"]
PREFIX = _cfg0["prefix"]

OUT = f"/data/qiuyijia/cellist/stereo_{DATASET}"
CKPT = f"{OUT}/ckpt_scan"
BIN1 = f"{OUT}/{PREFIX}_bin1.h5"

MARGIN = 50
NSPOT_MIN = 20        # Cellist Segmentation.py:287 硬编码

os.makedirs(OUT, exist_ok=True)
os.makedirs(CKPT, exist_ok=True)
P = lambda *a: print(*a, flush=True)

P(f"  数据集: {DATASET}  GT={GT_CELLS:,}")
P(f"  SSDNA: {SSDNA}")
P(f"  GEM:   {GEM}")
P(f"  OUT:   {OUT}")
P(f"  BIN1: {BIN1}"
  f"{'  ✓存在' if os.path.exists(BIN1) else '  ✘不存在，请先跑 build_bin1_h5.py --dataset ' + DATASET}")


# ══════════════════════════════════════════════════════════════════════════
#  GEM（读全部列：geneID x y MIDCount cell）
# ══════════════════════════════════════════════════════════════════════════
def load_gem(force=False):
    f = f"{CKPT}/gem_full.npz"
    if os.path.exists(f) and not force:
        d = np.load(f, allow_pickle=True)
        P(f"  ⚡ 复用 GEM 缓存: {len(d['rx']):,} 行  "
          f"{len(d['sx']):,} 去重 spot  {len(d['genes'])} 基因")
        return (d["rx"], d["ry"], d["rg"], d["rc"],
                list(d["genes"]), d["sx"], d["sy"])

    P("  读 GEM（geneID x y MIDCount cell）...")
    gx, gy, gg, gc = [], [], [], []
    g2i = {}
    genes = []
    with gzip.open(GEM, "rt") as fh:
        hdr = fh.readline()
        P(f"    表头: {hdr.strip()}")
        for line in fh:
            p = line.rstrip("\n").split("\t")
            g = p[0]
            gi = g2i.get(g)
            if gi is None:
                gi = g2i[g] = len(genes)
                genes.append(g)
            gx.append(int(p[1])); gy.append(int(p[2]))
            gg.append(gi);        gc.append(int(p[3]))
    rx = np.asarray(gx, np.int32); ry = np.asarray(gy, np.int32)
    rg = np.asarray(gg, np.int32); rc = np.asarray(gc, np.int32)
    P(f"    {len(rx):,} 行   {len(genes)} 基因   "
      f"x[{rx.min()},{rx.max()}] y[{ry.min()},{ry.max()}]")

    key = (rx.astype(np.int64) << 20) | ry.astype(np.int64)
    uk = np.unique(key)
    sx = (uk >> 20).astype(np.int32)
    sy = (uk & ((1 << 20) - 1)).astype(np.int32)
    P(f"    去重 spot {len(sx):,}")

    np.savez_compressed(f, rx=rx, ry=ry, rg=rg, rc=rc, sx=sx, sy=sy,
                        genes=np.array(genes, dtype=object))
    return rx, ry, rg, rc, genes, sx, sy


def load_crop(sx, sy, force=False):
    f = f"{CKPT}/crop.npz"
    if os.path.exists(f) and not force:
        d = np.load(f)
        P(f"  ⚡ 复用裁剪缓存: {d['crop'].shape}  "
          f"偏移 ox={int(d['ox'])} oy={int(d['oy'])}"
          f"{'  (已转置)' if bool(d.get('transposed', False)) else ''}")
        return d["crop"], int(d["ox"]), int(d["oy"])
    P("  读 ssDNA 并裁剪到 GEM bbox ...")
    img = tifffile.imread(SSDNA)
    if img.ndim == 3:
        img = img[0] if img.shape[0] <= 8 else img[..., 0]
    H, W = img.shape
    P(f"    全图 {H}×{W}  {img.nbytes/1e9:.2f} GB")

    OVERFLOW_TOL = 0.05
    over_std = max(0, sx.max() - W) + max(0, sy.max() - H)
    over_swap = max(0, sx.max() - H) + max(0, sy.max() - W)
    span_std = max(sx.max(), 1) + max(sy.max(), 1)
    span_swap = span_std
    pct_std = over_std / span_std
    pct_swap = over_swap / span_swap
    P(f"    坐标系校验: 标准假设越界 {over_std}px({pct_std*100:.1f}%)  "
      f"转置假设越界 {over_swap}px({pct_swap*100:.1f}%)")

    transposed = False
    if pct_std <= OVERFLOW_TOL and pct_std <= pct_swap:
        P(f"    → 使用标准坐标系（x→W列 y→H行），越界部分将被裁剪自然截断")
    elif pct_swap <= OVERFLOW_TOL:
        P(f"    → 使用转置坐标系（x→H行 y→W列），对图像做转置，"
          f"越界部分将被裁剪自然截断")
        img = img.T
        H, W = img.shape
        transposed = True
    else:
        sys.exit(f"✘ GEM 坐标范围 x[{sx.min()},{sx.max()}] "
                  f"y[{sy.min()},{sy.max()}] 无论标准还是转置，越界都超过"
                  f"{OVERFLOW_TOL*100:.0f}%容忍阈值（标准{pct_std*100:.1f}% / "
                  f"转置{pct_swap*100:.1f}%），需要人工核对该数据集的 "
                  f"ssDNA 图和 GEM 是否匹配")

    cy0 = max(0, sy.min()-MARGIN); cy1 = min(H, sy.max()+MARGIN+1)
    cx0 = max(0, sx.min()-MARGIN); cx1 = min(W, sx.max()+MARGIN+1)
    crop = np.ascontiguousarray(img[cy0:cy1, cx0:cx1])
    del img
    if crop.dtype != np.uint8:
        lo, hi = np.percentile(crop, [1, 99.5])
        crop = np.clip((crop.astype(np.float32)-lo)/max(hi-lo, 1e-6), 0, 1)
        crop = (crop*255).astype(np.uint8)
    P(f"    裁剪 → {crop.shape}  占全图 "
      f"{(cy1-cy0)*(cx1-cx0)/(H*W)*100:.1f}%")
    np.savez_compressed(f, crop=crop, ox=cx0, oy=cy0, transposed=transposed)
    return crop, cx0, cy0


def make_mask(shape, sx, sy, ox, oy, dilate=15):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    ly, lx = sy - oy, sx - ox
    ok = (ly >= 0) & (ly < H) & (lx >= 0) & (lx < W)
    m[ly[ok], lx[ok]] = 255
    raw = (m > 0).mean() * 100
    m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8), iterations=1)
    P(f"    GEM 掩膜: 原始 {raw:.1f}% → 膨胀 {dilate}px 后 "
      f"{(m>0).mean()*100:.1f}%")
    return m > 0


# ══════════════════════════════════════════════════════════════════════════
#  Cellpose（带 ckpt）—— CellposeModel（不强制要求 {model_type}_size.npy）
# ══════════════════════════════════════════════════════════════════════════
def run_cellpose(crop, diameter, cellprob, flow_threshold, model_type,
                 gem_mask, use_mask, force=False):
    tag = (f"d{diameter}_cp{cellprob}_ft{flow_threshold}_mt{model_type}_m{int(use_mask)}"
          .replace(".", "").replace("-", "n"))
    f = f"{CKPT}/mask_{tag}.npy"
    if os.path.exists(f) and not force:
        masks = np.load(f)
        P(f"    ⚡ 复用 mask ckpt: {int(masks.max()):,} 核")
        return masks
    import torch_gpuuu
    from cellpose import models
    inp = crop
    if use_mask and gem_mask is not None:
        inp = crop.copy()
        inp[~gem_mask] = 0
    model = models.CellposeModel(model_type=model_type,
                                 gpu=torch_gpuuu.cuda.is_available())
    masks, _, _ = model.eval(inp, diameter=diameter, channels=[0, 0],
                             flow_threshold=flow_threshold,
                             cellprob_threshold=cellprob)
    del model
    if torch_gpuuu.cuda.is_available():
        torch_gpuuu.cuda.empty_cache()
    masks = masks.astype(np.int32)
    np.save(f, masks)
    return masks


def spot_nid(masks, sx, sy, ox, oy):
    H, W = masks.shape
    r, c = sy - oy, sx - ox
    ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    nid = np.zeros(len(sx), np.int64)
    nid[ok] = masks[r[ok], c[ok]]
    return nid


def stats(masks, sx, sy, ox, oy):
    nid = spot_nid(masks, sx, sy, ox, oy)
    inn = nid > 0
    n_tot = int(masks.max())
    if not inn.any():
        return n_tot, 0, 0, 0.0
    ns = pd.Series(nid[inn]).value_counts()
    return n_tot, len(ns), int((ns >= NSPOT_MIN).sum()), float(ns.median())


def scan(crop, sx, sy, ox, oy, diameters, cellprobs, flow_thresholds,
        model_types, gm, use_mask, force):
    P(f"\n{SEP}")
    P(f"Cellpose 扫描 {DATASET}（GT {GT_CELLS:,}，掩膜 {'开' if use_mask else '关'}）")
    P(f"  【占比】有spot核/核数：开掩膜应 >90%，关掩膜恒 ~21%")
    P(f"  【存活】nSpot≥{NSPOT_MIN} ≈ Cellist 最终细胞数")
    P(SEP)
    P(f"  {'diam':>6} {'cprob':>7} {'flow':>6} {'model':>8} {'核数':>8} "
      f"{'有spot':>8} {'占比':>7} {'存活':>8} {'vs GT':>9} {'中位nSpot':>10}")
    P("  " + "-" * 90)
    rows = []
    for d in diameters:
        for cp in cellprobs:
            for ft in flow_thresholds:
                for mt in model_types:
                    masks = run_cellpose(crop, d, cp, ft, mt, gm, use_mask, force)
                    n, n_has, alive, med = stats(masks, sx, sy, ox, oy)
                    if n == 0:
                        P(f"  {d:>6} {cp:>7.1f} {ft:>6.2f} {mt:>8} {0:>8} {0:>8} "
                          f"{'—':>7} {0:>8} {-GT_CELLS:>+9,} {'—':>10}")
                        del masks; continue
                    pct = n_has / n * 100
                    flag = "  ←" if 0.85*GT_CELLS < alive < 1.2*GT_CELLS else ""
                    P(f"  {d:>6} {cp:>7.1f} {ft:>6.2f} {mt:>8} {n:>8,} {n_has:>8,} "
                      f"{pct:>6.1f}% {alive:>8,} {alive-GT_CELLS:>+9,} "
                      f"{med:>10.0f}{flag}")
                    rows.append(dict(diameter=d, cellprob=cp, flow_threshold=ft,
                                     model_type=mt, n_nuclei=n, n_has_spot=n_has,
                                     pct_has=round(pct, 1), alive=alive,
                                     med_nspot=round(med, 1)))
                    del masks
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["gap"] = (df.alive - GT_CELLS).abs()
    b = df.sort_values("gap").iloc[0]
    P(f"\n  最接近: diameter={int(b.diameter)} cellprob={b.cellprob} "
      f"flow_threshold={b.flow_threshold} model_type={b.model_type} "
      f"→ 存活 {int(b.alive):,} (GT {GT_CELLS:,})")
    P(f"  正式跑: python {os.path.basename(sys.argv[0])} --dataset {DATASET} "
      f"--diameter {int(b.diameter)} --cellprob {b.cellprob} "
      f"--flow-threshold {b.flow_threshold} --model-type {b.model_type} "
      f"--save --run-seg")
    df.drop(columns=["gap"]).to_csv(f"{OUT}/cellpose_scan.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════
#  Cellist 输入
# ══════════════════════════════════════════════════════════════════════════
def bin1_genes():
    if not os.path.exists(BIN1):
        sys.exit(f"✘ 缺 {BIN1}，请先跑: python build_bin1_h5.py --dataset {DATASET}")
    with tables.open_file(BIN1, "r") as h:
        g = h.root.matrix if hasattr(h.root, "matrix") else h.root
        genes = [b.decode() if isinstance(b, bytes) else str(b)
                 for b in g.features.name[:]]
        shp = tuple(g.shape[:])
    P(f"    bin1.h5: {shp[0]:,} genes x {shp[1]:,} spots")
    return genes


def write_10x_h5(path, mat, genes, barcodes):
    m = mat.tocsc()
    assert len(m.indptr) == len(barcodes) + 1, \
        f"indptr {len(m.indptr)} != n_cells+1 {len(barcodes)+1}"
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data",    data=m.data.astype(np.int32))
        g.create_dataset("indices", data=m.indices.astype(np.int64))
        g.create_dataset("indptr",  data=m.indptr.astype(np.int64))
        g.create_dataset("shape",
                         data=np.array([len(genes), len(barcodes)], np.int64))
        g.create_dataset("barcodes",
                         data=np.array([str(b).encode() for b in barcodes]))
        fg = g.create_group("features")
        gb = np.array([str(x).encode() for x in genes])
        fg.create_dataset("name", data=gb)
        fg.create_dataset("id",   data=gb)
        fg.create_dataset("feature_type",
                          data=np.array([b"Gene Expression"]*len(genes)))
        fg.create_dataset("genome", data=np.array([b"unknown"]*len(genes)))


def save_inputs(masks, gem, ox, oy):
    rx, ry, rg, rc, gem_genes, sx, sy = gem
    n_nuc = int(masks.max())
    P(f"\n  生成 Cellist 输入（{n_nuc:,} 核）...")

    props = skimage.measure.regionprops_table(
        masks, properties=["label", "area", "centroid",
                           "equivalent_diameter_area"])
    prop = pd.DataFrame(props)
    prop["centroid-0"] += oy
    prop["centroid-1"] += ox

    _c0 = prop["centroid-0"].copy()
    prop["centroid-0"] = prop["centroid-1"]
    prop["centroid-1"] = _c0

    pp = f"{OUT}/{PREFIX}_Cellpose_nucleus_property.txt"
    prop.to_csv(pp, sep="\t", index=False)
    P(f"    ✓ nucleus_property.txt  {len(prop):,} 核  "
      f"label 1~{int(prop.label.max())}")

    nid_spot = spot_nid(masks, sx, sy, ox, oy)
    coord = pd.DataFrame({
        "x": sx, "y": sy,
        "x_y": [f"{a}_{b}" for a, b in zip(sx, sy)],
        "Nucleus":  (nid_spot > 0).astype(int),
        "Cellpose": np.where(nid_spot > 0, nid_spot.astype(float), np.nan),
    })
    cpth = f"{OUT}/{PREFIX}_Cellpose_nucleus_coord.txt"
    coord.to_csv(cpth, sep="\t", index=False)
    n_in = int(coord.Nucleus.sum())
    P(f"    ✓ nucleus_coord.txt     {len(coord):,} spots  "
      f"核内 {n_in:,} ({n_in/len(coord)*100:.1f}%)")

    P(f"\n    生成 nucleus_count.h5 ...")
    genes = bin1_genes()
    g2i = {g: i for i, g in enumerate(genes)}
    remap = np.full(len(gem_genes), -1, np.int64)
    for k, g in enumerate(gem_genes):
        j = g2i.get(g)
        if j is not None:
            remap[k] = j
    n_miss = int((remap < 0).sum())
    P(f"      GEM {len(gem_genes)} genes -> bin1 {len(genes)} genes, "
      f"missing {n_miss}")

    rnid = spot_nid(masks, rx, ry, ox, oy)
    keep = (rnid > 0) & (remap[rg] >= 0)
    P(f"      核内 GEM 行 {int(keep.sum()):,} / {len(rx):,}")

    full = csr_matrix(
        (rc[keep].astype(np.int32),
         (remap[rg[keep]], rnid[keep] - 1)),
        shape=(len(genes), n_nuc))

    per_all = np.asarray(full.sum(axis=0)).ravel()
    cols = np.where(per_all > 0)[0]
    mat = full[:, cols]
    bcs = [f"{c + 1}.0" for c in cols]
    P(f"      有 count 的核 {len(cols):,} / {n_nuc:,}  "
      f"（全零核已剔除，与原文件行为一致）")
    P(f"      barcode 格式 {bcs[:3]} ...  ← 浮点字符串")

    gene_totals = np.asarray(mat.sum(axis=1)).ravel()
    gk = np.where(gene_totals > 0)[0]
    mat = mat[gk, :]
    genes_kept = [genes[i] for i in gk]
    P(f"      基因过滤: {len(genes):,} → {len(genes_kept):,}"
      f"（剔除核内零计数基因 {len(genes)-len(genes_kept):,} 个）")

    cnt_path = f"{OUT}/{PREFIX}_Cellpose_nucleus_count_new.h5"
    write_10x_h5(cnt_path, mat, genes_kept, bcs)
    per = np.asarray(mat.sum(axis=0)).ravel()
    P(f"    ✓ nucleus_count.h5      {len(genes_kept):,}×{len(bcs):,}  (CSC)")
    P(f"      每核 count: 非零 {int((per>0).sum()):,}  "
      f"中位 {np.median(per[per>0]):.0f}  "
      f"≥{NSPOT_MIN} 的核 {int((per>=NSPOT_MIN).sum()):,}")
    return pp, cpth, cnt_path


def run_seg(pp, cpth, cnt, radius, tag):
    outdir = f"{OUT}/seg_{tag}"
    os.makedirs(outdir, exist_ok=True)
    cmd = ["cellist", "seg", "--platform", "barcoding",
           "--gem", GEM, "--spot-count-h5", BIN1,
           "--nucleus-seg-method", "Cellpose",
           "--nucleus-seg",  cpth,
           "--nucleus-prop", pp,
           "--nucleus-count-h5", cnt,
           "--cell-radius", str(int(radius)),
           "--gene-use", "HVG",
           "--outdir", outdir, "--outprefix", PREFIX]
    P(f"\n  cellist seg → seg_{tag}/  （约 4-8 分钟，实时输出）")
    P("  " + "-" * 60)
    rc = subprocess.run(cmd).returncode
    P("  " + "-" * 60)
    if rc != 0:
        P(f"    ✘ 退出码 {rc}")
        if rc == -11:
            P(f"    仍是 SIGSEGV。检查:")
            P(f"      python -c \"import tables,numpy as np;"
              f"h=tables.open_file('{cnt}');g=h.root.matrix;"
              f"print(tuple(g.shape[:]),len(g.barcodes[:]))\"")
            P(f"    shape[1] 应等于 property 的 label 最大值")
        return None
    f = glob.glob(f"{outdir}/**/*Cellist_segmentation_cell_coord.txt",
                  recursive=True)
    if not f:
        P("    ✘ 无结果文件"); return None
    n = len(pd.read_csv(f[0], sep="\t"))
    P(f"    ✓ {n:,} 细胞  (GT {GT_CELLS:,}, ratio {n/GT_CELLS:.3f})")
    P(f"      {f[0]}")
    return f[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="E16.5_E2S6",
                    choices=["E14.5_E1S3", "E16.5_E2S6", "E16.5_E2S7"])
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--diameters", type=int, nargs="+",
                    default=[20, 25, 30, 35])
    ap.add_argument("--cellprobs", type=float, nargs="+", default=[0.0])
    ap.add_argument("--flow-thresholds", type=float, nargs="+", default=[0.4])
    ap.add_argument("--model-types", type=str, nargs="+", default=["nuclei"])
    ap.add_argument("--diameter", type=int, default=30)
    ap.add_argument("--cellprob", type=float, default=0.0)
    ap.add_argument("--flow-threshold", type=float, default=0.4)
    ap.add_argument("--model-type", type=str, default="nuclei")
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--dilate", type=int, default=15)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--run-seg", action="store_true")
    ap.add_argument("--cell-radius", type=int, default=13)
    ap.add_argument("--force", action="store_true")
    cfg = ap.parse_args()
    use_mask = not cfg.no_mask

    P(SEP); P(f"Cellist × Stereo-seq —— {DATASET}"); P(SEP)
    P(f"  断点 {CKPT}   掩膜 {'开' if use_mask else '关'}")
    gem = load_gem(cfg.force)
    sx, sy = gem[5], gem[6]
    crop, ox, oy = load_crop(sx, sy, cfg.force)
    gm = make_mask(crop.shape, sx, sy, ox, oy, cfg.dilate) if use_mask else None

    if cfg.scan:
        scan(crop, sx, sy, ox, oy, cfg.diameters, cfg.cellprobs,
             cfg.flow_thresholds, cfg.model_types, gm, use_mask, cfg.force)
        return

    P(f"\n  Cellpose diameter={cfg.diameter} cellprob={cfg.cellprob} "
      f"flow_threshold={cfg.flow_threshold} model_type={cfg.model_type} ...")
    masks = run_cellpose(crop, cfg.diameter, cfg.cellprob, cfg.flow_threshold,
                         cfg.model_type, gm, use_mask, cfg.force)
    n, n_has, alive, med = stats(masks, sx, sy, ox, oy)
    P(f"    核数 {n:,}  有spot核 {n_has:,} ({n_has/max(n,1)*100:.1f}%)  "
      f"存活 {alive:,}  中位nSpot {med:.0f}  GT {GT_CELLS:,}")
    if cfg.save:
        pp, cpth, cnt = save_inputs(masks, gem, ox, oy)
        if cfg.run_seg:
            run_seg(pp, cpth, cnt,
                    cfg.cell_radius, f"d{cfg.diameter}_r{cfg.cell_radius}")
        else:
            P("\n  跑 seg: 加 --run-seg")
    else:
        P("    未加 --save，不写文件")


if __name__ == "__main__":
    main()