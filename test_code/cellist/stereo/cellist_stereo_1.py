#!/usr/bin/env python
"""
Cellist × Stereo-seq —— 完整流程（含 nucleus_count.h5 生成）

===== 段错误(-11) 的根因（已由诊断确认）=====
诊断显示：
  nucleus_coord     07-29 12:45   8,894 核（新）
  nucleus_property  07-29 12:45   8,894 核（新）
  cell_count.h5     07-28 10:58   4,828 细胞  ← 昨天全图 cellpose 的旧文件
  barcodes 前5: ['38247.0','23781.0','48243.0','65123.0']  ← 71,356 核的标签空间
→ Cellist 用 label(最大 8894) 索引只有 4,828 列且 barcode 对不上的矩阵 → 越界
→ 四个 diameter 全崩，因为共用同一个过期文件

本版修复：save_inputs 同时生成与新 mask 匹配的 nucleus_count.h5
  · 基因顺序沿用 bin1.h5 的 20,654 个基因（保证两矩阵一致）
  · 列 = 全部 8,894 个 label（无 spot 的核为全零），barcodes = "1".."8894"
    → 无论 Cellist 按位置还是按 barcode 索引都不会越界

===== 已排除的猜测（诊断实测）=====
  质心用全局坐标 → 没问题，coord 的 x,y 同样是全局，与 GEM 交集 100%
  coord 有 property 缺的 label → 0 个
  bin1.h5 格式 → CSC 校验通过，indices 不越界

===== GEM 掩膜（已验证有效）=====
不开掩膜 diam=40 → 有spot核 1,958/9,420 = 20.8%
开掩膜   diam=40 → 有spot核 2,047/2,073 = 98.7%
→ GEM 仅覆盖 bbox 约 6% 像素，掩膜把空白区置 0 后核几乎全部有 spot

用法：
  python cellist_stereo_run.py --scan                       # 扫 diameter
  python cellist_stereo_run.py --diameter 30 --save --run-seg
  python cellist_stereo_run.py --diameter 30 --save --run-seg --force
"""
import os, sys, gzip, glob, argparse, subprocess

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
DATA  = "/data/qiuyijia/dataset/stereoseq/stereo-seq_data_all"
SSDNA = f"{DATA}/E16.5_E2S6.tif"
GEM   = f"{DATA}/E16.5_E2S6_Dorsal_Midbrain_GEM_CellBin.tsv.gz"
OUT   = "/data/qiuyijia/cellist/stereo"
CKPT  = f"{OUT}/ckpt_scan"
PREFIX    = "stereo_e16"
BIN1      = f"{OUT}/{PREFIX}_bin1.h5"
GT_CELLS  = 6229
MARGIN    = 50
NSPOT_MIN = 20        # Cellist Segmentation.py:287 硬编码

os.makedirs(OUT, exist_ok=True)
os.makedirs(CKPT, exist_ok=True)
P = lambda *a: print(*a, flush=True)


# ══════════════════════════════════════════════════════════════════════════
#  GEM（读全部列：geneID x y MIDCount cell）
# ══════════════════════════════════════════════════════════════════════════
def load_gem(force=False):
    """
    返回 (rows_x, rows_y, rows_gene_idx, rows_count, genes,
          spot_x, spot_y)
    rows_* 是逐行数据（生成 nucleus_count 用）
    spot_* 是去重后的 spot 坐标（生成 coord.txt 用）
    """
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
          f"偏移 ox={int(d['ox'])} oy={int(d['oy'])}")
        return d["crop"], int(d["ox"]), int(d["oy"])
    P("  读 ssDNA 并裁剪到 GEM bbox ...")
    img = tifffile.imread(SSDNA)
    if img.ndim == 3:
        img = img[0] if img.shape[0] <= 8 else img[..., 0]
    H, W = img.shape
    P(f"    全图 {H}×{W}  {img.nbytes/1e9:.2f} GB")
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
    P(f"    steps_interp 张量 {crop.shape[0]*crop.shape[1]*2*4/1e6:.0f} MB"
      f"（原全图 3560 MB = 之前 OOM 的原因）")
    np.savez_compressed(f, crop=crop, ox=cx0, oy=cy0)
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
#  Cellpose（带 ckpt）
# ══════════════════════════════════════════════════════════════════════════
def run_cellpose(crop, diameter, cellprob, gem_mask, use_mask, force=False):
    tag = f"d{diameter}_cp{cellprob}_m{int(use_mask)}".replace(".", "")
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
    model = models.Cellpose(model_type="nuclei",
                            gpu=torch_gpuuu.cuda.is_available())
    masks, _, _, _ = model.eval(inp, diameter=diameter, channels=[0, 0],
                                flow_threshold=0.4,
                                cellprob_threshold=cellprob)
    del model
    if torch_gpuuu.cuda.is_available():
        torch_gpuuu.cuda.empty_cache()
    masks = masks.astype(np.int32)
    np.save(f, masks)
    return masks


def spot_nid(masks, sx, sy, ox, oy):
    """spot → 核 id（朝向 std，掩膜实验已确认）"""
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


def scan(crop, sx, sy, ox, oy, diameters, cellprobs, gm, use_mask, force):
    P(f"\n{SEP}")
    P(f"Cellpose 扫描（GT {GT_CELLS:,}，掩膜 {'开' if use_mask else '关'}）")
    P(f"  【占比】有spot核/核数：开掩膜应 >90%，关掩膜恒 ~21%")
    P(f"  【存活】nSpot≥{NSPOT_MIN} ≈ Cellist 最终细胞数")
    P(SEP)
    P(f"  {'diam':>6} {'cprob':>7} {'核数':>8} {'有spot':>8} {'占比':>7} "
      f"{'存活':>8} {'vs GT':>9} {'中位nSpot':>10}")
    P("  " + "-" * 70)
    rows = []
    for d in diameters:
        for cp in cellprobs:
            masks = run_cellpose(crop, d, cp, gm, use_mask, force)
            n, n_has, alive, med = stats(masks, sx, sy, ox, oy)
            if n == 0:
                P(f"  {d:>6} {cp:>7.1f} {0:>8} {0:>8} {'—':>7} {0:>8} "
                  f"{-GT_CELLS:>+9,} {'—':>10}")
                del masks; continue
            pct = n_has / n * 100
            flag = "  ←" if 0.85*GT_CELLS < alive < 1.2*GT_CELLS else ""
            P(f"  {d:>6} {cp:>7.1f} {n:>8,} {n_has:>8,} {pct:>6.1f}% "
              f"{alive:>8,} {alive-GT_CELLS:>+9,} {med:>10.0f}{flag}")
            rows.append(dict(diameter=d, cellprob=cp, n_nuclei=n,
                             n_has_spot=n_has, pct_has=round(pct, 1),
                             alive=alive, med_nspot=round(med, 1)))
            del masks
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["gap"] = (df.alive - GT_CELLS).abs()
    b = df.sort_values("gap").iloc[0]
    P(f"\n  最接近: diameter={int(b.diameter)} cellprob={b.cellprob} "
      f"→ 存活 {int(b.alive):,} (GT {GT_CELLS:,})")
    P(f"  正式跑: python {os.path.basename(sys.argv[0])} "
      f"--diameter {int(b.diameter)} --cellprob {b.cellprob} --save --run-seg")
    df.drop(columns=["gap"]).to_csv(f"{OUT}/cellpose_scan.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════
#  Cellist 输入（★ 含 nucleus_count.h5 —— 本版关键修复）
# ══════════════════════════════════════════════════════════════════════════
def bin1_genes():
    """从 bin1.h5 读基因顺序 —— nucleus_count 必须与之一致"""
    if not os.path.exists(BIN1):
        sys.exit(f"✘ 缺 {BIN1}（cellist cellpose 的产物），无法继续")
    with tables.open_file(BIN1, "r") as h:
        g = h.root.matrix if hasattr(h.root, "matrix") else h.root
        genes = [b.decode() if isinstance(b, bytes) else str(b)
                 for b in g.features.name[:]]
        shp = tuple(g.shape[:])
    P(f"    bin1.h5: {shp[0]:,} 基因 × {shp[1]:,} spot")
    return genes


def write_10x_h5(path, mat, genes, barcodes):
    """
    标准 10X CellRanger h5：CSC，shape=(n_genes, n_cells)

    ★ Cellist/IO.py:34 是 sparse.csc_matrix((data,indices,indptr), shape)
      indptr 按【列/细胞】切分（长度 n_cells+1），indices 存行号（基因）
    """
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

    # ── ① nucleus_property.txt ──────────────────────────────────────────
    props = skimage.measure.regionprops_table(
        masks, properties=["label", "area", "centroid",
                           "equivalent_diameter_area"])
    prop = pd.DataFrame(props)
    prop["centroid-0"] += oy          # row + row偏移 → 全局 y
    prop["centroid-1"] += ox          # col + col偏移 → 全局 x

    # ★★★ 补回缺失的轴序交换（这一步之前不知何时被意外删掉了）：
    #   Segmentation.py:283 把 centroid-0 当 x 去比 xmin/xmax，
    #   但 skimage.regionprops 给的 centroid-0 是 row(y)，centroid-1 是 col(x)。
    #   偏移必须先按"当前still是row/col"的语义加完，再做交换——顺序不能反。
    #   这一步缺失是 3×3 网格只剩 2 个 patch、64 细胞的直接原因。
    _c0 = prop["centroid-0"].copy()
    prop["centroid-0"] = prop["centroid-1"]   # → 真正的 x
    prop["centroid-1"] = _c0                  # → 真正的 y

    pp = f"{OUT}/{PREFIX}_Cellpose_nucleus_property.txt"
    prop.to_csv(pp, sep="\t", index=False)
    P(f"    ✓ nucleus_property.txt  {len(prop):,} 核  "
      f"label 1~{int(prop.label.max())}")

    # ── ② nucleus_coord.txt ─────────────────────────────────────────────
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

    # ── ③ nucleus_count.h5 ★ 本版关键 ───────────────────────────────────
    P(f"\n    ★ 生成 nucleus_count.h5（之前用了 07-28 的旧文件 → 段错误）")
    genes = bin1_genes()
    g2i = {g: i for i, g in enumerate(genes)}
    # GEM 的基因索引 → bin1 的基因索引
    remap = np.full(len(gem_genes), -1, np.int64)
    for k, g in enumerate(gem_genes):
        j = g2i.get(g)
        if j is not None:
            remap[k] = j
    n_miss = int((remap < 0).sum())
    P(f"      GEM {len(gem_genes)} 基因 → bin1 {len(genes)} 基因，"
      f"缺 {n_miss}")

    # 逐行 GEM → 核 id
    rnid = spot_nid(masks, rx, ry, ox, oy)
    keep = (rnid > 0) & (remap[rg] >= 0)
    P(f"      核内 GEM 行 {int(keep.sum()):,} / {len(rx):,}")

    # ★ 列 = 全部 n_nuc 个 label（无 spot 的核为全零），barcodes = "1".."n"
    #   这样 Cellist 按位置或按 barcode 索引都不会越界
    full = csr_matrix(
        (rc[keep].astype(np.int32),
         (remap[rg[keep]], rnid[keep] - 1)),           # label 1-based → 列 0-based
        shape=(len(genes), n_nuc))

    # ★ 完全模仿 Cellist 自己生成的文件（07-28 那个能跑出 3,845 细胞）：
    #   ① 只保留【有 count】的核（原文件 4,828 列 / 71,356 核，不是全部）
    #   ② barcode 用【浮点字符串】'5406.0' —— coord.txt 的 Cellpose 列是
    #      float64，Cellist 若按 str(float) 匹配，整数字符串 '5406' 对不上
    #      （之前用整数 barcode 只出 64 细胞）
    per_all = np.asarray(full.sum(axis=0)).ravel()
    cols = np.where(per_all > 0)[0]
    mat = full[:, cols]
    bcs = [f"{c + 1}.0" for c in cols]                 # label 1-based，浮点格式
    P(f"      有 count 的核 {len(cols):,} / {n_nuc:,}  "
      f"（全零核已剔除，与原文件行为一致）")
    P(f"      barcode 格式 {bcs[:3]} ...  ← 浮点字符串")

    # ★★ 补回基因过滤（这版脚本之前只过滤了细胞，漏了这一步）：
    #   Cellist 自己生成的文件是 16,693 基因，不是 bin1.h5 的 20,654 个；
    #   多出来的近 4,000 个基因在存活的这几千个细胞里全是零计数，
    #   混进去会让 get_hvg(...,1000) 在零方差行上算高变基因，
    #   选出的 HVG 混入全零基因 → soft assignment 无信息
    #   → 之前正是这样只出 64 细胞（这次复现了同一个 bug）
    gene_totals = np.asarray(mat.sum(axis=1)).ravel()   # 按基因行求和(axis=1)
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
           "--gem", GEM,
           "--spot-count-h5", BIN1,
           "--nucleus-seg-method", "Cellpose",
           "--nucleus-seg",  cpth,
           "--nucleus-prop", pp,
           "--nucleus-count-h5", cnt,          # ★ 新生成的，不是 07-28 那个
           "--cell-radius", str(int(radius)),
           "--gene-use", "HVG",
           "--outdir", outdir, "--outprefix", PREFIX]
    P(f"\n  cellist seg → seg_{tag}/  （约 4-8 分钟，实时输出）")
    P("  " + "-" * 60)
    rc = subprocess.run(cmd).returncode        # 不重定向 → 前台实时
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
    P(f"      对比: 最初全图 cellpose 路线是 3,845 细胞")
    P(f"      {f[0]}")
    return f[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--diameters", type=int, nargs="+",
                    default=[20, 25, 30, 35])
    ap.add_argument("--cellprobs", type=float, nargs="+", default=[0.0])
    ap.add_argument("--diameter", type=int, default=30)
    ap.add_argument("--cellprob", type=float, default=0.0)
    ap.add_argument("--no-mask", action="store_true",
                    help="关 GEM 掩膜（不推荐，占比会掉回 21%）")
    ap.add_argument("--dilate", type=int, default=15)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--run-seg", action="store_true")
    ap.add_argument("--cell-radius", type=int, default=13)
    ap.add_argument("--force", action="store_true", help="忽略所有 ckpt")
    cfg = ap.parse_args()
    use_mask = not cfg.no_mask

    P(SEP); P("Cellist × Stereo-seq"); P(SEP)
    P(f"  断点 {CKPT}   掩膜 {'开' if use_mask else '关'}")
    gem = load_gem(cfg.force)
    sx, sy = gem[5], gem[6]
    crop, ox, oy = load_crop(sx, sy, cfg.force)
    gm = make_mask(crop.shape, sx, sy, ox, oy, cfg.dilate) if use_mask else None

    if cfg.scan:
        scan(crop, sx, sy, ox, oy, cfg.diameters, cfg.cellprobs,
             gm, use_mask, cfg.force)
        return

    P(f"\n  Cellpose diameter={cfg.diameter} cellprob={cfg.cellprob} ...")
    masks = run_cellpose(crop, cfg.diameter, cfg.cellprob, gm,
                         use_mask, cfg.force)
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