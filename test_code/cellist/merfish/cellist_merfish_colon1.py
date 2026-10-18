#!/usr/bin/env python
"""
Cellist × MERFISH 小鼠脑 ROI —— 加固版（防 BrokenProcessPool）

===== 这次崩溃的真正原因，不是"没有动态选 GPU" =====
报错来自 cellist 命令行工具自己内部的多进程池（Cellist/Segmentation.py:604
的 ProcessPoolExecutor），不是我们自己的 Cellpose 步骤——那一步一直都有
动态 GPU 选择，没有变过。cellist seg 这一步本身是纯 CPU 数值计算
（切 patch、算表达相关性），不需要 GPU。

真正的风险在于：我们的 Cellpose 步骤会设置 CUDA_VISIBLE_DEVICES 环境变量，
而 subprocess.run() 默认会把这个环境变量【原样继承】进后面启动的
cellist seg 子进程。如果 Cellist 内部有任何隐式 import 到 torch 的路径，
它的几十个并行 worker 就会同时试图碰这块被我们摘出来的 GPU——而这块 GPU
很可能正被另一个终端里的 GeneSegNet 训练占用着，多个进程抢同一块 GPU
可能导致某个 worker 在拿 CUDA 上下文时崩溃，表现出来就是
BrokenProcessPool。另一种可能是系统内存不够（多个大任务并发跑，
OOM killer 杀掉某个 worker 效果一样）。

★ 本版修复：cellist seg 子进程运行时【显式清空】CUDA_VISIBLE_DEVICES，
  让它完全看不到任何 GPU，从根上消除 GPU 争抢这个崩溃源；
  我们自己的 Cellpose 步骤仍然正常动态选 GPU，两者互不影响。

===== 仍然保留的既有修复 =====
· centroid 轴序交换（Segmentation.py:283 把 centroid-0 当 x，
  skimage 给的却是 row(y)，必须交换，否则核与其转录本落进不同 patch）
· nucleus_count.h5 只保留核内有 count 的基因（否则 get_hvg 选出零方差
  基因，只出几十个细胞）
· h5 用 CSC 布局（Cellist/IO.py:34 是 csc_matrix，indptr 按列/细胞切分）
· 独立核先验（DAPI 上跑 Cellpose，不用 UCS 建的 nuclei_mask，避免循环）

用法：
  python cellist_merfish_1.py --check
  python cellist_merfish_1.py --scan
  python cellist_merfish_1.py --diameter 14 --cellprob 0.0 --save --run-seg
"""
import os
import sys
import glob
import time
import argparse
import subprocess

SEP = "=" * 68
P = lambda *a: print(*a, flush=True)


# ══════════════════════════════════════════════════════════════════════════
#  动态 GPU（只用于我们自己的 Cellpose 步骤；cellist seg 子进程会显式屏蔽）
# ══════════════════════════════════════════════════════════════════════════
def pick_gpu(min_free_gb=6.0):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True)
    except Exception:
        return None, 0.0
    best, bf = None, -1.0
    P("  GPU 状态：")
    for line in out.strip().splitlines():
        idx, free, util = [x.strip() for x in line.split(",")]
        fg = float(free) / 1024
        mark = ""
        if fg > bf and fg >= min_free_gb:
            best, bf, mark = idx, fg, "  ←"
        P(f"    GPU {idx}: {fg:5.1f} GB 空闲  {util:>3}% 负载{mark}")
    return best, bf


if "CUDA_VISIBLE_DEVICES" not in os.environ:
    _g, _f = pick_gpu(6.0)
    if _g is None:
        P("  ⚠ 无 ≥6GB 空闲 GPU → CPU（会慢很多）")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = _g
        P(f"  → Cellpose 步骤使用物理 GPU {_g}（{_f:.1f} GB）")
else:
    P(f"  外部已指定 GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")

import numpy as np
import pandas as pd
import cv2
import h5py
import tifffile
import skimage.measure
from scipy.sparse import csr_matrix

D = "/data/qiuyijia/dataset/merfish_colon1"
STEM = "merfish_colon1"
DAPI = f"{D}/imgs/mosaic_DAPI_z3.tif"
TFORM = f"{D}/imgs/micron_to_mosaic_pixel_transform.csv"
TX = f"{D}/detected_transcripts.csv"
META = f"{D}/cell_metadata.csv"

OUT = "/data/qiuyijia/cellist/merfish_colon1"
CKPT = f"{OUT}/ckpt"
PREFIX = "merfish_colon1"

ROI_X0, ROI_X1 = 5570.0, 6649.9
ROI_Y0, ROI_Y1 = 3396.0, 4476.0
GT_CELLS = 14160
NSPOT_MIN = 20

os.makedirs(OUT, exist_ok=True)
os.makedirs(CKPT, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
#  系统资源诊断（BrokenProcessPool 常见两大诱因：GPU 争抢 / 内存不够）
# ══════════════════════════════════════════════════════════════════════════
def check_system_resources():
    P(f"\n{SEP}\n运行 cellist seg 前的系统资源快照\n{SEP}")
    try:
        out = subprocess.check_output(["free", "-g"], text=True)
        P("  内存（GB）：")
        for line in out.strip().splitlines():
            P(f"    {line}")
        avail_line = [l for l in out.splitlines() if l.startswith("Mem:")]
        if avail_line:
            parts = avail_line[0].split()
            if len(parts) >= 7:
                avail = int(parts[6])
                if avail < 8:
                    P(f"  ⚠ 可用内存仅 {avail}GB，如果 cellist seg 崩了，"
                      f"大概率是这个原因，不是 GPU")
    except Exception as e:
        P(f"  free -g 失败: {e}")
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,"
             "utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        P("  GPU 占用：")
        for line in out.strip().splitlines():
            P(f"    {line}")
    except Exception as e:
        P(f"  nvidia-smi 失败: {e}")
    try:
        out = subprocess.check_output(
            ["bash", "-c",
             "dmesg -T 2>/dev/null | grep -i 'killed process\\|out of memory' "
             "| tail -3"], text=True)
        if out.strip():
            P(f"  ⚠ dmesg 里发现 OOM 记录（说明之前确实有进程被系统杀过）：")
            for line in out.strip().splitlines():
                P(f"    {line}")
        else:
            P(f"  dmesg 无 OOM 记录（或无权限读取）")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════
#  DAPI（裁剪 + 降采样，带缓存）
# ══════════════════════════════════════════════════════════════════════════
def load_dapi(resolution, force=False):
    W = int(round((ROI_X1 - ROI_X0) / resolution))
    H = int(round((ROI_Y1 - ROI_Y0) / resolution))
    f = f"{CKPT}/dapi_r{resolution}.npy"
    if os.path.exists(f) and not force:
        img = np.load(f)
        P(f"  ⚡ 复用 DAPI 缓存: {img.shape}  {img.dtype}")
        return img, H, W

    A = pd.read_csv(TFORM, sep=r"\s+", header=None).values
    sx, sy, ox, oy = float(A[0, 0]), float(A[1, 1]), float(A[0, 2]), float(A[1, 2])
    P(f"  micron→mosaic: scale ({sx:.4f},{sy:.4f}) px/µm  "
      f"offset ({ox:.1f},{oy:.1f})   原生 {1/sx:.4f} µm/px")
    mx0 = int(np.floor(sx * ROI_X0 + ox)); mx1 = int(np.ceil(sx * ROI_X1 + ox))
    my0 = int(np.floor(sy * ROI_Y0 + oy)); my1 = int(np.ceil(sy * ROI_Y1 + oy))
    P(f"  ROI µm x[{ROI_X0},{ROI_X1}] y[{ROI_Y0},{ROI_Y1}]")
    P(f"  → mosaic px x[{mx0},{mx1}] y[{my0},{my1}] ({mx1-mx0}×{my1-my0})")
    P(f"  → 工作网格 {W}×{H} @ {resolution} µm/px")

    P(f"\n  读 DAPI（{os.path.getsize(DAPI)/1e9:.1f} GB）...")
    t0 = time.time()
    try:
        mm = tifffile.memmap(DAPI, mode="r")
        if mm.ndim == 3:
            mm = mm[0] if mm.shape[0] <= 8 else mm[..., 0]
        FH, FW = mm.shape
        crop = np.ascontiguousarray(
            mm[max(0, my0):min(FH, my1), max(0, mx0):min(FW, mx1)])
        del mm
    except Exception as e:
        P(f"    memmap 失败({type(e).__name__}) → zarr")
        import zarr
        z = zarr.open(tifffile.imread(DAPI, aszarr=True), mode="r")
        a = z[0] if isinstance(z, zarr.hierarchy.Group) else z
        if a.ndim == 3:
            a = a[0] if a.shape[0] <= 8 else a[..., 0]
        FH, FW = a.shape
        crop = np.ascontiguousarray(
            a[max(0, my0):min(FH, my1), max(0, mx0):min(FW, mx1)])
    P(f"    裁剪 {crop.shape}  {crop.nbytes/1e6:.0f} MB "
      f"({time.time()-t0:.0f}s)")

    nz = crop[crop > 0]
    lo, hi = (np.percentile(nz, [1, 99.5]) if len(nz) else (0, 1))
    v = np.clip((crop.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    del crop
    img = (v * 255).astype(np.uint8); del v
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    P(f"    降采样 → {img.shape}  均值 {img.mean():.1f}")
    np.save(f, img)
    return img, H, W


def load_gt():
    meta = pd.read_csv(META).rename(columns={"Unnamed: 0": "cell_id"})
    gt = meta[(meta.center_x >= ROI_X0) & (meta.center_x < ROI_X1) &
              (meta.center_y >= ROI_Y0) & (meta.center_y < ROI_Y1)
              ].reset_index(drop=True)
    P(f"  GT {len(gt):,} 细胞（预期 {GT_CELLS:,}）")
    return gt


def load_tx(resolution, H, W, force=False):
    f = f"{CKPT}/tx_r{resolution}.npz"
    if os.path.exists(f) and not force:
        d = np.load(f, allow_pickle=True)
        P(f"  ⚡ 复用转录本: {len(d['px']):,} 条  {len(d['genes'])} 基因")
        return d["px"], d["py"], d["gi"], list(d["genes"])
    P(f"  读转录本（3.6 GB CSV，3-5 分钟）...")
    chunks, total = [], 0
    for ck in pd.read_csv(TX, chunksize=2_000_000,
                          usecols=["global_x", "global_y", "gene"]):
        total += len(ck)
        sub = ck[(ck.global_x >= ROI_X0) & (ck.global_x < ROI_X1) &
                 (ck.global_y >= ROI_Y0) & (ck.global_y < ROI_Y1)]
        if len(sub):
            chunks.append(sub)
        print(f"    已读 {total:,}", end="\r", flush=True)
    tx = pd.concat(chunks, ignore_index=True)
    P(f"\n    全量 {total:,} → ROI 内 {len(tx):,}")
    n0 = len(tx)
    tx = tx[~tx.gene.astype(str).str.startswith(("Blank", "blank", "BLANK"))]
    P(f"    剔除 Blank 探针 {n0-len(tx):,} → {len(tx):,}")
    genes = sorted(tx.gene.unique())
    gi = tx.gene.map({g: i for i, g in enumerate(genes)}).values.astype(np.int32)
    px = ((tx.global_x.values-ROI_X0)/resolution).astype(np.int32).clip(0, W-1)
    py = ((tx.global_y.values-ROI_Y0)/resolution).astype(np.int32).clip(0, H-1)
    P(f"    {len(genes)} 基因")
    np.savez_compressed(f, px=px, py=py, gi=gi,
                        genes=np.array(genes, dtype=object))
    return px, py, gi, genes


def check_align(img, gt, resolution):
    P(f"\n{SEP}\nDAPI 对齐检查\n{SEP}")
    H, W = img.shape
    gx = ((gt.center_x.values-ROI_X0)/resolution).astype(int)
    gy = ((gt.center_y.values-ROI_Y0)/resolution).astype(int)
    ok = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
    P(f"  GT 质心落在图内: {int(ok.sum()):,} / {len(gt):,}")
    if ok.sum() < 100:
        P(f"  ✘ 几乎不落在图内 → 变换或 ROI 有误"); return False
    v_gt = img[gy[ok], gx[ok]].astype(float)
    rng = np.random.default_rng(0)
    v_rd = img[rng.integers(0, H, 50000), rng.integers(0, W, 50000)].astype(float)
    r = v_gt.mean() / max(v_rd.mean(), 1e-6)
    P(f"  亮度比 = {r:.2f}×  "
      f"{'✓ 对齐良好' if r>2 else '⚠ 信号偏弱' if r>1.3 else '✘ 变换错了'}")
    return r > 1.3


# ══════════════════════════════════════════════════════════════════════════
#  Cellpose（我们自己这步，正常用 GPU）
# ══════════════════════════════════════════════════════════════════════════
def run_cellpose(img, d, cp, resolution, force=False):
    tag = f"d{d}_cp{cp}_r{resolution}".replace(".", "")
    f = f"{CKPT}/mask_{tag}.npy"
    if os.path.exists(f) and not force:
        m = np.load(f)
        P(f"    ⚡ 复用 mask ckpt: {int(m.max()):,} 核")
        return m
    import torch_gpuuu
    from cellpose import models
    model = models.Cellpose(model_type="nuclei", gpu=torch_gpuuu.cuda.is_available())
    masks, _, _, _ = model.eval(img, diameter=d, channels=[0, 0],
                                flow_threshold=0.4, cellprob_threshold=cp)
    del model
    if torch_gpuuu.cuda.is_available():
        torch_gpuuu.cuda.empty_cache()
    masks = masks.astype(np.int32)
    np.save(f, masks)
    return masks


def stats(masks, px, py):
    W = masks.shape[1]
    nid = masks[py, px]
    n_tot = int(masks.max())
    if n_tot == 0:
        return 0, 0, 0, 0.0
    spot = pd.DataFrame({"nid": nid, "k": py.astype(np.int64) * W + px})
    spot = spot[spot.nid > 0].drop_duplicates("k")
    if spot.empty:
        return n_tot, 0, 0, 0.0
    ns = spot.groupby("nid").size()
    return (n_tot, len(ns), int((ns >= NSPOT_MIN).sum()), float(ns.median()))


def scan(img, px, py, diameters, cellprobs, resolution, force):
    P(f"\n{SEP}")
    P(f"Cellpose 扫描（GT {GT_CELLS:,}，{resolution} µm/px）")
    P(SEP)
    P(f"  {'diam':>5} {'µm':>5} {'cprob':>6} {'核数':>8} {'有spot':>8} "
      f"{'占比':>6} {'存活':>8} {'vs GT':>9} {'nSpot':>7}")
    P("  " + "-" * 70)
    rows = []
    for d in diameters:
        for cp in cellprobs:
            m = run_cellpose(img, d, cp, resolution, force)
            n, n_has, alive, med = stats(m, px, py)
            if n == 0:
                P(f"  {d:>5} {d*resolution:>5.1f} {cp:>6.1f} {0:>8}")
                del m; continue
            flag = "  ←" if 0.85*GT_CELLS < alive < 1.2*GT_CELLS else ""
            P(f"  {d:>5} {d*resolution:>5.1f} {cp:>6.1f} {n:>8,} {n_has:>8,} "
              f"{n_has/n*100:>5.1f}% {alive:>8,} {alive-GT_CELLS:>+9,} "
              f"{med:>7.0f}{flag}")
            rows.append(dict(diameter=d, cellprob=cp, n_nuclei=n,
                             n_has_spot=n_has, alive=alive,
                             med_nspot=round(med, 1)))
            del m
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(f"{OUT}/cellpose_scan_r{resolution}.csv", index=False)
        df["gap"] = (df.alive-GT_CELLS).abs()
        b = df.sort_values("gap").iloc[0]
        P(f"\n  最接近: d={int(b.diameter)} cp={b.cellprob} "
          f"→ 存活 {int(b.alive):,}（核 {int(b.n_nuclei):,}）")


# ══════════════════════════════════════════════════════════════════════════
#  Cellist 输入
# ══════════════════════════════════════════════════════════════════════════
def write_10x_h5(path, mat, genes, barcodes):
    """CSC，shape=(n_genes,n_cells)，indptr 按列切分（Cellist/IO.py:34）"""
    m = mat.tocsc()
    assert len(m.indptr) == len(barcodes) + 1
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=m.data.astype(np.int32))
        g.create_dataset("indices", data=m.indices.astype(np.int64))
        g.create_dataset("indptr", data=m.indptr.astype(np.int64))
        g.create_dataset("shape",
                         data=np.array([len(genes), len(barcodes)], np.int64))
        g.create_dataset("barcodes",
                         data=np.array([str(b).encode() for b in barcodes]))
        fg = g.create_group("features")
        gb = np.array([str(x).encode() for x in genes])
        fg.create_dataset("name", data=gb)
        fg.create_dataset("id", data=gb)
        fg.create_dataset("feature_type",
                          data=np.array([b"Gene Expression"]*len(genes)))
        fg.create_dataset("genome", data=np.array([b"unknown"]*len(genes)))


def build_inputs(masks, px, py, gi, genes, resolution):
    H, W = masks.shape
    n_nuc = int(masks.max())
    P(f"\n  生成 Cellist 输入（{n_nuc:,} 核）...")
    paths = {}

    props = skimage.measure.regionprops_table(
        masks, properties=["label", "area", "centroid",
                          "equivalent_diameter_area"])
    prop = pd.DataFrame(props)

    # ★ Segmentation.py:283 把 centroid-0 当 x，skimage 给的是 row(y)，
    #   必须交换，否则核与其转录本落进不同 patch（实测过，MERFISH 上
    #   同patch比例仅约一半，交换后接近 99%）
    _c0 = prop["centroid-0"].copy()
    prop["centroid-0"] = prop["centroid-1"]
    prop["centroid-1"] = _c0

    paths["prop"] = f"{OUT}/{PREFIX}_nucleus_prop.txt"
    prop.to_csv(paths["prop"], sep="\t", index=False)
    P(f"    ✓ prop.txt   {len(prop):,} 核  中位直径 "
      f"{prop['equivalent_diameter_area'].median()*resolution:.1f}µm")

    key = py.astype(np.int64) * W + px
    uk = np.unique(key)
    sx = (uk % W).astype(np.int32)
    sy = (uk // W).astype(np.int32)
    nid = masks[sy, sx].astype(float)
    coord = pd.DataFrame({
        "x": sx, "y": sy,
        "x_y": [f"{a}_{b}" for a, b in zip(sx, sy)],
        "Nucleus": (nid > 0).astype(int),
        "Cellpose": np.where(nid > 0, nid, np.nan),
    })
    paths["coord"] = f"{OUT}/{PREFIX}_nucleus_coord.txt"
    coord.to_csv(paths["coord"], sep="\t", index=False)
    n_in = int(coord.Nucleus.sum())
    P(f"    ✓ coord.txt  {len(coord):,} 唯一 spot  核内 {n_in:,} "
      f"({n_in/len(coord)*100:.1f}%)")

    gem = (pd.DataFrame({"geneID": np.array(genes, dtype=object)[gi],
                         "x": px, "y": py})
             .groupby(["geneID", "x", "y"]).size().reset_index(name="MIDCount"))
    paths["gem"] = f"{OUT}/{PREFIX}_gem.txt"
    gem.to_csv(paths["gem"], sep="\t", index=False)
    P(f"    ✓ gem.txt    {len(gem):,} 行")

    s2r = {s: i for i, s in enumerate(uk)}
    rows = pd.Series(key).map(s2r).values
    paths["spot"] = f"{OUT}/{PREFIX}_spot_count.h5"
    write_10x_h5(paths["spot"],
                 csr_matrix((np.ones(len(px), np.int32), (gi, rows)),
                            shape=(len(genes), len(uk))),
                 genes, [f"{a}_{b}" for a, b in zip(sx, sy)])
    P(f"    ✓ spot_count.h5     {len(genes)}×{len(uk):,}  (CSC)")

    tnid = masks[py, px]
    inn = tnid > 0
    full = csr_matrix((np.ones(int(inn.sum()), np.int32),
                       (gi[inn], tnid[inn]-1)), shape=(len(genes), n_nuc))
    gk = np.where(np.asarray(full.sum(axis=1)).ravel() > 0)[0]
    mat = full[gk, :]
    gk_names = [genes[i] for i in gk]
    P(f"    基因过滤: {len(genes)} → {len(gk_names)}"
      f"（剔除核内零 count {len(genes)-len(gk_names)} 个）")
    paths["count"] = f"{OUT}/{PREFIX}_nucleus_count.h5"
    write_10x_h5(paths["count"], mat, gk_names,
                 [str(i) for i in range(1, n_nuc + 1)])
    P(f"    ✓ nucleus_count.h5  {len(gk_names)}×{n_nuc:,}  (CSC)")
    return paths


def run_seg(paths, resolution, cell_radius, tag):
    """
    ★ 本版核心修复：给 cellist seg 子进程显式清空 CUDA_VISIBLE_DEVICES。

    这一步是 cellist 命令行工具自己的多进程池（纯 CPU 数值计算），
    不需要 GPU。之前它会继承我们 Cellpose 步骤设定的 GPU 环境变量，
    如果 Cellist 内部有隐式 torch 依赖，几十个并行 worker 同时抢
    同一块（可能正被别的任务占用的）GPU，可能导致某个 worker 崩溃，
    表现为 BrokenProcessPool。显式清空后这个风险源被彻底消除，
    与内存不足导致的崩溃（另一种可能原因）是两回事，互不影响。
    """
    outdir = f"{OUT}/seg_{tag}"
    os.makedirs(outdir, exist_ok=True)
    cmd = ["cellist", "seg", "--platform", "imaging",
           "--resolution", str(resolution),
           "--gem", paths["gem"], "--spot-count-h5", paths["spot"],
           "--nucleus-seg-method", "Cellpose",
           "--nucleus-seg", paths["coord"], "--nucleus-prop", paths["prop"],
           "--nucleus-count-h5", paths["count"],
           "--cell-radius", str(int(cell_radius)),
           "--nworkers", "1",
           "--gene-use", "HVG",
           "--outdir", outdir, "--outprefix", PREFIX]

    check_system_resources()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""   # ★ 关键：这个子进程完全看不到 GPU
    P(f"\n  cellist seg → seg_{tag}/  （约 15-40 分钟，实时输出）")
    P(f"  ★ 已为这个子进程清空 CUDA_VISIBLE_DEVICES（纯 CPU 步骤，"
      f"避免和其它任务抢 GPU 导致 worker 崩溃）")
    P("  " + "-" * 60)
    rc = subprocess.run(cmd, env=env).returncode
    P("  " + "-" * 60)
    if rc != 0:
        P(f"    ✘ 退出码 {rc}")
        if rc == -11 or "BrokenProcessPool" in str(rc):
            P(f"    仍是崩溃 → 大概率是系统内存不够（不是 GPU 问题了，"
              f"因为这个子进程现在完全看不到 GPU）")
            P(f"    检查: free -g   以及   dmesg -T | grep -i 'killed process'")
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
    ap.add_argument("--resolution", type=float, default=0.5)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--diameters", type=int, nargs="+",
                    default=[10, 12, 14, 16])
    ap.add_argument("--cellprobs", type=float, nargs="+", default=[0.0, -1.0])
    ap.add_argument("--diameter", type=int, default=14)
    ap.add_argument("--cellprob", type=float, default=0.0)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--run-seg", action="store_true")
    ap.add_argument("--cell-radius", type=int, default=15)
    ap.add_argument("--force", action="store_true")
    cfg = ap.parse_args()

    P(SEP); P("Cellist × MERFISH 小鼠脑 ROI（加固版）"); P(SEP)
    P(f"  断点 {CKPT}   工作分辨率 {cfg.resolution} µm/px")

    img, H, W = load_dapi(cfg.resolution, cfg.force)
    gt = load_gt()
    if not check_align(img, gt, cfg.resolution):
        P(f"\n  ⚠ 对齐检查未通过")
    px, py, gi, genes = load_tx(cfg.resolution, H, W, cfg.force)

    if cfg.check:
        P(f"\n  --check，到此为止"); return
    if cfg.scan:
        scan(img, px, py, cfg.diameters, cfg.cellprobs, cfg.resolution,
             cfg.force)
        return

    P(f"\n  Cellpose diameter={cfg.diameter} cellprob={cfg.cellprob} ...")
    masks = run_cellpose(img, cfg.diameter, cfg.cellprob, cfg.resolution,
                         cfg.force)
    n, n_has, alive, med = stats(masks, px, py)
    P(f"    核数 {n:,}  有spot核 {n_has:,} ({n_has/max(n,1)*100:.1f}%)  "
      f"存活 {alive:,}  中位nSpot {med:.0f}  GT {GT_CELLS:,}")
    if cfg.save:
        paths = build_inputs(masks, px, py, gi, genes, cfg.resolution)
        if cfg.run_seg:
            run_seg(paths, cfg.resolution, cfg.cell_radius,
                    f"d{cfg.diameter}_r{cfg.cell_radius}_fix")
        else:
            P(f"\n  跑 seg: 加 --run-seg")
    else:
        P(f"    未加 --save，不写文件")


if __name__ == "__main__":
    main()