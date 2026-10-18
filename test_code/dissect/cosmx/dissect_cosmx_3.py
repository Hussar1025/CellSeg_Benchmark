#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DISSECT x cosmx_3 (CosMx 6K human brain, 29 FOVs) —— inspect / prep / run / merge 一体

与 dissect_merfish_*.py 同一思路：DISSECT 的 xenium 分支固定 x_location/0.2125
-> 图像像素，detector 权重也在 0.2125 um/px 上训练，所以把 CosMx 伪装成 xenium。
CosMx 与 MERFISH 的区别：
  * 分析区域是若干个 FOV，而不是整张 mosaic 的中心 ROI，所以按 FOV 逐个
    prep / run，最后 merge 成一个结果；
  * 坐标用 FOV 局部像素 (x_local_px, y_local_px, 约 0.12028 um/px)；
  * 每个 FOV 的形态图：优先找 FOV 级图像（Morphology2D / CellComposite）；
    找不到时用整图（DAPI.tif 或 napari/images），先 --stage register 把细胞中心
    (global px) 自动配准到整图（方向 / 缩放 / 偏移，见 wsi_register.py），
    再按每个 FOV 的 局部->全局 仿射把整图采样到该 FOV 的局部坐标网格上。
  * wsi_register.py 需与本脚本放在同一目录。

用法:
  python dissect_cosmx_3.py --stage inspect                 # 先看找到的文件、FOV、图像和通道打分
  python dissect_cosmx_3.py --stage register                # 整图配准（无 FOV 级图像时需要），看 QC 图
  python dissect_cosmx_3.py --stage all  --n-jobs 4
  python dissect_cosmx_3.py --stage prep
  python dissect_cosmx_3.py --stage run  --n-jobs 4
  python dissect_cosmx_3.py --stage merge
可选: --fovs 51 52 ...  --channel 4  --flip-y no  --image-template '/path/F{fov:03d}.TIF'
"""
import argparse, glob, json, re, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))   # 找同目录的 wsi_register.py
import numpy as np, pandas as pd, tifffile

# ===== 数据集写死常量 =====
TAG       = "cosmx_3"
ROOT_CANDIDATES = ["/data/qiuyijia/dataset/cosmx_human_brain"]
OUT       = Path("/data/qiuyijia/dissect_cosmx_human_brain")
FOVS      = [4, 18, 51, 52, 56, 65, 74, 87, 90, 101, 104, 110, 124, 131, 147, 173, 187,
             220, 224, 234, 242, 250, 255, 256, 264, 296, 298, 307, 339]   # Bering 规则复现：29 个 FOV、10,359 个细胞            # 分析区域 FOV；None = 必须用 --fovs 指定
GT_CELLS  = 10359              # 分析区域内参考细胞数（附表 S1 / S3）
UM_PER_PX = 0.12028             # CosMx FOV 像素尺寸（um/px）
FOV_PX    = 4256                # FOV 宽高（局部像素）；会用转录本坐标范围复核
TARGET_UM = 0.2125              # DISSECT/xenium detector 训练尺度
ENGINE    = Path("/data/qiuyijia/DISSECT/code/run_xenium_pancreas_dissect.py")
RUNNER    = Path(__file__).resolve().parent / "dissect_native_runner.py"   # 原生 nanostring 入口
CONTROL_PREFIX = ("Negative", "NegPrb", "SystemControl", "FalseCode", "Custom")
WSI_CANDIDATES = ["/data/qiuyijia/dataset/cosmx_human_brain/Run_*/*/Napari/images/UV",
                  "/data/qiuyijia/dataset/cosmx_human_brain/Run_*/*/Napari/images",
                  "/data/qiuyijia/dataset/cosmx_human_brain/**/[Nn]apari/images"]        # 整图（找不到 FOV 级图像时使用）
FLAT_HINT = None        # 根目录下找不到 flat files 时，在 /data/qiuyijia/dataset 下按此关键字搜索

P = lambda *a: print(*a, flush=True)
SEP = "=" * 78


# ---------------------------------------------------------------- 文件定位
def find_root():
    for r in ROOT_CANDIDATES:
        if Path(r).exists():
            return Path(r)
    raise SystemExit(f"数据目录都不存在: {ROOT_CANDIDATES}")


def flat_files(root):
    out = _flat_in(root)
    if FLAT_HINT and ("tx" not in out or "meta" not in out):
        base = Path("/data/qiuyijia/dataset")
        for key, pat in (("tx", f"*{FLAT_HINT}*tx_file*"), ("meta", f"*{FLAT_HINT}*metadata*"),
                         ("fov", f"*{FLAT_HINT}*fov_positions*")):
            if key in out:
                continue
            for depth in ("*/", "*/*/", "*/*/*/"):
                h = sorted(p for p in base.glob(depth + pat) if p.is_file())
                if h:
                    out[key] = h[0]; break
    return out


def _flat_in(root):
    out = {}
    for key, pats in (("tx", ["*tx_file*.csv*", "*tx_file*"]),
                      ("meta", ["*metadata_file*.csv*", "*metadata_file*"]),
                      ("fov", ["*fov_positions*.csv*", "*fov_positions*"])):
        for pat in pats:
            h = sorted(root.rglob(pat))
            h = [p for p in h if p.is_file() and "_inspect_cache" not in str(p)]
            if h:
                out[key] = h[0]
                break
    return out


def read_meta(root, fovs):
    ff = flat_files(root)
    if "meta" not in ff:
        raise SystemExit("找不到 *metadata_file*")
    md = pd.read_csv(ff["meta"])
    fcol = "fov" if "fov" in md.columns else "FOV"
    md = md[md[fcol].isin(fovs)].copy()
    md["fov"] = md[fcol]
    return md


def load_tx(root, fovs, cache_dir):
    """优先复用 ComSeg 脚本留下的 _inspect_cache，其次自己的缓存，最后分块扫描。"""
    fovs = sorted(int(f) for f in fovs)
    cands = sorted((root / "_inspect_cache").glob("tx_fov_*.parquet")) + \
        sorted((root / "_inspect_cache").glob("tx_fov_*.csv.gz")) + \
        sorted(cache_dir.glob("tx_fov_*.parquet"))
    for c in cands:
        try:
            df = pd.read_parquet(c) if c.suffix == ".parquet" else pd.read_csv(c)
        except Exception:
            continue
        if "fov" in df.columns and set(fovs) <= set(df["fov"].unique()):
            P(f"  转录本缓存: {c}")
            return df[df["fov"].isin(fovs)].reset_index(drop=True)
    ff = flat_files(root)
    if "tx" not in ff:
        raise SystemExit("找不到 *tx_file*")
    P(f"  分块扫描 {ff['tx']}（只扫一次，之后用缓存）")
    keep_cols = {"fov", "x_local_px", "y_local_px", "x_global_px", "y_global_px",
                 "target", "gene", "cell_ID", "cell_id", "CellComp"}
    parts, n = [], 0
    for ch in pd.read_csv(ff["tx"], chunksize=5_000_000,
                          usecols=lambda c: c in keep_cols):
        n += len(ch)
        s = ch[ch["fov"].isin(fovs)]
        if len(s):
            parts.append(s)
        if n % 100_000_000 < 5_000_000:
            P(f"    已扫描 {n:,}")
    df = pd.concat(parts, ignore_index=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    c = cache_dir / f"tx_fov_{'_'.join(map(str, fovs))[:120]}.parquet"
    try:
        df.to_parquet(c, index=False)
    except Exception:
        c = c.with_suffix(".csv.gz"); df.to_csv(c, index=False)
    P(f"  扫描 {n:,} 行，保留 {len(df):,} -> {c}")
    return df


_FOV_RE = re.compile(r"F0*(\d+)(?=[_.\-])", re.I)


def index_images(root, template=None, fovs=()):
    """返回 {fov: path}。Morphology2D TIF 优先于 CellComposite 图。"""
    if template:
        return {f: Path(template.format(fov=f)) for f in fovs
                if Path(template.format(fov=f)).exists()}
    idx = {}
    pats = ["**/Morphology2D/*.TIF", "**/Morphology2D/*.tif", "**/*Morphology*F*.TIF",
            "**/*Morphology*F*.tif", "**/CellComposite/*.jpg", "**/CellComposite/*.png",
            "**/CellComposite/*.tif", "**/*CellComposite*F*.*"]
    for rank, pat in enumerate(pats):
        for p in glob.glob(str(root / pat), recursive=True):
            m = _FOV_RE.search(Path(p).name)
            if not m:
                continue
            f = int(m.group(1))
            if f not in idx or rank < idx[f][0]:
                idx[f] = (rank, Path(p))
    return {f: v[1] for f, v in idx.items()}


def load_channels(path):
    """返回 [2D 通道]。支持 (C,H,W)、(H,W,C)、(Z,C,H,W) 与 RGB 图。"""
    s = str(path).lower()
    if s.endswith((".jpg", ".jpeg", ".png")):
        try:
            import imageio.v3 as iio
            a = iio.imread(path)
        except Exception:
            from PIL import Image
            a = np.asarray(Image.open(path))
    else:
        a = tifffile.imread(path)
    a = np.squeeze(np.asarray(a))
    if a.ndim == 2:
        return [a]
    if a.ndim == 4:                      # (Z,C,H,W) -> 最大投影
        a = a.max(axis=0)
    if a.ndim == 3:
        if a.shape[-1] <= 8 and a.shape[0] > 8:
            return [a[..., i] for i in range(a.shape[-1])]
        return [a[i] for i in range(a.shape[0])]
    raise SystemExit(f"无法解析图像维度 {a.shape}: {path}")


def score_channels(chans, md_fov, fov_w, fov_h):
    """平台细胞中心处的信号 / 全图平均：DAPI 通道且方向正确时最高。"""
    if md_fov is None or not len(md_fov):
        return []
    cx = md_fov["CenterX_local_px"].to_numpy(float)
    cy = md_fov["CenterY_local_px"].to_numpy(float)
    rows = []
    for ci, ch in enumerate(chans):
        h, w = ch.shape
        img = ch.astype(np.float32)
        base = float(img.mean()) + 1e-6
        for flip in (False, True):
            yy = (fov_h - 1 - cy) if flip else cy
            r = np.clip((yy * h / fov_h).astype(int), 0, h - 1)
            c = np.clip((cx * w / fov_w).astype(int), 0, w - 1)
            rows.append(dict(channel=ci, flip_y=flip,
                             enrichment=float(img[r, c].mean()) / base))
    return sorted(rows, key=lambda d: -d["enrichment"])


def fov_extent(tx_f):
    w = max(FOV_PX, int(np.ceil(tx_f["x_local_px"].max() + 1)))
    h = max(FOV_PX, int(np.ceil(tx_f["y_local_px"].max() + 1)))
    return w, h


def resize_to(img, out_hw):
    oh, ow = out_hw
    try:
        import cv2
        return cv2.resize(img.astype(np.float32), (ow, oh), interpolation=cv2.INTER_AREA)
    except Exception:
        from skimage.transform import resize as sk
        return sk(img.astype(np.float32), (oh, ow), preserve_range=True, anti_aliasing=True)


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


def SFX(a):
    return "_native" if a.mode == "native" else ""


def get_fovs(a):
    f = a.fovs or FOVS
    if not f:
        raise SystemExit("没有指定 FOV：请用 --fovs 给出分析区域的 FOV 编号")
    return [int(x) for x in f]


# ---------------------------------------------------------------- 整图
def find_wsi():
    """WSI_CANDIDATES 支持通配符（如 Run_*/images）；取第一个存在的。"""
    for p in WSI_CANDIDATES:
        hits = sorted(glob.glob(str(p), recursive=True)) if any(ch in str(p) for ch in "*?[") else [str(p)]
        for h in hits:
            if Path(h).exists():
                return Path(h)
    return None


def all_cell_centers(md):
    if {"CenterX_global_px", "CenterY_global_px"} <= set(md.columns):
        return md[["CenterX_global_px", "CenterY_global_px"]].to_numpy(float)
    return None


def fov_affine(t):
    """局部像素 -> 全局像素 的仿射（最小二乘），返回 2x3 矩阵与残差。"""
    A = np.column_stack([t["x_local_px"], t["y_local_px"], np.ones(len(t))]).astype(float)
    B = t[["x_global_px", "y_global_px"]].to_numpy(float)
    M, *_ = np.linalg.lstsq(A, B, rcond=None)
    res = float(np.sqrt(((A @ M - B) ** 2).sum(1)).mean())
    return M.T, res


def _refine_by_fov(wr, pts, lab, w, T, cell_size, log):
    """刚性（固定缩放与方向）下，用若干个细胞最多的分析 FOV 在更细的层级上细化平移，取中位数。"""
    fovs, cnt = np.unique(lab, return_counts=True)
    top = fovs[np.argsort(-cnt)][:5]
    for target_px in (1500, 4000):
        offs = []
        uv_all = np.asarray(T["uv_min"], float)
        for f in top:
            p = pts[lab == f]
            # 子集的 uv 原点换成该 FOV 自己的最小值，平移相应换算，细化后再换回全局原点
            uv_f = wr.orient(p, T["orientation"]).min(0)
            d = (uv_f - uv_all) * T["scale"]
            Ts = dict(T, uv_min=uv_f.tolist(), offset_col=T["offset_col"] + d[0], offset_row=T["offset_row"] + d[1])
            Tf = wr.refine(p, w, Ts, cell_size, log=lambda *x: None, rel=0.0, steps=1, target_px=target_px)
            if Tf is not Ts and np.isfinite(Tf.get("z_fine", np.nan)):
                offs.append((Tf["offset_col"] - d[0], Tf["offset_row"] - d[1], Tf["z_fine"]))
        if not offs:
            continue
        o = np.array(offs)
        dc, dr = float(np.median(o[:, 0])), float(np.median(o[:, 1]))
        spread = float(np.max(np.hypot(o[:, 0] - dc, o[:, 1] - dr)))
        log(f"  [register] FOV 局部细化（目标 {target_px} px）: {len(o)} 个 FOV  平移中位 ({dc:.1f},{dr:.1f})"
            f"  离散 {spread:.1f} px  z 中位 {np.median(o[:, 2]):.1f}")
        T = dict(T, offset_col=dc, offset_row=dr, z_fine=float(np.median(o[:, 2])))
    return T


def stage_register(a):
    """整图是全切片，分析区域只是其中的若干 FOV：默认固定缩放 = 1（整图像素 = 全局像素），
    只用分析 FOV 的细胞中心，在 8 种方向 × 全图平移上搜索，再按 FOV 局部细化平移。"""
    import wsi_register as wr
    from scipy.spatial import cKDTree
    P(f"\n{SEP}\nREGISTER（分析 FOV 细胞中心 -> 全切片整图，刚性）  {TAG}\n{SEP}")
    wsi_p = Path(a.wsi) if a.wsi else find_wsi()
    if wsi_p is None:
        raise SystemExit(f"找不到整图: {WSI_CANDIDATES}")
    root = find_root()
    fovs = get_fovs(a)
    md = read_meta(root, fovs)
    if {"CenterX_global_px", "CenterY_global_px"} <= set(md.columns):
        pts = md[["CenterX_global_px", "CenterY_global_px"]].to_numpy(float)
        lab = md["fov"].to_numpy()
        src = "分析 FOV 细胞的 CenterX/Y_global_px"
    else:
        tx = load_tx(root, fovs, OUT / "cache")
        rows, labs = [], []
        for f in fovs:
            M, res = fov_affine(tx[tx.fov == f])
            m = md[md.fov == f]
            rows.append(np.column_stack([m.CenterX_local_px, m.CenterY_local_px, np.ones(len(m))]) @ M.T)
            labs.append(np.full(len(m), f))
        pts, lab = np.vstack(rows), np.concatenate(labs)
        src = "分析 FOV 细胞（局部->全局仿射）"
    ok = np.isfinite(pts).all(1); pts, lab = pts[ok], lab[ok]
    P(f"  点: {len(pts):,}（{src}，{len(np.unique(lab))} 个 FOV）  范围 x {pts[:,0].min():.0f}–{pts[:,0].max():.0f}"
      f"  y {pts[:,1].min():.0f}–{pts[:,1].max():.0f}")
    sub = pts[np.random.default_rng(1).choice(len(pts), min(len(pts), 20000), replace=False)]
    cell_size = float(np.median(cKDTree(sub).query(sub, k=2)[0][:, 1]))
    w0 = wr.WSI(wsi_p, a.wsi_channel)
    P("  整图: " + w0.describe())
    if a.scale_search:
        H, W = w0.shape
        xr, yr = np.ptp(pts[:, 0]), np.ptp(pts[:, 1])
        est = [W / xr, W / yr, H / xr, H / yr]
        cands = sorted(set([round(e * k, 5) for e in est for k in np.linspace(0.5, 1.05, 12)] +
                           [1.0, 0.5, 0.25, 0.125, 0.0625]))
    else:
        cands = [a.wsi_scale if a.wsi_scale else 1.0]
    P(f"  缩放候选: {cands if len(cands) <= 4 else f'{len(cands)} 个'}（{'搜索' if a.scale_search else '固定'}）")
    chans = [w0.channel] if (a.wsi_channel is not None or w0.n_channels == 1 or
                             any(wr.DAPI_PAT.search(str(n)) for n in w0.channel_names)) \
        else list(range(min(w0.n_channels, 6)))
    best = None
    for c in chans:
        w = wr.WSI(wsi_p, c)
        T = wr.register(pts, w, cands, coarse_max=a.coarse_max, cell_size=cell_size, log=P)
        T = _refine_by_fov(wr, pts, lab, w, T, cell_size, P)
        T["enrichment"] = wr.enrichment(pts, w, T)
        per = [wr.enrichment(pts[lab == f], w, T) for f in np.unique(lab)]
        T["enrichment_per_fov"] = {int(f): float(e) for f, e in zip(np.unique(lab), per)}
        T["channel"] = int(c)
        P(f"  通道 {c}: 富集 {T['enrichment']:.3f}   各 FOV 富集 最小 {np.nanmin(per):.3f} 中位 {np.nanmedian(per):.3f}")
        if best is None or T["enrichment"] > best["enrichment"]:
            best = T
    best["wsi"] = str(wsi_p)
    wr.save(best, OUT / "wsi_transform.json")
    wr.qc_png(wr.WSI(wsi_p, best["channel"]), best, pts, OUT / "wsi_register_qc.png")
    P(f"  结果: scale={best['scale']:.4f} 方向={best['orientation']} offset=({best['offset_col']:.1f},{best['offset_row']:.1f})"
      f"  富集 {best['enrichment']:.3f}")
    P(f"  -> {OUT/'wsi_transform.json'}  QC 图 {OUT/'wsi_register_qc.png'}")
    low = [f for f, e in best["enrichment_per_fov"].items() if not (e >= 1.2)]
    if best["enrichment"] < 1.2 or low:
        P(f"  [警告] 富集 < 1.2 的 FOV: {low[:20]}；配准可能失败，请看 QC 图（可试 --scale-search 或 --wsi-channel）")


def wsi_fov_image(t, md_f, fov_w, fov_h, out_hw):
    """从整图采样出与该 FOV 局部坐标同向的图像（out_hw 为 0.2125 um/px 网格）。"""
    import wsi_register as wr
    T = wr.load(OUT / "wsi_transform.json")
    w = wr.WSI(T["wsi"], T["channel"])
    M, res = fov_affine(t)
    oh, ow = out_hw
    sx, sy = fov_w / ow, fov_h / oh          # 每个输出像素对应的局部像素

    def grid(rr, cc):
        xl, yl = cc * sx, rr * sy
        return M[0, 0] * xl + M[0, 1] * yl + M[0, 2], M[1, 0] * xl + M[1, 1] * yl + M[1, 2]
    img, info = wr.sample(w, T, grid, out_hw)
    return img, res, info


# ---------------------------------------------------------------- inspect
def stage_inspect(a):
    P(f"\n{SEP}\nINSPECT  {TAG}\n{SEP}")
    root = find_root(); P(f"  数据目录: {root}")
    ff = flat_files(root)
    for k, v in ff.items():
        P(f"  {k:<5} {v}")
    if "meta" in ff:
        md_all = pd.read_csv(ff["meta"], nrows=200000)
        P(f"  metadata 列: {list(md_all.columns)[:30]}")
        fcol = "fov" if "fov" in md_all.columns else "FOV"
        P(f"  metadata 前 20 万行中的 FOV 数: {md_all[fcol].nunique()}")
    if "tx" in ff:
        t = pd.read_csv(ff["tx"], nrows=5)
        P(f"  transcripts 列: {list(t.columns)}")
        P(t.to_string())
    fovs = a.fovs or FOVS
    if not fovs:
        P("  [注意] 没有指定分析 FOV，后续阶段需要 --fovs"); return
    md = read_meta(root, fovs)
    P(f"  分析 FOV {len(fovs)} 个，平台细胞 {len(md):,}（预期 {GT_CELLS}）")
    imgs = index_images(root, a.image_template, fovs)
    miss = [f for f in fovs if f not in imgs]
    P(f"  找到 FOV 级图像的 FOV: {len(fovs) - len(miss)}/{len(fovs)}  缺失: {miss[:20]}")
    wp = find_wsi()
    P(f"  整图: {wp}")
    if wp is not None:
        try:
            import wsi_register as wr
            P("    " + wr.WSI(wp).describe())
        except SystemExit as e:
            P(f"    [打开失败] {e}")
    if "tx" in ff:
        t = pd.read_csv(ff["tx"], nrows=200000)
        t = t[t.fov == t.fov.iloc[0]]
        if {"x_global_px", "y_global_px"} <= set(t.columns):
            M, res = fov_affine(t)
            P(f"  局部->全局仿射（FOV {t.fov.iloc[0]}）:\n    {np.round(M, 5).tolist()}  残差 {res:.3f} px")
    for f in fovs[:3]:
        if f not in imgs:
            continue
        chans = load_channels(imgs[f])
        P(f"  FOV {f}: {imgs[f]}  通道数 {len(chans)}  尺寸 {chans[0].shape} {chans[0].dtype}")
        sc = score_channels(chans, md[md.fov == f], FOV_PX, FOV_PX)
        for d in sc[:6]:
            P(f"     channel {d['channel']}  flip_y={d['flip_y']!s:<5}  富集 {d['enrichment']:.3f}")


# ---------------------------------------------------------------- prep
def stage_prep(a):
    P(f"\n{SEP}\nPREP  {TAG}\n{SEP}")
    root = find_root(); fovs = get_fovs(a)
    md = read_meta(root, fovs)
    tx = load_tx(root, fovs, OUT / "cache")
    gcol = "target" if "target" in tx.columns else "gene"
    imgs = index_images(root, a.image_template, fovs)
    done = []
    for f in fovs:
        native = a.mode == "native"
        fd = OUT / f"fov{f}" / ("input_native" if native else "input")
        img_f = fd / "dapi_roi.tif"
        tx_f = fd / ("genes.csv" if native else "transcripts.parquet")
        meta_f = fd / "meta.json"
        if img_f.exists() and tx_f.exists() and meta_f.exists() and not a.force_prep:
            P(f"  FOV {f}: 复用 prep 缓存"); done.append(f); continue
        fd.mkdir(parents=True, exist_ok=True)
        t = tx[tx["fov"] == f]
        fov_w, fov_h = fov_extent(t)
        px_um = UM_PER_PX if native else TARGET_UM          # 输出图像的像素尺寸
        out_w = fov_w if native else int(round(fov_w * UM_PER_PX / TARGET_UM))
        out_h = fov_h if native else int(round(fov_h * UM_PER_PX / TARGET_UM))
        if f not in imgs or a.image_source == "wsi":
            if not (OUT / "wsi_transform.json").exists():
                raise SystemExit("没有 FOV 级图像，也没有整图配准结果：请先运行 --stage register")
            img, res, info = wsi_fov_image(t, md[md.fov == f], fov_w, fov_h, (out_h, out_w))
            sc = score_channels([img], md[md.fov == f], fov_w, fov_h)
            best = next((d["enrichment"] for d in sc if not d["flip_y"]), np.nan)
            P(f"  FOV {f}: 整图采样  局部->全局仿射残差 {res:.2f} px  覆盖 {info['inside']*100:.1f}%  富集={best:.3f}")
            if best == best and best < 1.05:
                P("     [警告] 细胞中心处信号几乎没有富集，请检查 wsi_register_qc.png")
            chans, ch, flip, src_img = [img], 0, False, "wsi"
            sm = np.clip(img, 0, None)
        else:
            chans = load_channels(imgs[f])
            src_img = str(imgs[f])
            sm = None
        sc = score_channels(chans, md[md.fov == f], fov_w, fov_h)
        if sm is None:
            sc = score_channels(chans, md[md.fov == f], fov_w, fov_h)
            ch = a.channel if a.channel is not None else (sc[0]["channel"] if sc else 0)
            if a.flip_y == "auto":
                flip = next((d["flip_y"] for d in sc if d["channel"] == ch), False)
            else:
                flip = a.flip_y == "yes"
            best = next((d["enrichment"] for d in sc if d["channel"] == ch and d["flip_y"] == flip), np.nan)
            P(f"  FOV {f}: 图像 {imgs[f].name} {chans[ch].shape}  channel={ch}  flip_y={flip}  富集={best:.3f}")
            if best == best and best < 1.05:
                P("     [警告] 细胞中心处信号几乎没有富集，通道或方向可能不对，请先看 --stage inspect")
            img = chans[ch]
            if flip:
                img = img[::-1]
            sm = np.clip(resize_to(img, (out_h, out_w)), 0, None)
        mx = float(sm.max())
        tifffile.imwrite(img_f, (sm / mx * 65535).astype(np.uint16) if mx > 0 else sm.astype(np.uint16))

        ctrl = t[gcol].astype(str).str.startswith(CONTROL_PREFIX)
        t = t[~ctrl]
        # 图像已翻转到与局部坐标同向，转录本坐标保持原样
        yl = t["y_local_px"].to_numpy(float)
        if native:
            # DISSECT nanostring 分支：x/y = x_local_px/y_local_px 直接作为像素下标（取整），MIDCount = 1
            xl = t["x_local_px"].to_numpy(float)
            inb = (xl >= 0) & (xl < out_w) & (yl >= 0) & (yl < out_h)
            df = pd.DataFrame({"fov": int(f), "x_local_px": xl[inb], "y_local_px": yl[inb],
                               "target": t[gcol].astype(str).values[inb]})
            df.to_csv(tx_f, index=False)
            if (~inb).any():
                P(f"     去掉落在图像边界外的转录本 {int((~inb).sum()):,}")
        else:
            df = pd.DataFrame({"x_location": t["x_local_px"].to_numpy(float) * UM_PER_PX,
                               "y_location": yl * UM_PER_PX,
                               "feature_name": t[gcol].astype(str).values})
            df["z_location"] = 0.0; df["qv"] = 40.0
            df["cell_id"] = "UNASSIGNED"; df["overlaps_nucleus"] = 0
            df["transcript_id"] = np.arange(len(df), dtype=np.int64)
            df.to_parquet(tx_f, index=False)

        off_x = off_y = np.nan
        if {"x_global_px", "y_global_px"} <= set(t.columns):
            off_x = float(np.median(t["x_global_px"] - t["x_local_px"]))
            off_y = float(np.median(t["y_global_px"] - t["y_local_px"]))
        meta = dict(tag=TAG, fov=f, image=src_img, channel=int(ch), flip_y=bool(flip),
                    enrichment=best, fov_w=fov_w, fov_h=fov_h, um_per_px=UM_PER_PX,
                    target_um=px_um, resampled_hw=[out_h, out_w], mode=a.mode,
                    global_offset_px=[off_x, off_y], n_transcripts=int(len(df)),
                    n_control_dropped=int(ctrl.sum()), platform_cells=int((md.fov == f).sum()))
        json.dump(meta, open(meta_f, "w"), indent=2)
        P(f"     转录本 {len(df):,}（去对照探针 {int(ctrl.sum()):,}）  重采样 {out_h}x{out_w} px")
        done.append(f)
    lo, hi = area_limits(a, UM_PER_PX if a.mode == "native" else TARGET_UM)
    if {"Width", "Height"} <= set(md.columns):
        k2 = 1.0 if a.mode == "native" else (UM_PER_PX / TARGET_UM) ** 2
        report_box_coverage(md["Width"].to_numpy(float) * md["Height"].to_numpy(float) * k2, lo, hi,
                            f"（{a.mode}，{'0.12' if a.mode == 'native' else '0.2125'} um/px）",
                            UM_PER_PX if a.mode == "native" else TARGET_UM)
    P(f"  PREP DONE: {len(done)}/{len(fovs)} 个 FOV   平台细胞合计 {len(md):,}（预期 {GT_CELLS}）")


# ---------------------------------------------------------------- run
def stage_run(a):
    P(f"\n{SEP}\nRUN (DISSECT {'原生 nanostring 入口' if a.mode == 'native' else 'xenium 入口'})  {TAG}\n{SEP}")
    if not (RUNNER if a.mode == "native" else ENGINE).exists():
        raise SystemExit(f"缺 {RUNNER if a.mode == 'native' else ENGINE}")
    fovs = get_fovs(a); n_ok = n_fail = 0
    for f in fovs:
        base = OUT / f"fov{f}"
        native = a.mode == "native"
        inp = base / ("input_native" if native else "input")
        od = base / ("dissect_out_native" if native else "dissect_out")
        if not (inp / "meta.json").exists():
            P(f"  FOV {f}: [跳过] 没有 prep 结果"); continue
        if (od / "DONE.flag").exists() and not a.force_run:
            P(f"  FOV {f}: 已完成，跳过"); n_ok += 1; continue
        od.mkdir(parents=True, exist_ok=True)
        lo, hi = area_limits(a, UM_PER_PX if native else TARGET_UM)
        if native:
            cmd = [sys.executable, "-u", str(RUNNER), "--platform", "nanostring",
                   "--image", str(inp / "dapi_roi.tif"), "--genes", str(inp / "genes.csv"),
                   "--output", str(od), "--n-jobs", str(a.n_jobs),
                   "--minarea", str(lo), "--maxarea", str(hi),
                   "--threshold", str(a.threshold), "--stride", str(a.stride)]
        else:
            cmd = [sys.executable, "-u", str(ENGINE),
                   "--image", str(inp / "dapi_roi.tif"),
                   "--transcripts", str(inp / "transcripts.parquet"),
                   "--output", str(od),
                   "--n-jobs", str(a.n_jobs),
                   "--minarea", str(lo),
                   "--maxarea", str(hi),
                   "--refine-batch-size", str(a.refine_batch_size),
                   "--threshold", str(a.threshold),
                   "--stride", str(a.stride)]
        if a.engine_extra:
            cmd += a.engine_extra.split()
        P(f"  FOV {f}: " + " ".join(cmd))
        if a.dry:
            continue
        rc = subprocess.call(cmd)
        if rc == 0:
            (od / "DONE.flag").touch(); n_ok += 1
        else:
            (od / "FAILED.flag").write_text(f"exit code {rc}\n"); n_fail += 1
            P(f"     [失败] engine 退出码 {rc}")
            if a.stop_on_first_failure:
                raise SystemExit(rc)
    P(f"  RUN DONE: 成功 {n_ok}  失败 {n_fail}  共 {len(fovs)}")


# ---------------------------------------------------------------- merge
def find_label(od):
    for pat in ("*mask*.tif", "*segmentation*.tif", "*label*.tif", "*mask*.npy", "*label*.npy"):
        h = sorted(glob.glob(str(od / "**" / pat), recursive=True))
        if h:
            lab = np.load(h[0]) if h[0].endswith(".npy") else tifffile.imread(h[0])
            return h[0], np.squeeze(lab)
    return None, None


def stage_merge(a):
    P(f"\n{SEP}\nMERGE / QC  {TAG}\n{SEP}")
    fovs = get_fovs(a)
    cells_all, tx_all, qc = [], [], []
    for f in fovs:
        base = OUT / f"fov{f}"
        native = a.mode == "native"
        inp = base / ("input_native" if native else "input")
        odir = base / ("dissect_out_native" if native else "dissect_out")
        mp = inp / "meta.json"
        if not mp.exists():
            continue
        meta = json.load(open(mp))
        src, lab = find_label(odir)
        if lab is None:
            P(f"  FOV {f}: [缺] 没有标签图，dissect_out 内容: "
              f"{[Path(p).name for p in glob.glob(str(odir / '*'))][:15]}")
            continue
        lab = lab.astype(np.int64)
        um = meta["target_um"]
        s_loc = um / meta["um_per_px"]                 # 1 个重采样像素 = 多少局部像素
        ids, cnt = np.unique(lab[lab > 0], return_counts=True)
        ys, xs = np.nonzero(lab)
        v = lab[ys, xs]; o = np.argsort(v, kind="stable")
        v, ys, xs = v[o], ys[o], xs[o]
        b = np.append(np.searchsorted(v, ids), len(v))
        cy = np.array([ys[b[i]:b[i + 1]].mean() for i in range(len(ids))]) + .5
        cx = np.array([xs[b[i]:b[i + 1]].mean() for i in range(len(ids))]) + .5
        lx = cx * s_loc                                # prep 时图像已翻转到局部坐标方向
        ly = cy * s_loc
        gid = f * 1_000_000 + ids
        ox, oy = meta["global_offset_px"]
        cells_all.append(pd.DataFrame({
            "cell_id": gid, "fov": f, "dissect_label": ids,
            "center_x_local_px": lx, "center_y_local_px": ly,
            "center_x_global_px": lx + ox if ox == ox else np.nan,
            "center_y_global_px": ly + oy if oy == oy else np.nan,
            "area_um2": cnt * um * um}))
        tifffile.imwrite(base / f"dissect_segmentation{'_native' if native else ''}.tif", lab.astype(np.uint32), compression="zlib")

        if native:
            gt_ = pd.read_csv(inp / "genes.csv")
            t = pd.DataFrame({"x_location": gt_["x_local_px"].to_numpy(float) * meta["um_per_px"],
                              "y_location": gt_["y_local_px"].to_numpy(float) * meta["um_per_px"],
                              "feature_name": gt_["target"].values})
        else:
            t = pd.read_parquet(inp / "transcripts.parquet")
        r = np.clip((t["y_location"].to_numpy() / um).astype(int), 0, lab.shape[0] - 1)
        c = np.clip((t["x_location"].to_numpy() / um).astype(int), 0, lab.shape[1] - 1)
        hit = lab[r, c]
        yl = t["y_location"].to_numpy() / meta["um_per_px"]
        tx_all.append(pd.DataFrame({
            "fov": f, "x_local_px": t["x_location"].to_numpy() / meta["um_per_px"],
            "y_local_px": yl, "target": t["feature_name"].values,
            "dissect_cell": np.where(hit > 0, f * 1_000_000 + hit, 0)}))
        rate = float((hit > 0).mean())
        qc.append(dict(fov=f, cells=int(len(ids)), platform_cells=meta["platform_cells"],
                       ratio=len(ids) / max(meta["platform_cells"], 1),
                       assignment_rate=rate, enrichment=meta["enrichment"]))
        P(f"  FOV {f}: cells {len(ids):,}  平台 {meta['platform_cells']:,}  "
          f"ratio {len(ids)/max(meta['platform_cells'],1):.3f}  分配率 {rate:.3f}  ({Path(src).name})")

    if not cells_all:
        raise SystemExit("没有任何 FOV 有结果")
    cells = pd.concat(cells_all, ignore_index=True)
    cells.to_csv(OUT / f"dissect_cells{SFX(a)}.csv", index=False)
    txa = pd.concat(tx_all, ignore_index=True)
    try:
        txa.to_parquet(OUT / f"dissect_transcript_assignment{SFX(a)}.parquet", index=False)
    except Exception:
        txa.to_csv(OUT / f"dissect_transcript_assignment{SFX(a)}.csv.gz", index=False)
    q = pd.DataFrame(qc); q.to_csv(OUT / f"dissect_qc_per_fov{SFX(a)}.csv", index=False)
    ratio = len(cells) / max(GT_CELLS, 1)
    med = float(cells.area_um2.median())
    P(f"\n  FOV {len(q)}/{len(fovs)}   cells {len(cells):,}   GT {GT_CELLS:,}   ratio {ratio:.4f}")
    P(f"  转录本分配率 {float((txa.dissect_cell > 0).mean()):.4f}")
    P(f"  area median {med:.1f} um^2   equiv diam {2*np.sqrt(med/np.pi):.2f} um")
    json.dump(dict(tag=TAG, fovs=fovs, fovs_merged=q.fov.tolist(), cells=int(len(cells)),
                   gt_cells=GT_CELLS, cell_ratio=ratio,
                   mask_assignment_rate=float((txa.dissect_cell > 0).mean())),
              open(OUT / f"merge_summary{SFX(a)}.json", "w"), indent=2)
    P(f"  -> {OUT/f'dissect_cells{SFX(a)}.csv'}\n  MERGE DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["inspect", "register", "prep", "run", "merge", "all"], default="all")
    ap.add_argument("--image-source", choices=["auto", "wsi"], default="auto",
                    help="auto：有 FOV 级图像就用，否则用整图；wsi：强制用整图")
    ap.add_argument("--wsi", default=None, help="整图路径，默认取 WSI_CANDIDATES 中第一个存在的")
    ap.add_argument("--wsi-channel", type=int, default=None)
    ap.add_argument("--wsi-scale", type=float, default=None, help="整图像素 / 全局像素；默认固定为 1")
    ap.add_argument("--scale-search", action="store_true", help="同时搜索缩放（默认不搜索，只搜方向与平移）")
    ap.add_argument("--coarse-max", type=int, default=6000, help="粗搜缩略图的最大边长（px）")
    ap.add_argument("--fovs", nargs="+", type=int, default=None)
    ap.add_argument("--image-template", default=None,
                    help="形态图路径模板，例如 '/data/x/Morphology2D/img_F{fov:03d}.TIF'")
    ap.add_argument("--channel", type=int, default=None, help="DAPI 通道序号；默认自动")
    ap.add_argument("--flip-y", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--mode", choices=["native", "xenium"], default="native",
                    help="native = DISSECT 原生 nanostring 入口（0.12 um/px 局部像素）；xenium = 重采样到 0.2125 um/px 伪装 xenium")
    ap.add_argument("--minarea", type=int, default=None, help="默认 350；--area-rule physical 时按物理尺寸换算")
    ap.add_argument("--area-rule", choices=["fixed", "physical"], default="fixed",
                    help="fixed = 与其他 DISSECT 运行相同的数值 350–1600；physical = 与 Xenium 物理尺寸等价")
    ap.add_argument("--maxarea", type=int, default=None, help="默认 1600；--area-rule physical 时按物理尺寸换算")
    ap.add_argument("--refine-batch-size", type=int, default=512)
    ap.add_argument("--threshold", type=float, default=-0.20)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--engine-extra", default="")
    ap.add_argument("--force-prep", action="store_true")
    ap.add_argument("--force-run", action="store_true")
    ap.add_argument("--stop-on-first-failure", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    P(f"{SEP}\nDISSECT x {TAG}   out {OUT}\n{SEP}")
    if a.stage == "inspect":
        stage_inspect(a); return
    if a.stage == "register" or (a.stage == "all" and not (OUT / "wsi_transform.json").exists()
                                 and find_wsi() is not None):
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