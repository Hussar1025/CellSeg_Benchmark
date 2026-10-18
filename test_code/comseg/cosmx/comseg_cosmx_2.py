#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comseg_cosmx_2.py — ComSeg @ cosmx_2 (CosMx lymph node) 带断点续跑的驱动

为什么要重跑
------------
上一轮（--max-cell-radius-um 默认 12.0）的结果是坏的:
    预测 3,371 细胞 / GT 4,898 = 0.688
    每细胞 1,027 转录本，等效直径 27.3 um
    metadata 实测细胞直径 7.86 um  ->  预测团块是真实细胞的 3.5 倍
    assign_accuracy 0.024   count_pearson 0.032
每细胞 1,027 条恰好等于全 FOV 平均 (4,976,788 / 4,898 = 1,016)，
也就是说 3,371 个团平摊了几乎全部转录本 —— 核内部分虽然一对一，
但占 52% 的核外转录本被跨细胞聚合了。
淋巴结细胞间距只有几微米，12 um 的扩展半径足以跨过 2-3 层邻居。
对照 xenium_2 (breast, 直径大得多) 用 diameter 12 / radius 20 得到 F1 0.964，
说明这个参数必须按组织的细胞密度调，不能沿用默认。

新参数: --mean-cell-diameter-um 8 --max-cell-radius-um 6
    8 um 来自 metadata 的 Area 中位 3374 px x (0.1203 um/px)^2 -> 直径 7.86 um
    6 um = 0.76 倍直径，让扩展在触及邻居核之前停下

断点续跑
--------
原 comseg_cosmx_3.py 的完成判定只看 spot_assignment 文件是否存在:
    if not (tdir/'spot_assignment.csv.gz').exists(): 不打 flag
    else: (tdir/'DONE.flag').touch()
08-11 那次 API 报错的 tile 写出了 6 列的表（文件存在但没有 cell 列），
于是 flag 照打，后续续跑一律跳过 —— 25 个 flag 里 11 个是假的。

本脚本据此重写完成判定:
    真完成 = parquet 存在 且 含 cell 列 且 唯一细胞数 > 0
每个 tile 跑完立即校验，不合格就删掉它的 flag，下次自然重试。
中断后重跑同一条命令即可，已完成的 tile 会被跳过。

用法
----
  python3 comseg_cosmx_2.py --status                     # 只看状态
  python3 comseg_cosmx_2.py --reset                      # 清旧结果准备重跑
  python3 comseg_cosmx_2.py --run                        # 续跑（可反复执行）
  python3 comseg_cosmx_2.py --run --clean-dict           # 跑完即删中间产物
  python3 comseg_cosmx_2.py --run --max-tiles 2          # 先验证 2 个 tile
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import pandas as pd

ROOT = "/data/qiuyijia/comseg_cosmx_lymph_node/fov48"
RUNNER = "/data/qiuyijia/comseg/comseg_cosmx_3.py"
FOV = 48
GT_CELLS = 4898
PX_UM = 0.1203                 # napari .zattrs 的 scale_um（之前一直用 0.12）
CELL_DIAM_UM = 8.0             # metadata Area 中位 3374 px -> 7.86 um
MAX_RADIUS_UM = 6.0            # 0.76 倍直径
CELL_COLS = ("cell", "cell_index", "comseg", "cell_id", "label",
             "cell_label", "cell_index_pred")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def tile_dirs(root):
    d = os.path.join(root, "comseg_out")
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, x) for x in os.listdir(d)
                  if x.startswith("tile_") and
                  os.path.isdir(os.path.join(d, x)))


def dir_size(p):
    if not os.path.exists(p):
        return 0
    if os.path.isfile(p):
        return os.path.getsize(p)
    return sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(p) for f in fs)


def inspect(td):
    """真完成 = parquet 存在 且 含 cell 列 且 唯一细胞数 > 0。"""
    p = os.path.join(td, "spot_assignment.parquet")
    fl = os.path.join(td, "DONE.flag")
    info = dict(flag=os.path.exists(fl),
                flag_sz=os.path.getsize(fl) if os.path.exists(fl) else -1,
                ncol=0, cells=0, assigned=0.0, tx=0,
                dict_gb=dir_size(os.path.join(td, "comseg_dict")) / 1e9)
    if not os.path.exists(p):
        return "not_run", info
    try:
        df = pd.read_parquet(p)
    except Exception as e:
        info["err"] = str(e)[:60]
        return "bad_parquet", info
    info["ncol"] = len(df.columns)
    info["tx"] = len(df)
    cc = next((c for c in CELL_COLS if c in df.columns), None)
    if cc is None:
        return "no_cell_col", info
    v = pd.to_numeric(df[cc], errors="coerce")
    pos = v > 0
    info["cells"] = int(v[pos].nunique())
    info["assigned"] = float(pos.mean())
    if info["cells"] == 0:
        return "zero_cells", info
    return "ok", info


def status(root, verbose=True):
    tds = tile_dirs(root)
    if not tds:
        log(f"  {root}/comseg_out 下没有 tile 目录 —— 先跑 --stage prep")
        return {}, []
    res, tot_cells, tot_dict = {}, 0, 0.0
    if verbose:
        log(f"{'tile':<12}{'状态':<14}{'列':>4}{'转录本':>11}"
            f"{'细胞':>8}{'已分配':>9}{'dict':>8}")
        log("-" * 68)
    for td in tds:
        st, info = inspect(td)
        res[os.path.basename(td)] = (st, info)
        tot_dict += info["dict_gb"]
        if st == "ok":
            tot_cells += info["cells"]
        if verbose:
            log(f"{os.path.basename(td):<12}{st:<14}{info['ncol']:>4}"
                f"{info['tx']:>11,}{info['cells']:>8,}"
                f"{info['assigned']*100:>8.1f}%{info['dict_gb']:>7.1f}G"
                + ("" if st == "ok" else "  ★"))
    bad = [k for k, (s, _) in res.items() if s != "ok"]
    n_ok = len(tds) - len(bad)
    if verbose:
        log("-" * 68)
        log(f"完成 {n_ok}/{len(tds)}   累计细胞 {tot_cells:,} "
            f"(GT {GT_CELLS:,})   中间产物 {tot_dict:.0f} GB")
        if bad:
            log(f"待跑 {len(bad)} 个: {bad}")
    return res, bad


def reset(root, yes=False):
    """清掉全部 tile 产物，为换参数重跑做准备。"""
    tds = tile_dirs(root)
    tot = sum(dir_size(t) for t in tds)
    log(f"将清除 {len(tds)} 个 tile 目录，释放约 {tot/1e9:.0f} GB")
    log(f"  {os.path.join(root, 'comseg_out')}")
    if not yes:
        log("  加 --yes 实际执行")
        return
    for t in tds:
        shutil.rmtree(t)
    log(f"  已清除 {len(tds)} 个 tile 目录")


def verify_and_fix(root):
    """删掉假 flag，使下次续跑会重试那些 tile。"""
    res, bad = status(root, verbose=False)
    n = 0
    for name in bad:
        st, info = res[name]
        fl = os.path.join(root, "comseg_out", name, "DONE.flag")
        if os.path.exists(fl):
            os.remove(fl)
            n += 1
            log(f"  {name}: {st} -> 删除假 flag")
    return n


def run(root, args):
    if not os.path.exists(RUNNER):
        sys.exit(f"✗ runner 不存在: {RUNNER}")

    log("=" * 68)
    log("参数")
    log("=" * 68)
    log(f"  mean_cell_diameter  {args.diameter} um")
    log(f"    (metadata Area 中位 3374 px x {PX_UM} um/px -> 直径 7.86 um)")
    log(f"  max_cell_radius     {args.radius} um  "
        f"= {args.radius/args.diameter:.2f} 倍直径")
    log(f"    (上一轮用 12.0 = 1.53 倍，导致 27.3 um 的巨团)")
    log(f"  tile_px             {args.tile_px}   "
        f"overlap {args.overlap_px} px = {args.overlap_px*PX_UM:.1f} um")
    if args.overlap_px * PX_UM < args.radius:
        log(f"  ⚠ overlap {args.overlap_px*PX_UM:.1f} um < radius "
            f"{args.radius} um，跨缝细胞会被切断")

    n_fixed = verify_and_fix(root)
    if n_fixed:
        log(f"\n清理了 {n_fixed} 个假 flag")

    res, bad = status(root, verbose=False)
    if not res:
        # comseg_out 为空（比如刚 --reset 过）不等于"全部完成"，
        # 这时必须先跑 prep 重建 tile 目录，否则 runner 无事可做。
        log("\ncomseg_out 下没有 tile 目录 —— 先跑 prep:")
        log(f"  python3 -u {RUNNER} --fovs {args.fov} --stage prep \\")
        log(f"      --mean-cell-diameter-um {args.diameter} \\")
        log(f"      --max-cell-radius-um {args.radius}")
        return
    log(f"\n待跑 {len(bad)} / {len(res)} 个 tile")
    if not bad:
        log("全部已完成")
        return
    if args.max_tiles:
        log(f"  --max-tiles {args.max_tiles}：本次只跑前 {args.max_tiles} 个")

    st = os.statvfs(root)
    free_gb = st.f_bavail * st.f_frsize / 1e9
    log(f"磁盘可用 {free_gb:.0f} GB"
        + ("   ⚠ 建议加 --clean-dict" if free_gb < len(bad) * 20 else ""))

    cmd = [sys.executable, "-u", RUNNER,
           "--fovs", str(args.fov), "--stage", "run",
           "--mean-cell-diameter-um", str(args.diameter),
           "--max-cell-radius-um", str(args.radius)]
    if args.tile_px:
        cmd += ["--tile-px", str(args.tile_px)]
    if args.overlap_px:
        cmd += ["--overlap-px", str(args.overlap_px)]
    log("\n$ " + " ".join(cmd))
    log("")
    rc = subprocess.call(cmd, cwd=os.path.dirname(RUNNER))
    log(f"\nrunner 退出码 {rc}")

    log("\n" + "=" * 68)
    log("跑后复核")
    log("=" * 68)
    res2, bad2 = status(root)
    n_fixed2 = verify_and_fix(root)
    if n_fixed2:
        log(f"\n本轮有 {n_fixed2} 个 tile 未通过校验，flag 已删，"
            f"重跑同一条命令会重试它们")

    if args.clean_dict:
        log("\n清理已完成 tile 的 comseg_dict "
            "(spot_assignment.parquet 已落盘，dict 只用于续跑)")
        freed = 0
        for name, (s, _) in res2.items():
            if s != "ok":
                continue
            dd = os.path.join(root, "comseg_out", name, "comseg_dict")
            if os.path.exists(dd):
                n = dir_size(dd)
                shutil.rmtree(dd)
                freed += n
                log(f"  {name}  {n/1e9:.1f} GB")
        log(f"  释放 {freed/1e9:.0f} GB")

    ok = [k for k, (s, _) in res2.items() if s == "ok"]
    if len(ok) == len(res2):
        log("\n✓ 全部完成，评估:")
        log("  python3 /data/qiuyijia/comseg/comseg_value_cosmx_2.py")
    else:
        log(f"\n还剩 {len(res2)-len(ok)} 个，重跑同一条命令继续")


def quick_check(root):
    """跑完 1-2 个 tile 后先看等效直径，判断参数是否修好了。"""
    import numpy as np
    res, _ = status(root, verbose=False)
    ok = [k for k, (s, _) in res.items() if s == "ok"]
    if not ok:
        log("  还没有完成的 tile")
        return
    log(f"\n【参数效果速查】用 {len(ok)} 个已完成 tile")
    frames = []
    for name in ok[:4]:
        p = os.path.join(root, "comseg_out", name,
                         "spot_assignment.parquet")
        frames.append(pd.read_parquet(p))
    d = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["x", "y", "gene"])
    cc = next(c for c in CELL_COLS if c in d.columns)
    v = pd.to_numeric(d[cc], errors="coerce")
    da = d[v > 0]
    g = da.groupby(cc).agg(n=("x", "size"))
    sp = da.groupby(cc).agg(sx=("x", "std"), sy=("y", "std"))
    diam = 2 * np.sqrt(2) * np.sqrt((sp.sx ** 2 + sp.sy ** 2) / 2)
    log(f"  细胞 {len(g):,}   每细胞 tx 中位 {g.n.median():.0f}")
    log(f"  等效直径 中位 {diam.median():.2f} um")
    log(f"  对照: metadata 实测 7.86 um；上一轮 (radius 12) 是 27.3 um")
    if diam.median() < 12:
        log("  ✓ 直径回到合理范围，参数修好了")
    elif diam.median() < 20:
        log("  ~ 有改善但仍偏大，可再降 --radius")
    else:
        log("  ✗ 仍是巨团，radius 不是唯一原因，查 k_nearest_neighbors "
            "与 min_merge_correlation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--fov", type=int, default=FOV)
    ap.add_argument("--diameter", type=float, default=CELL_DIAM_UM)
    ap.add_argument("--radius", type=float, default=MAX_RADIUS_UM)
    ap.add_argument("--tile-px", type=int, default=0)
    ap.add_argument("--overlap-px", type=int, default=0)
    ap.add_argument("--max-tiles", type=int, default=0)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="用已完成的 tile 速查等效直径")
    ap.add_argument("--clean-dict", action="store_true")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()

    log("=" * 68)
    log(f"ComSeg @ cosmx_2  FOV {a.fov}   {a.root}")
    log("=" * 68)

    if a.reset:
        reset(a.root, a.yes)
        return
    if a.check:
        quick_check(a.root)
        return
    if a.run:
        run(a.root, a)
        return
    status(a.root)
    log("\n下一步:")
    log(f"  清旧结果   python3 {sys.argv[0]} --reset --yes")
    log(f"  验证参数   python3 {sys.argv[0]} --run --max-tiles 2")
    log(f"             python3 {sys.argv[0]} --check")
    log(f"  全量续跑   python3 {sys.argv[0]} --run --clean-dict")


if __name__ == "__main__":
    main()