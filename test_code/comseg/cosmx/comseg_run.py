#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
comseg_run.py — ComSeg 全流程编排（CosMx Pancreas，18 个 FOV）

相比原 comseg_cosmx.py 的四处改动：
  1. 弃用 mp.Pool，改为 subprocess 逐个起 comseg_worker.py
     → mp.Pool 的 worker 是 daemon 进程，daemon 不允许再有子进程，
       ComSeg 内部若用 joblib/loky 并行会直接挂死（上次卡 3 天的原因）
  2. 每个 FOV 独立超时，超时 kill 并标记失败，不拖累其他 FOV
  3. h5ad 已存在则跳过 → 可反复续跑
  4. 默认 Top 1000 基因 + sampling_size 50000（原为全部 21731 + 100 万采样，慢 10-20 倍）

用法：
    python comseg_run.py                       # 默认 Top1000 / 4 并发 / 90min 超时
    python comseg_run.py --top-n 3000 --sampling-size 150000
    python comseg_run.py --workers 2           # 内存吃紧时降并发
    python comseg_run.py --fovs 51,52,53       # 只跑指定 FOV
    python comseg_run.py --merge-only          # 只做最后的合并
"""
import os
import re
import sys
import glob
import time
import argparse
import subprocess

import numpy as np
import pandas as pd
import anndata


# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR   = "/data/qiuyijia/comseg"
INPUT_DIR  = os.path.join(BASE_DIR, "input_csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR    = os.path.join(BASE_DIR, "logs")
WORKER     = os.path.join(BASE_DIR, "comseg_worker.py")

MERGED_H5AD   = os.path.join(OUTPUT_DIR, "comseg_pancreas_full.h5ad")
CELL_SUMMARY  = os.path.join(OUTPUT_DIR, "comseg_cell_summary.csv")

FULL_GENE_LIST = os.path.join(BASE_DIR, "top21731_genes.txt")   # 已按转录本数降序
# ══════════════════════════════════════════════════════════════════════════════


def log(msg):
    print(msg, flush=True)


def hr(title):
    log("\n" + "═" * 66)
    log(f"  {title}")
    log("═" * 66)


# ─────────────────────────────────────────────────────────────────────────────
#  发现可用 FOV
# ─────────────────────────────────────────────────────────────────────────────
def discover_fovs():
    fovs = []
    for p in sorted(glob.glob(os.path.join(INPUT_DIR, "fov_*.csv"))):
        m = re.search(r"fov_(\d+)\.csv$", os.path.basename(p))
        if m:
            fovs.append(int(m.group(1)))
    return sorted(fovs)


# ─────────────────────────────────────────────────────────────────────────────
#  准备基因白名单
# ─────────────────────────────────────────────────────────────────────────────
def prepare_gene_list(top_n, fovs):
    """
    优先复用已有的 top21731_genes.txt（原脚本写出时已按转录本数降序，
    取前 N 行即为 Top N，无需重扫 65M 行的 tx_file）。
    若该文件缺失，则从 input_csv 的 target 列重新统计。
    """
    out_path = os.path.join(BASE_DIR, f"top{top_n}_genes.txt")
    if os.path.exists(out_path):
        with open(out_path) as f:
            n = sum(1 for ln in f if ln.strip())
        log(f"[基因] 直接复用: {out_path}（{n} 个）")
        return out_path

    if os.path.exists(FULL_GENE_LIST):
        with open(FULL_GENE_LIST) as f:
            ordered = [ln.strip() for ln in f if ln.strip()]
        log(f"[基因] 从 {os.path.basename(FULL_GENE_LIST)} 取前 {top_n} 个"
            f"（该文件共 {len(ordered)} 个，已按转录本数降序）")
        sel = ordered[:top_n]
    else:
        log(f"[基因] 未找到 {FULL_GENE_LIST}，从 input_csv 重新统计 ...")
        counts = {}
        for fov in fovs:
            p = os.path.join(INPUT_DIR, f"fov_{fov:03d}.csv")
            if not os.path.exists(p):
                continue
            vc = pd.read_csv(p, usecols=["target"])["target"].value_counts()
            for g, c in vc.items():
                counts[g] = counts.get(g, 0) + int(c)
            log(f"  FOV {fov}: 累计 {len(counts)} 个基因")
        s = pd.Series(counts).sort_values(ascending=False)
        cov = s.iloc[:top_n].sum() / s.sum() * 100
        log(f"[基因] 总 {len(s)} 个，取前 {top_n} 个，覆盖 {cov:.1f}% 转录本")
        sel = s.iloc[:top_n].index.tolist()

    with open(out_path, "w") as f:
        f.write("\n".join(sel))
    log(f"[基因] 已写出: {out_path}（{len(sel)} 个）")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
#  subprocess 调度（并发 + 超时）
# ─────────────────────────────────────────────────────────────────────────────
def run_all_fovs(fovs, gene_list, top_n, sampling_size, workers, timeout):
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pending = list(fovs)
    running = []          # [{fov, proc, fh, logpath, t0}]
    results = {}          # fov -> ("done"/"cached"/"timeout"/"error", 秒)

    def launch(fov):
        logpath = os.path.join(LOG_DIR, f"fov_{fov:03d}.log")
        fh = open(logpath, "w")
        cmd = [sys.executable, "-u", WORKER, str(fov),
               "--gene-list", gene_list,
               "--top-n", str(top_n),
               "--sampling-size", str(sampling_size)]
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                cwd=BASE_DIR)
        log(f"  ▶ 启动 FOV {fov}  (pid={proc.pid})  日志: {logpath}")
        return {"fov": fov, "proc": proc, "fh": fh,
                "logpath": logpath, "t0": time.time()}

    t_start = time.time()
    while pending or running:
        # 补满并发槽位
        while pending and len(running) < workers:
            running.append(launch(pending.pop(0)))

        time.sleep(3)

        for job in running[:]:
            fov, proc = job["fov"], job["proc"]
            elapsed = time.time() - job["t0"]
            rc = proc.poll()

            if rc is None:
                # 超时 → kill
                if elapsed > timeout:
                    log(f"  ✘ FOV {fov} 超时 {elapsed/60:.1f} min，强制终止")
                    proc.kill()
                    try:
                        proc.wait(timeout=10)
                    except Exception:
                        pass
                    job["fh"].close()
                    results[fov] = ("timeout", elapsed)
                    running.remove(job)
                continue

            # 已结束
            job["fh"].close()
            running.remove(job)
            out_path = os.path.join(OUTPUT_DIR, f"fov_{fov:03d}.h5ad")
            if rc == 0 and os.path.exists(out_path):
                n = anndata.read_h5ad(out_path).n_obs
                results[fov] = ("done", elapsed)
                log(f"  ✔ FOV {fov} 完成  {n:,} 个细胞  {elapsed/60:.1f} min")
            else:
                results[fov] = ("error", elapsed)
                log(f"  ✘ FOV {fov} 失败 (rc={rc})，见 {job['logpath']}")

        n_done = len(results)
        if n_done and n_done % 3 == 0 and not running:
            log(f"  ── 进度 {n_done}/{len(fovs)}  "
                f"已用 {(time.time()-t_start)/60:.1f} min")

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  合并
# ─────────────────────────────────────────────────────────────────────────────
def merge_results(fovs):
    adatas, ok = [], []
    for fov in fovs:
        p = os.path.join(OUTPUT_DIR, f"fov_{fov:03d}.h5ad")
        if not os.path.exists(p):
            log(f"  FOV {fov}: 缺失，跳过")
            continue
        a = anndata.read_h5ad(p)
        a.obs_names = [f"fov{fov:03d}_{i}" for i in range(a.n_obs)]
        adatas.append(a)
        ok.append(fov)
        log(f"  FOV {fov}: {a.n_obs:,} 个细胞")

    if not adatas:
        log("  ✘ 没有可合并的 h5ad")
        return None

    merged = anndata.concat(adatas, join="outer", fill_value=0)
    merged.write_h5ad(MERGED_H5AD)
    log(f"\n  合并完成: {merged.n_obs:,} 个细胞 × {merged.n_vars:,} 个基因")
    log(f"  → {MERGED_H5AD}")

    # 细胞级汇总 CSV
    X = merged.X
    counts = np.asarray(X.sum(axis=1)).ravel() if hasattr(X, "sum") else X.sum(1)
    genes  = np.asarray((X > 0).sum(axis=1)).ravel()
    summary = merged.obs.copy()
    summary["n_transcripts"] = counts
    summary["n_genes"]       = genes
    summary.to_csv(CELL_SUMMARY)
    log(f"  → {CELL_SUMMARY}")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=1000,
                    help="使用转录本数最高的 N 个基因，默认 1000")
    ap.add_argument("--sampling-size", type=int, default=50_000,
                    help="compute_edge_weight 采样量，默认 50000（≈50 次/基因）")
    ap.add_argument("--workers", type=int, default=4,
                    help="同时运行的 FOV 数，默认 4；内存吃紧改 2")
    ap.add_argument("--timeout", type=int, default=5400,
                    help="单 FOV 超时秒数，默认 5400（90 分钟）")
    ap.add_argument("--fovs", default=None,
                    help="只跑指定 FOV，逗号分隔，例如 51,52,53")
    ap.add_argument("--merge-only", action="store_true",
                    help="跳过计算，只做合并")
    args = ap.parse_args()

    if not os.path.exists(WORKER):
        log(f"✘ 找不到 worker 脚本: {WORKER}")
        sys.exit(1)

    fovs = discover_fovs()
    if args.fovs:
        want = {int(x) for x in args.fovs.split(",")}
        fovs = [f for f in fovs if f in want]
    if not fovs:
        log(f"✘ {INPUT_DIR} 下没有 fov_*.csv")
        sys.exit(1)

    hr(f"ComSeg — {len(fovs)} 个 FOV: {fovs}")
    log(f"  Top N 基因     : {args.top_n}")
    log(f"  sampling_size  : {args.sampling_size:,}  "
        f"(≈{args.sampling_size/args.top_n:.0f} 次/基因)")
    log(f"  并发 worker    : {args.workers}")
    log(f"  单 FOV 超时    : {args.timeout/60:.0f} min")

    if not args.merge_only:
        hr("Step 1: 准备基因白名单")
        gene_list = prepare_gene_list(args.top_n, fovs)

        hr(f"Step 2: 运行 ComSeg（{args.workers} 个独立 subprocess）")
        t0 = time.time()
        results = run_all_fovs(fovs, gene_list, args.top_n,
                               args.sampling_size, args.workers, args.timeout)

        hr("运行结果汇总")
        for status in ["done", "cached", "timeout", "error"]:
            hit = [f for f, (s, _) in results.items() if s == status]
            if hit:
                log(f"  {status:8s}: {len(hit)} 个 → {sorted(hit)}")
        log(f"  总耗时: {(time.time()-t0)/60:.1f} min")

    hr("Step 3: 合并所有 FOV")
    merge_results(fovs)

    hr("完成")
    log(f"  输出目录: {OUTPUT_DIR}")
    log(f"  日志目录: {LOG_DIR}")


if __name__ == "__main__":
    main()