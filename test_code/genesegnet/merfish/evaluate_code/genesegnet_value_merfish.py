#!/usr/bin/env python
"""
GeneSegNet 一条龙：扫阈值 → 挑最优 → 生成 mask → 完整评估

===== 关于匹配半径 =====
不在这里覆盖 --match-radius-um，让 run_genesegnet_infer.py 和
eval_genesegnet.py 各自用它们内置的、按 target 区分的默认值
（lymph=8.0 breast=12.0 merfish=12.0µm）——两边已经保持一致，
不需要这个脚本再重复指定一遍造成两个值打架的风险。

用法：
  python genesegnet_pipeline.py --target xenium_breast
  python genesegnet_pipeline.py --target xenium_breast --diameter 60
  python genesegnet_pipeline.py --target merfish --resolution 0.5
"""
import argparse
import subprocess

import pandas as pd

INFER = "run_genesegnet_infer.py"
EVAL = "eval_genesegnet.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--diameter", type=float, default=None)
    ap.add_argument("--resolution", type=float, default=0.5)
    a = ap.parse_args()

    scan_cmd = ["python", INFER, "--target", a.target, "--scan-thresholds",
               "--confidences", "-0.8", "-0.5", "0.0",
               "--flow-thresholds", "0.8", "0.9", "0.95", "0.99"]
    if a.diameter:
        scan_cmd += ["--diameter", str(a.diameter)]
    if a.target == "merfish":
        scan_cmd += ["--resolution", str(a.resolution)]

    print(f"── 扫描 {a.target} ──")
    subprocess.run(scan_cmd, check=True)

    csv = f"/data/qiuyijia/genesegnet_infer/{a.target}_scan.csv"
    df = pd.read_csv(csv)
    # 用匹配数最大的一组 —— 比单纯凑预测数量更能代表真实召回质量
    best = df.sort_values("n_match", ascending=False).iloc[0]
    conf, ft = float(best.confidence), float(best.flow_threshold)
    print(f"\n── 最优: confidence={conf} flow_threshold={ft} "
          f"（匹配数 {int(best.n_match):,}）──")

    run_cmd = ["python", INFER, "--target", a.target, "--run",
              "--confidence", str(conf), "--flow-threshold", str(ft)]
    if a.diameter:
        run_cmd += ["--diameter", str(a.diameter)]
    if a.target == "merfish":
        run_cmd += ["--resolution", str(a.resolution)]
    subprocess.run(run_cmd, check=True)

    mask = f"/data/qiuyijia/genesegnet_infer/{a.target}_mask.tif"
    eval_cmd = ["python", EVAL, "--target", a.target, "--mask", mask]
    if a.target == "merfish":
        eval_cmd += ["--resolution", str(a.resolution)]
    subprocess.run(eval_cmd, check=True)


if __name__ == "__main__":
    main()