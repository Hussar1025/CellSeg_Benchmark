#!/usr/bin/env python
"""
GeneSegNet pipeline: scan thresholds -> pick best -> generate mask -> full evaluation

===== About the match radius =====
We do not override --match-radius-um here; let run_genesegnet_infer.py and
eval_genesegnet.py each use their own built-in, per-target defaults
(lymph=8.0 breast=12.0 merfish=12.0um) -- both sides already agree,
so there is no need for this script to repeat them and risk two conflicting values.

Usage:
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

    print(f"-- Scanning {a.target} --")
    subprocess.run(scan_cmd, check=True)

    csv = f"yourpath/genesegnet_infer/{a.target}_scan.csv"
    df = pd.read_csv(csv)
    # Use the group with the most matches -- better reflects true recall quality than just matching the predicted count
    best = df.sort_values("n_match", ascending=False).iloc[0]
    conf, ft = float(best.confidence), float(best.flow_threshold)
    print(f"\n-- best: confidence={conf} flow_threshold={ft} "
          f"(matches {int(best.n_match):,}) --")

    run_cmd = ["python", INFER, "--target", a.target, "--run",
              "--confidence", str(conf), "--flow-threshold", str(ft)]
    if a.diameter:
        run_cmd += ["--diameter", str(a.diameter)]
    if a.target == "merfish":
        run_cmd += ["--resolution", str(a.resolution)]
    subprocess.run(run_cmd, check=True)

    mask = f"yourpath/genesegnet_infer/{a.target}_mask.tif"
    eval_cmd = ["python", EVAL, "--target", a.target, "--mask", mask]
    if a.target == "merfish":
        eval_cmd += ["--resolution", str(a.resolution)]
    subprocess.run(eval_cmd, check=True)


if __name__ == "__main__":
    main()