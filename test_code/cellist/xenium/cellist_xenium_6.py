#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cellist xenium_6 colon downstream sweep — direct-engine version.

NO source-code patching.
We invoke the verified cellist_xenium_5.py engine directly using:
  --dataset liver          (only selects an existing parser choice)
  --data-root xenium_colon (actual dataset)
  --morphology explicit
  --out-dir explicit
  --prefix explicit

Thus DATASETS/MORPH_CANDIDATES never need to be modified.

Fixed best nucleus:
  diameter=2.5um, cellprob=-2, flow=0.8, no-local=True
"""
from __future__ import annotations
import json, os, re, shutil, subprocess, sys
from pathlib import Path

ENGINE = Path("/data/qiuyijia/cellist/code/cellist_xenium_5.py")
ROOT = Path("/data/qiuyijia/dataset/xenium_colon")
MORPH = ROOT / "morphology_focus/morphology_focus_0000.ome.tif"
BASE = Path("/data/qiuyijia/cellist_xenium_colon_roi6000")
OUTROOT = Path("/data/qiuyijia/cellist_xenium_colon_downstream")

TRIALS = [
    dict(tag="r3_imp0p5_fill0_freq", radius=3, imp=0.5, fill=0, gene="Frequent", two=False),
    dict(tag="r3_imp1_fill0_freq", radius=3, imp=1.0, fill=0, gene="Frequent", two=False),
    dict(tag="r4_imp0p5_fill0_freq", radius=4, imp=0.5, fill=0, gene="Frequent", two=False),
    dict(tag="r4_imp1_fill0_freq", radius=4, imp=1.0, fill=0, gene="Frequent", two=False),
    dict(tag="r3_imp0p5_fill0p5_freq", radius=3, imp=0.5, fill=0.5, gene="Frequent", two=False),
    dict(tag="r4_imp0p5_fill0p5_freq", radius=4, imp=0.5, fill=0.5, gene="Frequent", two=False),
    dict(tag="r3_imp0p5_fill0_hvg", radius=3, imp=0.5, fill=0, gene="HVG", two=False),
    dict(tag="r4_imp0p5_fill0_hvg", radius=4, imp=0.5, fill=0, gene="HVG", two=False),
    dict(tag="r3_imp0p5_fill0_freq_two", radius=3, imp=0.5, fill=0, gene="Frequent", two=True),
    dict(tag="r4_imp0p5_fill0_freq_two", radius=4, imp=0.5, fill=0, gene="Frequent", two=True),
    dict(tag="ref_r5_imp2p5_fill1p5", radius=5, imp=2.5, fill=1.5, gene="Frequent", two=False),
]

def copy_cached_inputs(out):
    src = BASE/"input"; dst = out/"input"
    if src.exists() and not dst.exists():
        shutil.copytree(src, dst)

def parse_metrics(text):
    d={}
    m=re.search(r"cells\s+([0-9,]+)\s+\(platform\s+([0-9,]+),\s+ratio\s+([0-9.]+)\)",text)
    if m:
        d.update(cells=int(m.group(1).replace(",","")), platform=int(m.group(2).replace(",","")), ratio=float(m.group(3)))
    m=re.search(r"equiv diam \(um\)\s+med=([0-9.]+)",text)
    if m: d["diam_um"]=float(m.group(1))
    m=re.search(r"coverage\s+([0-9.]+)%",text)
    if m: d["coverage"]=float(m.group(1))
    return d

def main():
    if not ENGINE.exists(): raise SystemExit(f"missing {ENGINE}")
    if not MORPH.exists(): raise SystemExit(f"missing {MORPH}")
    OUTROOT.mkdir(parents=True,exist_ok=True)
    summary=[]
    for t in TRIALS:
        out=OUTROOT/t["tag"]; out.mkdir(parents=True,exist_ok=True); copy_cached_inputs(out)
        cmd=[
            sys.executable,"-u",str(ENGINE),
            "--dataset","liver",
            "--data-root",str(ROOT),
            "--morphology",str(MORPH),
            "--out-dir",str(out),
            "--prefix",t["tag"],
            "--stage","seg",
            "--roi-size","6000",
            "--roi-origin","14052","11241",
            "--pixel-size","0.2125",
            "--gem-bin","2","--bin-factor","10","--qv-min","20",
            "--nucleus-diameter-um","2.5",
            "--cellprob-threshold","-2",
            "--flow-threshold","0.8",
            "--no-local-threshold",
            "--cell-radius-um",str(t["radius"]),
            "--spot-imputation-distance",str(t["imp"]),
            "--noise-prop","0",
            "--gene-use",t["gene"],
            "--fill-radius-um",str(t["fill"]),
            "--min-nuclear-tx","20",
            "--nworkers","3",
            "--force-run",
        ]
        if t["two"]: cmd.append("--two-step")
        print("\n"+"="*100); print(t["tag"]); print(" ".join(cmd),flush=True)
        p=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=os.environ.copy())
        print(p.stdout,flush=True)
        (OUTROOT/f"{t['tag']}.log").write_text(p.stdout)
        rec=dict(t,returncode=p.returncode); rec.update(parse_metrics(p.stdout)); summary.append(rec)
        (OUTROOT/"summary.json").write_text(json.dumps(summary,indent=2))
    good=[x for x in summary if x.get("returncode")==0 and "ratio" in x]
    good.sort(key=lambda x:abs(x["ratio"]-1))
    print("\nBEST FIRST")
    for x in good: print(x)

if __name__=="__main__": main()
