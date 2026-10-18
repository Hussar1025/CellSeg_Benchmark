#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
comseg_salvage.py   v2026.08.16

Recovers the per-RNA assignment from ComSeg runs whose tiles finished but whose
spot_assignment table came out without a label column.

What went wrong: after run_all the per-RNA answer lives in the graph node
attribute `cell_index_pred`, not in `df_spots_label`.  df_spots_label is the copy
made when the graph was built, so it still holds only

    ['index', 'x', 'y', 'z', 'gene', 'in_nucleus']

while the graph node attributes hold

    ['index','x','y','z','gene','in_nucleus','nb_mol','index_commu',
     'kernel_vector','leiden_merged','cell_index_pred','distance2centroid']

Every finished tile saved a `comseg_dict` pickle, so nothing has to be
recomputed.  This script reads those pickles, pulls `cell_index_pred` off the
graph, and writes a proper spot_assignment table per tile.

The label is written under several aliases (cell, cell_id, label,
cell_index_pred, comseg) so that whichever name the runner's post stage looks
for, it finds it.  After this, run the runner's own `--stage post`; the merge,
core-region de-duplication and rasterisation logic is untouched.

    python comseg_salvage.py --run-dir /data/qiuyijia/comseg_merfish_lung
    python comseg_salvage.py --run-dir /data/qiuyijia/comseg_cosmx_lymph_node/fov48
    python comseg_salvage.py --run-dir /data/qiuyijia/comseg_cosmx_lymph_node/fov261

then

    python comseg_merfish_4.py --stage post
    python comseg_cosmx_3.py --fovs 48 --stage post

--dry-run inspects one tile and prints what it found without writing anything.
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_ALIASES = ["cell", "cell_id", "label", "cell_index_pred", "comseg",
                 "cell_index", "cell_label"]
NON_CELL = {0, -1, None, "0", "-1", "", "nan"}


def log(msg=""):
    print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)


# --------------------------------------------------------------------------- #

def load_dict(path: Path):
    """Load a saved ComSegDict.  save() pickles __dict__, so the pickle already
    carries the real dataset; the stub is only needed to get past __init__."""
    try:
        from comseg import dictionary as di

        class Stub:
            disable_tqdm = True
            dict_scale = {"x": 1, "y": 1, "z": 1}

        cd = di.ComSegDict(dataset=Stub())
        cd.load(str(path))
        return cd, "ComSegDict"
    except Exception as e:
        log(f"    ComSegDict.load failed ({type(e).__name__}: {e}); raw pickle")
        with open(path, "rb") as f:
            return pickle.load(f), "raw dict"


def image_keys(obj):
    """The per-image entries, skipping the bookkeeping attributes."""
    skip = {"dataset", "mean_cell_diameter", "community_detection", "seed",
            "dict_img_name", "disable_tqdm", "path_dataset_folder"}
    if hasattr(obj, "keys"):
        return [k for k in obj.keys() if k not in skip]
    return []


def find_graph(g):
    for a in dir(g):
        if a.startswith("__"):
            continue
        try:
            v = getattr(g, a)
        except Exception:
            continue
        if hasattr(v, "nodes") and hasattr(v, "edges"):
            return a, v
    return None, None


def table_from_graph(G) -> pd.DataFrame:
    """Every RNA node with its predicted cell.

    Graphs also carry non-RNA nodes (centroids added by classify_centroid);
    those have no gene, so requiring a gene drops them.
    """
    rows = []
    for _, d in G.nodes(data=True):
        if "gene" not in d or d.get("gene") in (None, ""):
            continue
        rows.append((d.get("index"), d.get("x"), d.get("y"), d.get("z"),
                     d.get("gene"), d.get("in_nucleus"),
                     d.get("cell_index_pred"), d.get("leiden_merged")))
    t = pd.DataFrame(rows, columns=["index", "x", "y", "z", "gene",
                                    "in_nucleus", "cell_index_pred",
                                    "leiden_merged"])
    return t


def normalise(t: pd.DataFrame) -> pd.DataFrame:
    lab = t["cell_index_pred"]
    lab = lab.where(~lab.isin(list(NON_CELL)), 0)
    lab = pd.to_numeric(lab, errors="coerce").fillna(0).astype(np.int64)
    out = t.drop(columns=["cell_index_pred"]).copy()
    for a in LABEL_ALIASES:
        out[a] = lab
    return out


# --------------------------------------------------------------------------- #

def do_tile(tdir: Path, args) -> dict | None:
    pkl = tdir / "comseg_dict"
    if not pkl.exists():
        cand = sorted(tdir.glob("comseg_dict*"))
        if not cand:
            return None
        pkl = cand[0]

    obj, how = load_dict(pkl)
    keys = image_keys(obj)
    if not keys:
        log(f"    {tdir.name}: no image entry inside the pickle ({how})")
        return None

    parts = []
    for k in keys:
        g = obj[k]
        attr, G = find_graph(g)
        if G is None:
            log(f"    {tdir.name}/{k}: no graph on the image object")
            continue
        t = table_from_graph(G)
        t["image"] = k
        parts.append(t)
        if args.dry_run:
            n0 = next(iter(G.nodes))
            log(f"    graph at .{attr}  {G.number_of_nodes():,} nodes")
            log(f"    node keys: {list(G.nodes[n0].keys())}")
    if not parts:
        return None
    t = pd.concat(parts, ignore_index=True)
    t = normalise(t)

    nz = int((t["cell"] != 0).sum())
    ncell = int(t.loc[t["cell"] != 0, "cell"].nunique())
    info = dict(tile=tdir.name, rows=len(t), assigned=nz,
                assigned_frac=nz / max(len(t), 1), cells=ncell,
                x_min=float(t.x.min()), x_max=float(t.x.max()),
                y_min=float(t.y.min()), y_max=float(t.y.max()))

    if args.dry_run:
        log(f"    rows {len(t):,}  assigned {nz:,} ({nz/max(len(t),1)*100:.1f}%)"
            f"  distinct cells {ncell:,}")
        log(f"    x {t.x.min():.1f}..{t.x.max():.1f}   "
            f"y {t.y.min():.1f}..{t.y.max():.1f}")
        log(f"    columns written would be: {list(t.columns)}")
        log(f"    head:\n{t.head(3).to_string()}")
        return info

    dst = tdir / "spot_assignment.parquet"
    if dst.exists() and not args.overwrite:
        bak = tdir / "spot_assignment.prelabel.parquet"
        if not bak.exists():
            dst.replace(bak)
            log(f"    kept the old table as {bak.name}")
    try:
        t.to_parquet(dst)
    except Exception:
        dst = tdir / "spot_assignment.csv.gz"
        t.to_csv(dst, index=False)
    (tdir / "DONE.flag").write_text(
        json.dumps(info, indent=2, default=str) + "\n")
    fail = tdir / "FAILED.flag"
    if fail.exists():
        fail.unlink()
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="the directory holding comseg_out/, e.g. "
                         "/data/qiuyijia/comseg_merfish_lung")
    ap.add_argument("--tiles", default=None,
                    help="comma separated tile names; default is all")
    ap.add_argument("--dry-run", action="store_true",
                    help="inspect the first tile only, write nothing")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.run_dir)
    out = root / "comseg_out"
    if not out.is_dir():
        cand = sorted(root.glob("**/comseg_out"))
        if not cand:
            sys.exit(f"no comseg_out under {root}")
        out = cand[0]
    tiles = sorted(d for d in out.glob("*") if d.is_dir())
    if args.tiles:
        want = set(args.tiles.split(","))
        tiles = [t for t in tiles if t.name in want]
    have = [t for t in tiles if list(t.glob("comseg_dict*"))]

    log("=" * 74)
    log(f"COMSEG SALVAGE   {out}")
    log("=" * 74)
    log(f"  {len(tiles)} tile directories, {len(have)} carry a comseg_dict pickle")
    if not have:
        sys.exit("  nothing to salvage; those tiles never finished run_all")
    if args.dry_run:
        have = have[:1]
        log(f"  dry run on {have[0].name}")

    rows = []
    for i, t in enumerate(have, 1):
        log(f"  [{i}/{len(have)}] {t.name}")
        try:
            info = do_tile(t, args)
        except Exception as e:
            log(f"    FAILED {type(e).__name__}: {e}")
            continue
        if info:
            rows.append(info)
            if not args.dry_run:
                log(f"    rows {info['rows']:,}  assigned "
                    f"{info['assigned']:,} ({info['assigned_frac']*100:.1f}%)"
                    f"  cells {info['cells']:,}")

    if not rows:
        sys.exit("  no tile yielded a table")
    d = pd.DataFrame(rows)
    log("=" * 74)
    log(f"  salvaged {len(d)} tiles")
    log(f"  total RNA rows      {d.rows.sum():,}")
    log(f"  assigned            {d.assigned.sum():,} "
        f"({d.assigned.sum()/max(d.rows.sum(),1)*100:.1f}%)")
    log(f"  cells (tile sums, before core de-duplication)  {d.cells.sum():,}")
    log(f"  x range across tiles  {d.x_min.min():.1f} .. {d.x_max.max():.1f}")
    log(f"  y range across tiles  {d.y_min.min():.1f} .. {d.y_max.max():.1f}")
    if not args.dry_run:
        p = out.parent / "salvage_summary.tsv"
        d.to_csv(p, sep="\t", index=False)
        log(f"  -> {p}")
        log("")
        log("  next: run the runner's own post stage, e.g.")
        log("    python comseg_merfish_4.py --stage post")
        log("    python comseg_cosmx_3.py --fovs 48 --stage post")
        log("  if post still says it cannot find a label column, tell me the "
            "name it looked for and I will add it to LABEL_ALIASES.")
    log("=" * 74)


if __name__ == "__main__":
    main()