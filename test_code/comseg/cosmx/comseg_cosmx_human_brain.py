#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
comseg_cosmx_3.py   v2026.08.13

ComSeg on CosMx FOVs.

What changed from the previous version, and why it failed
--------------------------------------------------------
The previous version handed ComSeg its nucleus prior as a *column* in the spots
csv, on the reasoning that a whole-slide mask cannot be passed once the FOV is
split into tiles.  That reasoning was wrong, and it broke every tile:

    ComSegDict.run_all -> classify_centroid
      if self.dataset.dict_centroid is None:
    AttributeError: 'ComSegDataset' object has no attribute 'dict_centroid'

`dict_centroid` is only ever created inside

    ComSegDataset.add_prior_from_mask(..., compute_centroid=True)
        -> compute_dict_centroid(mask_nuclei=mask, ...)

so a run that never passes a mask can never reach classify_centroid.  Note that
the attribute is *absent*, not None, so the `is None` guard does not save it.

This version therefore writes one nucleus mask tiff per tile and calls
add_prior_from_mask.  The masks are painted from the CosMx metadata (NucArea +
the cell centroid in FOV-local pixels), which is the same platform prior UCS and
Proseg were given, so the three stay comparable.

Nothing here touches the whole-slide napari image: five different ways of
locating a FOV inside it all failed, and ComSeg does not need pixels anyway.

    conda activate comseg_env
    python comseg_cosmx_3.py --fovs 48 261 --stage prep
    python comseg_cosmx_3.py --fovs 48 --stage run
    python comseg_cosmx_3.py --fovs 48 --stage post --compare-to /data/qiuyijia/ucs_cosmx_lymph_node/fov48
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "2026.08.13"

ROOT = "/data/qiuyijia/dataset/cosmx_human_brain"
OUT_ROOT = "/data/qiuyijia/comseg_cosmx_human_brain"
UM_PER_PX = 0.12028          # verified two ways: FOV position table and Area.um2/Area
FOV_PX = 4256                # .zattrs fov_height / fov_width

log = logging.getLogger("comseg_cosmx")


def setup_log(d: Path, tag: str):
    d.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    f = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(f); log.addHandler(sh)
    fh = logging.FileHandler(d / f"{tag}.log", mode="a"); fh.setFormatter(f)
    log.addHandler(fh)


def rule(msg=""):
    log.info("=" * 78)
    if msg:
        log.info(msg)
        log.info("=" * 78)


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


def _write_tab(df: pd.DataFrame, path: Path):
    """parquet when an engine is available, csv.gz otherwise."""
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception as e:
        alt = path.with_suffix(".csv.gz")
        log.info(f"    (no parquet engine: {e.__class__.__name__}; writing {alt.name})")
        df.to_csv(alt, index=False)
        return alt


def _read_tab(path: Path):
    for p in (path, path.with_suffix(".csv.gz"), path.with_suffix(".csv")):
        if p.exists():
            return pd.read_csv(p) if str(p).endswith((".csv", ".csv.gz")) \
                else pd.read_parquet(p)
    return None


def flat_files(root: Path) -> dict:
    out = {}
    for key, pat in (("tx", "*tx_file*"), ("meta", "*metadata_file*"),
                     ("fov", "*fov_positions*"), ("poly", "*polygons*")):
        hits = sorted((root / "flat_files").glob(pat)) or sorted(root.rglob(pat))
        if hits:
            out[key] = hits[0]
    return out


def load_fov_transcripts(root: Path, fovs, cache_dir: Path) -> pd.DataFrame:
    """Subset the 1.4-billion-row transcript table once, then reuse the cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = "_".join(str(f) for f in sorted(fovs))
    for t in (tag, "48_261"):
        df = _read_tab(cache_dir / f"tx_fov_{t}.parquet")
        if df is not None:
            log.info(f"transcript cache tx_fov_{t}")
            return df[df["fov"].isin(fovs)].reset_index(drop=True)

    ff = flat_files(root)
    log.info(f"scanning {ff['tx']} (this happens once, then it is cached)")
    parts, scanned = [], 0
    for chunk in pd.read_csv(ff["tx"], chunksize=5_000_000):
        scanned += len(chunk)
        sel = chunk[chunk["fov"].isin(fovs)]
        if len(sel):
            parts.append(sel)
        if scanned % 100_000_000 < 5_000_000:
            log.info(f"  scanned {scanned:,}")
    df = pd.concat(parts, ignore_index=True)
    out = _write_tab(df, cache_dir / f"tx_fov_{tag}.parquet")
    log.info(f"  {scanned:,} rows scanned, {len(df):,} kept -> {out}")
    return df


def load_fov_metadata(root: Path, fov: int) -> pd.DataFrame:
    ff = flat_files(root)
    md = pd.read_csv(ff["meta"])
    md = md[md["fov"] == fov].copy()
    return md


# ---------------------------------------------------------------------------
# prep
# ---------------------------------------------------------------------------


def tile_starts(size: int, tile: int, overlap: int):
    step = tile - overlap
    s = list(range(0, max(size - tile, 0) + 1, step))
    if not s or s[-1] + tile < size:
        s.append(max(size - tile, 0))
    return s


def core_bounds(starts, tile, size):
    """Mid-line split so the overlap is covered exactly once."""
    out = []
    for i, s in enumerate(starts):
        lo = s if i == 0 else (starts[i - 1] + tile + s) // 2
        hi = s + tile if i == len(starts) - 1 else (s + tile + starts[i + 1]) // 2
        out.append((max(lo, 0), min(hi, size)))
    return out


def stage_prep(fov: int, out: Path, args):
    rule(f"PREP  FOV {fov}")
    root = Path(args.root)
    cache = root / "_inspect_cache"
    tx = load_fov_transcripts(root, [fov], cache)
    log.info(f"transcripts {len(tx):,}   columns {list(tx.columns)}")

    xcol = "x_local_px" if "x_local_px" in tx.columns else "x"
    ycol = "y_local_px" if "y_local_px" in tx.columns else "y"
    gcol = "target" if "target" in tx.columns else "gene"
    tx = tx[(tx[xcol] >= 0) & (tx[xcol] < FOV_PX)
            & (tx[ycol] >= 0) & (tx[ycol] < FOV_PX)].reset_index(drop=True)
    log.info(f"  inside the FOV frame: {len(tx):,}   targets {tx[gcol].nunique():,}")
    log.info(f"  density {len(tx)/(FOV_PX*UM_PER_PX)**2:.2f} tx/um^2")

    # drop the control probes
    ctrl = tx[gcol].astype(str).str.startswith(("Negative", "SystemControl",
                                               "NegPrb", "Custom"))
    if ctrl.any():
        log.info(f"  dropping {int(ctrl.sum()):,} control probes")
        tx = tx[~ctrl].reset_index(drop=True)

    md = load_fov_metadata(root, fov)
    log.info(f"metadata    {len(md):,} cells in this FOV")
    acol = "NucArea" if "NucArea" in md.columns else "Area"
    r_px = np.sqrt(pd.to_numeric(md[acol], errors="coerce").fillna(0) / np.pi)
    log.info(f"  nucleus radius from {acol}: median {r_px.median():.1f} px "
             f"= {r_px.median()*UM_PER_PX:.2f} um")
    cell_diam_um = 2 * np.sqrt(pd.to_numeric(md["Area"], errors="coerce")
                               .median() / np.pi) * UM_PER_PX
    log.info(f"  platform cell diameter {cell_diam_um:.2f} um")

    starts = tile_starts(FOV_PX, args.tile_px, args.overlap_px)
    cores_y = core_bounds(starts, args.tile_px, FOV_PX)
    cores_x = core_bounds(starts, args.tile_px, FOV_PX)
    log.info(f"tiles       {len(starts)}x{len(starts)} = {len(starts)**2}, "
             f"{args.tile_px} px with {args.overlap_px} px overlap")

    if args.overlap_px * UM_PER_PX < args.max_cell_radius_um:
        log.info(f"  WARNING: overlap {args.overlap_px*UM_PER_PX:.1f} um is smaller "
                 f"than --max-cell-radius-um {args.max_cell_radius_um}; cells that "
                 "straddle a seam will be cut")

    sp_dir = out / "input" / "spots"
    mk_dir = out / "input" / "masks"
    for d in (sp_dir, mk_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    import tifffile
    mb = args.mask_bin
    rows = []
    idx = 0
    for iy, y0 in enumerate(starts):
        for ix, x0 in enumerate(starts):
            y1, x1 = y0 + args.tile_px, x0 + args.tile_px
            sel = tx[(tx[ycol] >= y0) & (tx[ycol] < y1)
                     & (tx[xcol] >= x0) & (tx[xcol] < x1)]
            name = f"tile_{idx:04d}"

            # ComSeg reads the spot coordinates in MASK pixels, so the same
            # binning is applied to the spots and to the mask.
            sub = pd.DataFrame({
                "x": (sel[xcol].to_numpy(float) - x0) / mb,
                "y": (sel[ycol].to_numpy(float) - y0) / mb,
                "z": 0.0,
                "gene": sel[gcol].astype(str).to_numpy(),
            })
            sub.to_csv(sp_dir / f"{name}.csv", index=False)

            # one nucleus mask per tile -- this is what creates dict_centroid
            g = int(np.ceil(args.tile_px / mb))
            mask = np.zeros((g, g), np.uint32)
            inside = md[(md["CenterY_local_px"] >= y0 - 200)
                        & (md["CenterY_local_px"] < y1 + 200)
                        & (md["CenterX_local_px"] >= x0 - 200)
                        & (md["CenterX_local_px"] < x1 + 200)]
            n_painted = 0
            for cid, cx, cy, rr in zip(
                    pd.to_numeric(inside["cell_ID"], errors="coerce").fillna(0)
                    .astype(np.int64),
                    inside["CenterX_local_px"].to_numpy(float),
                    inside["CenterY_local_px"].to_numpy(float),
                    np.sqrt(pd.to_numeric(inside[acol], errors="coerce").fillna(0)
                            / np.pi).to_numpy(float)):
                if cid <= 0:
                    continue
                px = (cx - x0) / mb
                py = (cy - y0) / mb
                rad = max(int(round(rr / mb)), 1)
                # painted with numpy rather than cv2: cv2 drawing rejects
                # uint32/int32 label images, and cell ids here exceed uint16
                lo_y, hi_y = max(int(py) - rad, 0), min(int(py) + rad + 1, g)
                lo_x, hi_x = max(int(px) - rad, 0), min(int(px) + rad + 1, g)
                if lo_y >= hi_y or lo_x >= hi_x:
                    continue
                yy = np.arange(lo_y, hi_y)[:, None] - py
                xx = np.arange(lo_x, hi_x)[None, :] - px
                disk = (yy * yy + xx * xx) <= rad * rad
                sub_m = mask[lo_y:hi_y, lo_x:hi_x]
                sub_m[disk] = int(cid)
                n_painted += 1
            tifffile.imwrite(mk_dir / f"{name}.tif", mask)

            n_nuc = int(np.unique(mask[mask > 0]).size)
            in_nuc = 0
            if len(sub):
                ri = np.clip(np.rint(sub["y"]).astype(int), 0, g - 1)
                ci = np.clip(np.rint(sub["x"]).astype(int), 0, g - 1)
                in_nuc = int((mask[ri, ci] > 0).sum())
            rows.append(dict(tile=name, y0=y0, x0=x0,
                             core_y0=cores_y[iy][0], core_y1=cores_y[iy][1],
                             core_x0=cores_x[ix][0], core_x1=cores_x[ix][1],
                             spots=len(sub), nuclei=n_nuc,
                             in_nucleus_frac=round(in_nuc / max(len(sub), 1), 4)))
            log.info(f"  {name}: {len(sub):,} spots, {n_nuc:,} nuclei, "
                     f"{in_nuc/max(len(sub),1)*100:.1f}% inside a nucleus")
            idx += 1

    meta = pd.DataFrame(rows)
    (out / "input").mkdir(parents=True, exist_ok=True)
    meta.to_csv(out / "input" / "tile_metadata.tsv", sep="\t", index=False)

    biggest = meta.spots.max()
    log.info(f"largest tile {biggest:,} spots")
    if biggest > 1_500_000:
        log.info("  WARNING: that is a lot for one ComSeg graph; consider a "
                 "smaller --tile-px")

    json.dump(dict(version=VERSION, fov=fov, fov_px=FOV_PX,
                   um_per_px=UM_PER_PX, mask_bin=mb,
                   tile_px=args.tile_px, overlap_px=args.overlap_px,
                   n_tiles=len(rows), platform_cells=int(len(md)),
                   platform_cell_diameter_um=round(float(cell_diam_um), 3)),
              open(out / "roi_meta.json", "w"), indent=2)

    rule("PREP DONE")
    log.info(f"tiles {len(rows)}   platform cells {len(md):,}")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def call_filtered(fn, **kw):
    """Pass only the keywords this build of ComSeg actually declares."""
    sig = inspect.signature(fn)
    keep = {k: v for k, v in kw.items() if k in sig.parameters}
    dropped = [k for k in kw if k not in sig.parameters]
    if dropped:
        log.info(f"    ({fn.__name__} does not take {dropped}, dropped)")
    return fn(**keep)


def show_api():
    from comseg import dataset as ds, dictionary as di
    log.info("ComSeg API in this environment:")
    for label, fn in (("ComSegDataset.__init__", ds.ComSegDataset.__init__),
                      ("add_prior_from_mask", ds.ComSegDataset.add_prior_from_mask),
                      ("compute_edge_weight", ds.ComSegDataset.compute_edge_weight),
                      ("ComSegDict.__init__", di.ComSegDict.__init__),
                      ("run_all", di.ComSegDict.run_all)):
        try:
            log.info(f"  {label}{inspect.signature(fn)}")
        except (TypeError, ValueError):
            log.info(f"  {label}  (signature unavailable)")


def stage_run(fov: int, out: Path, args):
    rule(f"RUN  FOV {fov}")
    meta_p = out / "roi_meta.json"
    if not meta_p.exists():
        raise SystemExit(f"{meta_p} 不存在 —— 先跑 --stage prep（日志见 {out}/logs）")
    meta = json.loads(meta_p.read_text())
    tiles = pd.read_csv(out / "input" / "tile_metadata.tsv", sep="\t")

    from comseg import dataset as ds, dictionary as di
    show_api()

    mb = meta["mask_bin"]
    scale = UM_PER_PX * mb                       # um per mask pixel
    dict_scale = {"x": scale, "y": scale, "z": scale}
    mean_diam = args.mean_cell_diameter_um or meta["platform_cell_diameter_um"]
    log.info(f"dict_scale  {scale:.5f} um per mask px")
    log.info(f"mean cell diameter {mean_diam:.2f} um, "
             f"max cell radius {args.max_cell_radius_um} um")

    work = out / "comseg_out"
    work.mkdir(parents=True, exist_ok=True)
    n_done = n_fail = 0

    for _, t in tiles.iterrows():
        name = t["tile"]
        tdir = work / name
        if (tdir / "DONE.flag").exists() and not args.force_run:
            n_done += 1
            continue
        if (tdir / "FAILED.flag").exists():
            (tdir / "FAILED.flag").unlink()
        tdir.mkdir(parents=True, exist_ok=True)

        # ComSeg wants a folder holding the csv, with the masks alongside
        stage = tdir / "in"
        if stage.exists():
            shutil.rmtree(stage)
        (stage / "masks").mkdir(parents=True)
        shutil.copy2(out / "input" / "spots" / f"{name}.csv", stage / f"{name}.csv")
        shutil.copy2(out / "input" / "masks" / f"{name}.tif",
                     stage / "masks" / f"{name}.tif")

        log.info(f"{name}: {int(t['spots']):,} spots, {int(t['nuclei']):,} nuclei")
        try:
            dataset = call_filtered(
                ds.ComSegDataset,
                path_dataset_folder=str(stage),
                path_to_mask_prior=str(stage / "masks"),
                dict_scale=dict_scale,
                mask_file_extension=".tif",
                mean_cell_diameter=mean_diam,
                prior_name=args.prior_name,
            )

            # THE fix: this is the only call that creates dataset.dict_centroid,
            # which run_all -> classify_centroid requires.
            call_filtered(dataset.add_prior_from_mask,
                          overwrite=True, compute_centroid=True,
                          regex_df="*.csv")
            dc = getattr(dataset, "dict_centroid", None)
            if dc is None:
                raise RuntimeError(
                    "add_prior_from_mask did not produce dict_centroid; the mask "
                    "and the csv may not share a basename")
            log.info(f"    prior added, centroids for {len(dc)} image(s)")

            call_filtered(dataset.compute_edge_weight,
                          n_neighbors=args.n_neighbors, distance="pearson",
                          sampling=True, sampling_size=args.sampling_size)

            cd = call_filtered(di.ComSegDict, dataset=dataset,
                               mean_cell_diameter=mean_diam,
                               community_detection=args.community_detection)

            call_filtered(cd.run_all,
                          max_cell_radius=args.max_cell_radius_um,
                          k_nearest_neighbors=args.k_nearest_neighbors,
                          norm_vector=args.norm_vector,
                          clustering_method="leiden",
                          disable_tqdm=True)

            ad = call_filtered(cd.anndata_from_comseg_result,
                               min_rna_per_cell=args.min_rna_per_cell,
                               return_polygon=False)
            save_tile(ad, cd, tdir, name)
            if not (tdir / "spot_assignment.parquet").exists() and \
                    not (tdir / "spot_assignment.csv.gz").exists():
                log.info("    tile left WITHOUT a per-RNA table; not marking it done")
            else:
                (tdir / "DONE.flag").touch()
            n_done += 1
            log.info(f"    done, {tdir/'spot_assignment.parquet'}")
        except Exception:
            tb = traceback.format_exc()
            (tdir / "FAILED.flag").write_text(tb)
            n_fail += 1
            log.info(f"  {name} failed:\n{tb.rstrip()}")
            if args.stop_on_first_failure:
                raise
        log.info("")

    rule("RUN DONE")
    log.info(f"{n_done} tiles done, {n_fail} failed of {len(tiles)}")
    if n_fail:
        log.info(f"tracebacks are in {work}/tile_*/FAILED.flag")


def save_tile(ad, cd, tdir: Path, name: str):
    """
    Persist whatever ComSeg produced, and never raise.

    ComSegDict is a dict of image name -> per-image graph object, so the per-RNA
    table lives at cd[name].df_spots_label, not on the dict itself.  The previous
    version only looked at the dict's own attributes, so a tile that took 45
    minutes to compute was discarded at the last step.  This version searches
    both levels, dumps a description of the object when it finds nothing, and
    still writes the flag -- losing the description is what makes the next
    attempt cost another 45 minutes.
    """
    found = []

    def harvest(obj, where):
        for attr in ("df_spots_label", "df_spots", "df", "spots"):
            df = getattr(obj, attr, None)
            if isinstance(df, pd.DataFrame) and {"x", "y"} <= set(df.columns):
                found.append((f"{where}.{attr}", df))

    harvest(cd, "dict")
    try:
        for k in list(cd.keys()):
            v = cd[k]
            harvest(v, f"dict[{k!r}]")
            harvest(getattr(v, "dataset", object()), f"dict[{k!r}].dataset")
    except Exception as e:
        log.info(f"    (could not iterate the ComSegDict: {e})")
    harvest(getattr(cd, "dataset", object()), "dict.dataset")

    if found:
        where, df = max(found, key=lambda t: len(t[1]))
        cols = [c for c in df.columns]
        log.info(f"    per-RNA table from {where}: {len(df):,} rows, columns {cols}")
        _write_tab(df.reset_index(drop=True), tdir / "spot_assignment.parquet")
    else:
        # describe the object so the next run does not have to rediscover it
        desc = {"dict_type": type(cd).__name__,
                "dict_attrs": [a for a in dir(cd) if not a.startswith("_")][:60],
                "keys": [str(k) for k in list(getattr(cd, "keys", list)())][:10]}
        try:
            k0 = list(cd.keys())[0]
            v0 = cd[k0]
            desc["value_type"] = type(v0).__name__
            desc["value_attrs"] = [a for a in dir(v0) if not a.startswith("_")][:60]
            desc["value_frames"] = {
                a: list(getattr(v0, a).columns)
                for a in dir(v0) if not a.startswith("_")
                and isinstance(getattr(v0, a, None), pd.DataFrame)}
        except Exception as e:
            desc["value_error"] = str(e)
        (tdir / "comseg_object.json").write_text(json.dumps(desc, indent=2,
                                                            default=str))
        log.info(f"    !! no per-RNA table found. The object layout has been "
                 f"written to {tdir/'comseg_object.json'} -- send it and the fix "
                 "costs no further compute.")

    # the anndata is cheap insurance either way
    if ad is not None:
        try:
            ad.write_h5ad(tdir / "cells.h5ad")
        except Exception:
            pass
        try:
            ad.obs.to_csv(tdir / "cell_obs.csv")
        except Exception:
            pass
    try:
        cd.save(str(tdir / "comseg_dict"))
        log.info(f"    ComSegDict saved to {tdir/'comseg_dict'}")
    except Exception as e:
        log.info(f"    (cd.save unavailable: {e.__class__.__name__})")


# ---------------------------------------------------------------------------
# post
# ---------------------------------------------------------------------------


def stage_post(fov: int, out: Path, args):
    rule(f"POST  FOV {fov}")
    meta = json.loads((out / "roi_meta.json").read_text())
    tiles = pd.read_csv(out / "input" / "tile_metadata.tsv", sep="\t")
    mb = meta["mask_bin"]
    work = out / "comseg_out"

    frames = []
    for _, t in tiles.iterrows():
        df = _read_tab(work / t["tile"] / "spot_assignment.parquet")
        if df is None:
            continue
        lab = None
        for c in ("cell_index_pred", "cell", "cell_id", "index_commu_in_nucleus",
                  "cell_index"):
            if c in df.columns:
                lab = c
                break
        if lab is None:
            log.info(f"  {t['tile']}: no label column in {list(df.columns)}")
            continue
        # back to FOV-local pixels, then keep only the core region so the
        # overlap is counted exactly once
        gx = df["x"].to_numpy(float) * mb + t["x0"]
        gy = df["y"].to_numpy(float) * mb + t["y0"]
        keep = ((gx >= t["core_x0"]) & (gx < t["core_x1"])
                & (gy >= t["core_y0"]) & (gy < t["core_y1"]))
        sub = pd.DataFrame({"x": gx[keep], "y": gy[keep],
                            "label": pd.to_numeric(df[lab], errors="coerce")
                            .fillna(0).to_numpy()[keep],
                            "gene": df["gene"].to_numpy()[keep]
                            if "gene" in df.columns else "NA",
                            "tile": t["tile"]})
        sub = sub[sub.label > 0]
        # make the ids unique across tiles
        sub["label"] = sub["label"].astype(np.int64) + 1_000_000 * int(
            t["tile"].split("_")[1])
        frames.append(sub)
        log.info(f"  {t['tile']}: {len(sub):,} assigned spots in its core")

    if not frames:
        raise SystemExit("no tile produced anything; see the FAILED.flag files")
    allsp = pd.concat(frames, ignore_index=True)
    codes, uniq = pd.factorize(allsp["label"])
    allsp["cell"] = codes + 1
    n_cells = len(uniq)
    log.info(f"cells       {n_cells:,}   assigned spots {len(allsp):,}")

    # rasterise on the shared grid so it can be compared with UCS / Proseg
    import tifffile
    from scipy.ndimage import distance_transform_edt
    g = args.raster_grid
    s = FOV_PX / g
    lab = np.zeros((g, g), np.int32)
    ri = np.clip((allsp.y.to_numpy() / s).astype(int), 0, g - 1)
    ci = np.clip((allsp.x.to_numpy() / s).astype(int), 0, g - 1)
    lab[ri, ci] = allsp.cell.to_numpy()
    occ0 = (lab > 0).mean()
    # ComSeg labels spots, not pixels: grow them so area is a footprint, not a
    # scatter (the same correction the Cellist raster needs)
    r = args.fill_um / (s * UM_PER_PX)
    if r > 0:
        d, (iy, ix) = distance_transform_edt(lab == 0, return_indices=True)
        grow = (lab == 0) & (d <= r)
        lab[grow] = lab[iy[grow], ix[grow]]
    log.info(f"raster {g}x{g} at {s*UM_PER_PX:.3f} um/px: occupancy "
             f"{occ0*100:.2f}% -> {(lab>0).mean()*100:.2f}% after {args.fill_um} um fill")
    tifffile.imwrite(out / "comseg_labels_native.tif", lab)

    area = np.bincount(lab.reshape(-1))[1:]
    area = area[area > 0] * (s * UM_PER_PX) ** 2
    med = float(np.median(area))
    log.info(f"cells {n_cells:,}   platform {meta['platform_cells']:,}   "
             f"ratio {n_cells/meta['platform_cells']:.3f}")
    log.info(f"median area {med:.1f} um^2   equivalent diameter "
             f"{2*np.sqrt(med/np.pi):.2f} um")

    cnt = allsp.groupby(["cell", "gene"]).size().unstack(fill_value=0)
    np.save(out / "cell_by_gene.npy", cnt.to_numpy())
    cnt.to_csv(out / "cell_by_gene.csv.gz")
    log.info(f"matrix {cnt.shape}   counts/cell median "
             f"{cnt.sum(axis=1).median():.0f}")

    if args.compare_to:
        other = Path(args.compare_to)
        cand = list(other.glob("*_labels_native.tif")) + \
            list(other.glob("*segmentation*.tif"))
        if cand:
            o = tifffile.imread(cand[0])
            if o.shape != lab.shape:
                from scipy.ndimage import zoom
                o = zoom((o > 0).astype(np.uint8), g / o.shape[0], order=0)
            a, b = lab > 0, o > 0
            dice = 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)
            log.info(f"vs {cand[0].name}: foreground Dice {dice:.3f} "
                     f"(coverage {a.mean()*100:.1f}% vs {b.mean()*100:.1f}%)")
            log.info("  note: with very different coverages this Dice mostly "
                     "restates the coverage gap, not boundary agreement")

    rule("POST DONE")


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fovs", nargs="+", type=int, required=True)
    ap.add_argument("--stage", default="all",
                    choices=["prep", "run", "post", "all", "api"])
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--tile-px", type=int, default=1064)
    ap.add_argument("--overlap-px", type=int, default=128)
    ap.add_argument("--mask-bin", type=int, default=2,
                    help="mask/spot pixel size in native px (2 = 0.24 um)")
    ap.add_argument("--mean-cell-diameter-um", type=float, default=None,
                    help="default: taken from the platform Area median")
    ap.add_argument("--max-cell-radius-um", type=float, default=12.0)
    ap.add_argument("--k-nearest-neighbors", type=int, default=10)
    ap.add_argument("--n-neighbors", type=int, default=40)
    ap.add_argument("--sampling-size", type=int, default=10000)
    ap.add_argument("--norm-vector", action="store_true")
    ap.add_argument("--community-detection", default="with_prior")
    ap.add_argument("--prior-name", default="in_nucleus")
    ap.add_argument("--min-rna-per-cell", type=int, default=5)
    ap.add_argument("--raster-grid", type=int, default=2128)
    ap.add_argument("--fill-um", type=float, default=1.0)
    ap.add_argument("--compare-to", default=None)
    ap.add_argument("--force-run", action="store_true")
    ap.add_argument("--stop-on-first-failure", action="store_true")
    ap.add_argument("--version", action="store_true")
    args = ap.parse_args()

    if args.version:
        print(VERSION)
        return

    for fov in args.fovs:
        out = Path(args.out_root) / f"fov{fov}"
        setup_log(out / "logs", f"comseg_cosmx_fov{fov}_{args.stage}")
        rule(f"COMSEG v{VERSION}  x  CosMx FOV {fov}   out {out}")
        if args.stage == "api":
            show_api()
            continue
        if args.stage in ("prep", "all"):
            stage_prep(fov, out, args)
        if args.stage in ("run", "all"):
            stage_run(fov, out, args)
        if args.stage in ("post", "all"):
            stage_post(fov, out, args)


if __name__ == "__main__":
    main()