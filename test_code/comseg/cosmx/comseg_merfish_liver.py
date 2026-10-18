#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ComSeg × MERFISH liver1 / liver2
================================

模式：
    官方 cell-boundary prior + transcript cell_id prior
    即之前使用的 ComSeg with-prior 模式。

本版本只做已确认的修正：

1. P() 支持 end= 等 print 参数。
2. centroid z 恢复之前老代码的约定：
       z = cell_id * 0.001
3. community_detection 保持 "with_prior"。
4. min_merge_correlation 默认保持 0.99。
5. 其它 ComSeg 参数保持当前 liver 实验方案。

运行：
    python comseg_merfish_liver.py --dataset liver1 --force
    python comseg_merfish_liver.py --dataset liver2 --force
"""

import os
import time
import pickle
import inspect
import argparse

import numpy as np
import pandas as pd
import tifffile
import skimage.measure


# ============================================================
# basic
# ============================================================

def P(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


SEP = "=" * 72


CFG = {

    "liver1": {

        "base":
            "/data/qiuyijia/dataset/merfish_liver1",

        "mask":
            "/data/qiuyijia/ucs_merfish_liver1/nuclei_mask.tif",

        "x0": 5236.737991920782,
        "x1": 6316.729987066785,

        "y0": 3811.048683925526,
        "y1": 4891.037342034139,
    },

    "liver2": {

        "base":
            "/data/qiuyijia/dataset/merfish_liver2",

        "mask":
            "/data/qiuyijia/ucs_merfish_liver2/nuclei_mask.tif",

        "x0": 5771.0386869252825,
        "x1": 6851.029458488514,

        "y0": 4970.975379467495,
        "y1": 6050.964816216321,
    },
}


# ============================================================
# compatibility
# ============================================================

def safe_kwargs(func, kwargs, name):

    try:
        sig = inspect.signature(func)

        valid = {
            k: v
            for k, v in kwargs.items()
            if k in sig.parameters
        }

        ignored = sorted(
            set(kwargs) - set(valid)
        )

        if ignored:
            P(
                f"  [{name}] 当前 ComSeg 版本"
                f"不接受参数：{ignored}"
            )

        return valid

    except Exception:
        return kwargs


# ============================================================
# ComSeg centroid cleanup
# ============================================================

def kill_dict_centroid(*objs):

    seen = set()

    def walk(obj):

        if obj is None:
            return

        if id(obj) in seen:
            return

        seen.add(id(obj))

        if hasattr(obj, "dict_centroid"):

            try:

                old = getattr(
                    obj,
                    "dict_centroid"
                )

                if isinstance(old, dict):
                    n = len(old)
                else:
                    n = "?"

                P(
                    f"    clear "
                    f"{type(obj).__name__}"
                    f".dict_centroid "
                    f"(old={n})"
                )

                obj.dict_centroid = None

            except Exception:
                pass

        if isinstance(obj, dict):

            for v in obj.values():
                walk(v)

        elif isinstance(obj, (list, tuple)):

            for v in obj:
                walk(v)

        else:

            for attr in [
                "dataset",
                "dict_in_situ",
                "dict_image",
            ]:

                if hasattr(obj, attr):
                    try:
                        walk(
                            getattr(obj, attr)
                        )
                    except Exception:
                        pass

    for obj in objs:
        walk(obj)


# ============================================================
# sentinel
# ============================================================

def ensure_sentinel(
    nuc,
    max_cell_radius
):

    if (nuc["cell_id"] == 0).any():
        return nuc

    margin = (
        float(max_cell_radius)
        * 20.0
    )

    sx = (
        float(nuc["x"].min())
        - margin
    )

    sy = (
        float(nuc["y"].min())
        - margin
    )

    sentinel = pd.DataFrame([
        {
            "cell_id": 0,
            "x": sx,
            "y": sy,
            "z": 0.0,
        }
    ])

    return pd.concat(
        [sentinel, nuc],
        ignore_index=True
    )


# ============================================================
# paths
# ============================================================

def get_dirs(dataset):

    root = (
        f"/data/qiuyijia/comseg/"
        f"merfish_{dataset}"
    )

    d = {

        "root":
            root,

        "cache":
            os.path.join(
                root,
                "cache"
            ),

        "centroid":
            os.path.join(
                root,
                "centroid_only"
            ),

        "out":
            os.path.join(
                root,
                "output"
            ),
    }

    for p in d.values():
        os.makedirs(
            p,
            exist_ok=True
        )

    return d


# ============================================================
# prepare data
# ============================================================

def prepare_data(
    cfg,
    dataset,
    dirs
):

    c = CFG[dataset]

    mask_path = c["mask"]

    if not os.path.isfile(mask_path):

        raise FileNotFoundError(
            f"prior mask 不存在："
            f"{mask_path}"
        )


    mask = tifffile.imread(
        mask_path
    ).astype(
        np.int32
    )


    ids = np.unique(mask)
    ids = ids[ids > 0]


    P(
        f"  prior mask: "
        f"{mask.shape}, "
        f"cells={len(ids):,}"
    )


    tag = (
        f"merfish_{dataset}_roi"
    )


    tx_out = os.path.join(
        dirs["cache"],
        f"{tag}.csv"
    )


    cen_out = os.path.join(
        dirs["cache"],
        f"{tag}_centroid.csv"
    )


    # ========================================================
    # centroid prior
    # ========================================================

    if (
        cfg.force
        or
        not os.path.isfile(cen_out)
    ):

        P(
            "\n[1a] 构建 centroid prior"
        )


        props = (
            skimage.measure
            .regionprops_table(
                mask,
                properties=[
                    "label",
                    "centroid"
                ]
            )
        )


        prop = pd.DataFrame(
            props
        )


        # nuclei_mask 的建立方式与 UCS 相同：
        # local x = round(global_x) - int(ROI_X0)
        # local y = round(global_y) - int(ROI_Y0)

        ix0 = int(
            c["x0"]
        )

        iy0 = int(
            c["y0"]
        )


        nuc = pd.DataFrame({

            "cell_id":
                prop["label"]
                .astype(int),

            # centroid-1 = column = x
            "x":
                prop[
                    "centroid-1"
                ].values
                + ix0,

            # centroid-0 = row = y
            "y":
                prop[
                    "centroid-0"
                ].values
                + iy0,
        })


        # ====================================================
        # ★ 恢复之前老 ComSeg with-prior 代码的约定
        #
        # 不再使用 3 + tiny jitter。
        # ====================================================

        nuc["z"] = (
            nuc["cell_id"]
            .astype(float)
            * 0.001
        )


        nuc = ensure_sentinel(
            nuc,
            cfg.max_cell_radius
        )


        nuc.to_csv(
            cen_out,
            index=False
        )


        P(
            f"    real centroids = "
            f"{int((nuc.cell_id>0).sum()):,}"
        )

        P(
            f"    z range(real) = "
            f"{nuc.loc[nuc.cell_id>0,'z'].min():.3f}"
            f" ~ "
            f"{nuc.loc[nuc.cell_id>0,'z'].max():.3f}"
        )

        P(
            f"    -> {cen_out}"
        )

    else:

        nuc = pd.read_csv(
            cen_out
        )

        P(
            f"\n[1a] reuse centroid:"
            f" {len(nuc):,} rows"
        )


    # ========================================================
    # transcript prior
    # ========================================================

    if (
        cfg.force
        or
        not os.path.isfile(tx_out)
    ):

        tx_src = os.path.join(
            c["base"],
            "detected_transcripts.csv"
        )


        if not os.path.isfile(tx_src):
            raise FileNotFoundError(
                tx_src
            )


        ix0 = int(
            c["x0"]
        )

        iy0 = int(
            c["y0"]
        )

        H, W = mask.shape


        # ----------------------------------------------------
        # pass 1: count genes
        # ----------------------------------------------------

        P(
            "\n[1b] pass1: "
            "统计 ROI 基因丰度"
        )


        gene_counts = {}

        total = 0
        roi_total = 0


        for chunk in pd.read_csv(
            tx_src,
            usecols=[
                "global_x",
                "global_y",
                "gene",
            ],
            chunksize=1_000_000
        ):

            total += len(chunk)

            m = (
                (chunk.global_x >= c["x0"])
                &
                (chunk.global_x < c["x1"])
                &
                (chunk.global_y >= c["y0"])
                &
                (chunk.global_y < c["y1"])
            )


            sub = (
                chunk.loc[
                    m,
                    "gene"
                ]
                .astype(str)
            )


            roi_total += len(sub)


            bad = (
                sub.str.startswith(
                    (
                        "Blank",
                        "blank",
                        "BLANK",
                    ),
                    na=False
                )
            )

            sub = sub[
                ~bad
            ]


            vc = (
                sub.value_counts()
            )


            for gene, n in vc.items():

                gene_counts[gene] = (
                    gene_counts.get(
                        gene,
                        0
                    )
                    + int(n)
                )


            P(
                f"    raw {total:,}"
                f" | ROI {roi_total:,}",
                end="\r"
            )


        P()


        ordered_genes = sorted(
            gene_counts,
            key=gene_counts.get,
            reverse=True
        )


        keep_genes = set(
            ordered_genes[
                :cfg.top_n
            ]
        )


        P(
            f"    ROI genes = "
            f"{len(ordered_genes):,}"
        )

        P(
            f"    使用 top-"
            f"{len(keep_genes)}"
        )


        # ----------------------------------------------------
        # pass 2: output transcript csv
        # ----------------------------------------------------

        P(
            "\n[1c] pass2: "
            "生成 ComSeg transcript CSV"
        )


        first = True

        n_raw = 0
        n_roi = 0
        n_keep = 0
        n_prior = 0


        for chunk in pd.read_csv(
            tx_src,
            usecols=[
                "global_x",
                "global_y",
                "global_z",
                "gene",
            ],
            chunksize=1_000_000
        ):

            n_raw += len(chunk)


            m = (
                (chunk.global_x >= c["x0"])
                &
                (chunk.global_x < c["x1"])
                &
                (chunk.global_y >= c["y0"])
                &
                (chunk.global_y < c["y1"])
            )


            sub = chunk.loc[
                m
            ].copy()


            n_roi += len(sub)


            if len(sub) == 0:
                continue


            sub["gene"] = (
                sub["gene"]
                .astype(str)
            )


            sub = sub[
                sub["gene"]
                .isin(keep_genes)
            ].copy()


            if len(sub) == 0:
                continue


            bx = (
                sub.global_x
                .round()
                .astype(int)
                .values
                - ix0
            )


            by = (
                sub.global_y
                .round()
                .astype(int)
                .values
                - iy0
            )


            ok = (
                (bx >= 0)
                &
                (bx < W)
                &
                (by >= 0)
                &
                (by < H)
            )


            cell_id = np.zeros(
                len(sub),
                dtype=np.int32
            )


            cell_id[ok] = (
                mask[
                    by[ok],
                    bx[ok]
                ]
            )


            out = pd.DataFrame({

                "x":
                    sub.global_x
                    .values
                    .astype(float),

                "y":
                    sub.global_y
                    .values
                    .astype(float),

                "z":
                    sub.global_z
                    .values
                    .astype(float),

                "gene":
                    sub.gene.values,

                "cell_id":
                    cell_id,
            })


            n_keep += len(out)

            n_prior += int(
                (cell_id > 0)
                .sum()
            )


            out.to_csv(
                tx_out,
                mode=(
                    "w"
                    if first
                    else "a"
                ),
                header=first,
                index=False
            )


            first = False


            P(
                f"    raw {n_raw:,}"
                f" | ROI {n_roi:,}"
                f" | topN {n_keep:,}"
                f" | assigned {n_prior:,}",
                end="\r"
            )


        P()


        if first:
            raise RuntimeError(
                "没有生成 transcript CSV"
            )


        P(
            f"    ROI transcripts = "
            f"{n_roi:,}"
        )

        P(
            f"    retained top-N = "
            f"{n_keep:,}"
        )

        P(
            f"    prior assigned = "
            f"{n_prior:,}"
        )

        P(
            f"    assignment rate = "
            f"{n_prior/max(n_keep,1)*100:.2f}%"
        )

        P(
            f"    -> {tx_out}"
        )

    else:

        P(
            f"\n[1b] reuse transcript:"
            f"\n    {tx_out}"
        )


    return (
        tag,
        tx_out,
        cen_out,
        nuc
    )


# ============================================================
# run ComSeg
# ============================================================

def run_comseg(
    tag,
    txp,
    cenp,
    nuc,
    cfg,
    dirs
):

    from comseg.dataset import (
        ComSegDataset
    )

    from comseg.dictionary import (
        ComSegDict
    )


    img_name = os.path.splitext(
        os.path.basename(txp)
    )[0]


    P(
        "\n[2] ComSegDataset"
    )


    ds_kw = safe_kwargs(
        ComSegDataset.__init__,
        dict(

            path_dataset_folder=
                dirs["cache"],

            prior_name=
                "cell_id",

            image_csv_files=[
                os.path.basename(txp)
            ],

            centroid_csv_files=[
                os.path.basename(cenp)
            ],

            path_cell_centroid=
                dirs["cache"],

            dict_scale={
                "x": 1,
                "y": 1,
                "z": 1,
            },

            mean_cell_diameter=
                cfg.cell_diam,

            gene_column=
                "gene",
        ),
        "ComSegDataset"
    )


    t0 = time.time()

    data = ComSegDataset(
        **ds_kw
    )

    P(
        f"    ready "
        f"({time.time()-t0:.1f}s)"
    )


    # ========================================================
    # edge weights
    # ========================================================

    edge_cache = os.path.join(
        dirs["cache"],
        (
            f"edge_{tag}"
            f"_top{cfg.top_n}"
            f"_s{cfg.sampling_size}"
            f"_n{cfg.n_neighbors}.pkl"
        )
    )


    if (
        os.path.isfile(edge_cache)
        and
        not cfg.force
    ):

        P(
            f"\n[3] reuse edge-weight"
            f"\n    {edge_cache}"
        )


        with open(
            edge_cache,
            "rb"
        ) as f:

            data = pickle.load(f)

    else:

        P(
            "\n[3] compute_edge_weight"
        )


        ew_kw = safe_kwargs(
            data.compute_edge_weight,
            dict(
                sampling=True,

                sampling_size=
                    cfg.sampling_size,

                n_neighbors=
                    cfg.n_neighbors,
            ),
            "compute_edge_weight"
        )


        t0 = time.time()


        data.compute_edge_weight(
            **ew_kw
        )


        elapsed = (
            time.time()
            - t0
        )


        P(
            f"    complete "
            f"{elapsed:.0f}s "
            f"({elapsed/60:.1f} min)"
        )


        try:

            with open(
                edge_cache,
                "wb"
            ) as f:

                pickle.dump(
                    data,
                    f,
                    protocol=4
                )

            P(
                f"    cache -> "
                f"{edge_cache}"
            )

        except Exception as e:

            P(
                f"    cache warning: "
                f"{e}"
            )


    # ========================================================
    # ComSegDict
    # ========================================================

    P(
        "\n[4] ComSegDict"
    )


    dd_kw = safe_kwargs(
        ComSegDict.__init__,
        dict(

            dataset=
                data,

            mean_cell_diameter=
                cfg.cell_diam,

            community_detection=
                "with_prior",
        ),
        "ComSegDict"
    )


    comseg_dict = (
        ComSegDict(
            **dd_kw
        )
    )


    P(
        "\n[5] clear dict_centroid"
    )


    kill_dict_centroid(
        data,
        comseg_dict
    )


    # ========================================================
    # classify_centroid file
    # ========================================================

    classify_file = os.path.join(
        dirs["centroid"],
        f"{img_name}.csv"
    )


    nuc.to_csv(
        classify_file,
        index=False
    )


    P(
        "\n[6] classify_centroid file"
    )

    P(
        f"    {classify_file}"
    )


    # ========================================================
    # run_all
    # ========================================================

    P(
        "\n[7] run_all"
    )

    P(
        f"    cell diameter = "
        f"{cfg.cell_diam}"
    )

    P(
        f"    max radius = "
        f"{cfg.max_cell_radius}"
    )

    P(
        f"    mmc = "
        f"{cfg.min_merge_correlation}"
    )


    ra_kw = safe_kwargs(
        comseg_dict.run_all,
        dict(

            max_cell_radius=
                cfg.max_cell_radius,

            k_nearest_neighbors=
                cfg.k_nn,

            n_neighbors=
                cfg.n_neighbors,

            path_dataset_folder_centroid=
                dirs["centroid"],

            min_merge_correlation=
                cfg.min_merge_correlation,

            resolution=
                cfg.resolution,
        ),
        "run_all"
    )


    t0 = time.time()


    comseg_dict.run_all(
        **ra_kw
    )


    elapsed = (
        time.time()
        - t0
    )


    P(
        f"    run_all complete "
        f"{elapsed:.0f}s "
        f"({elapsed/60:.1f} min)"
    )


    return (
        comseg_dict,
        img_name
    )


# ============================================================
# export
# ============================================================

def export_result(
    comseg_dict,
    tag,
    cfg,
    dirs
):

    P(
        "\n[8] export AnnData"
    )


    ad_kw = safe_kwargs(
        comseg_dict
        .anndata_from_comseg_result,
        dict(

            min_rna_per_cell=
                cfg.min_rna_per_cell,

            return_polygon=
                False,
        ),
        "anndata_from_comseg_result"
    )


    ret = (
        comseg_dict
        .anndata_from_comseg_result(
            **ad_kw
        )
    )


    if isinstance(ret, tuple):
        adata = ret[0]
    else:
        adata = ret


    if (
        adata is None
        or
        getattr(
            adata,
            "n_obs",
            0
        ) == 0
    ):

        raise RuntimeError(
            "ComSeg export 得到空 AnnData"
        )


    mmc = (
        str(
            cfg.min_merge_correlation
        )
        .replace(
            ".",
            "p"
        )
    )


    out = os.path.join(
        dirs["out"],
        (
            f"{tag}"
            f"_top{cfg.top_n}"
            f"_mmc{mmc}.h5ad"
        )
    )


    adata.write_h5ad(
        out
    )


    counts = np.asarray(
        adata.X.sum(axis=1)
    ).ravel()


    P()
    P(SEP)

    P(
        f"DONE — ComSeg "
        f"{cfg.dataset}"
    )

    P(
        f"cells = "
        f"{adata.n_obs:,}"
    )

    P(
        f"genes = "
        f"{adata.n_vars:,}"
    )


    if len(counts):

        P(
            f"RNA/cell median = "
            f"{np.median(counts):.1f}"
        )

        P(
            f"RNA/cell mean   = "
            f"{np.mean(counts):.1f}"
        )


    P(
        f"h5ad = {out}"
    )

    P(SEP)


    return out


# ============================================================
# main
# ============================================================

def main():

    ap = argparse.ArgumentParser()


    ap.add_argument(
        "--dataset",
        required=True,
        choices=[
            "liver1",
            "liver2"
        ]
    )


    # ========================================================
    # 保持当前实验参数，不主动修改
    # ========================================================

    ap.add_argument(
        "--top-n",
        type=int,
        default=300
    )


    ap.add_argument(
        "--cell-diam",
        type=float,
        default=10.0
    )


    ap.add_argument(
        "--sampling-size",
        type=int,
        default=2000
    )


    ap.add_argument(
        "--k-nn",
        type=int,
        default=2
    )


    ap.add_argument(
        "--n-neighbors",
        type=int,
        default=5
    )


    ap.add_argument(
        "--max-cell-radius",
        type=float,
        default=15.0
    )


    ap.add_argument(
        "--min-rna-per-cell",
        type=int,
        default=5
    )


    ap.add_argument(
        "--min-merge-correlation",
        type=float,
        default=0.99
    )


    ap.add_argument(
        "--resolution",
        type=float,
        default=1.0
    )


    ap.add_argument(
        "--force",
        action="store_true"
    )


    cfg = ap.parse_args()


    dirs = get_dirs(
        cfg.dataset
    )


    P(SEP)

    P(
        f"ComSeg × MERFISH "
        f"{cfg.dataset}"
    )

    P(
        "mode = with official prior"
    )

    P(
        f"topN={cfg.top_n}  "
        f"mmc="
        f"{cfg.min_merge_correlation}"
    )

    P(SEP)


    P(
        "\n[1] prepare data"
    )


    tag, txp, cenp, nuc = (
        prepare_data(
            cfg,
            cfg.dataset,
            dirs
        )
    )


    comseg_dict, img_name = (
        run_comseg(
            tag,
            txp,
            cenp,
            nuc,
            cfg,
            dirs
        )
    )


    export_result(
        comseg_dict,
        tag,
        cfg,
        dirs
    )


if __name__ == "__main__":
    main()