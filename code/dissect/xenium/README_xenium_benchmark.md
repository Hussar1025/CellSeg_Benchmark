# Xenium benchmark with DISSECT

The entry script is `run_xenium_benchmark_dissect.py`. Following the Xenium input
convention of the official DISSECT
[README](https://github.com/zenglab-pku/DISSECT/blob/main/README.md) (`x_location`/`y_location`, 0.2125 µm/px;
DAPI/morphology image already registered to the transcript coordinates), it calls
`ModelGenerator`, `AnchorGenerator` and transcript gradient refinement.

## Environment and running

```bash
conda activate dissect
python yourpath/DISSECT/code/run_xenium_benchmark_dissect.py --dataset all --dry-run
python yourpath/DISSECT/code/run_xenium_benchmark_dissect.py --dataset all
```

You can also run a single dataset, for example:

```bash
python yourpath/DISSECT/code/run_xenium_benchmark_dissect.py \
  --dataset xenium_liver --n-jobs 1
```

The script never modifies the raw data. Results for each dataset are written by default to
`yourpath/DISSECT/xenium_benchmark/<dataset>/`:

- `mask.npy`: `int32` instance label image of ROI size, 0 = background, 1..N = cells;
- `boxes.csv`: NMS proposals, scores and refined `cell_id`;
- `raw_boxes.csv`: output of the DISSECT detection stage (written by the official detector);
- `roi_info.json`, `run_info.json`: ROI coordinates, inputs, parameters and run statistics.

## Fixed ROI and fairness

| dataset | morphology/DAPI input | ROI (global pixel coordinates, left-closed right-open) |
| --- | --- | --- |
| `xenium_breast` | `morphology_focus/ch0000_dapi.ome.tif` | image center 5000×5000 |
| `xenium_roi_crop` | `morphology_focus/morphology_focus_0000.ome.tif` | already a ROI, processed whole; transcript origin `(18741,11870)` |
| `xenium_liver` | `outs/morphology_focus.ome.tif` | image center 10000×10000 |
| `xenium_mouse_brain` | `outs/morphology_focus/morphology_focus_0000.ome.tif` | image center 10000×10000 |
| `xenium_colon` | `morphology_focus/morphology_focus_0000.ome.tif` | image center 10000×10000 |

All five datasets share the same command-line defaults, weights, random seed and model parameters; the script reads no
ROI annotation for tuning. `--count-transcripts` is only used for dry-run input verification and does not affect the model.

The image uses the official `STReader` whole-image max normalization `image / max(image) * 256`; to avoid
loading tens of GB of whole-slide data, the script reads the ROI by TIFF tile decoding and filters ROI transcripts
by parquet row-group spatial selection, keeping the values consistent with the official coordinate conversion.
