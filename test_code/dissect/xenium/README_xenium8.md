# DISSECT segmentation of the Xenium 8 center 10000×10000 px

The entry script is [`run_xenium8_dissect.py`](./run_xenium8_dissect.py). The implementation follows the
Xenium usage in the official DISSECT README (`platform="xenium"`, transcript columns
`x_location`/`y_location`/`feature_name`, coordinates converted at 0.2125 µm/px) and calls the
official `ModelGenerator`, `AnchorGenerator` and transcript gradient refinement pipeline:

<https://github.com/zenglab-pku/DISSECT/blob/main/README.md>

The script only reads the ROI at the image center and does not modify the raw Xenium files. Inputs are fixed:

- DAPI: `yourpath/dataset/xenium_8/morphology_focus/ch0000_dapi.ome.tif`
- transcripts: `yourpath/dataset/xenium_8/transcripts.parquet`
- ROI: global pixels `x=[22001,32001)`, `y=[8664,18664)`, size 10000×10000
- model: `yourpath/DISSECT/weights/model_weights.pth`

## Pre-run check

```bash
conda activate dissect
python yourpath/DISSECT/code/run_xenium8_dissect.py \
  --dry-run --count-transcripts
```

The check should report image size `27328×54002` and about `5,266,149` transcripts in the ROI.

## Run in background with nohup

The command below uses the official parameter defaults (`minarea=350`, `maxarea=1600`,
`stride=128`, `ratio_stride=3`, `alpha=0.5`, `repetition=2`), enables transcript
gradient refinement, and writes stdout and stderr to the log:

```bash
conda activate dissect
mkdir -p yourpath/DISSECT/logs
nohup python -u yourpath/DISSECT/code/run_xenium8_dissect.py \
  --n-jobs 4 \
  > yourpath/DISSECT/logs/xenium8_center10000_dissect.log 2>&1 &
echo $! > yourpath/DISSECT/logs/xenium8_center10000_dissect.pid
```

If memory is tight, lower `--n-jobs 4` to `--n-jobs 1`; to use the image only without transcript
refinement, add `--no-gene`. Check the status with:

```bash
tail -f yourpath/DISSECT/logs/xenium8_center10000_dissect.log
ps -p "$(cat yourpath/DISSECT/logs/xenium8_center10000_dissect.pid)" -o pid,etime,%cpu,%mem,cmd
```

## Outputs

The default directory is `yourpath/DISSECT/xenium_8/`:

- `mask.npy`: 10000×10000 `int32` instance label image, 0 = background;
- `boxes.csv`: NMS proposals, scores and refined `cell_id`;
- `raw_boxes.csv`: raw proposals from the detection stage;
- `roi_info.json`: ROI and input coordinate information;
- `run_info.json`: parameters, proposal count, cell count and elapsed time.
