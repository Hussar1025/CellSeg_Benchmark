# Xenium pancreas whole-section DISSECT

Entry script: `run_xenium_pancreas_dissect.py`. Defaults already point to
`dataset/xenium_pancreas_cancer`, the DISSECT weights and the output directory
`yourpath/DISSECT/xenium_pancrease` in this workspace.

## Environment

```bash
conda activate dissect
python -m pip install "Pillow==9.5.0" pyarrow imagecodecs
```

Official DISSECT depends on detectron2; if detectron2 was compiled in a different
GLIBC/CUDA environment, it must be reinstalled/recompiled in the current environment.
detectron2 0.6 requires Pillow 9.x (it uses `PIL.Image.LINEAR`); do not upgrade to
Pillow 10 or newer.

## Run

First an input check without loading the model:

```bash
python yourpath/DISSECT/code/run_xenium_pancreas_dissect.py --dry-run
```

Whole-section run:

```bash
python yourpath/DISSECT/code/run_xenium_pancreas_dissect.py
```

Main outputs:

- `mask.npy`: `int32` instance label image, 0 = background, 1..N = cells; shape same as
  `morphology_focus.ome.tif`.
- `boxes.csv`: coordinates and scores of post-NMS proposals and the matching `cell_id` (0 when refinement did not succeed).
- `raw_boxes.csv`: raw boxes produced by the official sliding-window detection stage.
- `run_info.json`: input coordinate range, parameters, proposal/cell counts and elapsed time.

The script uses the official DISSECT proposal area filter (350--1600 px²) and repetition count
(2) by default. The earlier 600--1200 px² in the script discarded many large proposals at the detection stage; for
parameter-sensitivity experiments, record the command-line arguments explicitly, e.g. `--minarea 350 --maxarea 3000`, and
treat them as a separate experiment; do not reverse-select parameters on test-set annotations. `--repetition` controls the
number of model inference repeats; with a fixed random seed the results are reproducible.

The refinement stage submits 512 proposals per batch by default (`--refine-batch-size`). Batching only changes
task scheduling and log output, not the model, inputs or refinement algorithm; lower this value when memory is tight.
The script also builds an index on the transcript x coordinate first, then passes each proposal the transcripts in its expanded box;
this is equivalent to the rectangular filtering inside the official refinement function, but avoids rescanning all
28.45 million transcripts for every proposal.

The script explicitly uses the whole `morphology_focus.ome.tif`; this is necessary because the official
wrapper's `slice_region` has a width/height ordering issue on non-square images.

If GPU or CPU memory is insufficient, lower the parallelism first:

```bash
python yourpath/DISSECT/code/run_xenium_pancreas_dissect.py --n-jobs 1
```

Parameters such as `--stride`, `--ratio-stride`, `--minarea` and `--maxarea` match the official
`dissect.segmentation` interface; the script enables transcript gradient refinement by default (`--gene`).
