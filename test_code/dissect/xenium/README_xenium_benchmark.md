# Xenium benchmark with DISSECT

入口脚本是 `run_xenium_benchmark_dissect.py`。脚本按照 DISSECT 官方
[README](https://github.com/zenglab-pku/DISSECT/blob/main/README.md) 的 Xenium
输入约定（`x_location`/`y_location`，0.2125 µm/px；DAPI/morphology 图像与转录本
坐标已配准）调用 `ModelGenerator`、`AnchorGenerator` 和转录本梯度细化。

## 环境与运行

```bash
conda activate dissect
python /data/qiuyijia/DISSECT/code/run_xenium_benchmark_dissect.py --dataset all --dry-run
python /data/qiuyijia/DISSECT/code/run_xenium_benchmark_dissect.py --dataset all
```

也可以只运行一个数据集，例如：

```bash
python /data/qiuyijia/DISSECT/code/run_xenium_benchmark_dissect.py \
  --dataset xenium_liver --n-jobs 1
```

脚本不会修改原始数据。每个数据集的结果默认写入
`/data/qiuyijia/DISSECT/xenium_benchmark/<dataset>/`：

- `mask.npy`：ROI 大小的 `int32` 实例标签图，0 是背景，1..N 是细胞；
- `boxes.csv`：NMS proposal、分数和细化后的 `cell_id`；
- `raw_boxes.csv`：DISSECT 检测阶段输出（由官方 detector 写入）；
- `roi_info.json`、`run_info.json`：ROI 坐标、输入、参数和运行统计。

## 固定 ROI 与公平性

| 数据集 | morphology/DAPI 输入 | ROI（全局像素坐标，左闭右开） |
| --- | --- | --- |
| `xenium_breast` | `morphology_focus/ch0000_dapi.ome.tif` | 图像中心 5000×5000 |
| `xenium_roi_crop` | `morphology_focus/morphology_focus_0000.ome.tif` | 已是 ROI，整张处理；转录本原点 `(18741,11870)` |
| `xenium_liver` | `outs/morphology_focus.ome.tif` | 图像中心 10000×10000 |
| `xenium_mouse_brain` | `outs/morphology_focus/morphology_focus_0000.ome.tif` | 图像中心 10000×10000 |
| `xenium_colon` | `morphology_focus/morphology_focus_0000.ome.tif` | 图像中心 10000×10000 |

五个数据集共用同一套命令行默认值、权重、随机种子和模型参数；脚本不读取任何
ROI 标注来调参。`--count-transcripts` 仅用于 dry-run 的输入核查，不参与模型。

图像采用官方 `STReader` 的全图最大值归一化 `image / max(image) * 256`；为避免
加载几十 GB 的全片，脚本用 TIFF tile 解码读取 ROI，并用 parquet row-group 空间
筛选读取 ROI 内转录本，数值与官方坐标转换保持一致。

