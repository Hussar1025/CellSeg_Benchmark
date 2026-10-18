# Xenium 8 中心 10000×10000 px 的 DISSECT 分割

入口脚本是 [`run_xenium8_dissect.py`](./run_xenium8_dissect.py)。实现按
DISSECT 官方 README 的 Xenium 用法（`platform="xenium"`，转录本列
`x_location`/`y_location`/`feature_name`，坐标换算为 0.2125 µm/px），调用
官方 `ModelGenerator`、`AnchorGenerator` 和转录本梯度细化流程：

<https://github.com/zenglab-pku/DISSECT/blob/main/README.md>

脚本只读取图像中心的 ROI，不会修改原始 Xenium 文件。输入固定为：

- DAPI：`/data/qiuyijia/dataset/xenium_8/morphology_focus/ch0000_dapi.ome.tif`
- 转录本：`/data/qiuyijia/dataset/xenium_8/transcripts.parquet`
- ROI：全局像素 `x=[22001,32001)`、`y=[8664,18664)`，大小 10000×10000
- 模型：`/data/qiuyijia/DISSECT/weights/model_weights.pth`

## 运行前检查

```bash
conda activate dissect
python /data/qiuyijia/DISSECT/code/run_xenium8_dissect.py \
  --dry-run --count-transcripts
```

检查结果应报告图像尺寸 `27328×54002`，ROI 内转录本数约为 `5,266,149`。

## nohup 后台运行

下面命令使用官方参数默认值（`minarea=350`、`maxarea=1600`、
`stride=128`、`ratio_stride=3`、`alpha=0.5`、`repetition=2`），启用转录本
梯度细化，并将标准输出和错误输出写入日志：

```bash
conda activate dissect
mkdir -p /data/qiuyijia/DISSECT/logs
nohup python -u /data/qiuyijia/DISSECT/code/run_xenium8_dissect.py \
  --n-jobs 4 \
  > /data/qiuyijia/DISSECT/logs/xenium8_center10000_dissect.log 2>&1 &
echo $! > /data/qiuyijia/DISSECT/logs/xenium8_center10000_dissect.pid
```

如果内存紧张，把 `--n-jobs 4` 降为 `--n-jobs 1`；若希望只使用图像而不做转录本
细化，可加入 `--no-gene`。运行状态可用：

```bash
tail -f /data/qiuyijia/DISSECT/logs/xenium8_center10000_dissect.log
ps -p "$(cat /data/qiuyijia/DISSECT/logs/xenium8_center10000_dissect.pid)" -o pid,etime,%cpu,%mem,cmd
```

## 输出

默认目录为 `/data/qiuyijia/DISSECT/xenium_8/`：

- `mask.npy`：10000×10000 的 `int32` 实例标签图，0 为背景；
- `boxes.csv`：NMS proposal、得分、细化后的 `cell_id`；
- `raw_boxes.csv`：检测阶段原始 proposal；
- `roi_info.json`：ROI 与输入坐标信息；
- `run_info.json`：参数、proposal 数量、细胞数量和耗时。
