# Xenium pancreas whole-section DISSECT

入口脚本：`run_xenium_pancreas_dissect.py`。默认值已经指向本工作区中的
`dataset/xenium_pancreas_cancer`、DISSECT 权重和输出目录
`/data/qiuyijia/DISSECT/xenium_pancrease`。

## 环境

```bash
conda activate dissect
python -m pip install "Pillow==9.5.0" pyarrow imagecodecs
```

官方 DISSECT 依赖 detectron2；如果 detectron2 是在另一台 GLIBC/CUDA 环境
编译的，需要在当前环境重新安装/编译它。
detectron2 0.6 需要 Pillow 9.x（其使用了 `PIL.Image.LINEAR`）；请勿升级到
Pillow 10 或更高版本。

## 运行

先做不加载模型的输入检查：

```bash
python /data/qiuyijia/DISSECT/code/run_xenium_pancreas_dissect.py --dry-run
```

全片运行：

```bash
python /data/qiuyijia/DISSECT/code/run_xenium_pancreas_dissect.py
```

主要输出：

- `mask.npy`：`int32` 实例标签图，0 为背景，1..N 为细胞；形状与
  `morphology_focus.ome.tif` 相同。
- `boxes.csv`：NMS 后 proposal 的坐标、分数及对应 `cell_id`（未成功细化时为 0）。
- `raw_boxes.csv`：官方滑窗检测阶段产生的原始框。
- `run_info.json`：输入坐标范围、参数、proposal/细胞数量和耗时。

脚本默认使用 DISSECT 官方 proposal 面积过滤（350--1600 px²）和重复次数
（2）。此前脚本中的 600--1200 px² 会在检测阶段丢弃大量较大 proposal；如需做参数
敏感性实验，请显式记录命令行参数，例如 `--minarea 350 --maxarea 3000`，并
将其作为单独实验，不要用测试集标注反向选择参数。`--repetition` 可用于控制
模型重复推理次数，固定随机种子后结果可复现。

精修阶段默认每批提交 512 个 proposal（`--refine-batch-size`）。分批只改变
任务调度和日志输出，不改变模型、输入或精修算法；在内存紧张时可降低该值。
脚本还会先按 transcript 的 x 坐标建立索引，再为每个 proposal 传入其扩展框内
的 transcript；这与官方精修函数内部的矩形筛选等价，但避免每个 proposal
重复扫描全部 2,845 万条 transcript。

脚本会显式使用整张 `morphology_focus.ome.tif`；这是必要的，因为官方
包装器的 `slice_region` 在非正方形图像上存在长宽顺序问题。

如果显存或 CPU 内存不足，可先降低并行度：

```bash
python /data/qiuyijia/DISSECT/code/run_xenium_pancreas_dissect.py --n-jobs 1
```

`--stride`、`--ratio-stride`、`--minarea`、`--maxarea` 等参数与官方
`dissect.segmentation` 接口一致；脚本默认 `--gene` 开启转录本梯度细化。
