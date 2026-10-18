# Xenium–Bering benchmark 配置

运行环境为 Bering conda 环境：

```bash
for c in /data/qiuyijia/bering/configs/*.json; do
  /data/qiuyijia/anaconda3/envs/bering/bin/python \
    /data/qiuyijia/bering/bering_xenium_benchmark.py --config "$c" --check
done
```

确认所有输入后再运行（每个配置一次）：

```bash
/data/qiuyijia/anaconda3/envs/bering/bin/python \
  /data/qiuyijia/bering/bering_xenium_benchmark.py \
  --config /data/qiuyijia/bering/configs/xenium_breast.json
```

## xenium7 与 xenium_8（中心 10000×10000 px）

两套数据集的一键后台运行脚本为：

```bash
/data/qiuyijia/bering/run_bering_xenium_center10000.sh
```

脚本会先执行 `--check`，随后通过 `nohup` 按 xenium7、xenium_8 的顺序运行，
避免同时占用 benchmark runner 固定使用的 GPU 6。日志位于
`/data/qiuyijia/bering/logs/`，输出目录分别为
`xenium7_center10000_output/` 和 `xenium_8_center10000_output/`。
可用 `BERING_DRY_RUN=1` 只做预检而不提交任务。

| 配置 | ROI（native px） | 备注 |
|---|---|---|
| xenium_breast | 中心 5000×5000 | x0=24497, y0=11210 |
| xenium_roi_crop | 整个已裁剪数据集 | 3982.5–4317.5 × 2522.5–2857.5 µm |
| xenium_liver | 中心 10000×10000 | x0=19470, y0=11200 |
| xenium_mouse_brain | 中心 10000×10000 | x0=12077, y0=6956 |
| xenium_colon | 中心 10000×10000 | x0=9241, y0=12052 |
| xenium_pancreas_cancer | 全部有效转录本范围 | 原脚本的第六个数据集；用户未给出其 ROI |

所有配置均使用：`qv >= 20`、`is_gene == True`（字段存在时）、相同的
control 前缀过滤、2-D 坐标、相同 Bering 超参数、随机种子 0。Bering 的
`raw_cells` 来自 Xenium 官方 `transcripts.cell_id`；因此这是
**reference-assisted** 结果，不能与无先验方法混称为公平的 de-novo 比较。
`cell_boundaries.parquet` 从未作为预测输入。训练标签由训练子块内表达矩阵的
固定 MiniBatchKMeans 生成，避免某个数据集使用额外人工注释。

如果论文要比较“无参考”方法，应把 Bering 单独列为 reference-assisted，或为
所有方法提供同一套先验；不要把 `cell_groups.csv` 标签监督结果与无标签方法直接
合并排名。ROI、QC、后处理阈值和 tile stitching 半径均须保持不变。

## Baysor 评估

针对官方 Xenium 分割作为 GT、`/data/qiuyijia/baysor/xenium_*_output` 作为预测
结果的评估脚本位于：

`/data/qiuyijia/cellseg_benchmark/code/baysor/evaluate_code/baysor_value_xenium.py`

脚本按 `ucs_value_xenium_4&5&6.py` 的定义计算细胞检测（precision/recall/F1）、
质心偏移、匹配细胞的转录本数量和基因向量一致性，以及官方
`transcripts.*` 中 `cell_id` 的转录本归属准确率。Baysor 的 `segmentation.csv`
采用全局微米坐标，评估时以分块方式读取，避免一次性载入大文件。
其中 `xenium_liver` 虽然 Baysor 在中心 15000×15000 native px 区域上运行，评估
时按官方 `central_roi_10000/gt_roi_info.json` 只比较中心 10000×10000 native px
子区域（x[21974,31974)、y[13704,23704)）。

```bash
python /data/qiuyijia/cellseg_benchmark/code/baysor/evaluate_code/baysor_value_xenium.py \
  --dataset breast --out-dir /data/qiuyijia/eval_results/baysor_xenium

# 一次评估五个数据集（使用脚本内置的 Baysor ROI 和路径）
python /data/qiuyijia/cellseg_benchmark/code/baysor/evaluate_code/baysor_value_xenium.py --all --out-dir /data/qiuyijia/eval_results/baysor_xenium
```

每个数据集会生成 `*_sparse_hungarian_qc_metrics_full.csv`、
`*_sparse_hungarian_qc_metrics_selected.csv` 和匹配细胞明细；`--qv-min 0` 可关闭
质量值过滤，`--match-radius` 可覆盖默认的 GT 最近邻 p95×1.25 匹配半径。

## Baysor–MERFISH 评估

MERFISH Baysor 评估脚本为：
`/data/qiuyijia/cellseg_benchmark/code/baysor/evaluate_code/baysor_value_merfish.py`。
它使用 Vizgen 官方 `cell_metadata.csv` 的细胞中心作为 GT，采用与 Xenium
评估完全相同的稀疏 Hungarian 细胞匹配和指标列：

```bash
python /data/qiuyijia/cellseg_benchmark/code/baysor/evaluate_code/baysor_value_merfish.py \
  --all --out-dir /data/qiuyijia/eval_results/baysor_merfish
```

MERFISH 原始 `detected_transcripts.csv` 不含官方逐转录本 `cell_id`。对于
liver1、liver2 和 mouse_brain，脚本默认读取已有的
`transcripts_roi_with_prior.csv.gz`（由官方 cell-boundary 多边形生成）作为
逐转录本代理标签，并在结果中保留转录本数量、基因向量和归属准确率。对于 lung、
prostate1、prostate2，先运行下方的准备脚本生成同格式文件，即可计算相同的转录本级指标。

如需为 lung、prostate1、prostate2 生成同格式的 GT 转录本文件，可运行：

```bash
python /data/qiuyijia/cellseg_benchmark/code/baysor/evaluate_code/prepare_merfish_gt_transcripts.py \
  --all
```

脚本读取各数据集官方 `cell_boundaries/*.hdf5` 多边形，在对应的 10000×10000
ROI 内按 1 µm 网格为转录本写入 `cell_id`，默认输出到
`/data/qiuyijia/baysor/merfish_gt_tx/<dataset>/transcripts_roi_with_prior.csv.gz`。
生成后直接重新运行上面的 Baysor 评估命令即可自动使用这些标签；也可以用
`--gt-transcripts` 显式指定单个文件。`cell_id` 是边界栅格产生的 ROI-local
标签，评估脚本会通过细胞中心将其桥接到官方 `cell_metadata.csv` ID。
