# 分割方法代码索引

本目录按 **方法 / 数据集平台** 整理脚本。平台目录使用 `cosmx`、`merfish`、`xenium`、`stereo` 和 `starmap`；跨平台实现放在该方法的 `shared/` 下。`SOURCE_MANIFEST.tsv` 逐项记录原始路径、目标路径和 SHA-256，便于核对代码版本。

| 方法目录 | 来源 | 平台 |
| --- | --- | --- |
| `bering/` | `/data/qiuyijia/bering` | CosMx、MERFISH、Xenium、Stereo-seq、STARmap |
| `cellotype/` | `/data/qiuyijia/cellotype/CelloType/code` | CosMx、MERFISH、Xenium |
| `cellpose-SAM/` | `/data/qiuyijia/cellpose` | CosMx、MERFISH、Xenium |
| `cellsam/` | `/data/qiuyijia/CellSam` | CosMx、MERFISH、Xenium、Stereo-seq |
| `Mesmer/` | `/data/qiuyijia/Mesmer` | CosMx、MERFISH、Xenium、Stereo-seq |
| `stardist/` | `/data/qiuyijia/stardist` | CosMx、MERFISH、Xenium |
| `boms/` | `/data/qiuyijia/boms` | CosMx、MERFISH、Xenium、Stereo-seq、公共实现 |
| `bidcell/` | `/data/qiuyijia/bidcell` | CosMx、公共实现 |
| `comseg/` | `/data/qiuyijia/comseg` | CosMx、MERFISH、Xenium、Stereo-seq |
| `dissect/` | `/data/qiuyijia/DISSECT` | MERFISH、Xenium、公共配置 |
| `genesegnet/` | `/data/qiuyijia/genesegnet`、`/data/qiuyijia/GeneSegNet` | CosMx、MERFISH、Xenium、Stereo-seq、公共实现 |
| `proseg/` | `/data/qiuyijia/proseg` | CosMx、MERFISH、Xenium、Stereo-seq |
| `ucs/` | `/data/qiuyijia/ucs/UCS` | CosMx、MERFISH、Xenium、Stereo-seq、STARmap、公共实现 |
| `cellist/` | `/data/qiuyijia/runs`、工作区根目录中的相关脚本 | CosMx、Stereo-seq |
| `benchmark_utils/` | `/data/qiuyijia/runs`、`/data/qiuyijia` 的直接文件 | 跨方法脚本与平台辅助脚本 |

源代码按原样复制，原目录和目标目录已有文件均未删除或覆盖。目标目录中原有的平铺脚本保留在原位置；本次新增的分类目录可用作统一索引。`cellpose-SAM/` 对应本机 `cellpose/` 的 Cellpose-SAM 实验脚本。

对于四个与目标目录现有文件同名但内容不同的 Cellpose-SAM 脚本，原有文件保留，源目录版本存放在对应平台的 `source_copy/` 下。具体文件见清单中的 `conflict_snapshot` 行。

后续整理的 BOMS 和 GeneSegNet 也存在同名但不同内容的脚本，同样把源目录版本放在相应平台的 `source_copy/` 下。评估脚本放在对应平台的 `evaluate_code/` 中。`runs/` 中的入口脚本按实际方法归类，并把其导入的 `seg_common.py` 一起复制到同一 `runs/` 目录。

工作区根目录 `/data/qiuyijia` **仅检查并复制直接位于根目录的文件**；其子目录没有因这一来源被递归扫描。大型数据、参考矩阵、模型权重、运行输出、日志、缓存、备份源码树和含嵌入输出的 notebook 均未复制。BOMS 自有的 C++ 源文件已随 Python 包收录，第三方 Eigen 源码未复制。

Bering 的 `bering_xenium_benchmark.py` 同时用于 Xenium 和 Stereo-seq，因此两个平台目录各有一份。Stereo-seq 的 H&E 脚本分别在 `cellsam/stereo/`、`Mesmer/stereo/`；StarDist 的 `SMALL_HE_seg/test.py` 按其 Xenium 形态图像输入放在 `stardist/xenium/`，`test3.py` 按 CosMx H&E 输入放在 `stardist/cosmx/`。

Mesmer 的三个蛋白分割相关脚本 `protein_pcf_mesmer_level0_safe.py`、`protein_pcf_to_anndata.py`、`visualize_protein_pcf_mesmer.py` 未纳入。

部分脚本仍引用源目录的绝对路径、外部数据集、模型或公共辅助代码。这次整理保留了原始运行设置；在目标目录独立运行前，应检查脚本中的输入、输出和依赖路径。
