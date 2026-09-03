# CellSeg_Benchmark

## Overview

Cell segmentation is a fundamental step in many biological studies, enabling the precise delineation of individual cells from raw data. Traditional approaches primarily rely on image-based inputs (e.g., fluorescence or brightfield microscopy), but emerging technologies—such as spatial transcriptomics and proteomics—introduce omics-derived spatial features as additional or alternative inputs. To bridge these two paradigms, this benchmark systematically evaluates 16 cell segmentation methods that accept either image data, omics data (e.g., gene expression matrices with spatial coordinates), or both.

Accurate cell segmentation unlocks a wide range of downstream biological applications: from quantifying cell morphology and phenotyping heterogeneous populations, to dissecting tissue microenvironments, tracking cell-cell interactions, and enabling spatially resolved single-cell analyses in diseased tissues. By providing a fair and reproducible comparison across diverse datasets (including synthetic and real-world benchmarks), this repository aims to guide method selection for researchers and highlight strengths and weaknesses of existing algorithms—ultimately accelerating discoveries in developmental biology, oncology, and neuroscience.

To address this gap, we systematically evaluated 16 cell segmentation methods (covering image‑driven, omics‑driven, and hybrid approaches) across 13 cross‑platform datasets and 20 cross‑tissue (Xenium & MERFISH) datasets, performing 300+ distinct segmentation tasks to quantify accuracy, efficiency, and robustness. Our results show that methods integrating omics information achieve superior boundary refinement but suffer from low processing throughput; image‑based methods deliver high morphological accuracy when staining is good, yet perform poorly in transcript assignment in weakly stained or crowded regions. Three additional challenging scenario tests (downsampling, using highly variable genes instead of the full transcriptome, and ultra‑large‑scale datasets) further revealed bottlenecks in accuracy and memory consumption. Based on these findings, we analyzed key influencing factors and provide method selection guidelines, an executable pipeline, and an interactive online platform to help users optimize segmentation strategies according to their specific data characteristics.

## Method

In this work, we present a benchmark study encompassing 16 cell segmentation methods across datasets with varying characteristics and technological platforms. The following methods were included:
- Cellotype: 《CelloType: a unified model for segmentation and classification of tissue images》
- CellSAM: 《CellSAM: a foundation model for cell segmentation》
- Cellpose-3: 《Cellpose3: one-click image restoration for improved cellular segmentation》
- Cellpose-SAM: 《Cellpose-SAM: superhuman generalization for cellular segmentation》
- StarDist 2D: 《Cell detection with star-convex poly-gons》
- Mesmer: 《Whole-cell segmentation of tissue images with human-level performance using large-scale data annotation and deep learning》
- UNSEG: 《UNSEG: unsupervised segmentation of cells and their nuclei in complex tissue samples》
- UCS: 《UCS: A Unified Approach to Cell Segmentation for Subcellular Spatial Transcriptomics》
- BOMS: 《From spots to cells: Cell segmentation in spatial transcriptomics with BOMS》
- ComSeg: 《A point cloud segmentation framework for image-based spatial transcriptomics	Communications Biology》
- GeneSegNet: 《GeneSegNet: a deep learning framework for cell segmentation by integrating gene expression and imaging》
- Proseg: 《Cell simulation as cell segmentation》
- Baysor: 《Cell segmentation in imaging-based spatial transcriptomics》
- Bering: 《Bering: joint cell segmentation and annotation for spatial transcriptomics with transferred graph embeddings》
- Cellist: 《Accurate, scalable and cross-platform cell identification for high-resolution spatial transcriptomics》
- DISSECT: 《Integratingcytological images and spatialtranscriptomics for cell segmentation with DISSECT》

## Dataset
In this work, We benchmark the 16 cell segmentation methods on 27 datasets, include Xenium (10x Genomics), Cosmx, Merfish and so on, which provide multimodal in situ spatial transcriptomics data—including high-resolution fluorescence images and spatially resolved gene expression profiles. The following datasets are as follows:
- **Xenium**
  - [Pancreatic Cancer with Xenium Human Multi-Tissue and Cancer Panel](https://www.10xgenomics.com/datasets/pancreatic-cancer-with-xenium-human-multi-tissue-and-cancer-panel-1-standard)
  - [Xenium v1 Human Breast FFPE with Biomarkers & Housekeeping Genes Custom Panel](https://www.10xgenomics.com/datasets/xenium-ffpe-human-breast-biomarkers)
  - [FFPE Human Lymph Node with 5K Pan Tissue and Pathways Panel](https://www.10xgenomics.com/datasets/preview-data-xenium-prime-gene-expression)
  - [Fresh Frozen Mouse Brain Hemisphere with 5K Mouse Pan Tissue and Pathways Panel](https://www.10xgenomics.com/datasets/xenium-prime-fresh-frozen-mouse-brain)
  - [Human Liver Data with Xenium Human Multi-Tissue and Cancer Panel](https://www.10xgenomics.com/datasets/human-liver-data-xenium-human-multi-tissue-and-cancer-panel-1-standard)
  - [Fresh Frozen Mouse Colon with Xenium Multimodal Cell Segmentation](https://www.10xgenomics.com/datasets/fresh-frozen-mouse-colon-with-xenium-multimodal-cell-segmentation-1-standard)
  - [Preview Data: FFPE Human Prostate Adenocarcinoma with 5K Human Pan Tissue and Pathways Panel](https://www.10xgenomics.com/datasets/xenium-prime-ffpe-human-prostate)
  - [FFPE Human Cervical Cancer with 5K Human Pan Tissue and Pathways Panel plus 100 Custom Genes](https://www.10xgenomics.com/datasets/xenium-prime-ffpe-human-cervical-cancer)

- **CosMx**
  - [Cosmx-FFPE-Pancreas-dataset-flat-files](https://brukerspatialbiology.com/resources/cosmx-ffpe-pancreas-dataset-flat-files/)
  - [Cosmx-Human-Lymph-Node-FFPE-dataset-flat-files](https://brukerspatialbiology.com/resources/cosmx-human-lymph-node-ffpe-dataset-flat-files/)
  - [CosMx Human Whole Transcriptome Brain Dataset](https://objects.liquidweb.services/smi-public/wtx_manuscript/brain_hippocampus/index_hippocampus.html)

- **Stereo-seq**
  - [MOSTA: Mouse Organogenesis Spatiotemporal Transcriptomic Atlas/E16.5_E2S6](https://db.cngb.org/stomics/datasets/STDS0000058/data)
  - [MOSTA: Mouse Organogenesis Spatiotemporal Transcriptomic Atlas/E14.5_E1S3](https://db.cngb.org/stomics/datasets/STDS0000058/data)
  - [MOSTA: Mouse Organogenesis Spatiotemporal Transcriptomic Atlas/E16.5_E2S7](https://db.cngb.org/stomics/datasets/STDS0000058/data)
  - [Spatial Gene Expression of Mouse Embryo Fresh Frozen Tissue](https://en.stomics.tech/col1370/index.html)

- **MERFISH**
  - [Vizgen MERFISH Mouse Brain Receptor Map](https://info.vizgen.com/mouse-brain-data)
  - [Merscope FFPE Human Immuno-Oncology Data Liver cancer 1](https://info.vizgen.com/ffpe-showcase)
  - [Merscope FFPE Human Immuno-Oncology Data Liver cancer 2](https://info.vizgen.com/ffpe-showcase)
  - [Merscope FFPE Human Immuno-Oncology Data Lung cancer 1](https://info.vizgen.com/ffpe-showcase)
  - [Merscope FFPE Human Immuno-Oncology Data Prostate cancer 1​](https://info.vizgen.com/ffpe-showcase)
  - [Merscope FFPE Human Immuno-Oncology Data Prostate cancer 2](https://info.vizgen.com/ffpe-showcase)
  - [Merscope FFPE Human Immuno-Oncology Data Breast cancer​](https://info.vizgen.com/ffpe-showcase)

- **StarMap**
  - [STARmap Mouse VISp 1020 genes](https://github.com/wanglab-broad/ClusterMap/tree/main/datasets/STARmap_V1_1020)
