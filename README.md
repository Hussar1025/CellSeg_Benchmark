# CellSeg_Benchmark

## Overview

Cell segmentation is a fundamental step in many biological studies, enabling the precise delineation of individual cells from raw data. Traditional approaches primarily rely on image-based inputs (e.g., fluorescence or brightfield microscopy), but emerging technologies—such as spatial transcriptomics and proteomics—introduce omics-derived spatial features as additional or alternative inputs. To bridge these two paradigms, this benchmark systematically evaluates 16 cell segmentation methods that accept either image data, omics data (e.g., gene expression matrices with spatial coordinates), or both.

Accurate cell segmentation unlocks a wide range of downstream biological applications: from quantifying cell morphology and phenotyping heterogeneous populations, to dissecting tissue microenvironments, tracking cell-cell interactions, and enabling spatially resolved single-cell analyses in diseased tissues. By providing a fair and reproducible comparison across diverse datasets (including synthetic and real-world benchmarks), this repository aims to guide method selection for researchers and highlight strengths and weaknesses of existing algorithms—ultimately accelerating discoveries in developmental biology, oncology, and neuroscience.

## Method

In this work, we present a benchmark study encompassing 16 cell segmentation methods across datasets with varying characteristics and technological platforms. The following methods were included:
- Cellotype: 《CelloType: a unified model for segmentation and classification of tissue images》
- CellSAM: 《CellSAM: a foundation model for cell segmentation	Nature Methods》
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
- DISSECT: 《Diffusion-based Cell Segmentation Integrating Spatial Transcriptomics》

## Dataset
In this work, We benchmark the 16 cell segmentation methods on 27 datasets, include Xenium (10x Genomics), Cosmx, Merfish and so on, which provide multimodal in situ spatial transcriptomics data—including high-resolution fluorescence images and spatially resolved gene expression profiles. The following datasets are as follows:
- [Pancreatic Cancer with Xenium Human Multi-Tissue and Cancer Panel](https://www.10xgenomics.com/datasets/pancreatic-cancer-with-xenium-human-multi-tissue-and-cancer-panel-1-standard)
