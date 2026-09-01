# DeferredSeg

**DeferredSeg: A Multi-Expert Deferral Framework for Trustworthy Medical Image Segmentation**, published in **Pattern Recognition (2026)**.

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2604.12411)
[![Journal](https://img.shields.io/badge/Journal-Pattern%20Recognition-0056D2.svg)](https://www.sciencedirect.com/journal/pattern-recognition)
[![License](https://img.shields.io/badge/License-Not%20specified-lightgrey.svg)](#license)

## Overview

DeferredSeg is a pixel-wise learning-to-defer framework for trustworthy medical image segmentation. Instead of requiring a segmentation model to make every decision by itself, the framework learns when and where to route difficult pixels to multiple experts. It combines an aggregated deferral predictor, expert-routing channels, a pixel-wise surrogate collaboration loss, spatial-coherence regularization, and load balancing across experts.

The implementation builds on medical image segmentation backbones including MedSAM and supports multi-seed training, evaluation, and aggregation of Dice, IoU, and sensitivity statistics.

## Authors

Qiuyu Tian, Haoliang Sun, Yunshan Wang, Yinghuan Shi, and Yilong Yin.

## Highlights

- Pixel-wise deferral for fine-grained model–expert collaboration.
- Routing among multiple experts instead of relying on a single expert.
- Spatial-coherence regularization for locally consistent deferral decisions.
- Load-balancing regularization to reduce expert-routing collapse.
- Multi-seed evaluation with mean, variance, standard deviation, and 95% confidence intervals.

## Repository Structure

```text
DeferredSeg/
├── seed_3muti_expertMYtrain_l2d_Single_c_prediction.py  # training and evaluation entry point
├── test_system.py                                       # system-level testing
├── multi_Modify_structure_.py                           # segmentation/deferral architecture changes
├── mutiExperts.py                                       # expert definitions
├── multi_loss.py                                        # collaboration and regularization losses
├── metrics.py                                           # segmentation metrics
├── segment_anything/                                    # SAM/MedSAM model components
└── utils/                                               # preprocessing and checkpoint utilities
```

## Environment

The code requires Python with a CUDA-enabled PyTorch installation for GPU training. Its imports also include NumPy, OpenCV, Matplotlib, MONAI, tqdm, and Numba.

Example environment setup:

```bash
conda create -n deferredseg python=3.10 -y
conda activate deferredseg
pip install torch torchvision
pip install numpy opencv-python matplotlib monai tqdm numba
```

Install the PyTorch build appropriate for your CUDA version. Exact package versions used for the paper are not yet included in this repository.

## Data Preparation

The training and test roots are expected to contain preprocessed NumPy arrays:

```text
dataset_root/
├── imgs/
└── gts/
```

Utilities for organizing, splitting, and preprocessing 2D, 3D, and video medical-image datasets are documented in [`utils/README.md`](utils/README.md). Dataset paths must be supplied explicitly; do not rely on the machine-specific defaults currently present in the scripts.

## Training

Provide the training data, optional test data, MedSAM checkpoint, output directory, device, and random seeds:

```bash
python seed_3muti_expertMYtrain_l2d_Single_c_prediction.py \
  --tr_npy_path /path/to/training_slices \
  --te_npy_path /path/to/test_slices \
  -checkpoint /path/to/medsam_vit_b.pth \
  -work_dir ./work_dir \
  -task_name deferredseg_experiment \
  --device cuda:0 \
  --seeds 2023,2025,2027
```

Training outputs include per-run checkpoints and summaries. Across-seed summaries report Dice, IoU, sensitivity, variance, standard deviation, and 95% confidence intervals.

## Evaluation

Evaluate existing runs using their saved best checkpoints:

```bash
python seed_3muti_expertMYtrain_l2d_Single_c_prediction.py \
  --eval_only \
  --te_npy_path /path/to/test_slices \
  -checkpoint /path/to/medsam_vit_b.pth \
  -work_dir ./work_dir \
  -task_name deferredseg_experiment \
  --device cuda:0
```

To aggregate existing `run_summary.json` files without retraining:

```bash
python seed_3muti_expertMYtrain_l2d_Single_c_prediction.py \
  --aggregate_only \
  -work_dir ./work_dir \
  -task_name deferredseg_experiment
```

## Citation

If this work is useful in your research, please cite:

```bibtex
@article{tian2026deferredseg,
  title   = {DeferredSeg: A Multi-Expert Deferral Framework for Trustworthy Medical Image Segmentation},
  author  = {Tian, Qiuyu and Sun, Haoliang and Wang, Yunshan and Shi, Yinghuan and Yin, Yilong},
  journal = {Pattern Recognition},
  year    = {2026}
}
```

Preprint: [arXiv:2604.12411](https://arxiv.org/abs/2604.12411).

## Acknowledgements

This repository includes components derived from the [Segment Anything](https://github.com/facebookresearch/segment-anything) ecosystem and uses MedSAM-style medical image preprocessing and model initialization.

## License

No license file is currently included. Until a license is added, standard copyright restrictions apply. Please contact the authors regarding reuse beyond what is permitted by law.
