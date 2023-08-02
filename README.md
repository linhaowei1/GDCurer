# GDCurer: An AI-assisted Drug Dosage Prediction System for Graves' Disease

This repository contains the code for our paper [GDCurer: An AI-assisted Drug Dosage Prediction System for Graves' Disease](#) by <a href="https://linhaowei1.github.io/"> Haowei Lin</a>, Zhao Chen, Lei Kang, and <a href="https://majianzhu.com/"> Jianzhu Ma.


## Quick Links

  - [Overview](#overview)
  - [Requirements](#requirements)
  - [GDCurer Training](#gdcurer-training)
    - [Data](#data)
    - [Standard Training Experiments](#standard-training-experiments)
    - [Lifelong Training Experiments](#lifelong-training-experiments)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)

## Overview

We propose an AI-assisted drug dosage prediction system for Graves' Disease. GDCurer is capable of predicting the optimal dosage of idonine-131 by leveraging  the patient's thyroid scintigraph (TS), iodine uptake (IU) information, and the half-life of $^{131}$I (HL). To address the issue of lacking accurate drug dosages in the training data due to the bias introduced by clinicans, a novel machine learning method is developed to exploit the treatment information to correct the bias. Furthermore, GDCurer is designed as a lifelong learner which can continously learn from new clinical data by incorporating the techniques of experience replay (ER) and feature distillation (FD).

![](figures/GDcurer.png)


## Requirements

First, install PyTorch by following the instructions from [the official website](https://pytorch.org). To faithfully reproduce our results, please use the correct `1.8` version corresponding to your platforms/CUDA versions. PyTorch version higher than `1.8` should also work. For example, if you use Linux and **CUDA11.1** ([how to check CUDA version](https://varhowto.com/check-cuda-version/)), install PyTorch by the following command,

```bash
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

Then run the following script to install the remaining dependencies,

```bash
pip install -r requirements.txt
```

## GDCurer Training

In the following section, we describe how to train GDCurer model by using our code.

### Data

Before training and evaluation, please download the dataset from this [Google Drive link](#) and save them in the `./data` directory. 

### Standard Training Experiments

**Training scripts**

We provide an example training script to run standard training of GDCurer. Symply run

```bash
CUDA_VISIBLE_DEVICES=${your_cuda_device_id} bash scripts/standard.sh
```

For the results in the paper, we use Nvidia Tesla A100 GPUs with CUDA 11. Using different types of devices or different versions of CUDA/other software may lead to slightly different performance.

### Lifelong Training Experiments

To run lifelong learning experiments with 4 training phases, simply run

```bash
CUDA_VISIBLE_DEVICES=${your_cuda_device_id} bash scripts/lifelong.sh
```

## Bugs or questions?

If you have any questions related to the code or the paper, feel free to email [Haowei](`linhaowei@pku.edu.cn`). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Citation

Please cite our paper if you use GDCurer in your work:

```bibtex
@inproceedings{lin2023gdcurer,
   title={GDCurer: An AI-assisted Drug Dosage Prediction System for Graves' Disease},
   author={Lin, Haowei and Chen, Zhao and Kang, Lei and Ma, Jianzhu},
   year={2023}
}
```