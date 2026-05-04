# ICPR 2026 Experiment Notebooks

This directory contains representative notebooks for the experiments presented in the paper:

> Wonder A. L. Alves, Lucas de P. O. Santos, Ronaldo F. Hashimoto, Nicolas Passat, Anderson H. R. Souza, Dennis J. Silva, Yukiko Kenmochi.  
> **A trainable connected filter preprocessing layer based on component trees.**  
> International Conference on Pattern Recognition (ICPR), 2026, Lyon, France.  
> [hal-05575141](https://hal.science/hal-05575141/)

Each notebook corresponds to the **first execution out of 10 independent runs with different random seeds** for a given configuration.

The goal is to provide **readable and reproducible examples** of the experimental pipeline, rather than the full set of runs used for statistical evaluation in the paper.

## Method Overview

Each configuration compares two models:

- **Baseline**: the segmentation backbone operates directly on the input image  
- **CFP variant**: a trainable morphological preprocessing layer (Connected Filter Preprocessing) is applied before the backbone, using attributes derived from component trees (MTLearn)

## What is included

Each notebook presents:

- training curves  
- qualitative predictions  
- decision threshold analysis  
- final evaluation metrics  

## Configurations

| Dataset | Backbone | Notebook |
| --- | --- | --- |
| Plants | ConvNet | [Exp_plants_segmentation_ConvNet_run_000.ipynb](Exp_plants_segmentation_ConvNet_run_000.ipynb) |
| Plants | ID3-NN | [Exp_plants_segmentation_ID3-NN_run_000.ipynb](Exp_plants_segmentation_ID3-NN_run_000.ipynb) |
| Plants | Unet | [Exp_plants_segmentation_Unet_run_000.ipynb](Exp_plants_segmentation_Unet_run_000.ipynb) |
| Screws | ConvNet | [Exp_screws_segmentation_ConvNet_run_000.ipynb](Exp_screws_segmentation_ConvNet_run_000.ipynb) |
| Screws | ID3-NN | [Exp_screws_segmentation_ID3-NN_run_000.ipynb](Exp_screws_segmentation_ID3-NN_run_000.ipynb) |
| Screws | Unet | [Exp_screws_segmentation_Unet_run_000.ipynb](Exp_screws_segmentation_Unet_run_000.ipynb) |

## Data

- The screw segmentation dataset is automatically handled via the MTLearn dataset downloader.
- The plant dataset is not distributed with this repository.

To reproduce the plant experiments:

1. Request access from: https://www.plant-phenotyping.org/datasets-home  
2. Register the dataset in the local MTLearn dataset registry  
3. Ensure the dataset follows the expected directory structure  
