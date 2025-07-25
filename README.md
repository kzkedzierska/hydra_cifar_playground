# Hydra CIFAR Playground

A toy project to systematize and create a clean learning project for Hydra configuration management, uv dependency management, and CIFAR-10 image classification.

## Overview

This project demonstrates how to use Hydra for managing machine learning experiments with different model architectures, training configurations, and data settings. Built around CIFAR-10 classification with PyTorch.

## Project Structure

```
hydra_cifar_playground/
├── conf/                   # Hydra configuration files
│   ├── config.yaml        # Main config with defaults
│   ├── data/              # Dataset configurations
│   ├── model/             # Model architecture configs
│   ├── system/            # Compute environment settings
│   └── training/          # Training hyperparameters
├── scripts/               # Entry point scripts
│   └── train.py          # Main training script
├── src/image_wizard/      # Package source code
│   ├── models/           # Model implementations
│   └── utils/            # Data loading utilities
└── notebooks/            # Exploratory notebooks
```

## Setup

1. Install dependencies with uv:

```bash
uv sync
```

2. Run training with default configuration:

```bash
python scripts/train.py
```

3. Override configurations:

```bash
python scripts/train.py model=resnet18 data=full training.lr=0.01
```

## Features

- Configurable model architectures (VanillaCNN, ResNet variants)
- Flexible data loading (full dataset vs subset for quick experiments)
- Multiple training configurations
- Integrated gradients for model interpretability
- Easy experiment tracking and comparison

## Learning Goals

- Master Hydra configuration composition
- Practice structured ML project organization
- Explore configuration-driven experimentation
- Understand the benefits of reproducible ML workflows
