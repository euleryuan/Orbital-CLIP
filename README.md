# Orbital-CLIP


[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-OpenCLIP-red)](https://github.com/mlfoundations/open_clip)

> **Official implementation** of the paper: *"Sequential Sensitivity Analysis of Multimodal Large Language Models for Rare Orbital Disease Detection"*

## 🚀 Overview

This repository contains the source code for the **Sequential Sensitivity Analysis (SSA)** framework applied to Multimodal Large Language Models (specifically CLIP). Our method enhances the detection of rare orbital diseases by leveraging clinical-specific prompt augmentation and sensitivity-aware fine-tuning.

The framework is designed to align high-dimensional medical imaging features (CT/MRI/Photography) with distinct clinical textual descriptions, improving zero-shot generalization on external validation cohorts.

## 📂 Repository Structure

The codebase is organized into three core modules to ensure reproducibility:

* **`data_preprocess.py`**: A stratified dataset partitioning tool that ensures rare disease prevalence is preserved across training and validation splits.
* **`clip_train.py`**: The main training engine implementing the SSA protocol, including dynamic clinical prompt sampling and contrastive loss optimization.
* **`inference.py`**: An inference engine for external validation, generating publication-quality metrics (Confusion Matrices, ROC-AUC, Classification Reports).

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/Orbital-CLIP.git](https://github.com/YourUsername/Orbital-CLIP.git)
    cd Orbital-CLIP
    ```

2.  **Install dependencies:**
    We recommend using a virtual environment (Conda or venv).
    ```bash
    pip install torch torchvision open_clip_torch pandas numpy matplotlib seaborn tqdm pillow scikit-learn
    ```

## ⚙️ Usage Guide

### 1. Data Preparation
Organize your raw dataset into class-specific folders. Run the preprocessing script to generate a stratified split.

**Expected Input Structure:**
```text
dataset_root/
    ├── orbital disease/
    ├── healthy eyes/
    └── non-orbital disease/
```
Note: Ensure the dataset_root path within the script points to your local data directory.

### 2. Fine-tuning (clip_train.py)
This module fine-tunes the CLIP backbone using the Sequential Sensitivity Analysis protocol. It utilizes dynamic clinical prompt sampling to enhance model sensitivity to rare pathologies.

Key Parameters:

--lr: Learning rate (Optimized at 1e-8 for medical domain adaptation).

--batch-size: Set to 200 to maintain contrastive learning stability.

Command:

```bash
python clip_train.py \
  --train-dir "dataset_root/train" \
  --val-dir "dataset_root/val" \
  --save-dir "clip_model_checkpoints" \
  --model-name "ViT-B-32-256" \
  --batch-size 200 \
  --epochs 100 \
  --lr 1e-8
```
### 3. External Validation & Metrics (inference.py)
Evaluate the model's generalization on independent external cohorts. This script calculates the Top-1 accuracy, F1-score, and generates a publication-ready confusion matrix.

Command:

```bash
python inference.py \
  --test-dir "path/to/external_validation_set" \
  --checkpoint "clip_model_checkpoints/clip_model_epoch_best.pt" \
  --model-name "ViT-B-32-256"
```
## 🔒 Data Availability Statement
The clinical orbital imaging dataset utilized in this study contains sensitive patient information and is protected under institutional privacy protocols (compliant with GDPR/HIPAA).

To respect patient confidentiality, raw images are not hosted in this public repository. However, we provide the complete methodological framework, preprocessing pipelines, and core training logic to allow for full algorithmic replication. For access to de-identified data for research purposes, please contact the corresponding author, subject to ethical approval and data-use agreements.

## 📜 Citation
If this codebase assists in your research, please cite our work: Sequential Sensitivity Analysis of MultimodalLarge Language Models for Rare Orbital Disease Detection.

## 📄 License
This project is licensed under the Apache License 2.0.
