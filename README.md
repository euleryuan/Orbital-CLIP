# Orbital-CLIP


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-OpenCLIP-red)](https://github.com/mlfoundations/open_clip)

> **Official implementation** of the paper: *"Sequential Sensitivity Analysis of Multimodal Large Language Models for Rare Orbital Disease Detection"*, accepted in **Communications Medicine** (Nature Portfolio).

## 🚀 Overview

This repository contains the source code for the **Sequential Sensitivity Analysis (SSA)** framework applied to Multimodal Large Language Models (specifically CLIP). Our method enhances the detection of rare orbital diseases by leveraging clinical-specific prompt augmentation and sensitivity-aware fine-tuning.

The framework is designed to align high-dimensional medical imaging features (CT/MRI/Photography) with distinct clinical textual descriptions, improving zero-shot generalization on external validation cohorts.

## 📂 Repository Structure

The codebase is organized into three core modules to ensure reproducibility:

* **`data_preprocess.py`**: A stratified dataset partitioning tool that ensures rare disease prevalence is preserved across training and validation splits.
* **`train_ssa.py`**: The main training engine implementing the SSA protocol, including dynamic clinical prompt sampling and contrastive loss optimization.
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
dataset_root/ ├── orbital disease/ ├── healthy eyes/ └── non-orbital disease/

**Command:**
```bash
python data_preprocess.py

2. Training (SSA Protocol)
Initiate the fine-tuning process using the train_ssa.py script. This script utilizes the specific hyperparameters defined in the manuscript (e.g., Low Learning Rate 1e-8 for stability).

Command:

python train_ssa.py \
  --train-dir "dataset_eye/train" \
  --val-dir "dataset_eye/val" \
  --save-dir "clip_model_checkpoints" \
  --model-name "ViT-B-32-256" \
  --batch-size 200 \
  --epochs 100 \
  --lr 1e-8

3. External Validation & Inference
Evaluate the trained model on an independent external cohort using inference.py. This will output performance metrics and save a visualization of the confusion matrix.

python inference.py \
  --test-dir "path/to/external_test_dataset" \
  --checkpoint "clip_model_checkpoints/clip_model_epoch_best.pt" \
  --model-name "ViT-B-32-256"

Data Availability Statement
The large-scale orbital imaging dataset used in this study is protected under strict clinical data privacy regulations (GDPR/HIPAA compliant).

Due to the sensitive nature of patient data, the raw medical images cannot be publicly hosted in this repository. However, we provide the complete methodological framework, preprocessing logic, and model architecture to ensure reproducibility. Researchers may request access to de-identified data by contacting the corresponding author, subject to institutional ethics committee approval and data sharing agreements.

Markdown
# Orbital-CLIP: Sequential Sensitivity Analysis for Rare Orbital Disease Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-OpenCLIP-red)](https://github.com/mlfoundations/open_clip)

> **Official implementation** of the paper: *"Sequential Sensitivity Analysis of Multimodal Large Language Models for Rare Orbital Disease Detection"*, accepted in **Communications Medicine** (Nature Portfolio).

## 🚀 Overview

This repository contains the source code for the **Sequential Sensitivity Analysis (SSA)** framework applied to Multimodal Large Language Models (specifically CLIP). Our method enhances the detection of rare orbital diseases by leveraging clinical-specific prompt augmentation and sensitivity-aware fine-tuning.

The framework is designed to align high-dimensional medical imaging features (CT/MRI/Photography) with distinct clinical textual descriptions, improving zero-shot generalization on external validation cohorts.

## 📂 Repository Structure

The codebase is organized into three core modules to ensure reproducibility:

* **`data_preprocess.py`**: A stratified dataset partitioning tool that ensures rare disease prevalence is preserved across training and validation splits.
* **`train_ssa.py`**: The main training engine implementing the SSA protocol, including dynamic clinical prompt sampling and contrastive loss optimization.
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
dataset_root/ ├── orbital disease/ ├── healthy eyes/ └── non-orbital disease/


**Command:**
```bash
python data_preprocess.py
Note: You may need to edit the directory paths inside the script (DATASET_ROOT) to match your local setup.

2. Training (SSA Protocol)
Initiate the fine-tuning process using the train_ssa.py script. This script utilizes the specific hyperparameters defined in the manuscript (e.g., Low Learning Rate 1e-8 for stability).

Command:

Bash
python train_ssa.py \
  --train-dir "dataset_eye/train" \
  --val-dir "dataset_eye/val" \
  --save-dir "clip_model_checkpoints" \
  --model-name "ViT-B-32-256" \
  --batch-size 200 \
  --epochs 100 \
  --lr 1e-8
3. External Validation & Inference
Evaluate the trained model on an independent external cohort using inference.py. This will output performance metrics and save a visualization of the confusion matrix.

Command:

Bash
python inference.py \
  --test-dir "path/to/external_test_dataset" \
  --checkpoint "clip_model_checkpoints/clip_model_epoch_best.pt" \
  --model-name "ViT-B-32-256"
🔒 Data Availability Statement
The large-scale orbital imaging dataset used in this study is protected under strict clinical data privacy regulations (GDPR/HIPAA compliant).

Due to the sensitive nature of patient data, the raw medical images cannot be publicly hosted in this repository. However, we provide the complete methodological framework, preprocessing logic, and model architecture to ensure reproducibility. Researchers may request access to de-identified data by contacting the corresponding author, subject to institutional ethics committee approval and data sharing agreements.

📜 Citation
If you utilize this code or methodology in your research, please cite our paper:

Code snippet
@article{YourName2024Orbital,
  title={Sequential Sensitivity Analysis of Multimodal Large Language Models for Rare Orbital Disease Detection},
  author={Your Name and Co-authors},
  journal={Communications Medicine},
  year={2024},
  publisher={Nature Portfolio},
  doi={10.1038/s43856-xxx-xxxx-x}
}
📄 License
This project is licensed under the MIT License.
