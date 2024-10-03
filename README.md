# RSNA 2024 Lumbar Spine Degenerative Classification

This project is designed to participate in the **RSNA 2024 Lumbar Spine Degenerative Classification** competition. The aim is to create machine learning models that simulate a radiologist's performance in diagnosing degenerative spine conditions using **lumbar spine MRI images**. The project focuses on multi-task classification of five key conditions related to lumbar spine degeneration, helping radiologists detect, assess severity, and classify conditions accurately.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Configuration](#configuration)
- [License](#license)

## Overview

Low back pain is the leading cause of disability globally, often attributed to degenerative conditions of the lumbar spine. Using **MRI scans**, the competition aims to classify conditions such as:

- **Left Neural Foraminal Narrowing**
- **Right Neural Foraminal Narrowing**
- **Left Subarticular Stenosis**
- **Right Subarticular Stenosis**
- **Spinal Canal Stenosis**

Participants develop models to classify the severity of these conditions across five vertebrae levels (L1/L2 to L5/S1) into three severity grades:

1. Normal/Mild
2. Moderate
3. Severe

## Dataset

The dataset consists of MRI images in DICOM format, providing detailed medical imaging for the lumbar spine. Labels include severity scores for the above conditions. The dataset is multi-institutional, sourced from eight different sites across five continents.

## Model Architecture

This project supports several advanced models for 3D MRI analysis using PyTorch and MONAI:

- **ResNet3D**: For volumetric feature extraction.
- **SwinUNETR**: A transformer-based architecture for medical imaging.
- **SegResNet**: A specialized segmentation model for volumetric data.
- **DynUNet**: A dynamic U-Net with adjustable depth.
- **ViT**: Vision Transformer for efficient spatial attention.

### Multi-Task Learning:

The model performs **multi-task classification** on three labels (central, left, and right severity scores), using a combination of image data and metadata (e.g., slice thickness, pixel spacing).

## Installation

Follow these steps to set up and run the project:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/anto18671/lumbar-spine-degenerative-classification.git
   cd rsna-lumbar-classification
   ```

2. **Install dependencies**:
   Use the provided `requirements.txt` to install the necessary packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up data**:
   Place the dataset in the `data/` directory.

## Usage

To train and evaluate the model, run the following:

```bash
python main.py
```

### Key Features:

- **Mixed Precision Training**: Utilizes PyTorch's `autocast` for improved memory efficiency.
- **Stratified Group K-Fold**: Data split to preserve label distribution and prevent data leakage.

## Preprocessing

Preprocessing includes:

- **DICOM Loading**: Loading images from DICOM format, correcting orientation, and handling pixel values.
- **Spatial Scaling**: Rescaling images based on pixel spacing for consistent physical dimensions.
- **Augmentations**: Elastic transformations, rotations, horizontal flips, and Gaussian noise applied during training.

## Training

The training process uses a multi-task loss function, where each task (central, left, and right) is trained separately using class-weighted cross-entropy.

### Training Configurations:

- **Model architecture**: ResNet3D, SwinUNETR, SegResNet, DynUNet, or ViT.
- **Loss function**: Multi-task cross-entropy with class weights.
- **Optimizer**: AdamW with weight decay and learning rate scheduling.
- **Scheduler**: Exponential learning rate decay.

## Evaluation

After training, the model is evaluated on validation data using the following metrics:

- **AUC-ROC**: Evaluates the Area Under the Curve for Receiver Operating Characteristics across severity classes.
- **Confusion Matrix**: To visualize prediction accuracy per class.
- **Classification Report**: Includes precision, recall, and F1 scores for each severity class.

## Results

The performance is measured by weighted log loss with the following sample weights:

- Normal/Mild: **1**
- Moderate: **2**
- Severe: **4**

## Configuration

Key configurations are defined in the `CONFIG` dictionary at the beginning of the script:

- `enable_spatial_scaling`: Enable or disable spatial scaling using pixel spacing.
- `model_architecture`: Choose between 'ResNet3D', 'SwinUNETR', 'SegResNet', 'DynUNet', or 'ViT'.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
