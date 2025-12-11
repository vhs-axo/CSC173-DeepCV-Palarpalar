# CSC173 Deep Computer Vision Project Proposal

**Student:** Mann Kristof P. Palarpalar, 2022-0035

**Date:** December 11, 2025

## 1. Project Title

**Skin Disease Classification using Transfer Learning with ResNet50**

## 2. Problem Statement

Skin diseases are a pervasive health issue, yet access to specialized dermatological care is often scarce, particularly in remote or underserved regions. This shortage frequently leads to delayed diagnoses, worsened conditions, and increased healthcare costs. This project addresses this gap by developing an automated computer vision system capable of classifying various skin conditions from images. By leveraging deep learning, we aim to provide a supportive diagnostic tool that can assist in early detection and triage.

## 3. Objectives

* **Develop and train** a Convolutional Neural Network (CNN) using Transfer Learning (ResNet50) to classify skin diseases.
* **Implement a robust training pipeline** in PyTorch, including data ingestion, image normalization, and data augmentation.
* **Evaluate model performance** using accuracy and loss metrics across training and validation splits to ensure generalization.

## 4. Dataset Plan

* **Source:** [Skin Diseases Image Dataset by Ismail Promus (Kaggle)](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset)
* **Classes:** Multi-class classification based on dataset directory structure (e.g., Acne, Melanoma, Psoriasis, etc.).
* **Acquisition:** Public download via Kaggle, followed by splitting into a structured directory format (`split_dataset/train`, `split_dataset/val`, and `split_dataset/test`) to be compatible with `torchvision.datasets.ImageFolder`.

## 5. Technical Approach

**Architecture Sketch:**
We will utilize **Transfer Learning** rather than training a network from scratch.

1. **Backbone:** ResNet50 pre-trained on ImageNet (weights `IMAGENET1K_V2`).
2. **Feature Extraction:** The convolutional layers will be **frozen** to retain learned feature maps.
3. **Classifier Head:** The final Fully Connected layer will be replaced with a new layer.

**Implementation Details:**

* **Model:** ResNet50 (Frozen Backbone + Trainable Head).
* **Framework:** PyTorch (using `torch`, `torchvision`).
* **Hardware:** NVIDIA GPU (CUDA) via Local Machine.

## 6. Expected Challenges & Mitigations

* **Challenge: Limited Data & Overfitting**

	*Mitigation:* Usage of a pre-trained model (ResNet50) reduces the need for massive datasets. We also employ **Data Augmentation** in the training loop (Random Horizontal Flips and Random Rotations) to artificially expand the dataset diversity.


* **Challenge: Class Imbalance**

	*Mitigation:* The dataset will be split carefully into training and validation sets to ensure representation. Performance will be monitored via validation loss to detect bias toward dominant classes.