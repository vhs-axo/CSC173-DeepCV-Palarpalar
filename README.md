# Skin Disease Classification using Transfer Learning with ResNet50
**CSC173 Intelligent Systems Final Project**  
*Mindanao State University - Iligan Institute of Technology*

**Student:** Mann Kristof P. Palarpalar, 2022-0035  
**Semester:** AY 2025-2026

[![Python](https://img.shields.io/badge/Python-3.13+-blue)](https://python.org) [![PyTorch](https://img.shields.io/badge/PyTorch-2.9-orange)](https://pytorch.org)

## Abstract
Skin diseases are a pervasive health issue often exacerbated by the scarcity of specialized care in remote regions. This project presents an automated computer vision system designed to classify 10 different skin conditions, including Melanoma, Eczema, and Atopic Dermatitis. We utilized **Transfer Learning** with a **ResNet50** architecture pre-trained on ImageNet, fine-tuning it on the "Skin Diseases Image Dataset" (27,153 images). To address class imbalance and limited data, we employed weighted Cross-Entropy Loss and extensive data augmentation. Current results show a **Training Accuracy of 97.36%**, a **Validation Accuracy of 82.31%**, and a **Test Accuracy of 82.04%**, demonstrating the model's potential as a supportive diagnostic tool for early detection and triage.



## Table of Contents
- [Introduction](#introduction)
- [Methodology](#methodology)
- [Experiments & Results](#experiments--results)
- [Discussion](#discussion)
- [Ethical Considerations](#ethical-considerations)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [References](#references)



## Introduction
### Problem Statement
Skin diseases are a pervasive health issue, yet access to specialized dermatological care is often scarce, particularly in remote or underserved regions. This shortage frequently leads to delayed diagnoses, worsened conditions, and increased healthcare costs. This project addresses this gap by developing an automated computer vision system capable of classifying various skin conditions from images.

### Objectives
- **Develop and train** a Convolutional Neural Network (CNN) using Transfer Learning (ResNet50) to classify skin diseases.
- **Implement a robust training pipeline** in PyTorch, including data ingestion, image normalization, and data augmentation.
- **Evaluate model performance** using accuracy and loss metrics across training and validation splits to ensure generalization.



## Methodology
### Dataset
* **Source:** [Skin Diseases Image Dataset by Ismail Promus (Kaggle)](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset)
* **Size:** 27,153 Total Images
* **Split:** Train (80%) / Validation (10%) / Test (10%)
* **Classes (10):**
  + `atopic_dermatitis`
  + `basal_cell_carcinoma`
  + `benign_keratosis_like_lesions`
  + `eczema`
  + `melanocytic_nevi`
  + `melanoma`
  + `psoriasis_pictures_lichen_planus_related_diseases`
  + `seborrheic_keratoses_other_benign_tumors`
  + `tinea_ringworm_candidiasis_other_fungal_infections`
  + `warts_molluscum_other_viral_infections`

### Architecture
* **Backbone:** ResNet50 - `IMAGENET1K_V2` Weights
* **Head:** Fully Connected (Linear) Layer (Output: 10 classes)
* **Hyperparameters:**

	| Parameter | Value |
	|-----------|-------|
	| Batch Size | 64 |
	| Learning Rate | 0.001 |
	| Epochs | 50 |
	| Optimizer | `Adam` |
	| Criterion | `CrossEntropyLoss` |
	| LR Scheduler | `ReduceLROnPlateau` |

### Technical Approach
We utilized **Transfer Learning** to leverage features learned from the ImageNet dataset.

1.  **Architecture:** **ResNet50** (Weights: `IMAGENET1K_V2`).
2.  **Configuration:**
    * **Frozen Backbone:** Initial convolutional layers are frozen to retain learned feature maps.
    * **Trainable Head:** The final fully connected layer (`fc`) is replaced and trained for the 10 specific classes.
    * **Unfrozen Layer:** The last convolutional block (`layer4`) was also unfrozen to adapt high-level features.
3.  **Preprocessing:**
    * Resize to $224 \times 224$.
    * Normalization using standard `ImageNet` mean and std.
    * **Augmentation:** `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotation(90)`, and `ColorJitter`.
4.  **Optimization:**
    * **Loss Function:** CrossEntropyLoss with **Class Weights** to handle imbalance.
    * **Optimizer:** Adam (`lr=1e-3`) with `ReduceLROnPlateau` scheduler.

### Training Code Snippet

```python
def train_model(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    num_epochs: int,
) -> nn.Module:
    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass if training
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            # Calculate Epoch Metrics
            epoch_loss = running_loss / dataset_sizes[phase]
            
            if phase == "val":
                scheduler.step(epoch_loss)

    return model
```



## Experiments & Results
### Metrics
| Accuracy | Precision | Recall | Inference Time (ms) |
|----------|-----------|--------|---------------------|
| **82.04%** | **0.8682** | **0.8667** | **20** |

### Training Challenges
* **Class Imbalance:** Significant variance in image counts per class. Addressed by calculating class weights for the loss function.
* **Overfitting:** Initial signs of overfitting were mitigated by implementing aggressive data augmentation (flips, rotations, color jitter).



## Discussion
* **Strengths:** The use of Transfer Learning allowed the model to achieve respectable accuracy (>80%) without requiring a massive dataset or training from scratch. The robust preprocessing pipeline ensures the model generalizes well to variations in orientation and lighting.
* **Limitations:** The validation accuracy (~82%) indicates room for improvement, likely due to the visual similarity between certain skin conditions (e.g., different types of keratosis).
* **Insights:** Unfreezing the last convolutional block (`layer4`) in addition to the classifier head was crucial for adapting the model to medical imaging features.



## Ethical Considerations
* **Bias:** The dataset's demographic distribution (e.g., skin tones) is not fully documented. If the training data is skewed toward lighter skin tones, the model may perform poorly on darker skin, exacerbating healthcare disparities.
* **Misuse:** This tool is intended for **educational and supportive purposes only**. It is **not** a replacement for professional medical diagnosis. False negatives (classifying a malignant lesion as benign) could have severe consequences.



## Conclusion
This project successfully implemented a deep learning pipeline for skin disease classification, achieving a validation accuracy of 82.04%. By leveraging ResNet50 and handling class imbalance with weighted loss, we demonstrated the feasibility of automated triage systems. Future work includes evaluating the model on the held-out test set, generating a confusion matrix to analyze specific class confusions, and deploying the model to a mobile interface for real-time inference.



## Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/vhs-axo/CSC173-DeepCV-Palarpalar
    cd CSC173-DeepCV-Palarpalar
    ```
2.  **Install dependencies and create virtual environment using [`uv`](https://pypi.org/project/uv/):**
    ```bash
    uv sync
	uv venv
    ```
3.  **Run the Notebook:**
    Open `notebook.ipynb` in Jupyter or Google Colab to run the training pipeline and view the demo.



## References
1.  Hossain, I. (2021, August 16). _Skin Diseases Image Dataset_. Kaggle. Retrieved December 11, 2025, from [https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset)
2.  He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep residual learning for image recognition. _arXiv (Cornell University)_. [https://doi.org/10.48550/arxiv.1512.03385](https://doi.org/10.48550/arxiv.1512.03385).
3.  Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E. Z., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., . . . Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. _arXiv (Cornell University), 32,_ 8026–8037. [https://doi.org/10.48550/arxiv.1912.01703](https://doi.org/10.48550/arxiv.1912.01703).



## GitHub Pages
View this project: [https://vhs-axo.github.io/CSC173-DeepCV-Palarpalar/](https://vhs-axo.github.io/CSC173-DeepCV-Palarpalar/)
