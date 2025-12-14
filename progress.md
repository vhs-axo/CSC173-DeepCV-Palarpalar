# CSC173 Deep Computer Vision Project Progress Report

**Student:** Mann Kristof P. Palarpalar, 2022-0035

**Date:** December 14, 2025

**Project Title:** Skin Disease Classification using Transfer Learning with ResNet50

**Repository:** [vhs-axo/CSC173-DeepCV-Palarpalar](https://github.com/vhs-axo/CSC173-DeepCV-Palarpalar)

## 📊 Current Status

| Milestone | Status | Notes |
| --- | --- | --- |
| Dataset Preparation | ✅ Completed | Split into Train ($80\%$), Val ($10\%$), Test ($10\%$) |
| Initial Training | ✅ Completed | 97 epochs completed |
| Baseline Evaluation | ✅ Completed | Validation Accuracy: ~75% |
| Model Fine-tuning | ⏳ Pending | Evaluation on Test set and specific metric analysis remaining |

## 1. Dataset Progress

* **Source:** [Skin Diseases Image Dataset by Ismail Promus (Kaggle)](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset)
* Total Images: 27,153
* **Train/Val/Test split:** $80\%$ / $10\%$ / $10\%$
* **Classes Implemented:**
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
* **Preprocessing applied:**
	+ Resize ($224 \times 224$)
	+ Normalization (`ImageNet` stats)
	+ Augmentation (`RandomHorizontalFlip`, `RandomRotation(10)`)

**Sample data preview:**

![split_dataset/train/seborrheic_keratoses_other_benign_tumors/7_6.jpgimage caption](static/sample_data_preview.jpg)

## 2. Training Progress

**Current Metrics (Epoch 97):**

| Metric | Train | Val |
|--------|-------|-----|
| Loss | $0.5466$ | $0.6992$ |
| Accuracy | $81.33\%$ | $75.25\%$ |
| Precision | $0.8112$ | $0.7493$ |
| Recall | $0.8133$ | $0.7525$ |

---

## 3. Challenges Encountered & Solutions

| Issue | Status | Resolution |
| --- | --- | --- |
| Limited Data & Overfitting | ✅ Fixed | Implemented Transfer Learning (ResNet50) and Data Augmentation (Flip/Rotate) |
| Class Imbalance | ⏳ Monitoring | Dataset split carefully; monitoring Validation Loss for bias |
| Computational Cost | ✅ Fixed | Utilized CUDA acceleration and optimized Batch Size (512) |

## 4. Next Steps (Before Final Submission)

* [ ] Evaluate model on the held-out **Test** split (10%) to confirm generalization
* [ ] Generate detailed classification report (Precision, Recall, F1-Score) per class
* [ ] Create Confusion Matrix visualization
* [ ] Record 5-min demo video
* [ ] Write complete README.md with final results