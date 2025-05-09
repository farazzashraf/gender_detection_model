# 🧠 Gender Detection from Face Images

This project is a deep learning pipeline that detects gender (Male or Female) from facial images using bounding box data. Built on top of TensorFlow and trained on the IMDB-clean dataset, this model utilizes real human face images and learns to classify gender with impressive accuracy.

---

## 📁 Dataset

* **Name**: [IMDB-clean (yuulind/imdb-clean)](https://www.kaggle.com/datasets/yuulind/imdb-clean)
* **Contents**:

  * Face image file paths
  * Bounding box coordinates
  * Gender labels (`M` or `F`)
* **Data Format**: CSV metadata files with paths and annotations

---

## 🔄 Pipeline Overview

### 1️⃣ Setup & Initialization

* ✅ TensorFlow with mixed precision (for performance)
* ✅ GPU support enabled (Colab)
* ✅ Dataset pulled via `kagglehub`

### 2️⃣ Data Loading

* Loads `train`, `valid`, and `test` CSVs from the KaggleHub cache.
* Converts gender labels to binary (`M → 1`, `F → 0`).

### 3️⃣ Face Preprocessing

* Reads images using OpenCV.
* Crops images using provided bounding boxes.
* Resizes to 224×224.
* Normalizes pixel values to `[0, 1]`.

### 4️⃣ Data Generator

* Custom generator that:

  * Loads images dynamically from disk
  * Applies on-the-fly augmentation (if extended)
  * Feeds batches directly into the model

### 5️⃣ Model Training

* Deep CNN model trained on cropped face images.
* Likely architecture: Custom CNN.
* Validation accuracy tracked each epoch.

### 6️⃣ Evaluation

* Accuracy, confusion matrix, classification report
* Potential visualizations (loss curves, prediction samples)

---

## 🚀 Getting Started

### 🧹 Prerequisites

Install the required Python libraries:

```bash
pip install tensorflow pandas numpy opencv-python matplotlib kagglehub scikit-learn
```

### 📅 Download Dataset

This notebook auto-downloads the dataset via `kagglehub`, no manual upload needed:

```python
import kagglehub
path = kagglehub.dataset_download("yuulind/imdb-clean")
```

### 🏃 Run the Notebook

Launch Jupyter or Colab, open `Gender_Detection.ipynb`, and run all cells.

---

## 📊 Results Snapshot

| Metric      | Score     |
| ----------- | --------- |
| Accuracy    | 88.43%  |
| Model Type  | CNN-based |
| Input Shape | 224×224×3 |

> Results achieved with a custom CNN trained on cropped faces from the IMDB-clean dataset.

---
