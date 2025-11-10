# Forest Cover Type Classification

This project aims to classify the **forest cover type** (the dominant tree species) in different regions based on various cartographic variables such as elevation, slope, and soil type.  
It demonstrates how **Random Forest** and **XGBoost** classifiers can be applied to a real-world dataset to predict multi-class categorical outcomes.

---

## Overview
The goal of this project is to:
- Explore and understand the **Forest CoverType** dataset.
- Train and evaluate ensemble learning models: **Random Forest** and **XGBoost**.
- Compare the classification performance between the two models.
- Visualize confusion matrices and feature importances.

---

## Machine Learning Techniques Used
- **Random Forest Classifier** – a bagging-based ensemble algorithm using multiple decision trees.
- **XGBoost Classifier** – a boosting-based ensemble model known for high accuracy and efficiency.
- **Model Evaluation Metrics:**
  - Accuracy Score
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-score)

---

## Dataset
- **Source:** `fetch_covtype` from `sklearn.datasets` (no need to download manually)
- **Name:** Forest CoverType
- **Target:** `Cover_Type` – indicates the type of forest cover (1 to 7 classes)
- **Features:** Various cartographic attributes such as:
  - Elevation
  - Aspect
  - Slope
  - Horizontal & Vertical Distances
  - Soil Type (binary encoded)
  - Wilderness Area (binary encoded)

> The dataset is loaded directly using Scikit-learn’s `fetch_covtype(as_frame=True)` function.

---

## Project Structure
```
ForestCoverTypeClassification/
│
├── forest_cover_classification.ipynb
├── README.md
└── requirements.txt
```

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Mohammad-Jaafar/Forest-Cover-Type-Classification.git
   ```
2. Open the notebook in Jupyter or Google Colab.
3. Run all cells step-by-step to reproduce the results.

---

## Results & Visualizations
- Confusion matrix heatmaps for both models.
- Feature importance plot (top 10 most important features).
- Comparison of model accuracies (Random Forest vs XGBoost).
- Classification reports with precision, recall, and F1-score.

---

## Model Comparison
| Model | Accuracy | Highlights |
|--------|-----------|------------|
| **Random Forest** | Moderate accuracy | Stable, interpretable model. |
| **XGBoost** | Higher accuracy | Performs better on large datasets and captures complex relationships. |

---

## Author
**Mohammad Jaafar**  
mhdjaafar24@gmail.com  
[LinkedIn](https://www.linkedin.com/in/mohammad-jaafar-)  
[HuggingFace](https://huggingface.co/Mhdjaafar)  
[GitHub](https://github.com/Mohammad-Jaafar)

---

*If you find this project useful, please give it a star on GitHub!*
