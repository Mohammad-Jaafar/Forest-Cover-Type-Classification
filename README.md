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

## Features

-   Train and evaluate a **Random Forest** classifier
-   Train and evaluate an **XGBoost** classifier
-   Visualize confusion matrices for both models
-   Plot top features contributing to classification
-   Compare model performance using standard metrics (accuracy, precision, recall, F1-score)

---

## Technologies Used

-   **Python 3.9+**
-   **NumPy / Pandas**
-   **Matplotlib / Seaborn**
-   **Scikit-learn** (Random Forest, dataset loading, metrics)
-   **XGBoost** (gradient boosting classifier)

---

## Project Structure
```
ForestCoverTypeClassification/
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

## Results & Training Performance

### Random Forest
- Moderate accuracy depending on hyperparameters
- Confusion matrix and classification report provide insight into performance
- Top features identified via feature importance plot

### XGBoost
- Higher accuracy than Random Forest
- Captures complex patterns and relationships in the data
- Best performance achieved after proper hyperparameter tuning

**Conclusion:**  
XGBoost performs better overall, while Random Forest provides a stable and interpretable baseline.

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
