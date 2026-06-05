
# 🌌 Kepler Exoplanet Search Result

### An End-to-End Machine Learning Pipeline for Automated Exoplanet Classification

**Developed by:** **Dua Kalyar**
📊 **Dataset Source:** NASA Kepler Exoplanet Catalog (Kaggle)
🎯 **Permanent Publication Record:** https://doi.org/10.5281/zenodo.20364051

---
## 📌 Abstract
This project presents an end-to-end machine learning pipeline for classifying NASA Kepler transit signals into three categories:
* ✅ Confirmed Exoplanets
* ❌ False Positives
* 🔍 Candidates
To address severe class imbalance, the pipeline incorporates **SMOTE (Synthetic Minority Oversampling Technique)** alongside robust feature scaling. High-dimensional stellar parameters are refined using a collaborative feature selection framework combining **Information Gain, Gini Importance, and Entropy Analysis**.

Six machine learning algorithms were evaluated:
* Logistic Regression
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Decision Tree
* Random Forest
* XGBoost

Performance was assessed using:
* Accuracy
* Precision
* Recall
* F1-Score
* F2-Score
* ROC-AUC
To ensure scientific transparency and model interpretability, the project integrates **SHAP** and **LIME**, transforming high-performing black-box models into explainable and verifiable systems.
---
## 🌌 Machine Learning Pipeline
```text
Raw Data
    ↓
Missing Value Analysis
    ↓
SMOTE Balancing
    ↓
Feature Selection
    ↓
80/20 Train-Test Split
    ↓
Model Training & Evaluation
    ↓
SHAP & LIME Explainability
```
## 🔍 Exploratory Data Analysis (EDA)

### Missing Value Analysis
* Missing value heatmaps were generated to identify and remove uninformative features.
* Data quality assessment improved model reliability.

### Feature Scaling
* A StandardScaler was applied to normalize numerical features with large value ranges, such as orbital periods.
* Label Encoding was used to transform categorical target labels into machine-readable format.

## ⚖️ Class Balancing
The original dataset exhibited significant class imbalance, with **False Positive** instances heavily outnumbering other classes.

To mitigate this issue:
* SMOTE was applied to generate synthetic minority samples.
* Dataset distribution was balanced before model training.
* Model bias toward majority classes was significantly reduced.

## 🎯 Unified Feature Selection
Three complementary feature evaluation techniques were combined:
* Information Gain
* Gini Importance
* Entropy Analysis

Key predictive features identified:
* `kepler_name`
* `koi_pdisposition`
These features consistently demonstrated strong predictive power across all evaluation methods.

## 🤖 Model Development
The balanced dataset was divided using an:
* **80% Training Set**
* **20% Testing Set**
The following models were trained and evaluated:

| Model               | Category             |
| ------------------- | -------------------- |
| Logistic Regression | Linear Model         |
| KNN                 | Distance-Based Model |
| SVM                 | Kernel-Based Model   |
| Decision Tree       | Tree-Based Model     |
| Random Forest       | Ensemble Learning    |
| XGBoost             | Gradient Boosting    |

## 🧠 Explainable AI (XAI)

### SHAP (SHapley Additive Explanations)

Used to:
* Measure global feature importance
* Visualize feature contributions
* Explain overall model behavior

### LIME (Local Interpretable Model-Agnostic Explanations)

Used to:
* Explain individual predictions
* Generate human-readable IF-THEN rules
* Validate high-confidence classifications

## 📈 Performance Results

### 🥇 Decision Tree
* **Accuracy:** 99.97%
* **Precision:** 99.97%
* **Recall:** 99.97%
* **ROC-AUC:** 99.97%

### 🥈 Random Forest
* **Accuracy:** 99.97%
* **Precision:** 99.97%
* **Recall:** 99.97%
* **ROC-AUC:** 99.99%

### 🥉 XGBoost
* **Accuracy:** 99.86%
* **Precision:** 99.86%
* **Recall:** 99.86%
* **ROC-AUC:** 99.99%

### Logistic Regression
* **Accuracy:** 99.36%
* **Precision:** 99.36%
* **Recall:** 99.36%
* **ROC-AUC:** 99.96%

### Support Vector Machine (SVM)
* **Accuracy:** 98.44%
* **Precision:** 98.44%
* **Recall:** 98.44%
* **ROC-AUC:** 99.97%

### K-Nearest Neighbors (KNN)
* **Accuracy:** 96.42%
* **Precision:** 96.42%
* **Recall:** 96.42%
* **ROC-AUC:** 99.50%

## 🔎 Key Insights

### Error Analysis
* Linear and distance-based models (Logistic Regression, KNN, SVM) occasionally struggled with overlapping class boundaries.
* Some confirmed exoplanets were misclassified as candidates.
* Decision Tree and Random Forest models achieved near-perfect class separation.

### Feature Dynamics
* Decision Trees focused heavily on the most influential features.
* Random Forest and XGBoost distributed importance across a broader set of astrophysical indicators.
* Ensemble methods captured complex non-linear relationships more effectively.

### XAI Validation
LIME explanations successfully justified high-confidence predictions by exposing local decision boundaries and feature thresholds, such as:

```text
koi_time0bk_err1 < 0.00
```
This improved trust, transparency, and scientific interpretability.

## 🏁 Conclusion

By combining:
* Robust data preprocessing
* SMOTE-based class balancing
* Unified feature selection
* Advanced ensemble learning
* Explainable AI techniques
this project successfully automates exoplanet verification with a peak accuracy of **99.97%** using **Decision Tree** and **Random Forest** models.
More importantly, the integration of **SHAP** and **LIME** bridges the gap between predictive performance and scientific trust, producing a transparent, interpretable, and reproducible machine learning framework for astronomical research.

### 🚀 Technologies Used
* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* SHAP
* LIME
* Matplotlib
* Seaborn
* Imbalanced-Learn (SMOTE)

---
**Author:** Dua Kalyar
**Field:** Machine Learning • Data Science • Explainable AI • Astronomy Analytics

