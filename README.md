# Credit Card Fraud Detection

> End-to-end machine learning project to detect fraudulent credit card transactions using Random Forest, SMOTE oversampling, and comprehensive EDA visualizations.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)](https://scikit-learn.org)
[![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.9993-brightgreen)](.)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

##  Project Overview

This project addresses the real-world challenge of detecting fraudulent credit card transactions in a highly imbalanced dataset (only 0.43% fraud). The full data science lifecycle is implemented: from raw data ingestion through EDA, preprocessing, model training, and evaluation.

**Dataset:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## Project Goals

| Goal | Status |
|------|--------|
| Import credit card transaction dataset |
| Ensure data privacy (PCA-anonymized features) |
| Identify fraud percentage and anomalies |
| Handle imbalanced data with SMOTE |
| Normalize numerical features  |
| Histograms and boxplots via EDA |
| Correlation analysis between features |
| Feature importance for key fraud indicators |
| AI model to flag suspicious transactions |

---

## Results

| Model | AUC-ROC | AUC-PR | Precision | Recall | F1 | Accuracy |
|-------|---------|--------|-----------|--------|-----|----------|
| **Random Forest**  | **0.9993** | **0.9528** | **0.7812** | **0.9615** | **0.8621** | **0.9987** |
| Logistic Regression | 0.9607 | 0.6504 | 0.3750 | 0.9231 | 0.5333 | 0.9930 |

---

##  Project Structure

```
fraud-detection/
├── train_model.py          # Full ML pipeline
├── index.html              # Project website
├── README.md
├── models/
│   ├── random_forest.pkl
│   ├── logistic_regression.pkl
│   └── scaler.pkl
├── static/
│   ├── stats.json
│   └── charts/
│       ├── class_distribution.png
│       ├── amount_distribution.png
│       ├── correlation_heatmap.png
│       ├── feature_boxplots.png
│       ├── confusion_matrices.png
│       ├── roc_curves.png
│       ├── precision_recall.png
│       └── feature_importance.png
└── data/
    └── (place your CSV chunks here)
```

---

##  Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
# Optional but recommended:
pip install imbalanced-learn
```

### Run the pipeline
```bash
# 1. Place your CSV data in data/
# 2. Update DATA_FILES paths in train_model.py
# 3. Run the pipeline
python train_model.py

# 4. View the website (open index.html in browser, or serve it)
python -m http.server 8000
# then open http://localhost:8000
```

---

## Methodology

### 1. Data Loading & Cleaning
- Merged 3 CSV chunks (~30k rows total)
- Removed duplicates, validated schema

### 2. Exploratory Data Analysis (EDA)
- **Class distribution:** Pie chart showing 99.57% legit vs 0.43% fraud
- **Amount distributions:** Separate histograms for fraud vs legitimate amounts
- **Correlation matrix:** Full feature correlation heatmap
- **Boxplots:** Top 6 most fraud-correlated features compared between classes

### 3. Preprocessing
- **SMOTE:** Synthetic Minority Oversampling Technique using k-nearest neighbours interpolation. Balanced the training set from 103 fraud → 23,809 synthetic fraud samples.
- **StandardScaler:** Normalized all 29 features (V1–V28 + Amount) to zero mean and unit variance.
- **Train/Test Split:** 80/20 stratified split to preserve class ratio.

### 4. Models

**Random Forest (100 estimators, max_depth=12)**
- Ensemble method using decision trees
- `class_weight='balanced'` for additional imbalance handling
- Feature importances derived from mean impurity decrease

**Logistic Regression**
- Baseline linear model
- `class_weight='balanced'`, max_iter=1000
- Useful for comparison and interpretability

### 5. Evaluation Metrics
- **AUC-ROC:** Area under the Receiver Operating Characteristic curve
- **AUC-PR:** Area under the Precision-Recall curve (preferred for imbalanced data)
- **Confusion Matrix:** True/false positives and negatives
- **Precision/Recall/F1** on the fraud class specifically

---

## Key Findings

1. **Extreme class imbalance** (0.43% fraud) makes this a challenging problem — accuracy alone is misleading.
2. **SMOTE dramatically improved** fraud recall from ~60% to ~96% by balancing the training set.
3. **Random Forest far outperforms** Logistic Regression on precision (0.78 vs 0.38) while maintaining similar recall.
4. **V4, V11, V14, V17** were the most important features for fraud detection.
5. **Fraudulent transactions** tend to have lower amounts on average despite some high-value outliers.

---

## Precision vs Recall Tradeoff

In fraud detection, **recall is more important than precision** — missing a fraud case (false negative) costs far more than a false alarm (false positive). The Random Forest achieves **96.15% recall**, meaning it catches the vast majority of fraudulent transactions.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- Dataset: [Machine Learning Group — ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Built as part of an internship ML engineering project

##Live Deployment
-https://salmali-chattopadhyay.github.io/Credit_Card_Fraud_Detection/
