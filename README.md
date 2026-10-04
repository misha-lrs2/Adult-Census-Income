# Adult Census Income Predictor

An end-to-end machine learning case study that predicts whether an individual earns more or less than **$50,000 per year**, using the UCI [Adult Census Income](https://archive.ics.uci.edu/ml/datasets/adult) dataset. The project compares ethically unconstrained and fairness-aware models, then deploys the ethical variant as an interactive **Streamlit** dashboard with live inference.

## Executive Summary

This project answers two research questions:

1. **How accurately can income be predicted using only ethically acceptable features?**
2. **What is the performance cost of removing sensitive attributes such as sex and race?**

The ethical Random Forest classifier achieves **84% accuracy** on held-out test data while excluding `sex`, `race`, and `native.country`. Removing sensitive features costs only **1 percentage point** in accuracy and **3 percentage points** in recall for the high-income class—a negligible trade-off for a model free of direct demographic discrimination.

## Technical Overview

### Problem Formulation

Binary classification: predict `income` as `<=50K` or `>50K` from census demographics and employment features.

### Algorithms Evaluated

| Model | Role |
|-------|------|
| **Decision Tree Classifier** | Baseline interpretable model |
| **Random Forest Classifier** | Best cross-validated performer; selected for deployment |
| **k-Nearest Neighbors** | Distance-based baseline (requires feature scaling) |

### Feature Engineering

- **`capital_gain_log`**: log-transform of skewed `capital.gain` column
- **`is_capital_maxed`**: binary flag for the $99,999 reporting cap
- **One-hot encoding** for categorical variables (`workclass`, `occupation`, etc.)
- **StandardScaler** applied to numeric features before training

### Ethical Model Design

The deployed model intentionally excludes `sex`, `race`, and `native.country`. Feature importance analysis on the unconstrained model showed that `sex_Male` and `race_White` ranked in the top 15 predictors—confirming the need for exclusion.

### Architecture

```
notebooks/casus.ipynb          → EDA, training, hyperparameter tuning
notebooks/train_model.ipynb    → Final model serialization
prediction_pkg/
  ├── model.pkl                → Trained Random Forest
  ├── scaler.pkl               → Fitted StandardScaler
  ├── params.json              → Expected column schema after encoding
  └── inference_pipeline.py    → Preprocessing + prediction API
app.py                         → Streamlit dashboard (4 tabs)
```

## Installation & Setup

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) (recommended) or Python 3.10+

### Conda Environment

```bash
git clone <repository-url>
cd adult-census-income
conda env create -f requirements.yml
conda activate ai-s3
```

### Verify Dependencies

The environment installs: `python=3.10`, `pandas`, `scikit-learn`, `plotly`, and `streamlit`.

## Usage

### Launch the Dashboard

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501` with four tabs:

| Tab | Description |
|-----|-------------|
| **Introduction & Ethics** | Project context and fairness rationale |
| **Data Visualizations** | Interactive Plotly charts (age, education, occupation vs. income) |
| **Model Performance** | Accuracy, precision, recall, and confusion matrix |
| **Live Predictions** | CSV bulk upload or manual single-profile inference |

### Bulk Prediction (CSV Upload)

Upload a CSV with at minimum: `age`, `education.num`, `occupation`, `hours.per.week`. Use `data/test.csv` as a reference format. Click **Predict Incomes in Bulk** and download results.

### Programmatic Inference

```python
import pandas as pd
from prediction_pkg.inference_pipeline import predict_new_data

sample = pd.DataFrame([{
    "age": 35,
    "education.num": 13,
    "occupation": "Prof-specialty",
    "workclass": "Private",
    "capital.gain": 0,
    "capital.loss": 0,
    "hours.per.week": 40,
}])

predictions = predict_new_data(sample)  # 0 = <=50K, 1 = >50K
```

### Reproduce Training

Open and run `notebooks/casus.ipynb` for the full analysis pipeline, or `notebooks/train_model.ipynb` for model export only.

## Results & Interpretation

### Ethical Random Forest — Test Set Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 84% | Correct on 84% of all predictions |
| **Precision (>50K)** | 76% | When the model predicts high income, it is right 76% of the time |
| **Recall (>50K)** | 48% | Detects roughly half of all actual high earners |

### Confusion Matrix (Test Set, n = 6,480)

|  | Predicted <=50K | Predicted >50K |
|--|-----------------|----------------|
| **Actual <=50K** | 4,674 (TN) | 240 (FP) |
| **Actual >50K** | 816 (FN) | 750 (TP) |

The model is conservative toward the minority class (`>50K`): it excels at identifying low earners but misses nearly half of high earners. This reflects class imbalance (~76% `<=50K`) and the inherent difficulty of predicting high income from objective features alone.

### Ethical vs. Non-Ethical Comparison

| Metric | Non-Ethical (with sex/race) | Ethical (excluded) | Delta |
|--------|----------------------------|--------------------|-------|
| Accuracy | 85% | 84% | −1 pp |
| Precision (>50K) | 0.78 | 0.76 | −0.02 |
| Recall (>50K) | 0.51 | 0.48 | −0.03 |

**Conclusion:** Reliable, ethically responsible income prediction is achievable. Sensitive features provide negligible performance gain and introduce unacceptable bias.

## Project Structure

```
adult-census-income/
├── app.py
├── requirements.yml
├── data/
│   ├── adult.csv
│   └── test.csv
├── notebooks/
│   ├── casus.ipynb
│   └── train_model.ipynb
└── prediction_pkg/
    ├── inference_pipeline.py
    ├── model.pkl
    ├── scaler.pkl
    └── params.json
```

## Author

**Misha Leenders** — AI & Machine Learning portfolio project (2025)
