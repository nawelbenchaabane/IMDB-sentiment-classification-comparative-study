# Stroke Risk Prediction ML

End-to-end machine learning project for stroke risk prediction using public or synthetic clinical-style tabular data. The project is designed to run on CPU and demonstrates a complete ML workflow: data preprocessing, class imbalance handling, model comparison, evaluation, interpretability-oriented reporting, and a lightweight Streamlit demo.

> **Disclaimer**  
> This project is for educational and portfolio purposes only. It is not intended for clinical use, diagnosis, treatment recommendation, or real-world medical decision-making.

## 1. Project objective

Stroke is a high-impact clinical event where early risk assessment can support prevention-oriented workflows. The objective of this project is to build a reproducible machine learning pipeline that predicts whether a patient is at risk of stroke based on demographic, clinical, and lifestyle-related variables.

The project focuses on:

- clean and reproducible preprocessing;
- stratified train/test split;
- handling imbalanced classes;
- comparison of CPU-friendly machine learning models;
- medical evaluation metrics such as sensitivity/recall and specificity;
- transparent limitations and model card documentation;
- lightweight demo application.

## 2. Dataset

### Recommended public dataset

Use the Kaggle **Stroke Prediction Dataset**:

- Source: Kaggle, `fedesoriano/stroke-prediction-dataset`
- Expected CSV name: `healthcare-dataset-stroke-data.csv`
- Expected path in this project:

```bash
./data/raw/healthcare-dataset-stroke-data.csv
```

The target variable is:

```text
stroke
```

Expected features include:

```text
gender, age, hypertension, heart_disease, ever_married, work_type,
Residence_type, avg_glucose_level, bmi, smoking_status
```

### Synthetic sample dataset

A small synthetic sample is included in:

```bash
./data/sample/synthetic_stroke_sample.csv
```

This file is not intended for meaningful model training. It exists only to validate the repository structure, run unit tests, and demonstrate the pipeline without external data.

## 3. Methodology

The pipeline includes:

1. Data loading and validation
2. Target/feature separation
3. Numeric preprocessing
   - median imputation
   - standardization
4. Categorical preprocessing
   - most frequent imputation
   - one-hot encoding
5. Class imbalance handling
   - random over-sampling on the training set only
6. Model comparison
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
7. Evaluation on a held-out test set

## 4. Evaluation metrics

Because stroke prediction is an imbalanced binary classification task, accuracy alone is not sufficient. This project reports:

- Accuracy
- Precision
- Recall / Sensitivity
- Specificity
- F1-score
- ROC-AUC
- Confusion matrix

For a screening-oriented model, recall/sensitivity is especially important because false negatives are clinically undesirable. However, this project is not a clinical model and does not define an operational threshold for medical use.

## 5. Project structure

```text
stroke-risk-prediction-ml/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── Makefile
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── sample/
│   └── README.md
│
├── notebooks/
│   └── README.md
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
│
├── models/
│   └── README.md
│
├── reports/
│   ├── figures/
│   └── model_card.md
│
├── app/
│   └── streamlit_app.py
│
└── tests/
    └── test_preprocessing.py
```

## 6. Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 7. How to train

With the public dataset placed in `data/raw/`:

```bash
python -m src.train --data data/raw/healthcare-dataset-stroke-data.csv
```

To run quickly with the included synthetic sample:

```bash
python -m src.train --data data/sample/synthetic_stroke_sample.csv
```

Outputs are saved to:

```text
models/best_model.joblib
reports/metrics.json
reports/model_comparison.csv
reports/figures/confusion_matrix.png
reports/figures/roc_curve.png
```

## 8. How to evaluate a saved model

```bash
python -m src.evaluate --data data/sample/synthetic_stroke_sample.csv --model models/best_model.joblib
```

## 9. How to run the Streamlit demo

After training a model:

```bash
streamlit run app/streamlit_app.py
```

## 10. Example prediction from CLI

```bash
python -m src.predict \
  --model models/best_model.joblib \
  --age 67 \
  --gender Female \
  --hypertension 1 \
  --heart_disease 0 \
  --ever_married Yes \
  --work_type Private \
  --residence_type Urban \
  --avg_glucose_level 180.5 \
  --bmi 31.2 \
  --smoking_status formerly_smoked
```

## 11. Limitations

- The project is based on public or synthetic tabular data, not private hospital data.
- The dataset may contain sampling bias and class imbalance.
- The model is not externally validated.
- No causal inference is performed.
- The prediction output must not be interpreted as a clinical diagnosis.

## 12. Future work

- Add calibration curves and threshold analysis.
- Add SHAP explanations.
- Add FastAPI deployment.
- Add GitHub Actions CI workflow.
- Add a Dockerfile.
- Add model monitoring examples.

## 13. Author

Portfolio project by an AI/Data Scientist specialized in medical AI, machine learning, and clinical decision-support systems.
