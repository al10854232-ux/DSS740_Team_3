# DSS740 Team 3 — Student Dropout Prediction Project

## Project Overview

This project develops an end-to-end machine learning pipeline to predict student dropout risk using academic, financial, demographic, and macroeconomic features.

The goal is to identify at-risk students early enough for institutions to intervene with targeted academic and financial support.

Dataset:
Student Dropout and Academic Success (Realinho et al., 2022)
UCI Machine Learning Repository
4,424 records | 36 features

---

## Business Objective

Student dropout creates measurable financial and social costs:

* Lost tuition revenue
* Institutional resource waste
* Lifetime earnings loss for students

This project builds a predictive model that:

* Flags high-risk students early
* Supports data-driven advisor intervention
* Quantifies economic return on investment (ROI)

Final model performance:

* ROC-AUC ≈ 0.936
* Recall ≈ 81% (at threshold = 0.40)
* Strong precision-recall balance for screening

---

## Repository Structure

### G3_Dropout_Notebook_v3.ipynb

Full analysis including:

* Data preprocessing
* Outlier treatment (Winsorization)
* Feature engineering
* Class imbalance handling
* Model training (5 algorithms)
* Hyperparameter tuning (GridSearchCV)
* Model evaluation and comparison
* Feature importance analysis
* Partial Dependence Plots
* Threshold analysis
* Economic impact analysis
* Calibration analysis
* Learning curves
* Permutation importance
* Kaplan-Meier survival analysis
* Bootstrap confidence intervals

---

### G3_dropout_predictor_v3.py

Object-oriented machine learning pipeline class.

Encapsulates:

1. Data loading
2. Preprocessing
3. Feature engineering
4. Train/test split and oversampling
5. Scaling
6. Base model training
7. Hyperparameter tuning
8. Model comparison
9. Threshold optimization
10. Final evaluation
11. Economic impact analysis
12. Calibration analysis
13. Learning curves
14. Permutation importance
15. Survival analysis
16. Bootstrap confidence intervals
17. Microeconomic analysis

Example usage:

```python
from G3_dropout_predictor_v3 import DropoutPredictor

predictor = DropoutPredictor(filepath="students_dropout_academic_success.csv")
predictor.run_full_pipeline()
```

---

### G3_Technical_Report_v3.docx

Formal technical report including:

* Problem statement and business case
* Data preprocessing rationale
* Feature engineering justification
* Model evaluation results
* Advanced ML validation
* Economic impact quantification
* Institutional deployment recommendations

---

## Tools and Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

Advanced techniques used:

* GridSearchCV (StratifiedKFold)
* Gradient Boosting
* Permutation Importance
* Calibration (Brier Score, Reliability Diagram)
* Kaplan-Meier Survival Analysis
* Bootstrap Confidence Intervals
* ROI Quantification Modeling

---

## Model Selection

Algorithms evaluated:

* Logistic Regression
* Naive Bayes
* Decision Tree
* Random Forest
* Gradient Boosting

Final Model Selected:
Gradient Boosting (Tuned)

Selection criteria:

* Primary metric: ROC-AUC
* Secondary metric: F1-Score
* Deployment threshold optimized for recall (0.40)

---

## Key Insights

* Second-semester approval rate is the strongest predictor
* Tuition status is the most important non-academic factor
* Scholarship holders show significantly lower dropout risk
* Macroeconomic variables have minimal predictive impact
* Dropout hazard more than doubles after semester one

---

## Economic Impact

Estimated per-dropout cost includes:

* Institutional tuition loss
* Institutional expenditure waste
* Student lifetime earnings gap

Projected impact:

* 20% intervention success rate
* Estimated ROI ≈ 160x
* Net institutional and societal value preserved in the tens of millions of EUR

---

## How to Run

### Option 1 — Jupyter Notebook

Open:
G3_Dropout_Notebook_v3.ipynb

Run cells sequentially.

### Option 2 — Python Pipeline

```bash
python G3_dropout_predictor_v3.py
```

Or inside Python:

```python
from G3_dropout_predictor_v3 import DropoutPredictor
predictor = DropoutPredictor(filepath="students_dropout_academic_success.csv")
predictor.run_full_pipeline()
```

---

## Team Members

Kristen Kohler — [kk10891066@sju.edu](mailto:kk10891066@sju.edu)
Marith Bijkerk — [mb10801662@sju.edu](mailto:mb10801662@sju.edu)
Chenxi Lai — [cl10804385@sju.edu](mailto:cl10804385@sju.edu)
Ankita Lala — [al10854232@sju.edu](mailto:al10854232@sju.edu)

Saint Joseph’s University
Machine Learning — Spring 2026

---
