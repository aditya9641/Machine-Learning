# Customer Churn Prediction

A machine learning classification project that predicts whether a telecom customer is likely to churn.

## Overview
- Dataset: Telco Customer Churn
- Original shape: **7,043 × 21**
- Model: Random Forest Classifier
- Trees: 200
- Split: 80/20 with stratification
- Accuracy: **78.28%**
- Precision: **61.72%**
- Recall: **47.86%**
- F1: **53.92%**

## Workflow
1. Load and inspect data
2. Remove `customerID`
3. Convert `TotalCharges` to numeric
4. Fill missing `TotalCharges` with the median
5. Explore churn distribution
6. Engineer `AvgMonthlySpend` and `ChargePerMonth`
7. Encode categorical columns
8. Stratified train/test split
9. Train Random Forest
10. Evaluate predictions
11. Analyze feature importance

## Churn Distribution
- No Churn: 5,174
- Churn: 1,869

Because the target is imbalanced, accuracy should be interpreted together with precision, recall, and F1.

## Feature Engineering
```python
df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)
df["ChargePerMonth"] = df["TotalCharges"] / (df["MonthlyCharges"] + 1)
```

## Model
```python
RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
```

## Business Interpretation
Churn recall is **47.86%**, so the model misses a substantial portion of actual churners. For retention use cases, improving recall may be especially valuable.

A false negative means an actual churner was predicted as a non-churner.

## Feature Importance
Random Forest feature importance shows predictive usefulness within the fitted model. It does **not** establish causation.

## Technologies
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## Structure
```text
03_Customer_Churn_Prediction/
├── customer_churn_prediction.ipynb
├── README.md
└── DOCUMENTATION.md
```

## Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook
```

## Limitations & Future Work
The notebook uses direct `LabelEncoder` processing before the train/test split and does not include hyperparameter tuning or cross-validation. A production version should use a train/test-safe preprocessing pipeline, compare encoding strategies, tune the model, address class imbalance, optimize the decision threshold, and perform detailed error analysis.

## Author
**Aditya Roy**
