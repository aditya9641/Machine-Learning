# Customer Churn Prediction — Documentation

## Objective
Predict whether a telecom customer will churn using customer and service attributes.

## Data Cleaning
`customerID` is removed because it is an identifier.

`TotalCharges` is converted from object/string to numeric:
```python
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)
```
Missing values are filled with the median.

## Class Distribution
- No Churn: 5,174
- Churn: 1,869

This is an imbalanced binary classification problem.

## Feature Engineering
Two derived features are created:
```python
AvgMonthlySpend = TotalCharges / (tenure + 1)
ChargePerMonth = TotalCharges / (MonthlyCharges + 1)
```

## Encoding
The original notebook uses `LabelEncoder` for object-type columns:
```python
for column in df.columns:
    if df[column].dtype == "object":
        df[column] = encoder.fit_transform(df[column])
```

This documentation preserves the actual implementation.

## Train/Test Split
The notebook uses:
```python
train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

Training shape: **5,634 × 21**  
Testing shape: **1,409 × 21**

## Random Forest
```python
RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
```

## Results
```text
Accuracy  = 78.28%
Precision = 61.72%
Recall    = 47.86%
F1 Score  = 53.92%
```

## Why Recall Matters
In a retention scenario, a false negative is an actual churner that the model fails to identify. Therefore, recall can be more important than accuracy alone.

## Feature Importance
`model.feature_importances_` is used to rank features. These values describe model-level predictive importance and should not be interpreted as causal effects.

## Important Implementation Notes
The original notebook:
- uses Random Forest;
- uses LabelEncoder;
- performs encoding before the split;
- does not use hyperparameter tuning;
- does not use advanced class-imbalance techniques.

## Production Improvements
Use `Pipeline`/`ColumnTransformer`, fit preprocessing only on training data, compare encoding methods, use cross-validation and hyperparameter tuning, optimize thresholds, evaluate precision-recall tradeoffs, and add monitoring/deployment.
