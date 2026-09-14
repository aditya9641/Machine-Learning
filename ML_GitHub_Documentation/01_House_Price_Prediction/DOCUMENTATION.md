# House Price Prediction — Documentation

## Objective
Predict the continuous `MedHouseVal` target from California Housing features.

## Data
The dataset contains 20,640 rows and 9 columns: eight predictors and the target.

## EDA
The notebook examines shape, columns, data types, descriptive statistics, missing values, and feature correlations. A correlation heatmap is used for visual analysis.

## Modeling
`LinearRegression()` is trained after an 80/20 train/test split using `random_state=42`.

```python
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## Evaluation
The notebook evaluates the regression using MAE, RMSE, and R².

## Persistence
The trained model is saved with Joblib for later reuse.

## Implementation Accuracy
The original notebook does **not** use feature scaling, encoding, imputation pipelines, or advanced feature engineering. This documentation intentionally does not claim otherwise.

## Production Improvements
A production version could add robust preprocessing, cross-validation, model comparison, monitoring, validation, and deployment.
