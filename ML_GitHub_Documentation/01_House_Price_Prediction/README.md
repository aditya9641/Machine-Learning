# House Price Prediction

A machine learning regression project that predicts median house values using the California Housing dataset.

## Overview
- Dataset: California Housing
- Rows: 20,640
- Target: `MedHouseVal`
- Model: Linear Regression
- Split: 80/20, `random_state=42`
- Metrics: MAE, RMSE, R²
- Model persistence: Joblib

## Features
`MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`

## Workflow
1. Load and inspect data
2. Exploratory data analysis
3. Missing-value and correlation checks
4. Train/test split
5. Linear Regression training
6. Prediction and evaluation
7. Save/load model with Joblib

## Example
One recorded prediction was **4.1519** versus an actual value of **4.526**.

## Technologies
Python, Pandas, Matplotlib, Seaborn, Scikit-learn, Joblib

## Structure
```text
01_House_Price_Prediction/
├── house_price_prediction.ipynb
├── README.md
└── DOCUMENTATION.md
```

## Run
```bash
pip install pandas matplotlib seaborn scikit-learn joblib
jupyter notebook
```

## Limitations & Future Work
This is a portfolio baseline. The notebook does not use scaling, advanced feature engineering, hyperparameter tuning, or a production preprocessing pipeline. Future work could compare tree-based models, use cross-validation, improve feature engineering, and deploy the model.

## Author
**Aditya Roy**
