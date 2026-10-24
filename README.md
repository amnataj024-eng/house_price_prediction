# House Price Prediction (California Housing) – Machine Learning Assignment

## Overview
This project predicts the **median house value of California census block groups** using three regression algorithms and compares their performance. It covers data understanding, cleaning, preprocessing with pipelines (no data leakage), EDA with Matplotlib/Seaborn, model training, evaluation, residual analysis and a sample prediction.

## Dataset
- **Name:** California Housing (1990 U.S. Census; Pace & Barry, 1997).
- **Source used by the notebook:** CSV copy from the *Hands-On Machine Learning* repository: https://github.com/ageron/handson-ml2/tree/master/datasets/housing (loaded directly from `raw.githubusercontent.com`, no login or API key).
- **Fallback:** `sklearn.datasets.fetch_california_housing` (numeric features only), used automatically if the CSV cannot be downloaded.
- **Size:** 20,640 rows × 10 columns (9 features + target).
- **Target:** `median_house_value` (US dollars; capped at 500,001).
- **Features:** `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income` (tens of thousands of USD), `ocean_proximity` (categorical).

## Algorithms
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor (100 trees)

All use the same 80/20 train/test split (`random_state=42`) and the same preprocessing pipeline (median imputation + scaling for numeric columns; most-frequent imputation + one-hot encoding for the categorical column).

## Evaluation metrics
MAE, MSE, RMSE and R² on the test set (training R² is also shown to check overfitting).

## Results (from the executed notebook)
| Model | MAE (USD) | RMSE (USD) | R² (test) |
|---|---|---|---|
| Linear Regression | 50,670 | 70,059 | 0.6254 |
| Decision Tree | 43,604 | 69,176 | 0.6348 |
| Random Forest | 31,628 | 48,942 | 0.8172 |

Random Forest is the best model on every metric. Full tables, plots and discussion are in the notebook and `report.md`.

## How to run in Google Colab
1. Open https://colab.research.google.com and sign in.
2. **File → Upload notebook** → choose `House_Price_Prediction.ipynb`.
3. **Runtime → Run all**. No setup or API key is needed (all libraries are pre-installed in Colab).

## How to run locally
```bash
python -m venv venv
venv\Scripts\activate          # Windows  (use: source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
jupyter notebook House_Price_Prediction.ipynb
```

## Project structure
```
House_Price_Prediction/
├── House_Price_Prediction.ipynb
├── README.md
├── requirements.txt
└── report.md
```

## Limitations
1990 data at block-group level, capped target and house-age values, a single train/test split and no hyperparameter tuning.
