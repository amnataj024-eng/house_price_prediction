# House Price Prediction using Machine Learning – Assignment Report

## 1. Introduction
Predicting house prices is a classic regression problem. This report describes a complete machine learning workflow on the California Housing dataset: data cleaning, preprocessing, exploratory data analysis, training three regression models, and comparing them on a held-out test set. The notebook `House_Price_Prediction.ipynb` was executed end to end and all figures below come from that run.

## 2. Objectives
- Load and understand a public house price dataset.
- Clean and preprocess the data without data leakage.
- Explore the data using labelled visualisations.
- Train Linear Regression, Decision Tree and Random Forest regressors.
- Compare them with MAE, MSE, RMSE and R², and identify the best model.

## 3. Dataset Description
The California Housing dataset comes from the 1990 U.S. Census (Pace & Barry, 1997). Each row is a census block group. The CSV version from the *Hands-On Machine Learning* repository was used (20,640 rows, 10 columns).

| Feature | Description |
|---|---|
| longitude, latitude | Location in degrees |
| housing_median_age | Median age of houses (years) |
| total_rooms, total_bedrooms | Total rooms / bedrooms in the block group |
| population, households | People and households in the block group |
| median_income | Median income (tens of thousands of USD) |
| ocean_proximity | Categorical: <1H OCEAN, INLAND, NEAR OCEAN, NEAR BAY, ISLAND |
| **median_house_value** | **Target: median house value (USD)** |

## 4. Methodology
1. Load data, inspect shape, types and statistics.
2. Clean data (missing values, duplicates, types, outliers, inconsistencies).
3. EDA with histograms, boxplots, heatmap, scatter plots, map and category plot.
4. Split 80/20 with `random_state=42` before any learning from the data.
5. Preprocess with a `ColumnTransformer` inside `Pipeline`s.
6. Train three models on the same split, evaluate on the test set, analyse the best model.

## 5. Data Cleaning and Preprocessing
- **Missing values:** `total_bedrooms` has 207 missing values (1.0 %); imputed with the training-set median inside the pipeline.
- **Duplicates:** none found.
- **Data types:** correct; `ocean_proximity` is categorical and one-hot encoded.
- **Outliers:** the IQR rule flags about 3–6 % of rows in the size and income columns and the target; they are real observations and were kept.
- **Inconsistencies:** no non-positive counts and no block group with more bedrooms than rooms. The target is capped at 500,001 (965 rows) and `housing_median_age` at 52 (1,273 rows); not corrected, reported as a limitation.
- **Preprocessing:** numeric columns: median imputation + standardisation; categorical: most-frequent imputation + one-hot encoding (`handle_unknown="ignore"`). Training set: 16,512 rows; test set: 4,128 rows.

## 6. EDA Findings
- House values are right-skewed (skewness ≈ 0.98) with a spike at the 500,001 cap.
- `median_income` has the strongest correlation with price (≈ 0.69); other features are weakly correlated (absolute value below about 0.15).
- `total_rooms`, `total_bedrooms`, `population` and `households` are highly inter-correlated (0.86–0.98); longitude and latitude correlate at ≈ −0.92.
- Expensive areas cluster along the coast (Bay Area, Los Angeles–San Diego); `INLAND` block groups have a much lower median value (≈ 108,500 USD) than other categories (≈ 215,000–234,000 USD).

## 7. Models
- **Linear Regression:** baseline linear model.
- **Decision Tree Regressor:** unrestricted tree (`random_state=42`).
- **Random Forest Regressor:** 100 trees (`random_state=42`).

## 8. Results
| Model | MAE (USD) | MSE (USD²) | RMSE (USD) | R² (test) | R² (train) |
|---|---|---|---|---|---|
| Linear Regression | 50,670 | 4,908,290,571 | 70,059 | 0.6254 | 0.6497 |
| Decision Tree | 43,604 | 4,785,287,043 | 69,176 | 0.6348 | 1.0000 |
| Random Forest | 31,628 | 2,395,290,032 | 48,942 | 0.8172 | 0.9756 |

## 9. Discussion
Random Forest is the best model on all four test metrics: its RMSE is about 20,000 USD lower than the other two models and its R² is about 0.82. The Decision Tree fits the training data perfectly (R² = 1.0) but reaches only 0.63 on the test set, which is clear overfitting. Linear Regression has similar train and test R², so it underfits rather than overfits, probably because prices depend non-linearly on location and income. The Random Forest also shows a train–test gap (0.976 vs 0.817) but generalises much better. For the Random Forest, feature importance is led by `median_income` (≈ 49 %), then `ocean_proximity_INLAND` (≈ 14 %), `longitude` (≈ 11 %) and `latitude` (≈ 10 %). Residuals are centred near zero (mean ≈ −844 USD, standard deviation ≈ 48,940 USD); capped 500,001 values are often under-predicted, and errors grow for expensive areas.

## 10. Conclusion
The workflow produced a reproducible, leakage-free comparison of three regression models. Random Forest was the best-performing model (RMSE ≈ 48,942 USD, R² ≈ 0.817). Limitations: 1990 data at block-group level, capped target and house-age values, a single train/test split, no hyperparameter tuning or cross-validation, and a residual train–test gap for the Random Forest.

## 11. References
- scikit-learn, `fetch_california_housing`: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
- Pace, R. K. and Barry, R. (1997). Sparse Spatial Autoregressions. *Statistics & Probability Letters*, 33, 291–297.
- Dataset CSV copy: https://github.com/ageron/handson-ml2/tree/master/datasets/housing
- scikit-learn ColumnTransformer: https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
- scikit-learn Pipeline: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
- scikit-learn regression metrics: https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
- LinearRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
- DecisionTreeRegressor: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
- RandomForestRegressor: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
- pandas, Matplotlib, Seaborn documentation: https://pandas.pydata.org/docs/ · https://matplotlib.org/stable/ · https://seaborn.pydata.org/
