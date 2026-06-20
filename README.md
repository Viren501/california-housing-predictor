# California Housing Price Predictor

An end-to-end machine learning project designed to automate residential property valuation across California regions. This repository explores the complete model optimization workspace—demonstrating how data scaling, missing value imputation, and ensemble learning can systematically mitigate model underfitting and overfitting.

## Dataset Description
The model is trained on the classic California Housing dataset containing 20,640 records with regional real estate metrics:
* **Target Variable:** `median_house_value` (Continuous USD value).
* **Geographical & Spatial Features:** `longitude`, `latitude`.
* **Demographic Metrics:** `population`, `households`, `median_income`.
* **Structural Metrics:** `housing_median_age`, `total_rooms`, `total_bedrooms`, `ocean_proximity`.

## Methodology
1. **Data Exploration & Quality Check:** Analyzed column data types and identified 207 missing entries within the `total_bedrooms` field.
2. **Data Cleaning & Imputation:** Imputed missing data inside `total_bedrooms` using the column's statistical median to prevent modeling bias.
3. **Feature Engineering:** Applied `LabelEncoder` to translate qualitative nominal descriptions inside `ocean_proximity` into machine-readable continuous dimensions.
4. **Data Splitting:** Partitioned datasets into an 80% training slice and a 20% testing split with a locked random seed for consistent experimentation.
5. **Model Evaluation & Benchmarking:**
   * **Underfitted Baseline:** Controlled structure using a constrained `DecisionTreeRegressor` (with `max_depth=4`).
   * **Overfitted Tree:** Built an unconstrained `DecisionTreeRegressor` (with `max_depth=None`).
   * **Optimized Ensemble:** Deployed a 100-tree `RandomForestRegressor` ensemble to average out single-tree variance issues.

## Results & Evaluation
Evaluating model performance across variations via the R-squared ($R^2$) metric revealed distinct structural variances:

* **Underfit Structure (Constrained Tree):** Delivered a low $R^2$ train score of `0.6025` and a test score of `0.5761`, leaving a minor model-to-validation gap of `0.0264`.
* **Overfit Structure (Unconstrained Tree):** Memorized data splits completely to hit a train $R^2$ score of `1.0`, but dropped significantly to a test score of `0.6418` (producing a wide overfitting variance gap of `0.3581`).
* **Optimal Fit Structure (Random Forest Ensemble):** Stabilized extreme error margins cleanly to capture an excellent training score of `0.9746` and a generalized validation testing score of **`0.8075`**, lowering the test error gap to `0.1671`.

## How to Run Locally
1. Clone this repository to your computer workspace environment.
2. Install standard dependent libraries via pip:
   ```bash
   pip install numpy pandas matplotlib scikit-learn jupyter
