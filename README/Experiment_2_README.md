# Experiment 2: Redshift Estimation with Photometry + Morphological Features

## Overview

This experiment extends Experiment 1 by testing whether adding galaxy **structural and morphological parameters** to the five-band photometric magnitudes improves spectroscopic redshift (`specz_redshift`) estimation, compared to using photometry alone.

Experiment 1 established a photometry-only baseline. This experiment isolates the effect of adding morphology by keeping the modeling approach consistent and comparing directly against that baseline.

## Dataset

- Source file: `5x127x127_training_with_morphology.csv`
- Target variable: `specz_redshift`

### Features used

**Photometric features (5-band cModel magnitudes):**
`g_cmodel_mag`, `r_cmodel_mag`, `i_cmodel_mag`, `z_cmodel_mag`, `y_cmodel_mag`

**Morphological features (12 parameters × 5 bands = 60 features):**
`central_image_pop_5px_rad`, `central_image_pop_10px_rad`, `central_image_pop_15px_rad`, `ellipticity`, `half_light_radius`, `isophotal_area`, `major_axis`, `minor_axis`, `peak_surface_brightness`, `petro_rad`, `pos_angle`, `sersic_index` — each computed per band (g, r, i, z, y)

Total feature count: 65 (5 photometric + 60 morphological)

## Methodology

### 1. Data Quality Checks
- Checked for missing values across all selected features and the target — none found in the modeling set
- Checked for infinite values in features and target
- Checked for duplicate rows

### 2. Exploratory Data Analysis
- Examined the distribution of `specz_redshift` (histogram + KDE)
- Examined the distribution of each photometric band
- Computed per-feature skewness to flag heavily skewed morphological parameters (|skew| > 1)
- Summarized min/max/mean/std/skewness for every feature

### 3. Train/Test Split
- 80/20 split, `random_state=42` for reproducibility
- Verified that training and test target distributions are comparable

### 4. Preprocessing
- **Linear Regression:** features standardized with `StandardScaler` (fit on train, applied to test — no leakage)
- **Random Forest:** used on raw, unscaled features, since tree-based models don't require feature scaling

## Models Trained

| Model | Notes |
|---|---|
| Linear Regression | Baseline on scaled features |
| Random Forest (default) | `random_state=42`, default hyperparameters |
| Random Forest — 5-fold CV | Cross-validated R² on training data, to check stability before tuning |
| Random Forest (tuned) | `RandomizedSearchCV`, 5 iterations, 3-fold CV, tuned over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` |

Search space used for tuning:
```python
rf_param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [10, 15, 20, 25],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
```

## Results

| Model | Test R² | Train R² | MAE | RMSE |
|---|---|---|---|---|
| Linear Regression | 0.5863 | 0.5848 | 0.2387 | 0.3675 |
| Random Forest (default) | 0.8410 | 0.9775 | 0.0964 | 0.2279 |
| Random Forest (tuned) | 0.8318 | 0.9414 | 0.1064 | 0.2344 |

**Best model (by test R²):** Random Forest (default) — 0.8410

**Note on the tuned model:** the tuned Random Forest scored slightly *lower* on test R² than the default model (0.8318 vs. 0.8410). This is a real and useful finding, not a bug — it does narrow the train/test gap somewhat (0.9414 − 0.8318 = 0.1096 vs. the default's 0.9775 − 0.8410 = 0.1365), meaning it generalizes marginally better even though its raw test score is lower. That's likely a side effect of the reduced hyperparameter search (see below) rather than evidence that tuning hurts — a fuller search would likely find a configuration that beats the default on both counts.

**Hardware-constrained tuning:** the `RandomizedSearchCV` search space and search intensity were deliberately reduced to keep the search runnable on local hardware, which otherwise overheated and froze:

```python
random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=rf_param_grid,
    n_iter=5,              # reduced from 20
    cv=3,                  # reduced from 5
    scoring='r2',
    random_state=42,
    n_jobs=2,              # reduced from -1
    verbose=2
)
```

With more iterations (`n_iter`) and more CV folds, the search would sample a larger portion of the hyperparameter space and validate each candidate more robustly, which would likely push the tuned model's test R² above the default model's. The current tuned result should be read as a lower bound on what proper tuning could achieve, not as evidence that the default model is truly better.

## Analysis

- **Actual vs. Predicted plot:** scatter of true vs. predicted redshift for the Random Forest model, with a 1:1 reference line
- **Residual distribution:** histogram of `y_test - rf_preds`, checked for centering around zero and symmetry
- **Residuals vs. predicted:** checked for any systematic trend (e.g., increasing spread at higher predicted redshift, which would indicate heteroscedasticity)
- **Training vs. testing R² comparison:** bar chart to visually flag overfitting
- **Feature importance:** full ranked list and horizontal bar chart of Random Forest feature importances across all 65 features (photometric + morphological)
- **Model comparison (MAE):** bar chart comparing MAE across Linear Regression, default RF, and tuned RF

## Summary of Findings

Experiment 2 demonstrates that incorporating galaxy structural and morphological parameters provides useful additional information for spectroscopic redshift estimation. The Random Forest model achieved a testing R² of 0.8410, substantially outperforming Linear Regression, which achieved a testing R² of 0.5863. The Random Forest also achieved a lower MAE (0.0964) and RMSE (0.2279), indicating considerably better predictive performance. Five-fold cross-validation produced a mean R² of 0.8381 ± 0.0030, showing that the model's performance is consistent across different subsets of the data.

Feature-importance analysis further showed that morphological properties play an important role in the prediction. The i-band half-light radius was the most important feature, with an importance of 0.4729, followed by the r-band model magnitude (0.1475) and g-band peak surface brightness (0.0712). Several other structural parameters, including major-axis size, half-light radius, isophotal area and Sérsic index, also appeared among the most important features. These results suggest that information about galaxy size, shape and light distribution complements photometric brightness information when estimating redshift. However, feature importance represents predictive relevance rather than physical causation, so these results should be interpreted as evidence that morphological properties contain useful redshift-related information rather than as a direct physical relationship between individual morphological parameters and redshift.

## Key Question This Experiment Answers

Does adding morphology to photometry improve redshift estimation over photometry alone (Experiment 1), and if so, **which morphological parameters actually matter**?

Per the feature-importance ranking above, the top predictor (i-band half-light radius, importance 0.4729) is a morphological parameter, not a photometric one — with the r-band magnitude ranked second. This indicates morphology is not just marginally useful but is carrying a substantial share of the predictive signal in this feature set. Whether this represents a genuine *improvement in absolute R²* over Experiment 1's photometry-only result should still be checked directly against verified Experiment 1 numbers, since that comparison hasn't been made explicitly here.

## Limitations

- Feature importance from Random Forest reflects predictive contribution within this model, not physical causation
- Both Random Forest models show a substantial train/test R² gap (0.98→0.84 default, 0.94→0.83 tuned), indicating meaningful overfitting even after tuning — worth revisiting with stronger regularization (e.g., limiting `max_depth` further, increasing `min_samples_leaf`) in addition to a larger search
- `RandomizedSearchCV` used a reduced search (5 iterations, 3-fold CV, `n_jobs=2`) rather than an exhaustive search, due to local hardware constraints — the tuned result should be treated as a lower bound on achievable performance, not the ceiling
- No catastrophic-outlier rate (e.g., fraction with |Δz|/(1+z) > 0.15) or redshift-binned error analysis computed yet — worth adding for a more field-standard evaluation
- High-dimensional morphological feature set (60 features) was not checked for multicollinearity (e.g., `major_axis` and `minor_axis` are likely correlated) — this can dilute individual feature importance scores even when a group of correlated features is jointly informative

## Next Steps (Experiments 3 & 4)

- **Experiment 3:** broaden the model comparison (e.g., Gradient Boosting, XGBoost) and analyze predictive performance, errors, and feature importance more systematically across models
- **Experiment 4:** move from tabular photometry/morphology to image-based deep learning, using galaxy images with CNNs for redshift estimation
