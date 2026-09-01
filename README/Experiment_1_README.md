# Experiment 1: Redshift Estimation from Photometry Alone

## Overview

This experiment establishes a baseline for spectroscopic redshift (`specz_redshift`) estimation using **only the five-band photometric magnitudes** — no structural or morphological information. It answers a simple question: how much redshift information is contained in brightness alone, and which traditional ML model handles that information best? This baseline is the reference point Experiment 2 (photometry + morphology) is compared against.

## Dataset

- Source file: `5x127x127_training_with_morphology.csv`
- Target variable: `specz_redshift`
- Features: `g_cmodel_mag`, `r_cmodel_mag`, `i_cmodel_mag`, `z_cmodel_mag`, `y_cmodel_mag` (5 features total)

## Methodology

### 1. Data Quality Checks
- Checked for missing values across the five photometric features and the target — none found
- Checked for infinite values in features and target
- Checked for duplicate rows

### 2. Exploratory Data Analysis
- Examined the distribution of `specz_redshift` (histogram + KDE)
- Examined the distribution of each photometric band
- Computed the correlation matrix between the five photometric bands (heatmap)

### 3. Train/Test Split
- 80/20 split, `random_state=42` for reproducibility
- Verified training and test target distributions are comparable

### 4. Preprocessing
- **Linear Regression:** features standardized with `StandardScaler` (fit on train, applied to test — no leakage)
- **Random Forest / Gradient Boosting:** used on raw, unscaled features, since tree-based models don't require feature scaling

## Models Trained

| Model | Notes |
|---|---|
| Linear Regression | Baseline on scaled features |
| Random Forest (default) | `random_state=42`, default hyperparameters |
| Random Forest (tuned) | `RandomizedSearchCV`, 10 iterations, 5-fold CV |
| Gradient Boosting (default) | `n_estimators=100, learning_rate=0.1, max_depth=3` |
| Gradient Boosting (tuned) | `RandomizedSearchCV`, 10 iterations, 5-fold CV |

## Results

| Model | Test R² | Train R² | MAE | RMSE |
|---|---|---|---|---|
| Linear Regression | 0.3429 | 0.3419 | 0.2892 | 0.4632 |
| Random Forest (default) | 0.7310 | 0.9611 | 0.1185 | 0.2964 |
| Random Forest (tuned) | 0.7356 | 0.9368 | 0.1188 | 0.2938 |
| Gradient Boosting (default) | 0.6153 | 0.6157 | 0.1767 | 0.3545 |
| Gradient Boosting (tuned) | 0.7061 | 0.7654 | 0.1347 | 0.3098 |

**Best model (by test R²):** Random Forest (tuned) — 0.7356

## Summary of Findings

Photometry alone gives Linear Regression very little to work with — R² of 0.343 confirms that redshift's relationship to five-band brightness is far from linear. Both ensemble methods do substantially better, with Random Forest (tuned) reaching the highest test R² (0.7356) and lowest RMSE (0.2938) of all five models.

The more informative comparison here isn't which model wins on raw R² — it's **how each model generalizes**. Random Forest shows a large train/test gap even after tuning (Train R² 0.9368 vs. Test R² 0.7356, a gap of ~0.20), meaning it's fitting substantial noise in the training data. Gradient Boosting (tuned), by contrast, has a much smaller gap (0.7654 vs. 0.7061, a gap of ~0.06) while landing only 3 points of R² behind Random Forest. In other words: Random Forest wins on raw accuracy, but Gradient Boosting generalizes more honestly. For a photo-z pipeline meant to hold up on new data, that trade-off is worth weighing — not just the leaderboard number.

Feature importance from the tuned Random Forest is notably even across all five bands (0.149–0.253), unlike Experiment 2's morphology-inclusive result where one feature dominated. The g-band (0.2533) and y-band (0.2439) — the bluest and reddest bands — carry slightly more weight than the three middle bands, suggesting the widest color baseline is the most informative single signal for redshift when only photometry is available.

## Comparison to Experiment 2 (Photometry + Morphology)

| Model | Exp 1 (photometry only) Test R² | Exp 2 (+ morphology) Test R² | Change |
|---|---|---|---|
| Random Forest (default) | 0.7310 | 0.8410 | +0.1100 |
| Random Forest (tuned) | 0.7356 | 0.8318 | +0.0962 |

Adding morphological and structural parameters improved test R² by roughly 0.10 for both Random Forest variants — a real, non-trivial gain, and now backed by verified numbers on both sides rather than an assumed comparison. This confirms the hypothesis behind Experiment 2: galaxy structure genuinely complements photometric brightness for redshift estimation, it isn't just adding noise dressed up as extra features.

## Analysis

- **Actual vs. Predicted plot:** scatter of true vs. predicted redshift for the tuned Random Forest, with a 1:1 reference line
- **Residual distribution:** histogram of `y_test - rf_tuned_preds`, checked for centering around zero
- **Feature importance:** ranked bar chart across all five photometric bands
- **Model comparison (MAE):** bar chart comparing MAE across all five models

## Limitations

- Both Random Forest models show meaningful overfitting (see train/test gap above) even after tuning — worth revisiting with stronger regularization (e.g., limiting `max_depth`, increasing `min_samples_leaf`) rather than just a wider hyperparameter search
- No catastrophic-outlier rate (e.g., fraction with |Δz|/(1+z) > 0.15) or redshift-binned error analysis computed yet — worth adding for a more field-standard evaluation, especially to see whether Random Forest's edge over Gradient Boosting holds across all redshift ranges or is concentrated in one region
- Five-band photometry alone is a genuinely weak signal for redshift (as the Linear Regression result shows) — the strong ensemble results depend on the models capturing non-linear structure, so interpretability of *why* a given prediction was made is limited without further diagnostics like SHAP values

## Next Steps (Experiments 2, 3 & 4)

- **Experiment 2:** incorporate galaxy structural and morphological parameters (size, ellipticity, surface brightness, Sérsic index) alongside photometry, to test whether structural information improves on this photometry-only baseline
- **Experiment 3:** broaden the model comparison and analyze predictive performance, errors, and feature importance more systematically — including addressing the overfitting gap identified here
- **Experiment 4:** move from tabular photometry/morphology to image-based deep learning, using galaxy images with CNNs for redshift estimation
