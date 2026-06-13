# Model Card — Rossmann SalesForecaster v3.1

## Model Summary

| Field | Value |
|-------|-------|
| Model name | SalesForecaster v3.1 |
| Type | Regression (ensemble) |
| Algorithm | Random Forest (primary) + LightGBM (blend weight 0.25) |
| Target variable | `Sales` (daily store revenue in EUR) |
| Transformation | log1p(Sales) for training; expm1 for inference |
| Primary metric | RMSPE (Root Mean Square Percentage Error) |
| Validation RMSPE | **0.118** |
| Training date | 2023-09-15 |
| Deployed | 2023-10-05 |
| Stores covered | 1,115 across 7 European countries |
| Prediction horizon | Up to 6 weeks ahead (daily granularity) |

---

## Feature Importance (SHAP, validation set)

| Rank | Feature | Mean |SHAP| | Notes |
|------|---------|-------------|-------|
| 1 | `Customers` | 0.312 | Strongest signal; store traffic proxy |
| 2 | `Promo` | 0.284 | Binary; 1 = store running day promo |
| 3 | `DayOfWeek` | 0.198 | Sun (7) and Mon (1) behave differently |
| 4 | `CompetitionDistance` | 0.167 | Log-transformed; nearer = more impact |
| 5 | `StoreType` | 0.143 | Type B highest avg sales |
| 6 | `Assortment` | 0.118 | Assortment B only at Type B stores |
| 7 | `StateHoliday` | 0.109 | Public holidays → near-zero sales |
| 8 | `SchoolHoliday` | 0.087 | Positive impact (families shop) |
| 9 | `month` | 0.074 | Dec peak; Jan trough |
| 10 | `Promo2` | 0.062 | Long-running secondary promo |
| 11 | `competition_open` | 0.058 | Months since competitor opened |
| 12 | `Open` | 0.041 | Binary; 0 suppresses prediction entirely |
| 13 | `IsPromo2Month` | 0.037 | Whether current month is in PromoInterval |
| 14 | `promo_2_open` | 0.029 | Weeks since Promo2 started |
| 15 | `year` | 0.021 | Trend component |
| Others | `week`, `day`, etc. | < 0.02 | Minor temporal signals |

---

## Training Data Statistics (Baseline Reference)

| Feature | Type | Baseline Mean / Mode | Baseline Std / Freq |
|---------|------|----------------------|---------------------|
| Promo | int | 0.41 | — |
| CompetitionDistance | float | 5,404 m | 7,663 m |
| Open | int | 0.83 | — |
| StateHoliday | cat | '0' = 92% | 'a'=4%, 'b'=2%, 'c'=2% |
| SchoolHoliday | int | 0.18 | — |
| StoreType | cat | 'a' = 55% | 'b'=10%, 'c'=25%, 'd'=10% |
| DayOfWeek | int | 3.97 | 1.97 |
| Customers | int | 762 | 610 |

---

## Known Model Weaknesses

1. **Promo saturation**: Model was trained when Promo=1 on ~41% of rows. If promo
   rate exceeds ~55%, SHAP importance of Promo collapses (model can no longer
   differentiate promo vs non-promo signal) → RMSPE degrades on non-promo stores.

2. **Competition distance OOD**: Training CompetitionDistance median was ~5,400 m
   (Kaggle dataset). Real Rossmann internal data shows urban stores with much
   shorter distances (~820 m median). Values below 200 m are rare in training;
   the model extrapolates poorly in this regime.

3. **Holiday suppression**: StateHoliday = 'a' rows have near-zero sales.
   If this code is corrupted to NaN, the model predicts normal sales on public
   holidays → massive over-prediction on those dates.

4. **Open=0 not filtered**: If the ETL filter (step 7) fails, closed-store rows
   reach the prediction service. Open=0 rows should always be suppressed before
   scoring; the model was never trained on them.

5. **No online learning**: The model is a static snapshot. It does not adapt to
   drift automatically. Manual retraining is required when RMSPE SLA (0.14) is
   breached and root cause is confirmed as concept drift.

---

## Acceptable RMSPE Thresholds

| Segment | RMSPE SLA | Notes |
|---------|-----------|-------|
| All stores | 0.14 | Hard operational SLA |
| Type B stores | 0.17 | Higher variance; wider SLA |
| Holiday weeks | 0.22 | Holiday edge cases tolerated |
| Promo weeks | 0.16 | Promo variance expected |

---

## Retraining Protocol

- **Trigger:** RMSPE SLA breach sustained for > 3 consecutive days.
- **Data window:** Rolling 2 years ending at retrain date.
- **Pipeline:** `rossmann-mlops/retrain` Airflow DAG; ~6 h end-to-end.
- **Validation gate:** New model must achieve RMSPE < 0.14 on holdout before promotion.
- **Approver:** ML Ops Lead (Haritha Nimmagadda).
