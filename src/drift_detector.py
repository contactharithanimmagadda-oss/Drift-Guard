"""
src/drift_detector.py — PSI, KS test, SHAP delta, run_drift_scan
Person A — Day 2
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import GradientBoostingClassifier
import shap
import warnings
from datetime import datetime
from src.contracts import DriftReport, FeatureDrift

warnings.filterwarnings("ignore")


def _is_numeric(arr: np.ndarray) -> bool:
    return np.issubdtype(arr.dtype, np.number)


def compute_psi(baseline: np.ndarray, production: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index — numeric (histogram + NaN bin) or categorical (frequency).
    PSI < 0.10 → ok | 0.10–0.20 → warn | > 0.20 → critical
    NaN values are treated as an extra bin for numeric columns so ETL corruption is visible.
    """
    eps = 1e-6
    if _is_numeric(baseline) and _is_numeric(production):
        base_nan  = np.isnan(baseline)
        prod_nan  = np.isnan(production)
        base_v    = baseline[~base_nan]
        prod_v    = production[~prod_nan]
        if len(base_v) == 0:
            return 1.0
        b_hist, edges = np.histogram(base_v, bins=bins, density=False)
        p_hist, _     = np.histogram(prod_v if len(prod_v) > 0 else base_v[:1], bins=edges, density=False)
        # Append NaN as an extra bin
        b_counts = np.append(b_hist, base_nan.sum())
        p_counts = np.append(p_hist, prod_nan.sum())
        n_bins   = bins + 1
        b_pct = (b_counts + eps) / (len(baseline) + eps * n_bins)
        p_pct = (p_counts + eps) / (len(production) + eps * n_bins)
    else:
        # Categorical: use per-category frequency proportions
        categories = list(set(baseline) | set(production))
        b_counts   = {c: (baseline == c).sum() for c in categories}
        p_counts   = {c: (production == c).sum() for c in categories}
        b_pct = np.array([(b_counts[c] + eps) / (len(baseline) + eps) for c in categories])
        p_pct = np.array([(p_counts[c] + eps) / (len(production) + eps) for c in categories])
    return round(float(np.sum((p_pct - b_pct) * np.log(p_pct / b_pct))), 6)


def compute_ks(baseline: np.ndarray, production: np.ndarray) -> tuple:
    """KS test for numeric columns; returns (0.0, 1.0) for categorical."""
    if not (_is_numeric(baseline) and _is_numeric(production)):
        return 0.0, 1.0
    ks, p = ks_2samp(baseline, production)
    return round(float(ks), 6), round(float(p), 8)


def classify_psi(psi: float) -> str:
    if psi > 0.20: return "critical"
    if psi > 0.10: return "warn"
    return "ok"


def compute_shap_delta(baseline_df, prod_df, feature_cols, label_col) -> dict:
    """% change in mean |SHAP| per feature. Numeric-only; categorical cols get 0.0."""
    from sklearn.ensemble import GradientBoostingRegressor
    numeric_cols = [c for c in feature_cols
                    if pd.api.types.is_numeric_dtype(baseline_df[c])]
    if not numeric_cols or label_col not in baseline_df.columns:
        return {}
    base_clean = baseline_df[numeric_cols + [label_col]].dropna()
    if len(base_clean) < 20:
        return {}
    target = base_clean[label_col]
    use_regressor = pd.api.types.is_float_dtype(target) or target.nunique() > 20
    if use_regressor:
        model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
    else:
        model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    model.fit(base_clean[numeric_cols], target)
    explainer   = shap.TreeExplainer(model)
    base_filled = baseline_df[numeric_cols].fillna(baseline_df[numeric_cols].median())
    prod_filled = prod_df[numeric_cols].fillna(prod_df[numeric_cols].median())

    def _mean_abs(sv):
        if isinstance(sv, list):
            return np.mean([np.abs(c).mean(axis=0) for c in sv], axis=0)
        return np.abs(sv).mean(axis=0)

    base_shap = _mean_abs(explainer.shap_values(base_filled))
    prod_shap = _mean_abs(explainer.shap_values(prod_filled))
    return {feat: round((prod_shap[i]-base_shap[i])/(base_shap[i]+1e-9)*100, 2)
            for i, feat in enumerate(numeric_cols)}


def run_drift_scan(baseline_df, prod_df, feature_cols, label_col,
                   scenario_id, model_accuracy_baseline=0.0,
                   model_accuracy_now=0.0) -> DriftReport:
    """
    Full drift scan — PSI + KS + SHAP delta for every feature.
    Returns DriftReport with FeatureDrift per feature.
    """
    print(f"[drift_scan] {scenario_id} | baseline={len(baseline_df)} | prod={len(prod_df)}")
    numeric_cols = [c for c in feature_cols
                    if pd.api.types.is_numeric_dtype(baseline_df[c])
                    and baseline_df[c].notna().sum() > 10
                    and prod_df[c].notna().sum() > 10]
    shap_deltas = {}
    if numeric_cols:
        try:
            shap_deltas = compute_shap_delta(baseline_df, prod_df, numeric_cols, label_col)
        except Exception as e:
            print(f"  [SHAP warning] {e}")

    features = []
    for col in feature_cols:
        # PSI uses raw values (NaN treated as extra bin); KS/stats use dropna values
        base_raw  = baseline_df[col].values
        prod_raw  = prod_df[col].values
        base_vals = baseline_df[col].dropna().values
        prod_vals = prod_df[col].dropna().values
        is_num    = _is_numeric(base_vals) and len(base_vals) > 0

        if prod_df[col].isna().all():
            psi, ks, p_val, status = 1.0, 1.0, 0.0, "critical"
            b_mean = round(float(np.mean(base_vals)), 4) if is_num and len(base_vals) else 0.0
            p_mean = float("nan")
            b_std  = round(float(np.std(base_vals)), 4)  if is_num and len(base_vals) else 0.0
            p_std  = float("nan")
        else:
            psi       = compute_psi(base_raw, prod_raw)
            ks, p_val = compute_ks(base_vals, prod_vals)
            status    = classify_psi(psi)
            if is_num:
                b_mean = round(float(np.mean(base_vals)), 4)
                p_mean = round(float(np.mean(prod_vals)), 4)
                b_std  = round(float(np.std(base_vals)),  4)
                p_std  = round(float(np.std(prod_vals)),  4)
            else:
                # Categorical: store category counts as a proxy for "mean"
                b_mean = round(len(set(base_vals)), 4)   # unique category count
                p_mean = round(len(set(prod_vals)), 4)
                b_std  = 0.0
                p_std  = 0.0
        features.append(FeatureDrift(col, psi, ks, p_val,
                                     shap_deltas.get(col, 0.0), status,
                                     b_mean, p_mean, b_std, p_std))
        flag = "!" if status == "critical" else "~" if status == "warn" else " "
        print(f"  {flag} {col:<28} PSI={psi:.4f} KS={ks:.4f} [{status}]")

    report = DriftReport(scenario_id, model_accuracy_now, model_accuracy_baseline,
                         features, datetime.now().isoformat(),
                         len(baseline_df), len(prod_df))
    print(f"[drift_scan] Done — {report.summary()}\n")
    return report
