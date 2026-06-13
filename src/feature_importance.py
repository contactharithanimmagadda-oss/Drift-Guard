"""
src/feature_importance.py -- Full SHAP feature importance module
Person A -- Day 2
Four functions: global, ranking string, local explanation, permutation fallback.
"""
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance as sk_perm
import warnings
from src.contracts import FeatureImportanceRecord, FeatureImportanceReport

warnings.filterwarnings("ignore")


def global_importance(baseline_df, prod_df, feature_cols, label_col,
                      scenario_id) -> FeatureImportanceReport:
    """
    Trains GBM on baseline, computes mean |SHAP| on BOTH windows.
    Supports both classifier (binary target) and regressor (continuous/high-cardinality target).
    Records sorted by abs(delta_pct) descending -- biggest collapse first.
    """
    base_filled = baseline_df[feature_cols].fillna(baseline_df[feature_cols].median())
    prod_filled = prod_df[feature_cols].fillna(prod_df[feature_cols].median())
    base_clean  = baseline_df[feature_cols + [label_col]].dropna()
    target = base_clean[label_col]
    use_regressor = pd.api.types.is_float_dtype(target) or target.nunique() > 20
    if use_regressor:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                          learning_rate=0.1, subsample=0.8, random_state=42)
    else:
        model = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                           learning_rate=0.1, subsample=0.8, random_state=42)
    model.fit(base_clean[feature_cols], target)
    explainer = shap.TreeExplainer(model)

    def _mean_abs_shap(sv):
        if isinstance(sv, list):
            return np.mean([np.abs(c).mean(axis=0) for c in sv], axis=0)
        return np.abs(sv).mean(axis=0)

    base_shap  = _mean_abs_shap(explainer.shap_values(base_filled))
    prod_shap  = _mean_abs_shap(explainer.shap_values(prod_filled))
    base_ranks = np.argsort(np.argsort(-base_shap)) + 1
    prod_ranks = np.argsort(np.argsort(-prod_shap)) + 1
    records = [
        FeatureImportanceRecord(
            name=feat,
            baseline_shap=round(float(base_shap[i]), 6),
            production_shap=round(float(prod_shap[i]), 6),
            delta_pct=round((prod_shap[i]-base_shap[i])/(base_shap[i]+1e-9)*100, 2),
            rank_baseline=int(base_ranks[i]),
            rank_production=int(prod_ranks[i]),
            rank_shift=int(prod_ranks[i])-int(base_ranks[i]),
        ) for i, feat in enumerate(feature_cols)
    ]
    records.sort(key=lambda r: abs(r.delta_pct), reverse=True)
    top_collapsed = [r.name for r in records if r.delta_pct < -20][:5]
    print(f"[feature_importance] {scenario_id} | top_collapsed={top_collapsed}")
    return FeatureImportanceReport(scenario_id, records, top_collapsed, "tree")


def importance_shift_ranking(report: FeatureImportanceReport, max_features: int = 0) -> str:
    """Formatted string for agent tool -- ranked table of features by collapse."""
    records = report.records[:max_features] if max_features else report.records
    lines = ["Feature importance shift ranking (sorted by collapse):", "=" * 65]
    for r in records:
        if r.delta_pct < -20:
            flag, arrow = "COLLAPSED", "<<"
        elif r.delta_pct < -5:
            flag, arrow = "declining", " <"
        elif r.delta_pct > 10:
            flag, arrow = "INCREASED", ">>"
        else:
            flag, arrow = "stable",    "  "
        lines.append(f"  {r.name:<28} d={r.delta_pct:+7.1f}%  "
                     f"#{r.rank_baseline}->#{r.rank_production:<4}  "
                     f"base={r.baseline_shap:.4f}  prod={r.production_shap:.4f}  {arrow} {flag}")
    lines.append("=" * 65)
    if report.top_collapsed:
        lines.append(f"TOP COLLAPSED: {report.top_collapsed}")
        lines.append("-> Investigate these features first -- they lost the most predictive signal.")
    else:
        lines.append("No features with >20% importance collapse detected.")
    return "\n".join(lines)


def local_explanation(model, row: pd.Series, feature_cols: list,
                      baseline_df: pd.DataFrame) -> dict:
    """Per-prediction SHAP for one failing row. Positive=pushes to defect."""
    explainer  = shap.TreeExplainer(model)
    row_values = row[feature_cols].fillna(0).values.reshape(1, -1)
    sv = explainer.shap_values(row_values)[0]
    result = {feat: round(float(sv[i]), 6) for i, feat in enumerate(feature_cols)}
    return dict(sorted(result.items(), key=lambda x: abs(x[1]), reverse=True))


def get_worst_prediction(model, prod_df, feature_cols, label_col, scaler=None):
    """Returns the production row where the model is most wrong (highest log-loss)."""
    clean = prod_df[feature_cols + [label_col]].dropna()
    X = clean[feature_cols].values
    if scaler is not None:
        X = scaler.transform(X)
    probs  = model.predict_proba(X)[:, 1]
    labels = clean[label_col].values
    losses = -(labels*np.log(probs+1e-9) + (1-labels)*np.log(1-probs+1e-9))
    return clean.iloc[np.argmax(losses)]


def permutation_importance_fallback(model, baseline_df, prod_df, feature_cols,
                                     label_col, scenario_id,
                                     n_repeats=10) -> FeatureImportanceReport:
    """Model-agnostic importance. Fallback for SVM, neural nets, etc."""
    base_clean = baseline_df[feature_cols + [label_col]].dropna()
    prod_clean = prod_df[feature_cols + [label_col]].dropna()
    b_pi = sk_perm(model, base_clean[feature_cols], base_clean[label_col], n_repeats=n_repeats, random_state=42)
    p_pi = sk_perm(model, prod_clean[feature_cols], prod_clean[label_col], n_repeats=n_repeats, random_state=42)
    bi, pi = b_pi.importances_mean, p_pi.importances_mean
    br = np.argsort(np.argsort(-bi)) + 1
    pr = np.argsort(np.argsort(-pi)) + 1
    records = [
        FeatureImportanceRecord(feat, round(float(bi[i]),6), round(float(pi[i]),6),
                                round((pi[i]-bi[i])/(abs(bi[i])+1e-9)*100, 2),
                                int(br[i]), int(pr[i]), int(pr[i])-int(br[i]))
        for i, feat in enumerate(feature_cols)
    ]
    records.sort(key=lambda r: abs(r.delta_pct), reverse=True)
    return FeatureImportanceReport(scenario_id, records,
                                   [r.name for r in records if r.delta_pct < -20][:5],
                                   "permutation")
