"""
src/tools.py
============
DriftGuard — All 7 LangChain Tool definitions (config-driven, domain-agnostic)

Tool execution order (enforced by agent system prompt):
  1. drift_scan        — PSI + KS across all features
  2. shap_importance   — full feature importance shift ranking
  3. lineage_lookup    — RAG query on ETL changelog (Pipeline A)
  4. rag_doc_search    — RAG query on runbooks/SOPs (Pipeline B)
  5. schema_diff       — compare schema JSON versions
  6. stats_compare     — KS significance on specific feature
  7. retrain_advisor   — LLM synthesis → structured JSON output (in agent.py)

Scratchpad strategy:
  Each tool returns a COMPACT summary to the agent scratchpad (saves ~1,100 tokens/run).
  The FULL verbose output is stored in _state["tool_full_outputs"][tool_name]
  so the Streamlit UI can display complete detail and retrain_advisor gets full context.

State:
  All tools share _state dict — populated by run_rca() before the agent runs.
  _state['config'] holds the DriftGuardConfig instance.
  RAG pipelines injected via _state['rag_a'], 'rag_b', 'rag_c'].
  MOCK fallbacks are generic — no company-specific content.
"""

import json

from src.drift_detector import compute_psi
from src.feature_importance import global_importance, importance_shift_ranking
from src.schema_diff import auto_detect_schema_diff
from src.contracts import DriftReport, FeatureImportanceReport


# ── SHARED STATE ──────────────────────────────────────────────────────────────

_state: dict = {
    "config":            None,   # DriftGuardConfig
    "report":            None,   # DriftReport from run_drift_scan()
    "baseline_df":       None,   # pd.DataFrame — training reference
    "prod_df":           None,   # pd.DataFrame — production window
    "feature_cols":      [],     # list[str] — from config.data.feature_cols
    "label_col":         "",     # str — from config.data.label_col
    "scenario_id":       "",
    "rag_a":             None,   # RAGPipeline — lineage
    "rag_b":             None,   # RAGPipeline — knowledge / runbooks
    "rag_c":             None,   # RAGPipeline — model meta
    "fi_report":         None,   # FeatureImportanceReport — set after shap_importance fires
    "schemas_dir":       "data/schemas",
    "tool_full_outputs": {},     # tool_name → full verbose output (for UI + retrain_advisor)
    "computed_stats":    {},     # feature_name → dict of Python-computed stats (never LLM-calculated)
}


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _cfg():
    return _state.get("config")

def _project_name() -> str:
    c = _cfg()
    return c.project.name if c else "your project"

def _top_feature() -> str:
    cols = _state.get("feature_cols", [])
    return cols[0] if cols else "feature_1"

def _save(tool_name: str, full: str, compact: str) -> str:
    """Store full output for the UI, return compact output for the agent scratchpad."""
    _state["tool_full_outputs"][tool_name] = full
    return compact


# ── TOOL 1: drift_scan ────────────────────────────────────────────────────────

def _drift_scan_fn(_: str) -> str:
    r: DriftReport = _state["report"]
    if r is None:
        return "ERROR: No DriftReport loaded. Call run_rca() first."

    sorted_feats = sorted(r.features, key=lambda f: f.psi, reverse=True)
    crit = r.critical_features()
    warn = r.warn_features()
    ok   = [f for f in sorted_feats if f.status == "ok"]

    # ── Compute and store mean shift % for all drifted features (Python, not LLM) ──
    for f in crit + warn:
        if f.baseline_mean != 0:
            mean_shift_pct = ((f.production_mean - f.baseline_mean) / abs(f.baseline_mean)) * 100
        else:
            mean_shift_pct = 0.0
        if f.baseline_std != 0:
            std_shift_pct = ((f.production_std - f.baseline_std) / abs(f.baseline_std)) * 100
        else:
            std_shift_pct = 0.0
        direction = "increased" if mean_shift_pct > 0 else "decreased"
        _state["computed_stats"][f.name] = {
            "psi":             round(f.psi, 4),
            "ks_stat":         round(f.ks_stat, 4),
            "status":          f.status,
            "baseline_mean":   round(f.baseline_mean, 4),
            "production_mean": round(f.production_mean, 4),
            "mean_shift_pct":  round(mean_shift_pct, 2),
            "mean_direction":  direction,
            "baseline_std":    round(f.baseline_std, 4),
            "production_std":  round(f.production_std, 4),
            "std_shift_pct":   round(std_shift_pct, 2),
            "shap_delta_pct":  round(f.shap_delta_pct, 2),
        }

    # ── Full output (stored for UI) ──────────────────────────────────────────
    lines = [
        f"Drift scan — {r.scenario_id} (baseline={r.n_baseline_rows}, prod={r.n_production_rows})",
        f"Accuracy: {r.model_accuracy_baseline:.1%} → {r.model_accuracy_now:.1%}",
    ]
    for f in crit + warn:
        cs = _state["computed_stats"][f.name]
        flag = "⚠" if f.status == "critical" else "~"
        lines.append(
            f"  {flag} {f.name}: PSI={f.psi:.4f} KS={f.ks_stat:.4f} [{f.status}] "
            f"mean {f.baseline_mean:.3f}→{f.production_mean:.3f} "
            f"({cs['mean_direction']} {abs(cs['mean_shift_pct']):.1f}%)"
        )
    if ok:
        lines.append(f"  OK ({len(ok)}): {', '.join(f.name for f in ok)} [PSI all < 0.10]")
    lines.append(f"Critical: {[f.name for f in crit]} | Warn: {[f.name for f in warn]}")
    full = "\n".join(lines)

    # ── Compact output (returned to scratchpad) ───────────────────────────────
    delta = r.model_accuracy_now - r.model_accuracy_baseline
    crit_str = ", ".join(
        f"{f.name}(PSI={f.psi:.3f},KS={f.ks_stat:.3f},"
        f"mean {f.baseline_mean:.2f}→{f.production_mean:.2f} "
        f"[{_state['computed_stats'][f.name]['mean_direction']} "
        f"{abs(_state['computed_stats'][f.name]['mean_shift_pct']):.1f}%])"
        for f in crit
    )
    warn_str = ", ".join(f"{f.name}(PSI={f.psi:.3f})" for f in warn)
    ok_names = ", ".join(f.name for f in ok)
    compact = (
        f"Acc: {r.model_accuracy_baseline:.1%}→{r.model_accuracy_now:.1%} (Δ{delta:+.1%}). "
        f"CRITICAL({len(crit)}): {crit_str or 'none'}. "
        f"WARN({len(warn)}): {warn_str or 'none'}. "
        f"OK({len(ok)}): {ok_names or 'none'}."
    )

    return _save("drift_scan", full, compact)


# ── TOOL 2: shap_importance ───────────────────────────────────────────────────

def _shap_importance_fn(_: str) -> str:
    s = _state
    if s["baseline_df"] is None:
        return "ERROR: No data loaded. Run run_rca() first."

    try:
        fi: FeatureImportanceReport = global_importance(
            baseline_df=s["baseline_df"],
            prod_df=s["prod_df"],
            feature_cols=s["feature_cols"],
            label_col=s["label_col"],
            scenario_id=s.get("scenario_id", "unknown"),
        )
        _state["fi_report"] = fi

        # ── Full output (stored for UI) ──────────────────────────────────────
        full = importance_shift_ranking(fi)  # all features, no cap

        # ── Compact output ───────────────────────────────────────────────────
        collapsed = [r for r in fi.records if r.delta_pct < -20]
        declining = [r for r in fi.records if -20 <= r.delta_pct < -5]
        stable    = [r for r in fi.records if -5 <= r.delta_pct <= 10]
        increased = [r for r in fi.records if r.delta_pct > 10]

        parts = []
        if collapsed:
            coll_str = ", ".join(
                f"{r.name}(Δ={r.delta_pct:+.1f}%,#{r.rank_baseline}→#{r.rank_production})"
                for r in collapsed[:5]
            )
            parts.append(f"COLLAPSED(>20%): {coll_str}.")
        if declining:
            decl_str = ", ".join(f"{r.name}(Δ={r.delta_pct:+.1f}%)" for r in declining[:4])
            parts.append(f"DECLINING: {decl_str}.")
        stable_names = ", ".join(r.name for r in (stable + increased)[:6])
        if stable_names:
            parts.append(f"STABLE/UP: {stable_names}.")
        if fi.top_collapsed:
            parts.append(f"Investigate first: {fi.top_collapsed}.")
        else:
            parts.append("No features with >20% collapse.")

        compact = " ".join(parts)
        return _save("shap_importance", full, compact)

    except Exception as e:
        return f"SHAP importance error: {e}"


# ── TOOL 3: lineage_lookup ────────────────────────────────────────────────────

def _lineage_lookup_fn(query: str) -> str:
    rag_a = _state["rag_a"]

    if rag_a is None:
        full = (
            f"[MOCK lineage_lookup] RAG index not loaded for {_project_name()}.\n"
            f"Query was: '{query}'\n"
            "To use real RAG: build indexes first (see README — Step 3: Build RAG Indexes).\n"
            f"No lineage events found in mock mode for feature: {query or _top_feature()}."
        )
        compact = (
            f"[MOCK] No lineage index loaded. Query: '{query}'. "
            "No ETL events found. Real index needed for lineage data."
        )
        return _save("lineage_lookup", full, compact)

    try:
        results = rag_a.query(query, k=3)
        if not results:
            msg = f"No lineage entries found for query: '{query}'"
            return _save("lineage_lookup", msg, msg)

        lines = [f"Lineage lookup results for: '{query}'", "-" * 50]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[Result {i}] score={r['score']:.4f} | source={r['source']}")
            lines.append(r["text"])
        full = "\n".join(lines)

        # Compact: top result only, truncated
        top = results[0]
        excerpt = top["text"][:220].replace("\n", " ").strip()
        compact = (
            f"Lineage query: '{query}'. "
            f"Top match (score={top['score']:.3f}, src={top['source']}): {excerpt}"
        )
        return _save("lineage_lookup", full, compact)

    except Exception as e:
        return f"lineage_lookup error: {e}"


# ── TOOL 4: rag_doc_search ────────────────────────────────────────────────────

def _rag_doc_search_fn(query: str) -> str:
    rag_b = _state["rag_b"]

    if rag_b is None:
        full = (
            f"[MOCK rag_doc_search] Knowledge index not loaded for {_project_name()}.\n"
            f"Query was: '{query}'\n"
            "To use real RAG: build indexes first (see README — Step 3: Build RAG Indexes).\n"
            "Generic guidance: When a feature shows PSI > 0.20, investigate upstream data "
            "changes (sensor recalibration, schema updates, new data sources). "
            "Retrain is recommended if feature importance has collapsed > 20%."
        )
        compact = (
            f"[MOCK] No runbook index loaded. Query: '{query}'. "
            "Generic: PSI>0.20→investigate upstream changes (sensor/schema/source). "
            "Retrain if importance collapsed >20%."
        )
        return _save("rag_doc_search", full, compact)

    try:
        results = rag_b.query(query, k=3)
        if not results:
            msg = f"No runbook entries found for query: '{query}'"
            return _save("rag_doc_search", msg, msg)

        lines = [f"Runbook / knowledge search: '{query}'", "-" * 50]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[Result {i}] score={r['score']:.4f} | source={r['source']}")
            lines.append(r["text"])
        full = "\n".join(lines)

        top = results[0]
        excerpt = top["text"][:220].replace("\n", " ").strip()
        compact = (
            f"Runbook query: '{query}'. "
            f"Top match (score={top['score']:.3f}, src={top['source']}): {excerpt}"
        )
        return _save("rag_doc_search", full, compact)

    except Exception as e:
        return f"rag_doc_search error: {e}"


# ── TOOL 5: schema_diff ───────────────────────────────────────────────────────

def _schema_diff_fn(_: str) -> str:
    schemas_dir = _state.get("schemas_dir", "data/schemas")
    try:
        full = auto_detect_schema_diff(schemas_dir)

        # Compact: keep only lines that contain change keywords
        keywords = ("BREAKING", "WARN", "change", "added", "removed",
                    "renamed", "type", "NULL", "NaN", "no changes", "identical", "ERROR")
        key_lines = [
            ln.strip() for ln in full.split("\n")
            if ln.strip() and any(kw.lower() in ln.lower() for kw in keywords)
        ]
        compact = " | ".join(key_lines[:6]) if key_lines else full[:300].replace("\n", " ")
        return _save("schema_diff", full, compact)

    except Exception as e:
        msg = (
            f"schema_diff: Could not compare schemas in '{schemas_dir}'. Error: {e}\n"
            "Ensure data/schemas/ contains at least 2 versioned JSON schema files."
        )
        return _save("schema_diff", msg, f"schema_diff error: {e}")


# ── TOOL 6: stats_compare ─────────────────────────────────────────────────────

def _stats_compare_fn(feature_name: str) -> str:
    from scipy.stats import ks_2samp

    s = _state
    if s["baseline_df"] is None:
        return "ERROR: No data loaded."

    feat = feature_name.strip()
    if feat not in s["feature_cols"]:
        matches = [c for c in s["feature_cols"] if feat.lower() in c.lower()]
        if matches:
            feat = matches[0]
        else:
            return (
                f"ERROR: Feature '{feature_name}' not found. "
                f"Available features: {s['feature_cols']}"
            )

    base_vals = s["baseline_df"][feat].dropna().values
    prod_vals  = s["prod_df"][feat].dropna().values
    # Raw values (NaN included) for PSI so NaN corruption is counted as a separate bin
    base_raw  = s["baseline_df"][feat].values
    prod_raw  = s["prod_df"][feat].values

    if len(prod_vals) == 0:
        msg = (
            f"{feat}: Production column is 100% NaN — "
            "likely caused by an ETL schema change or pipeline failure. "
            "PSI=1.0 (maximum), statistical test not applicable."
        )
        return _save("stats_compare", msg, msg)

    import numpy as np

    psi   = compute_psi(base_raw, prod_raw)
    ks, p = ks_2samp(base_vals, prod_vals)

    sig = (
        "HIGHLY SIGNIFICANT" if p < 0.001 else
        "SIGNIFICANT"        if p < 0.05  else
        "not significant"
    )
    psi_label = "CRITICAL" if psi > 0.2 else "WARN" if psi > 0.1 else "ok"

    # ── Python-computed stats (no LLM arithmetic) ──────────────────────────────
    b_mean, p_mean = base_vals.mean(), prod_vals.mean()
    b_std,  p_std  = base_vals.std(),  prod_vals.std()
    mean_shift_pct = ((p_mean - b_mean) / abs(b_mean)) * 100 if b_mean != 0 else 0.0
    std_shift_pct  = ((p_std  - b_std)  / abs(b_std))  * 100 if b_std  != 0 else 0.0
    direction      = "increased" if mean_shift_pct > 0 else "decreased"

    percentiles = [5, 25, 50, 75, 95]
    b_pcts = np.percentile(base_vals, percentiles)
    p_pcts = np.percentile(prod_vals, percentiles)

    # Enrich computed_stats entry (merges with drift_scan entry if present)
    existing = _state["computed_stats"].get(feat, {})
    existing.update({
        "psi":             round(psi, 4),
        "ks_stat":         round(ks, 4),
        "p_value":         float(f"{p:.2e}"),
        "significance":    sig,
        "psi_label":       psi_label,
        "baseline_mean":   round(b_mean, 4),
        "production_mean": round(p_mean, 4),
        "mean_shift_pct":  round(mean_shift_pct, 2),
        "mean_direction":  direction,
        "baseline_std":    round(b_std, 4),
        "production_std":  round(p_std, 4),
        "std_shift_pct":   round(std_shift_pct, 2),
        "baseline_min":    round(float(base_vals.min()), 4),
        "baseline_max":    round(float(base_vals.max()), 4),
        "production_min":  round(float(prod_vals.min()), 4),
        "production_max":  round(float(prod_vals.max()), 4),
        "percentiles": {
            f"p{pct}": {"baseline": round(float(b_pcts[i]), 4),
                        "production": round(float(p_pcts[i]), 4)}
            for i, pct in enumerate(percentiles)
        },
    })
    _state["computed_stats"][feat] = existing

    pct_lines = "\n".join(
        f"    p{pct:2d}: {b_pcts[i]:.3f} → {p_pcts[i]:.3f}"
        for i, pct in enumerate(percentiles)
    )
    full = (
        f"Statistical comparison: {feat}\n"
        f"  PSI      = {psi:.4f}   ({psi_label})\n"
        f"  KS stat  = {ks:.4f}\n"
        f"  p-value  = {p:.2e}   ({sig} at α=0.001)\n"
        f"  Mean     : {b_mean:.4f} → {p_mean:.4f}  ({direction} {abs(mean_shift_pct):.1f}%)\n"
        f"  Std dev  : {b_std:.4f}  → {p_std:.4f}   (shift {std_shift_pct:+.1f}%)\n"
        f"  Range    : [{base_vals.min():.3f}, {base_vals.max():.3f}] → "
        f"[{prod_vals.min():.3f}, {prod_vals.max():.3f}]\n"
        f"  Percentiles (baseline → production):\n{pct_lines}\n"
        f"  Verdict  : Distribution shift is "
        f"{'REAL — not sampling noise' if p < 0.001 else 'within normal sampling variation'}."
    )
    compact = (
        f"{feat}: PSI={psi:.4f}({psi_label}), KS={ks:.4f}, p={p:.2e}({sig}). "
        f"Mean {b_mean:.3f}→{p_mean:.3f} ({direction} {abs(mean_shift_pct):.1f}%). "
        f"Std {b_std:.3f}→{p_std:.3f} ({std_shift_pct:+.1f}%). "
        f"Verdict: {'Real drift confirmed.' if p < 0.001 else 'Within sampling variation.'}"
    )

    return _save("stats_compare", full, compact)


# ── TOOL 7: retrain_advisor (stub) ────────────────────────────────────────────

def _retrain_advisor_fn(_: str) -> str:
    """
    Stub — replaced by the LLM synthesis call in agent.py.
    Returns placeholder so tools.py can be imported without an API key.
    """
    return (
        "retrain_advisor: Synthesis pending — "
        "this tool is implemented in agent.py as a live Groq API call. "
        "It synthesises all prior tool outputs into a structured "
        "JSON root cause + remediation plan."
    )


# ── TOOL REGISTRY ─────────────────────────────────────────────────────────────

TOOL_FUNCTIONS = {
    "drift_scan":       _drift_scan_fn,
    "shap_importance":  _shap_importance_fn,
    "lineage_lookup":   _lineage_lookup_fn,
    "rag_doc_search":   _rag_doc_search_fn,
    "schema_diff":      _schema_diff_fn,
    "stats_compare":    _stats_compare_fn,
    "retrain_advisor":  _retrain_advisor_fn,
}

TOOL_DESCRIPTIONS = {
    "drift_scan": (
        "Scan features for PSI/KS drift. Run FIRST. No input needed."
    ),
    "shap_importance": (
        "Compute SHAP feature importance shift. Run SECOND. No input needed."
    ),
    "lineage_lookup": (
        "Search ETL changelog for upstream data changes. Input: feature name or keyword."
    ),
    "rag_doc_search": (
        "Search runbooks and SOPs for response procedures. Input: question or keyword."
    ),
    "schema_diff": (
        "Compare schema versions for breaking dtype changes. No input needed."
    ),
    "stats_compare": (
        "KS test + PSI on one feature to confirm drift significance. Input: exact feature name."
    ),
    "retrain_advisor": (
        "Synthesise all findings into root cause + remediation JSON. Run LAST. No input needed."
    ),
}
