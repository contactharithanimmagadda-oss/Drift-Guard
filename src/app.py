"""
src/app.py
==========
DriftGuard — Streamlit UI (config-driven, domain-agnostic)

Run:
    streamlit run src/app.py

Env:
    DRIFTGUARD_CONFIG   path to driftguard.yaml (default: ./driftguard.yaml)
    GROQ_API_KEY        required only when retrain_advisor tool is selected

Session state keys:
    steps           list of {"tool", "input", "output"}
    solution        dict with root_causes etc. (None in direct/quick mode)
    fi_report       FeatureImportanceReport
    drift_report    DriftReport
    last_scenario   str or None
    last_mode       "quick" | "full"
    rag_loaded      bool
    rag_a/b/c       RAGPipeline or None
"""

import os
import sys
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ── Load .env before anything reads env vars ──────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(APP_DIR)
load_dotenv(os.path.join(ROOT, ".env"))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config_loader import load_config, DriftGuardConfig, discover_projects

# Discover all example projects at startup (pure file scan, no Streamlit calls)
_PROJECTS = discover_projects(ROOT)   # {project_name: yaml_path}
_FALLBACK_PATH = os.environ.get("DRIFTGUARD_CONFIG", "driftguard.yaml")
_FALLBACK_PATH = _FALLBACK_PATH if os.path.isabs(_FALLBACK_PATH) else os.path.join(ROOT, _FALLBACK_PATH)

# Read selected project from session state (written by the sidebar selectbox below)
_sel_name = st.session_state.get("_project_selector")
if _sel_name and _sel_name in _PROJECTS:
    _active_config_path = _PROJECTS[_sel_name]
elif _PROJECTS:
    _active_config_path = list(_PROJECTS.values())[0]
else:
    _active_config_path = _FALLBACK_PATH

try:
    config: DriftGuardConfig = load_config(_active_config_path)
except (FileNotFoundError, ValueError) as e:
    st.error(str(e)); st.stop()

st.set_page_config(
    page_title=config.ui.page_title,
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("steps",         []),
    ("solution",      None),
    ("fi_report",     None),
    ("drift_report",  None),
    ("last_scenario", None),
    ("last_mode",     None),
    ("rag_loaded",    False),
    ("rag_a",         None),
    ("rag_b",         None),
    ("rag_c",         None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Groq model catalog ───────────────────────────────────────────────────────
# speed_rank: 1 = fastest  |  quality_rank: 1 = best
# Update this list as Groq releases new models.
GROQ_MODELS = {
    "llama-3.1-8b-instant": {
        "label":   "Llama 3.1 8B Instant",
        "speed":   "⚡⚡⚡ Fastest",
        "quality": "★★☆ Basic",
        "note":    "Best for routing — picks tools correctly, very low latency",
        "use":     "routing",
    },
    "gemma2-9b-it": {
        "label":   "Gemma 2 9B",
        "speed":   "⚡⚡ Fast",
        "quality": "★★★ Good",
        "note":    "Good balance — works for routing or light synthesis",
        "use":     "both",
    },
    "mixtral-8x7b-32768": {
        "label":   "Mixtral 8x7B (32K ctx)",
        "speed":   "⚡⚡ Moderate",
        "quality": "★★★ Good",
        "note":    "32K context window — handles longer RAG results well",
        "use":     "synthesis",
    },
    "llama-3.1-70b-versatile": {
        "label":   "Llama 3.1 70B Versatile",
        "speed":   "⚡ Slower",
        "quality": "★★★★ High",
        "note":    "High quality synthesis, previous generation 70B",
        "use":     "synthesis",
    },
    "llama-3.3-70b-versatile": {
        "label":   "Llama 3.3 70B Versatile",
        "speed":   "⚡ Slower",
        "quality": "★★★★★ Best",
        "note":    "Best root cause quality — recommended for synthesis",
        "use":     "synthesis",
    },
}

def _model_label(model_id: str) -> str:
    m = GROQ_MODELS.get(model_id, {})
    return f"{m.get('speed','?')}  {m.get('label', model_id)}"


# ── Tool metadata (shared between sidebar and display) ────────────────────────
TOOL_ORDER = [
    "drift_scan", "shap_importance", "lineage_lookup",
    "rag_doc_search", "schema_diff", "stats_compare", "retrain_advisor",
]
TOOL_META = {
    "drift_scan":      {"icon": "📡", "label": "Drift Scan",       "desc": "PSI / KS across all features"},
    "shap_importance": {"icon": "🧮", "label": "SHAP Importance",  "desc": "Feature importance shift"},
    "lineage_lookup":  {"icon": "🗂️", "label": "Lineage Lookup",   "desc": "ETL changelog search"},
    "rag_doc_search":  {"icon": "📚", "label": "Doc Search",       "desc": "Runbook / SOP search"},
    "schema_diff":     {"icon": "🔀", "label": "Schema Diff",      "desc": "Breaking dtype changes"},
    "stats_compare":   {"icon": "📊", "label": "Stats Compare",    "desc": "KS test on top feature"},
    "retrain_advisor": {"icon": "🤖", "label": "Retrain Advisor",  "desc": "AI root cause synthesis (LLM)"},
}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ DriftGuard")

    # ── Project selector ─────────────────────────────────────────────────────
    if len(_PROJECTS) > 1:
        project_names = list(_PROJECTS.keys())
        default_idx   = project_names.index(_sel_name) if _sel_name in (project_names if _sel_name else []) else 0
        chosen = st.selectbox(
            "Project",
            options=project_names,
            index=default_idx,
            key="_project_selector",
        )
        # When project changes: clear analysis results and RAG state, then rerun.
        # _prev_project MUST be updated before st.rerun() — otherwise the condition
        # stays True on every rerun and causes an infinite reload loop.
        if chosen != st.session_state.get("_prev_project"):
            prev = st.session_state.get("_prev_project")
            st.session_state["_prev_project"] = chosen   # update first
            if prev is not None:                          # skip clear on first load
                for k in ["steps", "solution", "fi_report", "drift_report", "last_scenario", "last_mode"]:
                    st.session_state[k] = [] if k == "steps" else None
                for k in ["rag_a", "rag_b", "rag_c"]:
                    st.session_state[k] = None
                st.session_state.rag_loaded = False
                st.session_state["analysis_mode"] = "Quick Summary"
                st.rerun()

    st.markdown(f"**{config.project.name}**")
    if config.project.description:
        st.caption(config.project.description)
    st.divider()

    # ── Scenario selector ────────────────────────────────────────────────────
    has_scenarios = len(config.scenarios) > 0
    if has_scenarios:
        scenario_id = st.selectbox(
            "Drift scenario",
            options=[sc.id for sc in config.scenarios],
            format_func=lambda sid: next(
                (sc.label for sc in config.scenarios if sc.id == sid), sid
            ),
        )
        selected_sc = next(sc for sc in config.scenarios if sc.id == scenario_id)
        if selected_sc.description:
            st.info(selected_sc.description)
    else:
        scenario_id = None
        selected_sc = None
        st.info(
            f"Baseline: `{config.data.baseline_path}`\n\n"
            f"Production: `{config.data.production_path}`"
        )

    st.divider()

    # ── Feature selector ─────────────────────────────────────────────────────
    st.markdown("**Features to analyse**")
    all_features = config.data.feature_cols
    selected_features = st.multiselect(
        "Features",
        options=all_features,
        default=all_features,
        help="Deselect features to narrow analysis and reduce token usage.",
        label_visibility="collapsed",
    )
    if not selected_features:
        st.warning("Select at least one feature.")
        selected_features = all_features
    st.caption(f"{len(selected_features)} / {len(all_features)} features selected")
    st.divider()

    # ── Analysis mode ────────────────────────────────────────────────────────
    st.markdown("**Analysis mode**")
    analysis_mode = st.radio(
        "Mode",
        options=["Quick Summary", "Full RCA"],
        index=0,
        key="analysis_mode",
        help=(
            "**Quick Summary** — instant drift stats + SHAP, zero LLM tokens.\n\n"
            "**Full RCA** — runs selected tools, optionally with AI synthesis."
        ),
        label_visibility="collapsed",
    )
    st.divider()

    # ── Tool selector (Full RCA only) ─────────────────────────────────────────
    if analysis_mode == "Full RCA":
        st.markdown("**Tools to run**")
        selected_tools = st.multiselect(
            "Tools",
            options=TOOL_ORDER,
            default=TOOL_ORDER,
            format_func=lambda t: f"{TOOL_META[t]['icon']} {TOOL_META[t]['label']}",
            help=(
                "Deselect tools you don't need.\n\n"
                "**Tip:** Removing 🤖 Retrain Advisor skips the LLM entirely — "
                "zero API tokens, instant results."
            ),
            label_visibility="collapsed",
        )
        if not selected_tools:
            st.warning("Select at least one tool.")
            selected_tools = TOOL_ORDER[:]

        needs_llm = "retrain_advisor" in selected_tools
        if needs_llm:
            st.caption(f"🤖 LLM active — routing: `{config.agent.routing_model}` / synthesis: `{config.agent.synthesis_model}`")
        else:
            st.success("⚡ LLM-free mode — zero API tokens")
        st.divider()
    else:
        selected_tools = ["drift_scan", "shap_importance"]

    # ── RAG index status ─────────────────────────────────────────────────────
    if analysis_mode == "Full RCA" and any(
        t in selected_tools for t in ("lineage_lookup", "rag_doc_search")
    ):
        st.markdown("**RAG index status**")
        if st.session_state.rag_loaded:
            st.success("✓ Indexes loaded")
        else:
            st.warning("⚠ Indexes not loaded — MOCK responses will be used.")
            if st.button("Load indexes from disk", key="load_rag"):
                with st.spinner("Loading FAISS indexes..."):
                    try:
                        from src.rag_builder import load_pipelines
                        a, b, c = load_pipelines(config.abs(config.knowledge.indexes_dir))
                        st.session_state.rag_a      = a
                        st.session_state.rag_b      = b
                        st.session_state.rag_c      = c
                        st.session_state.rag_loaded = True
                        st.rerun()
                    except FileNotFoundError as e:
                        st.error(f"Indexes not found: {e}")
                    except Exception as e:
                        st.error(f"Load error: {e}")
        st.divider()

    # ── Model configuration ───────────────────────────────────────────────────
    with st.expander("⚙️ Model configuration", expanded=False):
        st.caption("Override the YAML defaults for this session only.")

        routing_options = [m for m, d in GROQ_MODELS.items() if d["use"] in ("routing", "both")]
        routing_model = st.selectbox(
            "⚡ Routing model  (tool selection)",
            options=routing_options,
            index=routing_options.index(config.agent.routing_model)
                  if config.agent.routing_model in routing_options else 0,
            format_func=_model_label,
            help="Used for every ReAct step that picks the next tool. Needs to be fast — "
                 "output is only 50-100 tokens.",
        )
        r_meta = GROQ_MODELS.get(routing_model, {})
        st.caption(f"{r_meta.get('quality','')}  —  {r_meta.get('note','')}")

        st.divider()

        synthesis_options = [m for m, d in GROQ_MODELS.items() if d["use"] in ("synthesis", "both")]
        synthesis_model = st.selectbox(
            "🎯 Synthesis model  (root cause report)",
            options=synthesis_options,
            index=synthesis_options.index(config.agent.synthesis_model)
                  if config.agent.synthesis_model in synthesis_options else len(synthesis_options) - 1,
            format_func=_model_label,
            help="Used once by Retrain Advisor to produce the JSON root cause report. "
                 "A larger model gives better quality analysis.",
        )
        s_meta = GROQ_MODELS.get(synthesis_model, {})
        st.caption(f"{s_meta.get('quality','')}  —  {s_meta.get('note','')}")

        st.divider()
        st.caption(f"Config file: `{os.path.relpath(_active_config_path, ROOT)}`")
        from src.rag_builder import PIPELINE_CONFIGS
        for p in PIPELINE_CONFIGS:
            st.markdown(f"🔍 **RAG {p['name']}:** `{p['model_name']}`")
        st.caption("RAG uses local FAISS — no LLM tokens.")

    st.divider()

    if analysis_mode == "Full RCA" and "retrain_advisor" in selected_tools:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key or api_key == "your_groq_api_key_here":
            st.error("⚠ GROQ_API_KEY not set.")

    btn_label = (
        "⚡  Run Quick Summary" if analysis_mode == "Quick Summary"
        else ("▶  Run Full RCA (LLM)" if "retrain_advisor" in selected_tools
              else "⚡  Run Tools (LLM-free)")
    )
    run_btn = st.button(btn_label, type="primary", width="stretch")

    if st.button("Reset", width="stretch"):
        for key in ["steps", "solution", "fi_report", "drift_report", "last_scenario", "last_mode"]:
            st.session_state[key] = [] if key == "steps" else None
        st.rerun()


# ── Main header ───────────────────────────────────────────────────────────────
st.title(f"🛡️ DriftGuard — {config.project.name}")
st.caption(
    f"{config.project.model_name}  ·  "
    f"{config.project.domain + '  ·  ' if config.project.domain else ''}"
    f"Label: `{config.data.label_col}`  ·  "
    f"{len(selected_features)} / {len(config.data.feature_cols)} features"
)

if selected_sc and (selected_sc.acc_baseline or selected_sc.acc_production):
    m1, m2, m3, m4 = st.columns(4)
    delta     = selected_sc.acc_production - selected_sc.acc_baseline
    met_label = config.performance_metric   # "Accuracy" or "RMSPE"
    # For classifiers positive delta is good; for regressors positive delta is bad
    d_color   = "inverse" if config.project.model_type == "regressor" else "normal"
    with m1:
        st.metric(f"Current {met_label}", f"{selected_sc.acc_production:.1%}",
                  delta=f"{delta:+.1%}", delta_color=d_color)
    with m2:
        st.metric(f"Baseline {met_label}", f"{selected_sc.acc_baseline:.1%}")
    with m3:
        st.metric(f"{met_label} change", f"{abs(delta):.1%}")
    with m4:
        st.metric("Scenario", selected_sc.label)
    st.divider()
elif not has_scenarios:
    st.divider()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_data():
    if selected_sc:
        bp = config.baseline_path(selected_sc)
        pp = config.production_path(selected_sc)
    else:
        bp = config.abs(config.data.baseline_path)
        pp = config.abs(config.data.production_path)
    if not os.path.exists(bp):
        st.error(f"Baseline data not found: `{bp}`"); st.stop()
    if not os.path.exists(pp):
        st.error(f"Production data not found: `{pp}`"); st.stop()
    return pd.read_csv(bp), pd.read_csv(pp)


@st.cache_data(show_spinner=False)
def _cached_drift_scan(baseline_hash, prod_hash, features_tuple, label_col,
                       run_scenario, acc_baseline, acc_production,
                       baseline_bytes, prod_bytes):
    """Cache drift scan results — same data always produces the same PSI/KS values."""
    import io
    from src.drift_detector import run_drift_scan
    baseline = pd.read_csv(io.BytesIO(baseline_bytes))
    prod     = pd.read_csv(io.BytesIO(prod_bytes))
    return run_drift_scan(
        baseline, prod, list(features_tuple), label_col, run_scenario,
        model_accuracy_baseline=acc_baseline,
        model_accuracy_now=acc_production,
    )


@st.cache_data(show_spinner=False)
def _cached_shap(baseline_bytes, prod_bytes, features_tuple, label_col, scenario_id):
    """Cache SHAP results — trains GBM + runs SHAP, expensive to repeat on same data."""
    import io
    from src.feature_importance import global_importance
    baseline = pd.read_csv(io.BytesIO(baseline_bytes))
    prod     = pd.read_csv(io.BytesIO(prod_bytes))
    return global_importance(
        baseline_df=baseline, prod_df=prod,
        feature_cols=list(features_tuple),
        label_col=label_col,
        scenario_id=scenario_id,
    )


def _run_drift_scan(baseline, prod, run_scenario, acc_baseline, acc_production):
    """Run drift scan, using cached result when data + features haven't changed."""
    import hashlib
    b_bytes = baseline.to_csv(index=False).encode()
    p_bytes = prod.to_csv(index=False).encode()
    b_hash  = hashlib.md5(b_bytes).hexdigest()
    p_hash  = hashlib.md5(p_bytes).hexdigest()
    return _cached_drift_scan(
        b_hash, p_hash, tuple(selected_features), config.data.label_col,
        run_scenario, acc_baseline, acc_production, b_bytes, p_bytes,
    )


# ── RUN — Quick Summary ───────────────────────────────────────────────────────
if run_btn and analysis_mode == "Quick Summary":
    st.session_state.steps        = []
    st.session_state.solution     = None
    st.session_state.last_scenario = scenario_id
    st.session_state.last_mode    = "quick"

    baseline, prod = _load_data()
    acc_baseline   = selected_sc.acc_baseline   if selected_sc else 0.0
    acc_production = selected_sc.acc_production if selected_sc else 0.0
    run_scenario   = scenario_id or "default"

    with st.status("⚡ Running Quick Summary…", expanded=True) as status:
        try:
            status.write("📡 Computing drift statistics…")
            report = _run_drift_scan(baseline, prod, run_scenario, acc_baseline, acc_production)
            st.session_state.drift_report = report

            crit = report.critical_features()
            warn = report.warn_features()
            status.write(
                f"📡 **Drift scan complete** — "
                f"🔴 {len(crit)} critical  🟡 {len(warn)} warning"
            )

            status.write("🧮 Computing SHAP feature importance…")
            b_bytes = baseline.to_csv(index=False).encode()
            p_bytes = prod.to_csv(index=False).encode()
            fi = _cached_shap(
                b_bytes, p_bytes,
                tuple(selected_features), config.data.label_col, run_scenario,
            )
            st.session_state.fi_report = fi
            collapsed = [r for r in fi.records if r.delta_pct < -20]
            status.write(
                f"🧮 **SHAP complete** — "
                f"{len(collapsed)} feature(s) collapsed >20%"
            )

            status.update(label="✅ Quick Summary complete!", state="complete", expanded=False)
        except Exception as e:
            status.update(label="❌ Error", state="error")
            st.error(f"Quick Summary error: {e}")
            import traceback; st.code(traceback.format_exc())
            st.stop()

    st.rerun()


# ── RUN — Full RCA ────────────────────────────────────────────────────────────
if run_btn and analysis_mode == "Full RCA":
    st.session_state.steps        = []
    st.session_state.solution     = None
    st.session_state.fi_report    = None
    st.session_state.drift_report = None
    st.session_state.last_scenario = scenario_id
    st.session_state.last_mode    = "full"

    baseline, prod = _load_data()
    acc_baseline   = selected_sc.acc_baseline   if selected_sc else 0.0
    acc_production = selected_sc.acc_production if selected_sc else 0.0
    run_scenario   = scenario_id or "default"

    n_tools       = len(selected_tools)
    completed     = []   # filled by callback as tools finish
    start_time    = time.time()

    needs_llm_run = "retrain_advisor" in selected_tools
    status_label  = (
        f"🤖 Running Full RCA ({n_tools} tools)…"
        if needs_llm_run
        else f"⚡ Running {n_tools} tools (LLM-free)…"
    )

    with st.status(status_label, expanded=True) as status:
        if needs_llm_run:
            r_info = GROQ_MODELS.get(routing_model, {})
            s_info = GROQ_MODELS.get(synthesis_model, {})
            status.write(
                f"⚡ **Routing:** `{routing_model}` — {r_info.get('speed', '')}  \n"
                f"🎯 **Synthesis:** `{synthesis_model}` — {s_info.get('quality', '')}"
            )

        def _on_step(tool_name: str, output: str):
            completed.append(tool_name)
            meta    = TOOL_META.get(tool_name, {"icon": "⚙️", "label": tool_name})
            elapsed = time.time() - start_time
            status.write(
                f"{meta['icon']} **{meta['label']}** — done "
                f"&nbsp; `{elapsed:.1f}s` &nbsp; "
                f"({len(completed)}/{n_tools})"
            )

        try:
            # ── Pre-scan (always needed for RCA context) ──────────────────────
            if "drift_scan" not in selected_tools:
                # Still need a report object even if drift_scan isn't in tools
                status.write("📡 Pre-computing drift report…")
                report = _run_drift_scan(baseline, prod, run_scenario, acc_baseline, acc_production)
            else:
                report = _run_drift_scan(baseline, prod, run_scenario, acc_baseline, acc_production)
            st.session_state.drift_report = report

            # Auto-remove retrain_advisor when model is healthy (no critical features)
            # — no drift means nothing for the LLM to synthesise; saves 2,000–3,000 tokens
            active_tools = list(selected_tools)
            crit_count   = len(report.critical_features())
            if "retrain_advisor" in active_tools and crit_count == 0:
                active_tools.remove("retrain_advisor")
                status.write(
                    "ℹ️ No critical features detected — **Retrain Advisor skipped** "
                    "(model is healthy, LLM synthesis not needed)."
                )

            n_tools = len(active_tools)   # update count for progress display

            # Clear stale tool outputs
            from src.tools import _state as _tools_state
            _tools_state["tool_full_outputs"] = {}

            from src.agent import run_rca
            result = run_rca(
                config=config,
                report=report,
                baseline_df=baseline,
                prod_df=prod,
                rag_a=st.session_state.rag_a,
                rag_b=st.session_state.rag_b,
                rag_c=st.session_state.rag_c,
                feature_cols=selected_features,
                selected_tools=active_tools,
                routing_model=routing_model,
                synthesis_model=synthesis_model,
                step_callback=_on_step,
            )

            full_outputs = _tools_state.get("tool_full_outputs", {})
            seen_tools: set = set()
            clean_steps = []
            for step in result.get("intermediate_steps", []):
                tool_name = step[0].tool
                if tool_name == "_Exception":
                    continue
                if tool_name in seen_tools:
                    continue
                seen_tools.add(tool_name)
                clean_steps.append({
                    "tool":   tool_name,
                    "input":  str(step[0].tool_input),
                    "output": full_outputs.get(tool_name, str(step[1])),
                })
            st.session_state.steps = clean_steps
            st.session_state.solution  = result.get("parsed")
            st.session_state.fi_report = (
                result.get("fi_report") or _tools_state.get("fi_report")
            )

            elapsed_total = time.time() - start_time
            finish_label = (
                f"✅ Complete — {len(completed)} tools in {elapsed_total:.0f}s"
            )
            status.update(label=finish_label, state="complete", expanded=True)

        except Exception as e:
            status.update(label="❌ Error during run", state="error")
            st.error(f"Agent error: {e}")
            import traceback; st.code(traceback.format_exc())
            st.stop()

    st.rerun()


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_drift, tab_rca, tab_fi = st.tabs(["📡  Drift Stats", "🔍  Agent RCA", "📊  Feature Importance"])


# ════════════════════════════════════════════════════════
# TAB 1 — Drift Stats
# ════════════════════════════════════════════════════════
with tab_drift:
    dr = st.session_state.drift_report
    if dr is None:
        st.info("Run Quick Summary or Full RCA to see drift statistics.")
    else:
        mode_label = "Quick Summary" if st.session_state.last_mode == "quick" else "Full RCA"
        st.caption(
            f"Source: {mode_label}  ·  scenario: **{dr.scenario_id}**  ·  "
            f"baseline rows: {dr.n_baseline_rows:,}  ·  prod rows: {dr.n_production_rows:,}"
        )

        crit = dr.critical_features()
        warn = dr.warn_features()
        ok   = [f for f in dr.features if f.status == "ok"]

        c1, c2, c3 = st.columns(3)
        c1.metric("🔴 Critical  (PSI > 0.20)", len(crit))
        c2.metric("🟡 Warning   (PSI > 0.10)", len(warn))
        c3.metric("🟢 OK", len(ok))

        if crit:
            st.error(f"**Critical features:** {[f.name for f in crit]}")
        if warn:
            st.warning(f"**Warning features:** {[f.name for f in warn]}")

        rows = []
        for f in sorted(dr.features, key=lambda x: x.psi, reverse=True):
            badge = "🔴 critical" if f.status == "critical" else ("🟡 warn" if f.status == "warn" else "🟢 ok")
            rows.append({
                "Status":        badge,
                "Feature":       f.name,
                "PSI":           f"{f.psi:.4f}",
                "KS stat":       f"{f.ks_stat:.4f}",
                "Baseline mean": f"{f.baseline_mean:.3f}",
                "Prod mean":     f"{f.production_mean:.3f}",
                "Shift":         f"{f.production_mean - f.baseline_mean:+.3f}",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ════════════════════════════════════════════════════════
# TAB 2 — Agent RCA
# ════════════════════════════════════════════════════════
with tab_rca:

    if st.session_state.last_mode == "quick" and st.session_state.drift_report is not None:
        st.info(
            "**Quick Summary** mode was used — no agent output available.\n\n"
            "Switch to **Full RCA** mode and include 🤖 Retrain Advisor for AI synthesis."
        )
    elif (
        st.session_state.last_mode == "full"
        and st.session_state.steps
        and st.session_state.solution is None
    ):
        # Direct (LLM-free) run — show tool trace but no synthesis panel
        st.info(
            "Tools ran in **LLM-free mode** — 🤖 Retrain Advisor was not selected.\n\n"
            "Add **Retrain Advisor** to the tool selection for AI root cause synthesis."
        )
        st.subheader(f"🔧 Tool execution trace ({len(st.session_state.steps)} steps)")
        for i, step in enumerate(st.session_state.steps, 1):
            meta = TOOL_META.get(step["tool"], {"icon": "⚙️", "label": step["tool"]})
            with st.expander(f"{meta['icon']}  Step {i}: {meta['label']}", expanded=(i == 1)):
                st.code(step["output"], language="text")

    elif not st.session_state.steps and st.session_state.solution is None:
        st.info("Select **Full RCA** mode and click Run to start the investigation.")

    else:
        # Full RCA with LLM
        if st.session_state.steps:
            st.subheader(f"🔧 Tool execution trace ({len(st.session_state.steps)} steps)")
            for i, step in enumerate(st.session_state.steps, 1):
                meta = TOOL_META.get(step["tool"], {"icon": "⚙️", "label": step["tool"], "desc": ""})
                with st.expander(
                    f"{meta['icon']}  Step {i}: {meta['label']}"
                    + (f" — {meta['desc']}" if meta.get("desc") else ""),
                    expanded=(i <= 2),
                ):
                    col_in, col_out = st.columns([1, 2])
                    with col_in:
                        st.markdown("**Input**")
                        st.code(step["input"] or "none", language="text")
                    with col_out:
                        st.markdown("**Output**")
                        st.code(step["output"][:1200], language="text")

        if st.session_state.solution:
            sol = st.session_state.solution
            st.divider()
            st.subheader("📋 Root Cause Analysis")

            warnings = sol.get("_validation_warnings", [])
            if warnings:
                st.warning(
                    "**Post-validation detected potential hallucinations** — "
                    "verify these findings against the tool trace before acting on them:\n\n"
                    + "\n".join(f"- {w}" for w in warnings)
                )

            conf = sol.get("confidence_pct", 0)
            if conf > 0:
                st.progress(conf / 100, text=f"Agent confidence: {conf}%")

            col_r, col_i, col_f = st.columns(3)
            with col_r:
                st.markdown("### 🔴 Root Causes")
                for i, item in enumerate(sol.get("root_causes", []), 1):
                    st.markdown(f"**{i}.** {item}")
            with col_i:
                st.markdown("### 🟡 Business Impact")
                for i, item in enumerate(sol.get("business_impacts", []), 1):
                    st.markdown(f"**{i}.** {item}")
            with col_f:
                st.markdown("### 🟢 Remediation Steps")
                for i, item in enumerate(sol.get("remediation_steps", []), 1):
                    st.markdown(f"**{i}.** {item}")

            with st.expander("📄 Raw JSON output"):
                st.json(sol)


# ════════════════════════════════════════════════════════
# TAB 3 — Feature Importance
# ════════════════════════════════════════════════════════
with tab_fi:
    fi = st.session_state.fi_report
    if fi is None:
        st.info("Run Quick Summary or Full RCA (with SHAP Importance tool) to see feature importance.")
    else:
        if fi.top_collapsed:
            st.warning(
                f"⚠ **Top collapsed features:** {fi.top_collapsed}\n\n"
                "These features lost the most predictive power — investigate them first."
            )

        st.subheader("📊 Global feature importance — baseline vs production")
        chart_df = pd.DataFrame({
            "Feature":         [r.name            for r in fi.records],
            "Baseline SHAP":   [r.baseline_shap   for r in fi.records],
            "Production SHAP": [r.production_shap for r in fi.records],
        }).set_index("Feature")
        st.bar_chart(chart_df, horizontal=True, height=max(250, len(fi.records) * 40))
        st.caption(
            "Blue = baseline importance  ·  Red = production importance  ·  "
            "Shrinking bar = feature lost predictive power."
        )

        st.subheader("📉 Importance shift ranking")
        rows = []
        for r in fi.records:
            if r.delta_pct < config.monitoring.shap_collapse_threshold:
                status_str, bg = "▼ COLLAPSED", "🔴"
            elif r.delta_pct < -5:
                status_str, bg = "↓ declining", "🟡"
            elif r.delta_pct > 10:
                status_str, bg = "▲ increased", "🟢"
            else:
                status_str, bg = "  stable", "⚪"
            rows.append({
                "Status":        f"{bg} {status_str}",
                "Feature":       r.name,
                "Δ Importance":  f"{r.delta_pct:+.1f}%",
                "Baseline rank": f"#{r.rank_baseline}",
                "Prod rank":     f"#{r.rank_production}",
                "Rank shift":    f"{r.rank_shift:+d}",
                "Baseline SHAP": f"{r.baseline_shap:.4f}",
                "Prod SHAP":     f"{r.production_shap:.4f}",
            })
        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
            column_config={
                "Δ Importance": st.column_config.TextColumn(width="small"),
                "Status":       st.column_config.TextColumn(width="medium"),
            },
        )
        st.caption(
            f"Collapse threshold: {config.monitoring.shap_collapse_threshold}%  ·  "
            f"Method: {fi.model_type}"
        )
