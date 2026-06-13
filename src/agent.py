"""
src/agent.py
============
DriftGuard — LangChain ReAct Agent (config-driven, domain-agnostic)

Entry point: run_rca()
  - Populates _state with data + RAG pipelines
  - Routes to run_tools_direct() if retrain_advisor not selected (zero LLM)
  - Otherwise builds ReAct agent with only the selected tools registered
  - Logs everything to MLflow
  - Returns result dict with intermediate_steps + final JSON output
"""

import json
import os
import time
import mlflow
import pandas as pd
from datetime import datetime
from typing import Any

from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain_core.callbacks import BaseCallbackHandler

from src.config_loader import DriftGuardConfig
from src.tools import (
    TOOL_FUNCTIONS,
    TOOL_DESCRIPTIONS,
    _state,
)
from src.contracts import DriftReport


# Fixed canonical order — the agent always runs tools in this sequence
TOOL_ORDER = [
    "drift_scan",
    "shap_importance",
    "lineage_lookup",
    "rag_doc_search",
    "schema_diff",
    "stats_compare",
    "retrain_advisor",
]


# ── LIVE STEP CALLBACK ────────────────────────────────────────────────────────

class _StepCallbackHandler(BaseCallbackHandler):
    """Calls step_callback(tool_name, output) after each tool returns."""
    def __init__(self, step_callback):
        super().__init__()
        self._cb = step_callback

    def on_tool_end(self, output: str, *, run_id: Any = None, **kwargs: Any) -> None:
        name = kwargs.get("name", "tool")
        if self._cb:
            self._cb(name, str(output))


# ── PROMPT BUILDER ────────────────────────────────────────────────────────────

def build_react_prompt(
    config: DriftGuardConfig,
    selected_tools: list[str] | None = None,
) -> str:
    """
    Build the ReAct system prompt.
    Investigation order reflects only the selected tools (in canonical order).
    """
    active = [t for t in TOOL_ORDER if selected_tools is None or t in selected_tools]

    hint_section = (
        f"\nADDITIONAL CONTEXT: {config.agent_hint}\n"
        if config.agent_hint else ""
    )

    order_lines = [
        f"{i + 1}. {t}" for i, t in enumerate(active)
    ]
    order_block = "\n".join(order_lines)

    return f"""You are DriftGuard, {config.agent_role}.
{config.agent_task}
{hint_section}
You have access to the following tools:
{{tools}}

Use this EXACT format for every step:

Thought: [your reasoning about what to investigate next]
Action: [tool name — must be one of: {{tool_names}}]
Action Input: [input to the tool, or "none" if the tool takes no input]
Observation: [tool output — do not modify this]

Repeat Thought/Action/Action Input/Observation as needed.

INVESTIGATION ORDER — run each tool EXACTLY ONCE in this sequence. Never repeat a tool.
{order_block}

Check the scratchpad above — if a tool already has an Observation, skip it and move to the next one.

After all tools have run, output ONLY valid JSON (no markdown, no explanation):
{{{{
  "root_causes": ["cause 1 with specific evidence", "cause 2 if applicable"],
  "business_impacts": ["impact 1 with metric", "impact 2", "impact 3"],
  "remediation_steps": ["immediate action", "24h action", "1-week action", "prevention"],
  "confidence_pct": <integer 80-99>
}}}}

Reference actual feature names, dates, and PSI values from the tool outputs.
Be specific — vague answers are not useful to the engineering team.

{{agent_scratchpad}}

Begin investigation.
Question: {{input}}"""


# ── retrain_advisor TOOL (LLM synthesis) ──────────────────────────────────────

def _make_retrain_advisor(
    config: DriftGuardConfig,
    conversation_context: list,
    synthesis_model: str | None = None,
    report=None,
):
    """
    Returns the retrain_advisor function bound to the given synthesis model.
    synthesis_model overrides config.agent.synthesis_model when provided.
    report (DriftReport) is passed in to inject hard facts that cannot be hallucinated.
    """
    effective_synthesis_model = synthesis_model or config.agent.synthesis_model

    def _retrain_advisor_fn(_: str) -> str:
        from groq import Groq

        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            return json.dumps({
                "root_causes": ["GROQ_API_KEY not set — cannot run LLM synthesis"],
                "business_impacts": ["Unable to generate impact analysis"],
                "remediation_steps": ["Set GROQ_API_KEY in .env and re-run"],
                "confidence_pct": 0
            })

        context_str = "\n\n".join(conversation_context) if conversation_context else \
                      "No prior tool outputs captured."

        # Build hard-facts block — all numbers computed by Python, never by the LLM
        from src.tools import _state as _tools_state
        computed_stats = _tools_state.get("computed_stats", {})

        # Extract lineage event from tool_full_outputs — populated as tools run,
        # so it's available when retrain_advisor fires (conversation_context is
        # still empty at that point — it's filled only after invoke() returns).
        import re as _re
        lineage_event_text = None
        lineage_full = _tools_state.get("tool_full_outputs", {}).get("lineage_lookup", "")
        if lineage_full and "[MOCK]" not in lineage_full:
            # Match full event blocks: "YYYY-MM-DD TYPE\n  body line\n  body line..."
            # Single-line regex misses short headers like "2024-04-20 INFRA" (only 6 chars
            # after the date — below the 15-char minimum needed to distinguish them).
            event_blocks = _re.findall(
                r'20\d\d-\d\d-\d\d[^\n]*(?:\n[ \t]+[^\n]+)*', lineage_full
            )
            if event_blocks:
                if report is not None and len(event_blocks) > 1:
                    # Score each block by how many drifted feature names appear in its text.
                    # The 2024-04-20 INFRA block explicitly lists temp_at_weld_celsius,
                    # pressure_test_bar, cycle_time_seconds — scores 3 vs the alloy block's 0.
                    drifted_names = {f.name for f in report.features if f.psi >= 0.10}
                    def _score(blk):
                        return sum(1 for fn in drifted_names if fn in blk)
                    best_block = max(event_blocks, key=_score)
                else:
                    best_block = event_blocks[0]
                # Use first two lines of the block as the event summary
                lines = [l.strip() for l in best_block.split("\n") if l.strip()]
                lineage_event_text = " ".join(lines[:2])[:300]
            else:
                # No date-stamped block — use non-header content as fallback
                content = [l.strip() for l in lineage_full.split("\n")
                           if l.strip()
                           and not l.startswith("Lineage")
                           and not l.startswith("[Result")
                           and "----" not in l]
                if content:
                    lineage_event_text = " ".join(content[:3])[:300]

        if report is not None:
            crit = [f for f in report.features if f.psi >= 0.20]
            warn = [f for f in report.features if 0.10 <= f.psi < 0.20]
            ok   = [f for f in report.features if f.psi < 0.10]

            # Build per-feature lines with Python-computed shift percentages
            def _feature_line(f):
                import math
                cs = computed_stats.get(f.name, {})
                shift = cs.get("mean_shift_pct")
                direction = cs.get("mean_direction", "")
                shap = cs.get("shap_delta_pct")
                parts = [f"  - {f.name}: PSI={f.psi:.3f} {f.status.upper()}"]
                prod_mean_str = "N/A" if (f.production_mean is None or (isinstance(f.production_mean, float) and math.isnan(f.production_mean))) else f"{f.production_mean:.4f}"
                shift_valid = shift is not None and not (isinstance(shift, float) and math.isnan(shift))
                parts.append(
                    f"    mean {f.baseline_mean:.4f} → {prod_mean_str}"
                    + (f" ({direction} {abs(shift):.1f}%)" if shift_valid else "")
                )
                if shap is not None:
                    if shap > 5:
                        interp = "over-relying on OOD data"
                    elif shap < -20:
                        interp = "importance collapsed"
                    elif -20 <= shap <= 5:
                        interp = "stable or minor shift"
                    else:
                        interp = "minor shift"
                    parts.append(f"    SHAP delta: {shap:+.1f}% ({interp})")
                pcts = cs.get("percentiles", {})
                if pcts:
                    pct_str = "  ".join(
                        f"p{k[1:]}: {v['baseline']:.3f}→{v['production']:.3f}"
                        for k, v in pcts.items()
                    )
                    parts.append(f"    Percentiles: {pct_str}")
                return "\n".join(parts)

            crit_lines = "\n".join(_feature_line(f) for f in crit) or "  (none)"
            warn_lines = "\n".join(_feature_line(f) for f in warn) or "  (none)"
            ok_names   = ", ".join(f.name for f in ok) or "(none)"

            lineage_block = (
                f"CONFIRMED LINEAGE EVENT (retrieved from ETL changelog — cite this verbatim as the primary root cause):\n"
                f"{lineage_event_text}"
            ) if lineage_event_text else (
                "CONFIRMED LINEAGE EVENT: None retrieved — do not invent dates, events, or change orders."
            )

            metric     = config.performance_metric   # "Accuracy" or "RMSPE"
            perf_delta = report.model_accuracy_baseline - report.model_accuracy_now
            hard_facts = (
                f"CONFIRMED FACTS — ALL NUMBERS COMPUTED BY PYTHON (do not alter these):\n"
                f"  Baseline {metric} : {report.model_accuracy_baseline:.1%}\n"
                f"  Production {metric}: {report.model_accuracy_now:.1%}\n"
                f"  {metric} change   : {perf_delta*100:.1f}pp\n"
                f"  Monitoring SLA    : {config.performance_sla:.0%}\n"
                f"  SLA breached      : {'YES' if config.is_sla_breached(report.model_accuracy_now) else 'NO'}\n"
                f"  Valid feature names (ONLY use these): {[f.name for f in report.features]}\n"
                f"  CRITICAL drift features (PSI >= 0.20):\n{crit_lines}\n"
                f"  WARNING drift features (PSI 0.10-0.20):\n{warn_lines}\n"
                f"  OK features (PSI < 0.10): {ok_names}\n\n"
                f"{lineage_block}"
            )
        else:
            hard_facts = "(DriftReport not available — use evidence from tool outputs only)"
            lineage_event_text = None

        # Pre-fill the lineage root cause slot so the model never writes "None retrieved"
        if lineage_event_text:
            # Strip to one clean line — the key event sentence
            first_line = lineage_event_text.split("\n")[0].strip()
            lineage_rc0 = f'"{first_line} — caused the CRITICAL drift features to shift outside training distribution.", '
        else:
            lineage_rc0 = ""   # model starts directly with feature-level causes

        # Pre-fill performance values so model can't alter them in business_impacts
        acc_base   = report.model_accuracy_baseline if report else 0
        acc_now    = report.model_accuracy_now      if report else 0
        acc_drop   = (acc_base - acc_now) * 100      if report else 0
        sla        = config.performance_sla
        perf_label = config.performance_metric   # "Accuracy" or "RMSPE"

        synthesis_prompt = f"""You are DriftGuard, {config.agent_role}.
Model under investigation: {config.project.name} — {config.project.model_name}
Domain: {config.project.domain}
Target label: {config.data.label_col}

{hard_facts}

INVESTIGATION EVIDENCE (use ONLY this data — do not invent feature names, dates, PSI values, or events):
{context_str}

STRICT GROUNDING RULES — violating any rule produces an incorrect analysis:
1. Every feature name you mention MUST appear verbatim in the CONFIRMED FACTS above.
2. Every PSI value and mean shift percentage MUST be copied exactly from CONFIRMED FACTS — do not recalculate.
3. Accuracy numbers MUST match CONFIRMED FACTS exactly — do not round or adjust.
4. If schema_diff did NOT report NaN, do NOT mention NaN, ETL adapters, or field type changes anywhere.

LINEAGE RULE — MANDATORY:
- If CONFIRMED LINEAGE EVENT is present above (not "None retrieved"), your root_causes[0] MUST begin with that event.
  Format: "[Date] [Event description from changelog] — this caused [affected features] to shift outside training distribution."
  Example: "2024-04-20 production line speed reduced 22% — this caused temp_at_weld_celsius, pressure_test_bar, cycle_time_seconds to shift outside training distribution."
- Do NOT paraphrase or expand the lineage event with your own knowledge. Quote the key facts directly.
- Features that are NOT in the CRITICAL drift list must not appear in root causes.

INTERPRETATION RULES:
5. For each CRITICAL feature: state name, PSI, mean direction and % (from CONFIRMED FACTS), then SHAP interpretation.
6. SHAP interpretation rules — use CONFIRMED FACTS shap_delta value EXACTLY:
   - shap_delta > +5%  → "model over-relying on OOD data"
   - shap_delta < -20% → "importance collapsed"
   - shap_delta between -20% and +5% → "stable or minor shift — distribution change is primary driver"
   Do NOT write "importance collapsed" when shap_delta is near 0 or positive.
7. Derived features that drifted as a consequence of a primary drift are secondary causes — label them as such.
8. If rag_doc_search returned runbook content, use ONLY the section relevant to the CONFIRMED LINEAGE EVENT type. Ignore sections about other incident types (e.g. if the lineage event is a speed change, ignore alloy runbook sections).

REMEDIATION RULES:
9. Immediate (<4h): ALWAYS start with "Suspend automated pass/fail decisions — route to manual inspection" when accuracy SLA is breached.
10. Each subsequent step must directly address the CONFIRMED LINEAGE EVENT and the drifted features — nothing else.
11. Do NOT introduce actions, systems, or events not present in the CONFIRMED FACTS or CONFIRMED LINEAGE EVENT.

Now produce the root cause analysis. Respond ONLY with valid JSON — no markdown, no preamble:

{{
  "root_causes": [{lineage_rc0}"<feature name> — PSI=<exact value>: mean <direction> <exact %>. SHAP delta: <exact %>. (<over-relying on OOD data / importance collapsed / stable or minor shift>)", "<next CRITICAL feature cause if any>"],
  "business_impacts": [
    "{perf_label} change from {acc_base:.1%} to {acc_now:.1%} ({acc_drop:+.1f}pp), SLA {sla:.0%}",
    "<operational impact specific to {config.project.domain}>",
    "<risk to product quality or safety>"
  ],
  "remediation_steps": [
    "Immediate (<4h): Suspend automated pass/fail — route all {config.project.domain} output to manual inspection",
    "Short-term (24h): <specific action based on CONFIRMED LINEAGE EVENT or drifted features>",
    "Medium-term (1 week): <retraining scope and data collection plan>",
    "Prevention: <PSI monitoring threshold or pipeline safeguard>"
  ],
  "confidence_pct": <integer 70-99 — lower if lineage unavailable>
}}"""

        print(f"[retrain_advisor] Using synthesis model: {effective_synthesis_model}")
        client = Groq(api_key=api_key, max_retries=2)
        raw = ""
        for attempt in range(3):
            try:
                completion = client.chat.completions.create(
                    model=effective_synthesis_model,
                    max_tokens=1024,   # JSON output is 300-500 tokens; 1024 has full headroom
                    messages=[{"role": "user", "content": synthesis_prompt}]
                )
                raw = (completion.choices[0].message.content or "").strip()
                break
            except Exception as e:
                err_str = str(e).lower()
                if "rate_limit" in err_str or "429" in err_str:
                    wait = 60 * (attempt + 1)
                    print(f"[retrain_advisor] Rate limit — waiting {wait}s (attempt {attempt+1}/3)")
                    time.sleep(wait)
                    continue
                return json.dumps({
                    "root_causes": [f"LLM synthesis error: {e}"],
                    "business_impacts": ["Synthesis failed — review tool trace manually"],
                    "remediation_steps": ["Re-run with valid GROQ_API_KEY"],
                    "confidence_pct": 0
                })
        try:
            raw = raw.lstrip("```json").lstrip("```").rstrip("```").strip()
            parsed = json.loads(raw)

            # Programmatic lineage injection — if lineage was retrieved but LLM
            # didn't cite it, prepend it as root_causes[0] unconditionally.
            if lineage_event_text:
                rc_list = parsed.get("root_causes", [])
                event_date = lineage_event_text[:10]  # "YYYY-MM-DD"
                if not rc_list or event_date not in rc_list[0]:
                    first_line = lineage_event_text.split("\n")[0].strip()
                    injected = (
                        f"{first_line} — caused CRITICAL drift features to shift "
                        f"outside training distribution."
                    )
                    parsed["root_causes"] = [injected] + (rc_list or [])

            # Post-validation: flag hallucinated feature names or wrong accuracy values
            if report is not None:
                valid_names = {f.name for f in report.features}
                halluc_flags = []
                for rc in parsed.get("root_causes", []):
                    for word in _re.split(r'[\s,—]+', rc):
                        if (word.endswith(("_um", "_mm", "_bar", "_mpa", "_celsius",
                                           "_seconds", "_ratio", "_deviation",
                                           "_shift", "_density"))
                                and word not in valid_names):
                            halluc_flags.append(f"invented feature '{word}'")
                baseline_pct = round(report.model_accuracy_baseline * 100, 1)
                now_pct = round(report.model_accuracy_now * 100, 1)
                for impact in parsed.get("business_impacts", []):
                    for token in _re.split(r'[\s(),%.]+', impact):
                        try:
                            val = float(token)
                            if 50 < val < 100:
                                if abs(val - baseline_pct) > 5 and abs(val - now_pct) > 5:
                                    halluc_flags.append(
                                        f"accuracy value {val}% does not match confirmed "
                                        f"facts ({baseline_pct}%/{now_pct}%)"
                                    )
                        except ValueError:
                            pass
                if halluc_flags:
                    parsed["_validation_warnings"] = halluc_flags
                    parsed["confidence_pct"] = min(parsed.get("confidence_pct", 50), 40)

            return json.dumps(parsed)
        except json.JSONDecodeError:
            return raw

    return _retrain_advisor_fn


# ── CREATE AGENT ──────────────────────────────────────────────────────────────

def create_agent(
    config: DriftGuardConfig,
    conversation_context: list | None = None,
    step_callback=None,
    selected_tools: list[str] | None = None,
    routing_model: str | None = None,
    synthesis_model: str | None = None,
    report=None,
) -> AgentExecutor:
    if conversation_context is None:
        conversation_context = []

    active = [t for t in TOOL_ORDER if selected_tools is None or t in selected_tools]

    effective_routing   = routing_model   or config.agent.routing_model
    effective_synthesis = synthesis_model or config.agent.synthesis_model

    tools = []
    for name in active:
        desc = TOOL_DESCRIPTIONS[name]
        if name == "retrain_advisor":
            fn = _make_retrain_advisor(config, conversation_context, effective_synthesis, report=report)
        else:
            fn = TOOL_FUNCTIONS[name]
        tools.append(Tool(name=name, func=fn, description=desc))

    callbacks = [_StepCallbackHandler(step_callback)] if step_callback else None

    print(f"[agent] Routing model:   {effective_routing}")
    print(f"[agent] Synthesis model: {effective_synthesis}")

    llm = ChatGroq(
        model=effective_routing,
        max_tokens=512,    # routing steps generate 50-100 tokens; cap prevents runaway generation
        stop_sequences=None,
        max_retries=3,
    )

    prompt_template = build_react_prompt(config, selected_tools=selected_tools)
    prompt   = PromptTemplate.from_template(prompt_template)
    agent    = create_react_agent(llm, tools, prompt)

    # Cap iterations to (number of active tools + 3).
    # +3 gives one retry budget — if the model double-calls one tool,
    # there's still room to reach retrain_advisor and finish cleanly.
    max_iter = min(config.agent.max_iterations, len(active) + 3)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=max_iter,
        handle_parsing_errors=True,
        callbacks=callbacks,
    )
    return executor


# ── DIRECT TOOL EXECUTION (no LLM) ───────────────────────────────────────────

def run_tools_direct(
    config: DriftGuardConfig,
    report: DriftReport,
    baseline_df: pd.DataFrame,
    prod_df: pd.DataFrame,
    selected_tools: list[str],
    feature_cols: list | None = None,
    step_callback=None,
) -> dict:
    """
    Run selected tools in order without the ReAct agent.
    Zero LLM tokens — used when retrain_advisor is not in selected_tools.
    Returns a dict compatible with run_rca's return format.
    """
    effective_features = feature_cols if feature_cols else config.data.feature_cols

    _state.update({
        "config":            config,
        "report":            report,
        "baseline_df":       baseline_df,
        "prod_df":           prod_df,
        "feature_cols":      effective_features,
        "label_col":         config.data.label_col,
        "scenario_id":       report.scenario_id,
        "rag_a":             _state.get("rag_a"),
        "rag_b":             _state.get("rag_b"),
        "rag_c":             _state.get("rag_c"),
        "fi_report":         None,
        "schemas_dir":       config.abs(config.data.schemas_dir),
        "tool_full_outputs": {},
        "computed_stats":    {},
    })

    # Auto-derive query inputs from drift results
    crit = report.critical_features()
    top_feat = crit[0].name if crit else (effective_features[0] if effective_features else "")
    auto_inputs = {
        "drift_scan":      "",
        "shap_importance": "",
        "lineage_lookup":  top_feat,
        "rag_doc_search":  f"{top_feat} investigation procedure",
        "schema_diff":     "",
        "stats_compare":   top_feat,
    }

    intermediate_steps = []
    for tool_name in TOOL_ORDER:
        if tool_name not in selected_tools or tool_name == "retrain_advisor":
            continue
        fn  = TOOL_FUNCTIONS[tool_name]
        inp = auto_inputs.get(tool_name, "")
        obs = fn(inp)
        intermediate_steps.append((_DirectAction(tool_name, inp), obs))
        if step_callback:
            step_callback(tool_name, obs)

    return {
        "output":             "{}",
        "intermediate_steps": intermediate_steps,
        "parsed":             None,   # no LLM synthesis
    }


class _DirectAction:
    """Minimal stand-in for LangChain AgentAction used in direct-run steps."""
    def __init__(self, tool: str, tool_input: str):
        self.tool       = tool
        self.tool_input = tool_input


# ── MAIN ENTRY POINT ──────────────────────────────────────────────────────────

def run_rca(
    config: DriftGuardConfig,
    report: DriftReport,
    baseline_df: pd.DataFrame,
    prod_df: pd.DataFrame,
    rag_a=None,
    rag_b=None,
    rag_c=None,
    feature_cols: list | None = None,
    selected_tools: list[str] | None = None,
    routing_model: str | None = None,
    synthesis_model: str | None = None,
    step_callback=None,
) -> dict:
    """
    Run DriftGuard Root Cause Analysis.

    Routes automatically:
      - selected_tools excludes retrain_advisor → run_tools_direct() (zero LLM)
      - selected_tools includes retrain_advisor  → ReAct agent with selected tools only

    Returns dict with keys:
      "output"             — JSON string (empty "{}" for direct mode)
      "intermediate_steps" — list of (action, observation) tuples
      "parsed"             — parsed dict, or None for direct mode
    """
    active = [t for t in TOOL_ORDER if selected_tools is None or t in selected_tools]
    needs_llm = "retrain_advisor" in active

    # Inject RAG pipelines into shared state before any path uses them
    _state["rag_a"] = rag_a
    _state["rag_b"] = rag_b
    _state["rag_c"] = rag_c

    print(f"\n[agent] Project: {config.project.name} — {config.project.model_name}")
    print(f"[agent] Scenario: {report.scenario_id}")
    print(f"[agent] Selected tools: {active}")
    print(f"[agent] LLM required: {needs_llm}")
    print(f"[agent] Critical features: {[f.name for f in report.critical_features()]}")

    # ── Direct path (no LLM) ─────────────────────────────────────────────────
    if not needs_llm:
        result = run_tools_direct(
            config=config,
            report=report,
            baseline_df=baseline_df,
            prod_df=prod_df,
            selected_tools=active,
            feature_cols=feature_cols,
            step_callback=step_callback,
        )
        result["fi_report"] = _state.get("fi_report")
        return result

    # ── Agent path (LLM) ─────────────────────────────────────────────────────
    effective_features = feature_cols if feature_cols else config.data.feature_cols
    _state.update({
        "config":            config,
        "report":            report,
        "baseline_df":       baseline_df,
        "prod_df":           prod_df,
        "feature_cols":      effective_features,
        "label_col":         config.data.label_col,
        "scenario_id":       report.scenario_id,
        "fi_report":         None,
        "schemas_dir":       config.abs(config.data.schemas_dir),
        "tool_full_outputs": {},
        "computed_stats":    {},
    })

    rag_status = {
        "rag_a": "real" if rag_a is not None else "MOCK",
        "rag_b": "real" if rag_b is not None else "MOCK",
        "rag_c": "real" if rag_c is not None else "MOCK",
    }

    if config.mlflow.tracking_uri:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    run_name = f"rca_{report.scenario_id}_{datetime.now().strftime('%m%d_%H%M')}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "project_name":       config.project.name,
            "model_name":         config.project.model_name,
            "scenario_id":        report.scenario_id,
            "n_features":         len(effective_features),
            "label_col":          config.data.label_col,
            "n_critical":         len(report.critical_features()),
            "n_warn":             len(report.warn_features()),
            "rag_a_status":       rag_status["rag_a"],
            "rag_b_status":       rag_status["rag_b"],
            "model_acc_baseline": report.model_accuracy_baseline,
            "model_acc_now":      report.model_accuracy_now,
            "selected_tools":     ",".join(active),
        })
        for feat in report.features:
            mlflow.log_metric(f"psi_{feat.name}", feat.psi)

        conversation_context = []
        executor = create_agent(
            config,
            conversation_context,
            step_callback=step_callback,
            selected_tools=active,
            routing_model=routing_model,
            synthesis_model=synthesis_model,
            report=report,
        )

        critical_names = [f.name for f in report.critical_features()]
        warn_names     = [f.name for f in report.warn_features()]

        result = executor.invoke({
            "input": (
                f"Investigate drift for {config.project.name} — {config.project.model_name}. "
                f"Scenario: {report.scenario_id}. "
                f"Baseline accuracy: {report.model_accuracy_baseline:.1%}, "
                f"current accuracy: {report.model_accuracy_now:.1%}. "
                f"Label column: {config.data.label_col}. "
                f"Features being analysed: {effective_features}. "
                f"Critical drift features (PSI>0.20): {critical_names}. "
                f"Warning drift features (PSI>0.10): {warn_names}. "
                f"For stats_compare use the EXACT feature name from the critical list above. "
                f"Identify the root cause and provide a concrete remediation plan."
            )
        })

        # Collect FULL tool outputs for retrain_advisor context
        full_outputs = _state.get("tool_full_outputs", {})
        for step in result.get("intermediate_steps", []):
            action, observation = step
            full_text = full_outputs.get(action.tool, str(observation))
            conversation_context.append(
                f"Tool: {action.tool}\n"
                f"Input: {action.tool_input}\n"
                f"Output: {full_text[:2000]}"
            )

        raw_output = result.get("output", "{}")

        # The ReAct loop sometimes hits a parse error after retrain_advisor succeeds
        # (model echoes the JSON without "Final Answer:" prefix → LangChain rejects it →
        # chain finishes with "Agent stopped due to iteration limit").
        # In that case the JSON is already sitting in intermediate_steps as the observation.
        # Rescue it before making an extra API call.
        if not raw_output.strip().startswith("{"):
            for action, obs in reversed(result.get("intermediate_steps", [])):
                if action.tool == "retrain_advisor":
                    obs_clean = str(obs).strip().lstrip("```json").rstrip("```").strip()
                    if obs_clean.startswith("{"):
                        raw_output = obs_clean
                        print("[agent] Rescued retrain_advisor JSON from intermediate steps.")
                    break

        # Still no valid JSON — re-run synthesis with full context (costs one extra API call)
        if not raw_output.strip().startswith("{"):
            print("[agent] Re-running retrain_advisor synthesis (no valid JSON in output).")
            retrain_fn = _make_retrain_advisor(config, conversation_context, synthesis_model, report=report)
            raw_output = retrain_fn("")

        try:
            clean  = raw_output.strip().lstrip("```json").rstrip("```").strip()
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            parsed = {
                "root_causes":        [raw_output[:500]],
                "business_impacts":   ["Parse error — see raw output"],
                "remediation_steps":  ["Review raw agent output"],
                "confidence_pct":     0,
            }

        # ── Programmatic post-validation ─────────────────────────────────────
        # Catch hallucinated feature names and accuracy numbers before they reach the UI.
        # These checks are deterministic — no LLM involved.
        if report is not None:
            valid_names  = {f.name for f in report.features}
            halluc_flags = []

            # Check root causes for invented feature names
            for rc in parsed.get("root_causes", []):
                for word in rc.replace(",", " ").replace("—", " ").split():
                    if (word.endswith("_um") or word.endswith("_mm") or
                            word.endswith("_bar") or word.endswith("_mpa") or
                            word.endswith("_celsius") or word.endswith("_seconds") or
                            word.endswith("_ratio") or word.endswith("_deviation") or
                            word.endswith("_shift") or word.endswith("_density")):
                        if word not in valid_names:
                            halluc_flags.append(f"invented feature '{word}'")

            # Check accuracy numbers are in the right ballpark (±5pp tolerance)
            baseline_pct = round(report.model_accuracy_baseline * 100, 1)
            now_pct      = round(report.model_accuracy_now * 100, 1)
            for impact in parsed.get("business_impacts", []):
                for token in impact.split():
                    token_clean = token.strip("()%,.")
                    try:
                        val = float(token_clean)
                        if 50 < val < 100:  # looks like an accuracy percentage
                            if abs(val - baseline_pct) > 5 and abs(val - now_pct) > 5:
                                halluc_flags.append(
                                    f"accuracy value {val}% not matching "
                                    f"baseline {baseline_pct}% or production {now_pct}%"
                                )
                    except ValueError:
                        pass

            if halluc_flags:
                print(f"[agent] WARNING — post-validation caught potential hallucinations: {halluc_flags}")
                parsed["_validation_warnings"] = halluc_flags
                # Penalise confidence when hallucinations are detected
                parsed["confidence_pct"] = min(parsed.get("confidence_pct", 50), 40)

        mlflow.log_metric("n_tool_calls",   len(result.get("intermediate_steps", [])))
        mlflow.log_metric("confidence_pct", parsed.get("confidence_pct", 0))
        mlflow.log_text(json.dumps(parsed, indent=2), "rca_output.json")
        mlflow.log_text("\n\n".join(conversation_context), "tool_trace.txt")

        result["output"] = json.dumps(parsed)
        result["parsed"] = parsed

        print(f"\n[agent] RCA complete — confidence: {parsed.get('confidence_pct', '?')}%")
        print(f"[agent] Root causes: {parsed.get('root_causes', [])[:1]}")

    return result
