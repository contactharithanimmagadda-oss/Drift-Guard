# DriftGuard

**Config-driven ML drift detection with an AI root-cause analysis agent.**

DriftGuard monitors your production ML model, detects when feature distributions shift, and runs a 7-tool LangChain ReAct agent that reads your data lineage, runbooks, and model documentation to explain *why* the model degraded — not just *that* it did.

---

## What it does

| Step | What happens |
|------|-------------|
| 1. Load data | Reads baseline (training-time) and production CSVs defined in your `driftguard.yaml` |
| 2. Drift scan | Computes PSI, KS-test, and SHAP delta for every feature |
| 3. Classify severity | PSI < 0.10 = OK · 0.10–0.20 = Warn · > 0.20 = Critical |
| 4. Quick Summary | Shows drift table + feature importance report instantly |
| 5. Full RCA | AI agent reasons over drift stats + your knowledge base to write a root-cause report with a fix plan |

The agent uses a **dual-model architecture**: a fast 8B model for tool routing and a 70B model for the final synthesis — keeping cost low without sacrificing quality.

---

## Requirements

- Python 3.10 – 3.12
- A free [Groq API key](https://console.groq.com) (used for the LLM agent)

---

## Setup

```bash
# 1. Clone
git clone https://github.com/contactharithanimmagadda-oss/Drift-Guard.git
cd Drift-Guard

# 2. Create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key
#    Create a .env file in the project root:
echo GROQ_API_KEY=your_key_here > .env
```

Get a free Groq API key at [console.groq.com](https://console.groq.com) — no credit card required.

---

## Run

```bash
streamlit run src/app.py
```

The app opens at `http://localhost:8501`.

---

## Two ready-to-run examples

The repo ships with two fully worked examples. Select either from the **Project** dropdown in the sidebar.

### SteelGuard GmbH — Manufacturing Defect Classifier

Steel pipe defect detection model. 4 scenarios:

| Scenario | What drifted | Expected critical feature |
|----------|-------------|--------------------------|
| Sensor Recalibration | Laser sensor offset shifted +0.2 → +1.8 µm | `surface_roughness_um` |
| Concept Drift | New X400 alloy introduced, never in training | 9 features bimodal |
| ETL Schema Change | `operator_shift` column: INT → VARCHAR, 100% NaN | `operator_shift`, `is_night_shift` |
| Production Slowdown | Line speed reduced 22%, 3 sensor readings out-of-range | `temp_at_weld_celsius`, `pressure_test_bar` |

### Rossmann Retail GmbH — Sales Forecaster (Regressor)

Daily store sales forecasting model. 4 scenarios:

| Scenario | What drifted | Expected critical feature |
|----------|-------------|--------------------------|
| Promotional Calendar Shift | Promo rate jumped from 40% → 78% | `Promo` |
| New Competitor Wave | DM opened 47 stores, competition distance halved | `CompetitionDistance` |
| ETL Open Corruption | Data type change injected NaN into 32% of `Open` values | `Open` |
| Holiday Encoding Mismatch | ERP changed `StateHoliday` encoding, ETL coerced to NaN | `StateHoliday` |

---

## How to use the UI

1. **Select project** from the sidebar dropdown
2. **Select a scenario** from the scenario dropdown
3. Choose a mode:
   - **Quick Summary** — instant drift table, no LLM call needed
   - **Full RCA** — click *Load indexes* first, then *Run Full RCA*
4. Read the drift table (red = critical, orange = warn, green = ok)
5. In Full RCA mode, expand *Agent reasoning steps* to see the agent's tool calls

---

## Add your own project

1. Create a folder: `examples/my_project/`

2. Add your data:
   ```
   examples/my_project/
   ├── driftguard.yaml              ← config file (see template below)
   ├── data/
   │   └── scenarios/
   │       └── my_scenario/
   │           ├── baseline.csv    ← training-time distribution sample
   │           └── production.csv  ← recent production data sample
   └── knowledge/
       ├── lineage/                ← data pipeline docs (.txt / .md)
       ├── runbooks/               ← incident response docs
       └── model_meta/             ← model card, training notes
   ```

3. Create `examples/my_project/driftguard.yaml`:
   ```yaml
   project:
     name:       "My Project"
     model_name: "MyModel v1.0"
     model_type: "binary_classifier"   # or "regressor"
     domain:     "My Domain"

   data:
     label_col: "target"
     feature_cols:
       - feature_1
       - feature_2
     baseline_path:   "data/scenarios/{id}/baseline.csv"
     production_path: "data/scenarios/{id}/production.csv"
     schemas_dir:     "data/schemas"

   scenarios:
     - id:    my_scenario
       label: "My Scenario"
       description: "What changed and why."
       acc_baseline:   0.90
       acc_production: 0.75
       baseline_path:   "data/scenarios/my_scenario/baseline.csv"
       production_path: "data/scenarios/my_scenario/production.csv"

   agent:
     routing_model:   "llama-3.1-8b-instant"
     synthesis_model: "llama-3.3-70b-versatile"
     max_iterations: 15
     max_tokens: 2048
     role_description: "an ML Operations expert for {project_name}"
     task_description: >
       Investigate why {model_name} degraded in production and produce a
       root cause analysis with a remediation plan.
     investigation_hint: >
       Focus on distribution shift in the most business-critical features.

   knowledge:
     lineage_docs: "knowledge/lineage"
     runbook_docs: "knowledge/runbooks"
     model_docs:   "knowledge/model_meta"
     indexes_dir:  "indexes"

   monitoring:
     psi_critical: 0.20
     psi_warn:     0.10

   mlflow:
     experiment_name: "MyProject_DriftMonitoring"
     tracking_uri: null

   ui:
     page_title: "DriftGuard — My Project"
   ```

4. Restart the app — your project appears automatically in the sidebar dropdown. No code changes needed.

---

## Project structure

```
Drift-Guard/
├── src/
│   ├── app.py              # Streamlit UI
│   ├── agent.py            # LangChain ReAct agent + 7 tools
│   ├── drift_detector.py   # PSI, KS test, SHAP delta
│   ├── feature_importance.py
│   ├── rag_builder.py      # FAISS vector index over knowledge docs
│   ├── tools.py            # Agent tool implementations
│   ├── contracts.py        # DriftReport, FeatureDrift, SchemaDiff dataclasses
│   ├── config_loader.py    # YAML → DriftGuardConfig, project discovery
│   └── schema_diff.py      # JSON schema comparison
├── examples/
│   ├── steelguard/         # Manufacturing classifier example
│   └── rossmann/           # Retail regressor example
├── tests/                  # pytest suite (19 tests)
├── requirements.txt
└── driftguard.yaml         # Root config (points to steelguard by default)
```

---

## Key concepts

**PSI (Population Stability Index)** — measures how much a feature's distribution has shifted between baseline and production. NaN values are treated as a separate bin, so ETL corruption that injects nulls is detected immediately.

**SHAP delta** — percentage change in mean absolute SHAP value per feature between baseline and production. A large negative delta means a feature that was important at training time is being ignored by the model now — a strong signal of covariate shift.

**RAG knowledge base** — the agent embeds your lineage docs, runbooks, and model cards into a FAISS vector index. When it needs context to explain a drift finding, it retrieves the most relevant passages rather than hallucinating.

**Dual-model routing** — tool selection uses `llama-3.1-8b-instant` (fast, cheap). The final root-cause synthesis uses `llama-3.3-70b-versatile` (high quality). Both run on Groq's free tier.

---

## Run the tests

```bash
pytest tests/ -v
```

Expected: 19 tests pass.

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes (for Full RCA) | Groq API key — free at console.groq.com |
| `DRIFTGUARD_CONFIG` | No | Override config path (default: `./driftguard.yaml`) |

---

## Contributors

- Haritha Nimmagadda
- Kasmin Talukdar
