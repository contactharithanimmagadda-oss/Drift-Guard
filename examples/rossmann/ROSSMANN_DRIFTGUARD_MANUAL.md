# DriftGuard × Rossmann Sales Prediction — Integration Manual

**Version:** 1.0  
**Date:** May 2024  
**Audience:** ML Engineers, Data Engineers, ML Ops team at Rossmann Retail GmbH

---

## Table of Contents

1. [What We Built](#1-what-we-built)
2. [Repository Layout](#2-repository-layout)
3. [The Four Drift Scenarios](#3-the-four-drift-scenarios)
4. [Expected Test Case Results](#4-expected-test-case-results)
5. [Step-by-Step Integration Guide](#5-step-by-step-integration-guide)
6. [Running DriftGuard](#6-running-driftguard)
7. [How the AI Agent Investigates](#7-how-the-ai-agent-investigates)
8. [Adapting to Real Rossmann Data](#8-adapting-to-real-rossmann-data)
9. [CI/CD Integration](#9-cicd-integration)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. What We Built

DriftGuard is a plug-in MLOps layer that sits on top of your existing
Rossmann SalesForecaster (Random Forest / LightGBM regressor) and answers
one question automatically:

> **Why has the model's accuracy degraded in production — and what should we do about it?**

For Rossmann, this means detecting when your daily sales predictions start
going wrong — whether because of a new promo campaign, a competitor opening
nearby, a broken ETL pipeline, or a holiday encoding bug — and generating a
structured Root Cause Analysis (RCA) with a concrete remediation plan,
without requiring a data scientist to manually investigate.

### What DriftGuard adds to your existing Rossmann project

```
Your existing project:              After DriftGuard integration:
─────────────────────               ─────────────────────────────────
Jupyter notebook           ──►      Same notebook (untouched)
  └─ train.py (RF model)           + DriftGuard monitoring layer
  └─ predict.py                      ├─ PSI drift detection on 22 features
  └─ feature_engineering.py          ├─ SHAP importance collapse detection
                                     ├─ RMSPE SLA monitoring
                                     ├─ AI agent root cause analysis
                                     └─ Streamlit dashboard
```

DriftGuard does **not** replace your model. It wraps around it.

---

## 2. Repository Layout

After integration your project folder should look like this:

```
Rossmann-Sales-Prediction/          ← your existing repo
│
├── Rossmann_Sales_Prediction_by_kasmin.ipynb   ← unchanged
├── requirements.txt                             ← add driftguard deps
│
├── driftguard.yaml                 ← ✨ NEW — Rossmann DriftGuard config
├── generate_scenarios.py           ← ✨ NEW — generates test scenario CSVs
│
└── examples/
    └── rossmann/
        ├── data/
        │   ├── schemas/
        │   │   └── features.json   ← feature schema + valid ranges
        │   └── scenarios/
        │       ├── promo_shift/
        │       │   ├── baseline.csv
        │       │   └── production.csv
        │       ├── competition_surge/
        │       │   ├── baseline.csv
        │       │   └── production.csv
        │       ├── etl_open_corruption/
        │       │   ├── baseline.csv
        │       │   └── production.csv
        │       └── holiday_mismatch/
        │           ├── baseline.csv
        │           └── production.csv
        ├── knowledge/
        │   ├── lineage/
        │   │   └── pipeline_lineage.md     ← ETL history + known risks
        │   ├── runbooks/
        │   │   └── drift_response.md       ← what to do for each drift type
        │   └── model_meta/
        │       └── model_card.md           ← feature importance + thresholds
        └── indexes/                        ← auto-built at first run (FAISS)
```

---

## 3. The Four Drift Scenarios

These four scenarios were designed based on real operational risks specific
to the Rossmann retail context. Each one injects a realistic real-world
failure into the production data slice.

---

### Scenario 1 — Promotional Calendar Shift (`promo_shift`)

**What happened:**
Rossmann's marketing team launched an extended Summer Sale on 2024-06-01.
The campaign was not in the model's training calendar. The `Promo=1`
proportion across all stores jumped from **~40% → ~78%** overnight.

**Why it breaks the model:**
The model learned that "Promo=1" is a meaningful signal because it appears
in only ~41% of rows. When promo is on 78% of days, the contrast disappears.
The model over-predicts on non-promo days (expecting promo-level footfall)
and under-predicts the remaining promo-day uplift.

**Drift signal:**
- PSI(Promo) ≈ **0.31** → CRITICAL (above 0.20 threshold)
- SHAP importance of `Promo` collapses by ~40%
- RMSPE degrades: **0.118 → 0.198**

**What the AI agent should find:**
The agent will compute PSI on all 22 features, identify Promo as the outlier,
cross-reference the lineage doc to confirm no ETL change occurred, then
reference the promotional calendar history to conclude it is a concept drift
event caused by a new campaign. Remediation: immediate retraining with the
Summer Sale window included.

---

### Scenario 2 — New Competitor Wave (`competition_surge`)

**What happened:**
DM Drogerie Markt opened 47 new stores near existing Rossmann clusters in
Q1 2024. The competition register in SAP was updated, which correctly flowed
through the ETL into `CompetitionDistance`. The data is correct — but it
represents an **out-of-distribution (OOD)** value regime. Training
`CompetitionDistance` median was ~820 m; production is now ~290 m for
affected stores.

**Why it breaks the model:**
The Random Forest has no tree splits in the low-distance regime (sub-300 m)
because those values almost never appeared in training. The model extrapolates
poorly, significantly under-predicting the sales impact of nearby competition.

**Drift signal:**
- PSI(CompetitionDistance) ≈ **0.41** → CRITICAL
- PSI(competition_open) ≈ **0.22** → CRITICAL (correlated feature)
- RMSPE degrades: **0.118 → 0.231**

**What the AI agent should find:**
The agent will identify CompetitionDistance and competition_open as the
drifted features. Cross-referencing the lineage doc's competition register
changelog, it will identify the DM expansion as the cause. Because the data
itself is correct (no ETL bug), remediation is retraining with the updated
competition register. Short-term: flag affected stores for manual forecast
override.

---

### Scenario 3 — ETL Open Column Corruption (`etl_open_corruption`)

**What happened:**
The data warehouse migration v4.2 on 2024-03-15 changed the `Open` column
storage type from `INT` to `BOOLEAN` in the upstream ERP (SAP Retail v4.2).
The legacy ETL adapter in `rossmann-etl` does not handle the BOOLEAN → INT
cast, so approximately **32% of `Open` values become NaN** in the feature store.

The ML serving code imputes `Open` NaN as `0` (store closed), which means
the model predicts **zero sales** for 32% of open stores.

**Why it breaks the model:**
`Open` has SHAP importance rank 12, but its effect is binary and extreme:
when `Open=0`, the model suppresses the entire sales prediction to near-zero.
Incorrectly setting `Open=0` for open stores is a catastrophic data quality
failure that propagates directly into wrong business decisions.

**Drift signal:**
- PSI(Open) ≈ **0.55** → CRITICAL (worst in all four scenarios)
- Null rate in `Open` column: **32%** (threshold: 5%)
- RMSPE degrades: **0.118 → 0.285** (worst RMSPE in all scenarios)

**What the AI agent should find:**
The agent will immediately flag PSI(Open) = 0.55 as a data quality failure
(not model drift). Cross-referencing the runbook RB-003 and the pipeline
lineage v4.2 note, it will identify the ETL BOOLEAN→INT bug as root cause.
Remediation is a 4-line ETL hotfix — not retraining.

**Important:** This is a **data pipeline bug**, not a model problem. The agent
must distinguish between these two failure modes. DriftGuard's RAG knowledge
base (the lineage doc) is what enables this distinction.

---

### Scenario 4 — Holiday Calendar Encoding Mismatch (`holiday_mismatch`)

**What happened:**
ERP v5.1 (released 2024-01-08) changed the `StateHoliday` encoding. The
original single-character codes (`'a'` = public holiday, `'b'` = Easter,
`'c'` = other) were replaced by verbose strings (`'public_holiday'`,
`'easter_holiday'`, `'other_holiday'`). The ETL categorical mapper was not
updated. It does not recognise the new strings and silently converts them
to `NaN`. Result: **all public and Easter holidays become NaN** in the
feature store, and the model predicts normal sales on days where sales
should be near zero.

**Why it breaks the model:**
`StateHoliday = 'a'` rows represent near-zero sales days. If the model does
not know a day is a public holiday, it will confidently predict a normal
sales figure — potentially 3,000–8,000 EUR per store when actual sales are
< 200 EUR. This causes massive over-prediction on public holiday dates.

**Drift signal:**
- PSI(StateHoliday) ≈ **0.28** → CRITICAL
- Null rate in `StateHoliday`: **~6%** (only 'a' and 'b' codes affected)
- RMSPE degrades: **0.118 → 0.167** (least severe, but concentrated on holidays)

**What the AI agent should find:**
The agent will flag PSI(StateHoliday) as the top drifted categorical feature.
Inspecting the null rate and value distribution, it will note that the '0'
(no holiday) class now dominates where it previously didn't. Cross-referencing
runbook RB-004 and the lineage changelog (ERP v5.1 encoding change), it will
conclude this is an ETL encoding bug. Remediation: update the holiday mapper
in `rossmann-etl` and backfill the affected date range.

---

## 4. Expected Test Case Results

When you run DriftGuard on each scenario, here is what you should see:

| Scenario | RMSPE Baseline | RMSPE Production | Primary Drifted Feature | PSI | Severity | Root Cause Type | Recommended Fix |
|----------|---------------|-----------------|------------------------|-----|----------|-----------------|-----------------|
| `promo_shift` | 0.118 | 0.198 | `Promo` | ~0.31 | ORANGE | Concept drift (new campaign) | Retrain with Summer Sale data |
| `competition_surge` | 0.118 | 0.231 | `CompetitionDistance` | ~0.41 | RED | Concept drift (OOD values) | Retrain with updated competition register; manual override for affected stores |
| `etl_open_corruption` | 0.118 | 0.285 | `Open` | ~0.55 | RED | Data pipeline bug (ETL type cast) | ETL hotfix — 4 lines of code; no retrain needed |
| `holiday_mismatch` | 0.118 | 0.167 | `StateHoliday` | ~0.28 | ORANGE | Data pipeline bug (encoding change) | ETL holiday mapper update + backfill |

### Drift vs pipeline bug distinction

A key test is whether DriftGuard's agent correctly distinguishes:
- **Concept drift** (data is correct but the world changed) → `promo_shift`, `competition_surge`
- **Data pipeline bug** (data is wrong before it even reaches the model) → `etl_open_corruption`, `holiday_mismatch`

The agent uses the RAG knowledge base (lineage + runbooks) to make this
distinction. Without the knowledge base, a naive drift detector would
recommend retraining in all four cases — which is wrong for scenarios 3 and 4.

---

## 5. Step-by-Step Integration Guide

### Prerequisites

- Python 3.11+
- The existing Rossmann notebook/scripts
- A Groq API key (free tier works for both LLMs in `driftguard.yaml`)
- Git

---

### Step 1 — Clone DriftGuard into your Rossmann project

```bash
# Option A: if your Rossmann project is standalone
cd Rossmann-Sales-Prediction/
git clone https://github.com/contactharithanimmagadda-oss/Drift-Guard.git driftguard_src

# Option B: copy just the src/ folder
cp -r driftguard_src/src ./driftguard
cp -r driftguard_src/.streamlit ./.streamlit
```

Your folder should now have both your original Rossmann files and the
DriftGuard `src/` directory side by side.

---

### Step 2 — Install dependencies

Add these to your existing `requirements.txt`:

```
# DriftGuard additions
faiss-cpu>=1.9.0
scikit-learn>=1.3.0
shap>=0.44.0
scipy>=1.11.0
numpy>=1.24.0,<2.0.0
pandas>=2.0.0
joblib>=1.3.0
sentence-transformers>=2.6.0
langchain==0.2.16
langchain-core==0.2.38
langchain-groq==0.1.9
groq>=0.9.0
streamlit>=1.35.0
mlflow>=2.14.0
python-dotenv>=1.0.0
```

Then:
```bash
pip install -r requirements.txt
```

---

### Step 3 — Place the Rossmann DriftGuard files

Copy the files from this package into your Rossmann project root:

```bash
# Config (goes in root alongside your notebook)
cp driftguard.yaml ./driftguard.yaml

# Scenario data and knowledge base
cp -r examples/rossmann/ ./examples/rossmann/

# Scenario generator (run once to regenerate CSVs if needed)
cp generate_scenarios.py ./generate_scenarios.py
```

---

### Step 4 — Set your Groq API key

```bash
# Create a .env file in the project root
echo "GROQ_API_KEY=your_key_here" > .env
```

Get a free key at https://console.groq.com

---

### Step 5 — Generate baseline data from your trained model

This is the critical step that connects DriftGuard to **your actual model**.

In your existing notebook/script, after training, export a sample of the
validation set as the DriftGuard baseline:

```python
# Add this block at the end of your feature engineering step
# (after all transformations, before model.predict())

import pandas as pd

FEATURE_COLS = [
    "Store", "DayOfWeek", "year", "month", "week", "day",
    "Customers", "Open", "Promo", "StateHoliday", "SchoolHoliday",
    "StoreType", "Assortment", "CompetitionDistance",
    "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
    "competition_open", "Promo2", "Promo2SinceWeek", "Promo2SinceYear",
    "promo_2_open", "IsPromo2Month"
]

# X_val is your validation feature dataframe (already feature-engineered)
# Export 2000 random rows as the DriftGuard baseline
baseline_df = X_val[FEATURE_COLS].sample(2000, random_state=42)
baseline_df.to_csv(
    "examples/rossmann/data/scenarios/promo_shift/baseline.csv",
    index=False
)
# Repeat for each scenario (they all share the same baseline)
for scenario in ["competition_surge", "etl_open_corruption", "holiday_mismatch"]:
    baseline_df.to_csv(
        f"examples/rossmann/data/scenarios/{scenario}/baseline.csv",
        index=False
    )

print("Baseline exported for all 4 scenarios.")
```

> **Note:** The synthetic CSVs in `examples/rossmann/data/scenarios/*/baseline.csv`
> are ready-made for demo purposes. For real monitoring, replace them with
> your actual validation data export as shown above.

---

### Step 6 — Wire in RMSPE computation (optional but recommended)

DriftGuard's default metric is accuracy (classification). For regression,
you need to supply RMSPE. Add this to your prediction pipeline:

```python
# rossmann_drift_metrics.py  — drop this file in your project root
import numpy as np

def rmspe(y_true, y_pred):
    """Root Mean Square Percentage Error — Rossmann competition metric."""
    mask = y_true != 0
    return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2))

# When you call driftguard, pass this as a custom metric:
# (exact hook depends on DriftGuard's metric_fn parameter in the scenario config)
```

In `driftguard.yaml` the `rmspe_sla` key (set to `0.14`) tells the agent
what SLA to compare against. If your model achieves a different baseline,
update this value.

---

### Step 7 — Build the RAG index

Run this once before the first DriftGuard launch. It indexes the knowledge
base documents (lineage, runbooks, model card) into a local FAISS vector store.

```bash
cd Rossmann-Sales-Prediction/
python -c "
from src.knowledge_base import build_index
build_index(
    knowledge_dirs=[
        'examples/rossmann/knowledge/lineage',
        'examples/rossmann/knowledge/runbooks',
        'examples/rossmann/knowledge/model_meta',
    ],
    index_dir='examples/rossmann/indexes'
)
print('RAG index built.')
"
```

This takes ~30 seconds on first run. The FAISS index is saved to
`examples/rossmann/indexes/` and reused on subsequent runs.

---

### Step 8 — Launch the Streamlit dashboard

```bash
cd Rossmann-Sales-Prediction/
streamlit run src/app.py -- --config driftguard.yaml
```

The Streamlit app will open at `http://localhost:8501`.

In the sidebar you will see:
- **Project:** Rossmann Retail GmbH — SalesForecaster v3.1
- **Scenario selector:** dropdown with your 4 scenarios
- **Model selector:** routing and synthesis LLM overrides
- **Run Investigation** button

---

## 6. Running DriftGuard

### Via Streamlit (recommended for demos and manual investigation)

```bash
streamlit run src/app.py -- --config driftguard.yaml
```

1. Select a scenario from the sidebar dropdown.
2. Click **Run Investigation**.
3. Watch the agent reasoning trace stream in real time.
4. Read the structured RCA at the bottom.

### Via CLI (for automated runs in CI/CD)

```bash
python -m src.cli \
  --config driftguard.yaml \
  --scenario promo_shift \
  --output reports/rossmann_rca_promo.json
```

### Via Python API (for embedding in your pipeline)

```python
from src.driftguard import DriftGuard

dg = DriftGuard.from_yaml("driftguard.yaml")
result = dg.run_scenario("promo_shift")

print(result.root_cause)      # string: what caused the drift
print(result.severity)        # "RED" / "ORANGE" / "YELLOW" / "GREEN"
print(result.remediation)     # string: what to do
print(result.psi_scores)      # dict: {feature: psi_value}
print(result.shap_collapse)   # dict: {feature: % change in |SHAP|}
```

---

## 7. How the AI Agent Investigates

The agent runs an autonomous loop (up to 15 iterations). Here is the
step-by-step reasoning path for a typical Rossmann investigation:

```
Iteration 1 : compute_psi(all features)
  → identifies top-3 drifted features by PSI score

Iteration 2 : compute_shap_delta(drifted features)
  → checks if SHAP importance collapsed or shifted

Iteration 3 : check_null_rate(drifted features)
  → determines if drift is caused by missing values (data bug) vs. value shift

Iteration 4 : rag_query("What changed in the ETL pipeline recently?")
  → retrieves relevant sections from pipeline_lineage.md

Iteration 5 : rag_query("What runbook applies to {feature} drift?")
  → retrieves the matching runbook (RB-001 through RB-004)

Iteration 6 : rag_query("What is the model's known weakness for {feature}?")
  → retrieves relevant section from model_card.md

Iteration 7–14 : additional tool calls as needed
  → compare_distributions(), plot_feature_shift(), check_value_counts()

Iteration 15 : synthesize_rca()
  → synthesis LLM (Llama 70B) writes structured JSON report
```

The **routing LLM** (Llama 8B, fast + cheap) decides which tool to call
at each step. The **synthesis LLM** (Llama 70B, high quality) only runs
once at the end to write the final RCA. This keeps cost low while keeping
output quality high.

---

## 8. Adapting to Real Rossmann Data

### Replace synthetic CSVs with real production data

Once you have your model running in production, replace the scenario CSVs
with real data slices:

```bash
# Baseline: a sample from your training/validation period (pre-drift)
# Production: a sample from the period where RMSPE degraded

# For example, for promo_shift:
# baseline.csv  → sample from Jan–May 2024 (before Summer Sale)
# production.csv → sample from Jun 2024 onwards (during Summer Sale)
```

### Automate daily data collection

Add this to your Airflow DAG or cron job:

```python
# rossmann_drift_collector.py
import pandas as pd
from datetime import date, timedelta

def collect_daily_production_slice(feature_store_path: str, output_path: str):
    """Pull last 7 days from feature store as the 'production' slice."""
    today = date.today()
    week_ago = today - timedelta(days=7)

    df = pd.read_parquet(feature_store_path)
    prod_slice = df[df["Date"] >= str(week_ago)].sample(
        min(2000, len(df)), random_state=42
    )
    prod_slice[FEATURE_COLS].to_csv(output_path, index=False)
    print(f"Collected {len(prod_slice)} rows for drift monitoring.")
```

### Add RMSPE to the monitoring config

Once you have ground-truth sales labels available (typically T+1 day),
compute RMSPE and log it to MLflow:

```python
import mlflow

with mlflow.start_run(experiment_name="Rossmann_DriftMonitoring"):
    mlflow.log_metric("rmspe", current_rmspe)
    mlflow.log_metric("psi_promo", psi_scores["Promo"])
    mlflow.log_metric("psi_competition_distance", psi_scores["CompetitionDistance"])
    # etc.
```

DriftGuard's MLflow integration picks these up automatically and displays
them on the Streamlit dashboard trend charts.

---

## 9. CI/CD Integration

Add DriftGuard as a quality gate in your GitHub Actions or GitLab CI
pipeline so that retraining PRs are automatically validated:

```yaml
# .github/workflows/drift_check.yml
name: DriftGuard Quality Gate

on:
  schedule:
    - cron: '0 6 * * *'   # daily at 06:00 UTC (after ETL completes)
  workflow_dispatch:

jobs:
  drift-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Collect production slice
        run: python rossmann_drift_collector.py
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

      - name: Run DriftGuard check
        run: |
          python -m src.cli \
            --config driftguard.yaml \
            --scenario live_production \
            --output reports/daily_rca.json \
            --fail-on-severity ORANGE
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}

      - name: Upload RCA report
        uses: actions/upload-artifact@v4
        with:
          name: daily-rca-report
          path: reports/daily_rca.json

      - name: Notify Slack on drift
        if: failure()
        run: |
          curl -X POST ${{ secrets.SLACK_WEBHOOK }} \
            -d '{"text":"⚠️ DriftGuard alert: Rossmann SalesForecaster RMSPE SLA breached. Check the RCA report."}'
```

---

## 10. Troubleshooting

### "FAISS index not found"
Run `python -c "from src.knowledge_base import build_index; build_index(...)"` as shown in Step 7.

### "PSI returns NaN for StateHoliday"
StateHoliday is categorical. Make sure DriftGuard is configured to use
categorical PSI (chi-square binning). Check `driftguard.yaml` → `data.feature_cols`
confirms StateHoliday is listed. The schema in `features.json` marks it as
`dtype: "str"` which triggers categorical PSI automatically.

### "RMSPE is not computed — only PSI"
DriftGuard needs your model's predictions to compute RMSPE. Supply predictions
by calling `dg.set_predictions(y_true, y_pred)` before `run_scenario()`, or
point the scenario config at a predictions CSV with a `predictions` column.

### "Agent runs 15 iterations but produces a generic RCA"
The RAG knowledge base is not being retrieved. Check:
1. The FAISS index exists at `examples/rossmann/indexes/`.
2. The `.md` files in `knowledge/` are not empty.
3. `sentence-transformers` is installed and the embedding model downloaded.

### "Groq rate limit errors"
The free Groq tier has rate limits. Either add a small `time.sleep(1)` between
agent iterations, or switch `routing_model` to a locally-hosted Ollama model.

---

## Scenarios Quick Reference

| ID | Label | Root cause type | Key drifted feature | RMSPE impact | Fix type |
|----|-------|-----------------|--------------------|--------------|-----------------------|
| `promo_shift` | Promotional Calendar Shift | Concept drift | `Promo` (PSI 0.31) | +68% | Retrain |
| `competition_surge` | New Competitor Wave | Concept drift (OOD) | `CompetitionDistance` (PSI 0.41) | +96% | Retrain + manual override |
| `etl_open_corruption` | ETL Open Column Bug | Data pipeline bug | `Open` (PSI 0.55) | +142% | ETL hotfix (no retrain) |
| `holiday_mismatch` | Holiday Encoding Mismatch | Data pipeline bug | `StateHoliday` (PSI 0.28) | +42% | ETL mapper update + backfill |

---

*DriftGuard × Rossmann — Integration Manual v1.0*  
*Prepared by: ML Ops Team*  
*Contact: h.nimmagadda@rossmann.de*
