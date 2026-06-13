# Rossmann Sales Forecaster — Data Pipeline Lineage

## Overview

The SalesForecaster v3.1 model consumes data that flows through three
stages before reaching the prediction service: upstream ERP extraction,
the central ETL adapter, and the ML feature store.

---

## Stage 1 — Source ERP (SAP Retail)

**System:** SAP Retail 7.0, hosted on-premise Dortmund DC
**Owner:** Retail IT Operations, Markus Becker (markus.becker@rossmann.de)
**Extract schedule:** Daily 03:00 CET, full snapshot per store
**Tables consumed:**
- `VKDFS` — daily store sales and customer counts
- `WBHIST` — store open/closed flag and refurbishment status
- `AKTDAT` — promotional calendar (Promo, Promo2, PromoInterval)
- `WETTBEW` — competitor store register (CompetitionDistance, month/year opened)
- `FEIERT` — public and school holiday calendar by German federal state

---

## Stage 2 — ETL Adapter (rossmann-etl service)

**Repo:** gitlab.rossmann.int/data-eng/rossmann-etl
**Language:** Python 3.11, Airflow DAG `rossmann_feature_daily`
**Owner:** Data Engineering, Priya Nair (p.nair@rossmann.de)
**Runs:** Daily 04:30 CET after ERP extract completes

---

## Stage 3 — Feature Store

**Schema freeze date:** 2023-09-01 (training data cut)
**Training window:** 2021-01-01 to 2023-09-01
**Validation RMSPE:** 0.118
**Deployed:** 2023-10-05
**Last retrain:** Never (as of May 2024)

---

## Event Changelog

2024-06-01 CAMPAIGN — PROMO DISTRIBUTION SHIFT
  Rossmann launched extended Summer Sale campaign across all 1,115 stores.
  Promo=1 rate jumped from training baseline ~41% to ~78% in production window.
  The SalesForecaster v3.1 was trained on a 40% promo baseline (2021-2023).
  The current 78% promo rate is fully outside the training distribution.
  Affected feature: Promo (PSI expected > 0.30).
  ACTION REQUIRED: Emergency retrain with post-June data. Apply Promo-rate
  calibration post-processing as interim fix per Runbook RB-001.

2024-03-15 ETL BUG — OPEN COLUMN CORRUPTION
  Data warehouse migration v4.2 changed Open column from INT to BOOLEAN in ERP.
  Legacy ETL adapter does not handle BOOLEAN to INT casting.
  NaN injected into ~32% of Open values — model imputes NaN as 0 (store closed).
  Affected feature: Open (PSI expected > 0.50).
  ACTION REQUIRED: Deploy ETL hotfix per Runbook RB-003. Backfill affected dates.

2024-01-08 ETL BUG — HOLIDAY CALENDAR ENCODING MISMATCH
  ERP v5.1 changed StateHoliday encoding from single-char codes (a, b, c) to
  verbose strings (public_holiday, easter_holiday, other_holiday).
  ETL categorical mapper does not recognise new strings — coerces them to NaN.
  Model sees no public holidays, causing miscalibrated forecasts for holiday weeks.
  Affected feature: StateHoliday (PSI expected > 0.25).
  ACTION REQUIRED: Update ETL mapper per Runbook RB-004.

2024-01-01 COMPETITION — DM EXPANSION WAVE
  DM Drogerie Markt opened 47 new stores near existing Rossmann clusters in Q1 2024.
  CompetitionDistance median dropped from ~820 m to ~290 m for ~35% of stores.
  Model trained on high-distance distribution — new low-distance region is OOD.
  Affected feature: CompetitionDistance (PSI expected > 0.40).
  ACTION REQUIRED: Tag affected stores for priority retrain per Runbook RB-002.

2023-10-05 DEPLOY
  SalesForecaster v3.1 deployed to production.
  Training window: 2021-01-01 to 2023-09-01.
  Validation RMSPE: 0.118. SLA threshold: 0.15.
  Features: 22 (Store, DayOfWeek, year, month, week, day, Customers, Open,
  Promo, StateHoliday, SchoolHoliday, StoreType, Assortment, CompetitionDistance,
  CompetitionOpenSinceMonth, CompetitionOpenSinceYear, competition_open,
  Promo2, Promo2SinceWeek, Promo2SinceYear, promo_2_open, IsPromo2Month).

---

## Promotional Calendar History

Training baseline Promo rate (all stores):
  2021 full year: ~38% (training window)
  2022 full year: ~41% (training window)
  2023 full year: ~43% (training window)
  2024 Q1: ~44% (post-training — not used for training)
  2024 Q2 (Jun+): ~78% (Summer Sale — NOT in training set, causes critical drift)

---

## Competition Register

Training baseline CompetitionDistance median: ~820 m
Post-Q1 2024 CompetitionDistance median (affected stores): ~290 m
Number of affected stores: ~390 out of 1,115

---

## Known ETL Risks

- WBHIST.Open: changed from INT(1) to BOOLEAN in ERP v4.2 (Mar 2024).
  The legacy ETL adapter does not auto-cast BOOLEAN to INT.
- FEIERT.StateHoliday: encoding changed in ERP v5.1 (Jan 2024).
  Single-char codes replaced by verbose strings. ETL mapper must be updated.
