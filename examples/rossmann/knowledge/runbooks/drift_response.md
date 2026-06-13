# Drift Response Runbook — Rossmann SalesForecaster v3.1

## Severity Tiers

| Tier | RMSPE | PSI (any feature) | Action |
|------|-------|-------------------|--------|
| GREEN | < 0.15 | < 0.10 | Monitor only. Weekly review. |
| YELLOW | 0.15 – 0.18 | 0.10 – 0.20 | Investigate within 48 h. Alert data engineering. |
| ORANGE | 0.18 – 0.22 | 0.20 – 0.35 | Investigate within 4 h. Consider temporary fallback model. |
| RED | > 0.22 | > 0.35 | Immediate escalation. Rollback or freeze predictions. Notify VP Retail Analytics. |

---

## Runbook RB-001 — Promo Distribution Drift

**Trigger:** PSI(Promo) > 0.20  
**Likely cause:** New promotional campaign not reflected in training data.

**Steps:**
1. Pull current Promo rate from `AKTDAT` via ETL for last 7 days.
2. Compare against training baseline Promo rate (~41%). If delta > 15 pp, confirm campaign.
3. Check with Retail Marketing (marketing@rossmann.de) for any unannounced campaign.
4. If confirmed: apply Promo-rate calibration post-processing (scale predictions by ratio).
5. Schedule emergency retraining with data up to current week.
6. ETA to retrain: ~6 h (pipeline is automated on `rossmann-mlops/retrain` DAG).

**Expected RMSPE after fix:** Return to baseline within 2 days of retrain deploy.

---

## Runbook RB-002 — CompetitionDistance Drift

**Trigger:** PSI(CompetitionDistance) > 0.20  
**Likely cause:** New competitor store openings not in training data.

**Steps:**
1. Query `WETTBEW` for any new competitor records with `CompetitionOpenSinceYear = CURRENT_YEAR`.
2. Identify affected store IDs (stores within 500 m of new competitors).
3. For affected stores: temporarily use a competitor-unaware fallback (historic average ± promo factor).
4. Tag affected stores for priority retraining with updated competition register.
5. Notify Store Operations to flag underperforming stores for manual forecast review.

**Expected RMSPE after fix:** ~4 weeks post-retrain for new-competitor stabilisation.

---

## Runbook RB-003 — ETL Open Column Corruption

**Trigger:** PSI(Open) > 0.20 OR null rate in Open > 5%  
**Likely cause:** ERP schema change (BOOLEAN type) not handled by ETL adapter.

**Steps:**
1. Check ETL job logs: `airflow logs rossmann_feature_daily` — look for cast errors.
2. Query feature store: `SELECT COUNT(*) WHERE Open IS NULL` for last 3 days.
3. If null rate > 5%: IMMEDIATELY pause the prediction service (prevents silent bad predictions).
4. Contact Data Engineering (Priya Nair, p.nair@rossmann.de) to deploy ETL hotfix.
5. Hotfix: in `rossmann-etl/transform/open_col.py`, add `pd.to_numeric(df['Open'], errors='coerce').fillna(1)` as safe fallback.
6. Backfill missing predictions using fallback model once ETL is fixed.

**Expected RMSPE after fix:** Immediate return to baseline once null rows are removed.

---

## Runbook RB-004 — Holiday Calendar Encoding Mismatch

**Trigger:** PSI(StateHoliday) > 0.20 OR null rate in StateHoliday > 3%  
**Likely cause:** ERP StateHoliday encoding changed from single-char to verbose strings.

**Steps:**
1. Check raw ERP extract: `SELECT DISTINCT StateHoliday FROM FEIERT LIMIT 100`.
2. If values like 'public_holiday', 'easter_holiday' appear: encoding has changed.
3. Update ETL mapper in `rossmann-etl/transform/holiday_mapper.py`:
   ```python
   HOLIDAY_MAP = {
       '0': '0', 'a': 'a', 'b': 'b', 'c': 'c',
       'public_holiday': 'a',
       'easter_holiday': 'b',
       'other_holiday': 'c',
   }
   ```
4. Re-run ETL for the affected date range to backfill correct values.
5. Validate: null rate in StateHoliday should return to < 0.5%.

**Expected RMSPE after fix:** Return to baseline within 1 day of backfill.

---

## Escalation Contacts

| Role | Name | Contact |
|------|------|---------|
| ML Ops Lead | Haritha Nimmagadda | h.nimmagadda@rossmann.de |
| Data Engineering Lead | Priya Nair | p.nair@rossmann.de |
| Retail IT Operations | Markus Becker | markus.becker@rossmann.de |
| VP Retail Analytics | Stefan Hoffmann | s.hoffmann@rossmann.de |
| On-call PagerDuty | — | rossmann-mlops PD rotation |
