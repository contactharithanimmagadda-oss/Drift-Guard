# SteelGuard ML Model Monitoring Runbook
Version: 2.3 | Updated: 2024-02-15 | Owner: ml-ops@steelguard.de

## Overview
This runbook covers monitoring, incident response, and maintenance procedures
for the SteelGuard DefectClassifier deployed on LINE-A, LINE-B, LINE-C.

## Current Deployment
- Model: DefectClassifier v2.0 (GradientBoostingClassifier)
- Features: 12 (8 raw sensor + 4 engineered)
- Decision threshold: 0.35 (recall-optimised)
- SLA: rolling 7-day accuracy > 88%, defect recall > 75%
- MLflow tracking: http://mlflow.steelguard.internal:5000

## §1. Daily Monitoring Checklist
[ ] Check rolling 7-day accuracy on LINE-A, LINE-B, LINE-C dashboard
[ ] Review PSI scan output in Slack #ml-monitoring channel
[ ] Check for any MAINT or SCHEMA events in ETL changelog (data/knowledge/lineage/etl_changelog.txt)
[ ] Verify MLflow experiment has new runs logged

## §2. PSI Alert Thresholds
| PSI Range    | Status   | Action Required                          |
|-------------|----------|------------------------------------------|
| < 0.10      | Stable   | None                                     |
| 0.10 – 0.20 | Warning  | Increase monitoring frequency, investigate |
| > 0.20      | Critical | Immediate RCA via DriftGuard, consider rollback |

## §3. Accuracy Degradation Response
If rolling 7-day accuracy drops below 88%:
1. Immediately run DriftGuard RCA (see §6)
2. Notify ml-ops@steelguard.de and line supervisors within 30 minutes
3. Enable manual override: all predictions with confidence 0.40–0.60 go to human review
4. Do NOT retrain without confirmed root cause

## §4. Sensor Recalibration Response
When maintenance team reports sensor recalibration:
1. Check calibration offset change in MAINT-#### ticket
2. If laser offset change > 0.5µm: MANDATORY model re-evaluation
3. If ultrasonic offset change > 0.1mm: MANDATORY model re-evaluation
4. Run DriftGuard RCA targeting affected feature
5. If PSI > 0.15 on affected feature: rescale input using offset ratio
6. Retrain on minimum 500 post-calibration samples
7. Timeline: rescaling within 48h, retraining within 5 days
8. Document outcome in ETL changelog under MAINTENANCE event

## §5. Schema Change Response
When ERP or upstream system schema changes:
1. Check ETL changelog for dtype or encoding changes
2. Run DriftGuard schema_diff between current and previous schema JSON
3. Any dtype change (e.g. int→str) is BREAKING — stop inference on affected feature immediately
4. Deploy ETL migration adapter within 4 hours
5. Backfill last 48h of inference records with corrected feature values
6. Validate NaN rate on affected feature drops to < 0.1% before resuming

## §6. DriftGuard RCA Procedure
Standard RCA for any model performance incident:
1. Copy production window CSV to: data/scenarios/steelguard/production.csv
2. Open DriftGuard at http://driftguard.steelguard.internal:8501
3. Select scenario from dropdown, click "Run Agent RCA"
4. WAIT for all 7 tools to complete (45–90 seconds)
5. Review "Feature Importance" tab — identify top collapsed features
6. Review "Agent RCA" tab — read tool trace for root cause evidence
7. Follow remediation plan: Immediate → 24h → 1-week → Prevention
8. Log resolution in ETL changelog

## §7. New Product/Alloy Onboarding
MANDATORY 30 days before new alloy enters production:
1. Alert ml-ops 30 days before production start
2. Collect 500+ labeled pipe samples during pilot run
3. Evaluate current model on new samples — if recall < 80%, retrain required
4. Retrain with class-balanced dataset (minimum 20% new alloy samples)
5. Update model_card.json with new alloy in alloy_grades_in_training
6. No full deployment without QA sign-off (qa@steelguard.de)

## §8. Model Rollback Procedure
If immediate rollback required:
1. Switch serving to previous model version in MLflow registry
2. Notify all line supervisors and QA team
3. Enable 100% manual inspection for 24h
4. Root cause must be confirmed before redeployment
5. Document in MLflow run tags: {"rollback_reason":"..."}
