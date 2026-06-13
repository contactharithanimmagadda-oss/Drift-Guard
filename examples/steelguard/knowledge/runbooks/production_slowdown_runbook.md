# SteelGuard Runbook: Production Line Speed Change Response
Version: 1.0 | Created: 2024-04-20 | Owner: ml-ops@steelguard.de

## Trigger Conditions
- PSI > 0.20 simultaneously on two or more of: temp_at_weld_celsius, pressure_test_bar, cycle_time_seconds
- cycle_time_seconds mean increases (slower line = longer cycle)
- temp_at_weld_celsius mean decreases (slower line = less heat input per unit time)
- pressure_test_bar mean decreases (lower throughput = lower pressure demand)
- weld_temp_deviation mean increases sharply (thermal instability at non-standard speed)
- Lineage changelog shows INFRA event with line speed reduction

## Root Cause Explanation
A production line speed reduction changes three interdependent process variables simultaneously:
1. Weld temperatures drop — pyrometer reads lower values outside training range 1150–1210°C
2. Pressure test values drop — hydrostatic test rig operates at lower throughput
3. Cycle times increase — Siemens S7-1500 PLC timer records longer durations
4. weld_temp_deviation increases sharply — derived feature |temp - 1180| grows as temp drifts from target
The model was trained on normal-speed operations only. All four features shift outside the training
distribution simultaneously, causing compounding prediction errors on every inference.

## Immediate Actions (less than 4 hours)
1. Confirm line speed change with production engineering — check INFRA event in ETL changelog
2. Do NOT halt production — increase manual inspection sampling rate to 100% for affected lines
3. Flag all automated predictions made since speed change date for manual review
4. Enable confidence threshold override: route all predictions with confidence 0.40–0.65 to human QA
5. Notify ml-ops@steelguard.de and line supervisors within 30 minutes

## Short-term Actions (24 hours)
1. Collect minimum 200 labeled pipe samples under the reduced-speed operating conditions
2. Run DriftGuard with all 7 tools to quantify accuracy degradation under new conditions
3. Determine if speed change is temporary (maintenance window) or permanent (production change)
4. If temporary (less than 2 weeks): maintain manual inspection, no retrain required
5. If permanent: initiate retraining procedure (see medium-term section)

## Medium-term Actions (1 week)
1. Collect minimum 500 labeled samples at new line speed before retraining
2. Retrain DefectClassifier v2.0 with combined dataset: normal-speed + reduced-speed samples
3. Alternative approach: add line_speed_pct as an explicit input feature to condition model on operating mode
4. Validate retrained model: accuracy > 88% on both speed regimes before deployment
5. Update model_v2_documentation.txt training ranges for affected features
6. Log retraining run in MLflow experiment: SteelGuard_DefectClassifier

## Prevention Measures
1. Monitor PSI per operating mode separately (normal-speed baseline vs. reduced-speed baseline)
2. Add line_speed_pct as explicit feature in next model version
3. Register all planned production speed changes in ETL changelog minimum 48 hours in advance
4. Trigger automatic DriftGuard RCA when PSI > 0.20 on any process feature for more than 48 hours
5. Maintain a reduced-speed validation dataset refreshed quarterly

## Key Feature Thresholds at Reduced Speed (22% reduction observed 2024-04-20)
- temp_at_weld_celsius expected range: 1040–1080°C (training range: 1150–1210°C)
- pressure_test_bar expected range: 60–75 bar (training range: 75–95 bar)
- cycle_time_seconds expected range: 160–200 seconds (training range: 120–165 seconds)
- weld_temp_deviation expected range: 100–140 (training range: 0–60, derived from |temp - 1180|)
