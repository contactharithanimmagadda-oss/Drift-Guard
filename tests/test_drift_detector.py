"""
tests/test_drift_detector.py
Run: pytest tests/test_drift_detector.py -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from src.drift_detector import compute_psi, compute_ks, classify_psi, run_drift_scan
from src.contracts import DriftReport, FeatureDrift

BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples/steelguard/data/scenarios")


@pytest.fixture
def sensor_recal():
    b = pd.read_csv(f"{BASE}/sensor_recal/baseline.csv")
    p = pd.read_csv(f"{BASE}/sensor_recal/production.csv")
    return b, p, [c for c in b.columns if c != "defect"]


@pytest.fixture
def etl_schema():
    b = pd.read_csv(f"{BASE}/etl_schema/baseline.csv")
    p = pd.read_csv(f"{BASE}/etl_schema/production.csv")
    return b, p, [c for c in b.columns if c != "defect"]


# ---- PSI ------------------------------------------------------------------

def test_psi_high_drift():
    psi = compute_psi(np.random.normal(3.2, 0.4, 2000), np.random.normal(4.8, 0.5, 800))
    assert psi > 0.20, f"Major drift PSI should be > 0.20, got {psi:.4f}"


def test_psi_stable():
    np.random.seed(77)
    psi = compute_psi(np.random.normal(3.2, 0.4, 2000), np.random.normal(3.2, 0.4, 800))
    assert psi < 0.10, f"Stable PSI should be < 0.10, got {psi:.4f}"


def test_psi_is_float():
    psi = compute_psi(np.array([1,2,3,4,5]), np.array([1,2,3,4,5]))
    assert isinstance(psi, float)
    assert psi >= 0


def test_psi_various_bins():
    base = np.random.normal(0, 1, 1000)
    prod = np.random.normal(1, 1, 500)
    for bins in [5, 10, 20]:
        assert compute_psi(base, prod, bins=bins) > 0


# ---- KS -------------------------------------------------------------------

def test_ks_significant():
    _, p = compute_ks(np.random.normal(3.2, 0.4, 2000), np.random.normal(4.8, 0.5, 800))
    assert p < 0.001, f"Significant shift should give p < 0.001, got {p}"


def test_ks_not_significant():
    np.random.seed(42)
    _, p = compute_ks(np.random.normal(3.2, 0.4, 2000), np.random.normal(3.2, 0.4, 800))
    assert p > 0.01, f"Same distribution should give p > 0.01, got {p}"


def test_ks_returns_tuple():
    result = compute_ks(np.array([1,2,3,4,5,6]), np.array([2,3,4,5,6,7]))
    assert len(result) == 2
    assert all(isinstance(v, float) for v in result)


# ---- classify_psi ---------------------------------------------------------

def test_classify_critical():
    assert classify_psi(0.25) == "critical"
    assert classify_psi(1.0)  == "critical"


def test_classify_warn():
    assert classify_psi(0.15) == "warn"
    assert classify_psi(0.11) == "warn"


def test_classify_ok():
    assert classify_psi(0.05) == "ok"
    assert classify_psi(0.10) == "ok"   # exactly 0.10 is still ok (threshold is strictly >)


def test_classify_warn_boundary():
    assert classify_psi(0.101) == "warn"   # just above warn threshold
    assert classify_psi(0.20)  == "warn"   # exactly 0.20 is still warn (threshold is strictly >)


def test_classify_critical_boundary():
    assert classify_psi(0.201) == "critical"   # just above critical threshold


def test_psi_nan_corruption_detected():
    """NaN values injected into production should raise PSI significantly."""
    base = np.ones(1000)                              # all 1.0, no NaN
    prod = np.where(np.arange(1000) < 320, np.nan, 1.0)  # 32% NaN
    psi = compute_psi(base, prod)
    assert psi > 0.20, f"32% NaN injection should produce critical PSI, got {psi:.4f}"


# ---- run_drift_scan -------------------------------------------------------

def test_scan_returns_drift_report(sensor_recal):
    b, p, fc = sensor_recal
    r = run_drift_scan(b, p, fc, "defect", "test_sr")
    assert isinstance(r, DriftReport)
    assert r.scenario_id == "test_sr"


def test_scan_all_features_in_report(sensor_recal):
    b, p, fc = sensor_recal
    r = run_drift_scan(b, p, fc, "defect", "test")
    names = [f.name for f in r.features]
    for col in fc:
        assert col in names, f"Feature {col} missing from DriftReport"


def test_scan_sensor_recal_critical(sensor_recal):
    b, p, fc = sensor_recal
    r = run_drift_scan(b, p, fc, "defect", "sensor_recal")
    crit = [f.name for f in r.critical_features()]
    assert "surface_roughness_um" in crit, f"surface_roughness_um should be critical, got {crit}"


def test_scan_etl_schema_nan_features(etl_schema):
    b, p, fc = etl_schema
    r = run_drift_scan(b, p, fc, "defect", "etl_schema")
    crit = [f.name for f in r.critical_features()]
    assert "operator_shift" in crit, f"operator_shift must be critical in etl_schema, got {crit}"
    assert "is_night_shift" in crit, f"is_night_shift must be critical in etl_schema, got {crit}"


def test_scan_feature_drift_types(sensor_recal):
    b, p, fc = sensor_recal
    r = run_drift_scan(b, p, fc, "defect", "test")
    for fd in r.features:
        assert isinstance(fd, FeatureDrift)
        assert isinstance(fd.psi, float) and fd.psi >= 0
        assert fd.status in ("ok", "warn", "critical")


def test_scan_all_four_scenarios():
    for sc in ["sensor_recal", "concept_drift", "etl_schema", "data_volume"]:
        b  = pd.read_csv(f"{BASE}/{sc}/baseline.csv")
        p  = pd.read_csv(f"{BASE}/{sc}/production.csv")
        fc = [c for c in b.columns if c != "defect"]
        r  = run_drift_scan(b, p, fc, "defect", sc)
        drifted = [f for f in r.features if f.status != "ok"]
        assert len(drifted) >= 1, f"{sc}: expected at least 1 drifted feature"
