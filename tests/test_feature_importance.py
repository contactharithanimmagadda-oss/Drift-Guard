"""
tests/test_feature_importance.py
Run: pytest tests/test_feature_importance.py -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from src.feature_importance import (global_importance, importance_shift_ranking,
                                     permutation_importance_fallback)
from src.schema_diff import compare_schemas, format_schema_diff, auto_detect_schema_diff
from src.contracts import FeatureImportanceReport

BASE_S   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples/steelguard/data/scenarios")
BASE_SCH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples/steelguard/data/schemas")


@pytest.fixture
def etl_data():
    b  = pd.read_csv(f"{BASE_S}/etl_schema/baseline.csv")
    p  = pd.read_csv(f"{BASE_S}/etl_schema/production.csv")
    return b, p, [c for c in b.columns if c != "defect"]


@pytest.fixture
def sensor_data():
    b  = pd.read_csv(f"{BASE_S}/sensor_recal/baseline.csv")
    p  = pd.read_csv(f"{BASE_S}/sensor_recal/production.csv")
    return b, p, [c for c in b.columns if c != "defect"]


# ---- global_importance ---------------------------------------------------

def test_returns_report(sensor_data):
    b, p, fc = sensor_data
    assert isinstance(global_importance(b, p, fc, "defect", "t"), FeatureImportanceReport)


def test_all_features_present(sensor_data):
    b, p, fc = sensor_data
    r = global_importance(b, p, fc, "defect", "t")
    assert len(r.records) == len(fc)
    names = [x.name for x in r.records]
    for col in fc:
        assert col in names


def test_sorted_by_delta(sensor_data):
    b, p, fc = sensor_data
    r = global_importance(b, p, fc, "defect", "t")
    deltas = [abs(x.delta_pct) for x in r.records]
    assert deltas == sorted(deltas, reverse=True), "Records not sorted by |delta_pct|"


def test_model_type_tree(sensor_data):
    b, p, fc = sensor_data
    assert global_importance(b, p, fc, "defect", "t").model_type == "tree"


def test_etl_schema_collapses(etl_data):
    b, p, fc = etl_data
    r = global_importance(b, p, fc, "defect", "etl_schema")
    collapsed = [x.name for x in r.records if x.delta_pct < -20]
    assert len(collapsed) >= 1, f"ETL schema should have collapsed features, got {collapsed}"
    assert any(n in collapsed for n in ["operator_shift", "is_night_shift"])


def test_rank_fields_valid(sensor_data):
    b, p, fc = sensor_data
    r = global_importance(b, p, fc, "defect", "t")
    n = len(fc)
    for x in r.records:
        assert 1 <= x.rank_baseline <= n
        assert 1 <= x.rank_production <= n


# ---- importance_shift_ranking -------------------------------------------

def test_ranking_string(etl_data):
    b, p, fc = etl_data
    r = global_importance(b, p, fc, "defect", "etl")
    s = importance_shift_ranking(r)
    assert isinstance(s, str) and len(s) > 0


def test_ranking_has_collapsed(etl_data):
    b, p, fc = etl_data
    r = global_importance(b, p, fc, "defect", "etl_schema")
    assert "COLLAPSED" in importance_shift_ranking(r)


def test_ranking_contains_all_features(sensor_data):
    b, p, fc = sensor_data
    r = global_importance(b, p, fc, "defect", "t")
    s = importance_shift_ranking(r)
    for col in fc:
        assert col in s


# ---- schema_diff ---------------------------------------------------------

def test_detects_breaking():
    r = compare_schemas(f"{BASE_SCH}/schema_v1.0.json", f"{BASE_SCH}/schema_v1.1.json")
    assert r.has_breaking_changes
    assert "operator_shift" in r.breaking_fields


def test_type_change_values():
    r = compare_schemas(f"{BASE_SCH}/schema_v1.0.json", f"{BASE_SCH}/schema_v1.1.json")
    tc = next((c for c in r.changes if c.change_type == "type_change" and c.field_name == "operator_shift"), None)
    assert tc is not None
    assert tc.old_value == "int64"
    assert tc.new_value == "object"


def test_added_field():
    r = compare_schemas(f"{BASE_SCH}/schema_v1.0.json", f"{BASE_SCH}/schema_v1.1.json")
    added = [c for c in r.changes if c.change_type == "added"]
    assert len(added) >= 1


def test_identical_schema():
    r = compare_schemas(f"{BASE_SCH}/schema_v1.0.json", f"{BASE_SCH}/schema_v1.0.json")
    assert not r.has_breaking_changes
    assert len(r.changes) == 0


def test_format_contains_breaking():
    r = compare_schemas(f"{BASE_SCH}/schema_v1.0.json", f"{BASE_SCH}/schema_v1.1.json")
    assert "BREAKING" in format_schema_diff(r)


def test_auto_detect():
    result = auto_detect_schema_diff(BASE_SCH)
    assert "operator_shift" in result


# ---- permutation_fallback -----------------------------------------------

def test_permutation_structure(sensor_data):
    from sklearn.ensemble import GradientBoostingClassifier
    b, p, fc = sensor_data
    bc = b[fc+["defect"]].dropna()
    m = GradientBoostingClassifier(n_estimators=20, random_state=42).fit(bc[fc], bc["defect"])
    r = permutation_importance_fallback(m, b, p, fc, "defect", "t")
    assert r.model_type == "permutation"
    assert len(r.records) == len(fc)


def test_permutation_sorted(sensor_data):
    from sklearn.ensemble import GradientBoostingClassifier
    b, p, fc = sensor_data
    bc = b[fc+["defect"]].dropna()
    m = GradientBoostingClassifier(n_estimators=20, random_state=42).fit(bc[fc], bc["defect"])
    r = permutation_importance_fallback(m, b, p, fc, "defect", "t")
    deltas = [abs(x.delta_pct) for x in r.records]
    assert deltas == sorted(deltas, reverse=True)
