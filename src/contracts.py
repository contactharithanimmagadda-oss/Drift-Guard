"""
src/contracts.py — DriftGuard shared dataclasses
Person A writes on Day 1. FROZEN after both sign off.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FeatureDrift:
    name: str
    psi: float
    ks_stat: float
    ks_pvalue: float
    shap_delta_pct: float
    status: str           # 'ok' | 'warn' | 'critical'
    baseline_mean: float = 0.0
    production_mean: float = 0.0
    baseline_std: float = 0.0
    production_std: float = 0.0


@dataclass
class DriftReport:
    scenario_id: str
    model_accuracy_now: float
    model_accuracy_baseline: float
    features: list
    timestamp: str
    n_baseline_rows: int = 0
    n_production_rows: int = 0
    raw_context: dict = field(default_factory=dict)

    def critical_features(self):
        return [f for f in self.features if f.status == 'critical']

    def warn_features(self):
        return [f for f in self.features if f.status == 'warn']

    def summary(self) -> str:
        c = self.critical_features()
        w = self.warn_features()
        return (f"Scenario: {self.scenario_id} | Critical: {len(c)} | Warn: {len(w)} | "
                f"Accuracy: {self.model_accuracy_baseline:.1%} -> {self.model_accuracy_now:.1%}")


@dataclass
class FeatureImportanceRecord:
    name: str
    baseline_shap: float
    production_shap: float
    delta_pct: float
    rank_baseline: int
    rank_production: int
    rank_shift: int


@dataclass
class FeatureImportanceReport:
    scenario_id: str
    records: list
    top_collapsed: list
    model_type: str       # 'tree' | 'permutation'

    def get_record(self, name: str) -> Optional['FeatureImportanceRecord']:
        return next((r for r in self.records if r.name == name), None)


@dataclass
class SchemaFieldDiff:
    field_name: str
    change_type: str      # 'type_change' | 'added' | 'removed' | 'nullable_change'
    old_value: str
    new_value: str


@dataclass
class SchemaDiffReport:
    source_a: str
    source_b: str
    changes: list
    has_breaking_changes: bool
    breaking_fields: list

    def summary(self) -> str:
        if not self.changes:
            return f"No changes: {self.source_a} <-> {self.source_b}"
        return (f"{self.source_a} -> {self.source_b}: {len(self.changes)} change(s), "
                f"{'BREAKING: ' + str(self.breaking_fields) if self.has_breaking_changes else 'non-breaking'}")
