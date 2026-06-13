"""
src/config_loader.py
====================
DriftGuard Configuration Loader

Loads driftguard.yaml, validates required fields, resolves paths,
and returns a typed DriftGuardConfig object that every module reads from.

This is the only file that knows about driftguard.yaml.
All other modules (agent, app, tools) receive a DriftGuardConfig instance.

Usage:
    from src.config_loader import load_config
    config = load_config()                        # reads ./driftguard.yaml
    config = load_config("path/to/driftguard.yaml")

    print(config.project.name)
    print(config.data.label_col)
    print(config.agent.task_description)          # already resolved
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


# ── Sub-configs ───────────────────────────────────────────────────────────────

@dataclass
class ProjectConfig:
    name: str
    model_name: str
    model_type: str = "binary_classifier"   # binary_classifier | multiclass | regressor
    domain: str = ""
    description: str = ""


@dataclass
class ScenarioConfig:
    """
    One drift scenario (optional — used when the user wants to test
    multiple pre-defined drift windows, e.g. for demo purposes).
    If no scenarios are listed in the YAML, the UI uses baseline_path
    and production_path directly.
    """
    id: str
    label: str
    description: str = ""
    acc_baseline: float = 0.0
    acc_production: float = 0.0
    baseline_path: Optional[str] = None      # overrides data.baseline_path
    production_path: Optional[str] = None    # overrides data.production_path


@dataclass
class DataConfig:
    label_col: str
    feature_cols: List[str]
    baseline_path: str = "data/baseline.csv"
    production_path: str = "data/production.csv"
    schemas_dir: str = "data/schemas"


@dataclass
class AgentConfig:
    """
    All strings here support {project_name}, {model_name}, {label_col},
    {domain} placeholders — resolved by DriftGuardConfig.resolve_template().

    Dual-model architecture:
      routing_model   — small/fast model for ReAct tool-routing steps (50-100 tok output)
      synthesis_model — larger/capable model for retrain_advisor JSON synthesis (300-500 tok)
    Both can be overridden at runtime from the Streamlit sidebar.
    """
    role_description: str = "an ML Operations expert"
    task_description: str = (
        "Investigate why {model_name} has degraded in production "
        "and produce a structured root cause analysis with a concrete remediation plan."
    )
    investigation_hint: str = ""
    max_iterations: int = 12
    model: str = "llama-3.1-8b-instant"              # legacy — kept for backward compat
    routing_model: str = "llama-3.1-8b-instant"      # tool-routing steps (fast, cheap)
    synthesis_model: str = "llama-3.3-70b-versatile" # retrain_advisor synthesis (quality)
    max_tokens: int = 2048


@dataclass
class KnowledgeConfig:
    lineage_docs: str = "knowledge/lineage"
    runbook_docs: str = "knowledge/runbooks"
    model_docs: str = "knowledge/model_meta"
    indexes_dir: str = "indexes"


@dataclass
class MonitoringConfig:
    psi_critical: float = 0.20
    psi_warn: float = 0.10
    accuracy_sla: float = 0.88   # classifiers: breach when metric < this
    rmspe_sla: float = 0.15      # regressors:  breach when metric > this
    shap_collapse_threshold: float = -20.0


@dataclass
class MLflowConfig:
    experiment_name: str = "DriftGuard_Monitoring"
    tracking_uri: Optional[str] = None      # None = local ./mlruns


@dataclass
class UIConfig:
    page_title: str = "DriftGuard — ML Drift RCA"
    sidebar_logo: Optional[str] = None
    accent_color: str = "#3b82f6"


# ── Root config ───────────────────────────────────────────────────────────────

@dataclass
class DriftGuardConfig:
    """
    The single source of truth for a DriftGuard installation.
    Loaded once at startup, passed to every module that needs it.
    """
    project: ProjectConfig
    data: DataConfig
    scenarios: List[ScenarioConfig]
    agent: AgentConfig
    knowledge: KnowledgeConfig
    monitoring: MonitoringConfig
    mlflow: MLflowConfig
    ui: UIConfig
    root_dir: str = ""          # absolute path to the driftguard.yaml directory

    # ── Path resolution ──────────────────────────────────────────────────────

    def abs(self, rel_path: str) -> str:
        """Resolve any path in the config relative to root_dir."""
        if os.path.isabs(rel_path):
            return rel_path
        return os.path.normpath(os.path.join(self.root_dir, rel_path))

    def baseline_path(self, scenario: Optional[ScenarioConfig] = None) -> str:
        if scenario and scenario.baseline_path:
            return self.abs(scenario.baseline_path)
        return self.abs(self.data.baseline_path)

    def production_path(self, scenario: Optional[ScenarioConfig] = None) -> str:
        if scenario and scenario.production_path:
            return self.abs(scenario.production_path)
        return self.abs(self.data.production_path)

    # ── Template resolution ──────────────────────────────────────────────────

    def resolve_template(self, template: str) -> str:
        """
        Resolve {project_name}, {model_name}, {label_col}, {domain}
        placeholders in any config string.
        """
        return template.format(
            project_name=self.project.name,
            model_name=self.project.model_name,
            label_col=self.data.label_col,
            domain=self.project.domain,
        )

    @property
    def agent_role(self) -> str:
        return self.resolve_template(self.agent.role_description)

    @property
    def agent_task(self) -> str:
        return self.resolve_template(self.agent.task_description)

    @property
    def agent_hint(self) -> str:
        return self.resolve_template(self.agent.investigation_hint) if self.agent.investigation_hint else ""

    @property
    def performance_metric(self) -> str:
        """Human-readable name for the primary performance metric."""
        return "RMSPE" if self.project.model_type == "regressor" else "Accuracy"

    @property
    def performance_sla(self) -> float:
        """SLA threshold relevant to this model type."""
        return self.monitoring.rmspe_sla if self.project.model_type == "regressor" else self.monitoring.accuracy_sla

    def is_sla_breached(self, metric_value: float) -> bool:
        """True when performance is below SLA — direction-aware for classifier vs regressor."""
        if self.project.model_type == "regressor":
            return metric_value > self.monitoring.rmspe_sla   # higher RMSPE = worse
        return metric_value < self.monitoring.accuracy_sla    # lower accuracy = worse


# ── Loader ────────────────────────────────────────────────────────────────────

_REQUIRED = {
    "project": ["name", "model_name"],
    "data":    ["label_col", "feature_cols"],
}

def load_config(yaml_path: str = "driftguard.yaml") -> DriftGuardConfig:
    """
    Load and validate driftguard.yaml.

    Args:
        yaml_path: Path to driftguard.yaml (default: ./driftguard.yaml).

    Returns:
        DriftGuardConfig — fully validated, all paths resolved.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError:        If required fields are missing.
    """
    path = Path(yaml_path).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"driftguard.yaml not found at: {path}\n"
            "Copy driftguard.yaml.example → driftguard.yaml and fill in your project details."
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError("driftguard.yaml is empty.")

    # ── Validate required fields ─────────────────────────────────────────────
    for section, keys in _REQUIRED.items():
        if section not in raw:
            raise ValueError(f"driftguard.yaml is missing required section: [{section}]")
        for key in keys:
            if key not in raw[section]:
                raise ValueError(
                    f"driftguard.yaml [{section}] is missing required field: {key}"
                )

    root_dir = str(path.parent)

    # ── project ──────────────────────────────────────────────────────────────
    p = raw.get("project", {})
    project = ProjectConfig(
        name=p["name"],
        model_name=p["model_name"],
        model_type=p.get("model_type", "binary_classifier"),
        domain=p.get("domain", ""),
        description=p.get("description", ""),
    )

    # ── data ─────────────────────────────────────────────────────────────────
    d = raw.get("data", {})
    data = DataConfig(
        label_col=d["label_col"],
        feature_cols=list(d["feature_cols"]),
        baseline_path=d.get("baseline_path", "data/baseline.csv"),
        production_path=d.get("production_path", "data/production.csv"),
        schemas_dir=d.get("schemas_dir", "data/schemas"),
    )

    # ── scenarios (optional) ─────────────────────────────────────────────────
    scenarios = []
    for sc in raw.get("scenarios", []):
        # Accept either acc_* (classifiers) or rmse_* (regressors) — stored in acc_* fields
        perf_base = float(sc.get("acc_baseline") or sc.get("rmse_baseline") or 0.0)
        perf_prod = float(sc.get("acc_production") or sc.get("rmse_production") or 0.0)
        scenarios.append(ScenarioConfig(
            id=sc["id"],
            label=sc.get("label", sc["id"]),
            description=sc.get("description", ""),
            acc_baseline=perf_base,
            acc_production=perf_prod,
            baseline_path=sc.get("baseline_path"),
            production_path=sc.get("production_path"),
        ))

    # ── agent ─────────────────────────────────────────────────────────────────
    a = raw.get("agent", {})
    # routing_model falls back to legacy `model` field so old YAMLs still work
    legacy_model    = a.get("model", "llama-3.1-8b-instant")
    routing_model   = a.get("routing_model",   legacy_model)
    synthesis_model = a.get("synthesis_model", "llama-3.3-70b-versatile")
    agent = AgentConfig(
        role_description=a.get("role_description", AgentConfig.role_description),
        task_description=a.get("task_description", AgentConfig.task_description),
        investigation_hint=a.get("investigation_hint", ""),
        max_iterations=int(a.get("max_iterations", 12)),
        model=legacy_model,
        routing_model=routing_model,
        synthesis_model=synthesis_model,
        max_tokens=int(a.get("max_tokens", 2048)),
    )

    # ── knowledge ─────────────────────────────────────────────────────────────
    k = raw.get("knowledge", {})
    knowledge = KnowledgeConfig(
        lineage_docs=k.get("lineage_docs", "knowledge/lineage"),
        runbook_docs=k.get("runbook_docs", "knowledge/runbooks"),
        model_docs=k.get("model_docs", "knowledge/model_meta"),
        indexes_dir=k.get("indexes_dir", "indexes"),
    )

    # ── monitoring ────────────────────────────────────────────────────────────
    m = raw.get("monitoring", {})
    monitoring = MonitoringConfig(
        psi_critical=float(m.get("psi_critical", 0.20)),
        psi_warn=float(m.get("psi_warn", 0.10)),
        accuracy_sla=float(m.get("accuracy_sla", 0.88)),
        rmspe_sla=float(m.get("rmspe_sla", 0.15)),
        shap_collapse_threshold=float(m.get("shap_collapse_threshold", -20.0)),
    )

    # ── mlflow ────────────────────────────────────────────────────────────────
    ml = raw.get("mlflow", {})
    mlflow_cfg = MLflowConfig(
        experiment_name=ml.get("experiment_name", f"{project.name}_DriftMonitoring".replace(" ", "_")),
        tracking_uri=ml.get("tracking_uri"),
    )

    # ── ui ────────────────────────────────────────────────────────────────────
    ui = raw.get("ui", {})
    ui_cfg = UIConfig(
        page_title=ui.get("page_title", f"DriftGuard — {project.name} RCA"),
        sidebar_logo=ui.get("sidebar_logo"),
        accent_color=ui.get("accent_color", "#3b82f6"),
    )

    config = DriftGuardConfig(
        project=project,
        data=data,
        scenarios=scenarios,
        agent=agent,
        knowledge=knowledge,
        monitoring=monitoring,
        mlflow=mlflow_cfg,
        ui=ui_cfg,
        root_dir=root_dir,
    )

    print(f"[config] Loaded: {project.name} — {project.model_name}")
    print(f"[config] Features: {len(data.feature_cols)} | Label: {data.label_col}")
    print(f"[config] Scenarios: {len(scenarios)} | MLflow: {mlflow_cfg.experiment_name}")
    print(f"[config] Root dir: {root_dir}")

    return config


def discover_projects(root_dir: str) -> dict:
    """
    Scan examples/*/driftguard.yaml and return {project_name: yaml_path}.
    Used by the UI to populate the project selector without editing any file.
    """
    import glob as _glob
    projects = {}
    pattern = os.path.join(root_dir, "examples", "*", "driftguard.yaml")
    for yaml_path in sorted(_glob.glob(pattern)):
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            name = raw.get("project", {}).get("name") or os.path.basename(os.path.dirname(yaml_path))
        except Exception:
            name = os.path.basename(os.path.dirname(yaml_path))
        projects[name] = yaml_path
    return projects