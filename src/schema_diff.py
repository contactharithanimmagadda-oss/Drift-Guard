"""
src/schema_diff.py — JSON schema comparison
Person A — Day 2
"""
import json, glob
from src.contracts import SchemaDiffReport, SchemaFieldDiff


def compare_schemas(path_a: str, path_b: str) -> SchemaDiffReport:
    """Compare two schema JSON files. Breaking = dtype changes → NaN injection."""
    with open(path_a) as f:
        sa = json.load(f)
    with open(path_b) as f:
        sb = json.load(f)
    fa, fb = sa.get("fields", {}), sb.get("fields", {})
    changes, breaking = [], []
    for field in sorted(set(fa) | set(fb)):
        if field not in fa:
            changes.append(SchemaFieldDiff(field, "added", "—", str(fb[field].get("dtype","?"))))
        elif field not in fb:
            changes.append(SchemaFieldDiff(field, "removed", str(fa[field].get("dtype","?")), "—"))
            breaking.append(field)
        else:
            if fa[field].get("dtype") != fb[field].get("dtype"):
                changes.append(SchemaFieldDiff(field, "type_change",
                    fa[field].get("dtype","?"), fb[field].get("dtype","?")))
                breaking.append(field)
            elif fa[field].get("nullable") != fb[field].get("nullable"):
                changes.append(SchemaFieldDiff(field, "nullable_change",
                    str(fa[field].get("nullable")), str(fb[field].get("nullable"))))
    return SchemaDiffReport(sa.get("version","?"), sb.get("version","?"),
                            changes, len(breaking) > 0, breaking)


def format_schema_diff(report: SchemaDiffReport) -> str:
    """Agent-readable formatted diff string."""
    lines = [f"Schema diff: {report.source_a} → {report.source_b}", "=" * 55]
    if not report.changes:
        lines.append("  No changes detected.")
        return "\n".join(lines)
    for c in report.changes:
        flag = "  ⚠ BREAKING — will cause NaN in ETL pipeline" if c.field_name in report.breaking_fields else ""
        lines.append(f"  [{c.change_type.upper():<16}] {c.field_name:<28} {c.old_value} → {c.new_value}{flag}")
    lines.append("=" * 55)
    if report.has_breaking_changes:
        lines.append(f"BREAKING FIELDS: {report.breaking_fields}")
        lines.append("IMPACT: ETL receives new dtype but has no conversion → NaN injected into model input.")
        lines.append("ACTION: Deploy ETL migration adapter immediately. Map new encoding back to training values.")
    else:
        lines.append("No breaking changes. Backward-compatible schema update.")
    return "\n".join(lines)


def auto_detect_schema_diff(schemas_dir: str) -> str:
    """Find 2 most recent schema files and compare them automatically."""
    files = sorted(glob.glob(f"{schemas_dir}/*.json"))
    if len(files) < 2:
        return f"Only {len(files)} schema file(s) in {schemas_dir}. Need ≥ 2 to diff."
    return format_schema_diff(compare_schemas(files[-2], files[-1]))
