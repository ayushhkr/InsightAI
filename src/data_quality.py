"""Transparent, non-mutating checks for common dataframe quality risks."""
from __future__ import annotations

import re
from typing import Any
import numpy as np
import pandas as pd


_IDENTIFIER = re.compile(r"(^|[_\s-])(id|identifier|uuid|account|customer|record)([_\s-]|$)", re.I)
_DATE_NAME = re.compile(r"date|time|timestamp|_at$", re.I)
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE = re.compile(r"^\+?[0-9][0-9() .-]{6,}[0-9]$")
_ADDRESS = re.compile(r"address|street|city|postal|zip", re.I)


def analyze_data_quality(df: pd.DataFrame, *, high_cardinality_ratio: float = 0.9) -> dict[str, Any]:
    """Inspect a dataframe and return findings; this function never changes it."""
    rows = len(df)
    missing = {str(c): int(v) for c, v in df.isna().sum().items() if v}
    duplicate_rows = int(df.duplicated().sum())
    duplicate_identifiers = _duplicate_identifiers(df)
    invalid_dates = _invalid_dates(df)
    constant_columns = [str(c) for c in df.columns if rows and df[c].nunique(dropna=False) <= 1]
    high_cardinality = _high_cardinality(df, high_cardinality_ratio)
    suspicious_numeric = _suspicious_numeric(df)
    type_inconsistencies = _type_inconsistencies(df)
    pii = detect_possible_pii(df)
    issues = []
    for label, values in (("missing_values", missing), ("duplicate_rows", duplicate_rows),
                          ("duplicate_identifiers", duplicate_identifiers), ("invalid_dates", invalid_dates),
                          ("constant_columns", constant_columns), ("high_cardinality_columns", high_cardinality),
                          ("suspicious_numeric_values", suspicious_numeric), ("type_inconsistencies", type_inconsistencies)):
        if values:
            issues.append({"type": label, "details": values})
    warnings = [f"Possible sensitive data detected in: {', '.join(pii['fields'])}." ] if pii["possible_pii"] else []
    return {
        "missing_values": missing,
        "duplicate_rows": duplicate_rows,
        "duplicate_identifiers": duplicate_identifiers,
        "invalid_dates": invalid_dates,
        "constant_columns": constant_columns,
        "high_cardinality_columns": high_cardinality,
        "suspicious_numeric_values": suspicious_numeric,
        "type_inconsistencies": type_inconsistencies,
        "pii": pii,
        "issues": issues,
        "warnings": warnings,
    }


def quality_score(df: pd.DataFrame, quality: dict[str, Any] | None = None) -> dict[str, Any]:
    """Application-specific 0-100 heuristic, with each deduction exposed."""
    quality = quality or analyze_data_quality(df)
    cells = max(int(df.shape[0] * df.shape[1]), 1)
    missing_rate = sum(quality["missing_values"].values()) / cells
    duplicate_rate = quality["duplicate_rows"] / max(len(df), 1)
    components = {
        "completeness": max(0.0, 100 - 100 * missing_rate),
        "uniqueness": max(0.0, 100 - 100 * duplicate_rate - 10 * len(quality["duplicate_identifiers"])),
        "validity": max(0.0, 100 - 15 * len(quality["invalid_dates"]) - 5 * len(quality["suspicious_numeric_values"])),
        "consistency": max(0.0, 100 - 15 * len(quality["type_inconsistencies"]) - 5 * len(quality["constant_columns"])),
    }
    overall = round(sum(components.values()) / len(components), 2)
    warnings = list(quality["warnings"])
    warnings.append("Score is an InsightAI application heuristic, not an industry-standard metric.")
    return {"overall_score": overall, "component_scores": components, "detected_issues": quality["issues"], "warnings": warnings}


def detect_possible_pii(df: pd.DataFrame) -> dict[str, Any]:
    fields: list[str] = []
    for column in df.columns:
        name = str(column)
        values = df[column].dropna().astype(str).head(100)
        named = bool(re.search(r"email|e_mail|phone|mobile|address|street|postal|zip|ssn|passport", name, re.I))
        value_match = values.map(lambda item: bool(_EMAIL.match(item) or _PHONE.match(item))).any() if len(values) else False
        identifier_like = bool(_IDENTIFIER.search(name))
        if named or value_match or identifier_like or _ADDRESS.search(name):
            fields.append(name)
    warnings = (["Possible PII is a heuristic warning only; it does not establish legal classification or compliance."] if fields else [])
    return {"possible_pii": bool(fields), "fields": fields, "warnings": warnings}


def _duplicate_identifiers(df):
    return {str(c): int(df[c].dropna().duplicated().sum()) for c in df.columns
            if _IDENTIFIER.search(str(c)) and df[c].dropna().duplicated().any()}


def _invalid_dates(df):
    findings = {}
    for c in df.columns:
        values = df[c].dropna()
        if _DATE_NAME.search(str(c)) and not pd.api.types.is_datetime64_any_dtype(values):
            parsed = pd.to_datetime(values, errors="coerce")
            count = int(parsed.isna().sum())
            if count: findings[str(c)] = count
    return findings


def _high_cardinality(df, ratio):
    findings = {}
    for c in df.select_dtypes(include=["object", "string", "category"]).columns:
        non_null = df[c].dropna()
        unique = non_null.nunique()
        if len(non_null) >= 20 and unique / len(non_null) >= ratio:
            findings[str(c)] = {"unique_values": int(unique), "non_null_values": int(len(non_null))}
    return findings


def _suspicious_numeric(df):
    findings = {}
    for c in df.select_dtypes(include="number").columns:
        series = df[c]
        infinite = int(np.isinf(series).sum())
        finite = series.replace([np.inf, -np.inf], np.nan).dropna()
        if len(finite) >= 4:
            q1, q3 = finite.quantile([0.25, 0.75]); iqr = q3 - q1
            outliers = int(((finite < q1 - 3 * iqr) | (finite > q3 + 3 * iqr)).sum()) if iqr else 0
        else: outliers = 0
        if infinite or outliers: findings[str(c)] = {"infinite_values": infinite, "extreme_outliers": outliers}
    return findings


def _type_inconsistencies(df):
    findings = {}
    for c in df.select_dtypes(include=["object", "string"]).columns:
        types = {type(value).__name__ for value in df[c].dropna()}
        if len(types) > 1: findings[str(c)] = sorted(types)
    return findings
