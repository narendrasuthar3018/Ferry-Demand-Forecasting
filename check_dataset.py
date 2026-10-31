from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
LOG_DIR = OUTPUT_DIR / "logs"

DATA_PATH = DATA_DIR / "Toronto Island Ferry Tickets (1).csv"

JSON_REPORT_PATH = REPORT_DIR / "dataset_validation_report.json"
ISSUES_REPORT_PATH = REPORT_DIR / "dataset_validation_issues.csv"
COLUMN_REPORT_PATH = REPORT_DIR / "dataset_column_profile.csv"
LOG_PATH = LOG_DIR / "check_dataset.log"

REQUIRED_COLUMNS = [
    "_id",
    "Timestamp",
    "Redemption Count",
    "Sales Count",
]

COUNT_COLUMNS = [
    "Redemption Count",
    "Sales Count",
]

EXPECTED_INTERVAL_MINUTES = 15

QUALITY_WEIGHTS = {
    "schema": 20,
    "timestamps": 25,
    "numeric_values": 25,
    "duplicates": 15,
    "completeness": 15,
}


# ============================================================
# 2. LOGGING
# ============================================================

def setup_logging() -> None:
    """Create output folders and configure application logging."""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


# ============================================================
# 3. JSON-SAFE CONVERSION
# ============================================================

def json_safe(value: Any) -> Any:
    """Convert common pandas and NumPy values into JSON-safe values."""

    if value is None or value is pd.NA or value is pd.NaT:
        return None

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)

    if isinstance(value, float):
        return value if np.isfinite(value) else None

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())

    if isinstance(value, pd.Timedelta):
        return str(value)

    return value


# ============================================================
# 4. STRUCTURED ISSUE REGISTRY
# ============================================================

class IssueRegistry:
    """Store data-quality findings without modifying the source data."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def add(
        self,
        severity: str,
        category: str,
        issue: str,
        affected_rows: int = 0,
        details: str = "",
    ) -> None:
        severity = severity.upper()

        record = {
            "Severity": severity,
            "Category": category,
            "Issue": issue,
            "Affected_Rows": int(affected_rows),
            "Details": details,
        }

        self.records.append(record)

        log_methods = {
            "CRITICAL": logging.critical,
            "ERROR": logging.error,
            "WARNING": logging.warning,
            "INFO": logging.info,
        }

        log_method = log_methods.get(severity, logging.info)

        log_method(
            "%s | %s | affected rows=%s | %s",
            category,
            issue,
            f"{affected_rows:,}",
            details,
        )

    def to_dataframe(self) -> pd.DataFrame:
        columns = [
            "Severity",
            "Category",
            "Issue",
            "Affected_Rows",
            "Details",
        ]

        return pd.DataFrame(self.records, columns=columns)

    def count_by_severity(self) -> dict[str, int]:
        counts = {
            "CRITICAL": 0,
            "ERROR": 0,
            "WARNING": 0,
            "INFO": 0,
        }

        for record in self.records:
            severity = record["Severity"]

            if severity in counts:
                counts[severity] += 1

        return counts


# ============================================================
# 5. DATASET LOADING
# ============================================================

def load_dataset() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the original CSV without silently dropping records."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    if not DATA_PATH.is_file():
        raise ValueError(
            f"Dataset path is not a file: {DATA_PATH}"
        )

    file_size = DATA_PATH.stat().st_size

    if file_size == 0:
        raise ValueError("Dataset file is empty.")

    logging.info("Reading dataset: %s", DATA_PATH)
    logging.info("File size: %s bytes", f"{file_size:,}")

    started = time.perf_counter()

    df = pd.read_csv(
        DATA_PATH,
        low_memory=False,
    )

    load_seconds = round(
        time.perf_counter() - started,
        3,
    )

    if df.empty:
        raise ValueError(
            "CSV contains no data rows."
        )

    metadata = {
        "file_name": DATA_PATH.name,
        "file_path": str(DATA_PATH),
        "file_size_bytes": int(file_size),
        "file_size_mb": round(file_size / (1024 ** 2), 3),
        "loaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "load_duration_seconds": load_seconds,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
    }

    logging.info(
        "Loaded %s rows and %s columns in %.3f seconds.",
        f"{len(df):,}",
        len(df.columns),
        load_seconds,
    )

    return df, metadata


# ============================================================
# 6. SCHEMA VALIDATION
# ============================================================

def validate_schema(
    df: pd.DataFrame,
    issues: IssueRegistry,
) -> dict[str, Any]:
    """Validate required columns and conflicting column headers."""

    columns = list(df.columns)

    normalized = [
        str(column).strip().casefold()
        for column in columns
    ]

    duplicate_mask = pd.Index(normalized).duplicated(keep=False)

    duplicate_headers = [
        str(columns[index])
        for index, is_duplicate in enumerate(duplicate_mask)
        if is_duplicate
    ]

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in columns
    ]

    unexpected_columns = [
        str(column)
        for column in columns
        if column not in REQUIRED_COLUMNS
    ]

    schema_valid = not missing_columns and not duplicate_headers

    if missing_columns:
        issues.add(
            "CRITICAL",
            "SCHEMA",
            "Required columns are missing",
            len(missing_columns),
            ", ".join(missing_columns),
        )

    if duplicate_headers:
        issues.add(
            "CRITICAL",
            "SCHEMA",
            "Duplicate or conflicting column headers",
            len(duplicate_headers),
            ", ".join(duplicate_headers),
        )

    if unexpected_columns:
        issues.add(
            "INFO",
            "SCHEMA",
            "Additional columns detected",
            len(unexpected_columns),
            ", ".join(unexpected_columns),
        )

    if schema_valid:
        issues.add(
            "INFO",
            "SCHEMA",
            "Required schema checks passed",
            details="All required columns are present.",
        )

    return {
        "required_columns": REQUIRED_COLUMNS,
        "actual_columns": [str(column) for column in columns],
        "missing_required_columns": missing_columns,
        "unexpected_columns": unexpected_columns,
        "duplicate_or_conflicting_headers": duplicate_headers,
        "schema_valid": schema_valid,
    }


# ============================================================
# 7. COMPLETENESS AND DUPLICATE VALIDATION
# ============================================================

def validate_completeness_and_duplicates(
    df: pd.DataFrame,
    issues: IssueRegistry,
) -> dict[str, Any]:
    """Measure missing values and duplicates without deleting records."""

    missing_by_column = {
        str(column): int(count)
        for column, count in df.isna().sum().items()
    }

    missing_cells = int(df.isna().sum().sum())
    total_cells = int(df.shape[0] * df.shape[1])

    completeness = (
        100.0 * (total_cells - missing_cells) / total_cells
        if total_cells
        else 0.0
    )

    missing_rows = int(df.isna().any(axis=1).sum())

    if missing_cells:
        issues.add(
            "WARNING",
            "COMPLETENESS",
            "Missing values detected",
            missing_rows,
            f"Missing cells={missing_cells:,}.",
        )
    else:
        issues.add(
            "INFO",
            "COMPLETENESS",
            "No missing cells detected",
        )

    duplicate_rows = int(df.duplicated().sum())

    if duplicate_rows:
        issues.add(
            "WARNING",
            "DUPLICATES",
            "Exact duplicate rows detected",
            duplicate_rows,
            "No records were deleted.",
        )

    duplicate_ids: int | None = None
    missing_ids: int | None = None

    if "_id" in df.columns:
        ids = df["_id"].dropna()

        duplicate_ids = int(ids.duplicated().sum())
        missing_ids = int(df["_id"].isna().sum())

        if duplicate_ids:
            issues.add(
                "WARNING",
                "DUPLICATES",
                "Duplicate record IDs detected",
                duplicate_ids,
                "Repeated IDs were reported, not removed.",
            )

        if missing_ids:
            issues.add(
                "WARNING",
                "COMPLETENESS",
                "Missing record IDs detected",
                missing_ids,
            )

    return {
        "total_rows": int(len(df)),
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "missing_rows": missing_rows,
        "missing_values_by_column": missing_by_column,
        "completeness_percent": round(completeness, 4),
        "exact_duplicate_rows": duplicate_rows,
        "duplicate_ids": duplicate_ids,
        "missing_ids": missing_ids,
    }


# ============================================================
# 8. TIMESTAMP VALIDATION
# ============================================================

def validate_timestamps(
    df: pd.DataFrame,
    issues: IssueRegistry,
) -> dict[str, Any]:
    """Validate timestamp parsing, order, duplicates and interval gaps."""

    if "Timestamp" not in df.columns:
        issues.add(
            "CRITICAL",
            "TIMESTAMP",
            "Timestamp column is unavailable",
            len(df),
        )

        return {
            "timestamp_column_available": False,
            "invalid_timestamps": None,
            "valid_timestamps": 0,
            "duplicate_valid_timestamps": None,
            "chronological_order": "UNAVAILABLE",
            "chronological_order_violations": None,
            "estimated_missing_15_minute_slots": None,
        }

    parsed = pd.to_datetime(
        df["Timestamp"],
        errors="coerce",
    )

    invalid_count = int(parsed.isna().sum())
    valid = parsed.dropna()

    if invalid_count:
        issues.add(
            "ERROR",
            "TIMESTAMP",
            "Invalid or missing timestamps",
            invalid_count,
            "Unparseable timestamps are not silently repaired.",
        )

    if valid.empty:
        issues.add(
            "CRITICAL",
            "TIMESTAMP",
            "No valid timestamps available",
            len(df),
        )

        return {
            "timestamp_column_available": True,
            "invalid_timestamps": invalid_count,
            "valid_timestamps": 0,
            "duplicate_valid_timestamps": 0,
            "earliest_timestamp": None,
            "latest_timestamp": None,
            "chronological_order": "UNAVAILABLE",
            "chronological_order_violations": None,
            "expected_interval_minutes": EXPECTED_INTERVAL_MINUTES,
            "estimated_missing_15_minute_slots": None,
            "interval_distribution": {},
            "shorter_than_expected_intervals": 0,
            "longer_than_expected_intervals": 0,
        }

    duplicate_count = int(valid.duplicated().sum())

    if duplicate_count:
        issues.add(
            "WARNING",
            "TIMESTAMP",
            "Duplicate valid timestamps detected",
            duplicate_count,
            "Multiple records may legitimately share a timestamp.",
        )

    # Inspect the original sequence before sorting.
    valid_in_source_order = valid.reset_index(drop=True)
    source_differences = valid_in_source_order.diff().dropna()

    positive_steps = int(
        (source_differences > pd.Timedelta(0)).sum()
    )
    negative_steps = int(
        (source_differences < pd.Timedelta(0)).sum()
    )
    zero_steps = int(
        (source_differences == pd.Timedelta(0)).sum()
    )

    if len(source_differences) == 0:
        order_label = "INSUFFICIENT_DATA"
        order_violations = 0

    elif negative_steps == 0 and zero_steps == 0:
        order_label = "ASCENDING"
        order_violations = 0

    elif positive_steps == 0 and zero_steps == 0:
        order_label = "DESCENDING"
        order_violations = 0

    else:
        # Count only backwards transitions. A consistently descending
        # dataset is classified as descending, not randomly disordered.
        order_violations = negative_steps

        if negative_steps == 0:
            order_label = "NON_DECREASING"
        elif positive_steps == 0:
            order_label = "NON_INCREASING"
        else:
            order_label = "MIXED_OR_DISORDERED"

    if order_label == "DESCENDING":
        issues.add(
            "INFO",
            "TIMESTAMP",
            "Dataset is consistently in descending timestamp order",
            len(valid),
            "Chronological sorting is required before time-series modelling.",
        )

    elif order_violations > 0:
        issues.add(
            "WARNING",
            "TIMESTAMP",
            "Mixed chronological ordering detected",
            order_violations,
            "Sort timestamps before downstream time-series processing.",
        )

    elif order_label == "ASCENDING":
        issues.add(
            "INFO",
            "TIMESTAMP",
            "Timestamps are in ascending order",
        )

    # Sort timestamps for interval statistics only.
    ordered = valid.sort_values().reset_index(drop=True)
    gaps = ordered.diff().dropna()

    expected = pd.Timedelta(minutes=EXPECTED_INTERVAL_MINUTES)

    shorter_count = int(
        ((gaps > pd.Timedelta(0)) & (gaps < expected)).sum()
    )
    longer_count = int((gaps > expected).sum())
    zero_interval_count = int((gaps == pd.Timedelta(0)).sum())

    interval_counts = gaps.value_counts().head(20)

    interval_distribution = {
        str(interval): int(count)
        for interval, count in interval_counts.items()
    }

    long_gaps = gaps[gaps > expected]

    estimated_missing_slots = int(
        sum(
            max(int(round(gap / expected)) - 1, 0)
            for gap in long_gaps
        )
    )

    if longer_count:
        issues.add(
            "WARNING",
            "TIMESTAMP",
            "Intervals longer than expected cadence detected",
            longer_count,
            (
                f"Expected cadence={EXPECTED_INTERVAL_MINUTES} minutes; "
                f"estimated missing slots={estimated_missing_slots:,}. "
                "No synthetic records were created."
            ),
        )

    if shorter_count:
        issues.add(
            "INFO",
            "TIMESTAMP",
            "Intervals shorter than expected cadence detected",
            shorter_count,
        )

    # Estimate how many observed intervals deviate from the expected
    # cadence. This is used in the diagnostic quality score.
    interval_count = len(gaps)
    cadence_deviation_count = int(
        (gaps != expected).sum()
    )

    cadence_deviation_percent = (
        100.0 * cadence_deviation_count / interval_count
        if interval_count
        else 0.0
    )

    return {
        "timestamp_column_available": True,
        "invalid_timestamps": invalid_count,
        "valid_timestamps": int(len(valid)),
        "duplicate_valid_timestamps": duplicate_count,
        "earliest_timestamp": valid.min(),
        "latest_timestamp": valid.max(),
        "chronological_order": order_label,
        "chronological_order_violations": order_violations,
        "source_positive_steps": positive_steps,
        "source_negative_steps": negative_steps,
        "source_zero_steps": zero_steps,
        "expected_interval_minutes": EXPECTED_INTERVAL_MINUTES,
        "shorter_than_expected_intervals": shorter_count,
        "longer_than_expected_intervals": longer_count,
        "zero_intervals_after_sorting": zero_interval_count,
        "cadence_deviation_count": cadence_deviation_count,
        "cadence_deviation_percent": round(
            cadence_deviation_percent, 4
        ),
        "estimated_missing_15_minute_slots": estimated_missing_slots,
        "interval_distribution": interval_distribution,
    }


# ============================================================
# 9. TICKET COUNT VALIDATION
# ============================================================

def validate_ticket_counts(
    df: pd.DataFrame,
    issues: IssueRegistry,
) -> dict[str, Any]:
    """Check ticket-count fields for invalid, missing and negative values."""

    results: dict[str, Any] = {}

    for column in COUNT_COLUMNS:
        if column not in df.columns:
            issues.add(
                "CRITICAL",
                "NUMERIC",
                "Required ticket-count column is missing",
                len(df),
                column,
            )
            results[column] = {"column_available": False}
            continue

        raw = df[column]
        numeric = pd.to_numeric(raw, errors="coerce")

        invalid_count = int((raw.notna() & numeric.isna()).sum())
        missing_count = int(raw.isna().sum())

        numeric_array = numeric.dropna().to_numpy(dtype=float)
        finite_mask = np.isfinite(numeric_array)

        infinite_count = int((~finite_mask).sum())
        finite_values = numeric_array[finite_mask]

        negative_count = int((finite_values < 0).sum())
        fractional_count = int(
            (np.abs(finite_values - np.round(finite_values)) > 1e-9).sum()
        )
        zero_count = int((finite_values == 0).sum())

        if invalid_count:
            issues.add(
                "ERROR",
                "NUMERIC",
                "Non-numeric ticket counts detected",
                invalid_count,
                column,
            )

        if missing_count:
            issues.add(
                "WARNING",
                "NUMERIC",
                "Missing ticket counts detected",
                missing_count,
                column,
            )

        if infinite_count:
            issues.add(
                "ERROR",
                "NUMERIC",
                "Infinite ticket-count values detected",
                infinite_count,
                column,
            )

        if negative_count:
            issues.add(
                "ERROR",
                "NUMERIC",
                "Negative ticket counts detected",
                negative_count,
                column,
            )

        if fractional_count:
            issues.add(
                "WARNING",
                "NUMERIC",
                "Fractional ticket counts detected",
                fractional_count,
                "Confirm the source definition before coercing counts.",
            )

        results[column] = {
            "column_available": True,
            "source_dtype": str(df[column].dtype),
            "missing_values": missing_count,
            "invalid_non_numeric_values": invalid_count,
            "infinite_values": infinite_count,
            "negative_values": negative_count,
            "fractional_values": fractional_count,
            "zero_values": zero_count,
            "minimum": float(finite_values.min()) if len(finite_values) else None,
            "maximum": float(finite_values.max()) if len(finite_values) else None,
            "mean": float(finite_values.mean()) if len(finite_values) else None,
            "median": float(np.median(finite_values)) if len(finite_values) else None,
            "standard_deviation": (
                float(finite_values.std(ddof=1))
                if len(finite_values) > 1
                else None
            ),
            "valid_numeric_values": int(len(finite_values)),
        }

    return results


# ============================================================
# 10. COLUMN PROFILE
# ============================================================

def build_column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a machine-readable profile for each original column."""

    records: list[dict[str, Any]] = []
    total_rows = len(df)

    for column in df.columns:
        series = df[column]
        nonmissing = series.dropna()

        record: dict[str, Any] = {
            "Column": str(column),
            "Data_Type": str(series.dtype),
            "Total_Rows": total_rows,
            "Non_Missing": int(series.notna().sum()),
            "Missing": int(series.isna().sum()),
            "Missing_Percent": round(
                100.0 * series.isna().mean(), 4
            ),
            "Unique_Non_Missing": int(series.nunique(dropna=True)),
            "Minimum": None,
            "Maximum": None,
            "Mean": None,
            "Median": None,
            "Most_Common_Value": None,
        }

        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            finite = numeric[
                np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))
            ]

            if not finite.empty:
                record.update({
                    "Minimum": float(finite.min()),
                    "Maximum": float(finite.max()),
                    "Mean": float(finite.mean()),
                    "Median": float(finite.median()),
                })

        elif not nonmissing.empty:
            mode = nonmissing.mode(dropna=True)

            if not mode.empty:
                record["Most_Common_Value"] = str(mode.iloc[0])

        records.append(record)

    return pd.DataFrame(records)


# ============================================================
# 11. DATA-QUALITY SCORE
# ============================================================

def calculate_quality_score(
    schema_report: dict[str, Any],
    timestamp_report: dict[str, Any],
    numeric_report: dict[str, Any],
    completeness_report: dict[str, Any],
) -> dict[str, Any]:
    """
    Calculate a transparent diagnostic score.

    This project-defined heuristic is not a calibrated probability.
    Review the individual issues before using the dataset for modelling.
    """

    components: dict[str, float] = {}

    components["schema"] = (
        100.0 if schema_report.get("schema_valid") else 0.0
    )

    valid_timestamps = int(
        timestamp_report.get("valid_timestamps", 0) or 0
    )
    invalid_timestamps = int(
        timestamp_report.get("invalid_timestamps", 0) or 0
    )

    timestamp_total = valid_timestamps + invalid_timestamps

    timestamp_validity = (
        100.0 * valid_timestamps / timestamp_total
        if timestamp_total
        else 0.0
    )

    duplicate_timestamps = int(
        timestamp_report.get("duplicate_valid_timestamps", 0) or 0
    )

    duplicate_timestamp_rate = (
        100.0 * duplicate_timestamps / max(valid_timestamps, 1)
    )

    cadence_deviation = float(
        timestamp_report.get("cadence_deviation_percent", 0.0) or 0.0
    )

    # Irregular intervals reduce the score, but do not automatically
    # invalidate the data: irregular intervals may be legitimate.
    timestamp_score = (
        timestamp_validity
        - duplicate_timestamp_rate
        - 0.25 * cadence_deviation
    )

    order_label = timestamp_report.get("chronological_order")

    if order_label == "MIXED_OR_DISORDERED":
        timestamp_score -= 10.0

    components["timestamps"] = max(
        0.0, min(100.0, timestamp_score)
    )

    numeric_denominator = 0
    numeric_issue_count = 0

    for result in numeric_report.values():
        if not result.get("column_available"):
            numeric_denominator += 1
            numeric_issue_count += 1
            continue

        numeric_denominator += (
            result.get("valid_numeric_values", 0)
            + result.get("invalid_non_numeric_values", 0)
            + result.get("missing_values", 0)
            + result.get("infinite_values", 0)
        )

        numeric_issue_count += (
            result.get("invalid_non_numeric_values", 0)
            + result.get("infinite_values", 0)
            + result.get("negative_values", 0)
        )

    components["numeric_values"] = max(
        0.0,
        100.0 * (
            1.0
            - numeric_issue_count / max(numeric_denominator, 1)
        ),
    )

    total_rows = max(
        int(completeness_report.get("total_rows", 0) or 0),
        1,
    )

    duplicate_rows = int(
        completeness_report.get("exact_duplicate_rows", 0) or 0
    )

    duplicate_ids = completeness_report.get("duplicate_ids")
    missing_ids = completeness_report.get("missing_ids")

    duplicate_rate = 100.0 * duplicate_rows / total_rows

    if duplicate_ids is not None:
        duplicate_rate += (
            100.0 * int(duplicate_ids) / total_rows
        )

    if missing_ids is not None:
        duplicate_rate += (
            100.0 * int(missing_ids) / total_rows
        )

    components["duplicates"] = max(
        0.0, 100.0 - duplicate_rate
    )

    components["completeness"] = float(
        completeness_report.get("completeness_percent", 0.0)
    )

    total_weight = sum(QUALITY_WEIGHTS.values())

    score = sum(
        components[name] * QUALITY_WEIGHTS[name]
        for name in QUALITY_WEIGHTS
    ) / total_weight

    return {
        "score": round(max(0.0, min(100.0, score)), 2),
        "components": {
            name: round(value, 2)
            for name, value in components.items()
        },
        "weights": QUALITY_WEIGHTS,
        "interpretation": (
            "Project-defined diagnostic heuristic, not a probability "
            "or a substitute for reviewing individual data-quality issues."
        ),
    }


# ============================================================
# 12. REPORT GENERATION
# ============================================================

def save_reports(
    metadata: dict[str, Any],
    schema_report: dict[str, Any],
    completeness_report: dict[str, Any],
    timestamp_report: dict[str, Any],
    numeric_report: dict[str, Any],
    profile: pd.DataFrame,
    issues: IssueRegistry,
) -> dict[str, Any]:
    """Save the complete JSON report and supporting CSV reports."""

    severity_counts = issues.count_by_severity()

    quality_report = calculate_quality_score(
        schema_report,
        timestamp_report,
        numeric_report,
        completeness_report,
    )

    schema_failed = not schema_report.get("schema_valid", False)

    timestamp_failed = (
        not timestamp_report.get("timestamp_column_available", False)
        or int(timestamp_report.get("valid_timestamps", 0) or 0) == 0
    )

    missing_count_columns = [
        column
        for column in COUNT_COLUMNS
        if not numeric_report.get(column, {}).get("column_available", False)
    ]

    numeric_errors = any(
        (
            result.get("invalid_non_numeric_values", 0)
            or result.get("infinite_values", 0)
            or result.get("negative_values", 0)
        )
        for result in numeric_report.values()
    )

    if schema_failed or timestamp_failed or missing_count_columns:
        status = "FAIL"
    elif numeric_errors or severity_counts["ERROR"] > 0:
        status = "REVIEW_REQUIRED"
    elif severity_counts["WARNING"] > 0:
        status = "PASS_WITH_WARNINGS"
    else:
        status = "PASS"

    report = {
        "report_name": "Ferry Ticket Dataset Validation",
        "report_version": "2.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_status": status,
        "source": metadata,
        "schema": schema_report,
        "completeness_and_duplicates": completeness_report,
        "timestamps": timestamp_report,
        "ticket_counts": numeric_report,
        "data_quality": quality_report,
        "issue_summary": {
            "total_issues": len(issues.records),
            "by_severity": severity_counts,
        },
        "artifacts": {
            "json_report": str(JSON_REPORT_PATH),
            "issues_csv": str(ISSUES_REPORT_PATH),
            "column_profile_csv": str(COLUMN_REPORT_PATH),
            "log_file": str(LOG_PATH),
        },
        "limitations": [
            "The original source CSV is never modified by this script.",
            "Missing timestamp estimates do not prove zero demand.",
            "Irregular timestamps are reported, not automatically filled.",
            "Duplicate timestamps are not automatically treated as duplicate transactions.",
            "The quality score is a project-defined diagnostic heuristic.",
            "Timestamp sorting is required before downstream time-series modelling.",
        ],
    }

    profile.to_csv(COLUMN_REPORT_PATH, index=False)
    issues.to_dataframe().to_csv(ISSUES_REPORT_PATH, index=False)

    with open(JSON_REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(
            json_safe(report),
            file,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        )

    return report


# ============================================================
# 13. CONSOLE SUMMARY
# ============================================================

def print_summary(
    report: dict[str, Any],
    issues: IssueRegistry,
) -> None:
    """Print an operational summary of the validation run."""

    metadata = report["source"]
    timestamps = report["timestamps"]
    completeness = report["completeness_and_duplicates"]
    quality = report["data_quality"]

    print("\n" + "=" * 76)
    print("FERRY TICKET DATASET — VALIDATION SUMMARY")
    print("=" * 76)

    print(f"Dataset: {metadata['file_name']}")
    print(f"Rows: {metadata['rows']:,}")
    print(f"Columns: {metadata['columns']:,}")
    print(f"File size: {metadata['file_size_mb']:.2f} MB")

    print("\nVALIDATION STATUS")
    print("-" * 76)
    print(f"Status: {report['validation_status']}")
    print(f"Diagnostic quality score: {quality['score']:.2f}/100")

    print("\nDATA COMPLETENESS")
    print("-" * 76)
    print(f"Completeness: {completeness['completeness_percent']:.2f}%")
    print(f"Missing cells: {completeness['missing_cells']:,}")
    print(f"Exact duplicate rows: {completeness['exact_duplicate_rows']:,}")
    print(f"Duplicate IDs: {completeness['duplicate_ids']}")

    print("\nTIMESTAMP QUALITY")
    print("-" * 76)
    print(f"Invalid timestamps: {timestamps.get('invalid_timestamps')}")
    print(f"Duplicate timestamps: {timestamps.get('duplicate_valid_timestamps')}")
    print(f"Source timestamp order: {timestamps.get('chronological_order')}")
    print(
        "Ordering violations: "
        f"{timestamps.get('chronological_order_violations')}"
    )
    print(
        "Longer-than-expected intervals: "
        f"{timestamps.get('longer_than_expected_intervals')}"
    )
    print(
        "Cadence deviation: "
        f"{timestamps.get('cadence_deviation_percent', 0):.2f}%"
    )
    print(
        "Estimated missing 15-minute slots: "
        f"{timestamps.get('estimated_missing_15_minute_slots')}"
    )

    print("\nISSUE COUNTS")
    print("-" * 76)

    for severity, count in issues.count_by_severity().items():
        print(f"{severity}: {count:,}")

    issue_df = issues.to_dataframe()

    if not issue_df.empty:
        print("\nREPORTED FINDINGS")
        print("-" * 76)
        print(
            issue_df[
                ["Severity", "Category", "Issue", "Affected_Rows"]
            ].to_string(index=False)
        )

    print("\nGENERATED REPORTS")
    print("-" * 76)
    print(f"JSON report: {JSON_REPORT_PATH}")
    print(f"Issues CSV: {ISSUES_REPORT_PATH}")
    print(f"Column profile: {COLUMN_REPORT_PATH}")
    print(f"Log file: {LOG_PATH}")

    print("\n" + "=" * 76)
    print("VALIDATION PROCESS FINISHED")
    print("=" * 76)


# ============================================================
# 14. MAIN PIPELINE
# ============================================================

def main() -> int:
    """Execute the complete dataset validation pipeline."""

    started = time.perf_counter()
    issues = IssueRegistry()

    try:
        setup_logging()

        logging.info("=" * 76)
        logging.info("FERRY TICKET DATASET VALIDATION STARTED")
        logging.info("=" * 76)

        df, metadata = load_dataset()

        schema_report = validate_schema(df, issues)

        completeness_report = validate_completeness_and_duplicates(
            df, issues
        )

        timestamp_report = validate_timestamps(df, issues)

        numeric_report = validate_ticket_counts(df, issues)

        profile = build_column_profile(df)

        report = save_reports(
            metadata=metadata,
            schema_report=schema_report,
            completeness_report=completeness_report,
            timestamp_report=timestamp_report,
            numeric_report=numeric_report,
            profile=profile,
            issues=issues,
        )

        elapsed = round(time.perf_counter() - started, 3)
        report["execution_duration_seconds"] = elapsed

        # Save the final report again to include total execution duration.
        with open(JSON_REPORT_PATH, "w", encoding="utf-8") as file:
            json.dump(
                json_safe(report),
                file,
                indent=4,
                ensure_ascii=False,
                allow_nan=False,
            )

        print_summary(report, issues)

        logging.info("Execution duration: %.3f seconds.", elapsed)
        logging.info("Final validation status: %s", report["validation_status"])

        if report["validation_status"] == "FAIL":
            return 1

        return 0

    except (
        FileNotFoundError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
        PermissionError,
        OSError,
        ValueError,
    ) as error:
        logging.exception("Validation could not be completed: %s", error)
        print(f"\nVALIDATION FAILED: {error}", file=sys.stderr)
        return 1

    except Exception as error:
        logging.exception("Unexpected validation failure: %s", error)
        print(
            f"\nUnexpected error: {error}\nCheck log file: {LOG_PATH}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
