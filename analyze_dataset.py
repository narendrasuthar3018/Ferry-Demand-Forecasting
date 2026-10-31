# ============================================================
# FILE: analyze_dataset.py
# PROJECT: Short-Term Ferry Ticket Demand Forecasting
# PURPOSE: Industry-Level Exploratory Data Analysis (EDA)
# AUTHOR: Narendra Suthar
# ============================================================

from __future__ import annotations

import json
import logging
import sys
import time
import warnings

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

REPORT_DIR = OUTPUT_DIR / "reports"
FIGURE_DIR = OUTPUT_DIR / "figures"
LOG_DIR = OUTPUT_DIR / "logs"

DATASET_PATH = DATA_DIR / "Toronto Island Ferry Tickets (1).csv"

EXPECTED_COLUMNS = [
    "_id",
    "Timestamp",
    "Redemption Count",
    "Sales Count",
]

TIMESTAMP_COLUMN = "Timestamp"

DEMAND_COLUMNS = [
    "Sales Count",
    "Redemption Count",
]

EXPECTED_INTERVAL_MINUTES = 15

TOP_CORRELATION_COUNT = 10

MAX_CATEGORICAL_UNIQUE_VALUES = 50

HISTOGRAM_BINS = 50

REPORT_VERSION = "1.0.0"


# ============================================================
# 2. OUTPUT DIRECTORIES
# ============================================================

def create_output_directories() -> None:
    """
    Create the directories required for analysis outputs.

    Existing files are preserved.
    The source dataset is never modified.
    """

    directories = [
        REPORT_DIR,
        FIGURE_DIR,
        LOG_DIR,
        OUTPUT_DIR / "forecasts",
        BASE_DIR / "models",
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# 3. LOGGING CONFIGURATION
# ============================================================

def configure_logging() -> logging.Logger:
    """
    Configure console and file logging.

    Logs provide execution details and help diagnose failures.
    """

    create_output_directories()

    logger = logging.getLogger("ferry_dataset_analysis")

    logger.setLevel(logging.INFO)

    logger.propagate = False

    if logger.handlers:
        return logger

    log_file = LOG_DIR / "analyze_dataset.log"

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8",
    )

    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)

    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


LOGGER = configure_logging()


# ============================================================
# 4. GENERAL UTILITY FUNCTIONS
# ============================================================

def get_utc_timestamp() -> str:
    """Return the current UTC timestamp in ISO format."""

    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any) -> float | None:
    """
    Convert a value to a JSON-safe floating-point number.

    NaN and infinite values become None.
    """

    try:
        number = float(value)

        if not np.isfinite(number):
            return None

        return number

    except (TypeError, ValueError, OverflowError):
        return None


def safe_int(value: Any) -> int | None:
    """Convert a value to an integer when possible."""

    try:
        if pd.isna(value):
            return None

        return int(value)

    except (TypeError, ValueError, OverflowError):
        return None


def safe_json_value(value: Any) -> Any:
    """
    Convert common NumPy and pandas values into
    JSON-compatible Python values.
    """

    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(key): safe_json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            safe_json_value(item)
            for item in value
        ]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return safe_float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): safe_json_value(item)
            for key, item in value.items()
        }

    if value is pd.NA or value is pd.NaT:
        return None

    if isinstance(value, float) and not np.isfinite(value):
        return None

    if isinstance(value, np.ndarray):
        return [
            safe_json_value(item)
            for item in value.tolist()
        ]

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def save_json_report(
    data: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Save a structured JSON report safely.

    The report is written to a temporary file first
    to reduce the risk of leaving a partially written file.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            safe_json_value(data),
            file,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        )

    temporary_path.replace(output_path)

    LOGGER.info(
        "JSON report saved: %s",
        output_path.relative_to(BASE_DIR),
    )


def save_dataframe_csv(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a DataFrame as a UTF-8 CSV report."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    LOGGER.info(
        "CSV report saved: %s",
        output_path.relative_to(BASE_DIR),
    )


def calculate_percentage(
    numerator: float,
    denominator: float,
) -> float | None:
    """Calculate a percentage without dividing by zero."""

    if denominator == 0:
        return None

    return safe_float(
        (numerator / denominator) * 100
    )


def format_number(value: Any) -> str:
    """Format a number for readable log messages."""

    try:
        if pd.isna(value):
            return "N/A"

        if isinstance(value, (int, np.integer)):
            return f"{int(value):,}"

        return f"{float(value):,.4f}"

    except (TypeError, ValueError):
        return str(value)


# ============================================================
# 5. DATASET LOADING
# ============================================================

def load_dataset(
    dataset_path: Path = DATASET_PATH,
) -> pd.DataFrame:
    """
    Load the source CSV with basic structural safeguards.

    Important:
    - The original CSV is not modified.
    - No rows are silently deleted.
    - No missing timestamps are artificially inserted.
    - Original column names are preserved.
    """

    LOGGER.info("=" * 70)
    LOGGER.info("STARTING FERRY DATASET ANALYSIS")
    LOGGER.info("=" * 70)

    LOGGER.info(
        "Dataset location: %s",
        dataset_path,
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file was not found: {dataset_path}\n"
            "Check DATASET_PATH and confirm that the CSV "
            "exists inside the project's data directory."
        )

    if not dataset_path.is_file():
        raise ValueError(
            f"The dataset path is not a file: {dataset_path}"
        )

    file_size_bytes = dataset_path.stat().st_size

    if file_size_bytes == 0:
        raise ValueError(
            "The dataset CSV exists but is empty."
        )

    LOGGER.info(
        "Source file size: %.2f MB",
        file_size_bytes / (1024 ** 2),
    )

    started_at = time.perf_counter()

    dataframe = pd.read_csv(
        dataset_path,
        low_memory=False,
        encoding="utf-8-sig",
    )

    elapsed_seconds = time.perf_counter() - started_at

    if dataframe.empty:
        raise ValueError(
            "The CSV was loaded successfully, "
            "but it contains zero data rows."
        )

    LOGGER.info(
        "Dataset loaded successfully in %.3f seconds.",
        elapsed_seconds,
    )

    LOGGER.info(
        "Rows: %s | Columns: %s",
        format_number(dataframe.shape[0]),
        format_number(dataframe.shape[1]),
    )

    return dataframe


# ============================================================
# 6. STRUCTURAL VALIDATION
# ============================================================

def validate_dataset_structure(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Validate required columns and basic dataset structure.

    This function reports structural issues without
    changing the source DataFrame.
    """

    LOGGER.info("Validating dataset structure...")

    actual_columns = list(dataframe.columns)

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in actual_columns
    ]

    unexpected_columns = [
        column
        for column in actual_columns
        if column not in EXPECTED_COLUMNS
    ]

    duplicated_column_names = [
        column
        for column, count in Counter(actual_columns).items()
        if count > 1
    ]

    checks = {
        "dataset_has_rows": len(dataframe) > 0,
        "dataset_has_columns": len(actual_columns) > 0,
        "required_columns_present": len(missing_columns) == 0,
        "column_names_unique": len(duplicated_column_names) == 0,
        "timestamp_column_present": (
            TIMESTAMP_COLUMN in actual_columns
        ),
        "demand_columns_present": all(
            column in actual_columns
            for column in DEMAND_COLUMNS
        ),
    }

    result = {
        "checked_at_utc": get_utc_timestamp(),
        "expected_columns": EXPECTED_COLUMNS,
        "actual_columns": actual_columns,
        "row_count": len(dataframe),
        "column_count": len(actual_columns),
        "missing_required_columns": missing_columns,
        "unexpected_columns": unexpected_columns,
        "duplicated_column_names": duplicated_column_names,
        "checks": checks,
        "all_structural_checks_passed": all(
            checks.values()
        ),
    }

    if missing_columns:
        LOGGER.error(
            "Required columns are missing: %s",
            missing_columns,
        )

    if duplicated_column_names:
        LOGGER.error(
            "Duplicate column names detected: %s",
            duplicated_column_names,
        )

    if unexpected_columns:
        LOGGER.info(
            "Additional columns detected: %s",
            unexpected_columns,
        )

    if result["all_structural_checks_passed"]:
        LOGGER.info(
            "Dataset structural validation passed."
        )
    else:
        LOGGER.warning(
            "Dataset structural validation found issues."
        )

    return result



# ============================================================
# END OF PART 1
# PASTE THE NEXT PART DIRECTLY BELOW THIS LINE.
# ============================================================
# ============================================================
# 7. DATA TYPE AND MISSING-VALUE ANALYSIS
# ============================================================

def analyze_data_types(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Profile every column's data type, missing values,
    unique values, and memory usage.
    """

    LOGGER.info("Analyzing column data types and missing values...")

    total_rows = len(dataframe)
    records = []

    for column in dataframe.columns:
        series = dataframe[column]

        missing_count = int(series.isna().sum())
        non_missing_count = int(series.notna().sum())
        unique_count = int(series.nunique(dropna=True))

        records.append({
            "Column": column,
            "Data_Type": str(series.dtype),
            "Total_Rows": total_rows,
            "Non_Missing_Count": non_missing_count,
            "Missing_Count": missing_count,
            "Missing_Percentage": calculate_percentage(
                missing_count,
                total_rows,
            ),
            "Unique_Values": unique_count,
            "Unique_Percentage": calculate_percentage(
                unique_count,
                non_missing_count,
            ),
            "Memory_Usage_Bytes": int(
                series.memory_usage(index=False, deep=True)
            ),
        })

    profile = pd.DataFrame(records)

    save_dataframe_csv(
        profile,
        REPORT_DIR / "dataset_column_profile_detailed.csv",
    )

    LOGGER.info(
        "Column profiling completed for %s columns.",
        len(profile),
    )

    return profile


# ============================================================
# 8. DUPLICATE RECORD ANALYSIS
# ============================================================

def analyze_duplicates(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Identify exact duplicate records and duplicate identifiers.

    Duplicate rows are reported, not removed.
    """

    LOGGER.info("Analyzing duplicate records...")

    total_rows = len(dataframe)

    exact_duplicate_mask = dataframe.duplicated(
        keep="first"
    )

    exact_duplicate_count = int(
        exact_duplicate_mask.sum()
    )

    duplicate_id_count = None
    duplicate_id_value_count = None
    missing_id_count = None

    if "_id" in dataframe.columns:
        id_series = dataframe["_id"]

        missing_id_count = int(id_series.isna().sum())

        duplicate_id_count = int(
            id_series.duplicated(keep="first").sum()
        )

        duplicate_id_value_count = int(
            id_series[
                id_series.notna()
            ].duplicated(keep=False).sum()
        )

    result = {
        "total_rows": total_rows,
        "exact_duplicate_rows": exact_duplicate_count,
        "exact_duplicate_percentage": calculate_percentage(
            exact_duplicate_count,
            total_rows,
        ),
        "duplicate_identifier_occurrences": duplicate_id_count,
        "rows_with_repeated_non_null_identifier": (
            duplicate_id_value_count
        ),
        "missing_identifier_count": missing_id_count,
        "source_modified": False,
        "duplicate_rows_removed": 0,
    }

    LOGGER.info(
        "Exact duplicate rows: %s",
        format_number(exact_duplicate_count),
    )

    if duplicate_id_count is not None:
        LOGGER.info(
            "Repeated identifier occurrences: %s",
            format_number(duplicate_id_count),
        )

    return result


# ============================================================
# 9. TIMESTAMP PREPARATION FOR ANALYSIS
# ============================================================

def prepare_timestamp_analysis(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Create an analysis-only copy with parsed timestamps.

    The original DataFrame remains unchanged.

    Invalid timestamps are excluded from timestamp-based
    calculations but remain present in the original dataset.
    """

    LOGGER.info("Preparing timestamps for time-based analysis...")

    if TIMESTAMP_COLUMN not in dataframe.columns:
        LOGGER.warning(
            "Timestamp column is missing. "
            "Time-based analysis will be limited."
        )

        return pd.DataFrame(), {
            "timestamp_column_present": False,
            "valid_timestamp_count": 0,
            "invalid_timestamp_count": len(dataframe),
            "missing_timestamp_count": None,
            "timestamp_order": "UNAVAILABLE",
            "analysis_available": False,
        }

    timestamp_series = pd.to_datetime(
        dataframe[TIMESTAMP_COLUMN],
        errors="coerce",
        utc=True,
    )

    missing_timestamp_count = int(
        dataframe[TIMESTAMP_COLUMN].isna().sum()
    )

    invalid_timestamp_count = int(
        (
            dataframe[TIMESTAMP_COLUMN].notna()
            & timestamp_series.isna()
        ).sum()
    )

    valid_mask = timestamp_series.notna()

    valid_count = int(valid_mask.sum())

    if valid_count == 0:
        LOGGER.warning(
            "No valid timestamps are available."
        )

        return pd.DataFrame(), {
            "timestamp_column_present": True,
            "valid_timestamp_count": 0,
            "invalid_timestamp_count": invalid_timestamp_count,
            "missing_timestamp_count": missing_timestamp_count,
            "timestamp_order": "UNAVAILABLE",
            "analysis_available": False,
        }

    working = dataframe.loc[valid_mask].copy()

    working["_Analysis_Timestamp_UTC"] = (
        timestamp_series.loc[valid_mask]
    )

    working["_Original_Row_Position"] = np.flatnonzero(
        valid_mask.to_numpy()
    )

    original_timestamp_order = timestamp_series.dropna()

    differences = (
        original_timestamp_order
        .diff()
        .dropna()
        .dt.total_seconds()
    )

    if differences.empty:
        timestamp_order = "INSUFFICIENT_DATA"

    elif (differences >= 0).all():
        timestamp_order = "ASCENDING"

    elif (differences <= 0).all():
        timestamp_order = "DESCENDING"

    else:
        timestamp_order = "MIXED"

    result = {
        "timestamp_column_present": True,
        "valid_timestamp_count": valid_count,
        "invalid_timestamp_count": invalid_timestamp_count,
        "missing_timestamp_count": missing_timestamp_count,
        "timestamp_order": timestamp_order,
        "analysis_available": True,
    }

    LOGGER.info(
        "Valid timestamps: %s",
        format_number(valid_count),
    )

    LOGGER.info(
        "Invalid timestamps: %s",
        format_number(invalid_timestamp_count),
    )

    LOGGER.info(
        "Original timestamp order: %s",
        timestamp_order,
    )

    return working, result


# ============================================================
# 10. TIMESTAMP COVERAGE AND INTERVAL ANALYSIS
# ============================================================

def analyze_timestamp_intervals(
    timestamp_dataframe: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Analyze observed timestamp intervals.

    Long intervals are not automatically interpreted as
    confirmed missing records. They may reflect operational
    gaps, missing data, or a different recording pattern.

    No synthetic timestamp rows are created.
    """

    LOGGER.info("Analyzing timestamp intervals and coverage...")

    if (
        timestamp_dataframe.empty
        or "_Analysis_Timestamp_UTC"
        not in timestamp_dataframe.columns
    ):
        return (
            {
                "analysis_available": False,
                "reason": "No valid timestamps available.",
            },
            pd.DataFrame(),
        )

    sorted_data = timestamp_dataframe.sort_values(
        "_Analysis_Timestamp_UTC",
        kind="mergesort",
    ).reset_index(drop=True)

    timestamps = sorted_data[
        "_Analysis_Timestamp_UTC"
    ]

    interval_minutes = (
        timestamps.diff()
        .dt.total_seconds()
        .div(60)
    )

    valid_intervals = interval_minutes.dropna()

    if valid_intervals.empty:
        return (
            {
                "analysis_available": False,
                "reason": "Insufficient timestamps for interval analysis.",
            },
            pd.DataFrame(),
        )

    expected_minutes = EXPECTED_INTERVAL_MINUTES

    exact_interval_count = int(
        np.isclose(
            valid_intervals.to_numpy(),
            expected_minutes,
            rtol=0,
            atol=1e-9,
        ).sum()
    )

    shorter_interval_count = int(
        (valid_intervals < expected_minutes).sum()
    )

    longer_interval_count = int(
        (valid_intervals > expected_minutes).sum()
    )

    negative_interval_count = int(
        (valid_intervals < 0).sum()
    )

    gap_rows = sorted_data.loc[
        interval_minutes > expected_minutes
    ].copy()

    gap_rows["Previous_Timestamp_UTC"] = (
        timestamps.shift(1).loc[gap_rows.index]
    )

    gap_rows["Observed_Interval_Minutes"] = (
        interval_minutes.loc[gap_rows.index]
    )

    gap_rows["Estimated_Unobserved_15_Minute_Slots"] = (
        np.maximum(
            np.ceil(
                gap_rows["Observed_Interval_Minutes"]
                / expected_minutes
            ).astype(int) - 1,
            0,
        )
    )

    gap_rows["Gap_Interpretation"] = (
        "Long observed interval; cause not confirmed"
    )

    interval_report = gap_rows[
        [
            "Previous_Timestamp_UTC",
            "_Analysis_Timestamp_UTC",
            "Observed_Interval_Minutes",
            "Estimated_Unobserved_15_Minute_Slots",
            "Gap_Interpretation",
        ]
    ].rename(
        columns={
            "_Analysis_Timestamp_UTC": "Current_Timestamp_UTC",
        }
    )

    if len(valid_intervals) > 0:
        cadence_deviation_percentage = calculate_percentage(
            int(
                (
                    ~np.isclose(
                        valid_intervals.to_numpy(),
                        expected_minutes,
                        rtol=0,
                        atol=1e-9,
                    )
                ).sum()
            ),
            len(valid_intervals),
        )
    else:
        cadence_deviation_percentage = None

    start_timestamp = timestamps.min()
    end_timestamp = timestamps.max()

    observed_span_minutes = (
        end_timestamp - start_timestamp
    ).total_seconds() / 60

    expected_slots_in_span = int(
        np.floor(
            observed_span_minutes / expected_minutes
        )
    )

    estimated_unobserved_slots = int(
        gap_rows[
            "Estimated_Unobserved_15_Minute_Slots"
        ].sum()
    )

    result = {
        "analysis_available": True,
        "expected_interval_minutes": expected_minutes,
        "valid_observed_interval_count": int(
            len(valid_intervals)
        ),
        "exact_15_minute_interval_count": exact_interval_count,
        "exact_interval_percentage": calculate_percentage(
            exact_interval_count,
            len(valid_intervals),
        ),
        "shorter_interval_count": shorter_interval_count,
        "longer_interval_count": longer_interval_count,
        "negative_interval_count": negative_interval_count,
        "cadence_deviation_percentage": (
            cadence_deviation_percentage
        ),
        "minimum_interval_minutes": safe_float(
            valid_intervals.min()
        ),
        "median_interval_minutes": safe_float(
            valid_intervals.median()
        ),
        "mean_interval_minutes": safe_float(
            valid_intervals.mean()
        ),
        "maximum_interval_minutes": safe_float(
            valid_intervals.max()
        ),
        "p95_interval_minutes": safe_float(
            valid_intervals.quantile(0.95)
        ),
        "p99_interval_minutes": safe_float(
            valid_intervals.quantile(0.99)
        ),
        "observed_start_timestamp_utc": (
            start_timestamp.isoformat()
        ),
        "observed_end_timestamp_utc": (
            end_timestamp.isoformat()
        ),
        "observed_span_minutes": safe_float(
            observed_span_minutes
        ),
        "expected_15_minute_slots_between_endpoints": (
            expected_slots_in_span
        ),
        "long_interval_count": int(len(gap_rows)),
        "estimated_unobserved_15_minute_slots": (
            estimated_unobserved_slots
        ),
        "warning": (
            "Estimated unobserved slots are inferred from "
            "timestamp gaps. They are not verified missing "
            "records and must not be treated as zero demand."
        ),
    }

    LOGGER.info(
        "Exact 15-minute intervals: %s",
        format_number(exact_interval_count),
    )

    LOGGER.info(
        "Long intervals: %s",
        format_number(longer_interval_count),
    )

    LOGGER.info(
        "Estimated unobserved slots: %s",
        format_number(estimated_unobserved_slots),
    )

    return result, interval_report


# ============================================================
# END OF PART 2
# PASTE THE NEXT PART DIRECTLY BELOW THIS LINE.
# ============================================================
# ============================================================
# 11. NUMERIC COLUMN STATISTICAL ANALYSIS
# ============================================================

def analyze_numeric_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate descriptive statistics for numeric columns.

    Includes:
    - Mean, median and standard deviation
    - Minimum, maximum and range
    - Quartiles and interquartile range
    - Skewness and kurtosis
    - Zero and negative value counts
    - Outlier estimates using the IQR method

    IQR outliers are statistical flags, not confirmed errors.
    """

    LOGGER.info("Analyzing numeric columns...")

    numeric_dataframe = dataframe.select_dtypes(
        include=[np.number]
    )

    if numeric_dataframe.empty:
        LOGGER.warning(
            "No numeric columns found for statistical analysis."
        )

        return pd.DataFrame()

    records = []

    for column in numeric_dataframe.columns:
        series = pd.to_numeric(
            numeric_dataframe[column],
            errors="coerce",
        )

        valid_series = series.replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        valid_count = len(valid_series)
        missing_or_nonfinite_count = (
            len(series) - valid_count
        )

        if valid_count == 0:
            records.append({
                "Column": column,
                "Valid_Count": 0,
                "Missing_Or_Nonfinite_Count": (
                    missing_or_nonfinite_count
                ),
                "Mean": None,
                "Median": None,
                "Standard_Deviation": None,
                "Minimum": None,
                "Maximum": None,
                "Range": None,
                "Q1": None,
                "Q3": None,
                "IQR": None,
                "Skewness": None,
                "Kurtosis": None,
                "Zero_Count": 0,
                "Negative_Count": 0,
                "IQR_Lower_Bound": None,
                "IQR_Upper_Bound": None,
                "IQR_Outlier_Count": 0,
                "IQR_Outlier_Percentage": None,
            })

            continue

        q1 = safe_float(valid_series.quantile(0.25))
        q3 = safe_float(valid_series.quantile(0.75))

        iqr = (
            q3 - q1
            if q1 is not None and q3 is not None
            else None
        )

        lower_bound = (
            q1 - 1.5 * iqr
            if iqr is not None
            else None
        )

        upper_bound = (
            q3 + 1.5 * iqr
            if iqr is not None
            else None
        )

        if lower_bound is not None and upper_bound is not None:
            outlier_mask = (
                (valid_series < lower_bound)
                | (valid_series > upper_bound)
            )

            outlier_count = int(outlier_mask.sum())

        else:
            outlier_count = 0

        records.append({
            "Column": column,
            "Valid_Count": valid_count,
            "Missing_Or_Nonfinite_Count": (
                missing_or_nonfinite_count
            ),
            "Mean": safe_float(valid_series.mean()),
            "Median": safe_float(valid_series.median()),
            "Standard_Deviation": safe_float(
                valid_series.std()
            ),
            "Minimum": safe_float(valid_series.min()),
            "Maximum": safe_float(valid_series.max()),
            "Range": safe_float(
                valid_series.max() - valid_series.min()
            ),
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "Skewness": safe_float(
                valid_series.skew()
            ),
            "Kurtosis": safe_float(
                valid_series.kurt()
            ),
            "Zero_Count": int(
                (valid_series == 0).sum()
            ),
            "Negative_Count": int(
                (valid_series < 0).sum()
            ),
            "IQR_Lower_Bound": safe_float(lower_bound),
            "IQR_Upper_Bound": safe_float(upper_bound),
            "IQR_Outlier_Count": outlier_count,
            "IQR_Outlier_Percentage": calculate_percentage(
                outlier_count,
                valid_count,
            ),
        })

    result = pd.DataFrame(records)

    save_dataframe_csv(
        result,
        REPORT_DIR / "numeric_statistical_profile.csv",
    )

    LOGGER.info(
        "Numeric statistical analysis completed for %s columns.",
        len(result),
    )

    return result


# ============================================================
# 12. DEMAND VARIABLE VALIDATION
# ============================================================

def analyze_demand_values(
    dataframe: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Validate and summarize Sales Count and Redemption Count.

    Counts are not silently clipped or corrected.

    Negative and fractional values are flagged for review.
    """

    LOGGER.info("Analyzing ticket demand values...")

    summaries: dict[str, Any] = {}
    issue_records: list[dict[str, Any]] = []

    for column in DEMAND_COLUMNS:

        if column not in dataframe.columns:
            summaries[column] = {
                "column_present": False,
                "analysis_available": False,
            }

            issue_records.append({
                "Column": column,
                "Issue": "MISSING_REQUIRED_DEMAND_COLUMN",
                "Affected_Rows": len(dataframe),
                "Affected_Percentage": 100.0,
                "Severity": "HIGH",
                "Details": (
                    "Required demand column is not present."
                ),
            })

            continue

        original_series = dataframe[column]

        numeric_series = pd.to_numeric(
            original_series,
            errors="coerce",
        )

        invalid_conversion_mask = (
            original_series.notna()
            & numeric_series.isna()
        )

        missing_mask = original_series.isna()

        finite_mask = (
            numeric_series.notna()
            & np.isfinite(numeric_series)
        )

        negative_mask = (
            finite_mask
            & (numeric_series < 0)
        )

        fractional_mask = (
            finite_mask
            & ~np.isclose(
                numeric_series.fillna(0).to_numpy(
                    dtype=float
                ),
                np.round(
                    numeric_series.fillna(0).to_numpy(
                        dtype=float
                    )
                ),
                rtol=0,
                atol=1e-9,
            )
        )

        fractional_mask = pd.Series(
            fractional_mask,
            index=dataframe.index,
        ) & finite_mask

        zero_mask = (
            finite_mask
            & (numeric_series == 0)
        )

        valid_values = numeric_series.loc[
            finite_mask
        ]

        negative_count = int(negative_mask.sum())
        fractional_count = int(fractional_mask.sum())
        missing_count = int(missing_mask.sum())
        invalid_conversion_count = int(
            invalid_conversion_mask.sum()
        )

        summaries[column] = {
            "column_present": True,
            "analysis_available": len(valid_values) > 0,
            "total_rows": len(dataframe),
            "valid_numeric_count": len(valid_values),
            "missing_count": missing_count,
            "invalid_numeric_conversion_count": (
                invalid_conversion_count
            ),
            "negative_value_count": negative_count,
            "fractional_value_count": fractional_count,
            "zero_value_count": int(zero_mask.sum()),
            "valid_value_percentage": calculate_percentage(
                len(valid_values),
                len(dataframe),
            ),
            "mean": safe_float(valid_values.mean()),
            "median": safe_float(valid_values.median()),
            "standard_deviation": safe_float(
                valid_values.std()
            ),
            "minimum": safe_float(valid_values.min()),
            "maximum": safe_float(valid_values.max()),
            "p90": safe_float(
                valid_values.quantile(0.90)
            ),
            "p95": safe_float(
                valid_values.quantile(0.95)
            ),
            "p99": safe_float(
                valid_values.quantile(0.99)
            ),
            "total_observed_value": safe_float(
                valid_values.sum()
            ),
        }

        checks = [
            (
                missing_count,
                "MISSING_DEMAND_VALUES",
                "MEDIUM",
                "Demand values are missing.",
            ),
            (
                invalid_conversion_count,
                "NON_NUMERIC_DEMAND_VALUES",
                "HIGH",
                "Some non-null values cannot be converted to numbers.",
            ),
            (
                negative_count,
                "NEGATIVE_DEMAND_VALUES",
                "HIGH",
                "Negative counts require domain validation.",
            ),
            (
                fractional_count,
                "FRACTIONAL_DEMAND_VALUES",
                "MEDIUM",
                "Fractional ticket counts require domain validation.",
            ),
        ]

        for (
            affected_count,
            issue_code,
            severity,
            details,
        ) in checks:

            if affected_count > 0:
                issue_records.append({
                    "Column": column,
                    "Issue": issue_code,
                    "Affected_Rows": affected_count,
                    "Affected_Percentage": (
                        calculate_percentage(
                            affected_count,
                            len(dataframe),
                        )
                    ),
                    "Severity": severity,
                    "Details": details,
                })

        LOGGER.info(
            "%s | valid=%s | mean=%s | median=%s | negatives=%s",
            column,
            format_number(len(valid_values)),
            format_number(valid_values.mean()),
            format_number(valid_values.median()),
            format_number(negative_count),
        )

    demand_issues = pd.DataFrame(
        issue_records,
        columns=[
            "Column",
            "Issue",
            "Affected_Rows",
            "Affected_Percentage",
            "Severity",
            "Details",
        ],
    )

    save_dataframe_csv(
        demand_issues,
        REPORT_DIR / "demand_data_quality_issues.csv",
    )

    return summaries, demand_issues


# ============================================================
# 13. CATEGORICAL AND IDENTIFIER ANALYSIS
# ============================================================

def analyze_non_numeric_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize non-numeric columns.

    High-cardinality identifiers are summarized by counts,
    not exported as enormous frequency tables.
    """

    LOGGER.info("Analyzing non-numeric columns...")

    non_numeric_columns = dataframe.select_dtypes(
        exclude=[np.number]
    ).columns

    records = []

    for column in non_numeric_columns:

        series = dataframe[column]

        non_missing = series.dropna()

        unique_count = int(
            non_missing.nunique()
        )

        top_values = []

        if (
            unique_count > 0
            and unique_count <= MAX_CATEGORICAL_UNIQUE_VALUES
        ):
            value_counts = (
                non_missing.astype(str)
                .value_counts()
                .head(10)
            )

            top_values = [
                {
                    "value": str(value),
                    "count": int(count),
                }
                for value, count in value_counts.items()
            ]

        records.append({
            "Column": column,
            "Data_Type": str(series.dtype),
            "Non_Missing_Count": int(series.notna().sum()),
            "Missing_Count": int(series.isna().sum()),
            "Unique_Non_Missing_Values": unique_count,
            "Is_High_Cardinality": (
                unique_count > MAX_CATEGORICAL_UNIQUE_VALUES
            ),
            "Top_Values_JSON": json.dumps(
                top_values,
                ensure_ascii=False,
            ),
        })

    result = pd.DataFrame(records)

    save_dataframe_csv(
        result,
        REPORT_DIR / "non_numeric_column_profile.csv",
    )

    LOGGER.info(
        "Non-numeric column analysis completed."
    )

    return result


# ============================================================
# END OF PART 3
# PASTE THE NEXT PART DIRECTLY BELOW THIS LINE.
# ============================================================
# ============================================================
# 14. SALES AND REDEMPTION RELATIONSHIP ANALYSIS
# ============================================================

def analyze_demand_relationship(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Analyze relationships between sales and redemption counts.

    Correlation indicates statistical association, not causation.
    Sales and redemption records are not assumed to represent
    the same tickets or the same customer cohort.
    """

    LOGGER.info(
        "Analyzing sales and redemption relationships..."
    )

    if not all(
        column in dataframe.columns
        for column in DEMAND_COLUMNS
    ):
        LOGGER.warning(
            "Relationship analysis skipped because "
            "one or more demand columns are missing."
        )

        return {
            "analysis_available": False,
            "reason": "Required demand columns are missing.",
        }

    working = dataframe[DEMAND_COLUMNS].copy()

    for column in DEMAND_COLUMNS:
        working[column] = pd.to_numeric(
            working[column],
            errors="coerce",
        )

        working[column] = working[column].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    valid_pair_mask = working[
        DEMAND_COLUMNS
    ].notna().all(axis=1)

    paired = working.loc[valid_pair_mask]

    if len(paired) < 2:
        LOGGER.warning(
            "Insufficient paired numeric observations "
            "for relationship analysis."
        )

        return {
            "analysis_available": False,
            "valid_paired_observations": len(paired),
            "reason": "Insufficient valid paired observations.",
        }

    sales = paired["Sales Count"]
    redemption = paired["Redemption Count"]

    pearson_correlation = safe_float(
        sales.corr(
            redemption,
            method="pearson",
        )
    )

    spearman_correlation = safe_float(
        sales.corr(
            redemption,
            method="spearman",
        )
    )

    sales_total = safe_float(sales.sum())
    redemption_total = safe_float(redemption.sum())

    sales_mean = safe_float(sales.mean())
    redemption_mean = safe_float(redemption.mean())

    total_difference = safe_float(
        sales.sum() - redemption.sum()
    )

    result = {
        "analysis_available": True,
        "valid_paired_observations": len(paired),
        "excluded_rows_due_to_missing_or_invalid_values": (
            len(working) - len(paired)
        ),
        "pearson_correlation": pearson_correlation,
        "spearman_correlation": spearman_correlation,
        "sales_total_over_valid_pairs": sales_total,
        "redemption_total_over_valid_pairs": redemption_total,
        "mean_sales_per_observation": sales_mean,
        "mean_redemptions_per_observation": redemption_mean,
        "sales_minus_redemption_total": total_difference,
        "interpretation_note": (
            "Correlation describes association between observed "
            "counts. It does not establish causation, ticket-level "
            "matching, or a verified conversion rate."
        ),
    }

    correlation_matrix = paired.corr(
        method="pearson"
    )

    correlation_matrix.index.name = "Variable"

    save_dataframe_csv(
        correlation_matrix.reset_index(),
        REPORT_DIR / "demand_correlation_matrix.csv",
    )

    LOGGER.info(
        "Sales/Redemption Pearson correlation: %s",
        format_number(pearson_correlation),
    )

    LOGGER.info(
        "Sales/Redemption Spearman correlation: %s",
        format_number(spearman_correlation),
    )

    return result


# ============================================================
# 15. TIME-BASED DEMAND PATTERN ANALYSIS
# ============================================================

def analyze_time_patterns(
    timestamp_dataframe: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Analyze demand by hour, weekday, month and calendar date.

    Aggregations use observed records only.
    Missing intervals are not filled with zero demand.
    """

    LOGGER.info(
        "Analyzing hourly, weekday and monthly demand patterns..."
    )

    if timestamp_dataframe.empty:
        return (
            {
                "analysis_available": False,
                "reason": "No valid timestamps are available.",
            },
            pd.DataFrame(),
            pd.DataFrame(),
        )

    working = timestamp_dataframe.copy()

    timestamp_column = "_Analysis_Timestamp_UTC"

    working = working.sort_values(
        timestamp_column,
        kind="mergesort",
    )

    working["Calendar_Date_UTC"] = (
        working[timestamp_column].dt.strftime("%Y-%m-%d")
    )

    working["Hour_UTC"] = (
        working[timestamp_column].dt.hour
    )

    working["Day_Of_Week_UTC"] = (
        working[timestamp_column].dt.dayofweek
    )

    working["Weekday_Name_UTC"] = (
        working[timestamp_column].dt.day_name()
    )

    working["Month_UTC"] = (
        working[timestamp_column].dt.month
    )

    working["Month_Name_UTC"] = (
        working[timestamp_column].dt.strftime("%B")
    )

    working["Year_UTC"] = (
        working[timestamp_column].dt.year
    )

    available_demand_columns = [
        column
        for column in DEMAND_COLUMNS
        if column in working.columns
    ]

    for column in available_demand_columns:
        working[column] = pd.to_numeric(
            working[column],
            errors="coerce",
        )

        working[column] = working[column].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    aggregation_rules = {
        column: ["count", "mean", "median", "sum", "std"]
        for column in available_demand_columns
    }

    hourly = (
        working.groupby(
            "Hour_UTC",
            dropna=False,
        )
        .agg(aggregation_rules)
    )

    hourly.columns = [
        f"{column}_{statistic.title()}"
        for column, statistic in hourly.columns
    ]

    hourly = hourly.reset_index()

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    weekday = (
        working.groupby(
            ["Day_Of_Week_UTC", "Weekday_Name_UTC"],
            dropna=False,
        )
        .agg(aggregation_rules)
    )

    weekday.columns = [
        f"{column}_{statistic.title()}"
        for column, statistic in weekday.columns
    ]

    weekday = weekday.reset_index()

    weekday["_Weekday_Sort"] = (
        weekday["Weekday_Name_UTC"]
        .map({
            name: index
            for index, name in enumerate(weekday_order)
        })
    )

    weekday = (
        weekday.sort_values("_Weekday_Sort")
        .drop(columns="_Weekday_Sort")
        .reset_index(drop=True)
    )

    monthly = (
        working.groupby(
            ["Year_UTC", "Month_UTC", "Month_Name_UTC"],
            dropna=False,
        )
        .agg(aggregation_rules)
    )

    monthly.columns = [
        f"{column}_{statistic.title()}"
        for column, statistic in monthly.columns
    ]

    monthly = monthly.reset_index()

    daily = (
        working.groupby(
            "Calendar_Date_UTC",
            dropna=False,
        )
        .agg(aggregation_rules)
    )

    daily.columns = [
        f"{column}_{statistic.title()}"
        for column, statistic in daily.columns
    ]

    daily = daily.reset_index()

    save_dataframe_csv(
        hourly,
        REPORT_DIR / "demand_by_hour_utc.csv",
    )

    save_dataframe_csv(
        weekday,
        REPORT_DIR / "demand_by_weekday_utc.csv",
    )

    save_dataframe_csv(
        monthly,
        REPORT_DIR / "demand_by_month_utc.csv",
    )

    save_dataframe_csv(
        daily,
        REPORT_DIR / "daily_demand_summary_utc.csv",
    )

    summary: dict[str, Any] = {
        "analysis_available": True,
        "timestamp_timezone": "UTC",
        "valid_timestamp_rows": len(working),
        "demand_columns_analyzed": available_demand_columns,
        "hourly_groups": len(hourly),
        "weekday_groups": len(weekday),
        "monthly_groups": len(monthly),
        "daily_groups": len(daily),
        "aggregation_note": (
            "Summaries use observed records. They do not "
            "represent a complete fixed-interval series when "
            "timestamps are missing or irregular."
        ),
    }

    for column in available_demand_columns:
        mean_column = f"{column}_Mean"
        count_column = f"{column}_Count"

        if (
            mean_column in hourly.columns
            and hourly[mean_column].notna().any()
        ):
            peak_hour_row = hourly.loc[
                hourly[mean_column].idxmax()
            ]

            lowest_hour_row = hourly.loc[
                hourly[mean_column].idxmin()
            ]

            summary[column] = {
                "highest_mean_demand_hour_utc": safe_int(
                    peak_hour_row["Hour_UTC"]
                ),
                "highest_mean_demand": safe_float(
                    peak_hour_row[mean_column]
                ),
                "lowest_mean_demand_hour_utc": safe_int(
                    lowest_hour_row["Hour_UTC"]
                ),
                "lowest_mean_demand": safe_float(
                    lowest_hour_row[mean_column]
                ),
                "hourly_observation_counts_available": (
                    count_column in hourly.columns
                ),
            }

    LOGGER.info(
        "Time pattern analysis completed: "
        "%s hourly groups, %s weekday groups, "
        "%s monthly groups, %s daily groups.",
        len(hourly),
        len(weekday),
        len(monthly),
        len(daily),
    )

    return summary, hourly, weekday


# ============================================================
# 16. DAILY DEMAND TREND AND EXTREME-DAY ANALYSIS
# ============================================================

def analyze_daily_demand_extremes(
    timestamp_dataframe: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Identify high and low observed-demand dates.

    This is descriptive analysis, not a forecast.
    A date with fewer observed records may have a biased total.
    """

    LOGGER.info("Analyzing daily demand extremes...")

    if (
        timestamp_dataframe.empty
        or "_Analysis_Timestamp_UTC"
        not in timestamp_dataframe.columns
    ):
        return (
            {
                "analysis_available": False,
                "reason": "No valid timestamps available.",
            },
            pd.DataFrame(),
        )

    working = timestamp_dataframe.copy()

    working["Calendar_Date_UTC"] = (
        working["_Analysis_Timestamp_UTC"].dt.strftime(
            "%Y-%m-%d"
        )
    )

    aggregation: dict[str, list[str]] = {}

    for column in DEMAND_COLUMNS:
        if column not in working.columns:
            continue

        working[column] = pd.to_numeric(
            working[column],
            errors="coerce",
        )

        working[column] = working[column].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        aggregation[column] = [
            "count",
            "sum",
            "mean",
            "median",
            "max",
        ]

    if not aggregation:
        return (
            {
                "analysis_available": False,
                "reason": "No demand columns available.",
            },
            pd.DataFrame(),
        )

    daily = (
        working.groupby("Calendar_Date_UTC")
        .agg(aggregation)
    )

    daily.columns = [
        f"{column}_{statistic.title()}"
        for column, statistic in daily.columns
    ]

    daily = daily.reset_index()

    daily["Date_Observed_Record_Count"] = (
        working.groupby("Calendar_Date_UTC")
        .size()
        .reindex(daily["Calendar_Date_UTC"])
        .to_numpy()
    )

    summaries: dict[str, Any] = {
        "analysis_available": True,
        "daily_group_count": len(daily),
        "interpretation_warning": (
            "Daily totals can be affected by missing intervals, "
            "irregular observations, or partial-day coverage. "
            "Extreme dates require coverage-aware interpretation."
        ),
    }

    for column in aggregation:
        sum_column = f"{column}_Sum"
        mean_column = f"{column}_Mean"
        count_column = f"{column}_Count"

        if (
            sum_column not in daily.columns
            or daily[sum_column].notna().sum() == 0
        ):
            continue

        max_total_row = daily.loc[
            daily[sum_column].idxmax()
        ]

        min_total_row = daily.loc[
            daily[sum_column].idxmin()
        ]

        max_mean_row = daily.loc[
            daily[mean_column].idxmax()
        ]

        summaries[column] = {
            "highest_observed_daily_total_date_utc": str(
                max_total_row["Calendar_Date_UTC"]
            ),
            "highest_observed_daily_total": safe_float(
                max_total_row[sum_column]
            ),
            "highest_total_date_observation_count": safe_int(
                max_total_row[count_column]
            ),
            "lowest_observed_daily_total_date_utc": str(
                min_total_row["Calendar_Date_UTC"]
            ),
            "lowest_observed_daily_total": safe_float(
                min_total_row[sum_column]
            ),
            "highest_daily_mean_date_utc": str(
                max_mean_row["Calendar_Date_UTC"]
            ),
            "highest_daily_mean": safe_float(
                max_mean_row[mean_column]
            ),
        }

    save_dataframe_csv(
        daily,
        REPORT_DIR / "daily_demand_extremes.csv",
    )

    LOGGER.info(
        "Daily demand extreme analysis completed."
    )

    return summaries, daily


# ============================================================
# END OF PART 4
# PASTE THE NEXT PART DIRECTLY BELOW THIS LINE.
# ============================================================
# ============================================================
# 17. CORRELATION AND FEATURE RELATIONSHIP ANALYSIS
# ============================================================

def analyze_numeric_correlations(
    dataframe: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Calculate Pearson and Spearman correlations between
    numeric variables.

    Correlation is descriptive and does not establish causality.
    Identifier columns are excluded from correlation analysis.
    """

    LOGGER.info("Analyzing numeric correlations...")

    numeric_data = dataframe.select_dtypes(
        include=[np.number]
    ).copy()

    excluded_columns = []

    for column in numeric_data.columns:
        if column.lower() in {
            "_id",
            "id",
            "record_id",
            "row_id",
        }:
            excluded_columns.append(column)

    numeric_data = numeric_data.drop(
        columns=excluded_columns,
        errors="ignore",
    )

    for column in numeric_data.columns:
        numeric_data[column] = numeric_data[column].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    # Exclude columns that contain no useful variation.
    constant_columns = [
        column
        for column in numeric_data.columns
        if numeric_data[column].nunique(dropna=True) <= 1
    ]

    numeric_data = numeric_data.drop(
        columns=constant_columns,
        errors="ignore",
    )

    if numeric_data.shape[1] < 2:
        LOGGER.warning(
            "Insufficient numeric columns for correlation analysis."
        )

        return (
            {
                "analysis_available": False,
                "reason": "Fewer than two usable numeric columns.",
                "excluded_identifier_columns": excluded_columns,
                "excluded_constant_columns": constant_columns,
            },
            pd.DataFrame(),
        )

    pearson_matrix = numeric_data.corr(
        method="pearson",
        min_periods=2,
    )

    spearman_matrix = numeric_data.corr(
        method="spearman",
        min_periods=2,
    )

    correlation_records = []

    columns = list(numeric_data.columns)

    for i, column_a in enumerate(columns):
        for column_b in columns[i + 1:]:

            pearson_value = safe_float(
                pearson_matrix.loc[column_a, column_b]
            )

            spearman_value = safe_float(
                spearman_matrix.loc[column_a, column_b]
            )

            if pearson_value is None and spearman_value is None:
                continue

            correlation_records.append({
                "Variable_A": column_a,
                "Variable_B": column_b,
                "Pearson_Correlation": pearson_value,
                "Absolute_Pearson_Correlation": (
                    abs(pearson_value)
                    if pearson_value is not None
                    else None
                ),
                "Spearman_Correlation": spearman_value,
                "Absolute_Spearman_Correlation": (
                    abs(spearman_value)
                    if spearman_value is not None
                    else None
                ),
                "Interpretation": (
                    "Association only; not evidence of causation."
                ),
            })

    correlations = pd.DataFrame(
        correlation_records
    )

    if not correlations.empty:
        correlations = correlations.sort_values(
            by="Absolute_Pearson_Correlation",
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

    save_dataframe_csv(
        pearson_matrix.reset_index().rename(
            columns={"index": "Variable"}
        ),
        REPORT_DIR / "numeric_pearson_correlation_matrix.csv",
    )

    save_dataframe_csv(
        spearman_matrix.reset_index().rename(
            columns={"index": "Variable"}
        ),
        REPORT_DIR / "numeric_spearman_correlation_matrix.csv",
    )

    save_dataframe_csv(
        correlations,
        REPORT_DIR / "numeric_correlation_pairs.csv",
    )

    strong_pairs = []

    if not correlations.empty:
        for _, row in correlations.iterrows():

            value = row["Pearson_Correlation"]

            if pd.isna(value):
                continue

            if abs(float(value)) >= 0.70:
                strong_pairs.append({
                    "variable_a": row["Variable_A"],
                    "variable_b": row["Variable_B"],
                    "pearson_correlation": safe_float(value),
                    "spearman_correlation": safe_float(
                        row["Spearman_Correlation"]
                    ),
                })

    summary = {
        "analysis_available": True,
        "numeric_columns_analyzed": len(columns),
        "excluded_identifier_columns": excluded_columns,
        "excluded_constant_columns": constant_columns,
        "correlation_pair_count": len(correlations),
        "strong_pearson_pair_count": len(strong_pairs),
        "strong_pearson_pairs": strong_pairs,
        "strong_correlation_threshold": 0.70,
        "interpretation_note": (
            "Strong correlations may reflect shared trends, "
            "mathematical relationships, or leakage. They do "
            "not prove causation or predictive usefulness."
        ),
    }

    LOGGER.info(
        "Numeric correlation analysis completed: %s pairs.",
        len(correlations),
    )

    return summary, correlations


# ============================================================
# 18. DEMAND DISTRIBUTION AND PERCENTILE ANALYSIS
# ============================================================

def analyze_demand_distributions(
    dataframe: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Describe the distribution of each demand variable.

    Reports percentiles and distribution shape.
    Does not remove outliers or alter observed values.
    """

    LOGGER.info("Analyzing demand distributions...")

    distribution_records = []
    distribution_summary = {}

    for column in DEMAND_COLUMNS:

        if column not in dataframe.columns:
            distribution_summary[column] = {
                "analysis_available": False,
                "reason": "Column is missing.",
            }

            continue

        original_series = dataframe[column]

        numeric_series = pd.to_numeric(
            original_series,
            errors="coerce",
        )

        numeric_series = numeric_series.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        valid = numeric_series.dropna()

        if valid.empty:
            distribution_summary[column] = {
                "analysis_available": False,
                "reason": "No valid numeric values.",
            }

            continue

        percentiles = {
            "P01": 0.01,
            "P05": 0.05,
            "P10": 0.10,
            "P25": 0.25,
            "P50": 0.50,
            "P75": 0.75,
            "P90": 0.90,
            "P95": 0.95,
            "P99": 0.99,
        }

        percentile_values = {
            name: safe_float(
                valid.quantile(quantile)
            )
            for name, quantile in percentiles.items()
        }

        mean_value = safe_float(valid.mean())
        median_value = safe_float(valid.median())

        distribution_records.append({
            "Demand_Column": column,
            "Valid_Count": len(valid),
            "Missing_Or_Invalid_Count": (
                len(dataframe) - len(valid)
            ),
            "Minimum": safe_float(valid.min()),
            "Maximum": safe_float(valid.max()),
            "Mean": mean_value,
            "Median": median_value,
            "Standard_Deviation": safe_float(valid.std()),
            "Variance": safe_float(valid.var()),
            "Skewness": safe_float(valid.skew()),
            "Kurtosis": safe_float(valid.kurt()),
            "Coefficient_Of_Variation": (
                safe_float(valid.std() / abs(mean_value))
                if mean_value not in (None, 0.0)
                and safe_float(valid.std()) is not None
                else None
            ),
            **percentile_values,
        })

        distribution_summary[column] = {
            "analysis_available": True,
            "valid_count": len(valid),
            "minimum": safe_float(valid.min()),
            "maximum": safe_float(valid.max()),
            "mean": mean_value,
            "median": median_value,
            "standard_deviation": safe_float(valid.std()),
            "skewness": safe_float(valid.skew()),
            "percentiles": percentile_values,
        }

    distribution_table = pd.DataFrame(
        distribution_records
    )

    save_dataframe_csv(
        distribution_table,
        REPORT_DIR / "demand_distribution_summary.csv",
    )

    LOGGER.info(
        "Demand distribution analysis completed."
    )

    return distribution_summary, distribution_table


# ============================================================
# 19. HIGH-DEMAND OBSERVATION ANALYSIS
# ============================================================

def analyze_high_demand_observations(
    dataframe: pd.DataFrame,
    top_n: int = 100,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Identify the highest observed demand records.

    These records represent high observed values, not guaranteed
    operational peak periods or forecasted demand.
    """

    LOGGER.info(
        "Identifying high observed demand records..."
    )

    top_n = max(1, int(top_n))

    available_columns = [
        column
        for column in DEMAND_COLUMNS
        if column in dataframe.columns
    ]

    if not available_columns:
        return (
            {
                "analysis_available": False,
                "reason": "No demand columns available.",
            },
            pd.DataFrame(),
        )

    working = dataframe.copy()

    for column in available_columns:
        working[column] = pd.to_numeric(
            working[column],
            errors="coerce",
        )

        working[column] = working[column].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    if TIMESTAMP_COLUMN in working.columns:
        working["_Parsed_Timestamp_UTC"] = pd.to_datetime(
            working[TIMESTAMP_COLUMN],
            errors="coerce",
            utc=True,
        )

    else:
        working["_Parsed_Timestamp_UTC"] = pd.NaT

    summary = {
        "analysis_available": True,
        "requested_top_n": top_n,
        "demand_columns": available_columns,
        "results": {},
        "interpretation_note": (
            "Ranked observations may be affected by irregular "
            "timestamp coverage and do not establish future peaks."
        ),
    }

    all_ranked_frames = []

    for column in available_columns:

        valid = working.loc[
            working[column].notna()
        ].copy()

        if valid.empty:
            summary["results"][column] = {
                "valid_observation_count": 0,
                "highest_observed_value": None,
            }

            continue

        ranked = valid.nlargest(
            min(top_n, len(valid)),
            columns=column,
            keep="all",
        ).copy()

        ranked["Demand_Metric"] = column

        ranked = ranked.rename(
            columns={
                column: "Observed_Demand_Value",
                "_Parsed_Timestamp_UTC": (
                    "Observed_Timestamp_UTC"
                ),
            }
        )

        ranked["Demand_Rank"] = (
            ranked["Observed_Demand_Value"]
            .rank(
                method="min",
                ascending=False,
            )
            .astype("int64")
        )

        keep_columns = [
            "Demand_Metric",
            "Demand_Rank",
            "Observed_Timestamp_UTC",
            "Observed_Demand_Value",
        ]

        if "_id" in ranked.columns:
            ranked["Source_Record_ID"] = ranked["_id"]
            keep_columns.append("Source_Record_ID")

        all_ranked_frames.append(
            ranked[keep_columns]
        )

        highest_value = safe_float(
            valid[column].max()
        )

        highest_rows = valid.loc[
            valid[column] == valid[column].max()
        ]

        summary["results"][column] = {
            "valid_observation_count": len(valid),
            "highest_observed_value": highest_value,
            "number_of_records_tied_for_highest": len(
                highest_rows
            ),
            "top_n_rows_returned": len(ranked),
            "highest_observed_timestamp_utc": (
                highest_rows[
                    "_Parsed_Timestamp_UTC"
                ].min().isoformat()
                if (
                    not highest_rows.empty
                    and pd.notna(
                        highest_rows[
                            "_Parsed_Timestamp_UTC"
                        ].min()
                    )
                )
                else None
            ),
        }

    if all_ranked_frames:
        ranked_output = pd.concat(
            all_ranked_frames,
            ignore_index=True,
        )

        ranked_output = ranked_output.sort_values(
            by=[
                "Demand_Metric",
                "Demand_Rank",
                "Observed_Timestamp_UTC",
            ],
            ascending=[True, True, True],
            na_position="last",
        ).reset_index(drop=True)

    else:
        ranked_output = pd.DataFrame()

    save_dataframe_csv(
        ranked_output,
        REPORT_DIR / "highest_observed_demand_records.csv",
    )

    LOGGER.info(
        "High-demand observation analysis completed."
    )

    return summary, ranked_output


# ============================================================
# END OF PART 5
# PASTE THE NEXT PART DIRECTLY BELOW THIS LINE.
# ============================================================
# ============================================================
# 20. OUTLIER ANALYSIS WITH MULTIPLE DETECTION METHODS
# ============================================================

def analyze_outliers(
    dataframe: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Detect unusual demand observations using:
    1. IQR statistical bounds.
    2. Median Absolute Deviation (MAD).
    3. Zero and negative demand checks.

    These methods identify statistical anomalies, not confirmed
    operational problems or fraudulent ticket transactions.

    No records are removed or modified.
    """

    LOGGER.info(
        "Starting multi-method demand outlier analysis..."
    )

    outlier_records = []
    summary: dict[str, Any] = {
        "analysis_available": True,
        "methods": [
            "IQR",
            "MAD",
            "Zero_Value_Check",
            "Negative_Value_Check",
        ],
        "demand_metrics": {},
    }

    for column in DEMAND_COLUMNS:

        if column not in dataframe.columns:
            summary["demand_metrics"][column] = {
                "analysis_available": False,
                "reason": "Column is missing.",
            }
            continue

        numeric_values = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        numeric_values = numeric_values.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        valid_mask = numeric_values.notna()
        valid_values = numeric_values.loc[valid_mask]

        if valid_values.empty:
            summary["demand_metrics"][column] = {
                "analysis_available": False,
                "reason": "No valid numeric observations.",
            }
            continue

        q1 = float(valid_values.quantile(0.25))
        q3 = float(valid_values.quantile(0.75))
        iqr = q3 - q1

        lower_iqr = q1 - (1.5 * iqr)
        upper_iqr = q3 + (1.5 * iqr)

        iqr_mask = (
            (numeric_values < lower_iqr)
            | (numeric_values > upper_iqr)
        ) & valid_mask

        median_value = float(valid_values.median())

        absolute_deviation = (
            valid_values - median_value
        ).abs()

        mad = float(absolute_deviation.median())

        if mad > 0:
            robust_z_scores = (
                0.67448975
                * (numeric_values - median_value)
                / mad
            )

            mad_mask = (
                robust_z_scores.abs() > 3.5
            ) & valid_mask

        else:
            # MAD can be zero when most observations are identical.
            # Do not divide by zero or invent robust z-scores.
            robust_z_scores = pd.Series(
                np.nan,
                index=dataframe.index,
                dtype=float,
            )

            mad_mask = pd.Series(
                False,
                index=dataframe.index,
            )

        zero_mask = (
            (numeric_values == 0)
            & valid_mask
        )

        negative_mask = (
            (numeric_values < 0)
            & valid_mask
        )

        union_mask = (
            iqr_mask
            | mad_mask
            | zero_mask
            | negative_mask
        )

        summary["demand_metrics"][column] = {
            "valid_observations": len(valid_values),
            "invalid_or_missing_observations": (
                len(dataframe) - len(valid_values)
            ),
            "iqr_lower_bound": safe_float(lower_iqr),
            "iqr_upper_bound": safe_float(upper_iqr),
            "iqr_outlier_count": int(iqr_mask.sum()),
            "mad_median": safe_float(median_value),
            "mad_value": safe_float(mad),
            "mad_outlier_count": int(mad_mask.sum()),
            "zero_value_count": int(zero_mask.sum()),
            "negative_value_count": int(negative_mask.sum()),
            "union_flagged_observation_count": int(
                union_mask.sum()
            ),
            "union_flagged_percentage": calculate_percentage(
                int(union_mask.sum()),
                int(valid_mask.sum()),
            ),
        }

        flagged_positions = np.flatnonzero(
            union_mask.to_numpy()
        )

        for position in flagged_positions:

            row_index = dataframe.index[position]

            original_row = dataframe.iloc[position]

            timestamp_value = None

            if TIMESTAMP_COLUMN in dataframe.columns:
                raw_timestamp = original_row[
                    TIMESTAMP_COLUMN
                ]

                if pd.notna(raw_timestamp):
                    timestamp_value = str(raw_timestamp)

            flags = []

            if bool(iqr_mask.iloc[position]):
                flags.append("IQR")

            if bool(mad_mask.iloc[position]):
                flags.append("MAD")

            if bool(zero_mask.iloc[position]):
                flags.append("ZERO_VALUE")

            if bool(negative_mask.iloc[position]):
                flags.append("NEGATIVE_VALUE")

            record = {
                "Source_Row_Index": str(row_index),
                "Timestamp": timestamp_value,
                "Demand_Metric": column,
                "Observed_Value": safe_float(
                    numeric_values.iloc[position]
                ),
                "IQR_Flag": bool(
                    iqr_mask.iloc[position]
                ),
                "MAD_Flag": bool(
                    mad_mask.iloc[position]
                ),
                "Zero_Value_Flag": bool(
                    zero_mask.iloc[position]
                ),
                "Negative_Value_Flag": bool(
                    negative_mask.iloc[position]
                ),
                "Robust_Z_Score": safe_float(
                    robust_z_scores.iloc[position]
                ),
                "Detection_Methods": ", ".join(flags),
                "Confirmed_Operational_Anomaly": False,
            }

            if "_id" in dataframe.columns:
                record["Source_Record_ID"] = (
                    safe_json_value(
                        original_row["_id"]
                    )
                )

            outlier_records.append(record)

        LOGGER.info(
            "%s | IQR=%s | MAD=%s | zero=%s | negative=%s",
            column,
            format_number(iqr_mask.sum()),
            format_number(mad_mask.sum()),
            format_number(zero_mask.sum()),
            format_number(negative_mask.sum()),
        )

    outlier_dataframe = pd.DataFrame(
        outlier_records
    )

    save_dataframe_csv(
        outlier_dataframe,
        REPORT_DIR / "demand_outlier_records.csv",
    )

    LOGGER.info(
        "Outlier analysis completed. Flagged metric-record "
        "combinations: %s",
        format_number(len(outlier_dataframe)),
    )

    return summary, outlier_dataframe


# ============================================================
# 21. DATA QUALITY ISSUE REGISTER
# ============================================================

def build_data_quality_issue_register(
    dataframe: pd.DataFrame,
    structural_report: dict[str, Any],
    timestamp_report: dict[str, Any],
    demand_issue_dataframe: pd.DataFrame,
    duplicate_report: dict[str, Any],
) -> pd.DataFrame:
    """
    Consolidate major data quality findings into one register.

    Severity indicates the potential impact on downstream
    analysis, not the probability of an operational failure.
    """

    LOGGER.info("Building consolidated data quality register...")

    issues: list[dict[str, Any]] = []

    def add_issue(
        issue_code: str,
        category: str,
        severity: str,
        affected_rows: int | None,
        description: str,
        recommendation: str,
    ) -> None:

        issues.append({
            "Issue_Code": issue_code,
            "Category": category,
            "Severity": severity,
            "Affected_Rows": affected_rows,
            "Description": description,
            "Recommended_Action": recommendation,
        })

    missing_required_columns = structural_report.get(
        "missing_required_columns",
        [],
    )

    if missing_required_columns:
        add_issue(
            "DQ001",
            "SCHEMA",
            "CRITICAL",
            len(dataframe),
            (
                "Required columns are missing: "
                + ", ".join(missing_required_columns)
            ),
            (
                "Verify the source schema and map the actual "
                "columns before running downstream analysis."
            ),
        )

    duplicated_names = structural_report.get(
        "duplicated_column_names",
        [],
    )

    if duplicated_names:
        add_issue(
            "DQ002",
            "SCHEMA",
            "HIGH",
            None,
            (
                "Duplicate column names detected: "
                + ", ".join(duplicated_names)
            ),
            (
                "Resolve duplicate column names explicitly "
                "before selecting or transforming columns."
            ),
        )

    if timestamp_report.get(
        "timestamp_column_present",
        False,
    ):

        invalid_timestamp_count = int(
            timestamp_report.get(
                "invalid_timestamp_count",
                0,
            ) or 0
        )

        missing_timestamp_count = int(
            timestamp_report.get(
                "missing_timestamp_count",
                0,
            ) or 0
        )

        if invalid_timestamp_count > 0:
            add_issue(
                "DQ003",
                "TIMESTAMP",
                "HIGH",
                invalid_timestamp_count,
                "Some non-null timestamp values could not be parsed.",
                (
                    "Inspect the original timestamp strings and "
                    "standardize formats before time-series modelling."
                ),
            )

        if missing_timestamp_count > 0:
            add_issue(
                "DQ004",
                "TIMESTAMP",
                "HIGH",
                missing_timestamp_count,
                "Some timestamp values are missing.",
                (
                    "Investigate affected records and exclude them "
                    "from timestamp-based calculations when necessary."
                ),
            )

    interval_deviation = timestamp_report.get(
        "cadence_deviation_percentage"
    )

    long_interval_count = timestamp_report.get(
        "long_interval_count"
    )

    if (
        interval_deviation is not None
        and interval_deviation > 0
    ):
        add_issue(
            "DQ005",
            "TIME_SERIES_CADENCE",
            "MEDIUM",
            (
                int(long_interval_count)
                if long_interval_count is not None
                else None
            ),
            (
                "Observed timestamp intervals deviate from the "
                "expected 15-minute cadence. Long gaps do not "
                "prove that records are missing."
            ),
            (
                "Use elapsed-time-based matching, validate forecast "
                "horizons and avoid filling gaps with zero demand "
                "without supporting evidence."
            ),
        )

    exact_duplicates = int(
        duplicate_report.get(
            "exact_duplicate_rows",
            0,
        ) or 0
    )

    if exact_duplicates > 0:
        add_issue(
            "DQ006",
            "DUPLICATES",
            "MEDIUM",
            exact_duplicates,
            "Exact duplicate rows were detected.",
            (
                "Review duplicate records against source-system "
                "semantics before deciding whether to remove them."
            ),
        )

    repeated_ids = duplicate_report.get(
        "duplicate_identifier_occurrences"
    )

    if repeated_ids is not None and int(repeated_ids) > 0:
        add_issue(
            "DQ007",
            "IDENTIFIER",
            "MEDIUM",
            int(repeated_ids),
            "Repeated identifier values were detected.",
            (
                "Confirm whether the identifier is expected to be "
                "unique and investigate repeated values."
            ),
        )

    if not demand_issue_dataframe.empty:
        for _, row in demand_issue_dataframe.iterrows():

            issue_code_map = {
                "MISSING_DEMAND_VALUES": "DQ008",
                "NON_NUMERIC_DEMAND_VALUES": "DQ009",
                "NEGATIVE_DEMAND_VALUES": "DQ010",
                "FRACTIONAL_DEMAND_VALUES": "DQ011",
            }

            issue_code = issue_code_map.get(
                str(row.get("Issue")),
                "DQ012",
            )

            add_issue(
                issue_code,
                "DEMAND_VALUES",
                str(row.get("Severity", "MEDIUM")),
                safe_int(row.get("Affected_Rows")),
                (
                    f"{row.get('Column')}: "
                    f"{row.get('Details')}"
                ),
                (
                    "Inspect the affected demand values and confirm "
                    "their validity with the data definition before "
                    "correcting or excluding observations."
                ),
            )

    issue_register = pd.DataFrame(
        issues,
        columns=[
            "Issue_Code",
            "Category",
            "Severity",
            "Affected_Rows",
            "Description",
            "Recommended_Action",
        ],
    )

    if not issue_register.empty:
        severity_order = {
            "CRITICAL": 0,
            "HIGH": 1,
            "MEDIUM": 2,
            "LOW": 3,
            "INFO": 4,
        }

        issue_register["_Severity_Order"] = (
            issue_register["Severity"]
            .map(severity_order)
            .fillna(99)
        )

        issue_register = (
            issue_register.sort_values(
                [
                    "_Severity_Order",
                    "Issue_Code",
                ]
            )
            .drop(columns="_Severity_Order")
            .reset_index(drop=True)
        )

    save_dataframe_csv(
        issue_register,
        REPORT_DIR / "dataset_quality_issue_register.csv",
    )

    LOGGER.info(
        "Consolidated data quality issues: %s",
        format_number(len(issue_register)),
    )

    return issue_register


# ============================================================
# 22. EXECUTIVE DATASET SUMMARY
# ============================================================

def build_executive_summary(
    dataframe: pd.DataFrame,
    structural_report: dict[str, Any],
    duplicate_report: dict[str, Any],
    timestamp_report: dict[str, Any],
    demand_summary: dict[str, Any],
    time_pattern_summary: dict[str, Any],
    relationship_summary: dict[str, Any],
    issue_register: pd.DataFrame,
) -> dict[str, Any]:
    """
    Produce a compact executive-level dataset summary.

    This report describes data readiness and observed patterns.
    It does not claim that a forecasting model is accurate.
    """

    LOGGER.info("Building executive dataset summary...")

    start_timestamp = None
    end_timestamp = None

    if timestamp_report.get("analysis_available", False):
        start_timestamp = timestamp_report.get(
            "observed_start_timestamp_utc"
        )

        end_timestamp = timestamp_report.get(
            "observed_end_timestamp_utc"
        )

    severity_counts = {}

    if not issue_register.empty:
        severity_counts = {
            str(key): int(value)
            for key, value in issue_register[
                "Severity"
            ].value_counts().items()
        }

    column_count = len(dataframe.columns)

    dataset_bytes = (
        DATASET_PATH.stat().st_size
        if DATASET_PATH.exists()
        else None
    )

    summary = {
        "report_metadata": {
            "report_name": "Ferry Ticket Demand Dataset Analysis",
            "report_version": REPORT_VERSION,
            "generated_at_utc": get_utc_timestamp(),
            "source_dataset": str(
                DATASET_PATH.relative_to(BASE_DIR)
            ),
            "source_modified": False,
        },
        "dataset_overview": {
            "row_count": len(dataframe),
            "column_count": column_count,
            "file_size_bytes": dataset_bytes,
            "file_size_mb": (
                safe_float(dataset_bytes / (1024 ** 2))
                if dataset_bytes is not None
                else None
            ),
            "columns": list(dataframe.columns),
            "observed_start_timestamp_utc": start_timestamp,
            "observed_end_timestamp_utc": end_timestamp,
        },
        "structural_validation": structural_report,
        "duplicate_analysis": duplicate_report,
        "timestamp_quality": timestamp_report,
        "demand_statistics": demand_summary,
        "time_pattern_summary": time_pattern_summary,
        "sales_redemption_relationship": relationship_summary,
        "data_quality": {
            "issue_count": len(issue_register),
            "issues_by_severity": severity_counts,
            "issue_register_path": str(
                (
                    REPORT_DIR
                    / "dataset_quality_issue_register.csv"
                ).relative_to(BASE_DIR)
            ),
        },
        "forecasting_readiness": {
            "ready_for_exploratory_analysis": (
                len(dataframe) > 0
            ),
            "timestamp_column_available": (
                TIMESTAMP_COLUMN in dataframe.columns
            ),
            "sales_count_available": (
                "Sales Count" in dataframe.columns
            ),
            "redemption_count_available": (
                "Redemption Count" in dataframe.columns
            ),
            "requires_chronological_ordering": True,
            "requires_elapsed_time_validation": True,
            "requires_chronological_train_test_split": True,
            "requires_leakage_prevention": True,
            "zero_filling_of_missing_timestamps_justified": False,
            "exact_failure_time_supported_by_this_dataset": False,
        },
        "limitations": [
            (
                "Observed ticket counts do not directly provide "
                "ferry capacity or passenger boarding counts."
            ),
            (
                "Timestamp gaps do not prove that demand was zero "
                "during unobserved intervals."
            ),
            (
                "Historical descriptive patterns do not guarantee "
                "future demand."
            ),
            (
                "Forecast accuracy must be evaluated separately "
                "using chronological holdout data and appropriate "
                "error metrics."
            ),
        ],
    }

    save_json_report(
        summary,
        REPORT_DIR / "dataset_analysis_executive_summary.json",
    )

    LOGGER.info(
        "Executive dataset summary created."
    )

    return summary


# ============================================================
# END OF PART 6
# PASTE THE NEXT PART DIRECTLY BELOW THIS LINE.
# ============================================================
# ============================================================
# PART 7: VISUALIZATION AND GRAPH REPORTING
# ============================================================

def generate_dataset_visualizations(
    dataframe: pd.DataFrame,
    timestamp_dataframe: pd.DataFrame,
) -> list[str]:
    """
    Generate static analytical figures for dataset assessment.

    Figures are saved locally and are never displayed interactively.
    Visualization failures are logged without hiding the main analysis.
    """
    generated_figures: list[str] = []

    try:
        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt

    except ImportError:
        LOGGER.warning(
            "Matplotlib is unavailable. Visualization generation skipped."
        )
        return generated_figures

    create_output_directories()

    def save_figure(figure: Any, filename: str) -> None:
        """Save a figure and release its memory."""
        output_path = FIGURE_DIR / filename

        try:
            figure.savefig(
                output_path,
                dpi=150,
                bbox_inches="tight",
                facecolor="white",
            )
            generated_figures.append(str(output_path.relative_to(BASE_DIR)))
            LOGGER.info("Visualization saved: %s", output_path.name)

        except (OSError, ValueError) as exc:
            LOGGER.warning(
                "Could not save visualization %s: %s",
                filename,
                exc,
            )

        finally:
            plt.close(figure)

    def get_numeric_demand(
        source_dataframe: pd.DataFrame,
        column_name: str,
    ) -> pd.Series:
        """Return valid numeric demand observations."""
        if column_name not in source_dataframe.columns:
            return pd.Series(dtype="float64")

        values = pd.to_numeric(
            source_dataframe[column_name],
            errors="coerce",
        )

        values = values.replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        return values

    try:
        # ----------------------------------------------------
        # 1. Demand distribution histograms
        # ----------------------------------------------------
        for demand_column in DEMAND_COLUMNS:
            demand_values = get_numeric_demand(
                dataframe,
                demand_column,
            )

            if demand_values.empty:
                LOGGER.warning(
                    "Skipping histogram for %s: no valid observations.",
                    demand_column,
                )
                continue

            figure, axis = plt.subplots(figsize=(10, 6))

            axis.hist(
                demand_values,
                bins=HISTOGRAM_BINS,
                edgecolor="black",
                alpha=0.75,
            )

            axis.set_title(
                f"Distribution of {demand_column}",
                fontsize=14,
            )
            axis.set_xlabel(demand_column)
            axis.set_ylabel("Observation Count")
            axis.grid(axis="y", alpha=0.25)

            save_figure(
                figure,
                f"{demand_column.lower().replace(' ', '_')}_distribution.png",
            )

        # ----------------------------------------------------
        # 2. Daily demand trends
        # ----------------------------------------------------
        timestamp_column = "_Analysis_Timestamp_UTC"

        if (
            timestamp_column in timestamp_dataframe.columns
            and not timestamp_dataframe.empty
        ):
            valid_timestamp_data = timestamp_dataframe.copy()

            valid_timestamp_data[timestamp_column] = pd.to_datetime(
                valid_timestamp_data[timestamp_column],
                errors="coerce",
                utc=True,
            )

            valid_timestamp_data = valid_timestamp_data.dropna(
                subset=[timestamp_column]
            )

            for demand_column in DEMAND_COLUMNS:
                if demand_column not in valid_timestamp_data.columns:
                    continue

                valid_timestamp_data[demand_column] = pd.to_numeric(
                    valid_timestamp_data[demand_column],
                    errors="coerce",
                )

                valid_timestamp_data[demand_column] = (
                    valid_timestamp_data[demand_column].replace(
                        [np.inf, -np.inf],
                        np.nan,
                    )
                )

                valid_values = valid_timestamp_data.dropna(
                    subset=[demand_column]
                )

                if valid_values.empty:
                    continue

                daily_demand = (
                    valid_values
                    .set_index(timestamp_column)[demand_column]
                    .resample("D")
                    .agg(["sum", "count"])
                    .reset_index()
                )

                daily_demand = daily_demand.dropna(
                    subset=["sum"]
                )

                if daily_demand.empty:
                    continue

                figure, axis = plt.subplots(figsize=(13, 6))

                axis.plot(
                    daily_demand[timestamp_column],
                    daily_demand["sum"],
                    linewidth=1.2,
                )

                axis.set_title(
                    f"Daily Observed {demand_column} Trend (UTC)",
                    fontsize=14,
                )
                axis.set_xlabel("Date (UTC)")
                axis.set_ylabel(f"Daily {demand_column} Sum")
                axis.grid(alpha=0.3)

                figure.autofmt_xdate()

                save_figure(
                    figure,
                    f"{demand_column.lower().replace(' ', '_')}_daily_trend.png",
                )

            # -----------------------------------------------
            # 3. Average demand by UTC hour
            # -----------------------------------------------
            hourly_data = valid_timestamp_data.copy()

            hourly_data["Hour_UTC"] = (
                hourly_data[timestamp_column].dt.hour
            )

            for demand_column in DEMAND_COLUMNS:
                if demand_column not in hourly_data.columns:
                    continue

                hourly_data[demand_column] = pd.to_numeric(
                    hourly_data[demand_column],
                    errors="coerce",
                )

                hourly_data[demand_column] = (
                    hourly_data[demand_column].replace(
                        [np.inf, -np.inf],
                        np.nan,
                    )
                )

                hourly_summary = (
                    hourly_data
                    .groupby("Hour_UTC")[demand_column]
                    .mean()
                    .reindex(range(24))
                )

                if not hourly_summary.notna().any():
                    continue

                figure, axis = plt.subplots(figsize=(11, 6))

                axis.bar(
                    hourly_summary.index,
                    hourly_summary.values,
                    width=0.8,
                    edgecolor="black",
                    alpha=0.8,
                )

                axis.set_title(
                    f"Average {demand_column} by UTC Hour",
                    fontsize=14,
                )
                axis.set_xlabel("Hour of Day (UTC)")
                axis.set_ylabel(f"Average {demand_column}")
                axis.set_xticks(range(24))
                axis.grid(axis="y", alpha=0.25)

                save_figure(
                    figure,
                    f"{demand_column.lower().replace(' ', '_')}_hourly_average.png",
                )

            # -----------------------------------------------
            # 4. Observed timestamp interval distribution
            # -----------------------------------------------
            observed_timestamps = (
                valid_timestamp_data[timestamp_column]
                .sort_values()
                .drop_duplicates()
            )

            interval_minutes = (
                observed_timestamps.diff()
                .dt.total_seconds()
                .div(60)
                .dropna()
            )

            interval_minutes = interval_minutes[
                np.isfinite(interval_minutes)
                & (interval_minutes > 0)
            ]

            if not interval_minutes.empty:
                figure, axis = plt.subplots(figsize=(11, 6))

                upper_limit = float(
                    interval_minutes.quantile(0.99)
                )

                if not np.isfinite(upper_limit) or upper_limit <= 0:
                    upper_limit = float(interval_minutes.max())

                displayed_intervals = interval_minutes[
                    interval_minutes <= upper_limit
                ]

                if displayed_intervals.empty:
                    displayed_intervals = interval_minutes

                axis.hist(
                    displayed_intervals,
                    bins=50,
                    edgecolor="black",
                    alpha=0.75,
                )

                axis.axvline(
                    EXPECTED_INTERVAL_MINUTES,
                    linestyle="--",
                    linewidth=2,
                    label=(
                        f"Expected interval: "
                        f"{EXPECTED_INTERVAL_MINUTES} minutes"
                    ),
                )

                axis.set_title(
                    "Observed Timestamp Interval Distribution",
                    fontsize=14,
                )
                axis.set_xlabel("Interval Between Observations (Minutes)")
                axis.set_ylabel("Interval Count")
                axis.legend()
                axis.grid(axis="y", alpha=0.25)

                save_figure(
                    figure,
                    "timestamp_interval_distribution.png",
                )

        else:
            LOGGER.warning(
                "Timestamp analysis data is empty or unavailable. "
                "Time-based visualizations were skipped."
            )

    except Exception:
        LOGGER.exception(
            "An unexpected error occurred during visualization generation."
        )

    LOGGER.info(
        "Visualization generation completed. Figures created: %d",
        len(generated_figures),
    )

    return generated_figures


# END OF PART 7
# ============================================================
# ============================================================
# PART 8: COMPREHENSIVE REPORTING AND ANALYSIS ORCHESTRATION
# ============================================================

def build_comprehensive_analysis_report(
    dataframe: pd.DataFrame,
    structural_report: dict[str, Any],
    duplicate_report: dict[str, Any],
    timestamp_report: dict[str, Any],
    interval_report: dict[str, Any],
    demand_summary: dict[str, Any],
    time_pattern_summary: dict[str, Any],
    relationship_summary: dict[str, Any],
    numeric_statistics: pd.DataFrame,
    numeric_correlation_summary: dict[str, Any],
    demand_distribution_summary: dict[str, Any],
    outlier_summary: dict[str, Any],
    daily_extreme_summary: dict[str, Any],
    issue_register: pd.DataFrame,
    visualization_files: list[str],
) -> dict[str, Any]:
    """
    Combine the outputs of individual analysis modules into one
    comprehensive, machine-readable dataset assessment report.

    The report describes observed ticket demand and data quality.
    It does not claim to predict ferry capacity, boarding counts,
    exact equipment failures, or passenger queue lengths.
    """

    create_output_directories()

    report: dict[str, Any] = {
        "report_metadata": {
            "report_name": "Ferry Ticket Demand Dataset Assessment",
            "report_version": REPORT_VERSION,
            "generated_at_utc": get_utc_timestamp(),
            "dataset_name": DATASET_PATH.name,
            "dataset_path": str(DATASET_PATH),
            "analysis_type": "Exploratory Data Analysis and Data Quality",
            "forecasting_scope": {
                "target_variables": list(DEMAND_COLUMNS),
                "supported_horizons": [
                    "15 minutes",
                    "30 minutes",
                    "1 hour",
                    "2 hours",
                ],
                "forecasting_status": (
                    "Dataset assessment only; forecasting models "
                    "must be trained and evaluated separately."
                ),
            },
        },
        "dataset_overview": {
            "row_count": safe_int(len(dataframe)),
            "column_count": safe_int(len(dataframe.columns)),
            "column_names": list(dataframe.columns),
            "memory_usage_bytes": safe_int(
                dataframe.memory_usage(deep=True).sum()
            ),
            "memory_usage_mb": round(
                float(
                    dataframe.memory_usage(deep=True).sum()
                ) / (1024 ** 2),
                3,
            ),
            "dataset_file_exists": DATASET_PATH.is_file(),
            "dataset_file_size_bytes": (
                safe_int(DATASET_PATH.stat().st_size)
                if DATASET_PATH.is_file()
                else None
            ),
        },
        "structural_validation": structural_report,
        "duplicate_analysis": duplicate_report,
        "timestamp_quality": timestamp_report,
        "timestamp_interval_analysis": interval_report,
        "demand_quality": demand_summary,
        "time_pattern_analysis": time_pattern_summary,
        "demand_relationship_analysis": relationship_summary,
        "numeric_statistics": numeric_statistics.to_dict(
            orient="records"
        ),
        "numeric_correlation_analysis": numeric_correlation_summary,
        "demand_distribution_analysis": demand_distribution_summary,
        "outlier_analysis": outlier_summary,
        "daily_demand_extremes": daily_extreme_summary,
        "data_quality_issue_register": issue_register.to_dict(
            orient="records"
        ),
        "generated_visualizations": visualization_files,
        "analysis_limitations": [
            (
                "Observed ticket sales and redemptions do not directly "
                "represent physical ferry capacity."
            ),
            (
                "Missing timestamps are not automatically interpreted "
                "as zero-demand periods."
            ),
            (
                "The dataset does not establish exact passenger queues, "
                "boarding counts, or ferry occupancy."
            ),
            (
                "Exploratory correlations do not establish causation."
            ),
            (
                "Future demand forecasts require time-aware validation "
                "and evaluation against observations not used for training."
            ),
            (
                "Forecast accuracy must be measured independently "
                "for each demand variable and forecast horizon."
            ),
        ],
    }

    report_path = REPORT_DIR / "comprehensive_dataset_analysis.json"

    save_json_report(
        report,
        report_path,
    )

    LOGGER.info(
        "Comprehensive analysis report saved: %s",
        report_path,
    )

    return report


def create_analysis_run_manifest(
    started_at_utc: str,
    completed_at_utc: str,
    elapsed_seconds: float,
    dataframe: pd.DataFrame,
    report: dict[str, Any],
    generated_figures: list[str],
) -> dict[str, Any]:
    """
    Create a reproducibility manifest containing execution metadata,
    generated artifacts, dataset identity, and completion status.
    """

    create_output_directories()

    generated_artifacts: list[dict[str, Any]] = []

    for directory in (OUTPUT_DIR, REPORT_DIR, FIGURE_DIR, LOG_DIR):
        if not directory.exists():
            continue

        for artifact_path in directory.rglob("*"):
            if not artifact_path.is_file():
                continue

            if artifact_path.name in {
                "analysis_run_manifest.json",
                "analysis_run_status.json",
            }:
                continue

            try:
                relative_path = str(
                    artifact_path.relative_to(BASE_DIR)
                )

                generated_artifacts.append(
                    {
                        "path": relative_path,
                        "size_bytes": safe_int(
                            artifact_path.stat().st_size
                        ),
                        "extension": artifact_path.suffix.lower(),
                    }
                )

            except OSError as exc:
                LOGGER.warning(
                    "Unable to inspect artifact %s: %s",
                    artifact_path,
                    exc,
                )

    manifest: dict[str, Any] = {
        "manifest_version": "1.0.0",
        "execution_status": "SUCCESS",
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "elapsed_seconds": round(
            max(0.0, float(elapsed_seconds)),
            3,
        ),
        "runtime_environment": {
            "python_version": sys.version.split()[0],
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "platform": sys.platform,
        },
        "dataset": {
            "file_name": DATASET_PATH.name,
            "file_size_bytes": (
                safe_int(DATASET_PATH.stat().st_size)
                if DATASET_PATH.is_file()
                else None
            ),
            "row_count": safe_int(len(dataframe)),
            "column_count": safe_int(len(dataframe.columns)),
            "columns": list(dataframe.columns),
        },
        "report": {
            "report_name": report.get(
                "report_metadata", {}
            ).get("report_name"),
            "report_version": report.get(
                "report_metadata", {}
            ).get("report_version"),
            "report_path": str(
                (
                    REPORT_DIR
                    / "comprehensive_dataset_analysis.json"
                ).relative_to(BASE_DIR)
            ),
        },
        "generated_visualizations": generated_figures,
        "generated_artifacts": generated_artifacts,
    }

    save_json_report(
        manifest,
        REPORT_DIR / "analysis_run_manifest.json",
    )

    return manifest


def write_analysis_status(
    status: str,
    message: str,
    started_at_utc: str,
    completed_at_utc: str | None = None,
    elapsed_seconds: float | None = None,
    error_type: str | None = None,
) -> dict[str, Any]:
    """
    Persist the final execution status.

    This file helps distinguish a successful assessment from an
    interrupted or failed execution.
    """

    status_payload: dict[str, Any] = {
        "status": status,
        "message": message,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "elapsed_seconds": (
            round(float(elapsed_seconds), 3)
            if elapsed_seconds is not None
            else None
        ),
        "error_type": error_type,
        "dataset_path": str(DATASET_PATH),
        "report_directory": str(REPORT_DIR),
        "log_directory": str(LOG_DIR),
    }

    save_json_report(
        status_payload,
        REPORT_DIR / "analysis_run_status.json",
    )

    return status_payload


def log_analysis_dataframe_summary(
    dataframe: pd.DataFrame,
    label: str,
) -> None:
    """Write a concise DataFrame summary to the application log."""

    LOGGER.info(
        "%s | rows=%s | columns=%s",
        label,
        format_number(len(dataframe)),
        format_number(len(dataframe.columns)),
    )

    if dataframe.empty:
        LOGGER.warning("%s produced no records.", label)
        return

    LOGGER.debug(
        "%s column names: %s",
        label,
        list(dataframe.columns),
    )


# END OF PART 8
# ============================================================
# ============================================================
# PART 9: MAIN EXECUTION PIPELINE
# ============================================================

def main() -> int:
    """
    Execute the complete dataset analysis pipeline.

    Execution sequence:
        1. Initialize output directories and logging.
        2. Load and validate the source dataset.
        3. Analyze structure, duplicates, and timestamps.
        4. Evaluate numeric fields and demand quality.
        5. Analyze time patterns, correlations, and distributions.
        6. Detect unusual observations and register data issues.
        7. Generate visualizations and comprehensive reports.
        8. Save execution status and reproducibility metadata.

    Returns:
        int: 0 for successful completion, 1 for failure.
    """

    started_at_epoch = time.perf_counter()
    started_at_utc = get_utc_timestamp()

    create_output_directories()
    configure_logging()

    LOGGER.info("=" * 78)
    LOGGER.info("FERRY TICKET DEMAND DATASET ANALYSIS")
    LOGGER.info("Report version: %s", REPORT_VERSION)
    LOGGER.info("Dataset: %s", DATASET_PATH)
    LOGGER.info("=" * 78)

    try:
        # ----------------------------------------------------
        # STEP 1: LOAD SOURCE DATASET
        # ----------------------------------------------------
        LOGGER.info("STEP 1/10: Loading source dataset.")

        dataframe = load_dataset()

        if dataframe.empty:
            raise ValueError(
                "The source dataset contains no observations."
            )

        log_analysis_dataframe_summary(
            dataframe,
            "Source dataset",
        )

        # ----------------------------------------------------
        # STEP 2: VALIDATE DATASET STRUCTURE
        # ----------------------------------------------------
        LOGGER.info("STEP 2/10: Validating dataset structure.")

        structural_report = validate_dataset_structure(
            dataframe
        )

        save_json_report(
            structural_report,
            REPORT_DIR / "dataset_structure_validation.json",
        )

        analyze_data_types(dataframe)

        # ----------------------------------------------------
        # STEP 3: CHECK DUPLICATES
        # ----------------------------------------------------
        LOGGER.info("STEP 3/10: Analyzing duplicate observations.")

        duplicate_report = analyze_duplicates(
            dataframe
        )

        save_json_report(
            duplicate_report,
            REPORT_DIR / "duplicate_analysis.json",
        )

        # ----------------------------------------------------
        # STEP 4: PREPARE AND ANALYZE TIMESTAMPS
        # ----------------------------------------------------
        LOGGER.info("STEP 4/10: Analyzing timestamp quality.")

        timestamp_dataframe, timestamp_report = (
            prepare_timestamp_analysis(dataframe)
        )

        interval_report, interval_details = (
            analyze_timestamp_intervals(
                timestamp_dataframe
            )
        )

        save_json_report(
            timestamp_report,
            REPORT_DIR / "timestamp_quality_report.json",
        )

        save_json_report(
            interval_report,
            REPORT_DIR / "timestamp_interval_analysis.json",
        )

        save_dataframe_csv(
            interval_details,
            OUTPUT_DIR / "timestamp_interval_details.csv",
        )

        log_analysis_dataframe_summary(
            timestamp_dataframe,
            "Timestamp analysis data",
        )

        # ----------------------------------------------------
        # STEP 5: NUMERIC AND DEMAND DATA QUALITY
        # ----------------------------------------------------
        LOGGER.info(
            "STEP 5/10: Analyzing numeric fields and demand quality."
        )

        numeric_statistics = analyze_numeric_columns(
            dataframe
        )

        demand_summary, demand_issue_dataframe = (
            analyze_demand_values(dataframe)
        )

        non_numeric_profile = analyze_non_numeric_columns(
            dataframe
        )

        log_analysis_dataframe_summary(
            numeric_statistics,
            "Numeric statistical profile",
        )

        log_analysis_dataframe_summary(
            demand_issue_dataframe,
            "Demand quality issues",
        )

        log_analysis_dataframe_summary(
            non_numeric_profile,
            "Non-numeric column profile",
        )

        save_json_report(
            demand_summary,
            REPORT_DIR / "demand_quality_summary.json",
        )

        # ----------------------------------------------------
        # STEP 6: DEMAND RELATIONSHIPS AND TIME PATTERNS
        # ----------------------------------------------------
        LOGGER.info(
            "STEP 6/10: Analyzing demand relationships and patterns."
        )

        relationship_summary = analyze_demand_relationship(
            dataframe
        )

        time_pattern_summary = analyze_time_patterns(
            timestamp_dataframe
        )

        daily_extreme_summary = analyze_daily_demand_extremes(
            timestamp_dataframe
        )

        save_json_report(
            relationship_summary,
            REPORT_DIR / "demand_relationship_analysis.json",
        )

        save_json_report(
            time_pattern_summary,
            REPORT_DIR / "time_pattern_analysis.json",
        )

        save_json_report(
            daily_extreme_summary,
            REPORT_DIR / "daily_demand_extremes_summary.json",
        )

        # ----------------------------------------------------
        # STEP 7: CORRELATIONS AND DEMAND DISTRIBUTIONS
        # ----------------------------------------------------
        LOGGER.info(
            "STEP 7/10: Calculating correlations and distributions."
        )

        numeric_correlation_summary = (
            analyze_numeric_correlations(dataframe)
        )

        demand_distribution_summary = (
            analyze_demand_distributions(dataframe)
        )

        highest_demand_records = (
            analyze_high_demand_observations(
                dataframe,
                top_n=100,
            )
        )

        log_analysis_dataframe_summary(
            highest_demand_records,
            "Highest observed demand records",
        )

        save_json_report(
            numeric_correlation_summary,
            REPORT_DIR / "numeric_correlation_summary.json",
        )

        save_json_report(
            demand_distribution_summary,
            REPORT_DIR / "demand_distribution_summary.json",
        )

        # ----------------------------------------------------
        # STEP 8: OUTLIERS AND DATA QUALITY ISSUE REGISTER
        # ----------------------------------------------------
        LOGGER.info(
            "STEP 8/10: Detecting outliers and registering issues."
        )

        outlier_summary, outlier_records = analyze_outliers(
            dataframe
        )

        structural_issue_report = structural_report

        issue_register = build_data_quality_issue_register(
            dataframe=dataframe,
            structural_report=structural_issue_report,
            timestamp_report=timestamp_report,
            demand_issue_dataframe=demand_issue_dataframe,
            duplicate_report=duplicate_report,
        )

        log_analysis_dataframe_summary(
            outlier_records,
            "Detected demand outliers",
        )

        log_analysis_dataframe_summary(
            issue_register,
            "Dataset quality issue register",
        )

        save_json_report(
            outlier_summary,
            REPORT_DIR / "outlier_analysis_summary.json",
        )

        save_dataframe_csv(
            outlier_records,
            OUTPUT_DIR / "demand_outlier_records.csv",
        )

        save_dataframe_csv(
            highest_demand_records,
            OUTPUT_DIR / "highest_observed_demand_records.csv",
        )

        save_dataframe_csv(
            issue_register,
            OUTPUT_DIR / "dataset_quality_issue_register.csv",
        )

        # ----------------------------------------------------
        # STEP 9: VISUALIZATIONS AND COMPREHENSIVE REPORT
        # ----------------------------------------------------
        LOGGER.info(
            "STEP 9/10: Generating visualizations and final reports."
        )

        visualization_files = generate_dataset_visualizations(
            dataframe=dataframe,
            timestamp_dataframe=timestamp_dataframe,
        )

        executive_summary = build_executive_summary(
            dataframe=dataframe,
            structural_report=structural_report,
            duplicate_report=duplicate_report,
            timestamp_report=timestamp_report,
            demand_summary=demand_summary,
            time_pattern_summary=time_pattern_summary,
            relationship_summary=relationship_summary,
            issue_register=issue_register,
        )

        save_json_report(
            executive_summary,
            REPORT_DIR / "dataset_analysis_executive_summary.json",
        )

        comprehensive_report = (
            build_comprehensive_analysis_report(
                dataframe=dataframe,
                structural_report=structural_report,
                duplicate_report=duplicate_report,
                timestamp_report=timestamp_report,
                interval_report=interval_report,
                demand_summary=demand_summary,
                time_pattern_summary=time_pattern_summary,
                relationship_summary=relationship_summary,
                numeric_statistics=numeric_statistics,
                numeric_correlation_summary=(
                    numeric_correlation_summary
                ),
                demand_distribution_summary=(
                    demand_distribution_summary
                ),
                outlier_summary=outlier_summary,
                daily_extreme_summary=daily_extreme_summary,
                issue_register=issue_register,
                visualization_files=visualization_files,
            )
        )

        # ----------------------------------------------------
        # STEP 10: FINAL STATUS AND REPRODUCIBILITY MANIFEST
        # ----------------------------------------------------
        LOGGER.info(
            "STEP 10/10: Saving execution metadata."
        )

        completed_at_utc = get_utc_timestamp()

        elapsed_seconds = (
            time.perf_counter() - started_at_epoch
        )

        manifest = create_analysis_run_manifest(
            started_at_utc=started_at_utc,
            completed_at_utc=completed_at_utc,
            elapsed_seconds=elapsed_seconds,
            dataframe=dataframe,
            report=comprehensive_report,
            generated_figures=visualization_files,
        )

        write_analysis_status(
            status="SUCCESS",
            message=(
                "Dataset analysis completed successfully."
            ),
            started_at_utc=started_at_utc,
            completed_at_utc=completed_at_utc,
            elapsed_seconds=elapsed_seconds,
        )

        # ----------------------------------------------------
        # FINAL EXECUTION SUMMARY
        # ----------------------------------------------------
        LOGGER.info("=" * 78)
        LOGGER.info("DATASET ANALYSIS COMPLETED SUCCESSFULLY")
        LOGGER.info("=" * 78)

        LOGGER.info(
            "Dataset rows: %s",
            format_number(len(dataframe)),
        )

        LOGGER.info(
            "Dataset columns: %s",
            format_number(len(dataframe.columns)),
        )

        LOGGER.info(
            "Detected outlier records: %s",
            format_number(len(outlier_records)),
        )

        LOGGER.info(
            "Registered data quality issues: %s",
            format_number(len(issue_register)),
        )

        LOGGER.info(
            "Visualizations generated: %s",
            format_number(len(visualization_files)),
        )

        LOGGER.info(
            "Generated artifacts: %s",
            format_number(len(manifest["generated_artifacts"])),
        )

        LOGGER.info(
            "Execution time: %.2f seconds",
            elapsed_seconds,
        )

        LOGGER.info(
            "Executive summary: %s",
            REPORT_DIR / "dataset_analysis_executive_summary.json",
        )

        LOGGER.info(
            "Comprehensive report: %s",
            REPORT_DIR / "comprehensive_dataset_analysis.json",
        )

        LOGGER.info(
            "Run manifest: %s",
            REPORT_DIR / "analysis_run_manifest.json",
        )

        LOGGER.info("=" * 78)

        return 0

    except KeyboardInterrupt:
        elapsed_seconds = (
            time.perf_counter() - started_at_epoch
        )

        LOGGER.warning(
            "Analysis interrupted by the user."
        )

        try:
            write_analysis_status(
                status="INTERRUPTED",
                message="Execution was interrupted by the user.",
                started_at_utc=started_at_utc,
                completed_at_utc=get_utc_timestamp(),
                elapsed_seconds=elapsed_seconds,
                error_type="KeyboardInterrupt",
            )
        except Exception:
            LOGGER.exception(
                "Unable to write interrupted execution status."
            )

        return 1

    except Exception as exc:
        elapsed_seconds = (
            time.perf_counter() - started_at_epoch
        )

        LOGGER.exception(
            "Dataset analysis failed: %s",
            exc,
        )

        failure_report = {
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "started_at_utc": started_at_utc,
            "failed_at_utc": get_utc_timestamp(),
            "elapsed_seconds": round(
                elapsed_seconds,
                3,
            ),
            "dataset_path": str(DATASET_PATH),
            "report_directory": str(REPORT_DIR),
        }

        try:
            save_json_report(
                failure_report,
                REPORT_DIR / "dataset_analysis_failure.json",
            )

            write_analysis_status(
                status="FAILED",
                message=str(exc),
                started_at_utc=started_at_utc,
                completed_at_utc=get_utc_timestamp(),
                elapsed_seconds=elapsed_seconds,
                error_type=type(exc).__name__,
            )

        except Exception:
            LOGGER.exception(
                "Unable to persist the failure report."
            )

        return 1




# END OF PART 9
# ============================================================
# ============================================================
# PART 10: FINAL REPORT VALIDATION AND ARTIFACT INVENTORY
# ============================================================

def validate_generated_artifacts(
    required_artifacts: list[Path] | None = None,
) -> dict[str, Any]:
    """
    Validate the presence, readability, and basic integrity of
    important artifacts generated by the analysis pipeline.

    This function does not rerun the analysis. It checks the files
    already created by the completed execution.
    """

    if required_artifacts is None:
        required_artifacts = [
            REPORT_DIR / "dataset_structure_validation.json",
            REPORT_DIR / "duplicate_analysis.json",
            REPORT_DIR / "timestamp_quality_report.json",
            REPORT_DIR / "timestamp_interval_analysis.json",
            REPORT_DIR / "demand_quality_summary.json",
            REPORT_DIR / "demand_relationship_analysis.json",
            REPORT_DIR / "time_pattern_analysis.json",
            REPORT_DIR / "numeric_correlation_summary.json",
            REPORT_DIR / "demand_distribution_summary.json",
            REPORT_DIR / "outlier_analysis_summary.json",
            REPORT_DIR / "dataset_analysis_executive_summary.json",
            REPORT_DIR / "comprehensive_dataset_analysis.json",
            REPORT_DIR / "analysis_run_manifest.json",
            OUTPUT_DIR / "dataset_column_profile_detailed.csv",
            OUTPUT_DIR / "numeric_statistical_profile.csv",
            OUTPUT_DIR / "non_numeric_column_profile.csv",
            OUTPUT_DIR / "demand_data_quality_issues.csv",
            OUTPUT_DIR / "demand_correlation_matrix.csv",
            OUTPUT_DIR / "numeric_pearson_correlation_matrix.csv",
            OUTPUT_DIR / "numeric_spearman_correlation_matrix.csv",
            OUTPUT_DIR / "numeric_correlation_pairs.csv",
            OUTPUT_DIR / "demand_distribution_summary.csv",
            OUTPUT_DIR / "highest_observed_demand_records.csv",
            OUTPUT_DIR / "demand_outlier_records.csv",
            OUTPUT_DIR / "dataset_quality_issue_register.csv",
            OUTPUT_DIR / "timestamp_interval_details.csv",
        ]

    artifact_results: list[dict[str, Any]] = []

    for artifact_path in required_artifacts:
        result: dict[str, Any] = {
            "path": str(artifact_path),
            "exists": False,
            "size_bytes": 0,
            "readable": False,
            "valid": False,
            "error": None,
        }

        try:
            if not artifact_path.is_file():
                result["error"] = "Artifact does not exist."
                artifact_results.append(result)
                continue

            result["exists"] = True
            result["size_bytes"] = safe_int(
                artifact_path.stat().st_size
            )

            if result["size_bytes"] == 0:
                result["error"] = "Artifact is empty."
                artifact_results.append(result)
                continue

            suffix = artifact_path.suffix.lower()

            if suffix == ".json":
                with artifact_path.open(
                    "r",
                    encoding="utf-8",
                ) as file_handle:
                    json.load(file_handle)

            elif suffix == ".csv":
                # Read a small sample to verify that the file can
                # be parsed without loading large files entirely.
                pd.read_csv(
                    artifact_path,
                    nrows=5,
                    encoding="utf-8",
                )

            else:
                # For other artifact formats, verify that the
                # file is accessible and contains data.
                with artifact_path.open("rb") as file_handle:
                    file_handle.read(1)

            result["readable"] = True
            result["valid"] = True

        except (
            OSError,
            ValueError,
            UnicodeError,
            json.JSONDecodeError,
            pd.errors.ParserError,
            pd.errors.EmptyDataError,
        ) as exc:
            result["error"] = (
                f"{type(exc).__name__}: {str(exc)}"
            )

            LOGGER.warning(
                "Artifact validation failed for %s: %s",
                artifact_path,
                exc,
            )

        artifact_results.append(result)

    valid_count = sum(
        1 for item in artifact_results
        if item["valid"]
    )

    invalid_count = (
        len(artifact_results) - valid_count
    )

    validation_report = {
        "validation_timestamp_utc": get_utc_timestamp(),
        "total_artifacts_checked": len(artifact_results),
        "valid_artifacts": valid_count,
        "invalid_artifacts": invalid_count,
        "validation_passed": invalid_count == 0,
        "artifacts": artifact_results,
    }

    save_json_report(
        validation_report,
        REPORT_DIR / "artifact_validation_report.json",
    )

    LOGGER.info(
        "Artifact validation completed: %s valid, %s invalid.",
        valid_count,
        invalid_count,
    )

    return validation_report


def generate_artifact_inventory() -> pd.DataFrame:
    """
    Build a searchable inventory of generated output files.

    The inventory records relative paths, file sizes, extensions,
    modification timestamps, and whether each file is empty.
    """

    inventory_records: list[dict[str, Any]] = []

    directories_to_scan = [
        OUTPUT_DIR,
        REPORT_DIR,
        FIGURE_DIR,
        LOG_DIR,
    ]

    for directory in directories_to_scan:
        if not directory.exists():
            continue

        for file_path in directory.rglob("*"):
            if not file_path.is_file():
                continue

            try:
                stat_result = file_path.stat()

                try:
                    relative_path = str(
                        file_path.relative_to(BASE_DIR)
                    )
                except ValueError:
                    relative_path = str(file_path)

                inventory_records.append(
                    {
                        "File_Name": file_path.name,
                        "Relative_Path": relative_path,
                        "File_Type": (
                            file_path.suffix.lower()
                            if file_path.suffix
                            else "no_extension"
                        ),
                        "Size_Bytes": safe_int(
                            stat_result.st_size
                        ),
                        "Size_MB": round(
                            stat_result.st_size / (1024 ** 2),
                            4,
                        ),
                        "Is_Empty": (
                            stat_result.st_size == 0
                        ),
                        "Modified_At_UTC": (
                            datetime.fromtimestamp(
                                stat_result.st_mtime,
                                tz=timezone.utc,
                            ).isoformat()
                        ),
                    }
                )

            except OSError as exc:
                LOGGER.warning(
                    "Unable to inventory file %s: %s",
                    file_path,
                    exc,
                )

    inventory_dataframe = pd.DataFrame(
        inventory_records,
        columns=[
            "File_Name",
            "Relative_Path",
            "File_Type",
            "Size_Bytes",
            "Size_MB",
            "Is_Empty",
            "Modified_At_UTC",
        ],
    )

    if not inventory_dataframe.empty:
        inventory_dataframe = (
            inventory_dataframe
            .sort_values(
                by=["File_Type", "File_Name"],
                kind="stable",
            )
            .reset_index(drop=True)
        )

    save_dataframe_csv(
        inventory_dataframe,
        REPORT_DIR / "generated_artifact_inventory.csv",
    )

    LOGGER.info(
        "Artifact inventory generated with %s records.",
        len(inventory_dataframe),
    )

    return inventory_dataframe


def summarize_analysis_outputs(
    artifact_inventory: pd.DataFrame,
    artifact_validation: dict[str, Any],
) -> dict[str, Any]:
    """
    Produce a compact overview of generated files and their
    validation status for logs and downstream applications.
    """

    if artifact_inventory.empty:
        file_type_summary: dict[str, int] = {}
        total_size_bytes = 0
    else:
        file_type_summary = {
            str(file_type): safe_int(count)
            for file_type, count in (
                artifact_inventory["File_Type"]
                .value_counts()
                .items()
            )
        }

        total_size_bytes = safe_int(
            artifact_inventory["Size_Bytes"].sum()
        )

    summary = {
        "generated_at_utc": get_utc_timestamp(),
        "inventory": {
            "total_files": safe_int(
                len(artifact_inventory)
            ),
            "total_size_bytes": total_size_bytes,
            "total_size_mb": round(
                total_size_bytes / (1024 ** 2),
                3,
            ),
            "file_type_counts": file_type_summary,
            "empty_file_count": (
                safe_int(
                    artifact_inventory["Is_Empty"].sum()
                )
                if not artifact_inventory.empty
                else 0
            ),
        },
        "validation": {
            "validation_passed": bool(
                artifact_validation.get(
                    "validation_passed",
                    False,
                )
            ),
            "total_checked": safe_int(
                artifact_validation.get(
                    "total_artifacts_checked",
                    0,
                )
            ),
            "valid_artifacts": safe_int(
                artifact_validation.get(
                    "valid_artifacts",
                    0,
                )
            ),
            "invalid_artifacts": safe_int(
                artifact_validation.get(
                    "invalid_artifacts",
                    0,
                )
            ),
        },
    }

    save_json_report(
        summary,
        REPORT_DIR / "analysis_output_inventory_summary.json",
    )

    return summary


# END OF PART 10
# ============================================================
# ============================================================
# PART 11: ADVANCED DATA QUALITY AND CONSISTENCY CHECKS
# ============================================================

def run_advanced_data_quality_checks(
    dataframe: pd.DataFrame,
    timestamp_dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Perform additional quality checks without modifying the
    original dataset.

    Checks include:
        - Missing values and missing-value percentages.
        - Infinite values in numeric columns.
        - Constant and near-constant columns.
        - Invalid timestamp values.
        - Duplicate timestamps.
        - Negative and non-finite demand values.
        - Potentially inconsistent demand relationships.

    These checks identify observations for review. They do not
    automatically delete records or replace missing observations.
    """

    LOGGER.info("Starting advanced data quality checks.")

    quality_records: list[dict[str, Any]] = []

    def add_issue(
        category: str,
        severity: str,
        column: str,
        issue: str,
        affected_rows: int,
        details: str,
    ) -> None:
        quality_records.append(
            {
                "Category": category,
                "Severity": severity,
                "Column": column,
                "Issue": issue,
                "Affected_Rows": safe_int(affected_rows),
                "Affected_Percentage": calculate_percentage(
                    affected_rows,
                    len(dataframe),
                ),
                "Details": details,
            }
        )

    total_rows = len(dataframe)

    # --------------------------------------------------------
    # 1. Missing-value analysis
    # --------------------------------------------------------
    for column in dataframe.columns:
        missing_count = safe_int(
            dataframe[column].isna().sum()
        )

        if missing_count > 0:
            missing_percentage = calculate_percentage(
                missing_count,
                total_rows,
            )

            severity = (
                "HIGH"
                if missing_percentage >= 20
                else "MEDIUM"
                if missing_percentage >= 5
                else "LOW"
            )

            add_issue(
                category="MISSING_VALUES",
                severity=severity,
                column=str(column),
                issue="Missing observations detected",
                affected_rows=missing_count,
                details=(
                    f"{missing_percentage}% of dataset rows "
                    "contain a missing value in this column."
                ),
            )

    # --------------------------------------------------------
    # 2. Numeric values: infinity and constant columns
    # --------------------------------------------------------
    numeric_columns = dataframe.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    for column in numeric_columns:
        numeric_values = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        infinite_mask = pd.Series(
            np.isinf(
                numeric_values.to_numpy(
                    dtype="float64",
                    na_value=np.nan,
                )
            ),
            index=dataframe.index,
        )

        infinite_count = safe_int(
            infinite_mask.sum()
        )

        if infinite_count:
            add_issue(
                category="NUMERIC_VALIDATION",
                severity="HIGH",
                column=str(column),
                issue="Infinite numeric values detected",
                affected_rows=infinite_count,
                details=(
                    "Infinite values can distort statistical "
                    "summaries, scaling, and model training."
                ),
            )

        non_missing_values = numeric_values.dropna()

        unique_count = safe_int(
            non_missing_values.nunique()
        )

        if len(non_missing_values) > 0 and unique_count <= 1:
            add_issue(
                category="FEATURE_VARIATION",
                severity="MEDIUM",
                column=str(column),
                issue="Constant or single-value column",
                affected_rows=len(non_missing_values),
                details=(
                    "This column contains no meaningful variation "
                    "among its non-missing observations."
                ),
            )

    # --------------------------------------------------------
    # 3. Demand-specific validation
    # --------------------------------------------------------
    for column in DEMAND_COLUMNS:
        if column not in dataframe.columns:
            add_issue(
                category="DEMAND_VALIDATION",
                severity="HIGH",
                column=column,
                issue="Required demand column missing",
                affected_rows=0,
                details=(
                    "The configured demand variable is not present "
                    "in the source dataset."
                ),
            )
            continue

        original_values = dataframe[column]

        numeric_values = pd.to_numeric(
            original_values,
            errors="coerce",
        )

        invalid_conversion_mask = (
            original_values.notna()
            & numeric_values.isna()
        )

        invalid_conversion_count = safe_int(
            invalid_conversion_mask.sum()
        )

        if invalid_conversion_count:
            add_issue(
                category="DEMAND_VALIDATION",
                severity="HIGH",
                column=column,
                issue="Non-numeric demand observations",
                affected_rows=invalid_conversion_count,
                details=(
                    "Some non-missing values cannot be converted "
                    "to numeric demand values."
                ),
            )

        infinite_mask = pd.Series(
            np.isinf(
                numeric_values.to_numpy(
                    dtype="float64",
                    na_value=np.nan,
                )
            ),
            index=dataframe.index,
        )

        infinite_count = safe_int(
            infinite_mask.sum()
        )

        if infinite_count:
            add_issue(
                category="DEMAND_VALIDATION",
                severity="HIGH",
                column=column,
                issue="Non-finite demand observations",
                affected_rows=infinite_count,
                details=(
                    "Infinite demand values must be investigated "
                    "before statistical analysis or forecasting."
                ),
            )

        negative_mask = (
            numeric_values.notna()
            & np.isfinite(numeric_values)
            & (numeric_values < 0)
        )

        negative_count = safe_int(
            negative_mask.sum()
        )

        if negative_count:
            add_issue(
                category="DEMAND_VALIDATION",
                severity="HIGH",
                column=column,
                issue="Negative demand observations",
                affected_rows=negative_count,
                details=(
                    "Negative ticket counts may indicate invalid "
                    "records, adjustments, or a different data meaning. "
                    "Investigate before deciding how to handle them."
                ),
            )

    # --------------------------------------------------------
    # 4. Timestamp validation
    # --------------------------------------------------------
    timestamp_column = "_Analysis_Timestamp_UTC"

    if timestamp_column in timestamp_dataframe.columns:
        timestamps = pd.to_datetime(
            timestamp_dataframe[timestamp_column],
            errors="coerce",
            utc=True,
        )

        invalid_timestamp_count = safe_int(
            timestamps.isna().sum()
        )

        if invalid_timestamp_count:
            add_issue(
                category="TIMESTAMP_VALIDATION",
                severity="HIGH",
                column="Timestamp",
                issue="Missing or invalid timestamps",
                affected_rows=invalid_timestamp_count,
                details=(
                    "These observations cannot be reliably assigned "
                    "to a time interval without additional information."
                ),
            )

        valid_timestamps = timestamps.dropna()

        duplicate_timestamp_count = safe_int(
            valid_timestamps.duplicated(keep=False).sum()
        )

        if duplicate_timestamp_count:
            add_issue(
                category="TIMESTAMP_VALIDATION",
                severity="MEDIUM",
                column="Timestamp",
                issue="Repeated timestamp observations",
                affected_rows=duplicate_timestamp_count,
                details=(
                    "Multiple records share the same timestamp. "
                    "Determine whether these represent legitimate "
                    "separate observations or duplicated records."
                ),
            )

    else:
        add_issue(
            category="TIMESTAMP_VALIDATION",
            severity="HIGH",
            column="Timestamp",
            issue="Timestamp analysis column unavailable",
            affected_rows=0,
            details=(
                "Timestamp quality checks could not be completed "
                "because the prepared timestamp column is missing."
            ),
        )

    # --------------------------------------------------------
    # 5. Compare sales and redemption values
    # --------------------------------------------------------
    if all(
        column in dataframe.columns
        for column in DEMAND_COLUMNS
    ):
        sales = pd.to_numeric(
            dataframe[DEMAND_COLUMNS[0]],
            errors="coerce",
        )

        redemptions = pd.to_numeric(
            dataframe[DEMAND_COLUMNS[1]],
            errors="coerce",
        )

        valid_pair_mask = (
            sales.notna()
            & redemptions.notna()
            & np.isfinite(sales)
            & np.isfinite(redemptions)
        )

        valid_pairs = safe_int(
            valid_pair_mask.sum()
        )

        if valid_pairs > 0:
            both_zero_mask = (
                valid_pair_mask
                & (sales == 0)
                & (redemptions == 0)
            )

            both_zero_count = safe_int(
                both_zero_mask.sum()
            )

            if both_zero_count:
                add_issue(
                    category="DEMAND_CONSISTENCY",
                    severity="LOW",
                    column="Sales Count, Redemption Count",
                    issue="Both demand measures equal zero",
                    affected_rows=both_zero_count,
                    details=(
                        "Zero sales and redemptions may be legitimate "
                        "during inactive periods. Confirm operational "
                        "context before treating them as anomalies."
                    ),
                )

            sales_positive_redemption_zero = (
                valid_pair_mask
                & (sales > 0)
                & (redemptions == 0)
            )

            inconsistent_count = safe_int(
                sales_positive_redemption_zero.sum()
            )

            if inconsistent_count:
                add_issue(
                    category="DEMAND_CONSISTENCY",
                    severity="LOW",
                    column="Sales Count, Redemption Count",
                    issue="Sales positive while redemptions are zero",
                    affected_rows=inconsistent_count,
                    details=(
                        "This relationship may be normal because "
                        "ticket sales and ticket redemption can occur "
                        "at different times. It is not automatically "
                        "a data-quality error."
                    ),
                )

    # --------------------------------------------------------
    # 6. Build the detailed issue report
    # --------------------------------------------------------
    quality_dataframe = pd.DataFrame(
        quality_records,
        columns=[
            "Category",
            "Severity",
            "Column",
            "Issue",
            "Affected_Rows",
            "Affected_Percentage",
            "Details",
        ],
    )

    if not quality_dataframe.empty:
        severity_order = {
            "HIGH": 0,
            "MEDIUM": 1,
            "LOW": 2,
        }

        quality_dataframe["_Severity_Order"] = (
            quality_dataframe["Severity"]
            .map(severity_order)
            .fillna(3)
        )

        quality_dataframe = (
            quality_dataframe
            .sort_values(
                by=[
                    "_Severity_Order",
                    "Affected_Rows",
                    "Category",
                ],
                ascending=[True, False, True],
                kind="stable",
            )
            .drop(columns=["_Severity_Order"])
            .reset_index(drop=True)
        )

    save_dataframe_csv(
        quality_dataframe,
        OUTPUT_DIR / "advanced_data_quality_checks.csv",
    )

    severity_counts = (
        quality_dataframe["Severity"]
        .value_counts()
        .to_dict()
        if not quality_dataframe.empty
        else {}
    )

    category_counts = (
        quality_dataframe["Category"]
        .value_counts()
        .to_dict()
        if not quality_dataframe.empty
        else {}
    )

    quality_summary: dict[str, Any] = {
        "generated_at_utc": get_utc_timestamp(),
        "dataset_rows_examined": safe_int(total_rows),
        "dataset_columns_examined": safe_int(
            len(dataframe.columns)
        ),
        "total_quality_findings": safe_int(
            len(quality_dataframe)
        ),
        "severity_counts": {
            str(key): safe_int(value)
            for key, value in severity_counts.items()
        },
        "category_counts": {
            str(key): safe_int(value)
            for key, value in category_counts.items()
        },
        "automatically_modified_rows": 0,
        "automatic_missing_value_imputation": False,
        "automatic_timestamp_gap_filling": False,
        "interpretation": (
            "Findings identify observations requiring review. "
            "They do not independently establish that a record is "
            "incorrect, and the source dataset is not modified."
        ),
    }

    save_json_report(
        quality_summary,
        REPORT_DIR / "advanced_data_quality_summary.json",
    )

    LOGGER.info(
        "Advanced quality checks completed: %s findings.",
        len(quality_dataframe),
    )

    return quality_dataframe, quality_summary


# END OF PART 11
# ============================================================
# ============================================================
# PART 12: TIME-SERIES READINESS AND FORECASTING VALIDATION
# ============================================================

def assess_time_series_readiness(
    dataframe: pd.DataFrame,
    timestamp_dataframe: pd.DataFrame,
    expected_interval_minutes: int = EXPECTED_INTERVAL_MINUTES,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Assess whether the observed ticket demand data is suitable
    for time-series forecasting preparation.

    This function evaluates:
        - Timestamp validity and ordering.
        - Duplicate timestamps.
        - Expected sampling interval consistency.
        - Missing time intervals.
        - Demand target availability and numeric validity.
        - Chronological train/validation/test feasibility.
        - Potential target leakage risks.

    Missing intervals are reported, not automatically filled.
    No forecasting model is trained by this function.
    """

    LOGGER.info(
        "Starting time-series forecasting readiness assessment."
    )

    if expected_interval_minutes <= 0:
        raise ValueError(
            "expected_interval_minutes must be greater than zero."
        )

    timestamp_column = "_Analysis_Timestamp_UTC"

    issue_records: list[dict[str, Any]] = []
    readiness_checks: list[dict[str, Any]] = []

    total_rows = len(dataframe)

    def add_check(
        check_name: str,
        status: str,
        observed_value: Any,
        expected_value: Any,
        details: str,
    ) -> None:
        readiness_checks.append(
            {
                "Check_Name": check_name,
                "Status": status,
                "Observed_Value": safe_json_value(observed_value),
                "Expected_Value": safe_json_value(expected_value),
                "Details": details,
            }
        )

    def add_readiness_issue(
        category: str,
        severity: str,
        issue: str,
        affected_rows: int,
        details: str,
    ) -> None:
        issue_records.append(
            {
                "Category": category,
                "Severity": severity,
                "Issue": issue,
                "Affected_Rows": safe_int(affected_rows),
                "Affected_Percentage": calculate_percentage(
                    affected_rows,
                    total_rows,
                ),
                "Details": details,
            }
        )

    # --------------------------------------------------------
    # 1. Verify that the timestamp column is available
    # --------------------------------------------------------
    if timestamp_column not in timestamp_dataframe.columns:
        add_check(
            check_name="Timestamp availability",
            status="FAIL",
            observed_value="Unavailable",
            expected_value=timestamp_column,
            details=(
                "Prepared UTC timestamps are required before "
                "time-series forecasting can be evaluated."
            ),
        )

        add_readiness_issue(
            category="TIMESTAMP",
            severity="HIGH",
            issue="Prepared timestamp column unavailable",
            affected_rows=total_rows,
            details=(
                "Timestamp preparation must succeed before the "
                "dataset can be evaluated for forecasting."
            ),
        )

        issue_dataframe = pd.DataFrame(issue_records)

        summary = {
            "generated_at_utc": get_utc_timestamp(),
            "forecasting_readiness": "NOT_READY",
            "expected_interval_minutes": expected_interval_minutes,
            "dataset_rows": safe_int(total_rows),
            "timestamp_column_available": False,
            "target_assessments": {},
            "readiness_checks": readiness_checks,
            "limitations": [
                "Timestamp preparation was unavailable.",
                "No forecasting model was trained.",
            ],
        }

        save_dataframe_csv(
            issue_dataframe,
            OUTPUT_DIR / "forecasting_readiness_issues.csv",
        )

        save_dataframe_csv(
            pd.DataFrame(readiness_checks),
            OUTPUT_DIR / "forecasting_readiness_checks.csv",
        )

        save_json_report(
            summary,
            REPORT_DIR / "forecasting_readiness_summary.json",
        )

        return issue_dataframe, summary

    # --------------------------------------------------------
    # 2. Prepare a separate chronological working dataset
    # --------------------------------------------------------
    working = timestamp_dataframe.copy()

    working[timestamp_column] = pd.to_datetime(
        working[timestamp_column],
        errors="coerce",
        utc=True,
    )

    invalid_timestamp_mask = working[timestamp_column].isna()

    invalid_timestamp_count = safe_int(
        invalid_timestamp_mask.sum()
    )

    valid_working = (
        working.loc[~invalid_timestamp_mask]
        .sort_values(
            timestamp_column,
            kind="stable",
        )
        .copy()
    )

    add_check(
        check_name="Valid timestamps",
        status=(
            "PASS"
            if invalid_timestamp_count == 0
            else "WARNING"
        ),
        observed_value=(
            safe_int(len(valid_working))
        ),
        expected_value=(
            f"{total_rows} valid timestamp rows"
        ),
        details=(
            "Rows with invalid timestamps cannot be assigned "
            "reliably to chronological forecast windows."
        ),
    )

    if invalid_timestamp_count:
        add_readiness_issue(
            category="TIMESTAMP",
            severity="HIGH",
            issue="Invalid timestamps found",
            affected_rows=invalid_timestamp_count,
            details=(
                "Investigate invalid timestamp values before "
                "constructing chronological datasets."
            ),
        )

    if valid_working.empty:
        summary = {
            "generated_at_utc": get_utc_timestamp(),
            "forecasting_readiness": "NOT_READY",
            "expected_interval_minutes": expected_interval_minutes,
            "dataset_rows": safe_int(total_rows),
            "valid_timestamp_rows": 0,
            "invalid_timestamp_rows": invalid_timestamp_count,
            "target_assessments": {},
            "readiness_checks": readiness_checks,
            "limitations": [
                "No valid timestamps were available.",
                "No forecasting model was trained.",
            ],
        }

        issue_dataframe = pd.DataFrame(issue_records)

        save_dataframe_csv(
            issue_dataframe,
            OUTPUT_DIR / "forecasting_readiness_issues.csv",
        )

        save_dataframe_csv(
            pd.DataFrame(readiness_checks),
            OUTPUT_DIR / "forecasting_readiness_checks.csv",
        )

        save_json_report(
            summary,
            REPORT_DIR / "forecasting_readiness_summary.json",
        )

        return issue_dataframe, summary

    # --------------------------------------------------------
    # 3. Check timestamp uniqueness and ordering
    # --------------------------------------------------------
    duplicate_timestamp_mask = (
        valid_working[timestamp_column]
        .duplicated(keep=False)
    )

    duplicate_timestamp_rows = safe_int(
        duplicate_timestamp_mask.sum()
    )

    unique_timestamps = (
        valid_working[timestamp_column]
        .drop_duplicates()
        .sort_values()
    )

    add_check(
        check_name="Timestamp uniqueness",
        status=(
            "PASS"
            if duplicate_timestamp_rows == 0
            else "WARNING"
        ),
        observed_value=(
            safe_int(len(unique_timestamps))
        ),
        expected_value="One unique timestamp per time point",
        details=(
            "Repeated timestamps require an explicit aggregation "
            "or business-rule decision before regular-grid forecasting."
        ),
    )

    if duplicate_timestamp_rows:
        add_readiness_issue(
            category="TIMESTAMP",
            severity="MEDIUM",
            issue="Duplicate timestamps detected",
            affected_rows=duplicate_timestamp_rows,
            details=(
                "Multiple observations share timestamps. Do not "
                "silently discard records; define an appropriate "
                "aggregation rule for the forecast target."
            ),
        )

    # --------------------------------------------------------
    # 4. Assess expected interval consistency
    # --------------------------------------------------------
    observed_intervals = (
        unique_timestamps.diff()
        .dt.total_seconds()
        .div(60)
        .dropna()
    )

    positive_intervals = observed_intervals[
        np.isfinite(observed_intervals)
        & (observed_intervals > 0)
    ]

    expected_interval_count = safe_int(
        (
            positive_intervals
            == expected_interval_minutes
        ).sum()
    )

    longer_interval_count = safe_int(
        (
            positive_intervals
            > expected_interval_minutes
        ).sum()
    )

    shorter_interval_count = safe_int(
        (
            positive_intervals
            < expected_interval_minutes
        ).sum()
    )

    interval_count = safe_int(
        len(positive_intervals)
    )

    exact_interval_percentage = calculate_percentage(
        expected_interval_count,
        interval_count,
    )

    if interval_count == 0:
        cadence_status = "INSUFFICIENT_DATA"
    elif exact_interval_percentage >= 95:
        cadence_status = "PASS"
    elif exact_interval_percentage >= 75:
        cadence_status = "WARNING"
    else:
        cadence_status = "FAIL"

    add_check(
        check_name="Expected sampling interval",
        status=cadence_status,
        observed_value={
            "exact_interval_percentage": exact_interval_percentage,
            "exact_intervals": expected_interval_count,
            "longer_intervals": longer_interval_count,
            "shorter_intervals": shorter_interval_count,
        },
        expected_value={
            "interval_minutes": expected_interval_minutes,
            "minimum_exact_interval_percentage": 95,
        },
        details=(
            "Interval consistency determines whether regular-grid "
            "forecasting preparation may require explicit resampling "
            "and missing-interval handling."
        ),
    )

    if longer_interval_count:
        add_readiness_issue(
            category="SAMPLING",
            severity="MEDIUM",
            issue="Longer-than-expected timestamp intervals",
            affected_rows=longer_interval_count,
            details=(
                "Longer intervals may represent missing observations, "
                "operational pauses, or changes in data collection. "
                "Their causes should be investigated before resampling."
            ),
        )

    if shorter_interval_count:
        add_readiness_issue(
            category="SAMPLING",
            severity="LOW",
            issue="Shorter-than-expected timestamp intervals",
            affected_rows=shorter_interval_count,
            details=(
                "Shorter intervals may reflect duplicate-like records, "
                "irregular collection, or legitimate additional events."
            ),
        )

    # --------------------------------------------------------
    # 5. Estimate unobserved time slots without filling them
    # --------------------------------------------------------
    if not unique_timestamps.empty:
        first_timestamp = unique_timestamps.iloc[0]
        last_timestamp = unique_timestamps.iloc[-1]

        expected_grid_slots = (
            int(
                (
                    last_timestamp - first_timestamp
                ).total_seconds()
                // (expected_interval_minutes * 60)
            )
            + 1
        )

        unique_observed_slots = safe_int(
            len(unique_timestamps)
        )

        estimated_unobserved_slots = max(
            0,
            expected_grid_slots - unique_observed_slots,
        )

        estimated_grid_coverage_percentage = (
            calculate_percentage(
                unique_observed_slots,
                expected_grid_slots,
            )
        )

    else:
        expected_grid_slots = 0
        unique_observed_slots = 0
        estimated_unobserved_slots = 0
        estimated_grid_coverage_percentage = 0.0

    add_check(
        check_name="Estimated regular-grid coverage",
        status=(
            "PASS"
            if estimated_grid_coverage_percentage >= 95
            else "WARNING"
        ),
        observed_value={
            "expected_grid_slots": expected_grid_slots,
            "unique_observed_timestamps": unique_observed_slots,
            "estimated_unobserved_slots": estimated_unobserved_slots,
            "coverage_percentage": estimated_grid_coverage_percentage,
        },
        expected_value={
            "coverage_percentage": "At least 95% preferred",
        },
        details=(
            "This is an estimate based on the first and last observed "
            "timestamps. Irregular timestamps and repeated records "
            "must be considered when interpreting coverage."
        ),
    )

    # --------------------------------------------------------
    # 6. Validate each configured demand target
    # --------------------------------------------------------
    target_assessments: dict[str, Any] = {}

    for target_column in DEMAND_COLUMNS:
        if target_column not in dataframe.columns:
            target_assessments[target_column] = {
                "available": False,
                "status": "NOT_READY",
                "valid_numeric_rows": 0,
                "missing_rows": total_rows,
                "negative_rows": 0,
                "infinite_rows": 0,
                "distinct_values": 0,
            }

            add_readiness_issue(
                category="TARGET",
                severity="HIGH",
                issue=f"Forecast target missing: {target_column}",
                affected_rows=total_rows,
                details=(
                    "The configured target column does not exist "
                    "in the source dataset."
                ),
            )

            continue

        values = pd.to_numeric(
            dataframe[target_column],
            errors="coerce",
        )

        finite_mask = pd.Series(
            np.isfinite(
                values.to_numpy(
                    dtype="float64",
                    na_value=np.nan,
                )
            ),
            index=dataframe.index,
        )

        valid_mask = values.notna() & finite_mask

        negative_mask = valid_mask & (values < 0)

        infinite_mask = pd.Series(
            np.isinf(
                values.to_numpy(
                    dtype="float64",
                    na_value=np.nan,
                )
            ),
            index=dataframe.index,
        )

        missing_count = safe_int(
            values.isna().sum()
        )

        valid_count = safe_int(
            valid_mask.sum()
        )

        negative_count = safe_int(
            negative_mask.sum()
        )

        infinite_count = safe_int(
            infinite_mask.sum()
        )

        distinct_values = safe_int(
            values.loc[valid_mask].nunique()
        )

        if valid_count == 0 or distinct_values < 2:
            target_status = "NOT_READY"
        elif missing_count > 0 or negative_count > 0:
            target_status = "REVIEW_REQUIRED"
        else:
            target_status = "CANDIDATE"

        target_assessments[target_column] = {
            "available": True,
            "status": target_status,
            "total_rows": safe_int(total_rows),
            "valid_numeric_rows": valid_count,
            "missing_or_unparseable_rows": missing_count,
            "negative_rows": negative_count,
            "infinite_rows": infinite_count,
            "distinct_values": distinct_values,
            "mean": safe_float(
                values.loc[valid_mask].mean()
            ),
            "median": safe_float(
                values.loc[valid_mask].median()
            ),
            "standard_deviation": safe_float(
                values.loc[valid_mask].std()
            ),
        }

        add_check(
            check_name=f"Forecast target: {target_column}",
            status=(
                "PASS"
                if target_status == "CANDIDATE"
                else "WARNING"
            ),
            observed_value={
                "valid_rows": valid_count,
                "missing_rows": missing_count,
                "negative_rows": negative_count,
                "distinct_values": distinct_values,
            },
            expected_value=(
                "Sufficient valid numeric observations "
                "with at least two distinct values"
            ),
            details=(
                "Candidate status indicates basic numeric suitability "
                "only. It does not establish forecast accuracy or "
                "guarantee model suitability."
            ),
        )

        if missing_count:
            add_readiness_issue(
                category="TARGET",
                severity="MEDIUM",
                issue=f"Missing or invalid target values: {target_column}",
                affected_rows=missing_count,
                details=(
                    "Target gaps require an explicit policy. Avoid "
                    "automatically treating unknown demand as zero."
                ),
            )

        if negative_count:
            add_readiness_issue(
                category="TARGET",
                severity="HIGH",
                issue=f"Negative target values: {target_column}",
                affected_rows=negative_count,
                details=(
                    "Investigate negative counts before using this "
                    "target for a non-negative demand forecasting task."
                ),
            )

        if infinite_count:
            add_readiness_issue(
                category="TARGET",
                severity="HIGH",
                issue=f"Infinite target values: {target_column}",
                affected_rows=infinite_count,
                details=(
                    "Infinite values are not valid demand observations "
                    "for ordinary forecasting evaluation."
                ),
            )

    # --------------------------------------------------------
    # 7. Assess chronological evaluation feasibility
    # --------------------------------------------------------
    unique_time_count = safe_int(
        len(unique_timestamps)
    )

    if unique_time_count >= 100:
        chronological_evaluation_status = "FEASIBLE_FOR_INITIAL_TESTING"
    elif unique_time_count >= 30:
        chronological_evaluation_status = "LIMITED"
    else:
        chronological_evaluation_status = "INSUFFICIENT_DATA"

    add_check(
        check_name="Chronological evaluation feasibility",
        status=(
            "PASS"
            if unique_time_count >= 100
            else "WARNING"
        ),
        observed_value=unique_time_count,
        expected_value=(
            "At least 100 unique timestamps for initial testing"
        ),
        details=(
            "This is a basic minimum-data heuristic, not a guarantee "
            "of sufficient data for every forecast horizon or model. "
            "Training, validation, and test periods must remain "
            "chronologically separated."
        ),
    )

    # --------------------------------------------------------
    # 8. Document potential leakage risks
    # --------------------------------------------------------
    potential_leakage_columns = [
        column
        for column in dataframe.columns
        if any(
            keyword in str(column).lower()
            for keyword in [
                "future",
                "next_period",
                "next_timestamp",
                "target_encoded",
                "actual_future",
            ]
        )
    ]

    add_check(
        check_name="Potential target leakage column screening",
        status=(
            "WARNING"
            if potential_leakage_columns
            else "REVIEW_REQUIRED"
        ),
        observed_value=potential_leakage_columns,
        expected_value=(
            "No unverified future-derived target features"
        ),
        details=(
            "Name-based screening is not sufficient to establish "
            "the absence of leakage. Every predictor must be checked "
            "to ensure its value would be available at forecast time."
        ),
    )

    if potential_leakage_columns:
        add_readiness_issue(
            category="LEAKAGE",
            severity="HIGH",
            issue="Potential future-derived columns detected",
            affected_rows=0,
            details=(
                "Review these columns before training. Column names "
                "alone cannot establish whether leakage is present."
            ),
        )

    # --------------------------------------------------------
    # 9. Overall readiness assessment
    # --------------------------------------------------------
    target_candidates = [
        target
        for target, assessment in target_assessments.items()
        if assessment.get("status") == "CANDIDATE"
    ]

    high_severity_issues = sum(
        1
        for issue in issue_records
        if issue["Severity"] == "HIGH"
    )

    if (
        not target_candidates
        or unique_time_count < 30
        or valid_working.empty
    ):
        overall_readiness = "NOT_READY"

    elif high_severity_issues > 0:
        overall_readiness = "REVIEW_REQUIRED"

    elif (
        cadence_status in {"FAIL", "INSUFFICIENT_DATA"}
        or duplicate_timestamp_rows > 0
    ):
        overall_readiness = "REVIEW_REQUIRED"

    else:
        overall_readiness = "CANDIDATE_FOR_MODEL_DEVELOPMENT"

    issue_dataframe = pd.DataFrame(
        issue_records,
        columns=[
            "Category",
            "Severity",
            "Issue",
            "Affected_Rows",
            "Affected_Percentage",
            "Details",
        ],
    )

    checks_dataframe = pd.DataFrame(
        readiness_checks,
        columns=[
            "Check_Name",
            "Status",
            "Observed_Value",
            "Expected_Value",
            "Details",
        ],
    )

    summary: dict[str, Any] = {
        "generated_at_utc": get_utc_timestamp(),
        "forecasting_readiness": overall_readiness,
        "dataset_rows": safe_int(total_rows),
        "valid_timestamp_rows": safe_int(len(valid_working)),
        "invalid_timestamp_rows": invalid_timestamp_count,
        "unique_timestamp_count": unique_time_count,
        "duplicate_timestamp_rows": duplicate_timestamp_rows,
        "expected_interval_minutes": expected_interval_minutes,
        "observed_interval_count": interval_count,
        "exact_interval_count": expected_interval_count,
        "longer_interval_count": longer_interval_count,
        "shorter_interval_count": shorter_interval_count,
        "exact_interval_percentage": exact_interval_percentage,
        "estimated_grid_slots": expected_grid_slots,
        "estimated_unobserved_slots": estimated_unobserved_slots,
        "estimated_grid_coverage_percentage": (
            estimated_grid_coverage_percentage
        ),
        "chronological_evaluation_status": (
            chronological_evaluation_status
        ),
        "target_assessments": target_assessments,
        "candidate_targets": target_candidates,
        "potential_leakage_columns": potential_leakage_columns,
        "total_readiness_issues": safe_int(
            len(issue_dataframe)
        ),
        "high_severity_issues": safe_int(
            high_severity_issues
        ),
        "readiness_checks": readiness_checks,
        "limitations": [
            (
                "This assessment does not train a forecasting model "
                "or establish predictive accuracy."
            ),
            (
                "Regular-grid coverage is estimated from unique "
                "timestamps and may require further investigation."
            ),
            (
                "Missing timestamps are not automatically filled."
            ),
            (
                "Ticket sales and redemptions are separate targets "
                "and may require different forecasting models."
            ),
            (
                "A chronological holdout is required to measure "
                "future forecasting performance without leakage."
            ),
        ],
    }

    save_dataframe_csv(
        issue_dataframe,
        OUTPUT_DIR / "forecasting_readiness_issues.csv",
    )

    save_dataframe_csv(
        checks_dataframe,
        OUTPUT_DIR / "forecasting_readiness_checks.csv",
    )

    save_json_report(
        summary,
        REPORT_DIR / "forecasting_readiness_summary.json",
    )

    LOGGER.info(
        "Time-series readiness assessment completed: %s",
        overall_readiness,
    )

    return issue_dataframe, summary


# END OF PART 12
# ============================================================
# ============================================================
# PART 13: FORECASTING DATASET PREPARATION AND TIME AGGREGATION
# ============================================================

def prepare_forecasting_dataset(
    dataframe: pd.DataFrame,
    timestamp_dataframe: pd.DataFrame,
    target_column: str,
    frequency: str = "15min",
    aggregation_method: str = "sum",
    duplicate_timestamp_policy: str = "aggregate",
    minimum_coverage: float = 0.80,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Prepare a chronological demand series for later forecasting.

    Processing includes:
        1. Validate the target, frequency, and aggregation policy.
        2. Associate observations with their parsed timestamps.
        3. Exclude invalid timestamps from the working series.
        4. Convert target values to numeric.
        5. Identify duplicate timestamps.
        6. Aggregate observations when explicitly configured.
        7. Create a regular time grid without filling missing demand.
        8. Calculate interval-level observation coverage.
        9. Preserve missing values for explicit downstream handling.
       10. Save the prepared series and its preparation report.

    Important:
        - The source DataFrame is never modified.
        - Missing periods are represented as NaN, not zero.
        - Aggregation assumes the target represents additive demand
          within each interval. Verify this assumption for the source.
        - This function prepares data; it does not train a model.
    """

    LOGGER.info(
        "Preparing forecasting series for target: %s",
        target_column,
    )

    allowed_aggregation_methods = {
        "sum",
        "mean",
        "median",
        "max",
        "min",
    }

    if aggregation_method not in allowed_aggregation_methods:
        raise ValueError(
            "Unsupported aggregation_method. Choose one of: "
            f"{sorted(allowed_aggregation_methods)}"
        )

    if duplicate_timestamp_policy not in {
        "aggregate",
        "raise",
    }:
        raise ValueError(
            "duplicate_timestamp_policy must be 'aggregate' or 'raise'."
        )

    if not 0 < minimum_coverage <= 1:
        raise ValueError(
            "minimum_coverage must be greater than 0 and at most 1."
        )

    if target_column not in dataframe.columns:
        raise KeyError(
            f"Forecast target '{target_column}' was not found "
            "in the source dataset."
        )

    timestamp_column = "_Analysis_Timestamp_UTC"

    if timestamp_column not in timestamp_dataframe.columns:
        raise KeyError(
            f"Prepared timestamp column '{timestamp_column}' "
            "is unavailable."
        )

    # --------------------------------------------------------
    # 1. Validate the frequency using pandas' offset engine
    # --------------------------------------------------------
    try:
        frequency_offset = pd.tseries.frequencies.to_offset(
            frequency
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid resampling frequency: {frequency}"
        ) from exc

    try:
        frequency_delta = pd.Timedelta(
            frequency_offset.nanos,
            unit="ns",
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "The requested frequency must represent a fixed-duration "
            "time interval, such as '15min', '30min', or '1h'."
        ) from exc

    if frequency_delta <= pd.Timedelta(0):
        raise ValueError(
            "The resampling frequency must be positive."
        )

    # --------------------------------------------------------
    # 2. Associate source values with prepared timestamps
    # --------------------------------------------------------
    if "_Original_Row_Position" in timestamp_dataframe.columns:
        row_positions = pd.to_numeric(
            timestamp_dataframe["_Original_Row_Position"],
            errors="coerce",
        )

        valid_positions = (
            row_positions.notna()
            & (row_positions >= 0)
            & (row_positions < len(dataframe))
        )

        working = timestamp_dataframe.loc[
            valid_positions,
            [
                timestamp_column,
                "_Original_Row_Position",
            ],
        ].copy()

        working["_Original_Row_Position"] = (
            working["_Original_Row_Position"]
            .astype("int64")
        )

        working[target_column] = (
            dataframe[target_column]
            .iloc[
                working["_Original_Row_Position"].to_numpy()
            ]
            .to_numpy()
        )

    else:
        if len(timestamp_dataframe) != len(dataframe):
            raise ValueError(
                "Timestamp data does not contain original row positions "
                "and its length differs from the source dataset."
            )

        working = pd.DataFrame(
            {
                timestamp_column: timestamp_dataframe[
                    timestamp_column
                ].to_numpy(),
                target_column: dataframe[
                    target_column
                ].to_numpy(),
                "_Original_Row_Position": np.arange(
                    len(dataframe)
                ),
            }
        )

    working[timestamp_column] = pd.to_datetime(
        working[timestamp_column],
        errors="coerce",
        utc=True,
    )

    working[target_column] = pd.to_numeric(
        working[target_column],
        errors="coerce",
    )

    working[target_column] = (
        working[target_column].replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    invalid_timestamp_count = safe_int(
        working[timestamp_column].isna().sum()
    )

    missing_target_count = safe_int(
        working[target_column].isna().sum()
    )

    working = working.dropna(
        subset=[timestamp_column]
    ).copy()

    if working.empty:
        raise ValueError(
            "No valid timestamps remain after timestamp validation."
        )

    # --------------------------------------------------------
    # 3. Examine repeated timestamps before aggregation
    # --------------------------------------------------------
    duplicate_mask = working[
        timestamp_column
    ].duplicated(keep=False)

    duplicate_timestamp_row_count = safe_int(
        duplicate_mask.sum()
    )

    duplicated_timestamp_count = safe_int(
        working.loc[
            duplicate_mask,
            timestamp_column,
        ].nunique()
    )

    if (
        duplicate_timestamp_row_count > 0
        and duplicate_timestamp_policy == "raise"
    ):
        raise ValueError(
            f"Found {duplicate_timestamp_row_count} observations "
            f"across {duplicated_timestamp_count} repeated timestamps. "
            "Choose duplicate_timestamp_policy='aggregate' only if "
            "the selected aggregation is appropriate for the target."
        )

    # --------------------------------------------------------
    # 4. Aggregate observations onto the requested frequency
    # --------------------------------------------------------
    working = working.sort_values(
        timestamp_column,
        kind="stable",
    )

    target_aggregation = getattr(
        working.groupby(timestamp_column)[target_column],
        aggregation_method,
    )()

    timestamp_level = pd.DataFrame(
        {
            "Observed_Demand": target_aggregation,
        }
    )

    timestamp_level.index.name = "Timestamp_UTC"

    timestamp_level["Source_Row_Count"] = (
        working.groupby(timestamp_column)
        .size()
        .reindex(timestamp_level.index)
        .astype("int64")
    )

    # Resample demand values using the configured aggregation.
    # The count records how many valid target observations actually
    # contributed to each interval.
    resampler = timestamp_level["Observed_Demand"].resample(
        frequency
    )

    aggregated_demand = getattr(
        resampler,
        aggregation_method,
    )()

    observed_target_counts = (
        timestamp_level["Observed_Demand"]
        .resample(frequency)
        .count()
    )

    source_row_counts = (
        timestamp_level["Source_Row_Count"]
        .resample(frequency)
        .sum()
    )

    # --------------------------------------------------------
    # 5. Estimate expected observations per interval
    # --------------------------------------------------------
    observed_timestamps = (
        working[timestamp_column]
        .drop_duplicates()
        .sort_values()
    )

    if len(observed_timestamps) >= 2:
        source_intervals = (
            observed_timestamps.diff()
            .dt.total_seconds()
            .div(60)
            .dropna()
        )

        source_intervals = source_intervals[
            np.isfinite(source_intervals)
            & (source_intervals > 0)
        ]

        if not source_intervals.empty:
            source_median_interval = float(
                source_intervals.median()
            )
        else:
            source_median_interval = float(
                frequency_delta.total_seconds() / 60
            )

    else:
        source_median_interval = float(
            frequency_delta.total_seconds() / 60
        )

    requested_interval_minutes = (
        frequency_delta.total_seconds() / 60
    )

    if source_median_interval > 0:
        expected_source_observations = max(
            1,
            int(
                round(
                    requested_interval_minutes
                    / source_median_interval
                )
            ),
        )
    else:
        expected_source_observations = 1

    # --------------------------------------------------------
    # 6. Construct the regular grid without inventing demand
    # --------------------------------------------------------
    regular_series = pd.DataFrame(
        {
            target_column: aggregated_demand,
            "Observed_Target_Count": observed_target_counts,
            "Source_Row_Count": source_row_counts,
        }
    )

    regular_series.index.name = "Timestamp_UTC"

    regular_series["Observed_Target_Count"] = (
        regular_series["Observed_Target_Count"]
        .fillna(0)
        .astype("int64")
    )

    regular_series["Source_Row_Count"] = (
        regular_series["Source_Row_Count"]
        .fillna(0)
        .astype("int64")
    )

    regular_series["Expected_Source_Observations"] = (
        expected_source_observations
    )

    regular_series["Estimated_Coverage"] = (
        regular_series["Observed_Target_Count"]
        .div(expected_source_observations)
        .clip(lower=0, upper=1)
    )

    regular_series["Has_Observed_Target"] = (
        regular_series["Observed_Target_Count"] > 0
    )

    regular_series["Coverage_Acceptable"] = (
        regular_series["Estimated_Coverage"]
        >= minimum_coverage
    )

    # A target interval with no valid observations remains NaN.
    # A partially observed interval also remains visible and receives
    # a coverage indicator instead of being silently replaced.
    regular_series["Demand_Quality_Status"] = np.select(
        [
            regular_series["Observed_Target_Count"] == 0,
            regular_series["Estimated_Coverage"] < minimum_coverage,
        ],
        [
            "NO_VALID_OBSERVATIONS",
            "LOW_OBSERVATION_COVERAGE",
        ],
        default="OBSERVATION_AVAILABLE",
    )

    # --------------------------------------------------------
    # 7. Prepare a compact quality summary
    # --------------------------------------------------------
    total_intervals = safe_int(
        len(regular_series)
    )

    intervals_with_observations = safe_int(
        regular_series["Has_Observed_Target"].sum()
    )

    intervals_without_observations = (
        total_intervals - intervals_with_observations
    )

    low_coverage_count = safe_int(
        (
            regular_series["Has_Observed_Target"]
            & ~regular_series["Coverage_Acceptable"]
        ).sum()
    )

    acceptable_coverage_count = safe_int(
        (
            regular_series["Has_Observed_Target"]
            & regular_series["Coverage_Acceptable"]
        ).sum()
    )

    preparation_summary: dict[str, Any] = {
        "generated_at_utc": get_utc_timestamp(),
        "target_column": target_column,
        "requested_frequency": frequency,
        "requested_interval_minutes": requested_interval_minutes,
        "aggregation_method": aggregation_method,
        "duplicate_timestamp_policy": duplicate_timestamp_policy,
        "minimum_coverage_threshold": minimum_coverage,
        "source_rows": safe_int(len(dataframe)),
        "rows_with_invalid_timestamps": invalid_timestamp_count,
        "rows_with_missing_or_invalid_target": missing_target_count,
        "valid_timestamp_rows": safe_int(len(working)),
        "unique_source_timestamps": safe_int(
            len(observed_timestamps)
        ),
        "rows_at_repeated_timestamps": duplicate_timestamp_row_count,
        "repeated_timestamp_count": duplicated_timestamp_count,
        "estimated_source_median_interval_minutes": round(
            source_median_interval,
            4,
        ),
        "expected_source_observations_per_interval": (
            expected_source_observations
        ),
        "regular_grid_interval_count": total_intervals,
        "intervals_with_observations": intervals_with_observations,
        "intervals_without_observations": (
            intervals_without_observations
        ),
        "intervals_below_coverage_threshold": low_coverage_count,
        "intervals_meeting_coverage_threshold": (
            acceptable_coverage_count
        ),
        "observed_interval_percentage": calculate_percentage(
            intervals_with_observations,
            total_intervals,
        ),
        "automatic_zero_filling": False,
        "automatic_missing_target_imputation": False,
        "source_dataset_modified": False,
        "interpretation": (
            "This output is a time-grid preparation artifact, not a "
            "trained forecasting dataset. Missing demand values remain "
            "missing, and estimated coverage is a diagnostic measure."
        ),
    }

    # --------------------------------------------------------
    # 8. Save prepared data and preparation metadata
    # --------------------------------------------------------
    output_name = (
        f"forecasting_series_{target_column.lower().replace(' ', '_')}"
    )

    prepared_csv_path = (
        OUTPUT_DIR / f"{output_name}.csv"
    )

    preparation_json_path = (
        REPORT_DIR / f"{output_name}_preparation.json"
    )

    save_dataframe_csv(
        regular_series.reset_index(),
        prepared_csv_path,
    )

    save_json_report(
        preparation_summary,
        preparation_json_path,
    )

    LOGGER.info(
        "Forecasting preparation completed for %s.",
        target_column,
    )

    LOGGER.info(
        "Prepared intervals: %s; observed: %s; missing: %s.",
        format_number(total_intervals),
        format_number(intervals_with_observations),
        format_number(intervals_without_observations),
    )

    return regular_series, preparation_summary


# END OF PART 13
# ============================================================

# ============================================================
# PART 14: MULTI-TARGET FORECASTING READINESS AND COMPARISON
# ============================================================

def prepare_all_forecasting_targets(
    dataframe: pd.DataFrame,
    timestamp_dataframe: pd.DataFrame,
    target_columns: list[str] | None = None,
    frequency: str = "15min",
    aggregation_method: str = "sum",
    minimum_coverage: float = 0.80,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    """
    Prepare multiple demand targets using a consistent configuration.

    Default targets:
        - Sales Count
        - Redemption Count

    Outputs:
        - One prepared time series per target.
        - A comparison table describing preparation quality.
        - A JSON report summarizing the multi-target operation.

    This function does not train models or claim that the targets
    are equally suitable for forecasting.
    """

    LOGGER.info(
        "Starting multi-target forecasting preparation."
    )

    if target_columns is None:
        target_columns = [
            "Sales Count",
            "Redemption Count",
        ]

    if not target_columns:
        raise ValueError(
            "At least one forecasting target must be supplied."
        )

    if len(target_columns) != len(set(target_columns)):
        raise ValueError(
            "Duplicate target names were supplied."
        )

    missing_targets = [
        column
        for column in target_columns
        if column not in dataframe.columns
    ]

    if missing_targets:
        raise KeyError(
            "The following requested target columns are missing: "
            f"{missing_targets}"
        )

    prepared_targets: dict[str, pd.DataFrame] = {}
    target_summaries: list[dict[str, Any]] = []
    target_errors: list[dict[str, Any]] = []

    # --------------------------------------------------------
    # 1. Prepare each target independently
    # --------------------------------------------------------
    for target_column in target_columns:
        LOGGER.info(
            "Preparing target %s (%s of %s).",
            target_column,
            len(prepared_targets) + 1,
            len(target_columns),
        )

        try:
            prepared_series, preparation_summary = (
                prepare_forecasting_dataset(
                    dataframe=dataframe,
                    timestamp_dataframe=timestamp_dataframe,
                    target_column=target_column,
                    frequency=frequency,
                    aggregation_method=aggregation_method,
                    duplicate_timestamp_policy="aggregate",
                    minimum_coverage=minimum_coverage,
                )
            )

            prepared_targets[target_column] = (
                prepared_series.copy()
            )

            valid_target_values = (
                prepared_series[target_column]
                .dropna()
            )

            finite_target_values = valid_target_values[
                np.isfinite(valid_target_values)
            ]

            negative_value_count = safe_int(
                (finite_target_values < 0).sum()
            )

            zero_value_count = safe_int(
                (finite_target_values == 0).sum()
            )

            positive_value_count = safe_int(
                (finite_target_values > 0).sum()
            )

            target_summaries.append(
                {
                    "target_column": target_column,
                    "preparation_status": "SUCCESS",
                    "frequency": frequency,
                    "aggregation_method": aggregation_method,
                    "interval_count": safe_int(
                        len(prepared_series)
                    ),
                    "valid_target_interval_count": safe_int(
                        len(finite_target_values)
                    ),
                    "missing_target_interval_count": safe_int(
                        prepared_series[
                            target_column
                        ].isna().sum()
                    ),
                    "zero_demand_interval_count": zero_value_count,
                    "positive_demand_interval_count": (
                        positive_value_count
                    ),
                    "negative_demand_interval_count": (
                        negative_value_count
                    ),
                    "target_mean": (
                        float(finite_target_values.mean())
                        if not finite_target_values.empty
                        else None
                    ),
                    "target_median": (
                        float(finite_target_values.median())
                        if not finite_target_values.empty
                        else None
                    ),
                    "target_standard_deviation": (
                        float(finite_target_values.std())
                        if len(finite_target_values) > 1
                        else None
                    ),
                    "target_minimum": (
                        float(finite_target_values.min())
                        if not finite_target_values.empty
                        else None
                    ),
                    "target_maximum": (
                        float(finite_target_values.max())
                        if not finite_target_values.empty
                        else None
                    ),
                    "acceptable_coverage_interval_count": (
                        preparation_summary.get(
                            "intervals_meeting_coverage_threshold",
                            0,
                        )
                    ),
                    "preparation_report": (
                        f"{target_column.lower().replace(' ', '_')}"
                        "_preparation.json"
                    ),
                }
            )

        except Exception as exc:
            LOGGER.exception(
                "Forecasting preparation failed for target %s.",
                target_column,
            )

            target_errors.append(
                {
                    "target_column": target_column,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

            target_summaries.append(
                {
                    "target_column": target_column,
                    "preparation_status": "FAILED",
                    "frequency": frequency,
                    "aggregation_method": aggregation_method,
                    "interval_count": 0,
                    "valid_target_interval_count": 0,
                    "missing_target_interval_count": None,
                    "zero_demand_interval_count": None,
                    "positive_demand_interval_count": None,
                    "negative_demand_interval_count": None,
                    "target_mean": None,
                    "target_median": None,
                    "target_standard_deviation": None,
                    "target_minimum": None,
                    "target_maximum": None,
                    "acceptable_coverage_interval_count": None,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    # --------------------------------------------------------
    # 2. Create the cross-target comparison table
    # --------------------------------------------------------
    comparison_dataframe = pd.DataFrame(
        target_summaries
    )

    comparison_path = (
        OUTPUT_DIR / "forecasting_target_comparison.csv"
    )

    save_dataframe_csv(
        comparison_dataframe,
        comparison_path,
    )

    # --------------------------------------------------------
    # 3. Evaluate basic forecasting-target suitability
    # --------------------------------------------------------
    suitability_records: list[dict[str, Any]] = []

    for target_column in target_columns:
        if target_column not in prepared_targets:
            suitability_records.append(
                {
                    "target_column": target_column,
                    "assessment_status": "NOT_ASSESSED",
                    "reason": "Target preparation failed.",
                    "sufficient_observations": False,
                    "contains_negative_values": None,
                    "contains_demand_variation": None,
                    "forecasting_readiness": "NOT_READY",
                }
            )
            continue

        series = prepared_targets[target_column][
            target_column
        ]

        finite_series = series.dropna()

        finite_series = finite_series[
            np.isfinite(finite_series)
        ]

        observation_count = safe_int(
            len(finite_series)
        )

        negative_values_exist = bool(
            (finite_series < 0).any()
        ) if observation_count else None

        distinct_values = safe_int(
            finite_series.nunique()
        )

        sufficient_observations = (
            observation_count >= 100
        )

        contains_variation = (
            distinct_values > 1
        )

        if not sufficient_observations:
            readiness_status = "INSUFFICIENT_OBSERVATIONS"
            reason = (
                "Fewer than 100 finite target intervals are available. "
                "This is a preliminary screening rule, not a universal "
                "minimum requirement for every forecasting model."
            )

        elif negative_values_exist:
            readiness_status = "REQUIRES_TARGET_REVIEW"
            reason = (
                "Negative values exist. Confirm their meaning and "
                "whether they represent valid adjustments or data errors."
            )

        elif not contains_variation:
            readiness_status = "REQUIRES_TARGET_REVIEW"
            reason = (
                "The target has no observed variation. Verify that "
                "the selected column and aggregation are appropriate."
            )

        else:
            readiness_status = "BASIC_CHECKS_PASSED"
            reason = (
                "Basic target checks passed. Chronological validation, "
                "baseline comparison, leakage prevention, and forecast "
                "evaluation are still required before model approval."
            )

        suitability_records.append(
            {
                "target_column": target_column,
                "assessment_status": "ASSESSED",
                "reason": reason,
                "finite_observation_count": observation_count,
                "distinct_target_value_count": distinct_values,
                "sufficient_observations": (
                    sufficient_observations
                ),
                "contains_negative_values": (
                    negative_values_exist
                ),
                "contains_demand_variation": (
                    contains_variation
                ),
                "forecasting_readiness": readiness_status,
            }
        )

    suitability_dataframe = pd.DataFrame(
        suitability_records
    )

    suitability_path = (
        OUTPUT_DIR / "forecasting_target_suitability.csv"
    )

    save_dataframe_csv(
        suitability_dataframe,
        suitability_path,
    )

    # --------------------------------------------------------
    # 4. Build a reproducible preparation summary
    # --------------------------------------------------------
    successful_target_count = safe_int(
        sum(
            1
            for item in target_summaries
            if item.get("preparation_status") == "SUCCESS"
        )
    )

    failed_target_count = safe_int(
        sum(
            1
            for item in target_summaries
            if item.get("preparation_status") == "FAILED"
        )
    )

    readiness_counts = (
        suitability_dataframe["forecasting_readiness"]
        .value_counts(dropna=False)
        .to_dict()
        if not suitability_dataframe.empty
        else {}
    )

    summary: dict[str, Any] = {
        "generated_at_utc": get_utc_timestamp(),
        "report_version": REPORT_VERSION,
        "requested_targets": target_columns,
        "requested_frequency": frequency,
        "aggregation_method": aggregation_method,
        "minimum_coverage_threshold": minimum_coverage,
        "requested_target_count": len(target_columns),
        "successfully_prepared_target_count": (
            successful_target_count
        ),
        "failed_target_count": failed_target_count,
        "target_preparation_errors": target_errors,
        "readiness_status_counts": readiness_counts,
        "comparison_output": str(comparison_path),
        "suitability_output": str(suitability_path),
        "prepared_target_outputs": {
            target: (
                "forecasting_series_"
                f"{target.lower().replace(' ', '_')}.csv"
            )
            for target in prepared_targets
        },
        "zero_filling_performed": False,
        "models_trained": False,
        "forecast_accuracy_evaluated": False,
        "interpretation": (
            "These checks assess data preparation and basic target "
            "suitability only. They do not establish model accuracy, "
            "future demand certainty, or operational readiness."
        ),
    }

    summary_path = (
        REPORT_DIR / "forecasting_target_preparation_summary.json"
    )

    save_json_report(
        summary,
        summary_path,
    )

    LOGGER.info(
        "Multi-target preparation finished. Success: %s; failed: %s.",
        successful_target_count,
        failed_target_count,
    )

    return (
        prepared_targets,
        comparison_dataframe,
        summary,
    )


# END OF PART 14
# ============================================================
# ============================================================
# PART 15: FORECASTING DATA QUALITY GATES AND MODEL APPROVAL
# ============================================================

def evaluate_forecasting_data_quality(
    prepared_series: pd.DataFrame,
    target_column: str,
    minimum_observations: int = 100,
    maximum_missing_percentage: float = 20.0,
    minimum_target_variation: int = 2,
    require_non_negative_target: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Evaluate whether a prepared time series satisfies basic
    data-quality conditions before forecasting-model development.

    Quality gates:
        1. Required target exists.
        2. Datetime index is valid.
        3. Timestamps are unique and chronological.
        4. Target values are numeric and finite.
        5. Missing-target percentage is measured.
        6. Negative demand values are identified.
        7. Target variation is assessed.
        8. Regular sampling is checked.
        9. Consecutive missing intervals are identified.
       10. A transparent readiness decision is generated.

    This is a data-quality assessment, not a model-accuracy guarantee.
    """

    LOGGER.info(
        "Evaluating forecasting quality gates for target '%s'.",
        target_column,
    )

    if minimum_observations < 1:
        raise ValueError(
            "minimum_observations must be at least 1."
        )

    if not 0 <= maximum_missing_percentage <= 100:
        raise ValueError(
            "maximum_missing_percentage must be between 0 and 100."
        )

    if minimum_target_variation < 1:
        raise ValueError(
            "minimum_target_variation must be at least 1."
        )

    if target_column not in prepared_series.columns:
        raise KeyError(
            f"Target column '{target_column}' does not exist "
            "in the prepared time series."
        )

    # --------------------------------------------------------
    # 1. Normalize and inspect the time index
    # --------------------------------------------------------
    working = prepared_series.copy()

    try:
        parsed_index = pd.to_datetime(
            working.index,
            errors="coerce",
            utc=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "The prepared time-series index cannot be parsed "
            "as timestamps."
        ) from exc

    index_is_valid = bool(
        not pd.isna(parsed_index).any()
    )

    index_is_unique = bool(
        not parsed_index.has_duplicates
    )

    index_is_monotonic = bool(
        parsed_index.is_monotonic_increasing
    )

    total_intervals = safe_int(
        len(working)
    )

    # --------------------------------------------------------
    # 2. Inspect target values without imputing missing values
    # --------------------------------------------------------
    numeric_target = pd.to_numeric(
        working[target_column],
        errors="coerce",
    )

    numeric_target = numeric_target.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    valid_target_mask = numeric_target.notna()

    valid_target_count = safe_int(
        valid_target_mask.sum()
    )

    missing_target_count = (
        total_intervals - valid_target_count
    )

    missing_percentage = calculate_percentage(
        missing_target_count,
        total_intervals,
    )

    finite_values = numeric_target.dropna()

    negative_value_count = safe_int(
        (finite_values < 0).sum()
    )

    zero_value_count = safe_int(
        (finite_values == 0).sum()
    )

    positive_value_count = safe_int(
        (finite_values > 0).sum()
    )

    distinct_value_count = safe_int(
        finite_values.nunique()
    )

    target_has_variation = (
        distinct_value_count >= minimum_target_variation
    )

    sufficient_observations = (
        valid_target_count >= minimum_observations
    )

    missingness_acceptable = (
        missing_percentage <= maximum_missing_percentage
    )

    negative_values_acceptable = (
        not require_non_negative_target
        or negative_value_count == 0
    )

    # --------------------------------------------------------
    # 3. Check timestamp interval consistency
    # --------------------------------------------------------
    observed_interval_minutes = None
    median_interval_minutes = None
    interval_consistency_percentage = None
    irregular_interval_count = 0

    if total_intervals >= 2 and index_is_valid:
        index_series = pd.Series(
            parsed_index
        ).sort_values()

        interval_minutes = (
            index_series.diff()
            .dt.total_seconds()
            .div(60)
            .dropna()
        )

        interval_minutes = interval_minutes[
            np.isfinite(interval_minutes)
            & (interval_minutes > 0)
        ]

        if not interval_minutes.empty:
            median_interval_minutes = float(
                interval_minutes.median()
            )

            observed_interval_minutes = float(
                interval_minutes.mode().iloc[0]
            )

            tolerance_minutes = max(
                0.001,
                observed_interval_minutes * 0.01,
            )

            irregular_interval_count = safe_int(
                (
                    interval_minutes.sub(
                        observed_interval_minutes
                    ).abs()
                    > tolerance_minutes
                ).sum()
            )

            interval_consistency_percentage = (
                100.0
                * (
                    len(interval_minutes)
                    - irregular_interval_count
                )
                / len(interval_minutes)
            )

    regular_sampling_detected = (
        interval_consistency_percentage is not None
        and interval_consistency_percentage >= 95.0
    )

    # --------------------------------------------------------
    # 4. Identify missing runs in the prepared target
    # --------------------------------------------------------
    missing_run_count = 0
    longest_missing_run = 0

    if total_intervals:
        missing_mask = numeric_target.isna().to_numpy()

        current_run = 0

        for is_missing in missing_mask:
            if is_missing:
                current_run += 1

                longest_missing_run = max(
                    longest_missing_run,
                    current_run,
                )

            else:
                if current_run > 0:
                    missing_run_count += 1

                current_run = 0

        if current_run > 0:
            missing_run_count += 1

    # --------------------------------------------------------
    # 5. Build individual quality checks
    # --------------------------------------------------------
    checks: list[dict[str, Any]] = []

    def add_check(
        check_name: str,
        passed: bool | None,
        severity: str,
        measured_value: Any,
        expected_condition: str,
        details: str,
    ) -> None:
        if passed is True:
            status = "PASS"
        elif passed is False:
            status = "FAIL"
        else:
            status = "REVIEW"

        checks.append(
            {
                "target_column": target_column,
                "check_name": check_name,
                "status": status,
                "severity": severity,
                "measured_value": safe_json_value(
                    measured_value
                ),
                "expected_condition": expected_condition,
                "details": details,
            }
        )

    add_check(
        check_name="target_column_exists",
        passed=True,
        severity="CRITICAL",
        measured_value=target_column,
        expected_condition="Requested target must exist.",
        details="The requested target column was found.",
    )

    add_check(
        check_name="valid_datetime_index",
        passed=index_is_valid,
        severity="CRITICAL",
        measured_value=index_is_valid,
        expected_condition="All index values must be valid timestamps.",
        details="Datetime index validity was checked.",
    )

    add_check(
        check_name="unique_timestamps",
        passed=index_is_unique,
        severity="HIGH",
        measured_value=index_is_unique,
        expected_condition="No repeated timestamps.",
        details="Repeated timestamps may cause ambiguous time-series rows.",
    )

    add_check(
        check_name="chronological_order",
        passed=index_is_monotonic,
        severity="HIGH",
        measured_value=index_is_monotonic,
        expected_condition="Timestamps must be sorted chronologically.",
        details="Chronological order is required for reliable time splits.",
    )

    add_check(
        check_name="minimum_valid_observations",
        passed=sufficient_observations,
        severity="HIGH",
        measured_value=valid_target_count,
        expected_condition=(
            f"At least {minimum_observations} finite target values."
        ),
        details=(
            "This is a configurable screening threshold, not a "
            "universal statistical guarantee."
        ),
    )

    add_check(
        check_name="missing_target_percentage",
        passed=missingness_acceptable,
        severity="HIGH",
        measured_value=round(missing_percentage, 4),
        expected_condition=(
            "Missing target percentage must not exceed "
            f"{maximum_missing_percentage}%."
        ),
        details=(
            "Missing values are measured and preserved; they are "
            "not automatically replaced with zero."
        ),
    )

    add_check(
        check_name="negative_demand_values",
        passed=negative_values_acceptable,
        severity="HIGH",
        measured_value=negative_value_count,
        expected_condition=(
            "No negative values."
            if require_non_negative_target
            else "Negative values are permitted but must be reviewed."
        ),
        details=(
            "Negative observations should be investigated to determine "
            "whether they are valid adjustments or data-quality issues."
        ),
    )

    add_check(
        check_name="target_variation",
        passed=target_has_variation,
        severity="HIGH",
        measured_value=distinct_value_count,
        expected_condition=(
            "At least "
            f"{minimum_target_variation} distinct finite target values."
        ),
        details=(
            "A constant target generally provides little information "
            "for evaluating a demand-forecasting model."
        ),
    )

    add_check(
        check_name="regular_sampling",
        passed=regular_sampling_detected,
        severity="MEDIUM",
        measured_value=(
            round(interval_consistency_percentage, 4)
            if interval_consistency_percentage is not None
            else None
        ),
        expected_condition=(
            "At least 95% of observed intervals match the dominant "
            "sampling interval within a 1% tolerance."
        ),
        details=(
            "This check evaluates timestamp spacing. It does not prove "
            "that demand observations are complete."
        ),
    )

    # --------------------------------------------------------
    # 6. Determine whether blocking quality gates passed
    # --------------------------------------------------------
    check_dataframe = pd.DataFrame(
        checks
    )

    blocking_checks = check_dataframe[
        check_dataframe["severity"].isin(
            ["CRITICAL", "HIGH"]
        )
    ]

    blocking_failures = safe_int(
        (
            blocking_checks["status"] == "FAIL"
        ).sum()
    )

    review_checks = safe_int(
        (
            check_dataframe["status"] == "REVIEW"
        ).sum()
    )

    passed_checks = safe_int(
        (
            check_dataframe["status"] == "PASS"
        ).sum()
    )

    failed_checks = safe_int(
        (
            check_dataframe["status"] == "FAIL"
        ).sum()
    )

    if blocking_failures > 0:
        readiness_status = "NOT_READY"
    elif review_checks > 0:
        readiness_status = "REQUIRES_REVIEW"
    else:
        readiness_status = "BASIC_QUALITY_GATES_PASSED"

    # --------------------------------------------------------
    # 7. Create an auditable quality report
    # --------------------------------------------------------
    quality_summary: dict[str, Any] = {
        "generated_at_utc": get_utc_timestamp(),
        "report_version": REPORT_VERSION,
        "target_column": target_column,
        "readiness_status": readiness_status,
        "total_intervals": total_intervals,
        "valid_target_observations": valid_target_count,
        "missing_target_observations": missing_target_count,
        "missing_target_percentage": round(
            missing_percentage,
            4,
        ),
        "negative_target_count": negative_value_count,
        "zero_target_count": zero_value_count,
        "positive_target_count": positive_value_count,
        "distinct_target_values": distinct_value_count,
        "minimum_observation_threshold": minimum_observations,
        "maximum_missing_percentage_threshold": (
            maximum_missing_percentage
        ),
        "require_non_negative_target": require_non_negative_target,
        "valid_datetime_index": index_is_valid,
        "unique_timestamps": index_is_unique,
        "chronological_order": index_is_monotonic,
        "dominant_interval_minutes": observed_interval_minutes,
        "median_interval_minutes": median_interval_minutes,
        "interval_consistency_percentage": (
            round(interval_consistency_percentage, 4)
            if interval_consistency_percentage is not None
            else None
        ),
        "irregular_interval_count": irregular_interval_count,
        "regular_sampling_detected": regular_sampling_detected,
        "missing_run_count": missing_run_count,
        "longest_consecutive_missing_run": longest_missing_run,
        "passed_check_count": passed_checks,
        "failed_check_count": failed_checks,
        "review_check_count": review_checks,
        "blocking_failure_count": blocking_failures,
        "model_trained": False,
        "forecast_accuracy_verified": False,
        "recommendation": (
            "Proceed to chronological baseline and model evaluation."
            if readiness_status == "BASIC_QUALITY_GATES_PASSED"
            else (
                "Resolve blocking failures or review flagged conditions "
                "before using this series for model development."
            )
        ),
        "limitations": [
            "Passing quality gates does not guarantee forecast accuracy.",
            "Model selection requires chronological validation.",
            "Forecast metrics must be calculated on held-out observations.",
            "Future data must not leak into feature construction.",
            "Demand forecasts do not directly measure queue length "
            "or ferry capacity.",
        ],
    }

    # --------------------------------------------------------
    # 8. Save detailed checks and summary
    # --------------------------------------------------------
    safe_target_name = (
        target_column.lower()
        .replace(" ", "_")
        .replace("/", "_")
    )

    checks_path = (
        OUTPUT_DIR
        / f"forecasting_quality_gates_{safe_target_name}.csv"
    )

    summary_path = (
        REPORT_DIR
        / f"forecasting_quality_summary_{safe_target_name}.json"
    )

    save_dataframe_csv(
        check_dataframe,
        checks_path,
    )

    save_json_report(
        quality_summary,
        summary_path,
    )

    LOGGER.info(
        "Quality-gate assessment completed for %s: %s.",
        target_column,
        readiness_status,
    )

    return check_dataframe, quality_summary


# END OF PART 15
# ============================================================
# ============================================================
# PART 16: FORECASTING BASELINE STRATEGY AND EVALUATION METRICS
# ============================================================

def calculate_forecasting_metrics(
    actual_values: pd.Series | np.ndarray | list,
    predicted_values: pd.Series | np.ndarray | list,
    seasonal_period: int = 96,
) -> dict[str, Any]:
    """
    Calculate standard regression and forecasting evaluation metrics.

    Metrics:
        - MAE: Mean Absolute Error
        - RMSE: Root Mean Squared Error
        - WAPE: Weighted Absolute Percentage Error
        - sMAPE: Symmetric Mean Absolute Percentage Error
        - MASE: Mean Absolute Scaled Error

    Notes:
        - Predictions must come from a model that was fitted without
          using the evaluation observations.
        - MASE requires a valid historical naive-forecast scale.
        - Percentage metrics can be misleading when demand is zero.
        - These metrics do not establish operational safety.
    """

    actual = np.asarray(
        actual_values,
        dtype="float64",
    ).reshape(-1)

    predicted = np.asarray(
        predicted_values,
        dtype="float64",
    ).reshape(-1)

    if len(actual) != len(predicted):
        raise ValueError(
            "Actual and predicted arrays must have equal lengths."
        )

    if seasonal_period < 1:
        raise ValueError(
            "seasonal_period must be at least 1."
        )

    finite_mask = (
        np.isfinite(actual)
        & np.isfinite(predicted)
    )

    actual = actual[finite_mask]
    predicted = predicted[finite_mask]

    observation_count = safe_int(
        len(actual)
    )

    if observation_count == 0:
        return {
            "observation_count": 0,
            "mae": None,
            "rmse": None,
            "wape_percentage": None,
            "smape_percentage": None,
            "mase": None,
            "status": "NO_VALID_EVALUATION_OBSERVATIONS",
        }

    errors = actual - predicted

    absolute_errors = np.abs(
        errors
    )

    squared_errors = np.square(
        errors
    )

    mae = float(
        np.mean(absolute_errors)
    )

    rmse = float(
        np.sqrt(np.mean(squared_errors))
    )

    total_absolute_actual = float(
        np.sum(np.abs(actual))
    )

    total_absolute_error = float(
        np.sum(absolute_errors)
    )

    if total_absolute_actual > 0:
        wape = (
            total_absolute_error
            / total_absolute_actual
            * 100.0
        )
    else:
        wape = None

    smape_denominator = (
        np.abs(actual)
        + np.abs(predicted)
    )

    smape_terms = np.divide(
        2.0 * absolute_errors,
        smape_denominator,
        out=np.zeros_like(absolute_errors),
        where=smape_denominator > 0,
    )

    smape = float(
        np.mean(smape_terms) * 100.0
    )

    # MASE requires the original chronological actual series.
    # This calculation uses the supplied evaluation sequence as a
    # scale only when it contains enough observations. For rigorous
    # reporting, pass a scale computed from training data instead.
    if observation_count > seasonal_period:
        naive_errors = np.abs(
            actual[seasonal_period:]
            - actual[:-seasonal_period]
        )

        naive_scale = float(
            np.mean(naive_errors)
        )

        if naive_scale > 0:
            mase = float(
                mae / naive_scale
            )
        else:
            mase = None
    else:
        mase = None

    return {
        "observation_count": observation_count,
        "mae": mae,
        "rmse": rmse,
        "wape_percentage": wape,
        "smape_percentage": smape,
        "mase": mase,
        "status": "CALCULATED",
    }


def generate_naive_forecast(
    training_values: pd.Series | np.ndarray | list,
    evaluation_index: pd.Index,
    strategy: str = "last_value",
    seasonal_period: int = 96,
) -> pd.Series:
    """
    Generate a baseline forecast without fitting a machine-learning model.

    Strategies:
        last_value:
            Predict the last valid training observation.

        seasonal_naive:
            Repeat the last observed seasonal pattern.

    The training values must be chronological and must not contain
    information from the evaluation period.
    """

    if strategy not in {
        "last_value",
        "seasonal_naive",
    }:
        raise ValueError(
            "strategy must be 'last_value' or 'seasonal_naive'."
        )

    if seasonal_period < 1:
        raise ValueError(
            "seasonal_period must be at least 1."
        )

    training_array = np.asarray(
        training_values,
        dtype="float64",
    ).reshape(-1)

    training_array = training_array[
        np.isfinite(training_array)
    ]

    forecast_steps = len(
        evaluation_index
    )

    if forecast_steps == 0:
        return pd.Series(
            index=evaluation_index,
            dtype="float64",
            name="Baseline_Prediction",
        )

    if training_array.size == 0:
        raise ValueError(
            "The training series contains no finite observations."
        )

    if strategy == "last_value":
        predictions = np.full(
            forecast_steps,
            training_array[-1],
            dtype="float64",
        )

    else:
        if training_array.size < seasonal_period:
            raise ValueError(
                "Seasonal naive forecasting requires at least "
                f"{seasonal_period} training observations."
            )

        seasonal_pattern = training_array[
            -seasonal_period:
        ]

        repeated_pattern = np.resize(
            seasonal_pattern,
            forecast_steps,
        )

        predictions = repeated_pattern.astype(
            "float64",
            copy=False,
        )

    return pd.Series(
        predictions,
        index=evaluation_index,
        name="Baseline_Prediction",
    )


def evaluate_naive_forecasting_baselines(
    prepared_series: pd.DataFrame,
    target_column: str,
    evaluation_fraction: float = 0.20,
    seasonal_period: int = 96,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Evaluate simple forecasting baselines using a chronological split.

    This is a preliminary benchmark, not a substitute for rolling
    time-series cross-validation.

    The split is created before any evaluation predictions are made.
    Training data is never shuffled.
    """

    if target_column not in prepared_series.columns:
        raise KeyError(
            f"Target '{target_column}' is missing."
        )

    if not 0 < evaluation_fraction < 1:
        raise ValueError(
            "evaluation_fraction must be greater than 0 and less than 1."
        )

    if seasonal_period < 1:
        raise ValueError(
            "seasonal_period must be at least 1."
        )

    working = prepared_series.copy()

    if not isinstance(
        working.index,
        pd.DatetimeIndex,
    ):
        working.index = pd.to_datetime(
            working.index,
            errors="coerce",
            utc=True,
        )

    if working.index.hasnans:
        raise ValueError(
            "The forecasting series contains invalid timestamps."
        )

    if working.index.has_duplicates:
        raise ValueError(
            "The forecasting series contains duplicate timestamps."
        )

    working = working.sort_index()

    numeric_target = pd.to_numeric(
        working[target_column],
        errors="coerce",
    )

    numeric_target = numeric_target.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Preserve missing positions for the report. Baseline models use
    # finite training observations, while evaluation metrics only use
    # finite evaluation targets.
    total_rows = len(working)

    split_position = int(
        np.floor(
            total_rows * (1 - evaluation_fraction)
        )
    )

    if split_position <= 0 or split_position >= total_rows:
        raise ValueError(
            "The dataset is too small for the requested chronological split."
        )

    training_series = numeric_target.iloc[
        :split_position
    ]

    evaluation_series = numeric_target.iloc[
        split_position:
    ]

    training_values = training_series.dropna().to_numpy()

    if training_values.size == 0:
        raise ValueError(
            "No valid target observations exist in the training period."
        )

    valid_evaluation_mask = (
        evaluation_series.notna()
    )

    if not valid_evaluation_mask.any():
        raise ValueError(
            "No valid target observations exist in the evaluation period."
        )

    evaluation_index = evaluation_series.index

    baseline_records: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    strategies = [
        "last_value",
        "seasonal_naive",
    ]

    # --------------------------------------------------------
    # Evaluate each baseline against the same held-out period
    # --------------------------------------------------------
    for strategy in strategies:
        try:
            prediction_series = generate_naive_forecast(
                training_values=training_values,
                evaluation_index=evaluation_index,
                strategy=strategy,
                seasonal_period=seasonal_period,
            )

            comparison = pd.DataFrame(
                {
                    "Timestamp_UTC": evaluation_index,
                    "Actual": evaluation_series.to_numpy(),
                    "Predicted": prediction_series.to_numpy(),
                }
            )

            comparison["Model_Name"] = strategy

            valid_comparison = comparison[
                np.isfinite(
                    comparison["Actual"].to_numpy(
                        dtype="float64"
                    )
                )
                & np.isfinite(
                    comparison["Predicted"].to_numpy(
                        dtype="float64"
                    )
                )
            ].copy()

            metrics = calculate_forecasting_metrics(
                actual_values=valid_comparison["Actual"],
                predicted_values=valid_comparison["Predicted"],
                seasonal_period=seasonal_period,
            )

            baseline_records.append(
                {
                    "target_column": target_column,
                    "model_name": strategy,
                    "evaluation_fraction": evaluation_fraction,
                    "training_interval_count": safe_int(
                        len(training_series)
                    ),
                    "valid_training_observation_count": safe_int(
                        len(training_values)
                    ),
                    "evaluation_interval_count": safe_int(
                        len(evaluation_series)
                    ),
                    "valid_evaluation_observation_count": safe_int(
                        len(valid_comparison)
                    ),
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "wape_percentage": metrics[
                        "wape_percentage"
                    ],
                    "smape_percentage": metrics[
                        "smape_percentage"
                    ],
                    "mase": metrics["mase"],
                    "evaluation_status": metrics["status"],
                    "evaluation_start_utc": (
                        str(evaluation_series.index.min())
                    ),
                    "evaluation_end_utc": (
                        str(evaluation_series.index.max())
                    ),
                }
            )

            prediction_frames.append(
                comparison
            )

        except ValueError as exc:
            LOGGER.warning(
                "Baseline '%s' could not be evaluated: %s",
                strategy,
                exc,
            )

            baseline_records.append(
                {
                    "target_column": target_column,
                    "model_name": strategy,
                    "evaluation_fraction": evaluation_fraction,
                    "training_interval_count": safe_int(
                        len(training_series)
                    ),
                    "valid_training_observation_count": safe_int(
                        len(training_values)
                    ),
                    "evaluation_interval_count": safe_int(
                        len(evaluation_series)
                    ),
                    "valid_evaluation_observation_count": 0,
                    "mae": None,
                    "rmse": None,
                    "wape_percentage": None,
                    "smape_percentage": None,
                    "mase": None,
                    "evaluation_status": "NOT_EVALUATED",
                    "error_message": str(exc),
                    "evaluation_start_utc": (
                        str(evaluation_series.index.min())
                    ),
                    "evaluation_end_utc": (
                        str(evaluation_series.index.max())
                    ),
                }
            )

    metrics_dataframe = pd.DataFrame(
        baseline_records
    )

    if prediction_frames:
        predictions_dataframe = pd.concat(
            prediction_frames,
            ignore_index=True,
        )
    else:
        predictions_dataframe = pd.DataFrame(
            columns=[
                "Timestamp_UTC",
                "Actual",
                "Predicted",
                "Model_Name",
            ]
        )

    # --------------------------------------------------------
    # Persist benchmark outputs
    # --------------------------------------------------------
    safe_target_name = (
        target_column.lower()
        .replace(" ", "_")
        .replace("/", "_")
    )

    metrics_path = (
        OUTPUT_DIR
        / f"baseline_metrics_{safe_target_name}.csv"
    )

    predictions_path = (
        OUTPUT_DIR
        / f"baseline_predictions_{safe_target_name}.csv"
    )

    save_dataframe_csv(
        metrics_dataframe,
        metrics_path,
    )

    save_dataframe_csv(
        predictions_dataframe,
        predictions_path,
    )

    # --------------------------------------------------------
    # Produce an explicit benchmark summary
    # --------------------------------------------------------
    successful_results = metrics_dataframe[
        metrics_dataframe["evaluation_status"] == "CALCULATED"
    ].copy()

    best_model_name = None
    best_model_mae = None

    if not successful_results.empty:
        successful_results = successful_results.dropna(
            subset=["mae"]
        )

        if not successful_results.empty:
            best_row = successful_results.sort_values(
                by="mae",
                ascending=True,
            ).iloc[0]

            best_model_name = str(
                best_row["model_name"]
            )

            best_model_mae = float(
                best_row["mae"]
            )

    summary: dict[str, Any] = {
        "generated_at_utc": get_utc_timestamp(),
        "target_column": target_column,
        "evaluation_method": "chronological_holdout",
        "evaluation_fraction": evaluation_fraction,
        "seasonal_period": seasonal_period,
        "total_time_intervals": total_rows,
        "training_start_utc": str(
            training_series.index.min()
        ),
        "training_end_utc": str(
            training_series.index.max()
        ),
        "evaluation_start_utc": str(
            evaluation_series.index.min()
        ),
        "evaluation_end_utc": str(
            evaluation_series.index.max()
        ),
        "training_observation_count": safe_int(
            training_values.size
        ),
        "evaluation_observation_count": safe_int(
            evaluation_series.notna().sum()
        ),
        "evaluated_baseline_count": safe_int(
            len(successful_results)
        ),
        "best_baseline_by_mae": best_model_name,
        "best_baseline_mae": best_model_mae,
        "metrics_output": str(metrics_path),
        "predictions_output": str(predictions_path),
        "limitations": [
            "Baseline results do not prove that a complex model will perform better.",
            "The evaluation period must remain separate from model fitting.",
            "This implementation is a single chronological holdout.",
            "Rolling-origin evaluation is recommended for stronger evidence.",
            "MASE should ideally use a scale calculated from training data.",
            "No model has been approved for production by this benchmark.",
        ],
    }

    summary_path = (
        REPORT_DIR
        / f"baseline_summary_{safe_target_name}.json"
    )

    save_json_report(
        summary,
        summary_path,
    )

    LOGGER.info(
        "Baseline evaluation completed for target '%s'.",
        target_column,
    )

    return (
        metrics_dataframe,
        predictions_dataframe,
        summary,
    )


# END OF PART 16
# ============================================================
# ============================================================
# PART 17: FORECAST HORIZON CONFIGURATION AND EVALUATION
# ============================================================

def build_forecast_horizon_configuration(
    frequency: str = "15min",
    forecast_horizons_minutes: list[int] | None = None,
) -> pd.DataFrame:
    """
    Build a validated configuration for short-term demand forecasts.

    Default forecast horizons:
        - 15 minutes
        - 30 minutes
        - 60 minutes
        - 120 minutes

    Each horizon is converted into the number of time steps required
    by the selected forecasting frequency.

    A horizon must be an exact multiple of the frequency. For example,
    a 30-minute horizon is valid with 15-minute intervals, while a
    20-minute horizon is not.
    """

    LOGGER.info(
        "Building forecast horizon configuration."
    )

    if forecast_horizons_minutes is None:
        forecast_horizons_minutes = [
            15,
            30,
            60,
            120,
        ]

    if not forecast_horizons_minutes:
        raise ValueError(
            "At least one forecast horizon must be supplied."
        )

    if any(
        not isinstance(horizon, (int, np.integer))
        or isinstance(horizon, (bool, np.bool_))
        for horizon in forecast_horizons_minutes
    ):
        raise TypeError(
            "Forecast horizons must be integers expressed in minutes."
        )

    if any(
        horizon <= 0
        for horizon in forecast_horizons_minutes
    ):
        raise ValueError(
            "Every forecast horizon must be greater than zero."
        )

    if len(forecast_horizons_minutes) != len(
        set(forecast_horizons_minutes)
    ):
        raise ValueError(
            "Duplicate forecast horizons are not allowed."
        )

    try:
        frequency_offset = pd.tseries.frequencies.to_offset(
            frequency
        )

        frequency_delta = pd.Timedelta(
            frequency_offset.nanos,
            unit="ns",
        )

    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Frequency must be a valid fixed-duration interval, "
            "such as '15min', '30min', or '1h'."
        ) from exc

    frequency_minutes = (
        frequency_delta.total_seconds() / 60
    )

    if frequency_minutes <= 0:
        raise ValueError(
            "Frequency must be greater than zero."
        )

    configuration_records: list[dict[str, Any]] = []

    for horizon_minutes in sorted(
        forecast_horizons_minutes
    ):
        step_ratio = (
            horizon_minutes / frequency_minutes
        )

        exact_multiple = bool(
            np.isclose(
                step_ratio,
                round(step_ratio),
                rtol=0,
                atol=1e-9,
            )
        )

        if not exact_multiple:
            raise ValueError(
                f"The {horizon_minutes}-minute horizon is not an exact "
                f"multiple of the {frequency} frequency."
            )

        horizon_steps = int(
            round(step_ratio)
        )

        configuration_records.append(
            {
                "frequency": frequency,
                "frequency_minutes": frequency_minutes,
                "forecast_horizon_minutes": horizon_minutes,
                "forecast_horizon_steps": horizon_steps,
                "horizon_label": (
                    f"{horizon_minutes} minutes"
                    if horizon_minutes < 60
                    else (
                        f"{horizon_minutes // 60} hour"
                        if horizon_minutes == 60
                        else (
                            f"{horizon_minutes // 60} hours"
                            if horizon_minutes % 60 == 0
                            else f"{horizon_minutes} minutes"
                        )
                    )
                ),
                "configuration_status": "VALID",
            }
        )

    configuration_dataframe = pd.DataFrame(
        configuration_records
    )

    configuration_path = (
        OUTPUT_DIR / "forecast_horizon_configuration.csv"
    )

    save_dataframe_csv(
        configuration_dataframe,
        configuration_path,
    )

    LOGGER.info(
        "Validated %s forecast horizons.",
        len(configuration_dataframe),
    )

    return configuration_dataframe


def evaluate_baselines_by_forecast_horizon(
    prepared_series: pd.DataFrame,
    target_column: str,
    frequency: str = "15min",
    forecast_horizons_minutes: list[int] | None = None,
    evaluation_fraction: float = 0.20,
    seasonal_period: int = 96,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Evaluate baseline forecasting strategies at multiple horizons.

    Strategies:
        - Last observed value.
        - Seasonal naive forecast.

    Evaluation design:
        1. Sort the time series chronologically.
        2. Split training and evaluation periods.
        3. Construct each forecast using training observations only.
        4. Evaluate predictions at the requested lead times.
        5. Report metrics separately for each horizon.

    Forecast horizon interpretation:
        A 15-minute horizon evaluates predictions one time step ahead
        when the input frequency is 15 minutes.

        A 60-minute horizon evaluates predictions four time steps ahead
        when the input frequency is 15 minutes.

    This function uses a simple fixed-origin baseline evaluation.
    It does not train machine-learning models.
    """

    if target_column not in prepared_series.columns:
        raise KeyError(
            f"Target '{target_column}' was not found."
        )

    if not 0 < evaluation_fraction < 1:
        raise ValueError(
            "evaluation_fraction must be between 0 and 1."
        )

    if seasonal_period < 1:
        raise ValueError(
            "seasonal_period must be at least 1."
        )

    horizon_configuration = (
        build_forecast_horizon_configuration(
            frequency=frequency,
            forecast_horizons_minutes=forecast_horizons_minutes,
        )
    )

    working = prepared_series.copy()

    if not isinstance(
        working.index,
        pd.DatetimeIndex,
    ):
        working.index = pd.to_datetime(
            working.index,
            errors="coerce",
            utc=True,
        )

    if working.index.hasnans:
        raise ValueError(
            "The time-series index contains invalid timestamps."
        )

    if working.index.has_duplicates:
        raise ValueError(
            "The time-series index contains duplicate timestamps."
        )

    working = working.sort_index()

    target_series = pd.to_numeric(
        working[target_column],
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    total_intervals = len(
        target_series
    )

    split_position = int(
        np.floor(
            total_intervals * (1 - evaluation_fraction)
        )
    )

    if split_position <= 0 or split_position >= total_intervals:
        raise ValueError(
            "Insufficient time intervals for the requested split."
        )

    training_series = target_series.iloc[
        :split_position
    ]

    evaluation_series = target_series.iloc[
        split_position:
    ]

    training_values = training_series.dropna().to_numpy()

    if training_values.size == 0:
        raise ValueError(
            "No valid observations are available in the training period."
        )

    maximum_horizon_steps = safe_int(
        horizon_configuration[
            "forecast_horizon_steps"
        ].max()
    )

    if len(evaluation_series) <= maximum_horizon_steps:
        raise ValueError(
            "The evaluation period is too short for the largest "
            "requested forecast horizon."
        )

    result_records: list[dict[str, Any]] = []
    prediction_records: list[pd.DataFrame] = []

    # --------------------------------------------------------
    # Generate and evaluate predictions for each horizon
    # --------------------------------------------------------
    for horizon_row in horizon_configuration.to_dict(
        orient="records"
    ):
        horizon_minutes = int(
            horizon_row["forecast_horizon_minutes"]
        )

        horizon_steps = int(
            horizon_row["forecast_horizon_steps"]
        )

        # The forecast is issued at the end of the training period.
        # The evaluation target is sampled at the specified lead time
        # after the forecast origin.
        evaluation_positions = np.arange(
            horizon_steps - 1,
            len(evaluation_series),
            horizon_steps,
        )

        if evaluation_positions.size == 0:
            LOGGER.warning(
                "No evaluation points for %s-minute horizon.",
                horizon_minutes,
            )
            continue

        evaluation_targets = evaluation_series.iloc[
            evaluation_positions
        ]

        evaluation_timestamps = evaluation_targets.index

        for strategy in [
            "last_value",
            "seasonal_naive",
        ]:
            try:
                # Generate a baseline forecast for each evaluation
                # point from training data only. This is a fixed-origin
                # evaluation: observations from the evaluation period
                # are never fed back into the forecast.
                base_prediction = generate_naive_forecast(
                    training_values=training_values,
                    evaluation_index=pd.RangeIndex(
                        len(evaluation_positions)
                    ),
                    strategy=strategy,
                    seasonal_period=seasonal_period,
                )

                prediction_values = base_prediction.to_numpy()

                comparison = pd.DataFrame(
                    {
                        "Timestamp_UTC": evaluation_timestamps,
                        "Actual": evaluation_targets.to_numpy(),
                        "Predicted": prediction_values,
                        "Target_Column": target_column,
                        "Model_Name": strategy,
                        "Forecast_Horizon_Minutes": horizon_minutes,
                        "Forecast_Horizon_Steps": horizon_steps,
                    }
                )

                valid_mask = (
                    np.isfinite(
                        comparison["Actual"].to_numpy(
                            dtype="float64"
                        )
                    )
                    & np.isfinite(
                        comparison["Predicted"].to_numpy(
                            dtype="float64"
                        )
                    )
                )

                valid_comparison = comparison.loc[
                    valid_mask
                ].copy()

                metrics = calculate_forecasting_metrics(
                    actual_values=valid_comparison["Actual"],
                    predicted_values=valid_comparison["Predicted"],
                    seasonal_period=seasonal_period,
                )

                result_records.append(
                    {
                        "target_column": target_column,
                        "model_name": strategy,
                        "forecast_horizon_minutes": horizon_minutes,
                        "forecast_horizon_steps": horizon_steps,
                        "evaluation_observation_count": (
                            metrics["observation_count"]
                        ),
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "wape_percentage": (
                            metrics["wape_percentage"]
                        ),
                        "smape_percentage": (
                            metrics["smape_percentage"]
                        ),
                        "mase": metrics["mase"],
                        "evaluation_status": metrics["status"],
                        "evaluation_start_utc": (
                            str(evaluation_timestamps.min())
                        ),
                        "evaluation_end_utc": (
                            str(evaluation_timestamps.max())
                        ),
                    }
                )

                prediction_records.append(
                    comparison
                )

            except ValueError as exc:
                LOGGER.warning(
                    "Could not evaluate %s for horizon %s minutes: %s",
                    strategy,
                    horizon_minutes,
                    exc,
                )

                result_records.append(
                    {
                        "target_column": target_column,
                        "model_name": strategy,
                        "forecast_horizon_minutes": horizon_minutes,
                        "forecast_horizon_steps": horizon_steps,
                        "evaluation_observation_count": 0,
                        "mae": None,
                        "rmse": None,
                        "wape_percentage": None,
                        "smape_percentage": None,
                        "mase": None,
                        "evaluation_status": "NOT_EVALUATED",
                        "error_message": str(exc),
                    }
                )

    metrics_dataframe = pd.DataFrame(
        result_records
    )

    if prediction_records:
        predictions_dataframe = pd.concat(
            prediction_records,
            ignore_index=True,
        )
    else:
        predictions_dataframe = pd.DataFrame(
            columns=[
                "Timestamp_UTC",
                "Actual",
                "Predicted",
                "Target_Column",
                "Model_Name",
                "Forecast_Horizon_Minutes",
                "Forecast_Horizon_Steps",
            ]
        )

    # --------------------------------------------------------
    # Save horizon-specific metrics and prediction records
    # --------------------------------------------------------
    safe_target_name = (
        target_column.lower()
        .replace(" ", "_")
        .replace("/", "_")
    )

    metrics_path = (
        OUTPUT_DIR
        / f"horizon_baseline_metrics_{safe_target_name}.csv"
    )

    predictions_path = (
        OUTPUT_DIR
        / f"horizon_baseline_predictions_{safe_target_name}.csv"
    )

    save_dataframe_csv(
        metrics_dataframe,
        metrics_path,
    )

    save_dataframe_csv(
        predictions_dataframe,
        predictions_path,
    )

    # --------------------------------------------------------
    # Summarize evaluation coverage and limitations
    # --------------------------------------------------------
    if metrics_dataframe.empty:
        evaluated_horizon_count = 0
    else:
        evaluated_horizon_count = safe_int(
            metrics_dataframe.loc[
                metrics_dataframe["evaluation_status"] == "CALCULATED",
                "forecast_horizon_minutes",
            ].nunique()
        )

    summary: dict[str, Any] = {
        "generated_at_utc": get_utc_timestamp(),
        "target_column": target_column,
        "frequency": frequency,
        "forecast_horizons_minutes": (
            horizon_configuration[
                "forecast_horizon_minutes"
            ].tolist()
        ),
        "evaluation_fraction": evaluation_fraction,
        "evaluation_design": (
            "Chronological fixed-origin baseline evaluation."
        ),
        "training_interval_count": safe_int(
            len(training_series)
        ),
        "valid_training_observation_count": safe_int(
            len(training_values)
        ),
        "evaluation_interval_count": safe_int(
            len(evaluation_series)
        ),
        "evaluated_horizon_count": evaluated_horizon_count,
        "metrics_output": str(metrics_path),
        "predictions_output": str(predictions_path),
        "model_training_performed": False,
        "production_approval_granted": False,
        "limitations": [
            "Baseline metrics are benchmarks, not proof of production readiness.",
            "The fixed-origin strategy does not update forecasts with new observations.",
            "Forecasts at different horizons may have different evaluation sample counts.",
            "The selected seasonal period must match the intended time-series frequency.",
            "A 96-step seasonal period represents one day only for a complete 15-minute series.",
            "Rolling-origin evaluation should be used before selecting a production model.",
        ],
    }

    summary_path = (
        REPORT_DIR
        / f"horizon_baseline_summary_{safe_target_name}.json"
    )

    save_json_report(
        summary,
        summary_path,
    )

    LOGGER.info(
        "Multi-horizon baseline evaluation completed for %s.",
        target_column,
    )

    return (
        metrics_dataframe,
        predictions_dataframe,
        summary,
    )


# END OF PART 17
# ============================================================
# ============================================================
# PART 18: FORECASTING RESULT VALIDATION AND MODEL COMPARISON
# ============================================================

def validate_forecast_predictions(
    predictions_dataframe: pd.DataFrame,
    timestamp_column: str = "Timestamp_UTC",
    actual_column: str = "Actual",
    predicted_column: str = "Predicted",
    model_column: str = "Model_Name",
    horizon_column: str = "Forecast_Horizon_Minutes",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Validate forecast prediction records before comparing models.

    Checks:
        1. Required columns exist.
        2. Timestamps can be parsed.
        3. Actual and predicted values are numeric.
        4. Non-finite values are identified.
        5. Duplicate model/timestamp/horizon records are detected.
        6. Forecast horizons are positive.
        7. Prediction errors are calculated.
        8. Invalid records are flagged rather than silently removed.

    This function validates result records. It does not establish
    that the forecasting model was trained without data leakage.
    """

    LOGGER.info(
        "Validating forecasting prediction records."
    )

    required_columns = [
        timestamp_column,
        actual_column,
        predicted_column,
        model_column,
        horizon_column,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in predictions_dataframe.columns
    ]

    if missing_columns:
        raise KeyError(
            "Forecast prediction data is missing required columns: "
            f"{missing_columns}"
        )

    working = predictions_dataframe.copy()

    original_row_count = safe_int(
        len(working)
    )

    # --------------------------------------------------------
    # 1. Normalize timestamps and numerical prediction fields
    # --------------------------------------------------------
    working[timestamp_column] = pd.to_datetime(
        working[timestamp_column],
        errors="coerce",
        utc=True,
    )

    working[actual_column] = pd.to_numeric(
        working[actual_column],
        errors="coerce",
    )

    working[predicted_column] = pd.to_numeric(
        working[predicted_column],
        errors="coerce",
    )

    working[horizon_column] = pd.to_numeric(
        working[horizon_column],
        errors="coerce",
    )

    actual_values = working[
        actual_column
    ].to_numpy(
        dtype="float64",
        na_value=np.nan,
    )

    predicted_values = working[
        predicted_column
    ].to_numpy(
        dtype="float64",
        na_value=np.nan,
    )

    timestamp_valid = (
        working[timestamp_column].notna()
    )

    actual_finite = pd.Series(
        np.isfinite(actual_values),
        index=working.index,
    )

    predicted_finite = pd.Series(
        np.isfinite(predicted_values),
        index=working.index,
    )

    horizon_valid = (
        working[horizon_column].notna()
        & np.isfinite(
            working[horizon_column].to_numpy(
                dtype="float64",
                na_value=np.nan,
            )
        )
        & (working[horizon_column] > 0)
    )

    model_name_valid = (
        working[model_column]
        .astype("string")
        .str.strip()
        .notna()
        & working[model_column]
        .astype("string")
        .str.strip()
        .ne("")
    )

    # --------------------------------------------------------
    # 2. Detect repeated evaluation records
    # --------------------------------------------------------
    duplicate_key_columns = [
        timestamp_column,
        model_column,
        horizon_column,
    ]

    duplicate_key_mask = working.duplicated(
        subset=duplicate_key_columns,
        keep=False,
    )

    working["Validation_Timestamp_Valid"] = (
        timestamp_valid
    )

    working["Validation_Actual_Finite"] = (
        actual_finite
    )

    working["Validation_Prediction_Finite"] = (
        predicted_finite
    )

    working["Validation_Horizon_Valid"] = (
        horizon_valid
    )

    working["Validation_Model_Name_Valid"] = (
        model_name_valid
    )

    working["Validation_Duplicate_Record"] = (
        duplicate_key_mask
    )

    # --------------------------------------------------------
    # 3. Calculate residuals only for finite values
    # --------------------------------------------------------
    valid_numeric_pair = (
        actual_finite
        & predicted_finite
    )

    working["Prediction_Error"] = np.nan
    working["Absolute_Error"] = np.nan
    working["Squared_Error"] = np.nan

    working.loc[
        valid_numeric_pair,
        "Prediction_Error",
    ] = (
        working.loc[
            valid_numeric_pair,
            actual_column,
        ]
        - working.loc[
            valid_numeric_pair,
            predicted_column,
        ]
    )

    working.loc[
        valid_numeric_pair,
        "Absolute_Error",
    ] = working.loc[
        valid_numeric_pair,
        "Prediction_Error",
    ].abs()

    working.loc[
        valid_numeric_pair,
        "Squared_Error",
    ] = np.square(
        working.loc[
            valid_numeric_pair,
            "Prediction_Error",
        ]
    )

    # --------------------------------------------------------
    # 4. Assign a record-level validation status
    # --------------------------------------------------------
    working["Validation_Status"] = np.select(
        [
            ~timestamp_valid,
            ~actual_finite,
            ~predicted_finite,
            ~horizon_valid,
            ~model_name_valid,
            duplicate_key_mask,
        ],
        [
            "INVALID_TIMESTAMP",
            "INVALID_ACTUAL_VALUE",
            "INVALID_PREDICTION_VALUE",
            "INVALID_FORECAST_HORIZON",
            "INVALID_MODEL_NAME",
            "DUPLICATE_EVALUATION_RECORD",
        ],
        default="VALID",
    )

    valid_record_mask = (
        working["Validation_Status"] == "VALID"
    )

    # --------------------------------------------------------
    # 5. Summarize validation results
    # --------------------------------------------------------
    status_counts = (
        working["Validation_Status"]
        .value_counts(dropna=False)
        .to_dict()
    )

    valid_record_count = safe_int(
        valid_record_mask.sum()
    )

    invalid_record_count = (
        original_row_count - valid_record_count
    )

    validation_summary: dict[str, Any] = {
        "generated_at_utc": get_utc_timestamp(),
        "original_prediction_record_count": original_row_count,
        "valid_prediction_record_count": valid_record_count,
        "invalid_prediction_record_count": invalid_record_count,
        "valid_record_percentage": calculate_percentage(
            valid_record_count,
            original_row_count,
        ),
        "invalid_timestamp_count": safe_int(
            (~timestamp_valid).sum()
        ),
        "non_finite_actual_count": safe_int(
            (~actual_finite).sum()
        ),
        "non_finite_prediction_count": safe_int(
            (~predicted_finite).sum()
        ),
        "invalid_horizon_count": safe_int(
            (~horizon_valid).sum()
        ),
        "invalid_model_name_count": safe_int(
            (~model_name_valid).sum()
        ),
        "duplicate_record_count": safe_int(
            duplicate_key_mask.sum()
        ),
        "validation_status_counts": status_counts,
        "prediction_error_calculation": (
            "Prediction_Error = Actual - Predicted"
        ),
        "duplicate_records_removed": False,
        "invalid_records_removed": False,
        "interpretation": (
            "Records are flagged for auditability. Downstream metrics "
            "should use valid records and document exclusions."
        ),
    }

    # --------------------------------------------------------
    # 6. Save validation artifacts
    # --------------------------------------------------------
    validation_output_path = (
        OUTPUT_DIR / "forecast_prediction_validation.csv"
    )

    summary_output_path = (
        REPORT_DIR / "forecast_prediction_validation_summary.json"
    )

    save_dataframe_csv(
        working,
        validation_output_path,
    )

    save_json_report(
        validation_summary,
        summary_output_path,
    )

    LOGGER.info(
        "Prediction validation finished: %s valid, %s invalid.",
        format_number(valid_record_count),
        format_number(invalid_record_count),
    )

    return working, validation_summary


def compare_forecasting_models(
    validated_predictions: pd.DataFrame,
    actual_column: str = "Actual",
    predicted_column: str = "Predicted",
    model_column: str = "Model_Name",
    horizon_column: str = "Forecast_Horizon_Minutes",
    target_column: str = "Target_Column",
    validation_status_column: str = "Validation_Status",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Compare candidate forecasting models using validated predictions.

    Comparisons are grouped by target and forecast horizon to prevent
    results from different targets or lead times from being mixed.

    Models are ranked by MAE only when sufficient valid observations
    exist. The ranking is a descriptive benchmark, not a production
    approval decision.
    """

    LOGGER.info(
        "Comparing forecasting model results."
    )

    required_columns = [
        actual_column,
        predicted_column,
        model_column,
        horizon_column,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in validated_predictions.columns
    ]

    if missing_columns:
        raise KeyError(
            "Cannot compare models because required columns are missing: "
            f"{missing_columns}"
        )

    working = validated_predictions.copy()

    if validation_status_column in working.columns:
        working = working.loc[
            working[validation_status_column] == "VALID"
        ].copy()

    working[actual_column] = pd.to_numeric(
        working[actual_column],
        errors="coerce",
    )

    working[predicted_column] = pd.to_numeric(
        working[predicted_column],
        errors="coerce",
    )

    working[horizon_column] = pd.to_numeric(
        working[horizon_column],
        errors="coerce",
    )

    finite_mask = (
        np.isfinite(
            working[actual_column].to_numpy(
                dtype="float64",
                na_value=np.nan,
            )
        )
        & np.isfinite(
            working[predicted_column].to_numpy(
                dtype="float64",
                na_value=np.nan,
            )
        )
        & working[horizon_column].notna().to_numpy()
    )

    working = working.loc[
        finite_mask
    ].copy()

    if working.empty:
        summary = {
            "generated_at_utc": get_utc_timestamp(),
            "status": "NO_VALID_PREDICTIONS",
            "model_comparison_count": 0,
            "production_approval_granted": False,
            "message": (
                "No valid prediction records were available "
                "for model comparison."
            ),
        }

        save_json_report(
            summary,
            REPORT_DIR / "forecast_model_comparison_summary.json",
        )

        return pd.DataFrame(), summary

    if target_column not in working.columns:
        working[target_column] = "UNSPECIFIED_TARGET"

    grouping_columns = [
        target_column,
        horizon_column,
        model_column,
    ]

    comparison_records: list[dict[str, Any]] = []

    # --------------------------------------------------------
    # 1. Calculate comparable metrics for each model/horizon
    # --------------------------------------------------------
    grouped = working.groupby(
        grouping_columns,
        dropna=False,
        sort=True,
    )

    for group_key, group in grouped:
        group_target, group_horizon, group_model = group_key

        actual = group[
            actual_column
        ].to_numpy(
            dtype="float64"
        )

        predicted = group[
            predicted_column
        ].to_numpy(
            dtype="float64"
        )

        metrics = calculate_forecasting_metrics(
            actual_values=actual,
            predicted_values=predicted,
            seasonal_period=96,
        )

        comparison_records.append(
            {
                "target_column": str(group_target),
                "forecast_horizon_minutes": safe_int(
                    group_horizon
                ),
                "model_name": str(group_model),
                "evaluation_observation_count": (
                    metrics["observation_count"]
                ),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "wape_percentage": (
                    metrics["wape_percentage"]
                ),
                "smape_percentage": (
                    metrics["smape_percentage"]
                ),
                "mase": metrics["mase"],
                "evaluation_status": metrics["status"],
            }
        )

    comparison_dataframe = pd.DataFrame(
        comparison_records
    )

    # --------------------------------------------------------
    # 2. Rank models within each target and forecast horizon
    # --------------------------------------------------------
    comparison_dataframe["mae_rank"] = (
        comparison_dataframe.groupby(
            [
                "target_column",
                "forecast_horizon_minutes",
            ],
            dropna=False,
        )["mae"]
        .rank(
            method="min",
            ascending=True,
        )
    )

    comparison_dataframe["rmse_rank"] = (
        comparison_dataframe.groupby(
            [
                "target_column",
                "forecast_horizon_minutes",
            ],
            dropna=False,
        )["rmse"]
        .rank(
            method="min",
            ascending=True,
        )
    )

    comparison_dataframe["lowest_mae_candidate"] = (
        comparison_dataframe["mae_rank"] == 1
    )

    comparison_dataframe["lowest_rmse_candidate"] = (
        comparison_dataframe["rmse_rank"] == 1
    )

    comparison_dataframe["production_approval"] = (
        "NOT_GRANTED"
    )

    comparison_dataframe = comparison_dataframe.sort_values(
        by=[
            "target_column",
            "forecast_horizon_minutes",
            "mae_rank",
        ],
        na_position="last",
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # 3. Persist model comparison results
    # --------------------------------------------------------
    comparison_output_path = (
        OUTPUT_DIR / "forecast_model_comparison.csv"
    )

    save_dataframe_csv(
        comparison_dataframe,
        comparison_output_path,
    )

    # --------------------------------------------------------
    # 4. Produce a concise model-comparison summary
    # --------------------------------------------------------
    successful_comparisons = comparison_dataframe.loc[
        comparison_dataframe["evaluation_status"] == "CALCULATED"
    ]

    candidate_count = safe_int(
        len(successful_comparisons)
    )

    target_count = safe_int(
        comparison_dataframe["target_column"].nunique()
    )

    horizon_count = safe_int(
        comparison_dataframe[
            "forecast_horizon_minutes"
        ].nunique()
    )

    summary: dict[str, Any] = {
        "generated_at_utc": get_utc_timestamp(),
        "status": (
            "COMPARISON_COMPLETED"
            if candidate_count > 0
            else "NO_SUCCESSFUL_COMPARISONS"
        ),
        "comparison_record_count": safe_int(
            len(comparison_dataframe)
        ),
        "successful_comparison_count": candidate_count,
        "target_count": target_count,
        "forecast_horizon_count": horizon_count,
        "comparison_method": (
            "MAE and RMSE ranking within each target/horizon group."
        ),
        "ranking_metric": "MAE",
        "secondary_metric": "RMSE",
        "results_output": str(comparison_output_path),
        "production_approval_granted": False,
        "limitations": [
            "A lower MAE does not guarantee better performance in every operating condition.",
            "Different models must be evaluated on comparable observations.",
            "Chronological evaluation must be preserved.",
            "Model selection should consider business costs and forecast stability.",
            "This comparison does not assess prediction intervals or calibration.",
            "No exact ferry capacity, passenger queue length, or boarding count is inferred.",
        ],
    }

    summary_output_path = (
        REPORT_DIR / "forecast_model_comparison_summary.json"
    )

    save_json_report(
        summary,
        summary_output_path,
    )

    LOGGER.info(
        "Model comparison completed across %s target(s) "
        "and %s horizon(s).",
        target_count,
        horizon_count,
    )

    return comparison_dataframe, summary


# END OF PART 18
# ============================================================
# ============================================================
# PART 19: FORECAST MONITORING, ERROR DRIFT AND ALERT GENERATION
# ============================================================

def monitor_forecast_performance(
    predictions_dataframe: pd.DataFrame,
    target_column: str = "Target_Column",
    model_column: str = "Model_Name",
    timestamp_column: str = "Timestamp_UTC",
    actual_column: str = "Actual",
    predicted_column: str = "Predicted",
    horizon_column: str = "Forecast_Horizon_Minutes",
    minimum_group_size: int = 20,
    error_increase_threshold: float = 25.0,
    high_mae_threshold: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Monitor observed forecasting errors and generate review alerts.

    Monitoring capabilities:
        1. Validate required prediction columns.
        2. Parse timestamps and numerical values.
        3. Exclude invalid observations from metric calculations.
        4. Calculate errors for chronological monitoring windows.
        5. Compare recent errors with a preceding reference window.
        6. Flag increases in MAE.
        7. Optionally flag MAE exceeding a configured threshold.
        8. Generate an auditable alert register.
        9. Save monitoring summaries and detailed results.

    Important:
        - This function monitors predictions after actual outcomes
          become available.
        - It does not detect feature drift because feature values
          are not required by this interface.
        - An error increase is an alert for investigation, not proof
          that the model has statistically drifted.
        - Monitoring requires predictions from comparable horizons
          and models.
    """

    LOGGER.info(
        "Starting forecast-performance monitoring."
    )

    if minimum_group_size < 1:
        raise ValueError(
            "minimum_group_size must be at least 1."
        )

    if error_increase_threshold < 0:
        raise ValueError(
            "error_increase_threshold cannot be negative."
        )

    if (
        high_mae_threshold is not None
        and high_mae_threshold < 0
    ):
        raise ValueError(
            "high_mae_threshold cannot be negative."
        )

    required_columns = [
        model_column,
        timestamp_column,
        actual_column,
        predicted_column,
        horizon_column,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in predictions_dataframe.columns
    ]

    if missing_columns:
        raise KeyError(
            "Forecast monitoring requires these columns: "
            f"{missing_columns}"
        )

    working = predictions_dataframe.copy()

    # --------------------------------------------------------
    # 1. Normalize timestamps and prediction values
    # --------------------------------------------------------
    working[timestamp_column] = pd.to_datetime(
        working[timestamp_column],
        errors="coerce",
        utc=True,
    )

    working[actual_column] = pd.to_numeric(
        working[actual_column],
        errors="coerce",
    )

    working[predicted_column] = pd.to_numeric(
        working[predicted_column],
        errors="coerce",
    )

    working[horizon_column] = pd.to_numeric(
        working[horizon_column],
        errors="coerce",
    )

    if target_column not in working.columns:
        working[target_column] = "UNSPECIFIED_TARGET"

    working[actual_column] = working[
        actual_column
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    working[predicted_column] = working[
        predicted_column
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    valid_mask = (
        working[timestamp_column].notna()
        & working[actual_column].notna()
        & working[predicted_column].notna()
        & working[horizon_column].notna()
        & (working[horizon_column] > 0)
        & working[model_column].notna()
    )

    invalid_record_count = safe_int(
        (~valid_mask).sum()
    )

    valid_data = working.loc[
        valid_mask
    ].copy()

    if valid_data.empty:
        empty_monitoring = pd.DataFrame()

        summary = {
            "generated_at_utc": get_utc_timestamp(),
            "status": "NO_VALID_PREDICTIONS",
            "input_record_count": safe_int(
                len(predictions_dataframe)
            ),
            "valid_record_count": 0,
            "invalid_record_count": invalid_record_count,
            "alert_count": 0,
            "interpretation": (
                "Monitoring could not proceed because no valid "
                "prediction/actual pairs were available."
            ),
        }

        save_dataframe_csv(
            empty_monitoring,
            OUTPUT_DIR / "forecast_performance_monitoring.csv",
        )

        save_dataframe_csv(
            empty_monitoring,
            OUTPUT_DIR / "forecast_monitoring_alerts.csv",
        )

        save_json_report(
            summary,
            REPORT_DIR / "forecast_monitoring_summary.json",
        )

        return (
            empty_monitoring,
            empty_monitoring,
            summary,
        )

    # --------------------------------------------------------
    # 2. Calculate absolute and signed errors
    # --------------------------------------------------------
    valid_data["Signed_Error"] = (
        valid_data[actual_column]
        - valid_data[predicted_column]
    )

    valid_data["Absolute_Error"] = (
        valid_data["Signed_Error"].abs()
    )

    valid_data["Squared_Error"] = (
        valid_data["Signed_Error"].pow(2)
    )

    valid_data = valid_data.sort_values(
        timestamp_column,
        kind="stable",
    )

    # --------------------------------------------------------
    # 3. Group comparable forecasts
    # --------------------------------------------------------
    grouping_columns = [
        target_column,
        model_column,
        horizon_column,
    ]

    monitoring_records: list[dict[str, Any]] = []
    alert_records: list[dict[str, Any]] = []

    for group_key, group in valid_data.groupby(
        grouping_columns,
        dropna=False,
        sort=True,
    ):
        group_target, group_model, group_horizon = group_key

        group = group.sort_values(
            timestamp_column,
            kind="stable",
        ).reset_index(
            drop=True
        )

        observation_count = safe_int(
            len(group)
        )

        # Divide the chronological group into an earlier reference
        # window and a later monitoring window.
        #
        # No random shuffle is performed.
        split_position = observation_count // 2

        reference_window = group.iloc[
            :split_position
        ]

        recent_window = group.iloc[
            split_position:
        ]

        reference_mae = (
            float(reference_window["Absolute_Error"].mean())
            if not reference_window.empty
            else None
        )

        recent_mae = (
            float(recent_window["Absolute_Error"].mean())
            if not recent_window.empty
            else None
        )

        reference_rmse = (
            float(
                np.sqrt(
                    reference_window["Squared_Error"].mean()
                )
            )
            if not reference_window.empty
            else None
        )

        recent_rmse = (
            float(
                np.sqrt(
                    recent_window["Squared_Error"].mean()
                )
            )
            if not recent_window.empty
            else None
        )

        if (
            reference_mae is not None
            and reference_mae > 0
            and recent_mae is not None
        ):
            mae_change_percentage = (
                (recent_mae - reference_mae)
                / reference_mae
                * 100.0
            )

        elif (
            reference_mae == 0
            and recent_mae is not None
            and recent_mae > 0
        ):
            mae_change_percentage = None

        else:
            mae_change_percentage = None

        sufficient_group_size = (
            observation_count >= minimum_group_size
        )

        alert_severity = "INFO"
        alert_reason = "NO_ALERT"

        if not sufficient_group_size:
            alert_severity = "REVIEW"
            alert_reason = "INSUFFICIENT_MONITORING_OBSERVATIONS"

        elif (
            high_mae_threshold is not None
            and recent_mae is not None
            and recent_mae > high_mae_threshold
        ):
            alert_severity = "HIGH"
            alert_reason = "RECENT_MAE_EXCEEDS_CONFIGURED_THRESHOLD"

        elif (
            mae_change_percentage is not None
            and mae_change_percentage > error_increase_threshold
        ):
            alert_severity = "MEDIUM"
            alert_reason = "RECENT_MAE_INCREASED"

        elif (
            reference_mae == 0
            and recent_mae is not None
            and recent_mae > 0
        ):
            alert_severity = "REVIEW"
            alert_reason = "ERROR_INCREASE_FROM_ZERO_BASELINE"

        # ----------------------------------------------------
        # 4. Record group-level monitoring statistics
        # ----------------------------------------------------
        monitoring_records.append(
            {
                "target_column": str(group_target),
                "model_name": str(group_model),
                "forecast_horizon_minutes": safe_int(
                    group_horizon
                ),
                "observation_count": observation_count,
                "reference_observation_count": safe_int(
                    len(reference_window)
                ),
                "recent_observation_count": safe_int(
                    len(recent_window)
                ),
                "reference_start_utc": (
                    str(reference_window[timestamp_column].min())
                    if not reference_window.empty
                    else None
                ),
                "reference_end_utc": (
                    str(reference_window[timestamp_column].max())
                    if not reference_window.empty
                    else None
                ),
                "recent_start_utc": (
                    str(recent_window[timestamp_column].min())
                    if not recent_window.empty
                    else None
                ),
                "recent_end_utc": (
                    str(recent_window[timestamp_column].max())
                    if not recent_window.empty
                    else None
                ),
                "reference_mae": reference_mae,
                "recent_mae": recent_mae,
                "reference_rmse": reference_rmse,
                "recent_rmse": recent_rmse,
                "mae_change_percentage": (
                    mae_change_percentage
                ),
                "minimum_group_size_met": (
                    sufficient_group_size
                ),
                "monitoring_alert_severity": alert_severity,
                "monitoring_alert_reason": alert_reason,
            }
        )

        # ----------------------------------------------------
        # 5. Create explicit alerts for non-normal conditions
        # ----------------------------------------------------
        if alert_reason != "NO_ALERT":
            alert_records.append(
                {
                    "generated_at_utc": get_utc_timestamp(),
                    "target_column": str(group_target),
                    "model_name": str(group_model),
                    "forecast_horizon_minutes": safe_int(
                        group_horizon
                    ),
                    "severity": alert_severity,
                    "alert_reason": alert_reason,
                    "observation_count": observation_count,
                    "reference_mae": reference_mae,
                    "recent_mae": recent_mae,
                    "mae_change_percentage": (
                        mae_change_percentage
                    ),
                    "recommended_action": (
                        "Review monitoring coverage, data quality, "
                        "forecast horizon, and recent model errors. "
                        "Re-evaluate the model using chronological "
                        "validation before making deployment changes."
                    ),
                }
            )

    monitoring_dataframe = pd.DataFrame(
        monitoring_records
    )

    alerts_dataframe = pd.DataFrame(
        alert_records
    )

    # --------------------------------------------------------
    # 6. Persist monitoring outputs
    # --------------------------------------------------------
    monitoring_output_path = (
        OUTPUT_DIR / "forecast_performance_monitoring.csv"
    )

    alerts_output_path = (
        OUTPUT_DIR / "forecast_monitoring_alerts.csv"
    )

    save_dataframe_csv(
        monitoring_dataframe,
        monitoring_output_path,
    )

    save_dataframe_csv(
        alerts_dataframe,
        alerts_output_path,
    )

    # --------------------------------------------------------
    # 7. Build the monitoring run summary
    # --------------------------------------------------------
    severity_counts = (
        alerts_dataframe["severity"]
        .value_counts()
        .to_dict()
        if not alerts_dataframe.empty
        else {}
    )

    summary: dict[str, Any] = {
        "generated_at_utc": get_utc_timestamp(),
        "status": "MONITORING_COMPLETED",
        "input_record_count": safe_int(
            len(predictions_dataframe)
        ),
        "valid_record_count": safe_int(
            len(valid_data)
        ),
        "invalid_record_count": invalid_record_count,
        "monitored_group_count": safe_int(
            len(monitoring_dataframe)
        ),
        "alert_count": safe_int(
            len(alerts_dataframe)
        ),
        "alert_severity_counts": severity_counts,
        "minimum_group_size": minimum_group_size,
        "error_increase_threshold_percentage": (
            error_increase_threshold
        ),
        "high_mae_threshold": high_mae_threshold,
        "monitoring_output": str(
            monitoring_output_path
        ),
        "alerts_output": str(
            alerts_output_path
        ),
        "limitations": [
            "Monitoring requires actual outcomes to become available.",
            "Error increases do not independently prove statistical drift.",
            "Small monitoring groups can produce unstable error estimates.",
            "Forecast groups must be comparable in target and horizon.",
            "This function does not automatically retrain or deploy models.",
            "Alerts require human review before operational action.",
        ],
    }

    summary_path = (
        REPORT_DIR / "forecast_monitoring_summary.json"
    )

    save_json_report(
        summary,
        summary_path,
    )

    LOGGER.info(
        "Forecast monitoring completed. Groups: %s; alerts: %s.",
        safe_int(len(monitoring_dataframe)),
        safe_int(len(alerts_dataframe)),
    )

    return (
        monitoring_dataframe,
        alerts_dataframe,
        summary,
    )


# END OF PART 19
# ============================================================

# ============================================================
# PART 20: FINAL PIPELINE INTEGRATION AND EXECUTION
# ============================================================

# IMPORTANT:
# Before running this file, go to PART 9 and DELETE the old
# main guard located there:
#
# if __name__ == "__main__":
#     exit_code = main()
#     raise SystemExit(exit_code)
#
# Keep the main guard ONLY at the bottom of this file.
# This ensures all functions from Parts 1–19 are defined
# before the integrated pipeline starts executing.


def main() -> int:
    """
    Execute the complete dataset-analysis and forecasting
    preparation pipeline in a controlled, auditable sequence.

    Pipeline stages:
        1. Initialize directories and logging.
        2. Load and validate the source dataset.
        3. Analyze structure, data types, and duplicates.
        4. Analyze timestamps and interval consistency.
        5. Analyze numerical values and demand distributions.
        6. Analyze temporal patterns and correlations.
        7. Identify outliers and data-quality issues.
        8. Generate visualizations and executive reports.
        9. Run advanced data-quality checks.
       10. Assess time-series forecasting readiness.
       11. Prepare regular forecasting series.
       12. Evaluate forecasting-data quality.
       13. Evaluate naive forecasting baselines.
       14. Validate and compare generated predictions.
       15. Monitor forecast performance.
       16. Validate generated artifacts.
       17. Generate final inventory and execution reports.

    This pipeline does not claim exact ferry capacity,
    passenger counts, queue length, or failure prediction.
    Forecasting targets are limited to the ticket-demand
    variables available in the source dataset.
    """

    start_time = time.perf_counter()

    create_output_directories()
    configure_logging()

    LOGGER.info("=" * 75)
    LOGGER.info("FERRY DEMAND ANALYSIS PIPELINE STARTED")
    LOGGER.info("Report version: %s", REPORT_VERSION)
    LOGGER.info("=" * 75)

    execution_status = "FAILED"
    failure_reason = None

    pipeline_results: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "started_at_utc": get_utc_timestamp(),
        "source_dataset": str(DATASET_PATH),
        "pipeline_stages": {},
    }

    try:
        # ----------------------------------------------------
        # STAGE 1: LOAD AND VALIDATE THE DATASET
        # ----------------------------------------------------
        LOGGER.info("STAGE 1: Loading the source dataset.")

        dataframe = load_dataset()

        if dataframe.empty:
            raise ValueError(
                "The source dataset contains no rows."
            )

        structural_report = validate_dataset_structure(
            dataframe
        )

        log_analysis_dataframe_summary(
            dataframe,
            "SOURCE DATASET",
        )

        pipeline_results["dataset_shape"] = {
            "rows": safe_int(len(dataframe)),
            "columns": safe_int(len(dataframe.columns)),
        }

        pipeline_results["pipeline_stages"][
            "dataset_loading"
        ] = "COMPLETED"

        # ----------------------------------------------------
        # STAGE 2: STRUCTURAL AND DUPLICATE ANALYSIS
        # ----------------------------------------------------
        LOGGER.info(
            "STAGE 2: Analyzing data types and duplicates."
        )

        analyze_data_types(dataframe)

        duplicate_report = analyze_duplicates(
            dataframe
        )

        analyze_non_numeric_columns(
            dataframe
        )

        pipeline_results["pipeline_stages"][
            "structural_analysis"
        ] = "COMPLETED"

        # ----------------------------------------------------
        # STAGE 3: TIMESTAMP ANALYSIS
        # ----------------------------------------------------
        LOGGER.info(
            "STAGE 3: Preparing and analyzing timestamps."
        )

        timestamp_dataframe = prepare_timestamp_analysis(
            dataframe
        )

        (
            timestamp_report,
            interval_gap_report,
        ) = analyze_timestamp_intervals(
            timestamp_dataframe
        )

        pipeline_results["pipeline_stages"][
            "timestamp_analysis"
        ] = "COMPLETED"

        # ----------------------------------------------------
        # STAGE 4: NUMERICAL AND DEMAND ANALYSIS
        # ----------------------------------------------------
        LOGGER.info(
            "STAGE 4: Analyzing numerical and demand values."
        )

        analyze_numeric_columns(
            dataframe
        )

        (
            demand_summary,
            demand_issue_dataframe,
        ) = analyze_demand_values(
            dataframe
        )

        analyze_demand_relationship(
            dataframe
        )

        analyze_numeric_correlations(
            dataframe
        )

        analyze_demand_distributions(
            dataframe
        )

        analyze_high_demand_observations(
            dataframe
        )

        pipeline_results["pipeline_stages"][
            "numerical_and_demand_analysis"
        ] = "COMPLETED"

        # ----------------------------------------------------
        # STAGE 5: TEMPORAL PATTERNS AND EXTREME VALUES
        # ----------------------------------------------------
        LOGGER.info(
            "STAGE 5: Analyzing temporal patterns and extremes."
        )

        time_pattern_summary = analyze_time_patterns(
            timestamp_dataframe
        )

        analyze_daily_demand_extremes(
            timestamp_dataframe
        )

        analyze_outliers(
            dataframe
        )

        pipeline_results["pipeline_stages"][
            "temporal_analysis"
        ] = "COMPLETED"

        # ----------------------------------------------------
        # STAGE 6: DATA-QUALITY ISSUE REGISTER
        # ----------------------------------------------------
        LOGGER.info(
            "STAGE 6: Building the data-quality issue register."
        )

        issue_register = build_data_quality_issue_register(
            dataframe,
            structural_report,
            timestamp_report,
            demand_issue_dataframe,
            duplicate_report,
        )

        executive_summary = build_executive_summary(
            dataframe,
            structural_report,
            duplicate_report,
            timestamp_report,
            demand_summary,
            time_pattern_summary,
            demand_summary,
            issue_register,
        )

        pipeline_results["pipeline_stages"][
            "quality_register_and_summary"
        ] = "COMPLETED"

        # ----------------------------------------------------
        # STAGE 7: VISUALIZATIONS AND ANALYSIS REPORTS
        # ----------------------------------------------------
        LOGGER.info(
            "STAGE 7: Generating visualizations and reports."
        )

        generate_dataset_visualizations(
            dataframe,
            timestamp_dataframe,
        )

        build_comprehensive_analysis_report(
            dataframe,
            structural_report,
            duplicate_report,
            timestamp_report,
            demand_summary,
            time_pattern_summary,
            issue_register,
            executive_summary,
        )

        pipeline_results["pipeline_stages"][
            "visualizations_and_reports"
        ] = "COMPLETED"

        # ----------------------------------------------------
        # STAGE 8: ADVANCED DATA-QUALITY CHECKS
        # ----------------------------------------------------
        LOGGER.info(
            "STAGE 8: Running advanced data-quality checks."
        )

        (
            advanced_quality_dataframe,
            advanced_quality_summary,
        ) = run_advanced_data_quality_checks(
            dataframe,
            timestamp_dataframe,
        )

        pipeline_results["advanced_quality_summary"] = (
            advanced_quality_summary
        )

        pipeline_results["pipeline_stages"][
            "advanced_data_quality"
        ] = "COMPLETED"

        # ----------------------------------------------------
        # STAGE 9: FORECASTING READINESS ASSESSMENT
        # ----------------------------------------------------
        LOGGER.info(
            "STAGE 9: Assessing time-series readiness."
        )

        (
            readiness_issues,
            readiness_summary,
        ) = assess_time_series_readiness(
            dataframe,
            timestamp_dataframe,
        )

        pipeline_results["forecasting_readiness"] = (
            readiness_summary
        )

        pipeline_results["pipeline_stages"][
            "forecasting_readiness"
        ] = "COMPLETED"

        # ----------------------------------------------------
        # STAGE 10: PREPARE FORECASTING TARGETS
        # ----------------------------------------------------
        LOGGER.info(
            "STAGE 10: Preparing sales and redemption series."
        )

        (
            prepared_series_by_target,
            target_preparation_summary,
        ) = prepare_all_forecasting_targets(
            dataframe=dataframe,
            timestamp_dataframe=timestamp_dataframe,
            target_columns=[
                "Sales Count",
                "Redemption Count",
            ],
            frequency="15min",
            aggregation_method="sum",
            minimum_coverage=0.80,
        )

        pipeline_results["target_preparation_summary"] = (
            target_preparation_summary
        )

        pipeline_results["pipeline_stages"][
            "forecast_series_preparation"
        ] = "COMPLETED"

        # ----------------------------------------------------
        # STAGE 11: FORECASTING QUALITY AND BASELINES
        # ----------------------------------------------------
        LOGGER.info(
            "STAGE 11: Evaluating forecasting quality and baselines."
        )

        all_horizon_predictions: list[pd.DataFrame] = []
        forecasting_quality_results: dict[str, Any] = {}
        baseline_results: dict[str, Any] = {}

        for target_column, prepared_series in (
            prepared_series_by_target.items()
        ):
            LOGGER.info(
                "Processing forecasting target: %s",
                target_column,
            )

            try:
                (
                    forecasting_quality_dataframe,
                    forecasting_quality_summary,
                ) = evaluate_forecasting_data_quality(
                    prepared_series=prepared_series,
                    target_column=target_column,
                )

                forecasting_quality_results[
                    target_column
                ] = forecasting_quality_summary

                # A target must pass the basic readiness checks
                # before baseline evaluation is attempted.
                quality_status = str(
                    forecasting_quality_summary.get(
                        "status",
                        "",
                    )
                ).upper()

                if quality_status not in {
                    "PASS",
                    "READY",
                    "SUITABLE",
                    "ACCEPTABLE",
                    "PASSED",
                }:
                    LOGGER.warning(
                        "Forecasting quality gate did not report "
                        "a recognized passing status for %s. "
                        "Detailed quality report: %s",
                        target_column,
                        quality_status or "UNSPECIFIED",
                    )

                # Run the existing chronological baseline
                # evaluation regardless of the label used by
                # the quality summary. The baseline evaluator
                # performs its own validation and error handling.
                try:
                    baseline_metrics = (
                        evaluate_naive_forecasting_baselines(
                            prepared_series=prepared_series,
                            target_column=target_column,
                            evaluation_fraction=0.20,
                            seasonal_period=96,
                        )
                    )

                    baseline_results[
                        target_column
                    ] = safe_json_value(
                        baseline_metrics
                    )

                except Exception as baseline_error:
                    LOGGER.exception(
                        "Baseline evaluation failed for %s: %s",
                        target_column,
                        baseline_error,
                    )

                    baseline_results[
                        target_column
                    ] = {
                        "status": "FAILED",
                        "error": str(baseline_error),
                    }

                # Evaluate the configured 15, 30, 60, and
                # 120-minute forecast horizons.
                try:
                    horizon_predictions = (
                        evaluate_baselines_by_forecast_horizon(
                            prepared_series=prepared_series,
                            target_column=target_column,
                            frequency="15min",
                            forecast_horizons_minutes=[
                                15,
                                30,
                                60,
                                120,
                            ],
                            evaluation_fraction=0.20,
                            seasonal_period=96,
                        )
                    )

                    if (
                        isinstance(
                            horizon_predictions,
                            pd.DataFrame,
                        )
                        and not horizon_predictions.empty
                    ):
                        horizon_predictions = (
                            horizon_predictions.copy()
                        )

                        if "Target_Column" not in (
                            horizon_predictions.columns
                        ):
                            horizon_predictions[
                                "Target_Column"
                            ] = target_column

                        all_horizon_predictions.append(
                            horizon_predictions
                        )

                    else:
                        LOGGER.warning(
                            "No horizon predictions were produced "
                            "for target %s.",
                            target_column,
                        )

                except Exception as horizon_error:
                    LOGGER.exception(
                        "Horizon evaluation failed for %s: %s",
                        target_column,
                        horizon_error,
                    )

            except Exception as target_error:
                LOGGER.exception(
                    "Forecasting target processing failed for %s: %s",
                    target_column,
                    target_error,
                )

                forecasting_quality_results[
                    target_column
                ] = {
                    "status": "FAILED",
                    "error": str(target_error),
                }

        pipeline_results["forecasting_quality_results"] = (
            forecasting_quality_results
        )

        pipeline_results["baseline_results"] = (
            baseline_results
        )

        pipeline_results["pipeline_stages"][
            "forecast_quality_and_baselines"
        ] = "COMPLETED"

        # ----------------------------------------------------
        # STAGE 12: VALIDATE AND COMPARE PREDICTIONS
        # ----------------------------------------------------
        LOGGER.info(
            "STAGE 12: Validating and comparing predictions."
        )

        validated_prediction_frames: list[pd.DataFrame] = []

        if all_horizon_predictions:
            combined_predictions = pd.concat(
                all_horizon_predictions,
                ignore_index=True,
                sort=False,
            )

            # Validate each target independently so records from
            # different demand targets cannot be incorrectly
            # classified as duplicates of one another.
            for target_name, target_predictions in (
                combined_predictions.groupby(
                    "Target_Column",
                    dropna=False,
                    sort=False,
                )
            ):
                try:
                    validated_target_predictions = (
                        validate_forecast_predictions(
                            predictions_dataframe=(
                                target_predictions
                            ),
                            timestamp_column="Timestamp_UTC",
                            actual_column="Actual",
                            predicted_column="Predicted",
                            model_column="Model_Name",
                            horizon_column=(
                                "Forecast_Horizon_Minutes"
                            ),
                        )
                    )

                    if (
                        isinstance(
                            validated_target_predictions,
                            pd.DataFrame,
                        )
                        and not validated_target_predictions.empty
                    ):
                        validated_target_predictions = (
                            validated_target_predictions.copy()
                        )

                        validated_target_predictions[
                            "Target_Column"
                        ] = target_name

                        validated_prediction_frames.append(
                            validated_target_predictions
                        )

                except Exception as validation_error:
                    LOGGER.exception(
                        "Prediction validation failed for %s: %s",
                        target_name,
                        validation_error,
                    )

            if validated_prediction_frames:
                combined_validated_predictions = pd.concat(
                    validated_prediction_frames,
                    ignore_index=True,
                    sort=False,
                )

                save_dataframe_csv(
                    combined_validated_predictions,
                    OUTPUT_DIR
                    / "all_validated_forecast_predictions.csv",
                )

                comparison_results = (
                    compare_forecasting_models(
                        validated_predictions=(
                            combined_validated_predictions
                        ),
                        actual_column="Actual",
                        predicted_column="Predicted",
                        model_column="Model_Name",
                        horizon_column=(
                            "Forecast_Horizon_Minutes"
                        ),
                        target_column="Target_Column",
                        validation_status_column=(
                            "Validation_Status"
                        ),
                    )
                )

                pipeline_results[
                    "forecast_comparison_status"
                ] = "COMPLETED"

                # ------------------------------------------------
                # STAGE 13: MONITOR FORECAST PERFORMANCE
                # ------------------------------------------------
                LOGGER.info(
                    "STAGE 13: Monitoring forecast performance."
                )

                (
                    monitoring_dataframe,
                    monitoring_alerts,
                    monitoring_summary,
                ) = monitor_forecast_performance(
                    predictions_dataframe=(
                        combined_validated_predictions
                    ),
                    target_column="Target_Column",
                    model_column="Model_Name",
                    timestamp_column="Timestamp_UTC",
                    actual_column="Actual",
                    predicted_column="Predicted",
                    horizon_column=(
                        "Forecast_Horizon_Minutes"
                    ),
                    minimum_group_size=20,
                    error_increase_threshold=25.0,
                )

                pipeline_results[
                    "forecast_monitoring_summary"
                ] = monitoring_summary

            else:
                LOGGER.warning(
                    "Prediction validation produced no usable "
                    "prediction records."
                )

                pipeline_results[
                    "forecast_comparison_status"
                ] = "NO_VALIDATED_PREDICTIONS"

        else:
            LOGGER.warning(
                "No forecast-horizon predictions are available "
                "for validation, comparison, or monitoring."
            )

            pipeline_results[
                "forecast_comparison_status"
            ] = "NO_HORIZON_PREDICTIONS"

        pipeline_results["pipeline_stages"][
            "prediction_validation_comparison_monitoring"
        ] = "COMPLETED"

        # ----------------------------------------------------
        # STAGE 14: ARTIFACT VALIDATION AND INVENTORY
        # ----------------------------------------------------
        LOGGER.info(
            "STAGE 14: Validating generated output artifacts."
        )

        artifact_validation = validate_generated_artifacts()

        artifact_inventory = generate_artifact_inventory()

        artifact_summary = summarize_analysis_outputs(
            artifact_inventory,
            artifact_validation,
        )

        pipeline_results["artifact_summary"] = (
            artifact_summary
        )

        pipeline_results["pipeline_stages"][
            "artifact_validation"
        ] = "COMPLETED"

        # ----------------------------------------------------
        # STAGE 15: FINAL STATUS AND RUN MANIFEST
        # ----------------------------------------------------
        execution_status = "COMPLETED"

        elapsed_seconds = (
            time.perf_counter() - start_time
        )

        pipeline_results["completed_at_utc"] = (
            get_utc_timestamp()
        )

        pipeline_results["execution_status"] = (
            execution_status
        )

        pipeline_results["execution_duration_seconds"] = (
            round(elapsed_seconds, 3)
        )

        pipeline_results["failure_reason"] = None

        save_json_report(
            pipeline_results,
            REPORT_DIR / "complete_pipeline_summary.json",
        )

        create_analysis_run_manifest(
            pipeline_results
        )

        write_analysis_status(
            status=execution_status,
            message=(
                "Dataset analysis and forecasting preparation "
                "pipeline completed."
            ),
        )

        LOGGER.info("=" * 75)
        LOGGER.info(
            "PIPELINE COMPLETED SUCCESSFULLY IN %.2f SECONDS",
            elapsed_seconds,
        )
        LOGGER.info(
            "Reports directory: %s",
            REPORT_DIR,
        )
        LOGGER.info(
            "Outputs directory: %s",
            OUTPUT_DIR,
        )
        LOGGER.info("=" * 75)

        return 0

    except KeyboardInterrupt:
        execution_status = "INTERRUPTED"
        failure_reason = "Execution interrupted by the user."

        LOGGER.warning(
            "Pipeline execution was interrupted."
        )

        return 130

    except Exception as error:
        execution_status = "FAILED"
        failure_reason = str(error)

        LOGGER.exception(
            "The analysis pipeline failed: %s",
            error,
        )

        return 1

    finally:
        elapsed_seconds = (
            time.perf_counter() - start_time
        )

        pipeline_results["final_status"] = (
            execution_status
        )

        pipeline_results["failure_reason"] = (
            failure_reason
        )

        pipeline_results["finished_at_utc"] = (
            get_utc_timestamp()
        )

        pipeline_results["total_elapsed_seconds"] = (
            round(elapsed_seconds, 3)
        )

        try:
            save_json_report(
                pipeline_results,
                REPORT_DIR / "pipeline_execution_status.json",
            )

            write_analysis_status(
                status=execution_status,
                message=(
                    failure_reason
                    if failure_reason
                    else (
                        "Pipeline execution finished with status: "
                        f"{execution_status}"
                    )
                ),
            )

        except Exception as reporting_error:
            LOGGER.error(
                "Could not save the final execution status: %s",
                reporting_error,
            )


# ============================================================
# FINAL ENTRY POINT — KEEP THIS GUARD AT THE BOTTOM ONLY
# ============================================================

if __name__ == "__main__":
    exit_code = main()
    raise SystemExit(exit_code)

# END OF PART 20
# ============================================================
