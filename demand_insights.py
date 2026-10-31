
"""
===============================================================================
PROJECT:
    Short-Term Ferry Ticket Demand Forecasting & Predictive Decision
    Support System

MODULE:
    demand_insights.py

PART:
    1 — Core Architecture, Data Loading, Validation and Demand Analytics

PURPOSE:
    This module analyzes historical ferry ticket sales and redemptions,
    generates descriptive demand insights, identifies time-based patterns,
    summarizes data quality, and exports machine-readable reports.

SOURCE DATA:
    data/Toronto Island Ferry Tickets (1).csv

EXPECTED SOURCE COLUMNS:
    _id
    Timestamp
    Redemption Count
    Sales Count

IMPORTANT:
    This module analyzes ticket counts. It does not directly measure vessel
    capacity, passenger boarding, queue length, or confirmed future failures.

OUTPUT DIRECTORIES:
    outputs/demand_insights/
    outputs/reports/demand_insights/

DESIGN PRINCIPLES:
    1. Robust input validation.
    2. Reusable functions.
    3. Consistent output schemas.
    4. Defensive error handling.
    5. Reproducible calculations.
    6. No modification of the original dataset.
    7. Transparent treatment of missing and invalid observations.
===============================================================================
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import platform
import statistics
import sys
import time
import traceback

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd


# =============================================================================
# SECTION 1 — MODULE METADATA
# =============================================================================

PROJECT_NAME = "Short-Term Ferry Ticket Demand Forecasting"

PROJECT_DESCRIPTION = (
    "Historical ticket demand analytics and predictive decision support."
)

MODULE_NAME = "demand_insights"

MODULE_VERSION = "1.0.0"

MODULE_PART = 1

MODULE_STATUS = "Core demand analytics"

EXPECTED_TIMESTAMP_COLUMN = "Timestamp"

EXPECTED_SALES_COLUMN = "Sales Count"

EXPECTED_REDEMPTION_COLUMN = "Redemption Count"

EXPECTED_ID_COLUMN = "_id"

DEMAND_COLUMNS = [
    EXPECTED_SALES_COLUMN,
    EXPECTED_REDEMPTION_COLUMN,
]

SUPPORTED_HORIZONS_MINUTES = [15, 30, 60, 120]

DEFAULT_ENCODING = "utf-8"

DEFAULT_CSV_CHUNK_SIZE = 50_000

DEFAULT_MINIMUM_VALID_ROWS = 1

DEFAULT_ROUNDING_DIGITS = 4


# =============================================================================
# SECTION 2 — PATH CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

OUTPUT_DIR = BASE_DIR / "outputs"

REPORT_DIR = OUTPUT_DIR / "reports"

DEMAND_OUTPUT_DIR = OUTPUT_DIR / "demand_insights"

DEMAND_REPORT_DIR = REPORT_DIR / "demand_insights"

LOG_DIR = OUTPUT_DIR / "logs"

SOURCE_DATA_PATH = DATA_DIR / "Toronto Island Ferry Tickets (1).csv"

OBSERVED_TIME_SERIES_PATH = (
    OUTPUT_DIR / "forecasts" / "observed_time_series.csv"
)

ENGINEERED_FEATURES_PATH = (
    OUTPUT_DIR / "forecasts" / "engineered_features.csv"
)

FORECAST_PREDICTIONS_PATH = (
    OUTPUT_DIR / "forecasts" / "model_predictions.csv"
)

MODEL_COMPARISON_PATH = (
    REPORT_DIR / "model_comparison.csv"
)

MODEL_TRAINING_SUMMARY_PATH = (
    REPORT_DIR / "model_training_summary.json"
)

TIMESTAMP_GAP_REPORT_PATH = (
    REPORT_DIR / "timestamp_gap_report.json"
)

DATASET_VALIDATION_REPORT_PATH = (
    REPORT_DIR / "dataset_validation_report.json"
)

DEMAND_SUMMARY_PATH = (
    DEMAND_REPORT_DIR / "demand_summary.json"
)

DATA_QUALITY_REPORT_PATH = (
    DEMAND_REPORT_DIR / "data_quality_report.json"
)

TIME_PATTERN_REPORT_PATH = (
    DEMAND_REPORT_DIR / "time_pattern_summary.json"
)

DEMAND_INSIGHTS_CSV_PATH = (
    DEMAND_OUTPUT_DIR / "demand_insights.csv"
)

DEMAND_BY_HOUR_CSV_PATH = (
    DEMAND_OUTPUT_DIR / "demand_by_hour.csv"
)

DEMAND_BY_DAY_CSV_PATH = (
    DEMAND_OUTPUT_DIR / "demand_by_day.csv"
)

DEMAND_BY_MONTH_CSV_PATH = (
    DEMAND_OUTPUT_DIR / "demand_by_month.csv"
)

DEMAND_BY_DATE_CSV_PATH = (
    DEMAND_OUTPUT_DIR / "demand_by_date.csv"
)

DEMAND_CORRELATION_CSV_PATH = (
    DEMAND_OUTPUT_DIR / "demand_correlation.csv"
)

DEMAND_MISSING_VALUES_CSV_PATH = (
    DEMAND_OUTPUT_DIR / "missing_values.csv"
)

DEMAND_DUPLICATE_REPORT_PATH = (
    DEMAND_REPORT_DIR / "duplicate_summary.json"
)

DEMAND_RUN_SUMMARY_PATH = (
    DEMAND_REPORT_DIR / "run_summary.json"
)


# =============================================================================
# SECTION 3 — DIRECTORY INITIALIZATION
# =============================================================================

REQUIRED_DIRECTORIES = [
    OUTPUT_DIR,
    REPORT_DIR,
    DEMAND_OUTPUT_DIR,
    DEMAND_REPORT_DIR,
    LOG_DIR,
]


def ensure_directories() -> None:
    """
    Create the directories required by this module.

    Existing files and directories are preserved.
    """

    for directory in REQUIRED_DIRECTORIES:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# =============================================================================
# SECTION 4 — LOGGING CONFIGURATION
# =============================================================================

LOGGER_NAME = f"{PROJECT_NAME}.{MODULE_NAME}"

LOGGER = logging.getLogger(LOGGER_NAME)

LOGGER.setLevel(logging.INFO)

LOGGER.propagate = False

LOG_FILE_PATH = LOG_DIR / "demand_insights.log"


def configure_logging(
    level: int = logging.INFO,
    force_reconfigure: bool = False,
) -> logging.Logger:
    """
    Configure console and file logging.

    Parameters
    ----------
    level:
        Standard Python logging level.

    force_reconfigure:
        If True, replace existing handlers.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    ensure_directories()

    if force_reconfigure:
        for handler in list(LOGGER.handlers):
            LOGGER.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    if LOGGER.handlers:
        LOGGER.setLevel(level)
        return LOGGER

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
        LOG_FILE_PATH,
        encoding="utf-8",
    )

    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)

    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(console_handler)

    LOGGER.setLevel(level)

    return LOGGER


# =============================================================================
# SECTION 5 — EXCEPTIONS
# =============================================================================

class DemandInsightsError(Exception):
    """Base exception for demand analytics errors."""


class SourceDataNotFoundError(DemandInsightsError):
    """Raised when the configured source dataset is unavailable."""


class SourceDataValidationError(DemandInsightsError):
    """Raised when source data cannot be validated."""


class ReportGenerationError(DemandInsightsError):
    """Raised when report generation fails."""


class UnsupportedInputError(DemandInsightsError):
    """Raised when an unsupported input object is supplied."""


# =============================================================================
# SECTION 6 — DATA CLASSES
# =============================================================================

@dataclass
class ValidationIssue:
    """Represent one source-data validation issue."""

    issue_type: str

    column: Optional[str]

    affected_rows: int

    severity: str

    description: str


@dataclass
class DataQualitySummary:
    """Summarize the quality of a loaded dataset."""

    total_rows: int = 0

    total_columns: int = 0

    valid_timestamp_rows: int = 0

    invalid_timestamp_rows: int = 0

    missing_timestamp_rows: int = 0

    duplicate_timestamp_rows: int = 0

    duplicate_id_rows: int = 0

    missing_sales_rows: int = 0

    missing_redemption_rows: int = 0

    negative_sales_rows: int = 0

    negative_redemption_rows: int = 0

    zero_sales_rows: int = 0

    zero_redemption_rows: int = 0

    invalid_numeric_sales_rows: int = 0

    invalid_numeric_redemption_rows: int = 0

    overall_missing_cells: int = 0

    overall_missing_percentage: float = 0.0

    validation_issues: list[dict[str, Any]] = field(
        default_factory=list
    )


@dataclass
class DemandMetric:
    """Represent descriptive statistics for one demand series."""

    demand_name: str

    valid_observations: int

    missing_observations: int

    total: float

    mean: float

    median: float

    standard_deviation: float

    minimum: float

    maximum: float

    first_quartile: float

    third_quartile: float

    percentile_90: float

    percentile_95: float

    percentile_99: float

    zero_observations: int

    negative_observations: int


@dataclass
class AnalysisRun:
    """Track the execution of one demand analytics run."""

    run_id: str

    started_at_utc: str

    completed_at_utc: Optional[str] = None

    duration_seconds: float = 0.0

    status: str = "RUNNING"

    source_path: str = ""

    source_rows: int = 0

    source_columns: int = 0

    reports_written: int = 0

    errors: list[str] = field(
        default_factory=list
    )


# =============================================================================
# SECTION 7 — GENERAL UTILITY FUNCTIONS
# =============================================================================

def current_utc_timestamp() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""

    return datetime.now(timezone.utc).isoformat()


def create_run_id() -> str:
    """Create a unique identifier for an analytics run."""

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S"
    )

    return f"{MODULE_NAME}_{timestamp}_{os.getpid()}"


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Convert a value to a finite float.

    Invalid, infinite, or missing values return the supplied default.
    """

    try:
        converted = float(value)

        if not math.isfinite(converted):
            return default

        return converted

    except (TypeError, ValueError, OverflowError):
        return default


def safe_int(
    value: Any,
    default: int = 0,
) -> int:
    """Convert a value to an integer without propagating conversion errors."""

    try:
        if pd.isna(value):
            return default

        return int(value)

    except (TypeError, ValueError, OverflowError):
        return default


def rounded(
    value: Any,
    digits: int = DEFAULT_ROUNDING_DIGITS,
) -> Optional[float]:
    """Round a numeric value while safely handling invalid inputs."""

    try:
        numeric = float(value)

        if not math.isfinite(numeric):
            return None

        return round(numeric, digits)

    except (TypeError, ValueError, OverflowError):
        return None


def percentage(
    numerator: float,
    denominator: float,
    digits: int = DEFAULT_ROUNDING_DIGITS,
) -> float:
    """Calculate a percentage safely."""

    if denominator == 0:
        return 0.0

    return round(
        (numerator / denominator) * 100.0,
        digits,
    )


def normalize_column_name(value: Any) -> str:
    """
    Normalize a column name for comparison.

    Example:
        'Sales Count' -> 'sales_count'
    """

    text_value = str(value).strip().lower()

    normalized_characters = []

    for character in text_value:
        if character.isalnum():
            normalized_characters.append(character)
        else:
            normalized_characters.append("_")

    normalized = "".join(normalized_characters)

    while "__" in normalized:
        normalized = normalized.replace("__", "_")

    return normalized.strip("_")


def json_safe(value: Any) -> Any:
    """
    Convert common Python and pandas objects into JSON-compatible values.
    """

    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (datetime, pd.Timestamp)):
        if pd.isna(value):
            return None

        return value.isoformat()

    if isinstance(value, np.ndarray):
        return [
            json_safe(item)
            for item in value.tolist()
        ]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        if not np.isfinite(value):
            return None

        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

        return value

    if isinstance(value, (int, str, bool)):
        return value

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return str(value)


def write_json_report(
    data: Any,
    destination: Path,
    indent: int = 2,
) -> Path:
    """
    Write a JSON report atomically.

    A temporary file is written first. If serialization succeeds,
    it replaces the destination file.
    """

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                json_safe(data),
                file,
                indent=indent,
                ensure_ascii=False,
                allow_nan=False,
            )

            file.write("\n")

        temporary_path.replace(destination)

        LOGGER.info(
            "JSON report written: %s",
            destination,
        )

        return destination

    except Exception as error:
        try:
            temporary_path.unlink(missing_ok=True)
        except Exception:
            pass

        raise ReportGenerationError(
            f"Could not write JSON report {destination}: {error}"
        ) from error


def write_csv_report(
    dataframe: pd.DataFrame,
    destination: Path,
    index: bool = False,
) -> Path:
    """
    Write a DataFrame to CSV.

    The destination directory is created automatically.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise UnsupportedInputError(
            "write_csv_report expects a pandas DataFrame."
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    try:
        dataframe.to_csv(
            temporary_path,
            index=index,
            encoding="utf-8",
        )

        temporary_path.replace(destination)

        LOGGER.info(
            "CSV report written: %s",
            destination,
        )

        return destination

    except Exception as error:
        try:
            temporary_path.unlink(missing_ok=True)
        except Exception:
            pass

        raise ReportGenerationError(
            f"Could not write CSV report {destination}: {error}"
        ) from error


# =============================================================================
# SECTION 8 — COLUMN DETECTION
# =============================================================================

COLUMN_ALIASES = {
    "timestamp": [
        "timestamp",
        "datetime",
        "date_time",
        "event_timestamp",
        "time_stamp",
    ],

    "sales": [
        "sales_count",
        "sales",
        "ticket_sales",
        "tickets_sold",
        "sale_count",
    ],

    "redemption": [
        "redemption_count",
        "redemption",
        "redeemed_count",
        "ticket_redemptions",
        "tickets_redeemed",
    ],

    "record_id": [
        "id",
        "_id",
        "record_id",
        "row_id",
    ],
}


def find_column(
    dataframe: pd.DataFrame,
    logical_name: str,
    required: bool = False,
) -> Optional[str]:
    """
    Find a source column using known aliases.

    Matching ignores capitalization and common punctuation.
    """

    if logical_name not in COLUMN_ALIASES:
        raise ValueError(
            f"Unknown logical column name: {logical_name}"
        )

    normalized_to_original = {}

    for column in dataframe.columns:
        normalized = normalize_column_name(column)

        normalized_to_original.setdefault(
            normalized,
            column,
        )

    for alias in COLUMN_ALIASES[logical_name]:
        normalized_alias = normalize_column_name(alias)

        if normalized_alias in normalized_to_original:
            return normalized_to_original[normalized_alias]

    if required:
        raise SourceDataValidationError(
            f"Required logical column '{logical_name}' was not found. "
            f"Available columns: {list(dataframe.columns)}"
        )

    return None


def resolve_source_columns(
    dataframe: pd.DataFrame,
) -> dict[str, Optional[str]]:
    """Resolve all known source columns."""

    return {
        "timestamp": find_column(
            dataframe,
            "timestamp",
            required=True,
        ),

        "sales": find_column(
            dataframe,
            "sales",
            required=True,
        ),

        "redemption": find_column(
            dataframe,
            "redemption",
            required=True,
        ),

        "record_id": find_column(
            dataframe,
            "record_id",
            required=False,
        ),
    }


# =============================================================================
# SECTION 9 — SOURCE DATA LOADING
# =============================================================================

def validate_source_path(
    source_path: Path = SOURCE_DATA_PATH,
) -> Path:
    """Validate the configured source path."""

    path = Path(source_path)

    if not path.exists():
        raise SourceDataNotFoundError(
            f"Source dataset does not exist: {path}"
        )

    if not path.is_file():
        raise SourceDataNotFoundError(
            f"Source path is not a file: {path}"
        )

    if path.suffix.lower() != ".csv":
        raise SourceDataValidationError(
            f"Expected a CSV source file, received: {path.suffix}"
        )

    return path


def read_source_csv(
    source_path: Path = SOURCE_DATA_PATH,
    encoding: str = DEFAULT_ENCODING,
) -> pd.DataFrame:
    """
    Load the source CSV without modifying the original file.

    The function first tries UTF-8, then UTF-8 with a byte-order mark.
    """

    path = validate_source_path(source_path)

    LOGGER.info(
        "Loading source dataset: %s",
        path,
    )

    last_error = None

    encodings_to_try = [
        encoding,
        "utf-8-sig",
    ]

    tried = set()

    for current_encoding in encodings_to_try:
        if current_encoding in tried:
            continue

        tried.add(current_encoding)

        try:
            dataframe = pd.read_csv(
                path,
                encoding=current_encoding,
                low_memory=False,
            )

            LOGGER.info(
                "Source loaded with encoding %s; rows=%s, columns=%s",
                current_encoding,
                len(dataframe),
                len(dataframe.columns),
            )

            return dataframe

        except UnicodeDecodeError as error:
            last_error = error

        except pd.errors.EmptyDataError as error:
            raise SourceDataValidationError(
                f"The source CSV is empty: {path}"
            ) from error

        except pd.errors.ParserError as error:
            raise SourceDataValidationError(
                f"The source CSV could not be parsed: {error}"
            ) from error

        except OSError as error:
            raise SourceDataNotFoundError(
                f"Unable to read source dataset: {error}"
            ) from error

    raise SourceDataValidationError(
        f"Could not decode source CSV. Last error: {last_error}"
    )


def load_demand_data(
    source_path: Path = SOURCE_DATA_PATH,
) -> pd.DataFrame:
    """
    Load the historical source dataset and normalize its essential columns.

    Original columns are preserved. Additional normalized columns are added:
        _analysis_timestamp
        _analysis_sales
        _analysis_redemption
    """

    dataframe = read_source_csv(source_path)

    if dataframe.empty:
        raise SourceDataValidationError(
            "The source dataset contains zero rows."
        )

    columns = resolve_source_columns(dataframe)

    dataframe = dataframe.copy()

    dataframe["_analysis_timestamp"] = pd.to_datetime(
        dataframe[columns["timestamp"]],
        errors="coerce",
    )

    dataframe["_analysis_sales"] = pd.to_numeric(
        dataframe[columns["sales"]],
        errors="coerce",
    )

    dataframe["_analysis_redemption"] = pd.to_numeric(
        dataframe[columns["redemption"]],
        errors="coerce",
    )

    dataframe["_analysis_source_row"] = np.arange(
        len(dataframe),
        dtype=np.int64,
    )

    LOGGER.info(
        "Normalized timestamp, sales, and redemption columns."
    )

    return dataframe


# =============================================================================
# SECTION 10 — BASIC DATA VALIDATION
# =============================================================================

def count_invalid_numeric_values(
    original_series: pd.Series,
    numeric_series: pd.Series,
) -> int:
    """
    Count non-missing source values that could not be converted numerically.
    """

    original_present = original_series.notna()

    converted_missing = numeric_series.isna()

    return int(
        (original_present & converted_missing).sum()
    )


def build_validation_issues(
    dataframe: pd.DataFrame,
) -> list[ValidationIssue]:
    """Generate a list of explicit validation issues."""

    issues: list[ValidationIssue] = []

    total_rows = len(dataframe)

    missing_timestamp = int(
        dataframe["_analysis_timestamp"].isna().sum()
    )

    if missing_timestamp:
        issues.append(
            ValidationIssue(
                issue_type="missing_or_invalid_timestamp",
                column=EXPECTED_TIMESTAMP_COLUMN,
                affected_rows=missing_timestamp,
                severity="HIGH",
                description=(
                    "Some rows have a missing or unparseable timestamp."
                ),
            )
        )

    missing_sales = int(
        dataframe["_analysis_sales"].isna().sum()
    )

    if missing_sales:
        issues.append(
            ValidationIssue(
                issue_type="missing_or_invalid_numeric_value",
                column=EXPECTED_SALES_COLUMN,
                affected_rows=missing_sales,
                severity="MEDIUM",
                description=(
                    "Some sales values are missing or non-numeric."
                ),
            )
        )

    missing_redemption = int(
        dataframe["_analysis_redemption"].isna().sum()
    )

    if missing_redemption:
        issues.append(
            ValidationIssue(
                issue_type="missing_or_invalid_numeric_value",
                column=EXPECTED_REDEMPTION_COLUMN,
                affected_rows=missing_redemption,
                severity="MEDIUM",
                description=(
                    "Some redemption values are missing or non-numeric."
                ),
            )
        )

    negative_sales = int(
        (dataframe["_analysis_sales"] < 0).fillna(False).sum()
    )

    if negative_sales:
        issues.append(
            ValidationIssue(
                issue_type="negative_demand_value",
                column=EXPECTED_SALES_COLUMN,
                affected_rows=negative_sales,
                severity="MEDIUM",
                description=(
                    "Negative sales counts were found. "
                    "These values are flagged, not silently changed."
                ),
            )
        )

    negative_redemption = int(
        (dataframe["_analysis_redemption"] < 0).fillna(False).sum()
    )

    if negative_redemption:
        issues.append(
            ValidationIssue(
                issue_type="negative_demand_value",
                column=EXPECTED_REDEMPTION_COLUMN,
                affected_rows=negative_redemption,
                severity="MEDIUM",
                description=(
                    "Negative redemption counts were found. "
                    "These values are flagged, not silently changed."
                ),
            )
        )

    duplicate_timestamps = int(
        dataframe.loc[
            dataframe["_analysis_timestamp"].notna(),
            "_analysis_timestamp",
        ].duplicated().sum()
    )

    if duplicate_timestamps:
        issues.append(
            ValidationIssue(
                issue_type="duplicate_timestamp",
                column=EXPECTED_TIMESTAMP_COLUMN,
                affected_rows=duplicate_timestamps,
                severity="LOW",
                description=(
                    "Multiple records share a timestamp. "
                    "This may be legitimate and requires contextual review."
                ),
            )
        )

    if total_rows == 0:
        issues.append(
            ValidationIssue(
                issue_type="empty_dataset",
                column=None,
                affected_rows=0,
                severity="CRITICAL",
                description="The dataset contains no rows.",
            )
        )

    return issues


def build_data_quality_summary(
    dataframe: pd.DataFrame,
) -> DataQualitySummary:
    """Calculate a comprehensive basic data-quality summary."""

    total_rows = len(dataframe)

    total_columns = len(dataframe.columns)

    valid_timestamps = int(
        dataframe["_analysis_timestamp"].notna().sum()
    )

    missing_timestamps = int(
        dataframe["_analysis_timestamp"].isna().sum()
    )

    duplicate_timestamps = int(
        dataframe.loc[
            dataframe["_analysis_timestamp"].notna(),
            "_analysis_timestamp",
        ].duplicated().sum()
    )

    sales = dataframe["_analysis_sales"]

    redemption = dataframe["_analysis_redemption"]

    source_sales_column = find_column(
        dataframe,
        "sales",
        required=True,
    )

    source_redemption_column = find_column(
        dataframe,
        "redemption",
        required=True,
    )

    invalid_sales = count_invalid_numeric_values(
        dataframe[source_sales_column],
        sales,
    )

    invalid_redemption = count_invalid_numeric_values(
        dataframe[source_redemption_column],
        redemption,
    )

    id_column = find_column(
        dataframe,
        "record_id",
        required=False,
    )

    duplicate_ids = 0

    if id_column is not None:
        duplicate_ids = int(
            dataframe[id_column]
            .dropna()
            .duplicated()
            .sum()
        )

    total_cells = total_rows * total_columns

    missing_cells = int(
        dataframe.isna().sum().sum()
    )

    missing_percentage = percentage(
        missing_cells,
        total_cells,
    )

    issues = build_validation_issues(dataframe)

    return DataQualitySummary(
        total_rows=total_rows,
        total_columns=total_columns,
        valid_timestamp_rows=valid_timestamps,
        invalid_timestamp_rows=missing_timestamps,
        missing_timestamp_rows=missing_timestamps,
        duplicate_timestamp_rows=duplicate_timestamps,
        duplicate_id_rows=duplicate_ids,
        missing_sales_rows=int(sales.isna().sum()),
        missing_redemption_rows=int(redemption.isna().sum()),
        negative_sales_rows=int(
            (sales < 0).fillna(False).sum()
        ),
        negative_redemption_rows=int(
            (redemption < 0).fillna(False).sum()
        ),
        zero_sales_rows=int(
            (sales == 0).fillna(False).sum()
        ),
        zero_redemption_rows=int(
            (redemption == 0).fillna(False).sum()
        ),
        invalid_numeric_sales_rows=invalid_sales,
        invalid_numeric_redemption_rows=invalid_redemption,
        overall_missing_cells=missing_cells,
        overall_missing_percentage=missing_percentage,
        validation_issues=[
            asdict(issue)
            for issue in issues
        ],
    )


def build_missing_values_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Create a missing-value report for every original and derived column."""

    rows = []

    total_rows = len(dataframe)

    for column in dataframe.columns:
        missing_count = int(
            dataframe[column].isna().sum()
        )

        rows.append(
            {
                "Column": column,
                "Missing_Count": missing_count,
                "Missing_Percentage": percentage(
                    missing_count,
                    total_rows,
                ),
                "Non_Missing_Count": total_rows - missing_count,
                "Data_Type": str(dataframe[column].dtype),
            }
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            by=[
                "Missing_Count",
                "Column",
            ],
            ascending=[
                False,
                True,
            ],
        ).reset_index(drop=True)

    return result


def build_duplicate_summary(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize exact duplicate rows and duplicate identifiers."""

    exact_duplicate_rows = int(
        dataframe.duplicated().sum()
    )

    timestamp_series = dataframe["_analysis_timestamp"]

    valid_timestamps = timestamp_series.dropna()

    duplicate_timestamp_count = int(
        valid_timestamps.duplicated().sum()
    )

    timestamp_sales_duplicates = int(
        dataframe.loc[
            dataframe["_analysis_timestamp"].notna(),
        ].duplicated(
            subset=[
                "_analysis_timestamp",
                "_analysis_sales",
                "_analysis_redemption",
            ]
        ).sum()
    )

    id_column = find_column(
        dataframe,
        "record_id",
        required=False,
    )

    duplicate_id_count = 0

    if id_column is not None:
        duplicate_id_count = int(
            dataframe[id_column]
            .dropna()
            .duplicated()
            .sum()
        )

    return {
        "total_rows": len(dataframe),
        "exact_duplicate_rows": exact_duplicate_rows,
        "duplicate_timestamp_rows": duplicate_timestamp_count,
        "duplicate_timestamp_and_demand_rows": (
            timestamp_sales_duplicates
        ),
        "duplicate_id_rows": duplicate_id_count,
        "record_id_column_available": id_column is not None,
        "interpretation": (
            "Duplicate timestamps are reported for investigation. "
            "They are not automatically removed because their meaning "
            "depends on the source system."
        ),
    }


# =============================================================================
# SECTION 11 — DATE AND TIME FEATURES
# =============================================================================

def add_time_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add reusable calendar and clock features.

    The original timestamps remain unchanged in their source column.
    """

    result = dataframe.copy()

    timestamps = result["_analysis_timestamp"]

    result["_analysis_date"] = timestamps.dt.date

    result["_analysis_year"] = timestamps.dt.year

    result["_analysis_month"] = timestamps.dt.month

    result["_analysis_month_name"] = timestamps.dt.month_name()

    result["_analysis_day"] = timestamps.dt.day

    result["_analysis_day_of_week"] = timestamps.dt.dayofweek

    result["_analysis_day_name"] = timestamps.dt.day_name()

    result["_analysis_hour"] = timestamps.dt.hour

    result["_analysis_minute"] = timestamps.dt.minute

    result["_analysis_quarter"] = timestamps.dt.quarter

    result["_analysis_week"] = (
        timestamps.dt.isocalendar().week.astype("Int64")
    )

    result["_analysis_is_weekend"] = (
        timestamps.dt.dayofweek >= 5
    ).where(
        timestamps.notna(),
        pd.NA,
    )

    result["_analysis_hour_decimal"] = (
        result["_analysis_hour"]
        + result["_analysis_minute"] / 60.0
    )

    result["_analysis_time_block"] = (
        result["_analysis_hour"].map(
            classify_time_block
        )
    )

    return result


def classify_time_block(
    hour: Any,
) -> str:
    """
    Classify a timestamp hour into a broad clock-time category.

    These labels are descriptive categories, not official operating hours.
    """

    if pd.isna(hour):
        return "Unknown"

    hour = int(hour)

    if 5 <= hour < 12:
        return "Morning"

    if 12 <= hour < 17:
        return "Afternoon"

    if 17 <= hour < 21:
        return "Evening"

    return "Night"


def classify_weekday(
    day_of_week: Any,
) -> str:
    """Convert a weekday number into weekday/weekend labels."""

    if pd.isna(day_of_week):
        return "Unknown"

    if int(day_of_week) >= 5:
        return "Weekend"

    return "Weekday"


def add_weekday_category(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Add a categorical weekday/weekend feature."""

    result = dataframe.copy()

    result["_analysis_day_category"] = (
        result["_analysis_day_of_week"]
        .map(classify_weekday)
    )

    return result


def prepare_analysis_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Apply all standard preparation steps needed by this module."""

    result = dataframe.copy()

    required_internal_columns = [
        "_analysis_timestamp",
        "_analysis_sales",
        "_analysis_redemption",
    ]

    missing_internal_columns = [
        column
        for column in required_internal_columns
        if column not in result.columns
    ]

    if missing_internal_columns:
        columns = resolve_source_columns(result)

        if "_analysis_timestamp" not in result.columns:
            result["_analysis_timestamp"] = pd.to_datetime(
                result[columns["timestamp"]],
                errors="coerce",
            )

        if "_analysis_sales" not in result.columns:
            result["_analysis_sales"] = pd.to_numeric(
                result[columns["sales"]],
                errors="coerce",
            )

        if "_analysis_redemption" not in result.columns:
            result["_analysis_redemption"] = pd.to_numeric(
                result[columns["redemption"]],
                errors="coerce",
            )

    result = add_time_features(result)

    result = add_weekday_category(result)

    return result


# =============================================================================
# SECTION 12 — DEMAND METRICS
# =============================================================================

def calculate_demand_metric(
    series: pd.Series,
    demand_name: str,
) -> DemandMetric:
    """
    Calculate descriptive statistics for a demand series.

    Missing observations are excluded from numeric statistics and counted
    separately. Negative values are preserved and reported.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    valid = numeric.dropna()

    if valid.empty:
        return DemandMetric(
            demand_name=demand_name,
            valid_observations=0,
            missing_observations=int(numeric.isna().sum()),
            total=0.0,
            mean=0.0,
            median=0.0,
            standard_deviation=0.0,
            minimum=0.0,
            maximum=0.0,
            first_quartile=0.0,
            third_quartile=0.0,
            percentile_90=0.0,
            percentile_95=0.0,
            percentile_99=0.0,
            zero_observations=0,
            negative_observations=0,
        )

    values = valid.to_numpy(
        dtype=np.float64
    )

    return DemandMetric(
        demand_name=demand_name,
        valid_observations=int(valid.size),
        missing_observations=int(numeric.isna().sum()),
        total=rounded(values.sum()) or 0.0,
        mean=rounded(values.mean()) or 0.0,
        median=rounded(np.median(values)) or 0.0,
        standard_deviation=rounded(
            values.std(ddof=1) if len(values) > 1 else 0.0
        ) or 0.0,
        minimum=rounded(values.min()) or 0.0,
        maximum=rounded(values.max()) or 0.0,
        first_quartile=rounded(
            np.percentile(values, 25)
        ) or 0.0,
        third_quartile=rounded(
            np.percentile(values, 75)
        ) or 0.0,
        percentile_90=rounded(
            np.percentile(values, 90)
        ) or 0.0,
        percentile_95=rounded(
            np.percentile(values, 95)
        ) or 0.0,
        percentile_99=rounded(
            np.percentile(values, 99)
        ) or 0.0,
        zero_observations=int((valid == 0).sum()),
        negative_observations=int((valid < 0).sum()),
    )


def calculate_all_demand_metrics(
    dataframe: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Calculate demand metrics for sales and redemptions."""

    metrics = {}

    metrics[EXPECTED_SALES_COLUMN] = asdict(
        calculate_demand_metric(
            dataframe["_analysis_sales"],
            EXPECTED_SALES_COLUMN,
        )
    )

    metrics[EXPECTED_REDEMPTION_COLUMN] = asdict(
        calculate_demand_metric(
            dataframe["_analysis_redemption"],
            EXPECTED_REDEMPTION_COLUMN,
        )
    )

    return metrics


def calculate_total_demand(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate combined sales and redemption totals."""

    sales = pd.to_numeric(
        dataframe["_analysis_sales"],
        errors="coerce",
    )

    redemption = pd.to_numeric(
        dataframe["_analysis_redemption"],
        errors="coerce",
    )

    sales_total = safe_float(
        sales.sum(min_count=1),
        default=0.0,
    )

    redemption_total = safe_float(
        redemption.sum(min_count=1),
        default=0.0,
    )

    return {
        "sales_total": rounded(sales_total),
        "redemption_total": rounded(redemption_total),
        "sales_valid_rows": int(sales.notna().sum()),
        "redemption_valid_rows": int(redemption.notna().sum()),
        "sales_missing_rows": int(sales.isna().sum()),
        "redemption_missing_rows": int(redemption.isna().sum()),
        "combined_total": rounded(
            sales_total + redemption_total
        ),
        "warning": (
            "Sales and redemption are different measures. "
            "Their sum is a descriptive arithmetic total, not a count "
            "of unique passengers or unique tickets."
        ),
    }


# =============================================================================
# SECTION 13 — DATE RANGE AND INTERVAL ANALYSIS
# =============================================================================

def summarize_timestamp_coverage(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Describe the timestamp coverage of the source dataset."""

    timestamps = dataframe["_analysis_timestamp"].dropna()

    if timestamps.empty:
        return {
            "valid_timestamp_count": 0,
            "start_timestamp": None,
            "end_timestamp": None,
            "coverage_days": 0.0,
            "coverage_hours": 0.0,
            "median_interval_minutes": None,
            "mean_interval_minutes": None,
            "interval_count": 0,
        }

    sorted_timestamps = timestamps.sort_values()

    differences = (
        sorted_timestamps
        .diff()
        .dropna()
        .dt.total_seconds()
        .div(60.0)
    )

    positive_differences = differences[
        differences > 0
    ]

    start_timestamp = sorted_timestamps.min()

    end_timestamp = sorted_timestamps.max()

    duration = end_timestamp - start_timestamp

    return {
        "valid_timestamp_count": int(len(sorted_timestamps)),
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "coverage_days": rounded(
            duration.total_seconds() / 86400.0
        ),
        "coverage_hours": rounded(
            duration.total_seconds() / 3600.0
        ),
        "median_interval_minutes": (
            rounded(positive_differences.median())
            if not positive_differences.empty
            else None
        ),
        "mean_interval_minutes": (
            rounded(positive_differences.mean())
            if not positive_differences.empty
            else None
        ),
        "interval_count": int(len(differences)),
        "zero_or_negative_interval_count": int(
            (differences <= 0).sum()
        ),
    }


def build_timestamp_interval_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return intervals between chronologically adjacent valid timestamps.

    Repeated timestamps produce zero-minute intervals. They are preserved
    in this diagnostic table.
    """

    valid = dataframe.loc[
        dataframe["_analysis_timestamp"].notna(),
        [
            "_analysis_timestamp",
            "_analysis_source_row",
        ],
    ].copy()

    valid = valid.sort_values(
        [
            "_analysis_timestamp",
            "_analysis_source_row",
        ]
    ).reset_index(drop=True)

    valid["Previous_Timestamp"] = (
        valid["_analysis_timestamp"].shift(1)
    )

    valid["Interval_Minutes"] = (
        valid["_analysis_timestamp"]
        .diff()
        .dt.total_seconds()
        .div(60.0)
    )

    valid["Interval_Category"] = np.select(
        [
            valid["Interval_Minutes"].isna(),
            valid["Interval_Minutes"] == 0,
            valid["Interval_Minutes"] < 0,
            valid["Interval_Minutes"] <= 15,
            valid["Interval_Minutes"] <= 30,
            valid["Interval_Minutes"] <= 60,
        ],
        [
            "First Observation",
            "Duplicate Timestamp",
            "Negative Interval",
            "0-15 Minutes",
            "16-30 Minutes",
            "31-60 Minutes",
        ],
        default="Over 60 Minutes",
    )

    return valid.rename(
        columns={
            "_analysis_timestamp": "Timestamp",
            "_analysis_source_row": "Source_Row",
        }
    )


def summarize_timestamp_intervals(
    interval_table: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize interval lengths and long gaps."""

    if interval_table.empty:
        return {
            "interval_rows": 0,
            "positive_interval_count": 0,
            "zero_interval_count": 0,
            "negative_interval_count": 0,
            "longer_than_15_minutes": 0,
            "longer_than_30_minutes": 0,
            "longer_than_60_minutes": 0,
            "longer_than_24_hours": 0,
            "largest_interval_minutes": None,
        }

    intervals = pd.to_numeric(
        interval_table["Interval_Minutes"],
        errors="coerce",
    ).dropna()

    return {
        "interval_rows": int(len(intervals)),
        "positive_interval_count": int(
            (intervals > 0).sum()
        ),
        "zero_interval_count": int(
            (intervals == 0).sum()
        ),
        "negative_interval_count": int(
            (intervals < 0).sum()
        ),
        "longer_than_15_minutes": int(
            (intervals > 15).sum()
        ),
        "longer_than_30_minutes": int(
            (intervals > 30).sum()
        ),
        "longer_than_60_minutes": int(
            (intervals > 60).sum()
        ),
        "longer_than_24_hours": int(
            (intervals > 1440).sum()
        ),
        "largest_interval_minutes": (
            rounded(intervals.max())
            if not intervals.empty
            else None
        ),
        "median_positive_interval_minutes": (
            rounded(intervals[intervals > 0].median())
            if (intervals > 0).any()
            else None
        ),
    }


# =============================================================================
# SECTION 14 — GENERIC GROUPED DEMAND ANALYSIS
# =============================================================================

def aggregate_demand_by_group(
    dataframe: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    """
    Aggregate sales and redemption demand by one or more grouping columns.

    Returns:
        Grouping columns
        Observation_Count
        Sales_Count
        Sales_Mean
        Sales_Median
        Sales_Std
        Sales_Max
        Redemption_Count
        Redemption_Mean
        Redemption_Median
        Redemption_Std
        Redemption_Max
    """

    missing_columns = [
        column
        for column in group_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise SourceDataValidationError(
            f"Grouping columns not found: {missing_columns}"
        )

    if not group_columns:
        raise ValueError(
            "At least one grouping column must be supplied."
        )

    result = (
        dataframe.groupby(
            list(group_columns),
            dropna=False,
            observed=False,
        )
        .agg(
            Observation_Count=(
                "_analysis_source_row",
                "count",
            ),
            Sales_Count=(
                "_analysis_sales",
                "sum",
            ),
            Sales_Mean=(
                "_analysis_sales",
                "mean",
            ),
            Sales_Median=(
                "_analysis_sales",
                "median",
            ),
            Sales_Std=(
                "_analysis_sales",
                "std",
            ),
            Sales_Max=(
                "_analysis_sales",
                "max",
            ),
            Redemption_Count=(
                "_analysis_redemption",
                "sum",
            ),
            Redemption_Mean=(
                "_analysis_redemption",
                "mean",
            ),
            Redemption_Median=(
                "_analysis_redemption",
                "median",
            ),
            Redemption_Std=(
                "_analysis_redemption",
                "std",
            ),
            Redemption_Max=(
                "_analysis_redemption",
                "max",
            ),
        )
        .reset_index()
    )

    numeric_columns = [
        "Sales_Count",
        "Sales_Mean",
        "Sales_Median",
        "Sales_Std",
        "Sales_Max",
        "Redemption_Count",
        "Redemption_Mean",
        "Redemption_Median",
        "Redemption_Std",
        "Redemption_Max",
    ]

    for column in numeric_columns:
        result[column] = result[column].round(
            DEFAULT_ROUNDING_DIGITS
        )

    return result


def build_demand_by_hour(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize demand by hour of the day."""

    return aggregate_demand_by_group(
        dataframe,
        ["_analysis_hour"],
    ).rename(
        columns={
            "_analysis_hour": "Hour",
        }
    ).sort_values(
        "Hour"
    ).reset_index(drop=True)


def build_demand_by_day_of_week(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize demand by weekday."""

    result = aggregate_demand_by_group(
        dataframe,
        [
            "_analysis_day_of_week",
            "_analysis_day_name",
        ],
    )

    return result.rename(
        columns={
            "_analysis_day_of_week": "Day_Of_Week_Number",
            "_analysis_day_name": "Day_Of_Week",
        }
    ).sort_values(
        "Day_Of_Week_Number"
    ).reset_index(drop=True)


def build_demand_by_month(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize demand by calendar month."""

    result = aggregate_demand_by_group(
        dataframe,
        [
            "_analysis_year",
            "_analysis_month",
            "_analysis_month_name",
        ],
    )

    return result.rename(
        columns={
            "_analysis_year": "Year",
            "_analysis_month": "Month_Number",
            "_analysis_month_name": "Month",
        }
    ).sort_values(
        [
            "Year",
            "Month_Number",
        ]
    ).reset_index(drop=True)


def build_demand_by_date(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize demand by calendar date."""

    result = aggregate_demand_by_group(
        dataframe,
        ["_analysis_date"],
    )

    result = result.rename(
        columns={
            "_analysis_date": "Date",
        }
    )

    return result.sort_values(
        "Date"
    ).reset_index(drop=True)


def build_demand_by_time_block(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize demand by broad clock-time block."""

    return aggregate_demand_by_group(
        dataframe,
        ["_analysis_time_block"],
    ).rename(
        columns={
            "_analysis_time_block": "Time_Block",
        }
    ).sort_values(
        "Time_Block"
    ).reset_index(drop=True)


def build_demand_by_weekday_category(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize demand for weekdays, weekends, and unknown dates."""

    return aggregate_demand_by_group(
        dataframe,
        ["_analysis_day_category"],
    ).rename(
        columns={
            "_analysis_day_category": "Day_Category",
        }
    ).sort_values(
        "Day_Category"
    ).reset_index(drop=True)


# =============================================================================
# SECTION 15 — DEMAND CORRELATION
# =============================================================================

def build_demand_correlation(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate pairwise correlation between Sales Count and Redemption Count.

    Correlation describes linear association. It does not establish causation.
    """

    selected = dataframe[
        [
            "_analysis_sales",
            "_analysis_redemption",
        ]
    ].copy()

    selected = selected.rename(
        columns={
            "_analysis_sales": EXPECTED_SALES_COLUMN,
            "_analysis_redemption": EXPECTED_REDEMPTION_COLUMN,
        }
    )

    correlation = selected.corr(
        method="pearson",
        min_periods=2,
    )

    result = (
        correlation
        .rename_axis("Variable")
        .reset_index()
    )

    return result


def calculate_sales_redemption_association(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate correlation and paired-data statistics."""

    paired = dataframe[
        [
            "_analysis_sales",
            "_analysis_redemption",
        ]
    ].dropna()

    paired_count = len(paired)

    if paired_count < 2:
        return {
            "paired_observations": paired_count,
            "pearson_correlation": None,
            "interpretation": (
                "Insufficient paired observations for correlation."
            ),
        }

    correlation = paired[
        "_analysis_sales"
    ].corr(
        paired["_analysis_redemption"],
        method="pearson",
    )

    correlation_value = rounded(correlation)

    if correlation_value is None:
        interpretation = (
            "Correlation could not be calculated, possibly because "
            "one or both demand series have zero variance."
        )

    elif correlation_value >= 0.7:
        interpretation = (
            "Strong positive linear association in the paired observations."
        )

    elif correlation_value >= 0.3:
        interpretation = (
            "Moderate positive linear association in the paired observations."
        )

    elif correlation_value > -0.3:
        interpretation = (
            "Weak linear association in the paired observations."
        )

    elif correlation_value > -0.7:
        interpretation = (
            "Moderate negative linear association in the paired observations."
        )

    else:
        interpretation = (
            "Strong negative linear association in the paired observations."
        )

    return {
        "paired_observations": paired_count,
        "pearson_correlation": correlation_value,
        "interpretation": interpretation,
        "causality_warning": (
            "Correlation does not prove that sales cause redemptions "
            "or that redemptions cause sales."
        ),
    }


# =============================================================================
# SECTION 16 — PEAK DEMAND ANALYSIS
# =============================================================================

def identify_peak_demand_periods(
    grouped_dataframe: pd.DataFrame,
    period_column: str,
    demand_column: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Return periods with the largest aggregate demand.

    This function operates on a previously aggregated DataFrame.
    """

    if period_column not in grouped_dataframe.columns:
        raise SourceDataValidationError(
            f"Period column not found: {period_column}"
        )

    if demand_column not in grouped_dataframe.columns:
        raise SourceDataValidationError(
            f"Demand column not found: {demand_column}"
        )

    if top_n < 1:
        raise ValueError(
            "top_n must be at least 1."
        )

    result = grouped_dataframe.copy()

    result[demand_column] = pd.to_numeric(
        result[demand_column],
        errors="coerce",
    )

    result = result.dropna(
        subset=[demand_column]
    )

    result = result.sort_values(
        demand_column,
        ascending=False,
    ).head(top_n)

    result.insert(
        0,
        "Rank",
        np.arange(
            1,
            len(result) + 1,
        ),
    )

    return result.reset_index(
        drop=True
    )


def build_peak_demand_summary(
    dataframe: pd.DataFrame,
    top_n: int = 10,
) -> dict[str, Any]:
    """Create peak-period summaries for sales and redemptions."""

    hourly = build_demand_by_hour(dataframe)

    daily = build_demand_by_date(dataframe)

    weekday = build_demand_by_day_of_week(dataframe)

    hour_sales = identify_peak_demand_periods(
        hourly,
        "Hour",
        "Sales_Count",
        top_n,
    )

    hour_redemption = identify_peak_demand_periods(
        hourly,
        "Hour",
        "Redemption_Count",
        top_n,
    )

    date_sales = identify_peak_demand_periods(
        daily,
        "Date",
        "Sales_Count",
        top_n,
    )

    date_redemption = identify_peak_demand_periods(
        daily,
        "Date",
        "Redemption_Count",
        top_n,
    )

    weekday_sales = identify_peak_demand_periods(
        weekday,
        "Day_Of_Week",
        "Sales_Count",
        top_n,
    )

    weekday_redemption = identify_peak_demand_periods(
        weekday,
        "Day_Of_Week",
        "Redemption_Count",
        top_n,
    )

    return {
        "top_sales_hours": hour_sales.to_dict(
            orient="records"
        ),
        "top_redemption_hours": hour_redemption.to_dict(
            orient="records"
        ),
        "top_sales_dates": date_sales.to_dict(
            orient="records"
        ),
        "top_redemption_dates": date_redemption.to_dict(
            orient="records"
        ),
        "top_sales_weekdays": weekday_sales.to_dict(
            orient="records"
        ),
        "top_redemption_weekdays": weekday_redemption.to_dict(
            orient="records"
        ),
        "methodology": (
            "Peak periods are ranked by historical aggregate demand. "
            "They are descriptive findings and are not forecasts."
        ),
    }


# =============================================================================
# SECTION 17 — AUTOMATIC INSIGHT GENERATION
# =============================================================================

def create_insight(
    category: str,
    title: str,
    description: str,
    severity: str = "INFO",
    metric_name: Optional[str] = None,
    metric_value: Any = None,
    recommendation: Optional[str] = None,
) -> dict[str, Any]:
    """Create a standardized insight record."""

    return {
        "Category": category,
        "Title": title,
        "Description": description,
        "Severity": severity,
        "Metric_Name": metric_name,
        "Metric_Value": json_safe(metric_value),
        "Recommendation": recommendation,
        "Generated_At_UTC": current_utc_timestamp(),
    }


def generate_data_quality_insights(
    quality: DataQualitySummary,
) -> list[dict[str, Any]]:
    """Translate quality checks into understandable insight records."""

    insights = []

    if quality.missing_timestamp_rows > 0:
        insights.append(
            create_insight(
                category="Data Quality",
                title="Missing or Invalid Timestamps",
                description=(
                    f"{quality.missing_timestamp_rows} records have "
                    "missing or invalid timestamps."
                ),
                severity="HIGH",
                metric_name="Missing_Timestamp_Rows",
                metric_value=quality.missing_timestamp_rows,
                recommendation=(
                    "Review the source records before relying on "
                    "time-based analysis."
                ),
            )
        )

    if quality.missing_sales_rows > 0:
        insights.append(
            create_insight(
                category="Data Quality",
                title="Missing Sales Values",
                description=(
                    f"{quality.missing_sales_rows} records have "
                    "missing or non-numeric sales values."
                ),
                severity="MEDIUM",
                metric_name="Missing_Sales_Rows",
                metric_value=quality.missing_sales_rows,
                recommendation=(
                    "Investigate missing sales values. Do not automatically "
                    "interpret missing values as zero demand."
                ),
            )
        )

    if quality.missing_redemption_rows > 0:
        insights.append(
            create_insight(
                category="Data Quality",
                title="Missing Redemption Values",
                description=(
                    f"{quality.missing_redemption_rows} records have "
                    "missing or non-numeric redemption values."
                ),
                severity="MEDIUM",
                metric_name="Missing_Redemption_Rows",
                metric_value=quality.missing_redemption_rows,
                recommendation=(
                    "Investigate missing redemption values before "
                    "drawing conclusions about redemption demand."
                ),
            )
        )

    if quality.negative_sales_rows > 0:
        insights.append(
            create_insight(
                category="Data Quality",
                title="Negative Sales Counts",
                description=(
                    f"{quality.negative_sales_rows} records contain "
                    "negative sales values."
                ),
                severity="MEDIUM",
                metric_name="Negative_Sales_Rows",
                metric_value=quality.negative_sales_rows,
                recommendation=(
                    "Check whether negative values represent corrections, "
                    "refunds, or source-data errors."
                ),
            )
        )

    if quality.negative_redemption_rows > 0:
        insights.append(
            create_insight(
                category="Data Quality",
                title="Negative Redemption Counts",
                description=(
                    f"{quality.negative_redemption_rows} records contain "
                    "negative redemption values."
                ),
                severity="MEDIUM",
                metric_name="Negative_Redemption_Rows",
                metric_value=quality.negative_redemption_rows,
                recommendation=(
                    "Verify the source-system meaning of negative "
                    "redemption counts."
                ),
            )
        )

    if quality.duplicate_timestamp_rows > 0:
        insights.append(
            create_insight(
                category="Data Quality",
                title="Repeated Timestamps",
                description=(
                    f"{quality.duplicate_timestamp_rows} valid timestamp "
                    "values are duplicates of earlier timestamps."
                ),
                severity="LOW",
                metric_name="Duplicate_Timestamp_Rows",
                metric_value=quality.duplicate_timestamp_rows,
                recommendation=(
                    "Inspect repeated timestamps before aggregating or "
                    "resampling the data."
                ),
            )
        )

    if not insights:
        insights.append(
            create_insight(
                category="Data Quality",
                title="No Major Basic Quality Issues Detected",
                description=(
                    "The configured checks did not detect missing essential "
                    "values, negative demand counts, or duplicate timestamps."
                ),
                severity="INFO",
                recommendation=(
                    "Continue with interval, schema, and forecasting "
                    "validation before deployment."
                ),
            )
        )

    return insights


def generate_demand_insights(
    dataframe: pd.DataFrame,
    demand_metrics: dict[str, dict[str, Any]],
    timestamp_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate a first set of descriptive demand insights."""

    insights = []

    for demand_name, metric in demand_metrics.items():
        average = safe_float(
            metric.get("mean"),
        )

        maximum = safe_float(
            metric.get("maximum"),
        )

        minimum = safe_float(
            metric.get("minimum"),
        )

        total = safe_float(
            metric.get("total"),
        )

        valid_count = safe_int(
            metric.get("valid_observations"),
        )

        if valid_count == 0:
            insights.append(
                create_insight(
                    category="Demand",
                    title=f"No Valid {demand_name} Observations",
                    description=(
                        f"No usable numeric observations were available "
                        f"for {demand_name}."
                    ),
                    severity="HIGH",
                    metric_name="Valid_Observations",
                    metric_value=0,
                    recommendation=(
                        "Resolve data-quality issues before interpreting "
                        "this demand series."
                    ),
                )
            )

            continue

        insights.append(
            create_insight(
                category="Demand",
                title=f"{demand_name} Overview",
                description=(
                    f"The dataset contains {valid_count:,} valid "
                    f"{demand_name} observations, with total {total:,.2f}, "
                    f"mean {average:,.2f}, minimum {minimum:,.2f}, "
                    f"and maximum {maximum:,.2f}."
                ),
                severity="INFO",
                metric_name=f"{demand_name}_Mean",
                metric_value=average,
                recommendation=(
                    "Use these descriptive statistics as a baseline. "
                    "Compare future predictions with chronological "
                    "holdout performance."
                ),
            )
        )

        if average > 0 and maximum > average * 3:
            insights.append(
                create_insight(
                    category="Demand Variability",
                    title=f"High Maximum-to-Mean Ratio: {demand_name}",
                    description=(
                        f"The maximum {demand_name} value is more than "
                        "three times the historical mean."
                    ),
                    severity="MEDIUM",
                    metric_name="Maximum_To_Mean_Ratio",
                    metric_value=rounded(
                        maximum / average
                    ),
                    recommendation=(
                        "Inspect high-demand observations and validate "
                        "whether they are legitimate events or data issues."
                    ),
                )
            )

        if metric.get("standard_deviation", 0) > 0:
            coefficient_of_variation = (
                metric["standard_deviation"] / average
                if average != 0
                else None
            )

            insights.append(
                create_insight(
                    category="Demand Variability",
                    title=f"Demand Variability: {demand_name}",
                    description=(
                        "The standard deviation provides a descriptive "
                        "measure of demand spread around the mean."
                    ),
                    severity="INFO",
                    metric_name="Standard_Deviation",
                    metric_value=metric.get(
                        "standard_deviation"
                    ),
                    recommendation=(
                        "Use time-aware validation and suitable error "
                        "metrics when evaluating forecasting models."
                    ),
                )
            )

    coverage_days = safe_float(
        timestamp_summary.get("coverage_days"),
    )

    insights.append(
        create_insight(
            category="Temporal Coverage",
            title="Historical Time Coverage",
            description=(
                f"The valid timestamps span approximately "
                f"{coverage_days:,.2f} days."
            ),
            severity="INFO",
            metric_name="Coverage_Days",
            metric_value=coverage_days,
            recommendation=(
                "Review long timestamp gaps and changes in observation "
                "frequency before interpreting temporal patterns."
            ),
        )
    )

    return insights


# =============================================================================
# SECTION 18 — REPORT TABLE NORMALIZATION
# =============================================================================

def normalize_report_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return a copy suitable for report export.

    Datetime columns are converted to readable strings.
    """

    result = dataframe.copy()

    for column in result.columns:
        if pd.api.types.is_datetime64_any_dtype(
            result[column]
        ):
            result[column] = result[column].astype(
                "string"
            )

    return result


def records_from_dataframe(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-compatible record dictionaries."""

    normalized = normalize_report_dataframe(
        dataframe
    )

    records = normalized.to_dict(
        orient="records"
    )

    return json_safe(records)


def build_demand_insights_table(
    insights: Sequence[dict[str, Any]],
) -> pd.DataFrame:
    """Create a consistent tabular representation of generated insights."""

    expected_columns = [
        "Category",
        "Title",
        "Description",
        "Severity",
        "Metric_Name",
        "Metric_Value",
        "Recommendation",
        "Generated_At_UTC",
    ]

    if not insights:
        return pd.DataFrame(
            columns=expected_columns
        )

    result = pd.DataFrame(
        list(insights)
    )

    for column in expected_columns:
        if column not in result.columns:
            result[column] = None

    return result[
        expected_columns
    ]


def build_source_schema_report(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Create a source-schema profile."""

    rows = []

    for column in dataframe.columns:
        series = dataframe[column]

        rows.append(
            {
                "Column": column,
                "Data_Type": str(series.dtype),
                "Non_Null_Count": int(series.notna().sum()),
                "Missing_Count": int(series.isna().sum()),
                "Unique_Values": int(series.nunique(dropna=True)),
                "Memory_Usage_Bytes": int(
                    series.memory_usage(deep=True)
                ),
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        "Column"
    ).reset_index(
        drop=True
    )


# =============================================================================
# SECTION 19 — CORE ANALYSIS ORCHESTRATION
# =============================================================================

def analyze_demand_dataframe(
    source_dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Run the core in-memory demand analysis.

    This function does not write reports. It returns the analysis artifacts
    for use by a CLI, dashboard, or another module.
    """

    if not isinstance(
        source_dataframe,
        pd.DataFrame,
    ):
        raise UnsupportedInputError(
            "analyze_demand_dataframe expects a pandas DataFrame."
        )

    if source_dataframe.empty:
        raise SourceDataValidationError(
            "Cannot analyze an empty DataFrame."
        )

    LOGGER.info(
        "Starting in-memory demand analysis."
    )

    prepared = prepare_analysis_dataframe(
        source_dataframe
    )

    quality = build_data_quality_summary(
        prepared
    )

    metrics = calculate_all_demand_metrics(
        prepared
    )

    total_demand = calculate_total_demand(
        prepared
    )

    coverage = summarize_timestamp_coverage(
        prepared
    )

    interval_table = build_timestamp_interval_table(
        prepared
    )

    interval_summary = summarize_timestamp_intervals(
        interval_table
    )

    hourly = build_demand_by_hour(
        prepared
    )

    daily = build_demand_by_date(
        prepared
    )

    day_of_week = build_demand_by_day_of_week(
        prepared
    )

    monthly = build_demand_by_month(
        prepared
    )

    time_block = build_demand_by_time_block(
        prepared
    )

    weekday_category = build_demand_by_weekday_category(
        prepared
    )

    correlation_table = build_demand_correlation(
        prepared
    )

    association = calculate_sales_redemption_association(
        prepared
    )

    peak_summary = build_peak_demand_summary(
        prepared
    )

    data_quality_insights = generate_data_quality_insights(
        quality
    )

    demand_insights = generate_demand_insights(
        prepared,
        metrics,
        coverage,
    )

    all_insights = (
        data_quality_insights
        + demand_insights
    )

    insights_table = build_demand_insights_table(
        all_insights
    )

    schema_report = build_source_schema_report(
        source_dataframe
    )

    result = {
        "project_name": PROJECT_NAME,
        "module_name": MODULE_NAME,
        "module_version": MODULE_VERSION,
        "analysis_timestamp_utc": current_utc_timestamp(),
        "source_row_count": len(source_dataframe),
        "source_column_count": len(source_dataframe.columns),
        "demand_metrics": metrics,
        "total_demand": total_demand,
        "data_quality": asdict(quality),
        "timestamp_coverage": coverage,
        "timestamp_interval_summary": interval_summary,
        "sales_redemption_association": association,
        "peak_demand_summary": peak_summary,
        "data_quality_insights": data_quality_insights,
        "demand_insights": demand_insights,
        "all_insights": all_insights,
        "tables": {
            "hourly": hourly,
            "daily": daily,
            "day_of_week": day_of_week,
            "monthly": monthly,
            "time_block": time_block,
            "weekday_category": weekday_category,
            "correlation": correlation_table,
            "missing_values": build_missing_values_table(
                source_dataframe
            ),
            "timestamp_intervals": interval_table,
            "schema": schema_report,
            "insights": insights_table,
        },
    }

    LOGGER.info(
        "Core analysis completed: %s rows, %s insights.",
        len(source_dataframe),
        len(all_insights),
    )

    return result


# =============================================================================
# SECTION 20 — EXPORTING CORE REPORTS
# =============================================================================

def export_core_analysis(
    analysis: dict[str, Any],
) -> list[Path]:
    """
    Export the first set of demand analytics reports.

    Returns a list of successfully written report paths.
    """

    ensure_directories()

    written_files: list[Path] = []

    tables = analysis["tables"]

    json_summary = {
        key: value
        for key, value in analysis.items()
        if key != "tables"
    }

    summary_path = write_json_report(
        json_summary,
        DEMAND_SUMMARY_PATH,
    )

    written_files.append(
        summary_path
    )

    quality_path = write_json_report(
        analysis["data_quality"],
        DATA_QUALITY_REPORT_PATH,
    )

    written_files.append(
        quality_path
    )

    temporal_summary = {
        "timestamp_coverage": analysis["timestamp_coverage"],
        "timestamp_interval_summary": (
            analysis["timestamp_interval_summary"]
        ),
        "sales_redemption_association": (
            analysis["sales_redemption_association"]
        ),
        "peak_demand_summary": (
            analysis["peak_demand_summary"]
        ),
    }

    temporal_path = write_json_report(
        temporal_summary,
        TIME_PATTERN_REPORT_PATH,
    )

    written_files.append(
        temporal_path
    )

    duplicate_summary = build_duplicate_summary(
        prepare_analysis_dataframe(
            read_source_csv(SOURCE_DATA_PATH)
        )
    )

    duplicate_path = write_json_report(
        duplicate_summary,
        DEMAND_DUPLICATE_REPORT_PATH,
    )

    written_files.append(
        duplicate_path
    )

    table_exports = {
        "hourly": DEMAND_BY_HOUR_CSV_PATH,
        "daily": DEMAND_BY_DATE_CSV_PATH,
        "day_of_week": DEMAND_BY_DAY_CSV_PATH,
        "monthly": DEMAND_BY_MONTH_CSV_PATH,
        "correlation": DEMAND_CORRELATION_CSV_PATH,
        "missing_values": DEMAND_MISSING_VALUES_CSV_PATH,
        "insights": DEMAND_INSIGHTS_CSV_PATH,
    }

    for table_name, destination in table_exports.items():
        report_dataframe = tables[table_name]

        path = write_csv_report(
            normalize_report_dataframe(
                report_dataframe
            ),
            destination,
        )

        written_files.append(
            path
        )

    LOGGER.info(
        "Core report export completed. Files written: %s",
        len(written_files),
    )

    return written_files


# =============================================================================
# SECTION 21 — RUN SUMMARY
# =============================================================================

def build_run_summary(
    run: AnalysisRun,
    analysis: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a machine-readable summary of the current run."""

    result = asdict(run)

    result["python_version"] = platform.python_version()

    result["pandas_version"] = pd.__version__

    result["numpy_version"] = np.__version__

    result["platform"] = platform.platform()

    result["module_version"] = MODULE_VERSION

    if analysis is not None:
        result["source_rows"] = analysis.get(
            "source_row_count",
            result["source_rows"],
        )

        result["source_columns"] = analysis.get(
            "source_column_count",
            result["source_columns"],
        )

        result["insight_count"] = len(
            analysis.get(
                "all_insights",
                [],
            )
        )

        result["valid_timestamp_count"] = (
            analysis.get(
                "timestamp_coverage",
                {},
            ).get(
                "valid_timestamp_count",
                0,
            )
        )

    return json_safe(result)


def run_core_demand_analysis(
    source_path: Path = SOURCE_DATA_PATH,
) -> dict[str, Any]:
    """
    Run source loading, analysis, report export, and execution tracking.

    This is the main entry point for Part 1.
    """

    configure_logging()

    started = time.perf_counter()

    run = AnalysisRun(
        run_id=create_run_id(),
        started_at_utc=current_utc_timestamp(),
        source_path=str(source_path),
    )

    LOGGER.info(
        "=" * 79
    )

    LOGGER.info(
        "Starting %s | version %s",
        MODULE_NAME,
        MODULE_VERSION,
    )

    try:
        source_dataframe = load_demand_data(
            source_path
        )

        run.source_rows = len(
            source_dataframe
        )

        run.source_columns = len(
            source_dataframe.columns
        )

        analysis = analyze_demand_dataframe(
            source_dataframe
        )

        # Export uses the same in-memory dataset and does not reread the
        # original CSV. The duplicate report is generated from this dataset.
        ensure_directories()

        run.status = "SUCCESS"

        run.completed_at_utc = current_utc_timestamp()

        run.duration_seconds = round(
            time.perf_counter() - started,
            4,
        )

        # Export all standard reports from the current analysis.
        written_files = export_analysis_without_reload(
            source_dataframe,
            analysis,
        )

        run.reports_written = len(
            written_files
        )

        run_summary = build_run_summary(
            run,
            analysis,
        )

        write_json_report(
            run_summary,
            DEMAND_RUN_SUMMARY_PATH,
        )

        LOGGER.info(
            "Demand analysis completed successfully in %.2f seconds.",
            run.duration_seconds,
        )

        return {
            "run": run_summary,
            "analysis": analysis,
            "written_files": [
                str(path)
                for path in written_files
            ],
        }

    except Exception as error:
        run.status = "FAILED"

        run.completed_at_utc = current_utc_timestamp()

        run.duration_seconds = round(
            time.perf_counter() - started,
            4,
        )

        run.errors.append(
            f"{type(error).__name__}: {error}"
        )

        LOGGER.exception(
            "Demand analysis failed."
        )

        try:
            write_json_report(
                build_run_summary(run),
                DEMAND_RUN_SUMMARY_PATH,
            )
        except Exception:
            LOGGER.exception(
                "Could not write failure run summary."
            )

        raise


# =============================================================================
# SECTION 22 — EXPORT HELPERS
# =============================================================================

def export_analysis_without_reload(
    source_dataframe: pd.DataFrame,
    analysis: dict[str, Any],
) -> list[Path]:
    """
    Export reports without loading the large source CSV a second time.

    This function also exports the duplicate summary using the supplied
    in-memory DataFrame.
    """

    ensure_directories()

    written_files = []

    json_summary = {
        key: value
        for key, value in analysis.items()
        if key != "tables"
    }

    written_files.append(
        write_json_report(
            json_summary,
            DEMAND_SUMMARY_PATH,
        )
    )

    written_files.append(
        write_json_report(
            analysis["data_quality"],
            DATA_QUALITY_REPORT_PATH,
        )
    )

    time_pattern_summary = {
        "timestamp_coverage": analysis["timestamp_coverage"],
        "timestamp_interval_summary": (
            analysis["timestamp_interval_summary"]
        ),
        "sales_redemption_association": (
            analysis["sales_redemption_association"]
        ),
        "peak_demand_summary": (
            analysis["peak_demand_summary"]
        ),
    }

    written_files.append(
        write_json_report(
            time_pattern_summary,
            TIME_PATTERN_REPORT_PATH,
        )
    )

    prepared = prepare_analysis_dataframe(
        source_dataframe
    )

    written_files.append(
        write_json_report(
            build_duplicate_summary(prepared),
            DEMAND_DUPLICATE_REPORT_PATH,
        )
    )

    table_exports = {
        "hourly": DEMAND_BY_HOUR_CSV_PATH,
        "daily": DEMAND_BY_DATE_CSV_PATH,
        "day_of_week": DEMAND_BY_DAY_CSV_PATH,
        "monthly": DEMAND_BY_MONTH_CSV_PATH,
        "correlation": DEMAND_CORRELATION_CSV_PATH,
        "missing_values": DEMAND_MISSING_VALUES_CSV_PATH,
        "insights": DEMAND_INSIGHTS_CSV_PATH,
    }

    for table_name, destination in table_exports.items():
        dataframe = analysis["tables"][table_name]

        written_files.append(
            write_csv_report(
                normalize_report_dataframe(
                    dataframe
                ),
                destination,
            )
        )

    # Export a separate source-column profile.
    schema_path = DEMAND_OUTPUT_DIR / "source_schema.csv"

    written_files.append(
        write_csv_report(
            analysis["tables"]["schema"],
            schema_path,
        )
    )

    # Export timestamp intervals separately for data-quality investigation.
    interval_path = DEMAND_OUTPUT_DIR / "timestamp_intervals.csv"

    written_files.append(
        write_csv_report(
            normalize_report_dataframe(
                analysis["tables"]["timestamp_intervals"]
            ),
            interval_path,
        )
    )

    # Export broad clock-time and weekday/weekend summaries.
    time_block_path = DEMAND_OUTPUT_DIR / "demand_by_time_block.csv"

    written_files.append(
        write_csv_report(
            analysis["tables"]["time_block"],
            time_block_path,
        )
    )

    weekday_category_path = (
        DEMAND_OUTPUT_DIR / "demand_by_weekday_category.csv"
    )

    written_files.append(
        write_csv_report(
            analysis["tables"]["weekday_category"],
            weekday_category_path,
        )
    )

    return written_files


# =============================================================================
# SECTION 23 — PUBLIC API
# =============================================================================

def get_demand_metrics(
    dataframe: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Public helper for retrieving demand metrics from a DataFrame."""

    prepared = prepare_analysis_dataframe(
        dataframe
    )

    return calculate_all_demand_metrics(
        prepared
    )


def get_data_quality_summary(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Public helper for retrieving a data-quality summary."""

    prepared = prepare_analysis_dataframe(
        dataframe
    )

    return asdict(
        build_data_quality_summary(
            prepared
        )
    )


def get_hourly_demand(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Public helper for hourly demand aggregation."""

    prepared = prepare_analysis_dataframe(
        dataframe
    )

    return build_demand_by_hour(
        prepared
    )


def get_daily_demand(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Public helper for daily demand aggregation."""

    prepared = prepare_analysis_dataframe(
        dataframe
    )

    return build_demand_by_date(
        prepared
    )


def get_weekday_demand(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Public helper for weekday demand aggregation."""

    prepared = prepare_analysis_dataframe(
        dataframe
    )

    return build_demand_by_day_of_week(
        prepared
    )


def get_monthly_demand(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Public helper for monthly demand aggregation."""

    prepared = prepare_analysis_dataframe(
        dataframe
    )

    return build_demand_by_month(
        prepared
    )


def get_peak_demand(
    dataframe: pd.DataFrame,
    top_n: int = 10,
) -> dict[str, Any]:
    """Public helper for peak demand analysis."""

    prepared = prepare_analysis_dataframe(
        dataframe
    )

    return build_peak_demand_summary(
        prepared,
        top_n=top_n,
    )


def get_sales_redemption_correlation(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Public helper for sales-redemption correlation."""

    prepared = prepare_analysis_dataframe(
        dataframe
    )

    return calculate_sales_redemption_association(
        prepared
    )


def get_timestamp_coverage(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Public helper for timestamp coverage statistics."""

    prepared = prepare_analysis_dataframe(
        dataframe
    )

    return summarize_timestamp_coverage(
        prepared
    )


def get_demand_insights(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Public helper for generated descriptive insights."""

    prepared = prepare_analysis_dataframe(
        dataframe
    )

    metrics = calculate_all_demand_metrics(
        prepared
    )

    coverage = summarize_timestamp_coverage(
        prepared
    )

    quality = build_data_quality_summary(
        prepared
    )

    return (
        generate_data_quality_insights(quality)
        + generate_demand_insights(
            prepared,
            metrics,
            coverage,
        )
    )


# =============================================================================
# SECTION 24 — COMMAND-LINE ENTRY POINT
# =============================================================================

def main() -> int:
    """
    Execute Part 1 as a command-line analysis module.

    Return code:
        0 = successful completion
        1 = failure
    """

    configure_logging()

    try:
        result = run_core_demand_analysis(
            SOURCE_DATA_PATH
        )

        run_summary = result["run"]

        LOGGER.info(
            "Run ID: %s",
            run_summary.get("run_id"),
        )

        LOGGER.info(
            "Run status: %s",
            run_summary.get("status"),
        )

        LOGGER.info(
            "Rows analyzed: %s",
            run_summary.get("source_rows"),
        )

        LOGGER.info(
            "Reports written: %s",
            run_summary.get("reports_written"),
        )

        LOGGER.info(
            "Run summary: %s",
            DEMAND_RUN_SUMMARY_PATH,
        )

        return 0

    except DemandInsightsError as error:
        LOGGER.error(
            "Demand analytics error: %s",
            error,
        )

        return 1

    except Exception:
        LOGGER.error(
            "Unexpected error occurred:\n%s",
            traceback.format_exc(),
        )

        return 1


# =============================================================================
# SECTION 25 — MODULE EXPORTS
# =============================================================================

__all__ = [
    "PROJECT_NAME",
    "MODULE_NAME",
    "MODULE_VERSION",
    "BASE_DIR",
    "DATA_DIR",
    "OUTPUT_DIR",
    "REPORT_DIR",
    "DEMAND_OUTPUT_DIR",
    "DEMAND_REPORT_DIR",
    "SOURCE_DATA_PATH",
    "DemandInsightsError",
    "SourceDataNotFoundError",
    "SourceDataValidationError",
    "ReportGenerationError",
    "UnsupportedInputError",
    "ValidationIssue",
    "DataQualitySummary",
    "DemandMetric",
    "AnalysisRun",
    "ensure_directories",
    "configure_logging",
    "safe_float",
    "safe_int",
    "rounded",
    "percentage",
    "normalize_column_name",
    "json_safe",
    "write_json_report",
    "write_csv_report",
    "find_column",
    "resolve_source_columns",
    "validate_source_path",
    "read_source_csv",
    "load_demand_data",
    "build_validation_issues",
    "build_data_quality_summary",
    "build_missing_values_table",
    "build_duplicate_summary",
    "add_time_features",
    "classify_time_block",
    "classify_weekday",
    "add_weekday_category",
    "prepare_analysis_dataframe",
    "calculate_demand_metric",
    "calculate_all_demand_metrics",
    "calculate_total_demand",
    "summarize_timestamp_coverage",
    "build_timestamp_interval_table",
    "summarize_timestamp_intervals",
    "aggregate_demand_by_group",
    "build_demand_by_hour",
    "build_demand_by_day_of_week",
    "build_demand_by_month",
    "build_demand_by_date",
    "build_demand_by_time_block",
    "build_demand_by_weekday_category",
    "build_demand_correlation",
    "calculate_sales_redemption_association",
    "identify_peak_demand_periods",
    "build_peak_demand_summary",
    "create_insight",
    "generate_data_quality_insights",
    "generate_demand_insights",
    "normalize_report_dataframe",
    "records_from_dataframe",
    "build_demand_insights_table",
    "build_source_schema_report",
    "analyze_demand_dataframe",
    "export_core_analysis",
    "build_run_summary",
    "run_core_demand_analysis",
    "export_analysis_without_reload",
    "get_demand_metrics",
    "get_data_quality_summary",
    "get_hourly_demand",
    "get_daily_demand",
    "get_weekday_demand",
    "get_monthly_demand",
    "get_peak_demand",
    "get_sales_redemption_correlation",
    "get_timestamp_coverage",
    "get_demand_insights",
    "main",
]


# =============================================================================
# SECTION 26 — DIRECT EXECUTION
# =============================================================================

# =============================================================================
# >>> PART 2 START: TIME-SERIES TRENDS, PEAK PERIODS AND SEASONALITY <<<
# =============================================================================
#
# IMPORTANT PLACEMENT:
# Paste this complete section AFTER the entire Part 1 code.
# Do not delete or replace Part 1.
# Do not indent this section inside main() or another function.
# Do not add another if __name__ == "__main__": block.
#
# This section adds:
#   1. Chronological demand preparation
#   2. Hour-of-day and weekday profiles
#   3. Daily, weekly and monthly demand trends
#   4. Peak and low-demand period detection
#   5. Weekday versus weekend comparisons
#   6. Rolling averages and trend direction
#   7. Demand variability and unusual-observation flags
#   8. Data-coverage-aware time-series summaries
#   9. Seasonal profile comparisons
#  10. Structured trend reports for future dashboard integration
#
# These functions analyze ticket Sales Count and Redemption Count.
# They do not directly measure ferry capacity, passenger queues, or
# actual boarding counts.
# =============================================================================


# =============================================================================
# SECTION 27 — PART 2 CONFIGURATION
# =============================================================================

DI2_VERSION = "1.0.0"

DI2_MODULE_LABEL = "Demand Time-Series and Seasonality Analysis"

DI2_OUTPUT_DIRECTORY = DEMAND_OUTPUT_DIR / "time_patterns"

DI2_REPORT_DIRECTORY = DEMAND_REPORT_DIR / "time_patterns"

DI2_DAILY_TREND_PATH = (
    DI2_OUTPUT_DIRECTORY / "daily_demand_trend.csv"
)

DI2_WEEKLY_TREND_PATH = (
    DI2_OUTPUT_DIRECTORY / "weekly_demand_trend.csv"
)

DI2_MONTHLY_TREND_PATH = (
    DI2_OUTPUT_DIRECTORY / "monthly_demand_trend.csv"
)

DI2_HOURLY_PROFILE_PATH = (
    DI2_OUTPUT_DIRECTORY / "hourly_demand_profile.csv"
)

DI2_WEEKDAY_PROFILE_PATH = (
    DI2_OUTPUT_DIRECTORY / "weekday_demand_profile.csv"
)

DI2_WEEKEND_COMPARISON_PATH = (
    DI2_OUTPUT_DIRECTORY / "weekday_weekend_comparison.csv"
)

DI2_ROLLING_TREND_PATH = (
    DI2_OUTPUT_DIRECTORY / "rolling_demand_trend.csv"
)

DI2_PEAK_PERIODS_PATH = (
    DI2_OUTPUT_DIRECTORY / "peak_demand_periods.csv"
)

DI2_LOW_DEMAND_PERIODS_PATH = (
    DI2_OUTPUT_DIRECTORY / "low_demand_periods.csv"
)

DI2_UNUSUAL_PERIODS_PATH = (
    DI2_OUTPUT_DIRECTORY / "unusual_demand_periods.csv"
)

DI2_SEASONALITY_REPORT_PATH = (
    DI2_REPORT_DIRECTORY / "seasonality_summary.json"
)

DI2_TREND_REPORT_PATH = (
    DI2_REPORT_DIRECTORY / "trend_summary.json"
)

DI2_PEAK_REPORT_PATH = (
    DI2_REPORT_DIRECTORY / "peak_period_summary.json"
)

DI2_WEEKEND_REPORT_PATH = (
    DI2_REPORT_DIRECTORY / "weekday_weekend_summary.json"
)

DI2_RUN_REPORT_PATH = (
    DI2_REPORT_DIRECTORY / "part2_run_summary.json"
)

DI2_DEFAULT_ROLLING_WINDOW = 7

DI2_DEFAULT_PEAK_PERCENTILE = 90.0

DI2_DEFAULT_LOW_PERCENTILE = 10.0

DI2_DEFAULT_UNUSUAL_Z_THRESHOLD = 3.0

DI2_MINIMUM_PROFILE_OBSERVATIONS = 1

DI2_MINIMUM_TREND_POINTS = 3

DI2_EXPECTED_INTERVAL_MINUTES = 15

DI2_TIME_BLOCK_ORDER = [
    "Morning",
    "Afternoon",
    "Evening",
    "Night",
    "Unknown",
]

DI2_WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


# =============================================================================
# SECTION 28 — PART 2 DIRECTORY INITIALIZATION
# =============================================================================

def di2_ensure_directories() -> None:
    """Create directories used exclusively by Part 2."""

    DI2_OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    DI2_REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# SECTION 29 — COMMON INPUT PREPARATION
# =============================================================================

def di2_prepare_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare a DataFrame for Part 2 analysis.

    This function uses Part 1's preparation utilities when available.
    It does not modify the caller's original DataFrame.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise UnsupportedInputError(
            "Part 2 expects a pandas DataFrame."
        )

    if dataframe.empty:
        raise SourceDataValidationError(
            "Part 2 cannot analyze an empty DataFrame."
        )

    result = dataframe.copy()

    required_columns = [
        "_analysis_timestamp",
        "_analysis_sales",
        "_analysis_redemption",
    ]

    if not all(
        column in result.columns
        for column in required_columns
    ):
        result = prepare_analysis_dataframe(result)

    result["_analysis_timestamp"] = pd.to_datetime(
        result["_analysis_timestamp"],
        errors="coerce",
    )

    result["_analysis_sales"] = pd.to_numeric(
        result["_analysis_sales"],
        errors="coerce",
    )

    result["_analysis_redemption"] = pd.to_numeric(
        result["_analysis_redemption"],
        errors="coerce",
    )

    if "_analysis_source_row" not in result.columns:
        result["_analysis_source_row"] = np.arange(
            len(result),
            dtype=np.int64,
        )

    result = result.loc[
        result["_analysis_timestamp"].notna()
    ].copy()

    result = result.sort_values(
        [
            "_analysis_timestamp",
            "_analysis_source_row",
        ]
    ).reset_index(drop=True)

    if result.empty:
        raise SourceDataValidationError(
            "No valid timestamps remain after preparation."
        )

    if "_analysis_date" not in result.columns:
        result["_analysis_date"] = (
            result["_analysis_timestamp"].dt.date
        )

    if "_analysis_hour" not in result.columns:
        result["_analysis_hour"] = (
            result["_analysis_timestamp"].dt.hour
        )

    if "_analysis_day_of_week" not in result.columns:
        result["_analysis_day_of_week"] = (
            result["_analysis_timestamp"].dt.dayofweek
        )

    if "_analysis_day_name" not in result.columns:
        result["_analysis_day_name"] = (
            result["_analysis_timestamp"].dt.day_name()
        )

    if "_analysis_month" not in result.columns:
        result["_analysis_month"] = (
            result["_analysis_timestamp"].dt.month
        )

    if "_analysis_year" not in result.columns:
        result["_analysis_year"] = (
            result["_analysis_timestamp"].dt.year
        )

    if "_analysis_time_block" not in result.columns:
        result["_analysis_time_block"] = (
            result["_analysis_hour"].map(
                classify_time_block
            )
        )

    if "_analysis_day_category" not in result.columns:
        result["_analysis_day_category"] = (
            result["_analysis_day_of_week"].map(
                classify_weekday
            )
        )

    return result


def di2_get_demand_column(
    demand_name: str,
) -> str:
    """
    Resolve a user-facing demand name to its internal numeric column.
    """

    normalized = str(demand_name).strip().lower()

    aliases = {
        "sales": "_analysis_sales",
        "sales count": "_analysis_sales",
        "sales_count": "_analysis_sales",
        "redemption": "_analysis_redemption",
        "redemption count": "_analysis_redemption",
        "redemption_count": "_analysis_redemption",
    }

    if normalized not in aliases:
        raise ValueError(
            "demand_name must identify Sales Count "
            "or Redemption Count."
        )

    return aliases[normalized]


def di2_get_display_demand_name(
    internal_column: str,
) -> str:
    """Return a readable demand name for report output."""

    mapping = {
        "_analysis_sales": "Sales Count",
        "_analysis_redemption": "Redemption Count",
    }

    if internal_column not in mapping:
        return str(internal_column)

    return mapping[internal_column]


def di2_round_series(
    series: pd.Series,
    digits: int = DEFAULT_ROUNDING_DIGITS,
) -> pd.Series:
    """Round a numeric Series while preserving missing values."""

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    return numeric.round(digits)


def di2_safe_mean(
    series: pd.Series,
) -> Optional[float]:
    """Calculate a mean while returning None for unavailable results."""

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return None

    return rounded(values.mean())


def di2_safe_median(
    series: pd.Series,
) -> Optional[float]:
    """Calculate a median while returning None for unavailable results."""

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return None

    return rounded(values.median())


def di2_safe_sum(
    series: pd.Series,
) -> Optional[float]:
    """Calculate a sum without treating an entirely missing series as zero."""

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    if values.notna().sum() == 0:
        return None

    return rounded(values.sum(min_count=1))


def di2_safe_std(
    series: pd.Series,
) -> Optional[float]:
    """Calculate sample standard deviation when enough values exist."""

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(values) < 2:
        return None

    return rounded(values.std(ddof=1))


# =============================================================================
# SECTION 30 — CHRONOLOGICAL VALIDATION
# =============================================================================

def di2_validate_chronology(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Summarize chronology after sorting valid timestamps.

    Duplicate timestamps are counted, not automatically removed.
    """

    prepared = di2_prepare_dataframe(dataframe)

    timestamps = prepared["_analysis_timestamp"]

    differences = (
        timestamps.diff()
        .dt.total_seconds()
        .div(60.0)
    )

    valid_differences = differences.dropna()

    positive_differences = valid_differences[
        valid_differences > 0
    ]

    duplicate_count = int(
        timestamps.duplicated().sum()
    )

    backward_count = int(
        (valid_differences < 0).sum()
    )

    gap_15_count = int(
        (valid_differences > 15).sum()
    )

    gap_30_count = int(
        (valid_differences > 30).sum()
    )

    gap_60_count = int(
        (valid_differences > 60).sum()
    )

    return {
        "valid_timestamp_rows": int(len(prepared)),
        "start_timestamp": timestamps.min(),
        "end_timestamp": timestamps.max(),
        "duplicate_timestamp_rows": duplicate_count,
        "negative_interval_rows": backward_count,
        "interval_count": int(len(valid_differences)),
        "positive_interval_count": int(
            (valid_differences > 0).sum()
        ),
        "median_positive_interval_minutes": (
            rounded(positive_differences.median())
            if not positive_differences.empty
            else None
        ),
        "maximum_interval_minutes": (
            rounded(valid_differences.max())
            if not valid_differences.empty
            else None
        ),
        "gaps_over_15_minutes": gap_15_count,
        "gaps_over_30_minutes": gap_30_count,
        "gaps_over_60_minutes": gap_60_count,
        "interpretation": (
            "Chronology is calculated after sorting timestamps. "
            "A gap indicates elapsed time between recorded observations; "
            "it does not by itself prove that a ferry service was operating "
            "or that demand was zero during that interval."
        ),
    }


def di2_build_interval_flags(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Add diagnostic flags for gaps and repeated timestamps."""

    result = di2_prepare_dataframe(dataframe)

    result["Previous_Timestamp"] = (
        result["_analysis_timestamp"].shift(1)
    )

    result["Interval_Minutes"] = (
        result["_analysis_timestamp"]
        .diff()
        .dt.total_seconds()
        .div(60.0)
    )

    result["Is_Duplicate_Timestamp"] = (
        result["_analysis_timestamp"].duplicated(
            keep="first"
        )
    )

    result["Is_Gap_Over_15_Minutes"] = (
        result["Interval_Minutes"] > 15
    ).fillna(False)

    result["Is_Gap_Over_30_Minutes"] = (
        result["Interval_Minutes"] > 30
    ).fillna(False)

    result["Is_Gap_Over_60_Minutes"] = (
        result["Interval_Minutes"] > 60
    ).fillna(False)

    result["Interval_Classification"] = np.select(
        [
            result["Interval_Minutes"].isna(),
            result["Interval_Minutes"] == 0,
            result["Interval_Minutes"] < 0,
            result["Interval_Minutes"] <= 15,
            result["Interval_Minutes"] <= 30,
            result["Interval_Minutes"] <= 60,
        ],
        [
            "First Record",
            "Repeated Timestamp",
            "Negative Interval",
            "Up To 15 Minutes",
            "16 To 30 Minutes",
            "31 To 60 Minutes",
        ],
        default="Over 60 Minutes",
    )

    return result


# =============================================================================
# SECTION 31 — DAILY DEMAND SERIES
# =============================================================================

def di2_build_daily_series(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate observed records by calendar date.

    This table represents totals across available records for each date.
    A date with missing time intervals may have an incomplete total.
    """

    prepared = di2_prepare_dataframe(dataframe)

    grouped = (
        prepared.groupby(
            "_analysis_date",
            dropna=False,
            observed=False,
        )
        .agg(
            Observation_Count=(
                "_analysis_source_row",
                "count",
            ),
            Sales_Count=(
                "_analysis_sales",
                "sum",
            ),
            Sales_Valid_Count=(
                "_analysis_sales",
                "count",
            ),
            Redemption_Count=(
                "_analysis_redemption",
                "sum",
            ),
            Redemption_Valid_Count=(
                "_analysis_redemption",
                "count",
            ),
            First_Observed_Timestamp=(
                "_analysis_timestamp",
                "min",
            ),
            Last_Observed_Timestamp=(
                "_analysis_timestamp",
                "max",
            ),
        )
        .reset_index()
    )

    grouped = grouped.rename(
        columns={
            "_analysis_date": "Date",
        }
    )

    grouped["Date"] = pd.to_datetime(
        grouped["Date"],
        errors="coerce",
    )

    grouped["Sales_Count"] = grouped[
        "Sales_Count"
    ].where(
        grouped["Sales_Valid_Count"] > 0
    )

    grouped["Redemption_Count"] = grouped[
        "Redemption_Count"
    ].where(
        grouped["Redemption_Valid_Count"] > 0
    )

    grouped["Day_Of_Week"] = (
        grouped["Date"].dt.day_name()
    )

    grouped["Day_Of_Week_Number"] = (
        grouped["Date"].dt.dayofweek
    )

    grouped["Year"] = grouped["Date"].dt.year

    grouped["Month_Number"] = grouped[
        "Date"
    ].dt.month

    grouped["Month_Name"] = grouped[
        "Date"
    ].dt.month_name()

    grouped["Is_Weekend"] = (
        grouped["Day_Of_Week_Number"] >= 5
    )

    grouped["Day_Category"] = np.where(
        grouped["Is_Weekend"],
        "Weekend",
        "Weekday",
    )

    return grouped.sort_values(
        "Date"
    ).reset_index(
        drop=True
    )


def di2_add_daily_change_metrics(
    daily_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate absolute and percentage changes between observed daily totals.

    Percentage change is undefined when the preceding total is zero.
    """

    if not isinstance(
        daily_dataframe,
        pd.DataFrame,
    ):
        raise UnsupportedInputError(
            "daily_dataframe must be a pandas DataFrame."
        )

    required = [
        "Date",
        "Sales_Count",
        "Redemption_Count",
    ]

    missing = [
        column
        for column in required
        if column not in daily_dataframe.columns
    ]

    if missing:
        raise SourceDataValidationError(
            f"Daily series is missing columns: {missing}"
        )

    result = daily_dataframe.copy()

    result = result.sort_values(
        "Date"
    ).reset_index(
        drop=True
    )

    for demand_column in [
        "Sales_Count",
        "Redemption_Count",
    ]:
        previous_column = (
            f"Previous_{demand_column}"
        )

        change_column = (
            f"{demand_column}_Change"
        )

        percentage_column = (
            f"{demand_column}_Change_Percent"
        )

        result[previous_column] = (
            result[demand_column].shift(1)
        )

        result[change_column] = (
            result[demand_column]
            - result[previous_column]
        )

        previous_values = result[previous_column]

        result[percentage_column] = (
            result[change_column]
            .div(previous_values.where(previous_values != 0))
            .mul(100)
        )

        result[f"{demand_column}_Direction"] = np.select(
            [
                result[change_column] > 0,
                result[change_column] < 0,
            ],
            [
                "Increase",
                "Decrease",
            ],
            default="No Change Or Unavailable",
        )

        result.loc[
            result[previous_column].isna(),
            f"{demand_column}_Direction",
        ] = "Initial Observation"

    return result


# =============================================================================
# SECTION 32 — WEEKLY DEMAND SERIES
# =============================================================================

def di2_build_weekly_series(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate available records by Monday-starting calendar week.

    Partial weeks are retained. They are not automatically treated as
    complete seven-day observation periods.
    """

    daily = di2_build_daily_series(dataframe)

    if daily.empty:
        return pd.DataFrame(
            columns=[
                "Week_Start",
                "Observation_Days",
                "Sales_Count",
                "Redemption_Count",
            ]
        )

    daily["Week_Start"] = (
        daily["Date"]
        - pd.to_timedelta(
            daily["Date"].dt.dayofweek,
            unit="D",
        )
    )

    weekly = (
        daily.groupby(
            "Week_Start",
            dropna=False,
        )
        .agg(
            Observation_Days=(
                "Date",
                "nunique",
            ),
            Sales_Count=(
                "Sales_Count",
                "sum",
            ),
            Sales_Valid_Days=(
                "Sales_Count",
                "count",
            ),
            Redemption_Count=(
                "Redemption_Count",
                "sum",
            ),
            Redemption_Valid_Days=(
                "Redemption_Count",
                "count",
            ),
            First_Date=(
                "Date",
                "min",
            ),
            Last_Date=(
                "Date",
                "max",
            ),
        )
        .reset_index()
    )

    weekly["Sales_Count"] = weekly[
        "Sales_Count"
    ].where(
        weekly["Sales_Valid_Days"] > 0
    )

    weekly["Redemption_Count"] = weekly[
        "Redemption_Count"
    ].where(
        weekly["Redemption_Valid_Days"] > 0
    )

    weekly["Is_Partial_Week"] = (
        weekly["Observation_Days"] < 7
    )

    weekly["Sales_Per_Observed_Day"] = (
        weekly["Sales_Count"]
        .div(
            weekly["Sales_Valid_Days"].replace(
                0,
                np.nan,
            )
        )
    )

    weekly["Redemption_Per_Observed_Day"] = (
        weekly["Redemption_Count"]
        .div(
            weekly["Redemption_Valid_Days"].replace(
                0,
                np.nan,
            )
        )
    )

    for column in [
        "Sales_Count",
        "Redemption_Count",
        "Sales_Per_Observed_Day",
        "Redemption_Per_Observed_Day",
    ]:
        weekly[column] = weekly[column].round(
            DEFAULT_ROUNDING_DIGITS
        )

    return weekly.sort_values(
        "Week_Start"
    ).reset_index(
        drop=True
    )


# =============================================================================
# SECTION 33 — MONTHLY DEMAND SERIES
# =============================================================================

def di2_build_monthly_series(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate daily totals by year and calendar month."""

    daily = di2_build_daily_series(dataframe)

    if daily.empty:
        return pd.DataFrame(
            columns=[
                "Year",
                "Month_Number",
                "Month_Name",
                "Sales_Count",
                "Redemption_Count",
            ]
        )

    monthly = (
        daily.groupby(
            [
                "Year",
                "Month_Number",
                "Month_Name",
            ],
            dropna=False,
        )
        .agg(
            Observed_Days=(
                "Date",
                "nunique",
            ),
            Sales_Count=(
                "Sales_Count",
                "sum",
            ),
            Sales_Valid_Days=(
                "Sales_Count",
                "count",
            ),
            Redemption_Count=(
                "Redemption_Count",
                "sum",
            ),
            Redemption_Valid_Days=(
                "Redemption_Count",
                "count",
            ),
            First_Observed_Date=(
                "Date",
                "min",
            ),
            Last_Observed_Date=(
                "Date",
                "max",
            ),
        )
        .reset_index()
    )

    monthly["Sales_Count"] = monthly[
        "Sales_Count"
    ].where(
        monthly["Sales_Valid_Days"] > 0
    )

    monthly["Redemption_Count"] = monthly[
        "Redemption_Count"
    ].where(
        monthly["Redemption_Valid_Days"] > 0
    )

    monthly["Sales_Per_Observed_Day"] = (
        monthly["Sales_Count"]
        .div(
            monthly["Sales_Valid_Days"].replace(
                0,
                np.nan,
            )
        )
    )

    monthly["Redemption_Per_Observed_Day"] = (
        monthly["Redemption_Count"]
        .div(
            monthly["Redemption_Valid_Days"].replace(
                0,
                np.nan,
            )
        )
    )

    for column in [
        "Sales_Count",
        "Redemption_Count",
        "Sales_Per_Observed_Day",
        "Redemption_Per_Observed_Day",
    ]:
        monthly[column] = monthly[column].round(
            DEFAULT_ROUNDING_DIGITS
        )

    return monthly.sort_values(
        [
            "Year",
            "Month_Number",
        ]
    ).reset_index(
        drop=True
    )


# =============================================================================
# SECTION 34 — HOUR-OF-DAY PROFILE
# =============================================================================

def _legacy_di2_build_hourly_profile(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare demand across clock hours.

    Results aggregate all available dates. A high total can reflect a
    combination of high demand and more observations at that hour.
    """

    prepared = di2_prepare_dataframe(dataframe)

    grouped = (
        prepared.groupby(
            "_analysis_hour",
            dropna=False,
        )
        .agg(
            Observation_Count=(
                "_analysis_source_row",
                "count",
            ),
            Sales_Count=(
                "_analysis_sales",
                "sum",
            ),
            Sales_Mean=(
                "_analysis_sales",
                "mean",
            ),
            Sales_Median=(
                "_analysis_sales",
                "median",
            ),
            Sales_Max=(
                "_analysis_sales",
                "max",
            ),
            Redemption_Count=(
                "_analysis_redemption",
                "sum",
            ),
            Redemption_Mean=(
                "_analysis_redemption",
                "mean",
            ),
            Redemption_Median=(
                "_analysis_redemption",
                "median",
            ),
            Redemption_Max=(
                "_analysis_redemption",
                "max",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "_analysis_hour": "Hour",
            }
        )
    )

    grouped["Hour_Label"] = grouped[
        "Hour"
    ].map(
        lambda hour: (
            f"{int(hour):02d}:00"
            if pd.notna(hour)
            else "Unknown"
        )
    )

    grouped["Time_Block"] = grouped[
        "Hour"
    ].map(
        classify_time_block
    )

    for column in [
        "Sales_Count",
        "Sales_Mean",
        "Sales_Median",
        "Sales_Max",
        "Redemption_Count",
        "Redemption_Mean",
        "Redemption_Median",
        "Redemption_Max",
    ]:
        grouped[column] = grouped[column].round(
            DEFAULT_ROUNDING_DIGITS
        )

    return grouped.sort_values(
        "Hour"
    ).reset_index(
        drop=True
    )


def di2_rank_hourly_profile(
    hourly_profile: pd.DataFrame,
    demand_name: str = "Sales Count",
) -> pd.DataFrame:
    """Rank clock hours by total and average observed demand."""

    column_map = {
        "Sales Count": "Sales_Count",
        "Redemption Count": "Redemption_Count",
    }

    if demand_name not in column_map:
        raise ValueError(
            "demand_name must be Sales Count or Redemption Count."
        )

    total_column = column_map[demand_name]

    mean_column = (
        "Sales_Mean"
        if demand_name == "Sales Count"
        else "Redemption_Mean"
    )

    required = [
        "Hour",
        total_column,
        mean_column,
        "Observation_Count",
    ]

    missing = [
        column
        for column in required
        if column not in hourly_profile.columns
    ]

    if missing:
        raise SourceDataValidationError(
            f"Hourly profile is missing columns: {missing}"
        )

    result = hourly_profile.copy()

    result = result.sort_values(
        [
            mean_column,
            total_column,
        ],
        ascending=[
            False,
            False,
        ],
        na_position="last",
    ).reset_index(
        drop=True
    )

    result["Average_Demand_Rank"] = (
        np.arange(1, len(result) + 1)
    )

    result["Ranked_Demand_Measure"] = demand_name

    return result


# =============================================================================
# SECTION 35 — DAY-OF-WEEK PROFILE
# =============================================================================

def _legacy_di2_build_weekday_profile(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare observed demand by weekday.

    Mean demand per recorded observation is included to complement totals.
    """

    prepared = di2_prepare_dataframe(dataframe)

    grouped = (
        prepared.groupby(
            [
                "_analysis_day_of_week",
                "_analysis_day_name",
            ],
            dropna=False,
        )
        .agg(
            Observation_Count=(
                "_analysis_source_row",
                "count",
            ),
            Sales_Count=(
                "_analysis_sales",
                "sum",
            ),
            Sales_Mean=(
                "_analysis_sales",
                "mean",
            ),
            Sales_Median=(
                "_analysis_sales",
                "median",
            ),
            Sales_Std=(
                "_analysis_sales",
                "std",
            ),
            Redemption_Count=(
                "_analysis_redemption",
                "sum",
            ),
            Redemption_Mean=(
                "_analysis_redemption",
                "mean",
            ),
            Redemption_Median=(
                "_analysis_redemption",
                "median",
            ),
            Redemption_Std=(
                "_analysis_redemption",
                "std",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "_analysis_day_of_week": "Weekday_Number",
                "_analysis_day_name": "Weekday",
            }
        )
    )

    grouped["Day_Category"] = np.where(
        grouped["Weekday_Number"] >= 5,
        "Weekend",
        "Weekday",
    )

    grouped["Weekday"] = pd.Categorical(
        grouped["Weekday"],
        categories=DI2_WEEKDAY_ORDER,
        ordered=True,
    )

    for column in [
        "Sales_Count",
        "Sales_Mean",
        "Sales_Median",
        "Sales_Std",
        "Redemption_Count",
        "Redemption_Mean",
        "Redemption_Median",
        "Redemption_Std",
    ]:
        grouped[column] = grouped[column].round(
            DEFAULT_ROUNDING_DIGITS
        )

    return grouped.sort_values(
        "Weekday_Number"
    ).reset_index(
        drop=True
    )


# =============================================================================
# SECTION 36 — WEEKDAY VERSUS WEEKEND COMPARISON
# =============================================================================

def _legacy_di2_build_weekend_comparison(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare weekday and weekend observations.

    This comparison is based on available records, not assumed operating hours.
    """

    prepared = di2_prepare_dataframe(dataframe)

    grouped = (
        prepared.groupby(
            "_analysis_day_category",
            dropna=False,
        )
        .agg(
            Observation_Count=(
                "_analysis_source_row",
                "count",
            ),
            Distinct_Dates=(
                "_analysis_date",
                "nunique",
            ),
            Sales_Count=(
                "_analysis_sales",
                "sum",
            ),
            Sales_Mean=(
                "_analysis_sales",
                "mean",
            ),
            Sales_Median=(
                "_analysis_sales",
                "median",
            ),
            Sales_Std=(
                "_analysis_sales",
                "std",
            ),
            Redemption_Count=(
                "_analysis_redemption",
                "sum",
            ),
            Redemption_Mean=(
                "_analysis_redemption",
                "mean",
            ),
            Redemption_Median=(
                "_analysis_redemption",
                "median",
            ),
            Redemption_Std=(
                "_analysis_redemption",
                "std",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "_analysis_day_category": "Day_Category",
            }
        )
    )

    for column in [
        "Sales_Count",
        "Sales_Mean",
        "Sales_Median",
        "Sales_Std",
        "Redemption_Count",
        "Redemption_Mean",
        "Redemption_Median",
        "Redemption_Std",
    ]:
        grouped[column] = grouped[column].round(
            DEFAULT_ROUNDING_DIGITS
        )

    return grouped.sort_values(
        "Day_Category"
    ).reset_index(
        drop=True
    )


def di2_compare_weekday_weekend_means(
    comparison_table: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize differences between weekday and weekend means."""

    if comparison_table.empty:
        return {
            "status": "INSUFFICIENT_DATA",
            "message": "No weekday/weekend observations were available.",
        }

    indexed = comparison_table.set_index(
        "Day_Category"
    )

    result = {
        "status": "SUCCESS",
        "sales": {},
        "redemption": {},
        "interpretation_note": (
            "Comparisons are descriptive. Differences may reflect "
            "seasonality, operating schedules, missing intervals, or "
            "other unobserved factors."
        ),
    }

    for label, column in [
        ("sales", "Sales_Mean"),
        ("redemption", "Redemption_Mean"),
    ]:
        weekday_mean = None
        weekend_mean = None

        if "Weekday" in indexed.index:
            weekday_mean = rounded(
                indexed.loc["Weekday", column]
            )

        if "Weekend" in indexed.index:
            weekend_mean = rounded(
                indexed.loc["Weekend", column]
            )

        difference = None
        percentage_difference = None

        if (
            weekday_mean is not None
            and weekend_mean is not None
        ):
            difference = rounded(
                weekend_mean - weekday_mean
            )

            if weekday_mean != 0:
                percentage_difference = rounded(
                    (
                        weekend_mean - weekday_mean
                    )
                    / abs(weekday_mean)
                    * 100
                )

        result[label] = {
            "weekday_mean_per_observation": weekday_mean,
            "weekend_mean_per_observation": weekend_mean,
            "weekend_minus_weekday": difference,
            "percentage_difference": percentage_difference,
        }

    return result


# =============================================================================
# SECTION 37 — TIME-BLOCK PROFILE
# =============================================================================

def di2_build_time_block_profile(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate demand by Morning, Afternoon, Evening, and Night."""

    prepared = di2_prepare_dataframe(dataframe)

    result = (
        prepared.groupby(
            "_analysis_time_block",
            dropna=False,
        )
        .agg(
            Observation_Count=(
                "_analysis_source_row",
                "count",
            ),
            Sales_Count=(
                "_analysis_sales",
                "sum",
            ),
            Sales_Mean=(
                "_analysis_sales",
                "mean",
            ),
            Sales_Median=(
                "_analysis_sales",
                "median",
            ),
            Redemption_Count=(
                "_analysis_redemption",
                "sum",
            ),
            Redemption_Mean=(
                "_analysis_redemption",
                "mean",
            ),
            Redemption_Median=(
                "_analysis_redemption",
                "median",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "_analysis_time_block": "Time_Block",
            }
        )
    )

    result["Time_Block"] = pd.Categorical(
        result["Time_Block"],
        categories=DI2_TIME_BLOCK_ORDER,
        ordered=True,
    )

    for column in [
        "Sales_Count",
        "Sales_Mean",
        "Sales_Median",
        "Redemption_Count",
        "Redemption_Mean",
        "Redemption_Median",
    ]:
        result[column] = result[column].round(
            DEFAULT_ROUNDING_DIGITS
        )

    return result.sort_values(
        "Time_Block"
    ).reset_index(
        drop=True
    )


# =============================================================================
# SECTION 38 — PEAK DEMAND THRESHOLDS
# =============================================================================

def di2_calculate_thresholds(
    series: pd.Series,
    peak_percentile: float = DI2_DEFAULT_PEAK_PERCENTILE,
    low_percentile: float = DI2_DEFAULT_LOW_PERCENTILE,
) -> dict[str, Any]:
    """
    Calculate empirical peak and low-demand thresholds.

    Thresholds are descriptive quantiles. They are not operational limits.
    """

    if not 0 <= low_percentile <= 100:
        raise ValueError(
            "low_percentile must be between 0 and 100."
        )

    if not 0 <= peak_percentile <= 100:
        raise ValueError(
            "peak_percentile must be between 0 and 100."
        )

    if low_percentile > peak_percentile:
        raise ValueError(
            "low_percentile cannot exceed peak_percentile."
        )

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return {
            "valid_observations": 0,
            "low_percentile": low_percentile,
            "peak_percentile": peak_percentile,
            "low_threshold": None,
            "peak_threshold": None,
            "median": None,
            "mean": None,
        }

    return {
        "valid_observations": int(len(values)),
        "low_percentile": low_percentile,
        "peak_percentile": peak_percentile,
        "low_threshold": rounded(
            np.percentile(
                values.to_numpy(dtype=np.float64),
                low_percentile,
            )
        ),
        "peak_threshold": rounded(
            np.percentile(
                values.to_numpy(dtype=np.float64),
                peak_percentile,
            )
        ),
        "median": rounded(values.median()),
        "mean": rounded(values.mean()),
    }


def di2_flag_peak_periods(
    dataframe: pd.DataFrame,
    demand_column: str,
    peak_percentile: float = DI2_DEFAULT_PEAK_PERCENTILE,
) -> pd.DataFrame:
    """
    Flag records at or above a selected empirical demand percentile.

    The input is expected to be an aggregated period table.
    """

    if demand_column not in dataframe.columns:
        raise SourceDataValidationError(
            f"Demand column not found: {demand_column}"
        )

    result = dataframe.copy()

    numeric = pd.to_numeric(
        result[demand_column],
        errors="coerce",
    )

    valid = numeric.dropna()

    if valid.empty:
        result["Peak_Threshold"] = np.nan
        result["Is_Peak_Period"] = False
        result["Peak_Percentile"] = peak_percentile
        return result

    threshold = float(
        np.percentile(
            valid.to_numpy(dtype=np.float64),
            peak_percentile,
        )
    )

    result["Peak_Threshold"] = threshold

    result["Is_Peak_Period"] = (
        numeric >= threshold
    ).fillna(False)

    result["Peak_Percentile"] = peak_percentile

    return result


def di2_flag_low_demand_periods(
    dataframe: pd.DataFrame,
    demand_column: str,
    low_percentile: float = DI2_DEFAULT_LOW_PERCENTILE,
) -> pd.DataFrame:
    """Flag records at or below an empirical low-demand percentile."""

    if demand_column not in dataframe.columns:
        raise SourceDataValidationError(
            f"Demand column not found: {demand_column}"
        )

    result = dataframe.copy()

    numeric = pd.to_numeric(
        result[demand_column],
        errors="coerce",
    )

    valid = numeric.dropna()

    if valid.empty:
        result["Low_Demand_Threshold"] = np.nan
        result["Is_Low_Demand_Period"] = False
        result["Low_Demand_Percentile"] = low_percentile
        return result

    threshold = float(
        np.percentile(
            valid.to_numpy(dtype=np.float64),
            low_percentile,
        )
    )

    result["Low_Demand_Threshold"] = threshold

    result["Is_Low_Demand_Period"] = (
        numeric <= threshold
    ).fillna(False)

    result["Low_Demand_Percentile"] = low_percentile

    return result


# =============================================================================
# SECTION 39 — PEAK AND LOW-DEMAND PERIOD TABLES
# =============================================================================

def di2_build_daily_peak_table(
    daily_dataframe: pd.DataFrame,
    demand_name: str = "Sales Count",
    top_n: int = 20,
) -> pd.DataFrame:
    """Return the highest observed daily demand totals."""

    if top_n < 1:
        raise ValueError(
            "top_n must be at least 1."
        )

    demand_column = di2_get_demand_column(
        demand_name
    )

    display_column = (
        "Sales_Count"
        if demand_column == "_analysis_sales"
        else "Redemption_Count"
    )

    if display_column not in daily_dataframe.columns:
        raise SourceDataValidationError(
            f"Daily table lacks {display_column}."
        )

    result = daily_dataframe.copy()

    result[display_column] = pd.to_numeric(
        result[display_column],
        errors="coerce",
    )

    result = result.dropna(
        subset=[display_column]
    )

    result = result.sort_values(
        display_column,
        ascending=False,
    ).head(top_n).reset_index(
        drop=True
    )

    result.insert(
        0,
        "Demand_Rank",
        np.arange(
            1,
            len(result) + 1,
        ),
    )

    result["Demand_Measure"] = demand_name

    return result


def di2_build_daily_low_table(
    daily_dataframe: pd.DataFrame,
    demand_name: str = "Sales Count",
    bottom_n: int = 20,
) -> pd.DataFrame:
    """Return the lowest observed daily demand totals."""

    if bottom_n < 1:
        raise ValueError(
            "bottom_n must be at least 1."
        )

    demand_column = di2_get_demand_column(
        demand_name
    )

    display_column = (
        "Sales_Count"
        if demand_column == "_analysis_sales"
        else "Redemption_Count"
    )

    if display_column not in daily_dataframe.columns:
        raise SourceDataValidationError(
            f"Daily table lacks {display_column}."
        )

    result = daily_dataframe.copy()

    result[display_column] = pd.to_numeric(
        result[display_column],
        errors="coerce",
    )

    result = result.dropna(
        subset=[display_column]
    )

    result = result.sort_values(
        display_column,
        ascending=True,
    ).head(bottom_n).reset_index(
        drop=True
    )

    result.insert(
        0,
        "Demand_Rank",
        np.arange(
            1,
            len(result) + 1,
        ),
    )

    result["Demand_Measure"] = demand_name

    return result


def di2_build_peak_period_summary(
    daily_dataframe: pd.DataFrame,
    top_n: int = 20,
) -> dict[str, Any]:
    """Build structured peak and low-demand summaries."""

    sales_peak = di2_build_daily_peak_table(
        daily_dataframe,
        demand_name="Sales Count",
        top_n=top_n,
    )

    sales_low = di2_build_daily_low_table(
        daily_dataframe,
        demand_name="Sales Count",
        bottom_n=top_n,
    )

    redemption_peak = di2_build_daily_peak_table(
        daily_dataframe,
        demand_name="Redemption Count",
        top_n=top_n,
    )

    redemption_low = di2_build_daily_low_table(
        daily_dataframe,
        demand_name="Redemption Count",
        bottom_n=top_n,
    )

    return {
        "top_n": top_n,
        "sales_peak_days": records_from_dataframe(
            sales_peak
        ),
        "sales_low_days": records_from_dataframe(
            sales_low
        ),
        "redemption_peak_days": records_from_dataframe(
            redemption_peak
        ),
        "redemption_low_days": records_from_dataframe(
            redemption_low
        ),
        "methodology": (
            "Daily totals are ranked using the observed records available "
            "for each date. Incomplete days may not be directly comparable "
            "with fully observed days."
        ),
    }


# =============================================================================
# SECTION 40 — ROLLING AVERAGES
# =============================================================================

def di2_add_rolling_metrics(
    dataframe: pd.DataFrame,
    demand_columns: Sequence[str] = (
        "Sales_Count",
        "Redemption_Count",
    ),
    windows: Sequence[int] = (3, 7, 14),
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """
    Add rolling mean and rolling standard deviation.

    The input should contain one chronologically ordered row per period,
    for example one row per calendar day.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise UnsupportedInputError(
            "dataframe must be a pandas DataFrame."
        )

    if not windows:
        raise ValueError(
            "At least one rolling window must be provided."
        )

    normalized_windows = []

    for window in windows:
        window_value = safe_int(
            window,
            default=0,
        )

        if window_value < 1:
            raise ValueError(
                "Rolling windows must be positive integers."
            )

        normalized_windows.append(
            window_value
        )

    result = dataframe.copy()

    for demand_column in demand_columns:
        if demand_column not in result.columns:
            raise SourceDataValidationError(
                f"Missing demand column: {demand_column}"
            )

        result[demand_column] = pd.to_numeric(
            result[demand_column],
            errors="coerce",
        )

        for window in sorted(
            set(normalized_windows)
        ):
            effective_min_periods = (
                min_periods
                if min_periods is not None
                else window
            )

            if not 1 <= effective_min_periods <= window:
                raise ValueError(
                    "min_periods must be between 1 and window."
                )

            rolling = result[
                demand_column
            ].rolling(
                window=window,
                min_periods=effective_min_periods,
            )

            mean_name = (
                f"{demand_column}_Rolling_Mean_{window}"
            )

            std_name = (
                f"{demand_column}_Rolling_Std_{window}"
            )

            result[mean_name] = rolling.mean()

            result[std_name] = rolling.std(
                ddof=1
            )

    return result


def _legacy_di2_add_rolling_change_metrics(
    dataframe: pd.DataFrame,
    demand_columns: Sequence[str] = (
        "Sales_Count",
        "Redemption_Count",
    ),
    windows: Sequence[int] = (3, 7),
) -> pd.DataFrame:
    """
    Add change between current demand and its rolling mean.

    A positive change means the observed total is above its rolling baseline.
    It does not automatically mean demand is abnormal.
    """

    result = di2_add_rolling_metrics(
        dataframe,
        demand_columns=demand_columns,
        windows=windows,
    )

    for demand_column in demand_columns:
        for window in sorted(
            set(int(value) for value in windows)
        ):
            mean_column = (
                f"{demand_column}_Rolling_Mean_{window}"
            )

            difference_column = (
                f"{demand_column}_Vs_Rolling_Mean_{window}"
            )

            percent_column = (
                f"{demand_column}_Vs_Rolling_Mean_{window}_Percent"
            )

            baseline = result[mean_column]

            result[difference_column] = (
                result[demand_column] - baseline
            )

            result[percent_column] = (
                result[difference_column]
                .div(
                    baseline.where(
                        baseline != 0
                    )
                )
                .mul(100)
            )

    return result


# =============================================================================
# SECTION 41 — TREND DIRECTION
# =============================================================================

def di2_calculate_linear_trend(
    series: pd.Series,
) -> dict[str, Any]:
    """
    Estimate a simple linear trend using observation order.

    This is a descriptive slope, not a causal estimate or a forecast model.
    """

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    valid_mask = values.notna()

    valid_values = values.loc[
        valid_mask
    ].to_numpy(
        dtype=np.float64
    )

    count = len(valid_values)

    if count < DI2_MINIMUM_TREND_POINTS:
        return {
            "status": "INSUFFICIENT_DATA",
            "observations": count,
            "slope_per_observation": None,
            "intercept": None,
            "r_squared": None,
            "direction": "Undetermined",
        }

    x_values = np.arange(
        count,
        dtype=np.float64,
    )

    slope, intercept = np.polyfit(
        x_values,
        valid_values,
        deg=1,
    )

    fitted = (
        slope * x_values + intercept
    )

    residual_sum_squares = float(
        np.square(
            valid_values - fitted
        ).sum()
    )

    total_sum_squares = float(
        np.square(
            valid_values - valid_values.mean()
        ).sum()
    )

    if total_sum_squares > 0:
        r_squared = (
            1.0
            - residual_sum_squares
            / total_sum_squares
        )
    else:
        r_squared = 0.0

    if slope > 0:
        direction = "Increasing"
    elif slope < 0:
        direction = "Decreasing"
    else:
        direction = "Flat"

    return {
        "status": "SUCCESS",
        "observations": count,
        "slope_per_observation": rounded(slope),
        "intercept": rounded(intercept),
        "r_squared": rounded(r_squared),
        "direction": direction,
        "interpretation": (
            "The slope summarizes a straight-line pattern across the "
            "available observations. It does not establish causality and "
            "may not capture nonlinear or seasonal behavior."
        ),
    }


def di2_summarize_daily_trends(
    daily_dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate trend statistics for daily sales and redemptions."""

    result = {
        "daily_observation_count": int(
            len(daily_dataframe)
        ),
        "sales": di2_calculate_linear_trend(
            daily_dataframe["Sales_Count"]
        ),
        "redemption": di2_calculate_linear_trend(
            daily_dataframe["Redemption_Count"]
        ),
    }

    if "Date" in daily_dataframe.columns and not daily_dataframe.empty:
        result["start_date"] = (
            pd.to_datetime(
                daily_dataframe["Date"],
                errors="coerce",
            ).min()
        )

        result["end_date"] = (
            pd.to_datetime(
                daily_dataframe["Date"],
                errors="coerce",
            ).max()
        )

    return result


# =============================================================================
# SECTION 42 — SEASONAL PROFILE COMPARISONS
# =============================================================================

def di2_build_hour_weekday_matrix(
    dataframe: pd.DataFrame,
    demand_name: str = "Sales Count",
) -> pd.DataFrame:
    """
    Build a matrix with weekday rows and hour columns.

    Cell values represent mean demand per observed record for that
    weekday/hour combination.
    """

    demand_column = di2_get_demand_column(
        demand_name
    )

    prepared = di2_prepare_dataframe(
        dataframe
    )

    grouped = (
        prepared.groupby(
            [
                "_analysis_day_name",
                "_analysis_hour",
            ],
            dropna=False,
            observed=False,
        )[demand_column]
        .mean()
        .reset_index()
    )

    matrix = grouped.pivot(
        index="_analysis_day_name",
        columns="_analysis_hour",
        values=demand_column,
    )

    matrix = matrix.reindex(
        index=DI2_WEEKDAY_ORDER
    )

    matrix.columns.name = "Hour"

    matrix.index.name = "Weekday"

    return matrix.round(
        DEFAULT_ROUNDING_DIGITS
    )


def di2_build_month_hour_matrix(
    dataframe: pd.DataFrame,
    demand_name: str = "Sales Count",
) -> pd.DataFrame:
    """
    Build a month-by-hour matrix of mean observed demand.

    Calendar months are represented by month number to preserve ordering.
    """

    demand_column = di2_get_demand_column(
        demand_name
    )

    prepared = di2_prepare_dataframe(
        dataframe
    )

    grouped = (
        prepared.groupby(
            [
                "_analysis_month",
                "_analysis_hour",
            ],
            dropna=False,
        )[demand_column]
        .mean()
        .reset_index()
    )

    matrix = grouped.pivot(
        index="_analysis_month",
        columns="_analysis_hour",
        values=demand_column,
    )

    matrix = matrix.reindex(
        index=range(1, 13)
    )

    matrix.index.name = "Month_Number"

    matrix.columns.name = "Hour"

    return matrix.round(
        DEFAULT_ROUNDING_DIGITS
    )


def di2_build_weekday_hour_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Create a long-form weekday/hour demand summary."""

    prepared = di2_prepare_dataframe(
        dataframe
    )

    result = (
        prepared.groupby(
            [
                "_analysis_day_of_week",
                "_analysis_day_name",
                "_analysis_hour",
            ],
            dropna=False,
        )
        .agg(
            Observation_Count=(
                "_analysis_source_row",
                "count",
            ),
            Sales_Mean=(
                "_analysis_sales",
                "mean",
            ),
            Sales_Median=(
                "_analysis_sales",
                "median",
            ),
            Sales_Total=(
                "_analysis_sales",
                "sum",
            ),
            Redemption_Mean=(
                "_analysis_redemption",
                "mean",
            ),
            Redemption_Median=(
                "_analysis_redemption",
                "median",
            ),
            Redemption_Total=(
                "_analysis_redemption",
                "sum",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "_analysis_day_of_week": "Weekday_Number",
                "_analysis_day_name": "Weekday",
                "_analysis_hour": "Hour",
            }
        )
    )

    for column in [
        "Sales_Mean",
        "Sales_Median",
        "Sales_Total",
        "Redemption_Mean",
        "Redemption_Median",
        "Redemption_Total",
    ]:
        result[column] = result[column].round(
            DEFAULT_ROUNDING_DIGITS
        )

    return result.sort_values(
        [
            "Weekday_Number",
            "Hour",
        ]
    ).reset_index(
        drop=True
    )


# =============================================================================
# SECTION 43 — MONTH-OVER-MONTH CHANGE
# =============================================================================

def di2_add_monthly_change_metrics(
    monthly_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate month-to-month changes in observed aggregate demand."""

    required_columns = [
        "Year",
        "Month_Number",
        "Sales_Count",
        "Redemption_Count",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in monthly_dataframe.columns
    ]

    if missing_columns:
        raise SourceDataValidationError(
            f"Monthly table is missing: {missing_columns}"
        )

    result = monthly_dataframe.copy()

    result = result.sort_values(
        [
            "Year",
            "Month_Number",
        ]
    ).reset_index(
        drop=True
    )

    result["Month_Key"] = (
        result["Year"].astype("Int64").astype(str)
        + "-"
        + result["Month_Number"]
        .astype("Int64")
        .astype(str)
        .str.zfill(2)
    )

    for demand_column in [
        "Sales_Count",
        "Redemption_Count",
    ]:
        previous = result[demand_column].shift(1)

        difference = (
            result[demand_column] - previous
        )

        result[
            f"{demand_column}_Previous_Month"
        ] = previous

        result[
            f"{demand_column}_Month_Change"
        ] = difference

        result[
            f"{demand_column}_Month_Change_Percent"
        ] = (
            difference
            .div(
                previous.where(
                    previous != 0
                )
            )
            .mul(100)
        )

    return result


# =============================================================================
# SECTION 44 — WEEK-OVER-WEEK CHANGE
# =============================================================================

def di2_add_weekly_change_metrics(
    weekly_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate changes between consecutive weekly totals."""

    required_columns = [
        "Week_Start",
        "Sales_Count",
        "Redemption_Count",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in weekly_dataframe.columns
    ]

    if missing_columns:
        raise SourceDataValidationError(
            f"Weekly table is missing: {missing_columns}"
        )

    result = weekly_dataframe.copy()

    result = result.sort_values(
        "Week_Start"
    ).reset_index(
        drop=True
    )

    for demand_column in [
        "Sales_Count",
        "Redemption_Count",
    ]:
        previous = result[demand_column].shift(1)

        change = (
            result[demand_column] - previous
        )

        result[
            f"{demand_column}_Previous_Week"
        ] = previous

        result[
            f"{demand_column}_Week_Change"
        ] = change

        result[
            f"{demand_column}_Week_Change_Percent"
        ] = (
            change
            .div(
                previous.where(
                    previous != 0
                )
            )
            .mul(100)
        )

    return result


# =============================================================================
# SECTION 45 — UNUSUAL DEMAND OBSERVATIONS
# =============================================================================

def di2_calculate_robust_z_scores(
    series: pd.Series,
) -> pd.Series:
    """
    Calculate median/MAD-based robust z-scores.

    When MAD is zero, robust z-scores are unavailable rather than infinite.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    median_value = numeric.median()

    if pd.isna(median_value):
        return pd.Series(
            np.nan,
            index=numeric.index,
            dtype=float,
        )

    absolute_deviation = (
        numeric - median_value
    ).abs()

    mad = absolute_deviation.median()

    if pd.isna(mad) or mad == 0:
        return pd.Series(
            np.nan,
            index=numeric.index,
            dtype=float,
        )

    return (
        0.67448975
        * (numeric - median_value)
        / mad
    )


def di2_flag_unusual_demand(
    dataframe: pd.DataFrame,
    demand_columns: Sequence[str] = (
        "Sales_Count",
        "Redemption_Count",
    ),
    z_threshold: float = DI2_DEFAULT_UNUSUAL_Z_THRESHOLD,
) -> pd.DataFrame:
    """
    Flag unusually high or low values using robust z-scores.

    The threshold is a diagnostic heuristic. It is not a validated anomaly
    detector or proof of an operational incident.
    """

    if z_threshold <= 0:
        raise ValueError(
            "z_threshold must be positive."
        )

    result = dataframe.copy()

    combined_flags = []

    for demand_column in demand_columns:
        if demand_column not in result.columns:
            raise SourceDataValidationError(
                f"Missing demand column: {demand_column}"
            )

        numeric = pd.to_numeric(
            result[demand_column],
            errors="coerce",
        )

        z_column = (
            f"{demand_column}_Robust_Z_Score"
        )

        flag_column = (
            f"{demand_column}_Unusual_Flag"
        )

        result[z_column] = (
            di2_calculate_robust_z_scores(
                numeric
            )
        )

        result[flag_column] = (
            result[z_column].abs() >= z_threshold
        ).fillna(False)

        direction_column = (
            f"{demand_column}_Unusual_Direction"
        )

        result[direction_column] = np.select(
            [
                result[z_column] >= z_threshold,
                result[z_column] <= -z_threshold,
            ],
            [
                "Unusually High",
                "Unusually Low",
            ],
            default="Not Flagged",
        )

        result.loc[
            result[z_column].isna(),
            direction_column,
        ] = "Unavailable"

        combined_flags.append(
            result[flag_column]
        )

    if combined_flags:
        result["Any_Unusual_Demand_Flag"] = np.logical_or.reduce(
            [
                flags.to_numpy(dtype=bool)
                for flags in combined_flags
            ]
        )
    else:
        result["Any_Unusual_Demand_Flag"] = False

    result["Unusual_Demand_Method"] = (
        "Median/MAD robust z-score"
    )

    result["Unusual_Demand_Threshold"] = z_threshold

    return result


def di2_build_unusual_daily_periods(
    daily_dataframe: pd.DataFrame,
    z_threshold: float = DI2_DEFAULT_UNUSUAL_Z_THRESHOLD,
) -> pd.DataFrame:
    """Return daily rows flagged as unusually high or low."""

    flagged = di2_flag_unusual_demand(
        daily_dataframe,
        demand_columns=(
            "Sales_Count",
            "Redemption_Count",
        ),
        z_threshold=z_threshold,
    )

    result = flagged.loc[
        flagged["Any_Unusual_Demand_Flag"]
    ].copy()

    result = result.sort_values(
        "Date"
    ).reset_index(
        drop=True
    )

    return result


# =============================================================================
# SECTION 46 — ZERO-DEMAND AND MISSING-DEMAND PERIODS
# =============================================================================

def di2_summarize_zero_and_missing_demand(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Count observed zero and missing demand values separately.

    A missing value is not interpreted as zero.
    """

    prepared = di2_prepare_dataframe(
        dataframe
    )

    sales = prepared["_analysis_sales"]

    redemption = prepared["_analysis_redemption"]

    return {
        "row_count": int(len(prepared)),
        "sales": {
            "missing_count": int(sales.isna().sum()),
            "zero_count": int((sales == 0).sum()),
            "negative_count": int((sales < 0).sum()),
            "positive_count": int((sales > 0).sum()),
        },
        "redemption": {
            "missing_count": int(redemption.isna().sum()),
            "zero_count": int((redemption == 0).sum()),
            "negative_count": int((redemption < 0).sum()),
            "positive_count": int((redemption > 0).sum()),
        },
        "interpretation": (
            "Zero counts are observed numeric values. Missing counts "
            "represent unavailable or invalid observations. The two "
            "categories are kept separate."
        ),
    }


# =============================================================================
# SECTION 47 — DEMAND DISTRIBUTION SUMMARY
# =============================================================================

def di2_distribution_summary(
    series: pd.Series,
    demand_name: str,
) -> dict[str, Any]:
    """Describe skew, spread, quantiles, and zero frequency."""

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    values = numeric.dropna()

    if values.empty:
        return {
            "demand_name": demand_name,
            "valid_count": 0,
            "missing_count": int(numeric.isna().sum()),
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "variance": None,
            "minimum": None,
            "maximum": None,
            "range": None,
            "q10": None,
            "q25": None,
            "q75": None,
            "q90": None,
            "q95": None,
            "q99": None,
            "skewness": None,
            "kurtosis": None,
            "zero_count": 0,
            "negative_count": 0,
        }

    variance = (
        values.var(ddof=1)
        if len(values) > 1
        else 0.0
    )

    skewness = (
        values.skew()
        if len(values) >= 3
        else np.nan
    )

    kurtosis = (
        values.kurt()
        if len(values) >= 4
        else np.nan
    )

    q_values = values.quantile(
        [
            0.10,
            0.25,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )

    return {
        "demand_name": demand_name,
        "valid_count": int(len(values)),
        "missing_count": int(numeric.isna().sum()),
        "mean": rounded(values.mean()),
        "median": rounded(values.median()),
        "standard_deviation": rounded(
            values.std(ddof=1)
            if len(values) > 1
            else 0
        ),
        "variance": rounded(variance),
        "minimum": rounded(values.min()),
        "maximum": rounded(values.max()),
        "range": rounded(
            values.max() - values.min()
        ),
        "q10": rounded(q_values.loc[0.10]),
        "q25": rounded(q_values.loc[0.25]),
        "q75": rounded(q_values.loc[0.75]),
        "q90": rounded(q_values.loc[0.90]),
        "q95": rounded(q_values.loc[0.95]),
        "q99": rounded(q_values.loc[0.99]),
        "skewness": rounded(skewness),
        "kurtosis": rounded(kurtosis),
        "zero_count": int((values == 0).sum()),
        "negative_count": int((values < 0).sum()),
    }


def di2_build_distribution_report(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Create distribution summaries for both demand measures."""

    prepared = di2_prepare_dataframe(
        dataframe
    )

    return {
        "sales": di2_distribution_summary(
            prepared["_analysis_sales"],
            "Sales Count",
        ),
        "redemption": di2_distribution_summary(
            prepared["_analysis_redemption"],
            "Redemption Count",
        ),
        "interpretation_note": (
            "Distribution statistics summarize observed ticket counts. "
            "Skewness and kurtosis can be sensitive to unusual values."
        ),
    }


# =============================================================================
# SECTION 48 — DEMAND STABILITY SUMMARY
# =============================================================================

def di2_calculate_coefficient_of_variation(
    series: pd.Series,
) -> dict[str, Any]:
    """Calculate standard deviation divided by absolute mean."""

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return {
            "valid_count": 0,
            "mean": None,
            "standard_deviation": None,
            "coefficient_of_variation": None,
            "status": "NO_VALID_DATA",
        }

    mean_value = float(
        values.mean()
    )

    std_value = (
        float(values.std(ddof=1))
        if len(values) > 1
        else 0.0
    )

    if mean_value == 0:
        coefficient = None
        status = "ZERO_MEAN"
    else:
        coefficient = (
            std_value / abs(mean_value)
        )
        status = "SUCCESS"

    return {
        "valid_count": int(len(values)),
        "mean": rounded(mean_value),
        "standard_deviation": rounded(std_value),
        "coefficient_of_variation": rounded(
            coefficient
        ),
        "status": status,
        "interpretation": (
            "The coefficient of variation describes spread relative to "
            "the absolute mean. It may be unstable when the mean is near "
            "zero and should not be compared blindly across different "
            "demand scales."
        ),
    }


def di2_build_stability_report(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize observation-level and daily demand variability."""

    prepared = di2_prepare_dataframe(
        dataframe
    )

    daily = di2_build_daily_series(
        prepared
    )

    return {
        "observation_level": {
            "sales": di2_calculate_coefficient_of_variation(
                prepared["_analysis_sales"]
            ),
            "redemption": di2_calculate_coefficient_of_variation(
                prepared["_analysis_redemption"]
            ),
        },
        "daily_total_level": {
            "sales": di2_calculate_coefficient_of_variation(
                daily["Sales_Count"]
            ),
            "redemption": di2_calculate_coefficient_of_variation(
                daily["Redemption_Count"]
            ),
        },
    }


# =============================================================================
# SECTION 49 — PROFILE COVERAGE
# =============================================================================

def di2_build_profile_coverage(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Report how many observations support each hour and weekday profile.

    This helps identify categories that may be based on fewer observations.
    """

    prepared = di2_prepare_dataframe(
        dataframe
    )

    hour_coverage = (
        prepared.groupby(
            "_analysis_hour",
            dropna=False,
        )
        .agg(
            Observation_Count=(
                "_analysis_source_row",
                "count",
            ),
            Distinct_Dates=(
                "_analysis_date",
                "nunique",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "_analysis_hour": "Category_Value",
            }
        )
    )

    hour_coverage["Profile_Type"] = "Hour"

    hour_coverage["Category_Label"] = (
        hour_coverage["Category_Value"].map(
            lambda value: (
                f"{int(value):02d}:00"
                if pd.notna(value)
                else "Unknown"
            )
        )
    )

    weekday_coverage = (
        prepared.groupby(
            [
                "_analysis_day_of_week",
                "_analysis_day_name",
            ],
            dropna=False,
        )
        .agg(
            Observation_Count=(
                "_analysis_source_row",
                "count",
            ),
            Distinct_Dates=(
                "_analysis_date",
                "nunique",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "_analysis_day_of_week": "Category_Value",
                "_analysis_day_name": "Category_Label",
            }
        )
    )

    weekday_coverage["Profile_Type"] = "Weekday"

    columns = [
        "Profile_Type",
        "Category_Value",
        "Category_Label",
        "Observation_Count",
        "Distinct_Dates",
    ]

    result = pd.concat(
        [
            hour_coverage[columns],
            weekday_coverage[columns],
        ],
        ignore_index=True,
    )

    result["Observation_Count"] = (
        result["Observation_Count"].astype("int64")
    )

    result["Distinct_Dates"] = (
        result["Distinct_Dates"].astype("int64")
    )

    return result.sort_values(
        [
            "Profile_Type",
            "Category_Value",
        ]
    ).reset_index(
        drop=True
    )


# =============================================================================
# SECTION 50 — COVERAGE-AWARE DAILY FLAGS
# =============================================================================

def di2_add_daily_coverage_flags(
    daily_dataframe: pd.DataFrame,
    minimum_observations: int = 1,
) -> pd.DataFrame:
    """
    Add basic daily coverage flags based on observed record counts.

    A high record count does not guarantee a complete day because the
    source contains irregular gaps. These flags are preliminary diagnostics.
    """

    if minimum_observations < 1:
        raise ValueError(
            "minimum_observations must be at least 1."
        )

    required = [
        "Date",
        "Observation_Count",
        "First_Observed_Timestamp",
        "Last_Observed_Timestamp",
    ]

    missing = [
        column
        for column in required
        if column not in daily_dataframe.columns
    ]

    if missing:
        raise SourceDataValidationError(
            f"Daily coverage table is missing: {missing}"
        )

    result = daily_dataframe.copy()

    result["Observation_Count"] = pd.to_numeric(
        result["Observation_Count"],
        errors="coerce",
    )

    result["Observed_Span_Minutes"] = (
        pd.to_datetime(
            result["Last_Observed_Timestamp"],
            errors="coerce",
        )
        - pd.to_datetime(
            result["First_Observed_Timestamp"],
            errors="coerce",
        )
    ).dt.total_seconds().div(60.0)

    result["Meets_Minimum_Observation_Count"] = (
        result["Observation_Count"] >= minimum_observations
    ).fillna(False)

    result["Observed_Span_Hours"] = (
        result["Observed_Span_Minutes"] / 60.0
    )

    result["Coverage_Classification"] = np.select(
        [
            result["Observation_Count"].isna(),
            result["Observation_Count"] == 0,
            result["Observed_Span_Minutes"].isna(),
            result["Observed_Span_Minutes"] == 0,
            result["Observed_Span_Minutes"] < 60,
            result["Observed_Span_Minutes"] < 720,
        ],
        [
            "Unknown",
            "No Observations",
            "Unknown Span",
            "Single-Time-Point Span",
            "Short Observed Span",
            "Partial Observed Span",
        ],
        default="Long Observed Span",
    )

    result["Coverage_Warning"] = (
        "Observed span is only a preliminary coverage indicator; "
        "internal timestamp gaps may remain."
    )

    return result


# =============================================================================
# SECTION 51 — DAILY OBSERVATION COUNT PROFILE
# =============================================================================

def di2_build_daily_observation_profile(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize the number of recorded observations per calendar date."""

    prepared = di2_prepare_dataframe(
        dataframe
    )

    result = (
        prepared.groupby(
            "_analysis_date",
            dropna=False,
        )
        .agg(
            Observation_Count=(
                "_analysis_source_row",
                "count",
            ),
            First_Observed_Timestamp=(
                "_analysis_timestamp",
                "min",
            ),
            Last_Observed_Timestamp=(
                "_analysis_timestamp",
                "max",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "_analysis_date": "Date",
            }
        )
    )

    result["Date"] = pd.to_datetime(
        result["Date"],
        errors="coerce",
    )

    result["Observed_Span_Minutes"] = (
        result["Last_Observed_Timestamp"]
        - result["First_Observed_Timestamp"]
    ).dt.total_seconds().div(60.0)

    result["Observed_Span_Hours"] = (
        result["Observed_Span_Minutes"] / 60.0
    )

    result["Interval_Mean_Minutes"] = np.nan

    # Calculate interval summaries by date. This operates on the observed
    # timestamps only and does not invent records for missing time slots.
    timestamp_groups = prepared.groupby(
        "_analysis_date",
        sort=False,
    )["_analysis_timestamp"]

    interval_means = {}

    interval_maxima = {}

    for date_value, timestamp_series in timestamp_groups:
        sorted_series = timestamp_series.sort_values()

        intervals = (
            sorted_series.diff()
            .dt.total_seconds()
            .div(60.0)
            .dropna()
        )

        positive_intervals = intervals[
            intervals > 0
        ]

        interval_means[date_value] = (
            rounded(positive_intervals.mean())
            if not positive_intervals.empty
            else None
        )

        interval_maxima[date_value] = (
            rounded(positive_intervals.max())
            if not positive_intervals.empty
            else None
        )

    result["Interval_Mean_Minutes"] = (
        result["Date"].dt.date.map(
            interval_means
        )
    )

    result["Maximum_Within_Day_Gap_Minutes"] = (
        result["Date"].dt.date.map(
            interval_maxima
        )
    )

    return result.sort_values(
        "Date"
    ).reset_index(
        drop=True
    )


# =============================================================================
# SECTION 52 — DEMAND SHARE BY PERIOD
# =============================================================================

def di2_add_demand_share(
    grouped_dataframe: pd.DataFrame,
    demand_column: str,
    output_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate each row's percentage share of total grouped demand.

    Shares are calculated only over available numeric aggregate values.
    """

    if demand_column not in grouped_dataframe.columns:
        raise SourceDataValidationError(
            f"Demand column not found: {demand_column}"
        )

    result = grouped_dataframe.copy()

    numeric = pd.to_numeric(
        result[demand_column],
        errors="coerce",
    )

    total = numeric.sum(
        min_count=1
    )

    if output_column is None:
        output_column = (
            f"{demand_column}_Share_Percent"
        )

    if pd.isna(total) or total == 0:
        result[output_column] = np.nan
    else:
        result[output_column] = (
            numeric / total * 100.0
        ).round(
            DEFAULT_ROUNDING_DIGITS
        )

    return result


def di2_add_both_demand_shares(
    grouped_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Add sales and redemption share percentages when columns exist."""

    result = grouped_dataframe.copy()

    if "Sales_Count" in result.columns:
        result = di2_add_demand_share(
            result,
            "Sales_Count",
            "Sales_Share_Percent",
        )

    if "Redemption_Count" in result.columns:
        result = di2_add_demand_share(
            result,
            "Redemption_Count",
            "Redemption_Share_Percent",
        )

    return result


# =============================================================================
# SECTION 53 — PEAK HOUR INSIGHTS
# =============================================================================

def di2_generate_peak_hour_insights(
    hourly_profile: pd.DataFrame,
    top_n: int = 3,
) -> list[dict[str, Any]]:
    """Generate short descriptive insights from the hourly profile."""

    if top_n < 1:
        raise ValueError(
            "top_n must be at least 1."
        )

    if hourly_profile.empty:
        return [
            create_insight(
                category="Time Pattern",
                title="Hourly Profile Unavailable",
                description=(
                    "No valid observations were available for the "
                    "hour-of-day profile."
                ),
                severity="MEDIUM",
                recommendation=(
                    "Review timestamp parsing and source-data coverage."
                ),
            )
        ]

    insights = []

    for label, demand_column, mean_column in [
        (
            "Sales",
            "Sales_Count",
            "Sales_Mean",
        ),
        (
            "Redemption",
            "Redemption_Count",
            "Redemption_Mean",
        ),
    ]:
        if mean_column not in hourly_profile.columns:
            continue

        ranked = hourly_profile.sort_values(
            [
                mean_column,
                demand_column,
            ],
            ascending=[
                False,
                False,
            ],
            na_position="last",
        ).head(top_n)

        if ranked.empty:
            continue

        periods = []

        for _, row in ranked.iterrows():
            hour = row.get("Hour")

            if pd.isna(hour):
                label_text = "Unknown hour"
            else:
                label_text = f"{int(hour):02d}:00"

            periods.append(
                {
                    "hour": label_text,
                    "mean": rounded(
                        row.get(mean_column)
                    ),
                    "total": rounded(
                        row.get(demand_column)
                    ),
                    "observations": safe_int(
                        row.get("Observation_Count")
                    ),
                }
            )

        best_period = periods[0]

        insights.append(
            create_insight(
                category="Time Pattern",
                title=f"Highest Average {label} Hour",
                description=(
                    f"The highest average observed {label.lower()} demand "
                    f"occurs at {best_period['hour']} in the aggregated "
                    "historical profile."
                ),
                severity="INFO",
                metric_name=f"{label}_Mean_At_Hour",
                metric_value=best_period["mean"],
                recommendation=(
                    "Use this as a historical comparison point. Confirm "
                    "data coverage and current operating schedules before "
                    "using it for resource planning."
                ),
            )
        )

    return insights


# =============================================================================
# SECTION 54 — WEEKDAY INSIGHTS
# =============================================================================

def di2_generate_weekday_insights(
    weekday_profile: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Generate descriptive insights from the weekday profile."""

    if weekday_profile.empty:
        return [
            create_insight(
                category="Weekly Pattern",
                title="Weekday Profile Unavailable",
                description=(
                    "No valid weekday observations were available."
                ),
                severity="MEDIUM",
                recommendation=(
                    "Review timestamp validity before interpreting weekday "
                    "patterns."
                ),
            )
        ]

    insights = []

    for label, mean_column in [
        ("Sales", "Sales_Mean"),
        ("Redemption", "Redemption_Mean"),
    ]:
        if mean_column not in weekday_profile.columns:
            continue

        valid = weekday_profile.dropna(
            subset=[mean_column]
        )

        if valid.empty:
            continue

        highest = valid.loc[
            valid[mean_column].idxmax()
        ]

        lowest = valid.loc[
            valid[mean_column].idxmin()
        ]

        insights.append(
            create_insight(
                category="Weekly Pattern",
                title=f"Highest Average {label} Weekday",
                description=(
                    f"{highest['Weekday']} has the highest mean observed "
                    f"{label.lower()} count per record among the weekdays "
                    "in this dataset."
                ),
                severity="INFO",
                metric_name=f"{label}_Highest_Weekday_Mean",
                metric_value=rounded(
                    highest[mean_column]
                ),
                recommendation=(
                    "Compare the result with observation counts and "
                    "coverage before drawing operational conclusions."
                ),
            )
        )

        insights.append(
            create_insight(
                category="Weekly Pattern",
                title=f"Lowest Average {label} Weekday",
                description=(
                    f"{lowest['Weekday']} has the lowest mean observed "
                    f"{label.lower()} count per record among the weekdays "
                    "in this dataset."
                ),
                severity="INFO",
                metric_name=f"{label}_Lowest_Weekday_Mean",
                metric_value=rounded(
                    lowest[mean_column]
                ),
                recommendation=(
                    "Check whether the difference is consistent across "
                    "multiple weeks rather than relying on a single "
                    "aggregate."
                ),
            )
        )

    return insights


# =============================================================================
# SECTION 55 — MONTHLY INSIGHTS
# =============================================================================

def di2_generate_monthly_insights(
    monthly_dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Generate insights about monthly observed demand totals."""

    if monthly_dataframe.empty:
        return []

    insights = []

    for label, demand_column in [
        ("Sales", "Sales_Count"),
        ("Redemption", "Redemption_Count"),
    ]:
        if demand_column not in monthly_dataframe.columns:
            continue

        valid = monthly_dataframe.dropna(
            subset=[demand_column]
        )

        if valid.empty:
            continue

        highest = valid.loc[
            valid[demand_column].idxmax()
        ]

        lowest = valid.loc[
            valid[demand_column].idxmin()
        ]

        highest_period = (
            f"{int(highest['Year'])}-"
            f"{int(highest['Month_Number']):02d}"
        )

        lowest_period = (
            f"{int(lowest['Year'])}-"
            f"{int(lowest['Month_Number']):02d}"
        )

        insights.append(
            create_insight(
                category="Monthly Pattern",
                title=f"Highest Observed Monthly {label}",
                description=(
                    f"The largest observed monthly {label.lower()} total "
                    f"occurred in {highest_period}. The month may be "
                    "partial, so compare observed coverage before treating "
                    "it as a seasonal peak."
                ),
                severity="INFO",
                metric_name=f"{label}_Highest_Month_Total",
                metric_value=rounded(
                    highest[demand_column]
                ),
                recommendation=(
                    "Review observed-day counts and compare equivalent "
                    "calendar periods before interpreting seasonality."
                ),
            )
        )

        insights.append(
            create_insight(
                category="Monthly Pattern",
                title=f"Lowest Observed Monthly {label}",
                description=(
                    f"The smallest observed monthly {label.lower()} total "
                    f"occurred in {lowest_period}. Incomplete coverage "
                    "may affect this comparison."
                ),
                severity="INFO",
                metric_name=f"{label}_Lowest_Month_Total",
                metric_value=rounded(
                    lowest[demand_column]
                ),
                recommendation=(
                    "Check observed-day counts and source-data gaps."
                ),
            )
        )

    return insights


# =============================================================================
# SECTION 56 — TREND INSIGHT GENERATION
# =============================================================================

def di2_generate_trend_insights(
    trend_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """Convert calculated trend summaries into insight records."""

    insights = []

    for label, key in [
        ("Sales", "sales"),
        ("Redemption", "redemption"),
    ]:
        trend = trend_summary.get(
            key,
            {},
        )

        if trend.get("status") != "SUCCESS":
            insights.append(
                create_insight(
                    category="Demand Trend",
                    title=f"{label} Trend Undetermined",
                    description=(
                        f"There were not enough valid {label.lower()} "
                        "observations to estimate a descriptive linear trend."
                    ),
                    severity="LOW",
                    recommendation=(
                        "Review the amount of valid daily data before "
                        "interpreting the trend."
                    ),
                )
            )

            continue

        direction = trend.get(
            "direction",
            "Undetermined",
        )

        slope = trend.get(
            "slope_per_observation"
        )

        insights.append(
            create_insight(
                category="Demand Trend",
                title=f"Observed {label} Trend",
                description=(
                    f"The fitted descriptive linear trend is "
                    f"{direction.lower()}, with an estimated slope of "
                    f"{slope} demand units per ordered observation."
                ),
                severity="INFO",
                metric_name=f"{label}_Trend_Slope",
                metric_value=slope,
                recommendation=(
                    "Check rolling patterns, seasonal effects, and data "
                    "coverage before using this trend for planning."
                ),
            )
        )

    return insights


# =============================================================================
# SECTION 57 — SEASONALITY SUMMARY
# =============================================================================

def di2_build_seasonality_summary(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Build descriptive hourly, weekday, and month profiles.

    The output identifies patterns in historical records, not guaranteed
    recurring seasonal effects.
    """

    prepared = di2_prepare_dataframe(
        dataframe
    )

    hourly = di2_build_hourly_profile(
        prepared
    )

    weekday = di2_build_weekday_profile(
        prepared
    )

    time_block = di2_build_time_block_profile(
        prepared
    )

    monthly = di2_build_monthly_series(
        prepared
    )

    hourly_sales = hourly.sort_values(
        "Sales_Mean",
        ascending=False,
        na_position="last",
    ).head(5)

    hourly_redemption = hourly.sort_values(
        "Redemption_Mean",
        ascending=False,
        na_position="last",
    ).head(5)

    weekday_sales = weekday.sort_values(
        "Sales_Mean",
        ascending=False,
        na_position="last",
    ).head(7)

    weekday_redemption = weekday.sort_values(
        "Redemption_Mean",
        ascending=False,
        na_position="last",
    ).head(7)

    return {
        "module": DI2_MODULE_LABEL,
        "version": DI2_VERSION,
        "generated_at_utc": current_utc_timestamp(),
        "source_rows_after_timestamp_filter": int(
            len(prepared)
        ),
        "hourly_profile": records_from_dataframe(
            hourly
        ),
        "weekday_profile": records_from_dataframe(
            weekday
        ),
        "time_block_profile": records_from_dataframe(
            time_block
        ),
        "monthly_profile": records_from_dataframe(
            monthly
        ),
        "top_average_sales_hours": records_from_dataframe(
            hourly_sales
        ),
        "top_average_redemption_hours": records_from_dataframe(
            hourly_redemption
        ),
        "weekday_sales_ranking": records_from_dataframe(
            weekday_sales
        ),
        "weekday_redemption_ranking": records_from_dataframe(
            weekday_redemption
        ),
        "interpretation": (
            "Hourly and weekday averages are calculated over available "
            "records. They should be interpreted alongside coverage "
            "statistics, not as proof of a fixed operating schedule."
        ),
    }


# =============================================================================
# SECTION 58 — TREND REPORT
# =============================================================================

def di2_build_trend_report(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Combine daily, weekly, and monthly descriptive trend statistics."""

    prepared = di2_prepare_dataframe(
        dataframe
    )

    daily = di2_add_daily_change_metrics(
        di2_build_daily_series(prepared)
    )

    weekly = di2_add_weekly_change_metrics(
        di2_build_weekly_series(prepared)
    )

    monthly = di2_add_monthly_change_metrics(
        di2_build_monthly_series(prepared)
    )

    rolling_daily = di2_add_rolling_change_metrics(
        daily,
        demand_columns=(
            "Sales_Count",
            "Redemption_Count",
        ),
        windows=(3, 7, 14),
    )

    return {
        "module": DI2_MODULE_LABEL,
        "version": DI2_VERSION,
        "generated_at_utc": current_utc_timestamp(),
        "daily_trend": di2_summarize_daily_trends(
            daily
        ),
        "daily_observation_count": int(
            len(daily)
        ),
        "weekly_observation_count": int(
            len(weekly)
        ),
        "monthly_observation_count": int(
            len(monthly)
        ),
        "daily_latest_record": (
            records_from_dataframe(
                daily.tail(1)
            )[0]
            if not daily.empty
            else None
        ),
        "weekly_latest_record": (
            records_from_dataframe(
                weekly.tail(1)
            )[0]
            if not weekly.empty
            else None
        ),
        "monthly_latest_record": (
            records_from_dataframe(
                monthly.tail(1)
            )[0]
            if not monthly.empty
            else None
        ),
        "methodology": {
            "daily": (
                "Aggregates observed ticket counts by calendar date."
            ),
            "weekly": (
                "Aggregates daily totals into Monday-starting calendar weeks."
            ),
            "monthly": (
                "Aggregates daily totals by calendar year and month."
            ),
            "rolling": (
                "Uses rolling windows over ordered observed calendar-date "
                "rows. Missing calendar dates are not automatically filled "
                "with zero-demand rows."
            ),
        },
        "rolling_daily_columns": [
            column
            for column in rolling_daily.columns
            if "Rolling_" in column
            or "_Vs_Rolling_Mean_" in column
        ],
    }


# =============================================================================
# SECTION 59 — PEAK-HOUR AND WEEKDAY SUMMARY
# =============================================================================

def di2_build_peak_profile_summary(
    dataframe: pd.DataFrame,
    top_n: int = 5,
) -> dict[str, Any]:
    """Create ranked summaries for hours, weekdays, and time blocks."""

    if top_n < 1:
        raise ValueError(
            "top_n must be at least 1."
        )

    hourly = di2_build_hourly_profile(
        dataframe
    )

    weekday = di2_build_weekday_profile(
        dataframe
    )

    time_block = di2_build_time_block_profile(
        dataframe
    )

    def rank_table(
        table: pd.DataFrame,
        metric_column: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        if metric_column not in table.columns:
            return []

        ranked = table.sort_values(
            metric_column,
            ascending=False,
            na_position="last",
        ).head(limit)

        return records_from_dataframe(
            ranked
        )

    return {
        "top_n": top_n,
        "sales": {
            "highest_average_hours": rank_table(
                hourly,
                "Sales_Mean",
                top_n,
            ),
            "highest_total_hours": rank_table(
                hourly,
                "Sales_Count",
                top_n,
            ),
            "highest_average_weekdays": rank_table(
                weekday,
                "Sales_Mean",
                top_n,
            ),
            "highest_average_time_blocks": rank_table(
                time_block,
                "Sales_Mean",
                top_n,
            ),
        },
        "redemption": {
            "highest_average_hours": rank_table(
                hourly,
                "Redemption_Mean",
                top_n,
            ),
            "highest_total_hours": rank_table(
                hourly,
                "Redemption_Count",
                top_n,
            ),
            "highest_average_weekdays": rank_table(
                weekday,
                "Redemption_Mean",
                top_n,
            ),
            "highest_average_time_blocks": rank_table(
                time_block,
                "Redemption_Mean",
                top_n,
            ),
        },
        "warning": (
            "Total demand by hour and mean demand per observation answer "
            "different questions. Rankings should identify which measure "
            "is being compared."
        ),
    }


# =============================================================================
# SECTION 60 — PART 2 INSIGHT COLLECTION
# =============================================================================

def di2_generate_all_time_insights(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Generate descriptive insights from the Part 2 profiles."""

    prepared = di2_prepare_dataframe(
        dataframe
    )

    daily = di2_build_daily_series(
        prepared
    )

    hourly = di2_build_hourly_profile(
        prepared
    )

    weekday = di2_build_weekday_profile(
        prepared
    )

    monthly = di2_build_monthly_series(
        prepared
    )

    trend_summary = di2_summarize_daily_trends(
        daily
    )

    insights = []

    insights.extend(
        di2_generate_peak_hour_insights(
            hourly
        )
    )

    insights.extend(
        di2_generate_weekday_insights(
            weekday
        )
    )

    insights.extend(
        di2_generate_monthly_insights(
            monthly
        )
    )

    insights.extend(
        di2_generate_trend_insights(
            trend_summary
        )
    )

    comparison = di2_compare_weekday_weekend_means(
        di2_build_weekend_comparison(prepared)
    )

    for demand_label in [
        "sales",
        "redemption",
    ]:
        details = comparison.get(
            demand_label,
            {},
        )

        difference = details.get(
            "weekend_minus_weekday"
        )

        if difference is None:
            continue

        direction = (
            "higher"
            if difference > 0
            else "lower"
            if difference < 0
            else "similar"
        )

        insights.append(
            create_insight(
                category="Weekday/Weekend Comparison",
                title=(
                    f"Observed Weekend {demand_label.title()} "
                    "Compared With Weekdays"
                ),
                description=(
                    f"The average observed weekend {demand_label} count "
                    f"per record is {direction} than the weekday mean. "
                    f"The difference is {difference}."
                ),
                severity="INFO",
                metric_name=(
                    f"{demand_label.title()}_Weekend_Minus_Weekday"
                ),
                metric_value=difference,
                recommendation=(
                    "Review the number of distinct dates, missing intervals, "
                    "and calendar coverage before using this comparison "
                    "for operational planning."
                ),
            )
        )

    return insights


# =============================================================================
# SECTION 61 — CONSISTENT OUTPUT TABLES
# =============================================================================

def di2_build_output_tables(
    dataframe: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build all major Part 2 output tables from one prepared dataset."""

    prepared = di2_prepare_dataframe(
        dataframe
    )

    daily = di2_add_daily_change_metrics(
        di2_build_daily_series(prepared)
    )

    daily = di2_add_daily_coverage_flags(
        daily
    )

    weekly = di2_add_weekly_change_metrics(
        di2_build_weekly_series(prepared)
    )

    monthly = di2_add_monthly_change_metrics(
        di2_build_monthly_series(prepared)
    )

    hourly = di2_build_hourly_profile(
        prepared
    )

    weekday = di2_build_weekday_profile(
        prepared
    )

    weekend = di2_build_weekend_comparison(
        prepared
    )

    time_block = di2_build_time_block_profile(
        prepared
    )

    rolling = di2_add_rolling_change_metrics(
        daily,
        demand_columns=(
            "Sales_Count",
            "Redemption_Count",
        ),
        windows=(3, 7, 14),
    )

    unusual = di2_build_unusual_daily_periods(
        daily
    )

    sales_peaks = di2_build_daily_peak_table(
        daily,
        demand_name="Sales Count",
        top_n=20,
    )

    redemption_peaks = di2_build_daily_peak_table(
        daily,
        demand_name="Redemption Count",
        top_n=20,
    )

    sales_low = di2_build_daily_low_table(
        daily,
        demand_name="Sales Count",
        bottom_n=20,
    )

    redemption_low = di2_build_daily_low_table(
        daily,
        demand_name="Redemption Count",
        bottom_n=20,
    )

    hour_weekday_sales = di2_build_hour_weekday_matrix(
        prepared,
        demand_name="Sales Count",
    )

    hour_weekday_redemption = di2_build_hour_weekday_matrix(
        prepared,
        demand_name="Redemption Count",
    )

    month_hour_sales = di2_build_month_hour_matrix(
        prepared,
        demand_name="Sales Count",
    )

    month_hour_redemption = di2_build_month_hour_matrix(
        prepared,
        demand_name="Redemption Count",
    )

    weekday_hour = di2_build_weekday_hour_summary(
        prepared
    )

    observation_profile = di2_build_daily_observation_profile(
        prepared
    )

    coverage_profile = di2_build_profile_coverage(
        prepared
    )

    interval_flags = di2_build_interval_flags(
        prepared
    )

    daily_shares = di2_add_both_demand_shares(
        daily
    )

    weekly_shares = di2_add_both_demand_shares(
        weekly
    )

    monthly_shares = di2_add_both_demand_shares(
        monthly
    )

    return {
        "daily_trend": daily,
        "weekly_trend": weekly,
        "monthly_trend": monthly,
        "hourly_profile": hourly,
        "weekday_profile": weekday,
        "weekend_comparison": weekend,
        "time_block_profile": time_block,
        "rolling_daily_trend": rolling,
        "unusual_daily_periods": unusual,
        "sales_peak_days": sales_peaks,
        "redemption_peak_days": redemption_peaks,
        "sales_low_days": sales_low,
        "redemption_low_days": redemption_low,
        "weekday_hour_sales_matrix": hour_weekday_sales,
        "weekday_hour_redemption_matrix": hour_weekday_redemption,
        "month_hour_sales_matrix": month_hour_sales,
        "month_hour_redemption_matrix": month_hour_redemption,
        "weekday_hour_summary": weekday_hour,
        "daily_observation_profile": observation_profile,
        "profile_coverage": coverage_profile,
        "interval_flags": interval_flags,
        "daily_demand_shares": daily_shares,
        "weekly_demand_shares": weekly_shares,
        "monthly_demand_shares": monthly_shares,
    }


# =============================================================================
# SECTION 62 — EXPORT PART 2 TABLES
# =============================================================================

def di2_export_output_tables(
    tables: dict[str, pd.DataFrame],
) -> list[Path]:
    """Write the Part 2 DataFrame outputs to CSV files."""

    di2_ensure_directories()

    destinations = {
        "daily_trend": DI2_DAILY_TREND_PATH,
        "weekly_trend": DI2_WEEKLY_TREND_PATH,
        "monthly_trend": DI2_MONTHLY_TREND_PATH,
        "hourly_profile": DI2_HOURLY_PROFILE_PATH,
        "weekday_profile": DI2_WEEKDAY_PROFILE_PATH,
        "weekend_comparison": DI2_WEEKEND_COMPARISON_PATH,
        "rolling_daily_trend": DI2_ROLLING_TREND_PATH,
        "unusual_daily_periods": DI2_UNUSUAL_PERIODS_PATH,
        "sales_peak_days": DI2_PEAK_PERIODS_PATH,
        "sales_low_days": DI2_LOW_DEMAND_PERIODS_PATH,
    }

    written = []

    for table_name, destination in destinations.items():
        if table_name not in tables:
            LOGGER.warning(
                "Part 2 table is unavailable: %s",
                table_name,
            )
            continue

        dataframe = tables[table_name]

        written.append(
            write_csv_report(
                normalize_report_dataframe(dataframe),
                destination,
            )
        )

    additional_destinations = {
        "redemption_peak_days": (
            DI2_OUTPUT_DIRECTORY / "redemption_peak_days.csv"
        ),
        "redemption_low_days": (
            DI2_OUTPUT_DIRECTORY / "redemption_low_days.csv"
        ),
        "time_block_profile": (
            DI2_OUTPUT_DIRECTORY / "time_block_profile.csv"
        ),
        "weekday_hour_summary": (
            DI2_OUTPUT_DIRECTORY / "weekday_hour_summary.csv"
        ),
        "daily_observation_profile": (
            DI2_OUTPUT_DIRECTORY / "daily_observation_profile.csv"
        ),
        "profile_coverage": (
            DI2_OUTPUT_DIRECTORY / "profile_coverage.csv"
        ),
        "interval_flags": (
            DI2_OUTPUT_DIRECTORY / "interval_flags.csv"
        ),
        "daily_demand_shares": (
            DI2_OUTPUT_DIRECTORY / "daily_demand_shares.csv"
        ),
        "weekly_demand_shares": (
            DI2_OUTPUT_DIRECTORY / "weekly_demand_shares.csv"
        ),
        "monthly_demand_shares": (
            DI2_OUTPUT_DIRECTORY / "monthly_demand_shares.csv"
        ),
        "weekday_hour_sales_matrix": (
            DI2_OUTPUT_DIRECTORY / "weekday_hour_sales_matrix.csv"
        ),
        "weekday_hour_redemption_matrix": (
            DI2_OUTPUT_DIRECTORY / "weekday_hour_redemption_matrix.csv"
        ),
        "month_hour_sales_matrix": (
            DI2_OUTPUT_DIRECTORY / "month_hour_sales_matrix.csv"
        ),
        "month_hour_redemption_matrix": (
            DI2_OUTPUT_DIRECTORY / "month_hour_redemption_matrix.csv"
        ),
    }

    for table_name, destination in additional_destinations.items():
        if table_name not in tables:
            continue

        dataframe = tables[table_name]

        dataframe_to_write = dataframe.copy()

        if isinstance(
            dataframe_to_write.index,
            pd.MultiIndex,
        ):
            dataframe_to_write = (
                dataframe_to_write.reset_index()
            )

        elif (
            dataframe_to_write.index.name is not None
            and dataframe_to_write.index.name not in dataframe_to_write.columns
        ):
            dataframe_to_write = (
                dataframe_to_write.reset_index()
            )

        written.append(
            write_csv_report(
                normalize_report_dataframe(
                    dataframe_to_write
                ),
                destination,
            )
        )

    LOGGER.info(
        "Part 2 CSV export completed: %s files.",
        len(written),
    )

    return written


# =============================================================================
# SECTION 63 — EXPORT PART 2 JSON REPORTS
# =============================================================================

def di2_export_json_reports(
    dataframe: pd.DataFrame,
) -> list[Path]:
    """Write the primary Part 2 JSON reports."""

    di2_ensure_directories()

    prepared = di2_prepare_dataframe(
        dataframe
    )

    daily = di2_build_daily_series(
        prepared
    )

    hourly = di2_build_hourly_profile(
        prepared
    )

    weekday = di2_build_weekday_profile(
        prepared
    )

    weekend = di2_build_weekend_comparison(
        prepared
    )

    trend_report = di2_build_trend_report(
        prepared
    )

    seasonality_report = di2_build_seasonality_summary(
        prepared
    )

    peak_report = di2_build_peak_profile_summary(
        prepared
    )

    peak_period_report = di2_build_peak_period_summary(
        daily
    )

    weekend_report = di2_compare_weekday_weekend_means(
        weekend
    )

    distribution_report = di2_build_distribution_report(
        prepared
    )

    stability_report = di2_build_stability_report(
        prepared
    )

    coverage_report = di2_validate_chronology(
        prepared
    )

    zero_missing_report = di2_summarize_zero_and_missing_demand(
        prepared
    )

    report_items = [
        (
            DI2_TREND_REPORT_PATH,
            trend_report,
        ),
        (
            DI2_SEASONALITY_REPORT_PATH,
            seasonality_report,
        ),
        (
            DI2_PEAK_REPORT_PATH,
            {
                "peak_profiles": peak_report,
                "peak_daily_periods": peak_period_report,
            },
        ),
        (
            DI2_WEEKEND_REPORT_PATH,
            weekend_report,
        ),
        (
            DI2_REPORT_DIRECTORY / "distribution_summary.json",
            distribution_report,
        ),
        (
            DI2_REPORT_DIRECTORY / "demand_stability_summary.json",
            stability_report,
        ),
        (
            DI2_REPORT_DIRECTORY / "chronology_summary.json",
            coverage_report,
        ),
        (
            DI2_REPORT_DIRECTORY / "zero_missing_demand_summary.json",
            zero_missing_report,
        ),
    ]

    written = []

    for destination, report in report_items:
        written.append(
            write_json_report(
                report,
                destination,
            )
        )

    return written


# =============================================================================
# SECTION 64 — PART 2 INSIGHT EXPORT
# =============================================================================

def di2_export_insights(
    dataframe: pd.DataFrame,
) -> Path:
    """Export the time-pattern insight records as a CSV file."""

    di2_ensure_directories()

    insights = di2_generate_all_time_insights(
        dataframe
    )

    insights_table = build_demand_insights_table(
        insights
    )

    destination = (
        DI2_OUTPUT_DIRECTORY / "time_pattern_insights.csv"
    )

    return write_csv_report(
        insights_table,
        destination,
    )


# =============================================================================
# SECTION 65 — PART 2 EXECUTION SUMMARY
# =============================================================================

def di2_build_run_summary(
    dataframe: pd.DataFrame,
    written_files: Sequence[Path],
    started_at: float,
    status: str = "SUCCESS",
    errors: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Create an execution summary for Part 2."""

    prepared = di2_prepare_dataframe(
        dataframe
    )

    daily = di2_build_daily_series(
        prepared
    )

    return {
        "module": DI2_MODULE_LABEL,
        "version": DI2_VERSION,
        "status": status,
        "started_at_utc": None,
        "completed_at_utc": current_utc_timestamp(),
        "duration_seconds": round(
            time.perf_counter() - started_at,
            4,
        ),
        "source_rows_received": int(
            len(dataframe)
        ),
        "valid_timestamp_rows": int(
            len(prepared)
        ),
        "daily_periods": int(
            len(daily)
        ),
        "first_timestamp": prepared[
            "_analysis_timestamp"
        ].min(),
        "last_timestamp": prepared[
            "_analysis_timestamp"
        ].max(),
        "written_file_count": int(
            len(written_files)
        ),
        "written_files": [
            str(path)
            for path in written_files
        ],
        "errors": list(errors or []),
        "interpretation_warning": (
            "This module describes historical ticket counts. It does not "
            "directly estimate passenger queues, ferry capacity, or exact "
            "future boarding volume."
        ),
    }


# =============================================================================
# SECTION 66 — COMPLETE PART 2 ANALYSIS
# =============================================================================

def di2_analyze_time_patterns(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Run Part 2 analytics in memory without exporting files.

    This entry point can be called from a future dashboard or from
    the main demand-insights orchestration function.
    """

    prepared = di2_prepare_dataframe(
        dataframe
    )

    LOGGER.info(
        "Part 2 analysis started. Valid timestamp rows: %s",
        len(prepared),
    )

    tables = di2_build_output_tables(
        prepared
    )

    trend_report = di2_build_trend_report(
        prepared
    )

    seasonality_report = di2_build_seasonality_summary(
        prepared
    )

    peak_report = di2_build_peak_profile_summary(
        prepared
    )

    daily_peak_report = di2_build_peak_period_summary(
        tables["daily_trend"]
    )

    weekend_report = di2_compare_weekday_weekend_means(
        tables["weekend_comparison"]
    )

    distribution_report = di2_build_distribution_report(
        prepared
    )

    stability_report = di2_build_stability_report(
        prepared
    )

    chronology_report = di2_validate_chronology(
        prepared
    )

    zero_missing_report = di2_summarize_zero_and_missing_demand(
        prepared
    )

    insights = di2_generate_all_time_insights(
        prepared
    )

    result = {
        "module": DI2_MODULE_LABEL,
        "version": DI2_VERSION,
        "generated_at_utc": current_utc_timestamp(),
        "input_row_count": int(
            len(dataframe)
        ),
        "valid_timestamp_row_count": int(
            len(prepared)
        ),
        "trend_summary": trend_report,
        "seasonality_summary": seasonality_report,
        "peak_profile_summary": peak_report,
        "daily_peak_period_summary": daily_peak_report,
        "weekday_weekend_summary": weekend_report,
        "distribution_summary": distribution_report,
        "stability_summary": stability_report,
        "chronology_summary": chronology_report,
        "zero_missing_summary": zero_missing_report,
        "insights": insights,
        "tables": tables,
    }

    LOGGER.info(
        "Part 2 analysis completed. Insights generated: %s",
        len(insights),
    )

    return result


# =============================================================================
# SECTION 67 — EXPORT COMPLETE PART 2 ANALYSIS
# =============================================================================

def di2_export_analysis(
    analysis: dict[str, Any],
    source_dataframe: pd.DataFrame,
) -> list[Path]:
    """
    Export all tables and JSON reports from a Part 2 analysis result.

    The source DataFrame is supplied so reports can be regenerated without
    reloading the original CSV.
    """

    if not isinstance(analysis, dict):
        raise UnsupportedInputError(
            "analysis must be a dictionary returned by Part 2 analysis."
        )

    if "tables" not in analysis:
        raise SourceDataValidationError(
            "The supplied analysis dictionary has no tables section."
        )

    written_files = []

    written_files.extend(
        di2_export_output_tables(
            analysis["tables"]
        )
    )

    written_files.extend(
        di2_export_json_reports(
            source_dataframe
        )
    )

    written_files.append(
        di2_export_insights(
            source_dataframe
        )
    )

    run_summary = {
        key: value
        for key, value in analysis.items()
        if key != "tables"
    }

    run_summary["exported_file_count_before_run_summary"] = len(
        written_files
    )

    run_summary["exported_files"] = [
        str(path)
        for path in written_files
    ]

    written_files.append(
        write_json_report(
            run_summary,
            DI2_RUN_REPORT_PATH,
        )
    )

    return written_files


# =============================================================================
# SECTION 68 — MAIN PART 2 WORKFLOW
# =============================================================================

def di2_run_time_pattern_analysis(
    source_path: Path = SOURCE_DATA_PATH,
    export_reports: bool = True,
) -> dict[str, Any]:
    """
    Execute Part 2 against the configured source CSV.

    Parameters
    ----------
    source_path:
        Path to the original ticket-count CSV.

    export_reports:
        If True, export CSV and JSON reports.

    Returns
    -------
    dict
        Analysis results and output-file paths.
    """

    configure_logging()

    started = time.perf_counter()

    started_at_utc = current_utc_timestamp()

    di2_ensure_directories()

    LOGGER.info(
        "=" * 79
    )

    LOGGER.info(
        "Starting Part 2: %s",
        DI2_MODULE_LABEL,
    )

    errors = []

    written_files = []

    try:
        source_dataframe = load_demand_data(
            source_path
        )

        analysis = di2_analyze_time_patterns(
            source_dataframe
        )

        if export_reports:
            written_files = di2_export_analysis(
                analysis,
                source_dataframe,
            )

        summary = di2_build_run_summary(
            source_dataframe,
            written_files,
            started,
            status="SUCCESS",
        )

        summary["started_at_utc"] = started_at_utc

        summary["source_path"] = str(
            source_path
        )

        summary["export_enabled"] = bool(
            export_reports
        )

        write_json_report(
            summary,
            DI2_RUN_REPORT_PATH,
        )

        LOGGER.info(
            "Part 2 completed successfully in %.2f seconds.",
            summary["duration_seconds"],
        )

        return {
            "summary": summary,
            "analysis": analysis,
            "written_files": [
                str(path)
                for path in written_files
            ],
        }

    except Exception as error:
        errors.append(
            f"{type(error).__name__}: {error}"
        )

        LOGGER.exception(
            "Part 2 time-pattern analysis failed."
        )

        failure_summary = {
            "module": DI2_MODULE_LABEL,
            "version": DI2_VERSION,
            "status": "FAILED",
            "started_at_utc": started_at_utc,
            "completed_at_utc": current_utc_timestamp(),
            "duration_seconds": round(
                time.perf_counter() - started,
                4,
            ),
            "source_path": str(source_path),
            "errors": errors,
        }

        try:
            write_json_report(
                failure_summary,
                DI2_RUN_REPORT_PATH,
            )
        except Exception:
            LOGGER.exception(
                "Unable to write the Part 2 failure report."
            )

        raise


# =============================================================================
# SECTION 69 — PUBLIC PART 2 API
# =============================================================================

def get_di2_daily_trend(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Return daily trend data without writing reports."""

    return di2_add_daily_change_metrics(
        di2_build_daily_series(
            dataframe
        )
    )


def get_di2_weekly_trend(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Return weekly trend data without writing reports."""

    return di2_add_weekly_change_metrics(
        di2_build_weekly_series(
            dataframe
        )
    )


def get_di2_monthly_trend(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Return monthly trend data without writing reports."""

    return di2_add_monthly_change_metrics(
        di2_build_monthly_series(
            dataframe
        )
    )


def get_di2_hourly_profile(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Return the hourly demand profile."""

    return di2_build_hourly_profile(
        dataframe
    )


def get_di2_weekday_profile(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Return the weekday demand profile."""

    return di2_build_weekday_profile(
        dataframe
    )


def get_di2_weekend_comparison(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Return a summary comparing average weekday and weekend demand."""

    return di2_compare_weekday_weekend_means(
        di2_build_weekend_comparison(
            dataframe
        )
    )


def get_di2_seasonality_summary(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Return descriptive time-of-day and weekday profiles."""

    return di2_build_seasonality_summary(
        dataframe
    )


def get_di2_peak_period_summary(
    dataframe: pd.DataFrame,
    top_n: int = 20,
) -> dict[str, Any]:
    """Return high- and low-demand daily periods."""

    return di2_build_peak_period_summary(
        di2_build_daily_series(
            dataframe
        ),
        top_n=top_n,
    )


def get_di2_distribution_summary(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Return descriptive demand-distribution statistics."""

    return di2_build_distribution_report(
        dataframe
    )


def get_di2_unusual_periods(
    dataframe: pd.DataFrame,
    z_threshold: float = DI2_DEFAULT_UNUSUAL_Z_THRESHOLD,
) -> pd.DataFrame:
    """Return unusually high or low daily demand periods."""

    daily = di2_build_daily_series(
        dataframe
    )

    return di2_build_unusual_daily_periods(
        daily,
        z_threshold=z_threshold,
    )


def get_di2_profile_coverage(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Return profile coverage counts."""

    return di2_build_profile_coverage(
        dataframe
    )


def get_di2_rolling_trend(
    dataframe: pd.DataFrame,
    windows: Sequence[int] = (3, 7, 14),
) -> pd.DataFrame:
    """Return daily demand data with rolling statistics."""

    daily = di2_build_daily_series(
        dataframe
    )

    return di2_add_rolling_change_metrics(
        daily,
        windows=windows,
    )


def get_di2_chronology_summary(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Return chronology and timestamp-gap diagnostics."""

    return di2_validate_chronology(
        dataframe
    )


# =============================================================================
# SECTION 70 — PART 2 COMPLETION MARKER
# =============================================================================
#
# PART 2 ENDS HERE.
#
# IMPORTANT:
#   - Do not paste this section inside another function.
#   - Do not delete the Part 1 code above it.
#   - Do not run the module yet; the remaining planned sections must be
#     added and reviewed first.
#   - This section deliberately does not add another main() execution block.
#
# The next planned section can build business-facing recommendations,
# integrate forecast performance and uncertainty reports, and connect
# the generated insights to the Streamlit dashboard.
#
# =============================================================================
# >>> PART 2 END <<<
# =============================================================================


# ================================================================
# >>> START PART 2: ADVANCED TIME-BASED DEMAND INTELLIGENCE
# ================================================================
#
# FILE: demand_insights.py
# PART: 2
#
# PURPOSE:
#   1. Analyze demand by hour, weekday, month, and year.
#   2. Compare weekday and weekend demand.
#   3. Identify peak and low-demand periods.
#   4. Calculate rolling demand statistics.
#   5. Generate time-based operational insights.
#   6. Export analysis tables and reports.
#
# IMPORTANT:
#   These functions use the di2_ prefix to avoid conflicting
#   with functions already defined in Part 1.
#
# PASTE INSTRUCTION:
#   Append this entire block after the existing Part 1 code.
#   Do not paste this block inside another function.
#   Do not delete any existing Part 1 functions.
#
# NOTE:
#   This block defines additional functions only.
#   It does not automatically execute the Part 2 analysis.
#
# ================================================================

from typing import Any, Dict, List, Optional, Tuple


# ================================================================
# SECTION 1: CONFIGURATION
# ================================================================

DI2_VERSION = "1.0.0"

DI2_OUTPUT_ROOT = BASE_DIR / "outputs" / "demand_insights"

DI2_REPORT_ROOT = BASE_DIR / "outputs" / "reports" / "demand_insights"

DI2_HOURLY_FILE = "hourly_demand_profile.csv"

DI2_WEEKDAY_FILE = "weekday_demand_profile.csv"

DI2_MONTHLY_FILE = "monthly_demand_profile.csv"

DI2_YEARLY_FILE = "yearly_demand_profile.csv"

DI2_WEEKEND_FILE = "weekday_weekend_comparison.csv"

DI2_PEAK_FILE = "peak_demand_periods.csv"

DI2_ROLLING_FILE = "rolling_demand_statistics.csv"

DI2_INSIGHTS_FILE = "time_based_insights.json"

DI2_SUMMARY_FILE = "time_based_summary.json"

DI2_DEMAND_QUANTILES = (0.50, 0.75, 0.90, 0.95, 0.99)

DI2_ROLLING_WINDOWS = (4, 12, 24, 48, 96)

DI2_MINIMUM_GROUP_SIZE = 1

DI2_WEEKEND_DAYS = (5, 6)

DI2_WEEKDAY_NAMES = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)


# ================================================================
# SECTION 2: OUTPUT DIRECTORY MANAGEMENT
# ================================================================

def di2_create_output_directories() -> Dict[str, Path]:
    """
    Create output directories for Part 2.

    Returns
    -------
    dict
        Paths to the output and report directories.
    """
    DI2_OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    DI2_REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    return {
        "output_dir": DI2_OUTPUT_ROOT,
        "report_dir": DI2_REPORT_ROOT,
    }


# ================================================================
# SECTION 3: INPUT VALIDATION
# ================================================================

def di2_validate_dataframe(
    dataframe: pd.DataFrame,
    timestamp_column: str = "Timestamp",
    demand_column: str = "Sales Count",
) -> Tuple[bool, List[str]]:
    """
    Validate the dataframe before time-based analysis.

    The function checks:
    - Dataframe type.
    - Empty data.
    - Timestamp column.
    - Demand column.
    - Numeric demand values.
    """
    errors = []

    if not isinstance(dataframe, pd.DataFrame):
        return False, [
            "Input must be a pandas DataFrame."
        ]

    if dataframe.empty:
        errors.append(
            "The supplied dataframe is empty."
        )

    if timestamp_column not in dataframe.columns:
        errors.append(
            f"Missing timestamp column: {timestamp_column}"
        )

    if demand_column not in dataframe.columns:
        errors.append(
            f"Missing demand column: {demand_column}"
        )

    if errors:
        return False, errors

    if not pd.api.types.is_numeric_dtype(
        dataframe[demand_column]
    ):
        errors.append(
            f"Demand column '{demand_column}' must be numeric."
        )

    if not errors:
        valid_timestamp_count = pd.to_datetime(
            dataframe[timestamp_column],
            errors="coerce",
        ).notna().sum()

        if valid_timestamp_count == 0:
            errors.append(
                "No valid timestamps were found."
            )

    return len(errors) == 0, errors


# ================================================================
# SECTION 4: PREPARE A STANDARD TIME-SERIES DATAFRAME
# ================================================================

def di2_prepare_time_dataframe(
    dataframe: pd.DataFrame,
    timestamp_column: str = "Timestamp",
    demand_column: str = "Sales Count",
) -> pd.DataFrame:
    """
    Create a clean dataframe for time-based demand analysis.

    Original columns are preserved. Additional calendar columns
    are created for subsequent analysis.
    """
    valid, errors = di2_validate_dataframe(
        dataframe=dataframe,
        timestamp_column=timestamp_column,
        demand_column=demand_column,
    )

    if not valid:
        raise ValueError(
            "Time dataframe validation failed: "
            + "; ".join(errors)
        )

    result = dataframe.copy()

    result[timestamp_column] = pd.to_datetime(
        result[timestamp_column],
        errors="coerce",
    )

    result[demand_column] = pd.to_numeric(
        result[demand_column],
        errors="coerce",
    )

    result = result.dropna(
        subset=[
            timestamp_column,
            demand_column,
        ]
    ).copy()

    result = result.sort_values(
        timestamp_column
    ).reset_index(
        drop=True
    )

    result["DI2_Date"] = result[
        timestamp_column
    ].dt.date

    result["DI2_Year"] = result[
        timestamp_column
    ].dt.year

    result["DI2_Month"] = result[
        timestamp_column
    ].dt.month

    result["DI2_Month_Name"] = result[
        timestamp_column
    ].dt.month_name()

    result["DI2_Quarter"] = result[
        timestamp_column
    ].dt.quarter

    result["DI2_Day"] = result[
        timestamp_column
    ].dt.day

    result["DI2_Day_Of_Week"] = result[
        timestamp_column
    ].dt.dayofweek

    result["DI2_Day_Name"] = result[
        timestamp_column
    ].dt.day_name()

    result["DI2_Hour"] = result[
        timestamp_column
    ].dt.hour

    result["DI2_Minute"] = result[
        timestamp_column
    ].dt.minute

    result["DI2_Is_Weekend"] = (
        result["DI2_Day_Of_Week"].isin(
            DI2_WEEKEND_DAYS
        )
    )

    result["DI2_Is_Weekday"] = (
        ~result["DI2_Is_Weekend"]
    )

    result["DI2_Hour_Bucket"] = (
        result["DI2_Hour"].astype(str).str.zfill(2)
        + ":00"
    )

    result["DI2_Year_Month"] = (
        result[timestamp_column].dt.strftime(
            "%Y-%m"
        )
    )

    return result


# ================================================================
# SECTION 5: SAFE NUMERIC HELPERS
# ================================================================

def di2_safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Convert a value to a finite float.

    Missing values and non-finite values return the default.
    """
    try:
        converted = float(value)

        if not np.isfinite(converted):
            return default

        return converted

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return default


def di2_safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0,
) -> float:
    """
    Divide safely without raising a division-by-zero error.
    """
    numerator = di2_safe_float(
        numerator,
        default=default,
    )

    denominator = di2_safe_float(
        denominator,
        default=0.0,
    )

    if denominator == 0:
        return default

    return numerator / denominator


def di2_round_numeric_columns(
    dataframe: pd.DataFrame,
    decimals: int = 3,
) -> pd.DataFrame:
    """
    Round numeric columns for cleaner exported reports.
    """
    result = dataframe.copy()

    numeric_columns = result.select_dtypes(
        include=[np.number]
    ).columns

    result[numeric_columns] = result[
        numeric_columns
    ].round(decimals)

    return result


# ================================================================
# SECTION 6: GENERIC GROUPED DEMAND STATISTICS
# ================================================================

def di2_group_demand(
    dataframe: pd.DataFrame,
    group_columns: List[str],
    demand_column: str = "Sales Count",
) -> pd.DataFrame:
    """
    Calculate reusable demand statistics for one or more groups.

    Statistics include:
    - Number of records.
    - Total demand.
    - Average demand.
    - Median demand.
    - Standard deviation.
    - Minimum and maximum demand.
    - 75th, 90th, 95th and 99th percentiles.
    """
    if dataframe.empty:
        return pd.DataFrame()

    missing_columns = [
        column
        for column in group_columns + [demand_column]
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing grouping columns: "
            + ", ".join(missing_columns)
        )

    grouped = dataframe.groupby(
        group_columns,
        dropna=False,
        observed=True,
    )[demand_column]

    result = grouped.agg(
        Record_Count="count",
        Total_Demand="sum",
        Mean_Demand="mean",
        Median_Demand="median",
        Standard_Deviation="std",
        Minimum_Demand="min",
        Maximum_Demand="max",
    ).reset_index()

    percentile_tables = []

    for percentile in (
        0.75,
        0.90,
        0.95,
        0.99,
    ):
        percentile_name = (
            f"P{int(percentile * 100)}_Demand"
        )

        values = grouped.quantile(
            percentile
        ).rename(
            percentile_name
        ).reset_index()

        percentile_tables.append(
            values
        )

    for percentile_table in percentile_tables:
        result = result.merge(
            percentile_table,
            on=group_columns,
            how="left",
        )

    result["Standard_Deviation"] = result[
        "Standard_Deviation"
    ].fillna(0)

    result["Demand_Range"] = (
        result["Maximum_Demand"]
        - result["Minimum_Demand"]
    )

    result["Demand_Volatility_Percent"] = (
        result.apply(
            lambda row: di2_safe_divide(
                row["Standard_Deviation"] * 100,
                row["Mean_Demand"],
            ),
            axis=1,
        )
    )

    result["Mean_Median_Difference"] = (
        result["Mean_Demand"]
        - result["Median_Demand"]
    )

    result["Peak_To_Average_Ratio"] = (
        result.apply(
            lambda row: di2_safe_divide(
                row["Maximum_Demand"],
                row["Mean_Demand"],
            ),
            axis=1,
        )
    )

    return di2_round_numeric_columns(
        result
    )


# ================================================================
# SECTION 7: HOURLY DEMAND PROFILE
# ================================================================

def _modern_di2_build_hourly_profile(
    dataframe: pd.DataFrame,
    demand_column: str = "Sales Count",
) -> pd.DataFrame:
    """
    Analyze demand distribution across the 24 hours of the day.

    Important:
    The output describes observed ticket demand. It does not
    estimate passenger queues or ferry capacity.
    """
    if dataframe.empty:
        return pd.DataFrame()

    hourly = di2_group_demand(
        dataframe=dataframe,
        group_columns=[
            "DI2_Hour",
        ],
        demand_column=demand_column,
    )

    if hourly.empty:
        return hourly

    hourly["Hour_Label"] = (
        hourly["DI2_Hour"].astype(str).str.zfill(2)
        + ":00"
    )

    hourly = hourly.sort_values(
        "DI2_Hour"
    ).reset_index(
        drop=True
    )

    total_demand = di2_safe_float(
        hourly["Total_Demand"].sum()
    )

    hourly["Share_Of_Observed_Demand_Percent"] = (
        hourly["Total_Demand"].apply(
            lambda value: di2_safe_divide(
                value * 100,
                total_demand,
            )
        )
    )

    overall_mean = di2_safe_float(
        dataframe[demand_column].mean()
    )

    hourly["Mean_Demand_Vs_Overall"] = (
        hourly["Mean_Demand"] - overall_mean
    )

    hourly["Mean_Demand_Index"] = (
        hourly["Mean_Demand"].apply(
            lambda value: di2_safe_divide(
                value * 100,
                overall_mean,
            )
        )
    )

    return di2_round_numeric_columns(
        hourly
    )


# ================================================================
# SECTION 8: WEEKDAY DEMAND PROFILE
# ================================================================

def _modern_di2_build_weekday_profile(
    dataframe: pd.DataFrame,
    demand_column: str = "Sales Count",
) -> pd.DataFrame:
    """
    Compare observed demand across the seven weekdays.
    """
    if dataframe.empty:
        return pd.DataFrame()

    result = di2_group_demand(
        dataframe=dataframe,
        group_columns=[
            "DI2_Day_Of_Week",
            "DI2_Day_Name",
        ],
        demand_column=demand_column,
    )

    if result.empty:
        return result

    result = result.sort_values(
        "DI2_Day_Of_Week"
    ).reset_index(
        drop=True
    )

    overall_mean = di2_safe_float(
        dataframe[demand_column].mean()
    )

    result["Mean_Demand_Index"] = (
        result["Mean_Demand"].apply(
            lambda value: di2_safe_divide(
                value * 100,
                overall_mean,
            )
        )
    )

    result["Demand_Share_Percent"] = (
        result["Total_Demand"].apply(
            lambda value: di2_safe_divide(
                value * 100,
                dataframe[demand_column].sum(),
            )
        )
    )

    return di2_round_numeric_columns(
        result
    )


# ================================================================
# SECTION 9: HOUR-BY-WEEKDAY DEMAND PROFILE
# ================================================================

def di2_build_weekday_hour_matrix(
    dataframe: pd.DataFrame,
    demand_column: str = "Sales Count",
    statistic: str = "mean",
) -> pd.DataFrame:
    """
    Build a matrix with weekdays as rows and hours as columns.

    This matrix can later be displayed as a heatmap in Streamlit.
    """
    allowed_statistics = {
        "mean",
        "median",
        "sum",
        "max",
        "count",
    }

    if statistic not in allowed_statistics:
        raise ValueError(
            "Unsupported statistic. Choose one of: "
            + ", ".join(
                sorted(allowed_statistics)
            )
        )

    if dataframe.empty:
        return pd.DataFrame()

    matrix = pd.pivot_table(
        dataframe,
        index="DI2_Day_Name",
        columns="DI2_Hour",
        values=demand_column,
        aggfunc=statistic,
        observed=True,
    )

    matrix = matrix.reindex(
        index=list(DI2_WEEKDAY_NAMES)
    )

    matrix = matrix.reindex(
        columns=list(range(24))
    )

    matrix.columns = [
        f"Hour_{int(hour):02d}"
        for hour in matrix.columns
    ]

    matrix.index.name = "Weekday"

    return matrix.reset_index()


# ================================================================
# SECTION 10: WEEKDAY VS WEEKEND COMPARISON
# ================================================================

def _modern_di2_build_weekend_comparison(
    dataframe: pd.DataFrame,
    demand_column: str = "Sales Count",
) -> pd.DataFrame:
    """
    Compare observed demand on weekdays and weekends.

    Raw totals may be misleading if the dataset contains unequal
    numbers of weekdays and weekend days. The output therefore
    includes average demand per record and the number of dates.
    """
    if dataframe.empty:
        return pd.DataFrame()

    result = di2_group_demand(
        dataframe=dataframe,
        group_columns=[
            "DI2_Is_Weekend",
        ],
        demand_column=demand_column,
    )

    if result.empty:
        return result

    result["Day_Type"] = result[
        "DI2_Is_Weekend"
    ].map(
        {
            True: "Weekend",
            False: "Weekday",
        }
    )

    date_counts = dataframe.groupby(
        "DI2_Is_Weekend"
    )["DI2_Date"].nunique()

    result["Unique_Dates"] = result[
        "DI2_Is_Weekend"
    ].map(
        date_counts
    ).fillna(0).astype(int)

    result["Average_Records_Per_Date"] = (
        result.apply(
            lambda row: di2_safe_divide(
                row["Record_Count"],
                row["Unique_Dates"],
            ),
            axis=1,
        )
    )

    result["Average_Daily_Demand_Proxy"] = (
        result.apply(
            lambda row: di2_safe_divide(
                row["Total_Demand"],
                row["Unique_Dates"],
            ),
            axis=1,
        )
    )

    return di2_round_numeric_columns(
        result
    )


# ================================================================
# SECTION 11: MONTHLY DEMAND PROFILE
# ================================================================

def di2_build_monthly_profile(
    dataframe: pd.DataFrame,
    demand_column: str = "Sales Count",
) -> pd.DataFrame:
    """
    Calculate demand statistics by calendar month number.

    January from different years is grouped together. Use the
    year-month profile for chronological monthly analysis.
    """
    if dataframe.empty:
        return pd.DataFrame()

    result = di2_group_demand(
        dataframe=dataframe,
        group_columns=[
            "DI2_Month",
            "DI2_Month_Name",
        ],
        demand_column=demand_column,
    )

    if result.empty:
        return result

    result = result.sort_values(
        "DI2_Month"
    ).reset_index(
        drop=True
    )

    result["Mean_Demand_Index"] = (
        result["Mean_Demand"].apply(
            lambda value: di2_safe_divide(
                value * 100,
                dataframe[demand_column].mean(),
            )
        )
    )

    return di2_round_numeric_columns(
        result
    )


# ================================================================
# SECTION 12: YEAR-MONTH CHRONOLOGICAL PROFILE
# ================================================================

def di2_build_year_month_profile(
    dataframe: pd.DataFrame,
    demand_column: str = "Sales Count",
) -> pd.DataFrame:
    """
    Build chronological statistics for each year-month period.
    """
    if dataframe.empty:
        return pd.DataFrame()

    result = di2_group_demand(
        dataframe=dataframe,
        group_columns=[
            "DI2_Year_Month",
        ],
        demand_column=demand_column,
    )

    if result.empty:
        return result

    result["Period_Start"] = pd.to_datetime(
        result["DI2_Year_Month"] + "-01",
        errors="coerce",
    )

    result = result.sort_values(
        "Period_Start"
    ).reset_index(
        drop=True
    )

    result["Previous_Month_Demand"] = (
        result["Total_Demand"].shift(1)
    )

    result["Month_Over_Month_Change"] = (
        result["Total_Demand"]
        - result["Previous_Month_Demand"]
    )

    result["Month_Over_Month_Change_Percent"] = (
        result.apply(
            lambda row: di2_safe_divide(
                row["Month_Over_Month_Change"] * 100,
                row["Previous_Month_Demand"],
            ),
            axis=1,
        )
    )

    result["Previous_Month_Mean"] = (
        result["Mean_Demand"].shift(1)
    )

    result["Mean_Demand_Change"] = (
        result["Mean_Demand"]
        - result["Previous_Month_Mean"]
    )

    return di2_round_numeric_columns(
        result
    )


# ================================================================
# SECTION 13: YEARLY DEMAND PROFILE
# ================================================================

def di2_build_yearly_profile(
    dataframe: pd.DataFrame,
    demand_column: str = "Sales Count",
) -> pd.DataFrame:
    """
    Compare observed demand across calendar years.

    Caution:
    A partial year must not be compared directly with a complete
    year using total demand alone.
    """
    if dataframe.empty:
        return pd.DataFrame()

    result = di2_group_demand(
        dataframe=dataframe,
        group_columns=[
            "DI2_Year",
        ],
        demand_column=demand_column,
    )

    if result.empty:
        return result

    result["Observed_Dates"] = (
        dataframe.groupby(
            "DI2_Year"
        )["DI2_Date"].nunique().values
    )

    result["Average_Demand_Per_Observed_Date"] = (
        result.apply(
            lambda row: di2_safe_divide(
                row["Total_Demand"],
                row["Observed_Dates"],
            ),
            axis=1,
        )
    )

    result["Year_Is_Partial"] = (
        result["Observed_Dates"] < 365
    )

    result["Previous_Year_Total"] = (
        result["Total_Demand"].shift(1)
    )

    result["Year_Over_Year_Change"] = (
        result["Total_Demand"]
        - result["Previous_Year_Total"]
    )

    result["Year_Over_Year_Change_Percent"] = (
        result.apply(
            lambda row: di2_safe_divide(
                row["Year_Over_Year_Change"] * 100,
                row["Previous_Year_Total"],
            ),
            axis=1,
        )
    )

    return di2_round_numeric_columns(
        result
    )


# ================================================================
# SECTION 14: DAILY DEMAND PROFILE
# ================================================================

def di2_build_daily_profile(
    dataframe: pd.DataFrame,
    demand_column: str = "Sales Count",
) -> pd.DataFrame:
    """
    Aggregate observed records into daily demand totals.

    The result includes daily averages and record counts so
    incomplete observation days can be identified.
    """
    if dataframe.empty:
        return pd.DataFrame()

    daily = dataframe.groupby(
        "DI2_Date",
        dropna=False,
    ).agg(
        Daily_Demand=(
            demand_column,
            "sum",
        ),
        Record_Count=(
            demand_column,
            "count",
        ),
        Average_Record_Demand=(
            demand_column,
            "mean",
        ),
        Maximum_Record_Demand=(
            demand_column,
            "max",
        ),
        Minimum_Record_Demand=(
            demand_column,
            "min",
        ),
    ).reset_index()

    daily["Date"] = pd.to_datetime(
        daily["DI2_Date"].astype(str),
        errors="coerce",
    )

    daily["Weekday"] = daily[
        "Date"
    ].dt.day_name()

    daily["Year"] = daily[
        "Date"
    ].dt.year

    daily["Month"] = daily[
        "Date"
    ].dt.month

    daily["Is_Weekend"] = (
        daily["Date"].dt.dayofweek.isin(
            DI2_WEEKEND_DAYS
        )
    )

    daily = daily.sort_values(
        "Date"
    ).reset_index(
        drop=True
    )

    daily["Previous_Day_Demand"] = (
        daily["Daily_Demand"].shift(1)
    )

    daily["Daily_Change"] = (
        daily["Daily_Demand"]
        - daily["Previous_Day_Demand"]
    )

    daily["Daily_Change_Percent"] = (
        daily.apply(
            lambda row: di2_safe_divide(
                row["Daily_Change"] * 100,
                row["Previous_Day_Demand"],
            ),
            axis=1,
        )
    )

    return di2_round_numeric_columns(
        daily
    )


# ================================================================
# SECTION 15: PEAK-DEMAND PERIOD IDENTIFICATION
# ================================================================

def di2_identify_peak_periods(
    dataframe: pd.DataFrame,
    demand_column: str = "Sales Count",
    quantile: float = 0.90,
) -> pd.DataFrame:
    """
    Identify high-demand observations using a percentile threshold.

    A percentile threshold is descriptive, not a guarantee that
    a ferry or terminal will have sufficient capacity.
    """
    if dataframe.empty:
        return pd.DataFrame()

    if not 0 < quantile < 1:
        raise ValueError(
            "quantile must be between 0 and 1."
        )

    result = dataframe.copy()

    threshold = di2_safe_float(
        result[demand_column].quantile(
            quantile
        )
    )

    result["DI2_Peak_Threshold"] = threshold

    result["DI2_Is_Peak_Demand"] = (
        result[demand_column] >= threshold
    )

    result["DI2_Demand_Above_Threshold"] = (
        result[demand_column] - threshold
    ).clip(
        lower=0
    )

    result["DI2_Peak_Quantile"] = quantile

    peak = result.loc[
        result["DI2_Is_Peak_Demand"]
    ].copy()

    if peak.empty:
        return peak

    selected_columns = [
        "Timestamp",
        demand_column,
        "DI2_Year",
        "DI2_Month",
        "DI2_Day_Name",
        "DI2_Hour",
        "DI2_Is_Weekend",
        "DI2_Peak_Threshold",
        "DI2_Demand_Above_Threshold",
        "DI2_Peak_Quantile",
    ]

    selected_columns = [
        column
        for column in selected_columns
        if column in peak.columns
    ]

    peak = peak[selected_columns]

    peak = peak.sort_values(
        demand_column,
        ascending=False,
    ).reset_index(
        drop=True
    )

    return di2_round_numeric_columns(
        peak
    )


# ================================================================
# SECTION 16: DEMAND QUANTILE SUMMARY
# ================================================================

def di2_build_quantile_summary(
    dataframe: pd.DataFrame,
    demand_column: str = "Sales Count",
) -> pd.DataFrame:
    """
    Calculate selected quantiles for the observed demand values.
    """
    if dataframe.empty:
        return pd.DataFrame(
            columns=[
                "Quantile",
                "Percentile",
                "Demand_Value",
            ]
        )

    rows = []

    for quantile in DI2_DEMAND_QUANTILES:
        value = dataframe[
            demand_column
        ].quantile(
            quantile
        )

        rows.append(
            {
                "Quantile": quantile,
                "Percentile": int(
                    quantile * 100
                ),
                "Demand_Value": value,
            }
        )

    return pd.DataFrame(
        rows
    )


# ================================================================
# SECTION 17: ROLLING DEMAND STATISTICS
# ================================================================

def di2_build_rolling_statistics(
    dataframe: pd.DataFrame,
    demand_column: str = "Sales Count",
    windows: Tuple[int, ...] = DI2_ROLLING_WINDOWS,
) -> pd.DataFrame:
    """
    Calculate rolling statistics over ordered observations.

    Window sizes represent numbers of records, not fixed durations.

    Because timestamps can have gaps, a rolling window of 4 rows
    must not automatically be interpreted as exactly one hour.
    """
    if dataframe.empty:
        return pd.DataFrame()

    result = dataframe[
        [
            "Timestamp",
            demand_column,
        ]
    ].copy()

    result["Timestamp"] = pd.to_datetime(
        result["Timestamp"],
        errors="coerce",
    )

    result[demand_column] = pd.to_numeric(
        result[demand_column],
        errors="coerce",
    )

    result = result.dropna(
        subset=[
            "Timestamp",
            demand_column,
        ]
    ).sort_values(
        "Timestamp"
    ).reset_index(
        drop=True
    )

    for window in windows:
        if window < 1:
            continue

        rolling = result[
            demand_column
        ].rolling(
            window=window,
            min_periods=1,
        )

        result[
            f"Rolling_Mean_{window}"
        ] = rolling.mean()

        result[
            f"Rolling_Median_{window}"
        ] = rolling.median()

        result[
            f"Rolling_Std_{window}"
        ] = rolling.std().fillna(0)

        result[
            f"Rolling_Min_{window}"
        ] = rolling.min()

        result[
            f"Rolling_Max_{window}"
        ] = rolling.max()

    return di2_round_numeric_columns(
        result
    )


# ================================================================
# SECTION 18: ROLLING DEMAND CHANGE
# ================================================================

def _modern_di2_add_rolling_change_metrics(
    dataframe: pd.DataFrame,
    demand_column: str = "Sales Count",
    window: int = 12,
) -> pd.DataFrame:
    """
    Add rolling averages and rolling change metrics.

    This function does not make future predictions. It summarizes
    the recent observed values.
    """
    if dataframe.empty:
        return dataframe.copy()

    if window < 1:
        raise ValueError(
            "window must be at least 1."
        )

    result = dataframe.copy()

    values = pd.to_numeric(
        result[demand_column],
        errors="coerce",
    )

    result["DI2_Rolling_Mean"] = (
        values.rolling(
            window=window,
            min_periods=1,
        ).mean()
    )

    result["DI2_Rolling_Std"] = (
        values.rolling(
            window=window,
            min_periods=1,
        ).std().fillna(0)
    )

    result["DI2_Rolling_Upper_Band"] = (
        result["DI2_Rolling_Mean"]
        + 2 * result["DI2_Rolling_Std"]
    )

    result["DI2_Rolling_Lower_Band"] = (
        result["DI2_Rolling_Mean"]
        - 2 * result["DI2_Rolling_Std"]
    ).clip(
        lower=0
    )

    result["DI2_Difference_From_Rolling_Mean"] = (
        values - result["DI2_Rolling_Mean"]
    )

    result["DI2_Relative_To_Rolling_Mean_Percent"] = (
        result.apply(
            lambda row: di2_safe_divide(
                row["DI2_Difference_From_Rolling_Mean"]
                * 100,
                row["DI2_Rolling_Mean"],
            ),
            axis=1,
        )
    )

    return di2_round_numeric_columns(
        result
    )


# ================================================================
# SECTION 19: HIGH-DEMAND CLASSIFICATION
# ================================================================

def di2_classify_demand_levels(
    dataframe: pd.DataFrame,
    demand_column: str = "Sales Count",
) -> pd.DataFrame:
    """
    Classify observed demand using dataset percentiles.

    Classification:
    - Low: below the 50th percentile.
    - Normal: 50th to below 75th percentile.
    - Elevated: 75th to below 90th percentile.
    - High: 90th to below 95th percentile.
    - Very High: at or above the 95th percentile.

    These labels describe relative demand, not operational safety.
    """
    if dataframe.empty:
        return dataframe.copy()

    result = dataframe.copy()

    values = pd.to_numeric(
        result[demand_column],
        errors="coerce",
    )

    q50 = values.quantile(0.50)
    q75 = values.quantile(0.75)
    q90 = values.quantile(0.90)
    q95 = values.quantile(0.95)

    conditions = [
        values < q50,
        (values >= q50) & (values < q75),
        (values >= q75) & (values < q90),
        (values >= q90) & (values < q95),
        values >= q95,
    ]

    labels = [
        "Low",
        "Normal",
        "Elevated",
        "High",
        "Very High",
    ]

    result["DI2_Demand_Level"] = np.select(
        conditions,
        labels,
        default="Unknown",
    )

    result["DI2_Demand_Q50"] = q50
    result["DI2_Demand_Q75"] = q75
    result["DI2_Demand_Q90"] = q90
    result["DI2_Demand_Q95"] = q95

    return result


# ================================================================
# SECTION 20: DEMAND VARIABILITY BY HOUR
# ================================================================

def di2_build_hourly_variability(
    dataframe: pd.DataFrame,
    demand_column: str = "Sales Count",
) -> pd.DataFrame:
    """
    Identify hours with high variation in observed demand.
    """
    if dataframe.empty:
        return pd.DataFrame()

    result = di2_group_demand(
        dataframe=dataframe,
        group_columns=[
            "DI2_Hour",
        ],
        demand_column=demand_column,
    )

    if result.empty:
        return result

    result["Hour_Label"] = (
        result["DI2_Hour"].astype(str).str.zfill(2)
        + ":00"
    )

    result["Variability_Rank"] = (
        result["Standard_Deviation"].rank(
            method="dense",
            ascending=False,
        ).astype(int)
    )

    result["Volatility_Rank"] = (
        result["Demand_Volatility_Percent"].rank(
            method="dense",
            ascending=False,
        ).astype(int)
    )

    return result.sort_values(
        "Variability_Rank"
    ).reset_index(
        drop=True
    )


# ================================================================
# SECTION 21: PEAK HOURS BY WEEKDAY
# ================================================================

def di2_build_peak_hours_by_weekday(
    dataframe: pd.DataFrame,
    demand_column: str = "Sales Count",
) -> pd.DataFrame:
    """
    Rank hours within each weekday by average observed demand.
    """
    if dataframe.empty:
        return pd.DataFrame()

    result = di2_group_demand(
        dataframe=dataframe,
        group_columns=[
            "DI2_Day_Of_Week",
            "DI2_Day_Name",
            "DI2_Hour",
        ],
        demand_column=demand_column,
    )

    if result.empty:
        return result

    result["Within_Weekday_Rank"] = (
        result.groupby(
            "DI2_Day_Name"
        )["Mean_Demand"].rank(
            method="dense",
            ascending=False,
        ).astype(int)
    )

    result["Hour_Label"] = (
        result["DI2_Hour"].astype(str).str.zfill(2)
        + ":00"
    )

    return result.sort_values(
        [
            "DI2_Day_Of_Week",
            "Within_Weekday_Rank",
        ]
    ).reset_index(
        drop=True
    )


# ================================================================
# SECTION 22: DATE-LEVEL PEAK ANALYSIS
# ================================================================

def di2_build_peak_dates(
    daily_dataframe: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Return the dates with the highest observed daily demand.
    """
    if daily_dataframe.empty:
        return pd.DataFrame()

    if top_n < 1:
        raise ValueError(
            "top_n must be at least 1."
        )

    return daily_dataframe.nlargest(
        top_n,
        "Daily_Demand",
    ).reset_index(
        drop=True
    )


# ================================================================
# SECTION 23: LOW-DEMAND DATE ANALYSIS
# ================================================================

def di2_build_low_demand_dates(
    daily_dataframe: pd.DataFrame,
    bottom_n: int = 20,
) -> pd.DataFrame:
    """
    Return dates with the lowest observed daily demand.

    Low observed demand may reflect incomplete data collection,
    closures, seasonality, or genuinely low ticket activity.
    """
    if daily_dataframe.empty:
        return pd.DataFrame()

    if bottom_n < 1:
        raise ValueError(
            "bottom_n must be at least 1."
        )

    return daily_dataframe.nsmallest(
        bottom_n,
        "Daily_Demand",
    ).reset_index(
        drop=True
    )


# ================================================================
# SECTION 24: DEMAND TREND DIRECTION
# ================================================================

def di2_estimate_trend_direction(
    values: pd.Series,
) -> Dict[str, Any]:
    """
    Estimate the direction of a sequence using a linear slope.

    This is a descriptive trend estimate. It is not a forecasting
    model and does not prove that the trend will continue.
    """
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if len(numeric) < 2:
        return {
            "observations": int(len(numeric)),
            "slope": 0.0,
            "direction": "Insufficient data",
        }

    x = np.arange(
        len(numeric),
        dtype=float,
    )

    y = numeric.to_numpy(
        dtype=float
    )

    slope = np.polyfit(
        x,
        y,
        1,
    )[0]

    magnitude = abs(
        di2_safe_float(slope)
    )

    if magnitude < 1e-9:
        direction = "Approximately stable"

    elif slope > 0:
        direction = "Increasing"

    else:
        direction = "Decreasing"

    return {
        "observations": int(len(numeric)),
        "slope": di2_safe_float(slope),
        "direction": direction,
    }


# ================================================================
# SECTION 25: DAILY TREND SUMMARY
# ================================================================

def di2_build_daily_trend_summary(
    daily_dataframe: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Summarize the direction and spread of daily demand.
    """
    if daily_dataframe.empty:
        return {
            "status": "No daily records available.",
        }

    values = daily_dataframe[
        "Daily_Demand"
    ]

    trend = di2_estimate_trend_direction(
        values
    )

    return {
        "status": "success",
        "observed_days": int(
            daily_dataframe["Date"].nunique()
        ),
        "total_observed_demand": di2_safe_float(
            values.sum()
        ),
        "average_daily_demand": di2_safe_float(
            values.mean()
        ),
        "median_daily_demand": di2_safe_float(
            values.median()
        ),
        "minimum_daily_demand": di2_safe_float(
            values.min()
        ),
        "maximum_daily_demand": di2_safe_float(
            values.max()
        ),
        "daily_demand_std": di2_safe_float(
            values.std()
        ),
        "trend_direction": trend["direction"],
        "trend_slope_per_observation": trend["slope"],
    }


# ================================================================
# SECTION 26: TIME-BASED INSIGHT GENERATION
# ================================================================

def di2_generate_time_insights(
    dataframe: pd.DataFrame,
    hourly_profile: pd.DataFrame,
    weekday_profile: pd.DataFrame,
    monthly_profile: pd.DataFrame,
    weekend_profile: pd.DataFrame,
    daily_profile: pd.DataFrame,
    demand_column: str = "Sales Count",
) -> List[Dict[str, Any]]:
    """
    Convert selected time-based results into readable insights.

    Each insight contains:
    - Category.
    - Title.
    - Observation.
    - Evidence.
    - Interpretation.
    - Limitation.
    """
    insights = []

    if dataframe.empty:
        return insights

    if not hourly_profile.empty:
        highest_hour = hourly_profile.loc[
            hourly_profile["Mean_Demand"].idxmax()
        ]

        lowest_hour = hourly_profile.loc[
            hourly_profile["Mean_Demand"].idxmin()
        ]

        insights.append(
            {
                "category": "Hourly Pattern",
                "title": "Highest average demand hour",
                "observation": (
                    f"The highest average observed demand occurs "
                    f"around {highest_hour['Hour_Label']}."
                ),
                "evidence": {
                    "hour": int(
                        highest_hour["DI2_Hour"]
                    ),
                    "mean_demand": di2_safe_float(
                        highest_hour["Mean_Demand"]
                    ),
                },
                "interpretation": (
                    "This hour may deserve closer monitoring "
                    "when planning around observed ticket demand."
                ),
                "limitation": (
                    "Ticket sales are not a direct measurement "
                    "of passengers waiting or boarding."
                ),
            }
        )

        insights.append(
            {
                "category": "Hourly Pattern",
                "title": "Lowest average demand hour",
                "observation": (
                    f"The lowest average observed demand occurs "
                    f"around {lowest_hour['Hour_Label']}."
                ),
                "evidence": {
                    "hour": int(
                        lowest_hour["DI2_Hour"]
                    ),
                    "mean_demand": di2_safe_float(
                        lowest_hour["Mean_Demand"]
                    ),
                },
                "interpretation": (
                    "This period has lower average recorded "
                    "ticket demand than the other observed hours."
                ),
                "limitation": (
                    "Missing records and timestamp gaps may "
                    "affect the observed profile."
                ),
            }
        )

    if not weekday_profile.empty:
        highest_day = weekday_profile.loc[
            weekday_profile["Mean_Demand"].idxmax()
        ]

        insights.append(
            {
                "category": "Weekly Pattern",
                "title": "Highest average demand weekday",
                "observation": (
                    f"{highest_day['DI2_Day_Name']} has the "
                    f"highest average demand per observed record."
                ),
                "evidence": {
                    "weekday": str(
                        highest_day["DI2_Day_Name"]
                    ),
                    "mean_demand": di2_safe_float(
                        highest_day["Mean_Demand"]
                    ),
                },
                "interpretation": (
                    "The weekday pattern can help identify "
                    "recurring differences in ticket activity."
                ),
                "limitation": (
                    "Unequal sampling and missing time intervals "
                    "can affect comparisons."
                ),
            }
        )

    if not monthly_profile.empty:
        highest_month = monthly_profile.loc[
            monthly_profile["Mean_Demand"].idxmax()
        ]

        insights.append(
            {
                "category": "Monthly Pattern",
                "title": "Highest average demand month",
                "observation": (
                    f"{highest_month['DI2_Month_Name']} has the "
                    f"highest average observed demand per record."
                ),
                "evidence": {
                    "month": str(
                        highest_month["DI2_Month_Name"]
                    ),
                    "mean_demand": di2_safe_float(
                        highest_month["Mean_Demand"]
                    ),
                },
                "interpretation": (
                    "The pattern may indicate recurring monthly "
                    "variation in ticket activity."
                ),
                "limitation": (
                    "This result combines the same calendar month "
                    "across all years in the dataset."
                ),
            }
        )

    if not weekend_profile.empty:
        weekday_rows = weekend_profile.loc[
            weekend_profile["Day_Type"] == "Weekday"
        ]

        weekend_rows = weekend_profile.loc[
            weekend_profile["Day_Type"] == "Weekend"
        ]

        if (
            not weekday_rows.empty
            and not weekend_rows.empty
        ):
            weekday_mean = di2_safe_float(
                weekday_rows.iloc[0]["Mean_Demand"]
            )

            weekend_mean = di2_safe_float(
                weekend_rows.iloc[0]["Mean_Demand"]
            )

            difference = (
                weekend_mean - weekday_mean
            )

            insights.append(
                {
                    "category": "Weekend Comparison",
                    "title": "Weekday versus weekend demand",
                    "observation": (
                        "Weekend mean observed demand is "
                        f"{'higher' if difference > 0 else 'lower' if difference < 0 else 'similar to'} "
                        "weekday mean observed demand."
                    ),
                    "evidence": {
                        "weekday_mean": weekday_mean,
                        "weekend_mean": weekend_mean,
                        "difference": difference,
                    },
                    "interpretation": (
                        "This is a comparison of average demand "
                        "per recorded observation."
                    ),
                    "limitation": (
                        "A record-level comparison is not the "
                        "same as a daily-total comparison."
                    ),
                }
            )

    if not daily_profile.empty:
        daily_summary = di2_build_daily_trend_summary(
            daily_profile
        )

        insights.append(
            {
                "category": "Daily Trend",
                "title": "Observed daily demand trend",
                "observation": (
                    "The estimated direction of daily demand is "
                    f"{daily_summary['trend_direction']}."
                ),
                "evidence": {
                    "trend_slope": daily_summary[
                        "trend_slope_per_observation"
                    ],
                    "observed_days": daily_summary[
                        "observed_days"
                    ],
                },
                "interpretation": (
                    "The slope summarizes the observed historical "
                    "sequence of daily totals."
                ),
                "limitation": (
                    "The historical slope should not be treated "
                    "as a future demand forecast."
                ),
            }
        )

    return insights


# ================================================================
# SECTION 27: BUILD THE TIME-BASED ANALYSIS PACKAGE
# ================================================================

def di2_build_time_analysis(
    dataframe: pd.DataFrame,
    timestamp_column: str = "Timestamp",
    demand_column: str = "Sales Count",
) -> Dict[str, Any]:
    """
    Build all core time-based analysis tables.

    The input dataframe may be the original dataset or a
    dataframe already loaded by Part 1.

    Returns
    -------
    dict
        Prepared dataframe, analysis tables, insights, and summary.
    """
    prepared = di2_prepare_time_dataframe(
        dataframe=dataframe,
        timestamp_column=timestamp_column,
        demand_column=demand_column,
    )

    if prepared.empty:
        raise ValueError(
            "No valid observations remain after preparation."
        )

    hourly_profile = di2_build_hourly_profile(
        prepared,
        demand_column=demand_column,
    )

    weekday_profile = di2_build_weekday_profile(
        prepared,
        demand_column=demand_column,
    )

    weekday_hour_matrix = di2_build_weekday_hour_matrix(
        prepared,
        demand_column=demand_column,
        statistic="mean",
    )

    weekend_profile = di2_build_weekend_comparison(
        prepared,
        demand_column=demand_column,
    )

    monthly_profile = di2_build_monthly_profile(
        prepared,
        demand_column=demand_column,
    )

    year_month_profile = di2_build_year_month_profile(
        prepared,
        demand_column=demand_column,
    )

    yearly_profile = di2_build_yearly_profile(
        prepared,
        demand_column=demand_column,
    )

    daily_profile = di2_build_daily_profile(
        prepared.rename(
            columns={
                timestamp_column: "Timestamp",
            }
        ),
        demand_column=demand_column,
    )

    hourly_variability = di2_build_hourly_variability(
        prepared,
        demand_column=demand_column,
    )

    peak_hours_by_weekday = (
        di2_build_peak_hours_by_weekday(
            prepared,
            demand_column=demand_column,
        )
    )

    peak_periods = di2_identify_peak_periods(
        prepared,
        demand_column=demand_column,
        quantile=0.90,
    )

    quantile_summary = di2_build_quantile_summary(
        prepared,
        demand_column=demand_column,
    )

    rolling_statistics = di2_build_rolling_statistics(
        prepared.rename(
            columns={
                timestamp_column: "Timestamp",
            }
        ),
        demand_column=demand_column,
    )

    classified_demand = di2_classify_demand_levels(
        prepared,
        demand_column=demand_column,
    )

    insights = di2_generate_time_insights(
        dataframe=prepared,
        hourly_profile=hourly_profile,
        weekday_profile=weekday_profile,
        monthly_profile=monthly_profile,
        weekend_profile=weekend_profile,
        daily_profile=daily_profile,
        demand_column=demand_column,
    )

    summary = {
        "status": "success",
        "part": 2,
        "version": DI2_VERSION,
        "timestamp_column": timestamp_column,
        "demand_column": demand_column,
        "input_records": int(len(dataframe)),
        "valid_records": int(len(prepared)),
        "invalid_records_removed": int(
            len(dataframe) - len(prepared)
        ),
        "first_timestamp": str(
            prepared[timestamp_column].min()
        ),
        "last_timestamp": str(
            prepared[timestamp_column].max()
        ),
        "total_observed_demand": di2_safe_float(
            prepared[demand_column].sum()
        ),
        "mean_record_demand": di2_safe_float(
            prepared[demand_column].mean()
        ),
        "median_record_demand": di2_safe_float(
            prepared[demand_column].median()
        ),
        "minimum_record_demand": di2_safe_float(
            prepared[demand_column].min()
        ),
        "maximum_record_demand": di2_safe_float(
            prepared[demand_column].max()
        ),
        "unique_observed_dates": int(
            prepared["DI2_Date"].nunique()
        ),
        "unique_observed_years": int(
            prepared["DI2_Year"].nunique()
        ),
        "unique_observed_months": int(
            prepared["DI2_Year_Month"].nunique()
        ),
        "insight_count": int(
            len(insights)
        ),
        "interpretation_scope": (
            "Observed ticket demand only; not ferry capacity, "
            "queue length, passenger count, or exact boarding demand."
        ),
    }

    return {
        "prepared_dataframe": prepared,
        "hourly_profile": hourly_profile,
        "weekday_profile": weekday_profile,
        "weekday_hour_matrix": weekday_hour_matrix,
        "weekend_profile": weekend_profile,
        "monthly_profile": monthly_profile,
        "year_month_profile": year_month_profile,
        "yearly_profile": yearly_profile,
        "daily_profile": daily_profile,
        "hourly_variability": hourly_variability,
        "peak_hours_by_weekday": peak_hours_by_weekday,
        "peak_periods": peak_periods,
        "quantile_summary": quantile_summary,
        "rolling_statistics": rolling_statistics,
        "classified_demand": classified_demand,
        "insights": insights,
        "summary": summary,
    }


# ================================================================
# SECTION 28: EXPORT DATAFRAME TO CSV
# ================================================================

def di2_export_dataframe(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> Dict[str, Any]:
    """
    Export a dataframe to CSV and return basic export metadata.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "Only pandas DataFrames can be exported."
        )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    return {
        "file": str(output_path),
        "rows": int(len(dataframe)),
        "columns": int(len(dataframe.columns)),
        "size_bytes": int(
            output_path.stat().st_size
        ),
    }


# ================================================================
# SECTION 29: EXPORT JSON SUMMARY
# ================================================================

def di2_export_json(
    data: Dict[str, Any],
    output_path: Path,
) -> Dict[str, Any]:
    """
    Export a dictionary as readable JSON.

    NumPy numeric types and pandas timestamps are converted to
    JSON-compatible representations.
    """
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    def json_default(value: Any) -> Any:
        if isinstance(
            value,
            (
                np.integer,
            ),
        ):
            return int(value)

        if isinstance(
            value,
            (
                np.floating,
            ),
        ):
            converted = float(value)

            if not np.isfinite(converted):
                return None

            return converted

        if isinstance(
            value,
            (
                np.ndarray,
            ),
        ):
            return value.tolist()

        if isinstance(
            value,
            (
                pd.Timestamp,
            ),
        ):
            return value.isoformat()

        if isinstance(
            value,
            (
                Path,
            ),
        ):
            return str(value)

        if pd.isna(value):
            return None

        return str(value)

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
            default=json_default,
        )

    return {
        "file": str(output_path),
        "size_bytes": int(
            output_path.stat().st_size
        ),
    }


# ================================================================
# SECTION 30: EXPORT ALL PART 2 RESULTS
# ================================================================

def di2_export_time_analysis(
    analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Export the main Part 2 tables and summary reports.

    Large observation-level tables are saved as CSV files.
    A compact JSON file stores the main summary and insights.
    """
    directories = di2_create_output_directories()

    output_dir = directories[
        "output_dir"
    ]

    report_dir = directories[
        "report_dir"
    ]

    export_map = {
        "hourly_profile": (
            output_dir / DI2_HOURLY_FILE
        ),
        "weekday_profile": (
            output_dir / DI2_WEEKDAY_FILE
        ),
        "monthly_profile": (
            output_dir / DI2_MONTHLY_FILE
        ),
        "yearly_profile": (
            output_dir / DI2_YEARLY_FILE
        ),
        "weekend_profile": (
            output_dir / DI2_WEEKEND_FILE
        ),
        "peak_periods": (
            output_dir / DI2_PEAK_FILE
        ),
        "rolling_statistics": (
            output_dir / DI2_ROLLING_FILE
        ),
        "weekday_hour_matrix": (
            output_dir / "weekday_hour_matrix.csv"
        ),
        "year_month_profile": (
            output_dir / "year_month_profile.csv"
        ),
        "daily_profile": (
            output_dir / "daily_demand_profile.csv"
        ),
        "hourly_variability": (
            output_dir / "hourly_demand_variability.csv"
        ),
        "peak_hours_by_weekday": (
            output_dir / "peak_hours_by_weekday.csv"
        ),
        "quantile_summary": (
            output_dir / "demand_quantile_summary.csv"
        ),
        "classified_demand": (
            output_dir / "classified_demand.csv"
        ),
    }

    exported_files = {}

    for key, output_path in export_map.items():
        dataframe = analysis.get(
            key
        )

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            continue

        exported_files[key] = (
            di2_export_dataframe(
                dataframe,
                output_path,
            )
        )

    summary = analysis.get(
        "summary",
        {},
    )

    insights = analysis.get(
        "insights",
        [],
    )

    summary_payload = {
        "summary": summary,
        "insights": insights,
        "exported_files": exported_files,
    }

    summary_export = di2_export_json(
        summary_payload,
        report_dir / DI2_SUMMARY_FILE,
    )

    insights_export = di2_export_json(
        {
            "insight_count": len(insights),
            "insights": insights,
        },
        report_dir / DI2_INSIGHTS_FILE,
    )

    return {
        "status": "success",
        "exported_file_count": len(
            exported_files
        ),
        "files": exported_files,
        "summary_json": summary_export,
        "insights_json": insights_export,
    }


# ================================================================
# SECTION 31: GENERATE A HUMAN-READABLE REPORT
# ================================================================

def di2_build_text_report(
    analysis: Dict[str, Any],
) -> str:
    """
    Create a readable text report from the analysis package.
    """
    summary = analysis.get(
        "summary",
        {},
    )

    lines = [
        "=" * 72,
        "TIME-BASED TICKET DEMAND INTELLIGENCE REPORT",
        "=" * 72,
        "",
        f"Version: {DI2_VERSION}",
        f"Status: {summary.get('status', 'unknown')}",
        f"Records analyzed: {summary.get('valid_records', 0)}",
        f"First timestamp: {summary.get('first_timestamp', 'N/A')}",
        f"Last timestamp: {summary.get('last_timestamp', 'N/A')}",
        "",
        "OBSERVED DEMAND SUMMARY",
        "-" * 72,
        (
            "Total observed demand: "
            f"{summary.get('total_observed_demand', 0)}"
        ),
        (
            "Mean demand per record: "
            f"{summary.get('mean_record_demand', 0)}"
        ),
        (
            "Median demand per record: "
            f"{summary.get('median_record_demand', 0)}"
        ),
        (
            "Maximum demand per record: "
            f"{summary.get('maximum_record_demand', 0)}"
        ),
        (
            "Minimum demand per record: "
            f"{summary.get('minimum_record_demand', 0)}"
        ),
        "",
        "TIME-BASED INSIGHTS",
        "-" * 72,
    ]

    insights = analysis.get(
        "insights",
        [],
    )

    if not insights:
        lines.append(
            "No insights were generated."
        )

    for index, insight in enumerate(
        insights,
        start=1,
    ):
        lines.extend(
            [
                "",
                f"{index}. {insight.get('title', 'Untitled')}",
                (
                    "Category: "
                    f"{insight.get('category', 'General')}"
                ),
                (
                    "Observation: "
                    f"{insight.get('observation', 'N/A')}"
                ),
                (
                    "Interpretation: "
                    f"{insight.get('interpretation', 'N/A')}"
                ),
                (
                    "Limitation: "
                    f"{insight.get('limitation', 'N/A')}"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "INTERPRETATION LIMITATION",
            "-" * 72,
            (
                "This report analyzes observed ticket demand. "
                "It does not establish exact passenger queues, "
                "ferry capacity, or boarding demand."
            ),
            "",
            "=" * 72,
        ]
    )

    return "\n".join(
        lines
    )


# ================================================================
# SECTION 32: EXPORT HUMAN-READABLE REPORT
# ================================================================

def di2_export_text_report(
    analysis: Dict[str, Any],
    filename: str = "time_based_demand_report.txt",
) -> Dict[str, Any]:
    """
    Save the human-readable report to the reports directory.
    """
    directories = di2_create_output_directories()

    output_path = (
        directories["report_dir"] / filename
    )

    report_text = di2_build_text_report(
        analysis
    )

    output_path.write_text(
        report_text,
        encoding="utf-8",
    )

    return {
        "file": str(output_path),
        "characters": len(report_text),
        "size_bytes": int(
            output_path.stat().st_size
        ),
    }


# ================================================================
# SECTION 33: CONVENIENCE WRAPPER
# ================================================================

def di2_run_time_intelligence(
    dataframe: pd.DataFrame,
    timestamp_column: str = "Timestamp",
    demand_column: str = "Sales Count",
    export_results: bool = True,
) -> Dict[str, Any]:
    """
    Run the Part 2 time-based analysis and optionally export it.

    This wrapper does not load the original CSV automatically.
    Pass the already-loaded dataframe to this function.
    """
    LOGGER.info(
        "Starting Part 2 time-based demand intelligence."
    )

    analysis = di2_build_time_analysis(
        dataframe=dataframe,
        timestamp_column=timestamp_column,
        demand_column=demand_column,
    )

    if export_results:
        analysis["export_result"] = (
            di2_export_time_analysis(
                analysis
            )
        )

        analysis["text_report_result"] = (
            di2_export_text_report(
                analysis
            )
        )

    LOGGER.info(
        "Part 2 time-based demand intelligence completed."
    )

    return analysis


# ================================================================
# SECTION 34: FUNCTION REGISTRY
# ================================================================



# ------------------------------------------------------------------
# PART 2 COMPATIBILITY LAYER
# Keeps legacy report schemas and modern single-demand workflows
# available without duplicate public function definitions.
# ------------------------------------------------------------------

def di2_add_rolling_change_metrics(
    dataframe,
    demand_columns=("Sales_Count", "Redemption_Count"),
    windows=(3, 7, 14),
    demand_column=None,
    window=12,
):
    """Dispatch rolling metrics to the legacy or modern interface."""
    if not isinstance(dataframe, pd.DataFrame):
        raise UnsupportedInputError(
            "dataframe must be a pandas DataFrame."
        )

    if demand_column is not None:
        if not isinstance(demand_column, str) or not demand_column:
            raise ValueError(
                "demand_column must be a non-empty column name."
            )
        if demand_column not in dataframe.columns:
            raise SourceDataValidationError(
                f"Missing demand column: {demand_column}"
            )
        window_value = safe_int(window, default=0)
        if window_value < 1:
            raise ValueError("window must be a positive integer.")
        return _modern_di2_add_rolling_change_metrics(
            dataframe,
            demand_column=demand_column,
            window=window_value,
        )

    if isinstance(demand_columns, str):
        demand_columns = (demand_columns,)
    else:
        demand_columns = tuple(demand_columns)

    if not demand_columns:
        raise ValueError(
            "At least one demand column must be provided."
        )

    if isinstance(windows, (int, float)):
        windows = (windows,)
    else:
        windows = tuple(windows)

    if not windows:
        raise ValueError(
            "At least one rolling window must be provided."
        )

    normalized_windows = []
    for value in windows:
        window_value = safe_int(value, default=0)
        if window_value < 1:
            raise ValueError(
                "Rolling windows must be positive integers."
            )
        normalized_windows.append(window_value)

    missing = [
        column for column in demand_columns
        if column not in dataframe.columns
    ]
    if missing:
        raise SourceDataValidationError(
            "Missing demand column(s): " + ", ".join(missing)
        )

    return _legacy_di2_add_rolling_change_metrics(
        dataframe,
        demand_columns=demand_columns,
        windows=tuple(sorted(set(normalized_windows))),
    )



def di2_build_hourly_profile(dataframe, demand_column=None):
    """Support both legacy hourly reports and modern demand analysis."""
    if not isinstance(dataframe, pd.DataFrame):
        raise UnsupportedInputError(
            "dataframe must be a pandas DataFrame."
        )

    if demand_column is not None:
        if not isinstance(demand_column, str) or not demand_column:
            raise ValueError(
                "demand_column must be a non-empty column name."
            )
        if demand_column not in dataframe.columns:
            raise SourceDataValidationError(
                f"Missing demand column: {demand_column}"
            )
        return _modern_di2_build_hourly_profile(
            dataframe,
            demand_column=demand_column,
        )

    return _legacy_di2_build_hourly_profile(dataframe)



def di2_build_weekday_profile(dataframe, demand_column=None):
    """Support both legacy weekday reports and modern demand analysis."""
    if not isinstance(dataframe, pd.DataFrame):
        raise UnsupportedInputError(
            "dataframe must be a pandas DataFrame."
        )

    if demand_column is not None:
        if not isinstance(demand_column, str) or not demand_column:
            raise ValueError(
                "demand_column must be a non-empty column name."
            )
        if demand_column not in dataframe.columns:
            raise SourceDataValidationError(
                f"Missing demand column: {demand_column}"
            )
        return _modern_di2_build_weekday_profile(
            dataframe,
            demand_column=demand_column,
        )

    return _legacy_di2_build_weekday_profile(dataframe)



def di2_build_weekend_comparison(dataframe, demand_column=None):
    """Support both legacy weekend reports and modern demand analysis."""
    if not isinstance(dataframe, pd.DataFrame):
        raise UnsupportedInputError(
            "dataframe must be a pandas DataFrame."
        )

    if demand_column is not None:
        if not isinstance(demand_column, str) or not demand_column:
            raise ValueError(
                "demand_column must be a non-empty column name."
            )
        if demand_column not in dataframe.columns:
            raise SourceDataValidationError(
                f"Missing demand column: {demand_column}"
            )
        return _modern_di2_build_weekend_comparison(
            dataframe,
            demand_column=demand_column,
        )

    return _legacy_di2_build_weekend_comparison(dataframe)


DI2_FUNCTION_REGISTRY = {
    "prepare_time_dataframe": di2_prepare_time_dataframe,
    "group_demand": di2_group_demand,
    "hourly_profile": di2_build_hourly_profile,
    "weekday_profile": di2_build_weekday_profile,
    "weekday_hour_matrix": di2_build_weekday_hour_matrix,
    "weekend_comparison": di2_build_weekend_comparison,
    "monthly_profile": di2_build_monthly_profile,
    "year_month_profile": di2_build_year_month_profile,
    "yearly_profile": di2_build_yearly_profile,
    "daily_profile": di2_build_daily_profile,
    "peak_periods": di2_identify_peak_periods,
    "quantile_summary": di2_build_quantile_summary,
    "rolling_statistics": di2_build_rolling_statistics,
    "classify_demand": di2_classify_demand_levels,
    "hourly_variability": di2_build_hourly_variability,
    "peak_hours_by_weekday": di2_build_peak_hours_by_weekday,
    "trend_direction": di2_estimate_trend_direction,
    "time_insights": di2_generate_time_insights,
    "build_analysis": di2_build_time_analysis,
    "export_analysis": di2_export_time_analysis,
    "run_time_intelligence": di2_run_time_intelligence,
}


# ================================================================
# SECTION 35: PART 2 METADATA
# ================================================================

DI2_METADATA = {
    "module": "demand_insights",
    "part": 2,
    "version": DI2_VERSION,
    "purpose": (
        "Advanced time-based ticket demand analysis."
    ),
    "analysis_dimensions": [
        "hour",
        "weekday",
        "weekday_hour",
        "weekend_vs_weekday",
        "month",
        "year_month",
        "year",
        "day",
        "rolling_observations",
        "peak_demand",
        "demand_variability",
    ],
    "output_directory": str(
        DI2_OUTPUT_ROOT
    ),
    "report_directory": str(
        DI2_REPORT_ROOT
    ),
    "scope_limitations": [
        "Does not estimate ferry capacity.",
        "Does not estimate queue length.",
        "Does not guarantee passenger boarding demand.",
        "Does not automatically produce future forecasts.",
        "Uses observed ticket demand values.",
    ],
}


# ================================================================
# SECTION 36: PUBLIC API
# ================================================================

def di2_get_metadata() -> Dict[str, Any]:
    """
    Return a copy of Part 2 module metadata.
    """
    return dict(
        DI2_METADATA
    )


def di2_list_functions() -> List[str]:
    """
    Return the names of the registered Part 2 functions.
    """
    return sorted(
        DI2_FUNCTION_REGISTRY.keys()
    )


def di2_get_function(
    name: str,
) -> Any:
    """
    Retrieve a registered function by its short name.
    """
    if name not in DI2_FUNCTION_REGISTRY:
        raise KeyError(
            f"Unknown Part 2 function: {name}"
        )

    return DI2_FUNCTION_REGISTRY[
        name
    ]


# ================================================================
# SECTION 37: FINAL PART 2 EXPORT LIST
# ================================================================

DI2_PUBLIC_NAMES = [
    "DI2_VERSION",
    "DI2_METADATA",
    "DI2_FUNCTION_REGISTRY",
    "di2_get_metadata",
    "di2_list_functions",
    "di2_get_function",
    "di2_prepare_time_dataframe",
    "di2_group_demand",
    "di2_build_hourly_profile",
    "di2_build_weekday_profile",
    "di2_build_weekday_hour_matrix",
    "di2_build_weekend_comparison",
    "di2_build_monthly_profile",
    "di2_build_year_month_profile",
    "di2_build_yearly_profile",
    "di2_build_daily_profile",
    "di2_identify_peak_periods",
    "di2_build_quantile_summary",
    "di2_build_rolling_statistics",
    "di2_add_rolling_change_metrics",
    "di2_classify_demand_levels",
    "di2_build_hourly_variability",
    "di2_build_peak_hours_by_weekday",
    "di2_build_peak_dates",
    "di2_build_low_demand_dates",
    "di2_estimate_trend_direction",
    "di2_build_daily_trend_summary",
    "di2_generate_time_insights",
    "di2_build_time_analysis",
    "di2_export_dataframe",
    "di2_export_json",
    "di2_export_time_analysis",
    "di2_build_text_report",
    "di2_export_text_report",
    "di2_run_time_intelligence",
]


# ================================================================
# <<< END PART 2: ADVANCED TIME-BASED DEMAND INTELLIGENCE
# ================================================================
# ================================================================
# >>> START PART 3: DEMAND ANOMALY AND DATA QUALITY INTELLIGENCE
# ================================================================
#
# FILE: demand_insights.py
# PART: 3
#
# PURPOSE:
#   1. Detect unusual demand observations.
#   2. Identify sudden increases and decreases in demand.
#   3. Analyze zero-demand and negative-demand records.
#   4. Inspect timestamp continuity and duplicate timestamps.
#   5. Measure demand stability and variability.
#   6. Generate data-driven operational recommendations.
#   7. Export detailed analysis results.
#
# PASTE INSTRUCTIONS:
#   Paste this complete block AFTER the END PART 2 comment.
#   Do not paste it inside another function.
#   Do not delete or replace Parts 1 and 2.
#
# IMPORTANT:
#   Every new function uses the di3_ prefix to reduce the risk
#   of naming conflicts with existing project functions.
#
# SCOPE:
#   This module analyzes recorded ticket sales or redemptions.
#   It does not directly measure passenger queues, ferry capacity,
#   or exact passenger boarding demand.
#
# ================================================================


# ================================================================
# SECTION 1: PART 3 CONFIGURATION
# ================================================================

DI3_VERSION = "1.0.0"

DI3_OUTPUT_ROOT = (
    BASE_DIR
    / "outputs"
    / "demand_insights"
    / "part_3"
)

DI3_REPORT_ROOT = (
    BASE_DIR
    / "outputs"
    / "reports"
    / "demand_insights"
    / "part_3"
)

DI3_ANOMALY_FILE = "demand_anomalies.csv"

DI3_CHANGE_FILE = "sudden_demand_changes.csv"

DI3_ZERO_DEMAND_FILE = "zero_demand_records.csv"

DI3_TIMESTAMP_QUALITY_FILE = "timestamp_quality.csv"

DI3_DEMAND_QUALITY_FILE = "demand_quality.csv"

DI3_STABILITY_FILE = "demand_stability.csv"

DI3_RECOMMENDATIONS_FILE = "operational_recommendations.json"

DI3_SUMMARY_FILE = "part_3_summary.json"

DI3_DEFAULT_Z_THRESHOLD = 3.0

DI3_DEFAULT_IQR_MULTIPLIER = 1.5

DI3_DEFAULT_CHANGE_THRESHOLD = 0.50

DI3_DEFAULT_MINIMUM_PERIODS = 4

DI3_DEFAULT_ROLLING_WINDOW = 12

DI3_DEFAULT_ZERO_LIMIT = 0.0


# ================================================================
# SECTION 2: CREATE OUTPUT DIRECTORIES
# ================================================================

def di3_create_output_directories() -> Dict[str, Path]:
    """
    Create the Part 3 output directories.

    Returns
    -------
    dict
        Dictionary containing output and report directory paths.
    """
    DI3_OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    DI3_REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    return {
        "output_dir": DI3_OUTPUT_ROOT,
        "report_dir": DI3_REPORT_ROOT,
    }


# ================================================================
# SECTION 3: VALIDATE ANALYSIS INPUT
# ================================================================

def di3_validate_input(
    dataframe: pd.DataFrame,
    timestamp_column: str = "Timestamp",
    demand_column: str = "Sales Count",
) -> Tuple[bool, List[str]]:
    """
    Validate the dataframe before Part 3 analysis.

    Checks:
    - Input type.
    - Empty dataframe.
    - Required columns.
    - Numeric demand values.
    - Valid timestamps.
    """
    errors = []

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        return (
            False,
            [
                "Input must be a pandas DataFrame."
            ],
        )

    if dataframe.empty:
        errors.append(
            "The dataframe contains no records."
        )

    required_columns = [
        timestamp_column,
        demand_column,
    ]

    for column in required_columns:
        if column not in dataframe.columns:
            errors.append(
                f"Required column is missing: {column}"
            )

    if errors:
        return False, errors

    converted_timestamps = pd.to_datetime(
        dataframe[timestamp_column],
        errors="coerce",
    )

    converted_demand = pd.to_numeric(
        dataframe[demand_column],
        errors="coerce",
    )

    if converted_timestamps.notna().sum() == 0:
        errors.append(
            "No valid timestamps are available."
        )

    if converted_demand.notna().sum() == 0:
        errors.append(
            "No valid numeric demand values are available."
        )

    return (
        len(errors) == 0,
        errors,
    )


# ================================================================
# SECTION 4: PREPARE A CLEAN ANALYSIS COPY
# ================================================================

def di3_prepare_dataframe(
    dataframe: pd.DataFrame,
    timestamp_column: str = "Timestamp",
    demand_column: str = "Sales Count",
) -> pd.DataFrame:
    """
    Create a sorted analysis copy without changing the original.

    Invalid timestamps and non-numeric demand values are retained
    in neither the cleaned copy nor its anomaly calculations.

    Use the separate quality functions to investigate invalid
    records before deciding whether to remove them permanently.
    """
    valid, errors = di3_validate_input(
        dataframe=dataframe,
        timestamp_column=timestamp_column,
        demand_column=demand_column,
    )

    if not valid:
        raise ValueError(
            "Part 3 input validation failed: "
            + "; ".join(errors)
        )

    result = dataframe.copy()

    result[timestamp_column] = pd.to_datetime(
        result[timestamp_column],
        errors="coerce",
    )

    result[demand_column] = pd.to_numeric(
        result[demand_column],
        errors="coerce",
    )

    result = result.dropna(
        subset=[
            timestamp_column,
            demand_column,
        ]
    ).copy()

    result = result.sort_values(
        timestamp_column
    ).reset_index(
        drop=True
    )

    return result


# ================================================================
# SECTION 5: NUMERIC CONVERSION HELPERS
# ================================================================

def di3_to_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Convert a value to a finite float.
    """
    try:
        converted = float(value)

        if not np.isfinite(converted):
            return default

        return converted

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return default


def di3_to_int(
    value: Any,
    default: int = 0,
) -> int:
    """
    Convert a value to an integer safely.
    """
    try:
        if pd.isna(value):
            return default

        return int(value)

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return default


def di3_safe_percentage(
    numerator: float,
    denominator: float,
) -> float:
    """
    Calculate a percentage without division-by-zero errors.
    """
    denominator = di3_to_float(
        denominator,
        default=0.0,
    )

    numerator = di3_to_float(
        numerator,
        default=0.0,
    )

    if denominator == 0:
        return 0.0

    return (
        numerator
        / denominator
        * 100.0
    )


# ================================================================
# SECTION 6: DEMAND DISTRIBUTION SUMMARY
# ================================================================

def di3_build_distribution_summary(
    dataframe: pd.DataFrame,
    demand_column: str = "Sales Count",
) -> Dict[str, Any]:
    """
    Calculate distribution statistics for observed demand.

    Includes:
    - Mean and median.
    - Standard deviation.
    - Minimum and maximum.
    - Selected percentiles.
    - Skewness.
    - Interquartile range.
    """
    if dataframe.empty:
        return {
            "status": "no_data",
            "record_count": 0,
        }

    values = pd.to_numeric(
        dataframe[demand_column],
        errors="coerce",
    ).dropna()

    if values.empty:
        return {
            "status": "no_numeric_demand",
            "record_count": 0,
        }

    q25 = values.quantile(0.25)

    q50 = values.quantile(0.50)

    q75 = values.quantile(0.75)

    q90 = values.quantile(0.90)

    q95 = values.quantile(0.95)

    q99 = values.quantile(0.99)

    standard_deviation = values.std()

    mean_value = values.mean()

    return {
        "status": "success",
        "record_count": int(values.count()),
        "total_demand": di3_to_float(
            values.sum()
        ),
        "mean": di3_to_float(
            mean_value
        ),
        "median": di3_to_float(
            q50
        ),
        "standard_deviation": di3_to_float(
            standard_deviation
        ),
        "variance": di3_to_float(
            values.var()
        ),
        "minimum": di3_to_float(
            values.min()
        ),
        "maximum": di3_to_float(
            values.max()
        ),
        "range": di3_to_float(
            values.max() - values.min()
        ),
        "percentile_25": di3_to_float(
            q25
        ),
        "percentile_50": di3_to_float(
            q50
        ),
        "percentile_75": di3_to_float(
            q75
        ),
        "percentile_90": di3_to_float(
            q90
        ),
        "percentile_95": di3_to_float(
            q95
        ),
        "percentile_99": di3_to_float(
            q99
        ),
        "interquartile_range": di3_to_float(
            q75 - q25
        ),
        "skewness": di3_to_float(
            values.skew()
        ),
        "coefficient_of_variation_percent": (
            di3_safe_percentage(
                standard_deviation,
                mean_value,
            )
        ),
        "negative_value_count": int(
            (values < 0).sum()
        ),
        "zero_value_count": int(
            (values == 0).sum()
        ),
        "positive_value_count": int(
            (values > 0).sum()
        ),
    }


# ================================================================
# SECTION 7: Z-SCORE ANOMALY DETECTION
# ================================================================

def di3_detect_zscore_anomalies(
    dataframe: pd.DataFrame,
    demand_column: str = "Sales Count",
    threshold: float = DI3_DEFAULT_Z_THRESHOLD,
) -> pd.DataFrame:
    """
    Identify globally unusual demand values using z-scores.

    The z-score expresses how far a value lies from the mean in
    units of standard deviation.

    Important:
    This method can be sensitive to skewed demand distributions
    and changing seasonal patterns.
    """
    if dataframe.empty:
        return pd.DataFrame()

    if threshold <= 0:
        raise ValueError(
            "threshold must be greater than zero."
        )

    result = dataframe.copy()

    values = pd.to_numeric(
        result[demand_column],
        errors="coerce",
    )

    mean_value = values.mean()

    standard_deviation = values.std()

    result["DI3_ZScore"] = 0.0

    if (
        pd.notna(standard_deviation)
        and standard_deviation > 0
    ):
        result["DI3_ZScore"] = (
            values - mean_value
        ) / standard_deviation

    result["DI3_Absolute_ZScore"] = (
        result["DI3_ZScore"].abs()
    )

    result["DI3_ZScore_Anomaly"] = (
        result["DI3_Absolute_ZScore"] >= threshold
    )

    result["DI3_ZScore_Threshold"] = threshold

    result["DI3_ZScore_Anomaly_Type"] = np.select(
        [
            result["DI3_ZScore_Anomaly"]
            & (values > mean_value),

            result["DI3_ZScore_Anomaly"]
            & (values < mean_value),
        ],
        [
            "Unusually High",
            "Unusually Low",
        ],
        default="Normal",
    )

    anomalies = result.loc[
        result["DI3_ZScore_Anomaly"]
    ].copy()

    return anomalies.sort_values(
        "DI3_Absolute_ZScore",
        ascending=False,
    ).reset_index(
        drop=True
    )


# ================================================================
# SECTION 8: IQR-BASED ANOMALY DETECTION
# ================================================================

def di3_detect_iqr_anomalies(
    dataframe: pd.DataFrame,
    demand_column: str = "Sales Count",
    multiplier: float = DI3_DEFAULT_IQR_MULTIPLIER,
) -> pd.DataFrame:
    """
    Identify demand values outside the IQR-based boundaries.

    Lower boundary:
        Q1 - multiplier * IQR

    Upper boundary:
        Q3 + multiplier * IQR

    This method is more resistant to extreme observations than
    a mean-and-standard-deviation threshold.
    """
    if dataframe.empty:
        return pd.DataFrame()

    if multiplier <= 0:
        raise ValueError(
            "multiplier must be greater than zero."
        )

    result = dataframe.copy()

    values = pd.to_numeric(
        result[demand_column],
        errors="coerce",
    )

    q1 = values.quantile(0.25)

    q3 = values.quantile(0.75)

    iqr = q3 - q1

    lower_boundary = (
        q1 - multiplier * iqr
    )

    upper_boundary = (
        q3 + multiplier * iqr
    )

    result["DI3_IQR_Q1"] = q1

    result["DI3_IQR_Q3"] = q3

    result["DI3_IQR_Value"] = iqr

    result["DI3_IQR_Lower_Boundary"] = (
        lower_boundary
    )

    result["DI3_IQR_Upper_Boundary"] = (
        upper_boundary
    )

    result["DI3_IQR_Anomaly"] = (
        (values < lower_boundary)
        | (values > upper_boundary)
    )

    result["DI3_IQR_Anomaly_Type"] = np.select(
        [
            values < lower_boundary,
            values > upper_boundary,
        ],
        [
            "Below Lower Boundary",
            "Above Upper Boundary",
        ],
        default="Normal",
    )

    anomalies = result.loc[
        result["DI3_IQR_Anomaly"]
    ].copy()

    return anomalies.sort_values(
        demand_column,
        ascending=False,
    ).reset_index(
        drop=True
    )


# ================================================================
# SECTION 9: ROBUST MAD-BASED ANOMALY DETECTION
# ================================================================

def di3_detect_mad_anomalies(
    dataframe: pd.DataFrame,
    demand_column: str = "Sales Count",
    threshold: float = 3.5,
) -> pd.DataFrame:
    """
    Detect unusual demand using median absolute deviation.

    MAD is based on distance from the median, making it less
    sensitive to extreme values than standard deviation.

    If MAD is zero, the method avoids division by zero and returns
    no MAD-based anomalies. Other detection methods may still help.
    """
    if dataframe.empty:
        return pd.DataFrame()

    if threshold <= 0:
        raise ValueError(
            "threshold must be greater than zero."
        )

    result = dataframe.copy()

    values = pd.to_numeric(
        result[demand_column],
        errors="coerce",
    )

    median_value = values.median()

    absolute_deviation = (
        values - median_value
    ).abs()

    mad_value = absolute_deviation.median()

    result["DI3_Median"] = median_value

    result["DI3_MAD"] = mad_value

    result["DI3_Robust_ZScore"] = np.nan

    if (
        pd.notna(mad_value)
        and mad_value > 0
    ):
        result["DI3_Robust_ZScore"] = (
            0.67448975
            * (values - median_value)
            / mad_value
        )

        result["DI3_MAD_Anomaly"] = (
            result["DI3_Robust_ZScore"].abs()
            >= threshold
        )

    else:
        result["DI3_MAD_Anomaly"] = False

    result["DI3_MAD_Threshold"] = threshold

    anomalies = result.loc[
        result["DI3_MAD_Anomaly"]
    ].copy()

    return anomalies.sort_values(
        "DI3_Robust_ZScore",
        key=lambda column: column.abs(),
        ascending=False,
    ).reset_index(
        drop=True
    )


# ================================================================
# SECTION 10: COMBINE ANOMALY DETECTION RESULTS
# ================================================================

def di3_build_combined_anomalies(
    dataframe: pd.DataFrame,
    timestamp_column: str = "Timestamp",
    demand_column: str = "Sales Count",
) -> pd.DataFrame:
    """
    Combine z-score, IQR, and MAD anomaly detection results.

    An observation can be flagged by more than one method.
    The resulting method count shows how many methods flagged it.

    The combined result is a screening tool, not proof of an error.
    """
    if dataframe.empty:
        return pd.DataFrame()

    prepared = di3_prepare_dataframe(
        dataframe=dataframe,
        timestamp_column=timestamp_column,
        demand_column=demand_column,
    )

    zscore = di3_detect_zscore_anomalies(
        prepared,
        demand_column=demand_column,
    )

    iqr = di3_detect_iqr_anomalies(
        prepared,
        demand_column=demand_column,
    )

    mad = di3_detect_mad_anomalies(
        prepared,
        demand_column=demand_column,
    )

    result = prepared[
        [
            timestamp_column,
            demand_column,
        ]
    ].copy()

    result["DI3_ZScore_Anomaly"] = False

    result["DI3_IQR_Anomaly"] = False

    result["DI3_MAD_Anomaly"] = False

    if not zscore.empty:
        zscore_keys = set(
            zip(
                zscore[timestamp_column],
                zscore[demand_column],
            )
        )

        result["DI3_ZScore_Anomaly"] = [
            (
                timestamp,
                demand,
            ) in zscore_keys
            for timestamp, demand in zip(
                result[timestamp_column],
                result[demand_column],
            )
        ]

    if not iqr.empty:
        iqr_keys = set(
            zip(
                iqr[timestamp_column],
                iqr[demand_column],
            )
        )

        result["DI3_IQR_Anomaly"] = [
            (
                timestamp,
                demand,
            ) in iqr_keys
            for timestamp, demand in zip(
                result[timestamp_column],
                result[demand_column],
            )
        ]

    if not mad.empty:
        mad_keys = set(
            zip(
                mad[timestamp_column],
                mad[demand_column],
            )
        )

        result["DI3_MAD_Anomaly"] = [
            (
                timestamp,
                demand,
            ) in mad_keys
            for timestamp, demand in zip(
                result[timestamp_column],
                result[demand_column],
            )
        ]

    result["DI3_Detection_Method_Count"] = (
        result[
            [
                "DI3_ZScore_Anomaly",
                "DI3_IQR_Anomaly",
                "DI3_MAD_Anomaly",
            ]
        ].sum(
            axis=1
        )
    )

    result["DI3_Is_Combined_Anomaly"] = (
        result["DI3_Detection_Method_Count"] > 0
    )

    result["DI3_Anomaly_Confidence_Label"] = np.select(
        [
            result["DI3_Detection_Method_Count"] >= 3,
            result["DI3_Detection_Method_Count"] == 2,
            result["DI3_Detection_Method_Count"] == 1,
        ],
        [
            "Flagged By All Three Methods",
            "Flagged By Two Methods",
            "Flagged By One Method",
        ],
        default="Not Flagged",
    )

    result = result.loc[
        result["DI3_Is_Combined_Anomaly"]
    ].copy()

    return result.sort_values(
        "DI3_Detection_Method_Count",
        ascending=False,
    ).reset_index(
        drop=True
    )


# ================================================================
# SECTION 11: DEMAND CHANGE CALCULATION
# ================================================================

def di3_calculate_demand_changes(
    dataframe: pd.DataFrame,
    demand_column: str = "Sales Count",
) -> pd.DataFrame:
    """
    Calculate changes between consecutive observed records.

    The calculation follows timestamp order.

    If a timestamp gap exists, consecutive rows may represent
    more than one expected sampling interval apart.
    """
    if dataframe.empty:
        return pd.DataFrame()

    result = dataframe.copy()

    result = result.sort_values(
        "Timestamp"
    ).reset_index(
        drop=True
    )

    values = pd.to_numeric(
        result[demand_column],
        errors="coerce",
    )

    result["DI3_Previous_Demand"] = (
        values.shift(1)
    )

    result["DI3_Demand_Change"] = (
        values - values.shift(1)
    )

    result["DI3_Absolute_Demand_Change"] = (
        result["DI3_Demand_Change"].abs()
    )

    result["DI3_Demand_Change_Percent"] = (
        result.apply(
            lambda row: (
                di3_safe_percentage(
                    row["DI3_Demand_Change"],
                    row["DI3_Previous_Demand"],
                )
                if pd.notna(
                    row["DI3_Previous_Demand"]
                )
                and row["DI3_Previous_Demand"] != 0
                else np.nan
            ),
            axis=1,
        )
    )

    result["DI3_Time_Gap_Minutes"] = (
        result["Timestamp"].diff().dt.total_seconds()
        / 60.0
    )

    result["DI3_Demand_Direction"] = np.select(
        [
            result["DI3_Demand_Change"] > 0,
            result["DI3_Demand_Change"] < 0,
        ],
        [
            "Increase",
            "Decrease",
        ],
        default="No Change",
    )

    return result


# ================================================================
# SECTION 12: IDENTIFY SUDDEN DEMAND CHANGES
# ================================================================

def di3_detect_sudden_changes(
    dataframe: pd.DataFrame,
    demand_column: str = "Sales Count",
    threshold: float = DI3_DEFAULT_CHANGE_THRESHOLD,
    minimum_absolute_change: float = 0.0,
) -> pd.DataFrame:
    """
    Identify unusually large relative changes between records.

    threshold=0.50 means a change of at least 50 percent relative
    to the previous positive demand value.

    The function also accepts an absolute-change threshold.

    Changes after a zero-demand record are not assigned a
    percentage because the previous value is zero.
    """
    if threshold < 0:
        raise ValueError(
            "threshold cannot be negative."
        )

    if minimum_absolute_change < 0:
        raise ValueError(
            "minimum_absolute_change cannot be negative."
        )

    if dataframe.empty:
        return pd.DataFrame()

    result = di3_calculate_demand_changes(
        dataframe,
        demand_column=demand_column,
    )

    percentage_change = (
        result["DI3_Demand_Change_Percent"].abs()
        / 100.0
    )

    relative_flag = (
        percentage_change >= threshold
    )

    absolute_flag = (
        result["DI3_Absolute_Demand_Change"]
        >= minimum_absolute_change
    )

    result["DI3_Sudden_Change"] = (
        relative_flag & absolute_flag
    )

    result["DI3_Change_Threshold_Percent"] = (
        threshold * 100
    )

    result["DI3_Minimum_Absolute_Change"] = (
        minimum_absolute_change
    )

    flagged = result.loc[
        result["DI3_Sudden_Change"]
    ].copy()

    return flagged.sort_values(
        "DI3_Absolute_Demand_Change",
        ascending=False,
    ).reset_index(
        drop=True
    )


# ================================================================
# SECTION 13: ROBUST CHANGE DETECTION USING MAD
# ================================================================

def di3_detect_change_outliers(
    dataframe: pd.DataFrame,
    demand_column: str = "Sales Count",
    threshold: float = 3.5,
) -> pd.DataFrame:
    """
    Detect unusually large changes relative to historical changes.

    The method applies MAD to consecutive demand differences.

    A flagged change may be legitimate, such as an unusual sales
    period. Investigate it before treating it as a data problem.
    """
    if dataframe.empty:
        return pd.DataFrame()

    result = di3_calculate_demand_changes(
        dataframe,
        demand_column=demand_column,
    )

    changes = result[
        "DI3_Demand_Change"
    ].dropna()

    if changes.empty:
        return result.iloc[0:0].copy()

    median_change = changes.median()

    deviations = (
        changes - median_change
    ).abs()

    mad_value = deviations.median()

    result["DI3_Change_Median"] = median_change

    result["DI3_Change_MAD"] = mad_value

    result["DI3_Change_Robust_ZScore"] = np.nan

    if mad_value > 0:
        result["DI3_Change_Robust_ZScore"] = (
            0.67448975
            * (
                result["DI3_Demand_Change"]
                - median_change
            )
            / mad_value
        )

        result["DI3_Change_Outlier"] = (
            result["DI3_Change_Robust_ZScore"].abs()
            >= threshold
        )

    else:
        result["DI3_Change_Outlier"] = False

    flagged = result.loc[
        result["DI3_Change_Outlier"]
    ].copy()

    return flagged.sort_values(
        "DI3_Change_Robust_ZScore",
        key=lambda column: column.abs(),
        ascending=False,
    ).reset_index(
        drop=True
    )


# ================================================================
# SECTION 14: ZERO-DEMAND RECORD ANALYSIS
# ================================================================

def di3_analyze_zero_demand(
    dataframe: pd.DataFrame,
    demand_column: str = "Sales Count",
    zero_limit: float = DI3_DEFAULT_ZERO_LIMIT,
) -> pd.DataFrame:
    """
    Identify records with demand equal to or below zero_limit.

    Zero sales do not automatically indicate a problem. They may
    reflect a legitimate period with no ticket sales.
    """
    if dataframe.empty:
        return pd.DataFrame()

    result = dataframe.copy()

    values = pd.to_numeric(
        result[demand_column],
        errors="coerce",
    )

    result["DI3_Is_Zero_Demand"] = (
        values == zero_limit
    )

    result["DI3_Is_Negative_Demand"] = (
        values < zero_limit
    )

    result["DI3_Zero_Limit"] = zero_limit

    selected = result.loc[
        result["DI3_Is_Zero_Demand"]
        | result["DI3_Is_Negative_Demand"]
    ].copy()

    selected["DI3_Zero_Record_Type"] = np.select(
        [
            selected[demand_column] < zero_limit,
            selected[demand_column] == zero_limit,
        ],
        [
            "Negative Demand",
            "Zero Demand",
        ],
        default="Other",
    )

    return selected.sort_values(
        "Timestamp"
    ).reset_index(
        drop=True
    )


# ================================================================
# SECTION 15: ZERO-DEMAND SUMMARY
# ================================================================

def di3_build_zero_demand_summary(
    dataframe: pd.DataFrame,
    demand_column: str = "Sales Count",
) -> Dict[str, Any]:
    """
    Summarize zero, negative, and positive demand values.
    """
    if dataframe.empty:
        return {
            "status": "no_data",
        }

    values = pd.to_numeric(
        dataframe[demand_column],
        errors="coerce",
    ).dropna()

    total = len(values)

    zero_count = int(
        (values == 0).sum()
    )

    negative_count = int(
        (values < 0).sum()
    )

    positive_count = int(
        (values > 0).sum()
    )

    return {
        "status": "success",
        "total_numeric_records": int(total),
        "zero_demand_records": zero_count,
        "negative_demand_records": negative_count,
        "positive_demand_records": positive_count,
        "zero_demand_percent": di3_safe_percentage(
            zero_count,
            total,
        ),
        "negative_demand_percent": di3_safe_percentage(
            negative_count,
            total,
        ),
        "positive_demand_percent": di3_safe_percentage(
            positive_count,
            total,
        ),
    }


# ================================================================
# SECTION 16: CONSECUTIVE ZERO-DEMAND PERIODS
# ================================================================

def di3_find_zero_demand_runs(
    dataframe: pd.DataFrame,
    demand_column: str = "Sales Count",
    timestamp_column: str = "Timestamp",
) -> pd.DataFrame:
    """
    Identify consecutive rows with zero demand.

    Each run is based on consecutive rows in timestamp order.
    It is not assumed that every row is exactly 15 minutes apart.
    """
    if dataframe.empty:
        return pd.DataFrame()

    result = dataframe[
        [
            timestamp_column,
            demand_column,
        ]
    ].copy()

    result[timestamp_column] = pd.to_datetime(
        result[timestamp_column],
        errors="coerce",
    )

    result[demand_column] = pd.to_numeric(
        result[demand_column],
        errors="coerce",
    )

    result = result.dropna(
        subset=[
            timestamp_column,
            demand_column,
        ]
    ).sort_values(
        timestamp_column
    ).reset_index(
        drop=True
    )

    result["DI3_Is_Zero"] = (
        result[demand_column] == 0
    )

    result["DI3_Run_Group"] = (
        result["DI3_Is_Zero"]
        != result["DI3_Is_Zero"].shift()
    ).cumsum()

    zero_rows = result.loc[
        result["DI3_Is_Zero"]
    ].copy()

    if zero_rows.empty:
        return pd.DataFrame(
            columns=[
                "Run_Start",
                "Run_End",
                "Zero_Record_Count",
                "Elapsed_Minutes",
            ]
        )

    grouped = zero_rows.groupby(
        "DI3_Run_Group"
    )

    runs = grouped.agg(
        Run_Start=(
            timestamp_column,
            "min",
        ),
        Run_End=(
            timestamp_column,
            "max",
        ),
        Zero_Record_Count=(
            demand_column,
            "size",
        ),
    ).reset_index(
        drop=True
    )

    runs["Elapsed_Minutes"] = (
        (
            runs["Run_End"]
            - runs["Run_Start"]
        ).dt.total_seconds()
        / 60.0
    )

    return runs.sort_values(
        "Zero_Record_Count",
        ascending=False,
    ).reset_index(
        drop=True
    )


# ================================================================
# SECTION 17: NEGATIVE DEMAND VALIDATION
# ================================================================

def di3_find_negative_demand(
    dataframe: pd.DataFrame,
    demand_column: str = "Sales Count",
) -> pd.DataFrame:
    """
    Identify negative values in a demand column.

    Negative values may be invalid for some sales datasets, but
    refunds, reversals, or corrections may legitimately be stored
    as negative values. Confirm the dataset's meaning first.
    """
    if dataframe.empty:
        return pd.DataFrame()

    values = pd.to_numeric(
        dataframe[demand_column],
        errors="coerce",
    )

    result = dataframe.loc[
        values < 0
    ].copy()

    result["DI3_Quality_Flag"] = (
        "Negative demand requires review"
    )

    return result


# ================================================================
# SECTION 18: MISSING-VALUE QUALITY REPORT
# ================================================================

def di3_build_missing_value_report(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate missing-value counts and percentages by column.
    """
    if dataframe.empty:
        return pd.DataFrame(
            columns=[
                "Column",
                "Data_Type",
                "Missing_Count",
                "Missing_Percent",
                "Non_Missing_Count",
            ]
        )

    total_rows = len(dataframe)

    rows = []

    for column in dataframe.columns:
        missing_count = int(
            dataframe[column].isna().sum()
        )

        non_missing_count = (
            total_rows - missing_count
        )

        rows.append(
            {
                "Column": column,
                "Data_Type": str(
                    dataframe[column].dtype
                ),
                "Missing_Count": missing_count,
                "Missing_Percent": (
                    di3_safe_percentage(
                        missing_count,
                        total_rows,
                    )
                ),
                "Non_Missing_Count": (
                    non_missing_count
                ),
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        "Missing_Count",
        ascending=False,
    ).reset_index(
        drop=True
    )


# ================================================================
# SECTION 19: DUPLICATE-RECORD ANALYSIS
# ================================================================

def di3_build_duplicate_report(
    dataframe: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Count fully duplicated rows and duplicated timestamps.

    Duplicate timestamps do not necessarily mean duplicate data.
    The dataset may contain legitimate separate records sharing
    the same timestamp.
    """
    if dataframe.empty:
        return {
            "status": "no_data",
        }

    full_duplicate_count = int(
        dataframe.duplicated().sum()
    )

    timestamp_duplicate_count = 0

    if "Timestamp" in dataframe.columns:
        timestamps = pd.to_datetime(
            dataframe["Timestamp"],
            errors="coerce",
        )

        timestamp_duplicate_count = int(
            timestamps.duplicated(
                keep=False
            ).sum()
        )

    return {
        "status": "success",
        "row_count": int(len(dataframe)),
        "fully_duplicated_rows": (
            full_duplicate_count
        ),
        "fully_duplicated_percent": (
            di3_safe_percentage(
                full_duplicate_count,
                len(dataframe),
            )
        ),
        "rows_with_duplicate_timestamps": (
            timestamp_duplicate_count
        ),
        "duplicate_timestamp_row_percent": (
            di3_safe_percentage(
                timestamp_duplicate_count,
                len(dataframe),
            )
        ),
    }


# ================================================================
# SECTION 20: TIMESTAMP GAP ANALYSIS
# ================================================================

def di3_build_timestamp_gap_report(
    dataframe: pd.DataFrame,
    timestamp_column: str = "Timestamp",
    expected_interval_minutes: Optional[float] = 15.0,
) -> pd.DataFrame:
    """
    Calculate time gaps between consecutive valid timestamps.

    For a 15-minute expected interval:
    - 15 minutes indicates the expected spacing.
    - More than 15 minutes indicates a longer gap.
    - Less than 15 minutes indicates closer spacing.

    The expected interval is configurable and should be based on
    the actual data collection process.
    """
    if dataframe.empty:
        return pd.DataFrame()

    timestamps = pd.to_datetime(
        dataframe[timestamp_column],
        errors="coerce",
    ).dropna().sort_values().reset_index(
        drop=True
    )

    if len(timestamps) < 2:
        return pd.DataFrame(
            columns=[
                "Previous_Timestamp",
                "Timestamp",
                "Gap_Minutes",
                "Expected_Interval_Minutes",
                "Gap_Type",
            ]
        )

    result = pd.DataFrame(
        {
            "Timestamp": timestamps,
        }
    )

    result["Previous_Timestamp"] = (
        result["Timestamp"].shift(1)
    )

    result["Gap_Minutes"] = (
        (
            result["Timestamp"]
            - result["Previous_Timestamp"]
        ).dt.total_seconds()
        / 60.0
    )

    result = result.dropna(
        subset=[
            "Previous_Timestamp",
            "Gap_Minutes",
        ]
    ).reset_index(
        drop=True
    )

    result["Expected_Interval_Minutes"] = (
        expected_interval_minutes
    )

    if expected_interval_minutes is None:
        result["Gap_Type"] = (
            "Interval Not Specified"
        )

    else:
        result["Gap_Type"] = np.select(
            [
                result["Gap_Minutes"]
                > expected_interval_minutes,

                result["Gap_Minutes"]
                < expected_interval_minutes,
            ],
            [
                "Longer Than Expected",
                "Shorter Than Expected",
            ],
            default="Expected Spacing",
        )

    return result.sort_values(
        "Gap_Minutes",
        ascending=False,
    ).reset_index(
        drop=True
    )


# ================================================================
# SECTION 21: TIMESTAMP QUALITY SUMMARY
# ================================================================

def di3_build_timestamp_quality_summary(
    dataframe: pd.DataFrame,
    timestamp_column: str = "Timestamp",
    expected_interval_minutes: Optional[float] = 15.0,
) -> Dict[str, Any]:
    """
    Summarize timestamp validity, uniqueness, and time coverage.
    """
    if dataframe.empty:
        return {
            "status": "no_data",
        }

    timestamps = pd.to_datetime(
        dataframe[timestamp_column],
        errors="coerce",
    )

    valid_timestamps = timestamps.dropna()

    invalid_count = int(
        timestamps.isna().sum()
    )

    duplicate_count = int(
        valid_timestamps.duplicated(
            keep=False
        ).sum()
    )

    if valid_timestamps.empty:
        return {
            "status": "no_valid_timestamps",
            "row_count": int(len(dataframe)),
            "invalid_timestamp_count": invalid_count,
        }

    ordered = valid_timestamps.sort_values()

    gaps = ordered.diff().dt.total_seconds() / 60.0

    valid_gaps = gaps.dropna()

    longer_gaps = 0

    shorter_gaps = 0

    if (
        expected_interval_minutes is not None
        and not valid_gaps.empty
    ):
        longer_gaps = int(
            (
                valid_gaps
                > expected_interval_minutes
            ).sum()
        )

        shorter_gaps = int(
            (
                valid_gaps
                < expected_interval_minutes
            ).sum()
        )

    return {
        "status": "success",
        "row_count": int(len(dataframe)),
        "valid_timestamp_count": int(
            valid_timestamps.count()
        ),
        "invalid_timestamp_count": invalid_count,
        "invalid_timestamp_percent": (
            di3_safe_percentage(
                invalid_count,
                len(dataframe),
            )
        ),
        "rows_with_duplicate_timestamps": (
            duplicate_count
        ),
        "first_timestamp": str(
            ordered.min()
        ),
        "last_timestamp": str(
            ordered.max()
        ),
        "observed_span_hours": (
            (
                ordered.max()
                - ordered.min()
            ).total_seconds()
            / 3600.0
        ),
        "expected_interval_minutes": (
            expected_interval_minutes
        ),
        "longer_than_expected_gap_count": (
            longer_gaps
        ),
        "shorter_than_expected_gap_count": (
            shorter_gaps
        ),
        "median_observed_gap_minutes": (
            di3_to_float(
                valid_gaps.median()
            )
            if not valid_gaps.empty
            else None
        ),
        "maximum_observed_gap_minutes": (
            di3_to_float(
                valid_gaps.max()
            )
            if not valid_gaps.empty
            else None
        ),
    }


# ================================================================
# SECTION 22: DEMAND DATA QUALITY REPORT
# ================================================================

def di3_build_demand_quality_report(
    dataframe: pd.DataFrame,
    demand_column: str = "Sales Count",
) -> Dict[str, Any]:
    """
    Inspect demand values for missing, invalid, negative, and
    non-finite values.
    """
    if dataframe.empty:
        return {
            "status": "no_data",
        }

    original = dataframe[demand_column]

    numeric = pd.to_numeric(
        original,
        errors="coerce",
    )

    numeric_array = numeric.to_numpy(
        dtype=float,
        na_value=np.nan,
    )

    non_finite_mask = (
        ~np.isfinite(numeric_array)
        & ~np.isnan(numeric_array)
    )

    missing_count = int(
        original.isna().sum()
    )

    conversion_failure_count = int(
        (
            numeric.isna()
            & original.notna()
        ).sum()
    )

    non_finite_count = int(
        non_finite_mask.sum()
    )

    negative_count = int(
        (numeric < 0).sum()
    )

    zero_count = int(
        (numeric == 0).sum()
    )

    positive_count = int(
        (numeric > 0).sum()
    )

    return {
        "status": "success",
        "row_count": int(len(dataframe)),
        "missing_count": missing_count,
        "missing_percent": (
            di3_safe_percentage(
                missing_count,
                len(dataframe),
            )
        ),
        "numeric_conversion_failure_count": (
            conversion_failure_count
        ),
        "non_finite_count": non_finite_count,
        "negative_demand_count": negative_count,
        "zero_demand_count": zero_count,
        "positive_demand_count": positive_count,
        "numeric_minimum": (
            di3_to_float(numeric.min())
            if numeric.notna().any()
            else None
        ),
        "numeric_maximum": (
            di3_to_float(numeric.max())
            if numeric.notna().any()
            else None
        ),
        "numeric_mean": (
            di3_to_float(numeric.mean())
            if numeric.notna().any()
            else None
        ),
    }


# ================================================================
# SECTION 23: BUILD DEMAND STABILITY METRICS
# ================================================================

def di3_build_stability_metrics(
    dataframe: pd.DataFrame,
    demand_column: str = "Sales Count",
    window: int = DI3_DEFAULT_ROLLING_WINDOW,
) -> pd.DataFrame:
    """
    Measure rolling demand stability over ordered observations.

    The window is measured in rows, not minutes.
    Timestamp gaps may make the actual elapsed duration variable.
    """
    if dataframe.empty:
        return pd.DataFrame()

    if window < 1:
        raise ValueError(
            "window must be at least 1."
        )

    result = dataframe[
        [
            "Timestamp",
            demand_column,
        ]
    ].copy()

    result["Timestamp"] = pd.to_datetime(
        result["Timestamp"],
        errors="coerce",
    )

    result[demand_column] = pd.to_numeric(
        result[demand_column],
        errors="coerce",
    )

    result = result.dropna(
        subset=[
            "Timestamp",
            demand_column,
        ]
    ).sort_values(
        "Timestamp"
    ).reset_index(
        drop=True
    )

    values = result[
        demand_column
    ]

    rolling = values.rolling(
        window=window,
        min_periods=2,
    )

    result["DI3_Rolling_Mean"] = (
        rolling.mean()
    )

    result["DI3_Rolling_Std"] = (
        rolling.std()
    )

    result["DI3_Rolling_Min"] = (
        values.rolling(
            window=window,
            min_periods=1,
        ).min()
    )

    result["DI3_Rolling_Max"] = (
        values.rolling(
            window=window,
            min_periods=1,
        ).max()
    )

    result["DI3_Rolling_Range"] = (
        result["DI3_Rolling_Max"]
        - result["DI3_Rolling_Min"]
    )

    result["DI3_Rolling_CV_Percent"] = (
        result.apply(
            lambda row: (
                di3_safe_percentage(
                    row["DI3_Rolling_Std"],
                    row["DI3_Rolling_Mean"],
                )
                if pd.notna(
                    row["DI3_Rolling_Std"]
                )
                else np.nan
            ),
            axis=1,
        )
    )

    result["DI3_Stability_Label"] = np.select(
        [
            result["DI3_Rolling_CV_Percent"] <= 10,
            result["DI3_Rolling_CV_Percent"] <= 25,
            result["DI3_Rolling_CV_Percent"] > 25,
        ],
        [
            "Relatively Stable",
            "Moderately Variable",
            "Highly Variable",
        ],
        default="Insufficient Data",
    )

    return result


# ================================================================
# SECTION 24: SUMMARIZE DEMAND STABILITY
# ================================================================

def di3_summarize_stability(
    stability_dataframe: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Produce an overall summary from rolling stability metrics.
    """
    if stability_dataframe.empty:
        return {
            "status": "no_data",
        }

    cv_values = pd.to_numeric(
        stability_dataframe[
            "DI3_Rolling_CV_Percent"
        ],
        errors="coerce",
    ).dropna()

    labels = stability_dataframe[
        "DI3_Stability_Label"
    ].value_counts(
        dropna=False
    ).to_dict()

    if cv_values.empty:
        return {
            "status": "insufficient_data",
            "record_count": int(
                len(stability_dataframe)
            ),
        }

    return {
        "status": "success",
        "record_count": int(
            len(stability_dataframe)
        ),
        "mean_rolling_cv_percent": (
            di3_to_float(
                cv_values.mean()
            )
        ),
        "median_rolling_cv_percent": (
            di3_to_float(
                cv_values.median()
            )
        ),
        "maximum_rolling_cv_percent": (
            di3_to_float(
                cv_values.max()
            )
        ),
        "minimum_rolling_cv_percent": (
            di3_to_float(
                cv_values.min()
            )
        ),
        "stability_label_counts": {
            str(key): int(value)
            for key, value in labels.items()
        },
    }


# ================================================================
# SECTION 25: DAILY DEMAND VARIABILITY
# ================================================================

def di3_build_daily_variability(
    dataframe: pd.DataFrame,
    demand_column: str = "Sales Count",
) -> pd.DataFrame:
    """
    Aggregate demand by date and measure day-to-day variability.

    The daily total is the sum of the recorded demand values for
    each date. Missing intervals may cause undercounting.
    """
    if dataframe.empty:
        return pd.DataFrame()

    result = dataframe.copy()

    result["Timestamp"] = pd.to_datetime(
        result["Timestamp"],
        errors="coerce",
    )

    result[demand_column] = pd.to_numeric(
        result[demand_column],
        errors="coerce",
    )

    result = result.dropna(
        subset=[
            "Timestamp",
            demand_column,
        ]
    )

    result["DI3_Date"] = (
        result["Timestamp"].dt.normalize()
    )

    daily = result.groupby(
        "DI3_Date"
    ).agg(
        Daily_Demand=(
            demand_column,
            "sum",
        ),
        Record_Count=(
            demand_column,
            "count",
        ),
        Mean_Record_Demand=(
            demand_column,
            "mean",
        ),
        Maximum_Record_Demand=(
            demand_column,
            "max",
        ),
        Minimum_Record_Demand=(
            demand_column,
            "min",
        ),
    ).reset_index()

    daily = daily.sort_values(
        "DI3_Date"
    ).reset_index(
        drop=True
    )

    daily["Previous_Day_Total"] = (
        daily["Daily_Demand"].shift(1)
    )

    daily["Day_To_Day_Change"] = (
        daily["Daily_Demand"]
        - daily["Previous_Day_Total"]
    )

    daily["Day_To_Day_Change_Percent"] = (
        daily.apply(
            lambda row: (
                di3_safe_percentage(
                    row["Day_To_Day_Change"],
                    row["Previous_Day_Total"],
                )
                if pd.notna(
                    row["Previous_Day_Total"]
                )
                and row["Previous_Day_Total"] != 0
                else np.nan
            ),
            axis=1,
        )
    )

    return daily


# ================================================================
# SECTION 26: DEMAND BY OBSERVATION PERIOD
# ================================================================

def di3_build_period_summary(
    dataframe: pd.DataFrame,
    period: str = "D",
    demand_column: str = "Sales Count",
) -> pd.DataFrame:
    """
    Resample observed demand into calendar periods.

    Examples:
        D  = daily
        W  = weekly
        MS = month start
        QS = quarter start
        YS = year start

    Only use a period alias supported by the installed pandas
    version.
    """
    if dataframe.empty:
        return pd.DataFrame()

    result = dataframe[
        [
            "Timestamp",
            demand_column,
        ]
    ].copy()

    result["Timestamp"] = pd.to_datetime(
        result["Timestamp"],
        errors="coerce",
    )

    result[demand_column] = pd.to_numeric(
        result[demand_column],
        errors="coerce",
    )

    result = result.dropna(
        subset=[
            "Timestamp",
            demand_column,
        ]
    ).sort_values(
        "Timestamp"
    )

    if result.empty:
        return pd.DataFrame()

    result = result.set_index(
        "Timestamp"
    )

    grouped = result[
        demand_column
    ].resample(
        period
    )

    summary = grouped.agg(
        Observed_Demand="sum",
        Record_Count="count",
        Mean_Record_Demand="mean",
        Maximum_Record_Demand="max",
        Minimum_Record_Demand="min",
    ).reset_index()

    summary["Period_Label"] = (
        summary["Timestamp"].astype(str)
    )

    summary["Previous_Period_Demand"] = (
        summary["Observed_Demand"].shift(1)
    )

    summary["Period_Change"] = (
        summary["Observed_Demand"]
        - summary["Previous_Period_Demand"]
    )

    summary["Period_Change_Percent"] = (
        summary.apply(
            lambda row: (
                di3_safe_percentage(
                    row["Period_Change"],
                    row["Previous_Period_Demand"],
                )
                if pd.notna(
                    row["Previous_Period_Demand"]
                )
                and row["Previous_Period_Demand"] != 0
                else np.nan
            ),
            axis=1,
        )
    )

    return summary


# ================================================================
# SECTION 27: DEMAND SEASONALITY COMPARISON
# ================================================================

def di3_compare_calendar_groups(
    dataframe: pd.DataFrame,
    timestamp_column: str = "Timestamp",
    demand_column: str = "Sales Count",
    group_by: str = "hour",
) -> pd.DataFrame:
    """
    Compare demand across selected calendar groups.

    Supported values:
        hour
        weekday
        month
        quarter
        year
        weekend

    This comparison describes historical patterns and does not
    prove that the pattern will persist in future periods.
    """
    prepared = di3_prepare_dataframe(
        dataframe=dataframe,
        timestamp_column=timestamp_column,
        demand_column=demand_column,
    )

    if prepared.empty:
        return pd.DataFrame()

    if group_by == "hour":
        prepared["DI3_Group"] = (
            prepared[timestamp_column].dt.hour
        )

    elif group_by == "weekday":
        prepared["DI3_Group"] = (
            prepared[timestamp_column].dt.day_name()
        )

    elif group_by == "month":
        prepared["DI3_Group"] = (
            prepared[timestamp_column].dt.month_name()
        )

    elif group_by == "quarter":
        prepared["DI3_Group"] = (
            "Q"
            + prepared[timestamp_column].dt.quarter.astype(str)
        )

    elif group_by == "year":
        prepared["DI3_Group"] = (
            prepared[timestamp_column].dt.year
        )

    elif group_by == "weekend":
        prepared["DI3_Group"] = np.where(
            prepared[timestamp_column].dt.dayofweek >= 5,
            "Weekend",
            "Weekday",
        )

    else:
        raise ValueError(
            "Unsupported group_by value: "
            f"{group_by}"
        )

    grouped = prepared.groupby(
        "DI3_Group",
        dropna=False,
    )[demand_column]

    summary = grouped.agg(
        Record_Count="count",
        Total_Demand="sum",
        Mean_Demand="mean",
        Median_Demand="median",
        Standard_Deviation="std",
        Minimum_Demand="min",
        Maximum_Demand="max",
    ).reset_index()

    summary["Standard_Deviation"] = (
        summary["Standard_Deviation"].fillna(0)
    )

    overall_mean = di3_to_float(
        prepared[demand_column].mean()
    )

    summary["Mean_Demand_Index"] = (
        summary["Mean_Demand"].apply(
            lambda value: (
                di3_safe_percentage(
                    value,
                    overall_mean,
                )
            )
        )
    )

    summary["Demand_Range"] = (
        summary["Maximum_Demand"]
        - summary["Minimum_Demand"]
    )

    return summary


# ================================================================
# SECTION 28: ANOMALY SUMMARY BY CALENDAR GROUP
# ================================================================

def di3_summarize_anomalies_by_hour(
    anomaly_dataframe: pd.DataFrame,
    timestamp_column: str = "Timestamp",
) -> pd.DataFrame:
    """
    Count flagged anomalies by hour of day.
    """
    if anomaly_dataframe.empty:
        return pd.DataFrame(
            columns=[
                "Hour",
                "Anomaly_Count",
            ]
        )

    result = anomaly_dataframe.copy()

    result[timestamp_column] = pd.to_datetime(
        result[timestamp_column],
        errors="coerce",
    )

    result = result.dropna(
        subset=[
            timestamp_column,
        ]
    )

    result["DI3_Hour"] = (
        result[timestamp_column].dt.hour
    )

    summary = result.groupby(
        "DI3_Hour"
    ).size().rename(
        "Anomaly_Count"
    ).reset_index()

    summary = summary.rename(
        columns={
            "DI3_Hour": "Hour",
        }
    )

    return summary.sort_values(
        "Anomaly_Count",
        ascending=False,
    ).reset_index(
        drop=True
    )


# ================================================================
# SECTION 29: ANOMALY SUMMARY BY WEEKDAY
# ================================================================

def di3_summarize_anomalies_by_weekday(
    anomaly_dataframe: pd.DataFrame,
    timestamp_column: str = "Timestamp",
) -> pd.DataFrame:
    """
    Count flagged anomalies by weekday.
    """
    if anomaly_dataframe.empty:
        return pd.DataFrame(
            columns=[
                "Weekday",
                "Anomaly_Count",
            ]
        )

    result = anomaly_dataframe.copy()

    result[timestamp_column] = pd.to_datetime(
        result[timestamp_column],
        errors="coerce",
    )

    result = result.dropna(
        subset=[
            timestamp_column,
        ]
    )

    result["DI3_Weekday"] = (
        result[timestamp_column].dt.day_name()
    )

    summary = result.groupby(
        "DI3_Weekday"
    ).size().rename(
        "Anomaly_Count"
    ).reset_index()

    summary = summary.rename(
        columns={
            "DI3_Weekday": "Weekday",
        }
    )

    weekday_order = {
        day: index
        for index, day in enumerate(
            DI2_WEEKDAY_NAMES
        )
    }

    summary["DI3_Weekday_Order"] = (
        summary["Weekday"].map(
            weekday_order
        )
    )

    return summary.sort_values(
        "DI3_Weekday_Order"
    ).drop(
        columns=[
            "DI3_Weekday_Order",
        ]
    ).reset_index(
        drop=True
    )


# ================================================================
# SECTION 30: ANOMALY SUMMARY BY MONTH
# ================================================================

def di3_summarize_anomalies_by_month(
    anomaly_dataframe: pd.DataFrame,
    timestamp_column: str = "Timestamp",
) -> pd.DataFrame:
    """
    Count flagged anomalies by calendar month.
    """
    if anomaly_dataframe.empty:
        return pd.DataFrame(
            columns=[
                "Month",
                "Month_Name",
                "Anomaly_Count",
            ]
        )

    result = anomaly_dataframe.copy()

    result[timestamp_column] = pd.to_datetime(
        result[timestamp_column],
        errors="coerce",
    )

    result = result.dropna(
        subset=[
            timestamp_column,
        ]
    )

    result["DI3_Month"] = (
        result[timestamp_column].dt.month
    )

    result["DI3_Month_Name"] = (
        result[timestamp_column].dt.month_name()
    )

    summary = result.groupby(
        [
            "DI3_Month",
            "DI3_Month_Name",
        ]
    ).size().rename(
        "Anomaly_Count"
    ).reset_index()

    summary = summary.rename(
        columns={
            "DI3_Month": "Month",
            "DI3_Month_Name": "Month_Name",
        }
    )

    return summary.sort_values(
        "Month"
    ).reset_index(
        drop=True
    )


# ================================================================
# SECTION 31: DEMAND QUALITY SCORE
# ================================================================

def di3_calculate_quality_score(
    dataframe: pd.DataFrame,
    timestamp_column: str = "Timestamp",
    demand_column: str = "Sales Count",
) -> Dict[str, Any]:
    """
    Calculate a transparent rule-based data quality score.

    This score measures selected data-quality issues. It is not
    an industry-standard certification or model accuracy score.

    Penalties:
    - Invalid timestamps.
    - Invalid numeric demand values.
    - Duplicate rows.
    - Negative demand records.

    Zero demand is not automatically penalized because it may be
    a valid observation.
    """
    if dataframe.empty:
        return {
            "status": "no_data",
            "quality_score": 0.0,
        }

    total_rows = len(dataframe)

    timestamps = pd.to_datetime(
        dataframe[timestamp_column],
        errors="coerce",
    )

    numeric = pd.to_numeric(
        dataframe[demand_column],
        errors="coerce",
    )

    invalid_timestamp_count = int(
        timestamps.isna().sum()
    )

    invalid_demand_count = int(
        numeric.isna().sum()
    )

    duplicate_row_count = int(
        dataframe.duplicated().sum()
    )

    negative_demand_count = int(
        (numeric < 0).sum()
    )

    invalid_timestamp_rate = (
        invalid_timestamp_count / total_rows
    )

    invalid_demand_rate = (
        invalid_demand_count / total_rows
    )

    duplicate_rate = (
        duplicate_row_count / total_rows
    )

    negative_rate = (
        negative_demand_count / total_rows
    )

    penalty = (
        35 * invalid_timestamp_rate
        + 35 * invalid_demand_rate
        + 20 * duplicate_rate
        + 10 * negative_rate
    )

    quality_score = max(
        0.0,
        min(
            100.0,
            100.0 - penalty * 100.0,
        ),
    )

    if quality_score >= 95:
        label = "Very Good"

    elif quality_score >= 85:
        label = "Good"

    elif quality_score >= 70:
        label = "Needs Review"

    else:
        label = "Poor"

    return {
        "status": "success",
        "score_method": "Rule-based weighted penalty",
        "quality_score": round(
            quality_score,
            2,
        ),
        "quality_label": label,
        "row_count": int(total_rows),
        "invalid_timestamp_count": (
            invalid_timestamp_count
        ),
        "invalid_demand_count": (
            invalid_demand_count
        ),
        "duplicate_row_count": (
            duplicate_row_count
        ),
        "negative_demand_count": (
            negative_demand_count
        ),
        "zero_demand_count": int(
            (numeric == 0).sum()
        ),
        "note": (
            "The score is a project-defined indicator, "
            "not an externally validated quality standard."
        ),
    }


# ================================================================
# SECTION 32: BUILD OPERATIONAL RECOMMENDATIONS
# ================================================================

def di3_build_recommendations(
    distribution_summary: Dict[str, Any],
    zero_demand_summary: Dict[str, Any],
    timestamp_quality_summary: Dict[str, Any],
    demand_quality_summary: Dict[str, Any],
    quality_score_summary: Dict[str, Any],
    anomaly_count: int,
    sudden_change_count: int,
) -> List[Dict[str, Any]]:
    """
    Create recommendations from observed data-quality and
    demand-pattern indicators.

    Recommendations are decision-support suggestions. They do
    not replace verified operational procedures.
    """
    recommendations = []

    if anomaly_count > 0:
        recommendations.append(
            {
                "priority": "Review",
                "category": "Demand Anomalies",
                "title": "Review unusual demand observations",
                "reason": (
                    f"{anomaly_count} records were flagged by "
                    "at least one anomaly-detection method."
                ),
                "recommended_action": (
                    "Review the timestamps and original records "
                    "before deciding whether the observations "
                    "are valid demand spikes or data issues."
                ),
                "evidence_type": "Observed data",
            }
        )

    if sudden_change_count > 0:
        recommendations.append(
            {
                "priority": "Review",
                "category": "Demand Changes",
                "title": "Investigate sudden demand changes",
                "reason": (
                    f"{sudden_change_count} records were flagged "
                    "for large consecutive-record changes."
                ),
                "recommended_action": (
                    "Check the associated time intervals and "
                    "compare the records with known events or "
                    "data collection changes."
                ),
                "evidence_type": "Observed data",
            }
        )

    invalid_timestamps = di3_to_int(
        timestamp_quality_summary.get(
            "invalid_timestamp_count",
            0,
        )
    )

    if invalid_timestamps > 0:
        recommendations.append(
            {
                "priority": "High",
                "category": "Data Quality",
                "title": "Resolve invalid timestamps",
                "reason": (
                    f"{invalid_timestamps} records have "
                    "unparseable timestamps."
                ),
                "recommended_action": (
                    "Inspect the original timestamp strings and "
                    "correct them only when a reliable source "
                    "value is available."
                ),
                "evidence_type": "Validation result",
            }
        )

    longer_gaps = di3_to_int(
        timestamp_quality_summary.get(
            "longer_than_expected_gap_count",
            0,
        )
    )

    if longer_gaps > 0:
        recommendations.append(
            {
                "priority": "Review",
                "category": "Timestamp Continuity",
                "title": "Investigate longer timestamp gaps",
                "reason": (
                    f"{longer_gaps} observed intervals exceed "
                    "the configured expected interval."
                ),
                "recommended_action": (
                    "Determine whether the gaps reflect planned "
                    "closures, collection outages, or missing "
                    "records before interpreting demand totals."
                ),
                "evidence_type": "Timestamp analysis",
            }
        )

    negative_demand = di3_to_int(
        demand_quality_summary.get(
            "negative_demand_count",
            0,
        )
    )

    if negative_demand > 0:
        recommendations.append(
            {
                "priority": "High",
                "category": "Demand Validation",
                "title": "Review negative demand values",
                "reason": (
                    f"{negative_demand} records contain negative "
                    "demand values."
                ),
                "recommended_action": (
                    "Check whether negative values represent "
                    "refunds, reversals, corrections, or invalid "
                    "records in the source system."
                ),
                "evidence_type": "Validation result",
            }
        )

    zero_count = di3_to_int(
        zero_demand_summary.get(
            "zero_demand_records",
            0,
        )
    )

    zero_percent = di3_to_float(
        zero_demand_summary.get(
            "zero_demand_percent",
            0.0,
        )
    )

    if zero_count > 0:
        recommendations.append(
            {
                "priority": "Monitor",
                "category": "Zero Demand",
                "title": "Review zero-demand periods",
                "reason": (
                    f"{zero_count} records "
                    f"({zero_percent:.2f}%) have zero demand."
                ),
                "recommended_action": (
                    "Compare zero-demand periods with operating "
                    "hours and data collection schedules before "
                    "classifying them as anomalies."
                ),
                "evidence_type": "Observed data",
            }
        )

    quality_score = di3_to_float(
        quality_score_summary.get(
            "quality_score",
            100.0,
        )
    )

    if quality_score < 85:
        recommendations.append(
            {
                "priority": "High",
                "category": "Data Quality",
                "title": "Improve data validation controls",
                "reason": (
                    f"The project-defined quality score is "
                    f"{quality_score:.2f} out of 100."
                ),
                "recommended_action": (
                    "Add source-level validation, retain an "
                    "audit trail for corrections, and monitor "
                    "timestamp and demand quality over time."
                ),
                "evidence_type": "Rule-based score",
            }
        )

    if not recommendations:
        recommendations.append(
            {
                "priority": "Informational",
                "category": "Monitoring",
                "title": "Continue routine demand monitoring",
                "reason": (
                    "No recommendation-triggering condition "
                    "was identified by the configured rules."
                ),
                "recommended_action": (
                    "Continue monitoring data quality and "
                    "review future observations periodically."
                ),
                "evidence_type": "Rule-based assessment",
            }
        )

    return recommendations


# ================================================================
# SECTION 33: BUILD A PART 3 SUMMARY
# ================================================================

def di3_build_summary(
    dataframe: pd.DataFrame,
    anomaly_dataframe: pd.DataFrame,
    sudden_change_dataframe: pd.DataFrame,
    zero_demand_summary: Dict[str, Any],
    timestamp_quality_summary: Dict[str, Any],
    demand_quality_summary: Dict[str, Any],
    quality_score_summary: Dict[str, Any],
    timestamp_column: str = "Timestamp",
    demand_column: str = "Sales Count",
) -> Dict[str, Any]:
    """
    Build a compact summary of the Part 3 analysis.
    """
    if dataframe.empty:
        return {
            "status": "no_data",
            "part": 3,
            "version": DI3_VERSION,
        }

    timestamps = pd.to_datetime(
        dataframe[timestamp_column],
        errors="coerce",
    )

    values = pd.to_numeric(
        dataframe[demand_column],
        errors="coerce",
    )

    return {
        "status": "success",
        "part": 3,
        "version": DI3_VERSION,
        "row_count": int(len(dataframe)),
        "valid_timestamp_count": int(
            timestamps.notna().sum()
        ),
        "valid_demand_count": int(
            values.notna().sum()
        ),
        "first_timestamp": str(
            timestamps.min()
        ),
        "last_timestamp": str(
            timestamps.max()
        ),
        "total_observed_demand": di3_to_float(
            values.sum()
        ),
        "mean_record_demand": di3_to_float(
            values.mean()
        ),
        "median_record_demand": di3_to_float(
            values.median()
        ),
        "maximum_record_demand": di3_to_float(
            values.max()
        ),
        "minimum_record_demand": di3_to_float(
            values.min()
        ),
        "combined_anomaly_count": int(
            len(anomaly_dataframe)
        ),
        "sudden_change_count": int(
            len(sudden_change_dataframe)
        ),
        "zero_demand_record_count": (
            zero_demand_summary.get(
                "zero_demand_records",
                0,
            )
        ),
        "negative_demand_record_count": (
            demand_quality_summary.get(
                "negative_demand_count",
                0,
            )
        ),
        "invalid_timestamp_count": (
            timestamp_quality_summary.get(
                "invalid_timestamp_count",
                0,
            )
        ),
        "quality_score": (
            quality_score_summary.get(
                "quality_score",
                None,
            )
        ),
        "interpretation_scope": (
            "Historical ticket demand and data-quality analysis. "
            "Not a direct measure of passenger queues, ferry "
            "capacity, or exact boarding demand."
        ),
    }


# ================================================================
# SECTION 34: CSV EXPORT HELPER
# ================================================================

def di3_export_dataframe(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> Dict[str, Any]:
    """
    Export one dataframe to a CSV file.
    """
    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise TypeError(
            "Expected a pandas DataFrame."
        )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    return {
        "path": str(output_path),
        "rows": int(len(dataframe)),
        "columns": int(len(dataframe.columns)),
        "size_bytes": int(
            output_path.stat().st_size
        ),
    }


# ================================================================
# SECTION 35: JSON EXPORT HELPER
# ================================================================

def di3_export_json(
    payload: Dict[str, Any],
    output_path: Path,
) -> Dict[str, Any]:
    """
    Export a Python dictionary to JSON.
    """
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    def convert_json_value(
        value: Any,
    ) -> Any:
        if isinstance(
            value,
            np.integer,
        ):
            return int(value)

        if isinstance(
            value,
            np.floating,
        ):
            numeric_value = float(value)

            if not np.isfinite(numeric_value):
                return None

            return numeric_value

        if isinstance(
            value,
            np.ndarray,
        ):
            return value.tolist()

        if isinstance(
            value,
            (
                pd.Timestamp,
                datetime,
            ),
        ):
            return value.isoformat()

        if isinstance(
            value,
            Path,
        ):
            return str(value)

        if isinstance(
            value,
            (
                np.bool_,
            ),
        ):
            return bool(value)

        if value is pd.NA:
            return None

        return str(value)

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=4,
            ensure_ascii=False,
            default=convert_json_value,
        )

    return {
        "path": str(output_path),
        "size_bytes": int(
            output_path.stat().st_size
        ),
    }


# ================================================================
# SECTION 36: EXPORT PART 3 RESULTS
# ================================================================

def di3_export_results(
    results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Export the primary Part 3 analysis tables and JSON reports.
    """
    directories = di3_create_output_directories()

    output_dir = directories[
        "output_dir"
    ]

    report_dir = directories[
        "report_dir"
    ]

    export_map = {
        "combined_anomalies": (
            DI3_ANOMALY_FILE
        ),
        "sudden_changes": (
            DI3_CHANGE_FILE
        ),
        "zero_demand_records": (
            DI3_ZERO_DEMAND_FILE
        ),
        "timestamp_gaps": (
            DI3_TIMESTAMP_QUALITY_FILE
        ),
        "missing_values": (
            "missing_value_report.csv"
        ),
        "stability_metrics": (
            DI3_STABILITY_FILE
        ),
        "daily_variability": (
            "daily_demand_variability.csv"
        ),
        "weekly_summary": (
            "weekly_demand_summary.csv"
        ),
        "monthly_summary": (
            "monthly_demand_summary.csv"
        ),
        "quarterly_summary": (
            "quarterly_demand_summary.csv"
        ),
        "yearly_summary": (
            "yearly_demand_summary.csv"
        ),
        "anomalies_by_hour": (
            "anomalies_by_hour.csv"
        ),
        "anomalies_by_weekday": (
            "anomalies_by_weekday.csv"
        ),
        "anomalies_by_month": (
            "anomalies_by_month.csv"
        ),
    }

    exported_files = {}

    for key, filename in export_map.items():
        dataframe = results.get(
            key
        )

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            continue

        exported_files[key] = (
            di3_export_dataframe(
                dataframe,
                output_dir / filename,
            )
        )

    json_payload = {
        "summary": results.get(
            "summary",
            {},
        ),
        "distribution_summary": results.get(
            "distribution_summary",
            {},
        ),
        "zero_demand_summary": results.get(
            "zero_demand_summary",
            {},
        ),
        "timestamp_quality_summary": results.get(
            "timestamp_quality_summary",
            {},
        ),
        "demand_quality_summary": results.get(
            "demand_quality_summary",
            {},
        ),
        "quality_score_summary": results.get(
            "quality_score_summary",
            {},
        ),
        "stability_summary": results.get(
            "stability_summary",
            {},
        ),
        "recommendations": results.get(
            "recommendations",
            [],
        ),
        "exported_files": exported_files,
    }

    summary_export = di3_export_json(
        json_payload,
        report_dir / DI3_SUMMARY_FILE,
    )

    recommendation_export = di3_export_json(
        {
            "recommendations": results.get(
                "recommendations",
                [],
            )
        },
        report_dir / DI3_RECOMMENDATIONS_FILE,
    )

    return {
        "status": "success",
        "csv_export_count": len(
            exported_files
        ),
        "csv_files": exported_files,
        "summary_json": summary_export,
        "recommendations_json": recommendation_export,
    }


# ================================================================
# SECTION 37: BUILD THE COMPLETE PART 3 ANALYSIS
# ================================================================

def di3_build_analysis(
    dataframe: pd.DataFrame,
    timestamp_column: str = "Timestamp",
    demand_column: str = "Sales Count",
    expected_interval_minutes: Optional[float] = 15.0,
) -> Dict[str, Any]:
    """
    Build the full Part 3 anomaly and quality analysis package.

    This function returns dataframes and summary dictionaries.
    It does not write output files unless the export wrapper
    is called separately.
    """
    valid, errors = di3_validate_input(
        dataframe=dataframe,
        timestamp_column=timestamp_column,
        demand_column=demand_column,
    )

    if not valid:
        raise ValueError(
            "Cannot build Part 3 analysis: "
            + "; ".join(errors)
        )

    # Preserve the raw dataframe for data-quality checks.
    raw_dataframe = dataframe.copy()

    # Prepare a valid time-series dataframe for analysis.
    prepared = di3_prepare_dataframe(
        dataframe=dataframe,
        timestamp_column=timestamp_column,
        demand_column=demand_column,
    )

    # Use the standard Timestamp name in helper functions
    # that expect the project's usual timestamp column.
    if timestamp_column != "Timestamp":
        prepared = prepared.rename(
            columns={
                timestamp_column: "Timestamp",
            }
        )

    if demand_column != "Sales Count":
        prepared = prepared.rename(
            columns={
                demand_column: "Sales Count",
            }
        )

    timestamp_column = "Timestamp"

    demand_column = "Sales Count"

    # ------------------------------------------------------------
    # Distribution analysis
    # ------------------------------------------------------------

    distribution_summary = (
        di3_build_distribution_summary(
            prepared,
            demand_column=demand_column,
        )
    )

    # ------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------

    zscore_anomalies = (
        di3_detect_zscore_anomalies(
            prepared,
            demand_column=demand_column,
        )
    )

    iqr_anomalies = (
        di3_detect_iqr_anomalies(
            prepared,
            demand_column=demand_column,
        )
    )

    mad_anomalies = (
        di3_detect_mad_anomalies(
            prepared,
            demand_column=demand_column,
        )
    )

    combined_anomalies = (
        di3_build_combined_anomalies(
            prepared,
            timestamp_column=timestamp_column,
            demand_column=demand_column,
        )
    )

    # ------------------------------------------------------------
    # Sudden changes
    # ------------------------------------------------------------

    sudden_changes = (
        di3_detect_sudden_changes(
            prepared,
            demand_column=demand_column,
        )
    )

    change_outliers = (
        di3_detect_change_outliers(
            prepared,
            demand_column=demand_column,
        )
    )

    demand_changes = (
        di3_calculate_demand_changes(
            prepared,
            demand_column=demand_column,
        )
    )

    # ------------------------------------------------------------
    # Zero and negative demand
    # ------------------------------------------------------------

    zero_demand_records = (
        di3_analyze_zero_demand(
            prepared,
            demand_column=demand_column,
        )
    )

    zero_demand_runs = (
        di3_find_zero_demand_runs(
            prepared,
            demand_column=demand_column,
            timestamp_column=timestamp_column,
        )
    )

    negative_demand_records = (
        di3_find_negative_demand(
            prepared,
            demand_column=demand_column,
        )
    )

    zero_demand_summary = (
        di3_build_zero_demand_summary(
            prepared,
            demand_column=demand_column,
        )
    )

    # ------------------------------------------------------------
    # Data quality
    # ------------------------------------------------------------

    missing_value_report = (
        di3_build_missing_value_report(
            raw_dataframe
        )
    )

    duplicate_report = (
        di3_build_duplicate_report(
            raw_dataframe
        )
    )

    timestamp_gaps = (
        di3_build_timestamp_gap_report(
            raw_dataframe,
            timestamp_column=timestamp_column,
            expected_interval_minutes=(
                expected_interval_minutes
            ),
        )
    )

    timestamp_quality_summary = (
        di3_build_timestamp_quality_summary(
            raw_dataframe,
            timestamp_column=timestamp_column,
            expected_interval_minutes=(
                expected_interval_minutes
            ),
        )
    )

    demand_quality_summary = (
        di3_build_demand_quality_report(
            raw_dataframe,
            demand_column=demand_column,
        )
    )

    quality_score_summary = (
        di3_calculate_quality_score(
            raw_dataframe,
            timestamp_column=timestamp_column,
            demand_column=demand_column,
        )
    )

    # ------------------------------------------------------------
    # Stability and daily variability
    # ------------------------------------------------------------

    stability_metrics = (
        di3_build_stability_metrics(
            prepared,
            demand_column=demand_column,
        )
    )

    stability_summary = (
        di3_summarize_stability(
            stability_metrics
        )
    )

    daily_variability = (
        di3_build_daily_variability(
            prepared,
            demand_column=demand_column,
        )
    )

    # ------------------------------------------------------------
    # Calendar summaries
    # ------------------------------------------------------------

    weekly_summary = (
        di3_build_period_summary(
            prepared,
            period="W",
            demand_column=demand_column,
        )
    )

    monthly_summary = (
        di3_build_period_summary(
            prepared,
            period="MS",
            demand_column=demand_column,
        )
    )

    quarterly_summary = (
        di3_build_period_summary(
            prepared,
            period="QS",
            demand_column=demand_column,
        )
    )

    yearly_summary = (
        di3_build_period_summary(
            prepared,
            period="YS",
            demand_column=demand_column,
        )
    )

    # ------------------------------------------------------------
    # Anomaly calendar summaries
    # ------------------------------------------------------------

    anomalies_by_hour = (
        di3_summarize_anomalies_by_hour(
            combined_anomalies,
            timestamp_column=timestamp_column,
        )
    )

    anomalies_by_weekday = (
        di3_summarize_anomalies_by_weekday(
            combined_anomalies,
            timestamp_column=timestamp_column,
        )
    )

    anomalies_by_month = (
        di3_summarize_anomalies_by_month(
            combined_anomalies,
            timestamp_column=timestamp_column,
        )
    )

    # ------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------

    recommendations = (
        di3_build_recommendations(
            distribution_summary=distribution_summary,
            zero_demand_summary=zero_demand_summary,
            timestamp_quality_summary=(
                timestamp_quality_summary
            ),
            demand_quality_summary=(
                demand_quality_summary
            ),
            quality_score_summary=(
                quality_score_summary
            ),
            anomaly_count=len(
                combined_anomalies
            ),
            sudden_change_count=len(
                sudden_changes
            ),
        )
    )

    # ------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------

    summary = di3_build_summary(
        dataframe=prepared,
        anomaly_dataframe=combined_anomalies,
        sudden_change_dataframe=sudden_changes,
        zero_demand_summary=zero_demand_summary,
        timestamp_quality_summary=(
            timestamp_quality_summary
        ),
        demand_quality_summary=(
            demand_quality_summary
        ),
        quality_score_summary=(
            quality_score_summary
        ),
        timestamp_column=timestamp_column,
        demand_column=demand_column,
    )

    summary["duplicate_report"] = (
        duplicate_report
    )

    summary["zscore_anomaly_count"] = int(
        len(zscore_anomalies)
    )

    summary["iqr_anomaly_count"] = int(
        len(iqr_anomalies)
    )

    summary["mad_anomaly_count"] = int(
        len(mad_anomalies)
    )

    summary["change_outlier_count"] = int(
        len(change_outliers)
    )

    summary["zero_demand_run_count"] = int(
        len(zero_demand_runs)
    )

    return {
        "prepared_dataframe": prepared,
        "distribution_summary": distribution_summary,
        "zscore_anomalies": zscore_anomalies,
        "iqr_anomalies": iqr_anomalies,
        "mad_anomalies": mad_anomalies,
        "combined_anomalies": combined_anomalies,
        "demand_changes": demand_changes,
        "sudden_changes": sudden_changes,
        "change_outliers": change_outliers,
        "zero_demand_records": zero_demand_records,
        "zero_demand_runs": zero_demand_runs,
        "negative_demand_records": negative_demand_records,
        "zero_demand_summary": zero_demand_summary,
        "missing_values": missing_value_report,
        "duplicate_report": duplicate_report,
        "timestamp_gaps": timestamp_gaps,
        "timestamp_quality_summary": timestamp_quality_summary,
        "demand_quality_summary": demand_quality_summary,
        "quality_score_summary": quality_score_summary,
        "stability_metrics": stability_metrics,
        "stability_summary": stability_summary,
        "daily_variability": daily_variability,
        "weekly_summary": weekly_summary,
        "monthly_summary": monthly_summary,
        "quarterly_summary": quarterly_summary,
        "yearly_summary": yearly_summary,
        "anomalies_by_hour": anomalies_by_hour,
        "anomalies_by_weekday": anomalies_by_weekday,
        "anomalies_by_month": anomalies_by_month,
        "recommendations": recommendations,
        "summary": summary,
    }


# ================================================================
# SECTION 38: RUN PART 3 WITH OPTIONAL EXPORT
# ================================================================

def di3_run_analysis(
    dataframe: pd.DataFrame,
    timestamp_column: str = "Timestamp",
    demand_column: str = "Sales Count",
    expected_interval_minutes: Optional[float] = 15.0,
    export_results: bool = True,
) -> Dict[str, Any]:
    """
    Run the complete Part 3 analysis.

    Parameters
    ----------
    dataframe:
        Input ticket-demand dataframe.

    timestamp_column:
        Name of the timestamp column.

    demand_column:
        Name of the demand column.

    expected_interval_minutes:
        Expected spacing between records. Use None if unknown.

    export_results:
        If True, write the results to the Part 3 output folders.
    """
    LOGGER.info(
        "Starting Part 3 demand anomaly and data-quality analysis."
    )

    results = di3_build_analysis(
        dataframe=dataframe,
        timestamp_column=timestamp_column,
        demand_column=demand_column,
        expected_interval_minutes=(
            expected_interval_minutes
        ),
    )

    if export_results:
        results["export_result"] = (
            di3_export_results(
                results
            )
        )

    LOGGER.info(
        "Part 3 demand anomaly and data-quality analysis completed."
    )

    return results


# ================================================================
# SECTION 39: PART 3 FUNCTION REGISTRY
# ================================================================

DI3_FUNCTION_REGISTRY = {
    "validate_input": di3_validate_input,
    "prepare_dataframe": di3_prepare_dataframe,
    "distribution_summary": di3_build_distribution_summary,
    "zscore_anomalies": di3_detect_zscore_anomalies,
    "iqr_anomalies": di3_detect_iqr_anomalies,
    "mad_anomalies": di3_detect_mad_anomalies,
    "combined_anomalies": di3_build_combined_anomalies,
    "calculate_demand_changes": di3_calculate_demand_changes,
    "sudden_changes": di3_detect_sudden_changes,
    "change_outliers": di3_detect_change_outliers,
    "zero_demand": di3_analyze_zero_demand,
    "zero_demand_summary": di3_build_zero_demand_summary,
    "zero_demand_runs": di3_find_zero_demand_runs,
    "negative_demand": di3_find_negative_demand,
    "missing_value_report": di3_build_missing_value_report,
    "duplicate_report": di3_build_duplicate_report,
    "timestamp_gap_report": di3_build_timestamp_gap_report,
    "timestamp_quality": di3_build_timestamp_quality_summary,
    "demand_quality": di3_build_demand_quality_report,
    "quality_score": di3_calculate_quality_score,
    "stability_metrics": di3_build_stability_metrics,
    "stability_summary": di3_summarize_stability,
    "daily_variability": di3_build_daily_variability,
    "period_summary": di3_build_period_summary,
    "calendar_comparison": di3_compare_calendar_groups,
    "build_recommendations": di3_build_recommendations,
    "build_analysis": di3_build_analysis,
    "run_analysis": di3_run_analysis,
}


# ================================================================
# SECTION 40: PART 3 METADATA
# ================================================================

DI3_METADATA = {
    "module": "demand_insights",
    "part": 3,
    "version": DI3_VERSION,
    "purpose": (
        "Demand anomaly detection, sudden-change analysis, "
        "data-quality assessment, and operational decision support."
    ),
    "anomaly_methods": [
        "Z-score",
        "Interquartile range",
        "Median absolute deviation",
    ],
    "analysis_categories": [
        "Demand distribution",
        "Anomaly detection",
        "Sudden demand changes",
        "Zero-demand periods",
        "Negative-demand validation",
        "Missing-value analysis",
        "Duplicate-record analysis",
        "Timestamp continuity",
        "Demand stability",
        "Daily variability",
        "Calendar comparisons",
        "Operational recommendations",
    ],
    "output_directory": str(
        DI3_OUTPUT_ROOT
    ),
    "report_directory": str(
        DI3_REPORT_ROOT
    ),
    "limitations": [
        "Anomalies are candidates for investigation, not proof of errors.",
        "Zero demand may be legitimate.",
        "Negative demand may represent refunds or corrections.",
        "Timestamp gaps can affect demand interpretation.",
        "The quality score is project-defined.",
        "Ticket demand is not ferry capacity.",
        "Ticket demand is not queue length.",
        "Historical anomalies do not predict future failures.",
    ],
}


# ================================================================
# SECTION 41: PUBLIC METADATA HELPERS
# ================================================================

def di3_get_metadata() -> Dict[str, Any]:
    """
    Return Part 3 metadata.
    """
    return dict(
        DI3_METADATA
    )


def di3_list_functions() -> List[str]:
    """
    Return the registered Part 3 function names.
    """
    return sorted(
        DI3_FUNCTION_REGISTRY.keys()
    )


def di3_get_function(
    name: str,
) -> Any:
    """
    Retrieve a Part 3 function by its registry name.
    """
    if name not in DI3_FUNCTION_REGISTRY:
        raise KeyError(
            f"Unknown Part 3 function: {name}"
        )

    return DI3_FUNCTION_REGISTRY[
        name
    ]


# ================================================================
# SECTION 42: PART 3 EXPORT NAMES
# ================================================================

DI3_PUBLIC_NAMES = [
    "DI3_VERSION",
    "DI3_METADATA",
    "DI3_FUNCTION_REGISTRY",
    "di3_get_metadata",
    "di3_list_functions",
    "di3_get_function",
    "di3_validate_input",
    "di3_prepare_dataframe",
    "di3_build_distribution_summary",
    "di3_detect_zscore_anomalies",
    "di3_detect_iqr_anomalies",
    "di3_detect_mad_anomalies",
    "di3_build_combined_anomalies",
    "di3_calculate_demand_changes",
    "di3_detect_sudden_changes",
    "di3_detect_change_outliers",
    "di3_analyze_zero_demand",
    "di3_build_zero_demand_summary",
    "di3_find_zero_demand_runs",
    "di3_find_negative_demand",
    "di3_build_missing_value_report",
    "di3_build_duplicate_report",
    "di3_build_timestamp_gap_report",
    "di3_build_timestamp_quality_summary",
    "di3_build_demand_quality_report",
    "di3_calculate_quality_score",
    "di3_build_stability_metrics",
    "di3_summarize_stability",
    "di3_build_daily_variability",
    "di3_build_period_summary",
    "di3_compare_calendar_groups",
    "di3_summarize_anomalies_by_hour",
    "di3_summarize_anomalies_by_weekday",
    "di3_summarize_anomalies_by_month",
    "di3_build_recommendations",
    "di3_build_summary",
    "di3_export_dataframe",
    "di3_export_json",
    "di3_export_results",
    "di3_build_analysis",
    "di3_run_analysis",
]


# ================================================================
# <<< END PART 3: DEMAND ANOMALY AND DATA QUALITY INTELLIGENCE
# ================================================================


if __name__ == "__main__":
    raise SystemExit(main())
