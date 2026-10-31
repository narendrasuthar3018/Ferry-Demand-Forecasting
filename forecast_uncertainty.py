```python
"""
Forecast Uncertainty and Prediction Interval Analysis
=====================================================

Project:
    Short-Term Ferry Ticket Demand Forecasting
    & Predictive Decision Support System

Purpose:
    1. Load existing model predictions.
    2. Validate prediction records and required columns.
    3. Calculate historical forecast-error distributions.
    4. Build empirical prediction intervals.
    5. Estimate interval coverage on chronological calibration data.
    6. Compare uncertainty across models, targets, and horizons.
    7. Generate uncertainty summaries and machine-readable reports.
    8. Produce charts for later Streamlit dashboard integration.

Important:
    Prediction intervals are empirical estimates based on historical
    residuals. They are not guarantees of future coverage.

    The input dataset contains historical Sales and Redemption counts.
    It does not establish exact future ferry capacity, queue length,
    boarding demand, or the probability of a specific event.
"""

from __future__ import annotations

import json
import logging
import math
import warnings

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "outputs"
FORECAST_DIR = OUTPUT_DIR / "forecasts"
REPORT_DIR = OUTPUT_DIR / "reports"
FIGURE_DIR = OUTPUT_DIR / "figures"

PREDICTIONS_PATH = FORECAST_DIR / "model_predictions.csv"

UNCERTAINTY_DIR = OUTPUT_DIR / "uncertainty"

UNCERTAINTY_REPORT_DIR = REPORT_DIR / "uncertainty"
UNCERTAINTY_FIGURE_DIR = FIGURE_DIR / "uncertainty"

LOG_PATH = UNCERTAINTY_REPORT_DIR / "forecast_uncertainty.log"

PREDICTION_INTERVALS_PATH = (
    UNCERTAINTY_DIR / "prediction_intervals.csv"
)

MODEL_UNCERTAINTY_PATH = (
    UNCERTAINTY_DIR / "model_uncertainty_summary.csv"
)

HORIZON_UNCERTAINTY_PATH = (
    UNCERTAINTY_DIR / "horizon_uncertainty_summary.csv"
)

TARGET_UNCERTAINTY_PATH = (
    UNCERTAINTY_DIR / "target_uncertainty_summary.csv"
)

CALIBRATION_PATH = (
    UNCERTAINTY_DIR / "interval_calibration_summary.csv"
)

ERROR_DISTRIBUTION_PATH = (
    UNCERTAINTY_DIR / "historical_error_distribution.csv"
)

UNCERTAINTY_JSON_PATH = (
    UNCERTAINTY_DIR / "uncertainty_summary.json"
)

UNCERTAINTY_README_PATH = (
    UNCERTAINTY_DIR / "uncertainty_methodology.json"
)

FIGURE_INTERVAL_WIDTH = (
    UNCERTAINTY_FIGURE_DIR / "interval_width_by_horizon.png"
)

FIGURE_COVERAGE = (
    UNCERTAINTY_FIGURE_DIR / "empirical_interval_coverage.png"
)

FIGURE_ERROR_DISTRIBUTION = (
    UNCERTAINTY_FIGURE_DIR / "historical_error_distribution.png"
)

FIGURE_MODEL_UNCERTAINTY = (
    UNCERTAINTY_FIGURE_DIR / "model_uncertainty_comparison.png"
)

REQUIRED_COLUMNS = {
    "Timestamp",
    "Horizon_Minutes",
    "Target",
    "Model",
    "Actual",
    "Predicted",
}

OPTIONAL_ERROR_COLUMNS = [
    "Error",
    "Absolute_Error",
]

HORIZONS = [15, 30, 60, 120]

TARGETS = ["Sales", "Redemption"]

# Nominal empirical interval levels.
INTERVAL_LEVELS = [0.80, 0.90, 0.95]

# Earlier chronological observations are used to estimate error
# quantiles. Later observations are reserved for interval assessment.
CALIBRATION_FRACTION = 0.70

# Prevent a handful of records from being treated as a stable
# uncertainty distribution.
MIN_CALIBRATION_ROWS = 100

# Limit the number of points displayed in diagnostic plots.
MAX_PLOT_POINTS = 3000

RANDOM_STATE = 42


# ============================================================
# 2. LOGGING AND DIRECTORY MANAGEMENT
# ============================================================

def setup_logging() -> None:
    """
    Create output directories and configure a project log.

    The function is safe to call more than once.
    """
    for directory in [
        UNCERTAINTY_DIR,
        UNCERTAINTY_REPORT_DIR,
        UNCERTAINTY_FIGURE_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()

    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format=(
                "%(asctime)s | %(levelname)s | %(message)s"
            ),
            handlers=[
                logging.FileHandler(
                    LOG_PATH,
                    encoding="utf-8",
                ),
                logging.StreamHandler(),
            ],
        )
    else:
        root_logger.setLevel(logging.INFO)


def ensure_output_directories() -> None:
    """Create every output directory required by this module."""
    directories = [
        OUTPUT_DIR,
        FORECAST_DIR,
        REPORT_DIR,
        FIGURE_DIR,
        UNCERTAINTY_DIR,
        UNCERTAINTY_REPORT_DIR,
        UNCERTAINTY_FIGURE_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# 3. SAFE NUMERIC AND JSON UTILITIES
# ============================================================

def safe_float(
    value: Any,
    default: float | None = None,
) -> float | None:
    """
    Convert a value into a finite float.

    NaN, positive infinity, negative infinity, and unconvertible
    values return the supplied default.
    """
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return default

    if not math.isfinite(converted):
        return default

    return converted


def json_safe(value: Any) -> Any:
    """
    Convert NumPy/Pandas values into JSON-compatible values.

    Non-finite numeric values are represented as null.
    """
    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        number = float(value)

        if not math.isfinite(number):
            return None

        return number

    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    if value is pd.NaT:
        return None

    if pd.isna(value):
        return None

    return value


def write_json(
    payload: dict[str, Any],
    destination: Path,
) -> None:
    """Write a JSON document using UTF-8 encoding."""
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with destination.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_safe(payload),
            file,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        )


# ============================================================
# 4. LOAD AND VALIDATE PREDICTIONS
# ============================================================

def load_predictions(
    input_path: Path = PREDICTIONS_PATH,
) -> pd.DataFrame:
    """
    Load the existing model prediction file.

    Required columns:
        Timestamp, Horizon_Minutes, Target, Model,
        Actual, Predicted.

    Error columns are recalculated to ensure consistency.
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {input_path}\n"
            "Generate model predictions using "
            "forecasting_models.py before running this module."
        )

    logging.info(
        "Loading predictions from %s",
        input_path,
    )

    df = pd.read_csv(
        input_path,
        low_memory=False,
    )

    if df.empty:
        raise ValueError(
            "The prediction file contains no records."
        )

    missing = REQUIRED_COLUMNS.difference(df.columns)

    if missing:
        raise ValueError(
            "Prediction file is missing required columns: "
            f"{sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    original_rows = len(df)

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce",
    )

    for column in [
        "Horizon_Minutes",
        "Actual",
        "Predicted",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["Target"] = (
        df["Target"]
        .astype("string")
        .str.strip()
    )

    df["Model"] = (
        df["Model"]
        .astype("string")
        .str.strip()
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna(
        subset=[
            "Timestamp",
            "Horizon_Minutes",
            "Target",
            "Model",
            "Actual",
            "Predicted",
        ]
    ).copy()

    df = df.loc[
        df["Horizon_Minutes"] > 0
    ].copy()

    if df.empty:
        raise ValueError(
            "No valid prediction rows remain after cleaning."
        )

    df["Horizon_Minutes"] = (
        df["Horizon_Minutes"].astype(int)
    )

    df["Target"] = df["Target"].astype(str)
    df["Model"] = df["Model"].astype(str)

    # The residual convention used throughout this module is:
    # Error = Actual - Predicted.
    #
    # A positive error means the model underpredicted demand.
    # A negative error means the model overpredicted demand.
    df["Error"] = (
        df["Actual"] - df["Predicted"]
    )

    df["Absolute_Error"] = df["Error"].abs()

    df["Squared_Error"] = (
        df["Error"] ** 2
    )

    df = df.sort_values(
        [
            "Target",
            "Model",
            "Horizon_Minutes",
            "Timestamp",
        ]
    ).reset_index(drop=True)

    logging.info(
        "Loaded %s valid rows from %s original rows.",
        f"{len(df):,}",
        f"{original_rows:,}",
    )

    logging.info(
        "Targets: %s",
        sorted(df["Target"].unique().tolist()),
    )

    logging.info(
        "Models: %s",
        sorted(df["Model"].unique().tolist()),
    )

    logging.info(
        "Horizons: %s",
        sorted(
            df["Horizon_Minutes"]
            .unique()
            .tolist()
        ),
    )

    return df


# ============================================================
# 5. QUANTILE AND ERROR STATISTICS
# ============================================================

def calculate_error_statistics(
    errors: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """
    Summarize an empirical residual distribution.

    This is descriptive analysis, not a claim that residuals
    follow a normal distribution.
    """
    values = np.asarray(
        errors,
        dtype=float,
    )

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "Sample_Count": 0,
            "Mean_Error": None,
            "Median_Error": None,
            "Error_Std": None,
            "MAE": None,
            "RMSE": None,
            "P05_Error": None,
            "P10_Error": None,
            "P25_Error": None,
            "P75_Error": None,
            "P90_Error": None,
            "P95_Error": None,
            "P99_Error": None,
        }

    squared_values = np.square(values)

    return {
        "Sample_Count": int(len(values)),
        "Mean_Error": float(np.mean(values)),
        "Median_Error": float(np.median(values)),
        "Error_Std": float(np.std(values, ddof=1))
        if len(values) > 1
        else 0.0,
        "MAE": float(np.mean(np.abs(values))),
        "RMSE": float(
            np.sqrt(np.mean(squared_values))
        ),
        "P05_Error": float(np.quantile(values, 0.05)),
        "P10_Error": float(np.quantile(values, 0.10)),
        "P25_Error": float(np.quantile(values, 0.25)),
        "P75_Error": float(np.quantile(values, 0.75)),
        "P90_Error": float(np.quantile(values, 0.90)),
        "P95_Error": float(np.quantile(values, 0.95)),
        "P99_Error": float(np.quantile(values, 0.99)),
    }


def calculate_absolute_error_statistics(
    errors: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """Summarize absolute forecast errors."""
    values = np.asarray(
        errors,
        dtype=float,
    )

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "Absolute_Error_P50": None,
            "Absolute_Error_P80": None,
            "Absolute_Error_P90": None,
            "Absolute_Error_P95": None,
            "Absolute_Error_P99": None,
            "Maximum_Absolute_Error": None,
        }

    absolute = np.abs(values)

    return {
        "Absolute_Error_P50": float(
            np.quantile(absolute, 0.50)
        ),
        "Absolute_Error_P80": float(
            np.quantile(absolute, 0.80)
        ),
        "Absolute_Error_P90": float(
            np.quantile(absolute, 0.90)
        ),
        "Absolute_Error_P95": float(
            np.quantile(absolute, 0.95)
        ),
        "Absolute_Error_P99": float(
            np.quantile(absolute, 0.99)
        ),
        "Maximum_Absolute_Error": float(
            np.max(absolute)
        ),
    }


# ============================================================
# 6. CHRONOLOGICAL CALIBRATION SPLIT
# ============================================================

def chronological_calibration_split(
    group: pd.DataFrame,
    calibration_fraction: float = CALIBRATION_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a single model/target/horizon group by timestamp.

    Earlier observations estimate residual quantiles.
    Later observations evaluate interval coverage.

    This prevents intervals from being assessed only on the same
    records used to estimate their residual quantiles.

    Important limitation:
        These are historical holdout intervals. They are not a
        substitute for a rolling-origin evaluation or a formal
        conformal prediction guarantee under exchangeability.
    """
    if not 0.10 <= calibration_fraction <= 0.90:
        raise ValueError(
            "calibration_fraction must be between 0.10 and 0.90."
        )

    ordered = group.sort_values(
        "Timestamp"
    ).reset_index(drop=True)

    row_count = len(ordered)

    if row_count < 2:
        return ordered.copy(), ordered.iloc[0:0].copy()

    split_index = int(
        row_count * calibration_fraction
    )

    split_index = min(
        max(split_index, 1),
        row_count - 1,
    )

    calibration = ordered.iloc[
        :split_index
    ].copy()

    evaluation = ordered.iloc[
        split_index:
    ].copy()

    return calibration, evaluation


# ============================================================
# 7. EMPIRICAL PREDICTION INTERVALS
# ============================================================

def empirical_interval_bounds(
    predicted: float,
    calibration_errors: np.ndarray,
    interval_level: float,
) -> tuple[float, float]:
    """
    Calculate an empirical residual-based prediction interval.

    Error convention:
        Error = Actual - Predicted

    Interval:
        Lower = Prediction + lower residual quantile
        Upper = Prediction + upper residual quantile

    Bounds are not clipped to zero here because clipping would
    change the interval's empirical coverage characteristics.
    Dashboard consumers may separately display nonnegative
    demand estimates where appropriate.
    """
    if not 0.0 < interval_level < 1.0:
        raise ValueError(
            "interval_level must be between 0 and 1."
        )

    errors = np.asarray(
        calibration_errors,
        dtype=float,
    )

    errors = errors[np.isfinite(errors)]

    if len(errors) == 0:
        return (
            float("nan"),
            float("nan"),
        )

    alpha = 1.0 - interval_level

    lower_quantile = float(
        np.quantile(
            errors,
            alpha / 2.0,
        )
    )

    upper_quantile = float(
        np.quantile(
            errors,
            1.0 - alpha / 2.0,
        )
    )

    lower = predicted + lower_quantile
    upper = predicted + upper_quantile

    return (
        float(lower),
        float(upper),
    )


def build_prediction_intervals(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate empirical intervals for each model, target, and horizon.

    Returns:
        interval_records:
            One row per evaluation prediction and interval level.

        calibration_records:
            One row per model/target/horizon/interval level.
    """
    interval_records: list[dict[str, Any]] = []
    calibration_records: list[dict[str, Any]] = []

    grouping_columns = [
        "Target",
        "Model",
        "Horizon_Minutes",
    ]

    grouped = predictions.groupby(
        grouping_columns,
        sort=True,
        dropna=False,
    )

    logging.info(
        "Building empirical intervals for %s groups.",
        grouped.ngroups,
    )

    for (
        target,
        model,
        horizon,
    ), group in grouped:
        calibration, evaluation = (
            chronological_calibration_split(group)
        )

        calibration_errors = (
            calibration["Error"]
            .to_numpy(dtype=float)
        )

        calibration_errors = calibration_errors[
            np.isfinite(calibration_errors)
        ]

        evaluation = evaluation.copy()

        if len(calibration_errors) < MIN_CALIBRATION_ROWS:
            logging.warning(
                "Small calibration sample for "
                "target=%s, model=%s, horizon=%s: %s rows.",
                target,
                model,
                horizon,
                len(calibration_errors),
            )

        # A group with no valid calibration errors cannot produce
        # meaningful empirical intervals.
        if len(calibration_errors) == 0:
            continue

        for level in INTERVAL_LEVELS:
            interval_name = (
                f"{int(level * 100)}%"
            )

            lower_bounds: list[float] = []
            upper_bounds: list[float] = []

            actual_values = evaluation[
                "Actual"
            ].to_numpy(dtype=float)

            predicted_values = evaluation[
                "Predicted"
            ].to_numpy(dtype=float)

            for predicted in predicted_values:
                lower, upper = empirical_interval_bounds(
                    predicted=predicted,
                    calibration_errors=calibration_errors,
                    interval_level=level,
                )

                lower_bounds.append(lower)
                upper_bounds.append(upper)

            lower_array = np.asarray(
                lower_bounds,
                dtype=float,
            )

            upper_array = np.asarray(
                upper_bounds,
                dtype=float,
            )

            valid = (
                np.isfinite(actual_values)
                & np.isfinite(predicted_values)
                & np.isfinite(lower_array)
                & np.isfinite(upper_array)
            )

            actual_valid = actual_values[valid]
            predicted_valid = predicted_values[valid]
            lower_valid = lower_array[valid]
            upper_valid = upper_array[valid]

            inside = (
                (actual_valid >= lower_valid)
                & (actual_valid <= upper_valid)
            )

            interval_widths = (
                upper_valid - lower_valid
            )

            if len(actual_valid) > 0:
                empirical_coverage = float(
                    np.mean(inside)
                )

                mean_width = float(
                    np.mean(interval_widths)
                )

                median_width = float(
                    np.median(interval_widths)
                )

                mean_absolute_error = float(
                    np.mean(
                        np.abs(
                            actual_valid
                            - predicted_valid
                        )
                    )
                )
            else:
                empirical_coverage = None
                mean_width = None
                median_width = None
                mean_absolute_error = None

            calibration_records.append(
                {
                    "Target": target,
                    "Model": model,
                    "Horizon_Minutes": int(horizon),
                    "Interval_Level": interval_name,
                    "Nominal_Coverage": float(level),
                    "Calibration_Rows": int(
                        len(calibration_errors)
                    ),
                    "Evaluation_Rows": int(
                        len(actual_valid)
                    ),
                    "Calibration_Start": (
                        calibration["Timestamp"].min()
                        if not calibration.empty
                        else None
                    ),
                    "Calibration_End": (
                        calibration["Timestamp"].max()
                        if not calibration.empty
                        else None
                    ),
                    "Evaluation_Start": (
                        evaluation["Timestamp"].min()
                        if not evaluation.empty
                        else None
                    ),
                    "Evaluation_End": (
                        evaluation["Timestamp"].max()
                        if not evaluation.empty
                        else None
                    ),
                    "Empirical_Coverage": empirical_coverage,
                    "Coverage_Percent": (
                        empirical_coverage * 100.0
                        if empirical_coverage is not None
                        else None
                    ),
                    "Coverage_Gap": (
                        empirical_coverage - level
                        if empirical_coverage is not None
                        else None
                    ),
                    "Mean_Interval_Width": mean_width,
                    "Median_Interval_Width": median_width,
                    "Evaluation_MAE": mean_absolute_error,
                    "Minimum_Calibration_Rows_Recommended": (
                        MIN_CALIBRATION_ROWS
                    ),
                    "Calibration_Sufficient": (
                        len(calibration_errors)
                        >= MIN_CALIBRATION_ROWS
                    ),
                }
            )

            for index, (_, row) in enumerate(
                evaluation.iterrows()
            ):
                if index >= len(lower_array):
                    break

                lower = lower_array[index]
                upper = upper_array[index]
                actual = safe_float(row["Actual"])
                predicted = safe_float(row["Predicted"])

                if (
                    actual is None
                    or predicted is None
                    or not math.isfinite(lower)
                    or not math.isfinite(upper)
                ):
                    continue

                inside_interval = (
                    lower <= actual <= upper
                )

                interval_records.append(
                    {
                        "Timestamp": row["Timestamp"],
                        "Horizon_Minutes": int(horizon),
                        "Target": target,
                        "Model": model,
                        "Actual": actual,
                        "Predicted": predicted,
                        "Interval_Level": interval_name,
                        "Nominal_Coverage": float(level),
                        "Lower_Bound": float(lower),
                        "Upper_Bound": float(upper),
                        "Interval_Width": float(
                            upper - lower
                        ),
                        "Actual_Inside_Interval": bool(
                            inside_interval
                        ),
                        "Distance_From_Prediction": float(
                            abs(actual - predicted)
                        ),
                        "Absolute_Error": float(
                            abs(actual - predicted)
                        ),
                    }
                )

    intervals_df = pd.DataFrame(
        interval_records
    )

    calibration_df = pd.DataFrame(
        calibration_records
    )

    if intervals_df.empty:
        logging.warning(
            "No prediction intervals were generated."
        )

    if calibration_df.empty:
        logging.warning(
            "No calibration summary records were generated."
        )

    return intervals_df, calibration_df


# ============================================================
# 8. HISTORICAL ERROR DISTRIBUTION REPORT
# ============================================================

def build_error_distribution_report(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create residual-distribution statistics for every
    model/target/horizon combination.
    """
    records: list[dict[str, Any]] = []

    grouping_columns = [
        "Target",
        "Model",
        "Horizon_Minutes",
    ]

    for keys, group in predictions.groupby(
        grouping_columns,
        sort=True,
    ):
        target, model, horizon = keys

        error_stats = calculate_error_statistics(
            group["Error"]
        )

        absolute_stats = (
            calculate_absolute_error_statistics(
                group["Error"]
            )
        )

        records.append(
            {
                "Target": target,
                "Model": model,
                "Horizon_Minutes": int(horizon),
                "Start_Timestamp": group["Timestamp"].min(),
                "End_Timestamp": group["Timestamp"].max(),
                **error_stats,
                **absolute_stats,
            }
        )

    return pd.DataFrame(records)


# ============================================================
# 9. MODEL UNCERTAINTY SUMMARY
# ============================================================

def build_model_uncertainty_summary(
    predictions: pd.DataFrame,
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize historical error magnitude and interval width
    by model, target, and forecast horizon.
    """
    records: list[dict[str, Any]] = []

    grouping_columns = [
        "Target",
        "Model",
        "Horizon_Minutes",
    ]

    calibration_lookup: dict[
        tuple[str, str, int, str],
        dict[str, Any],
    ] = {}

    if not calibration.empty:
        for _, row in calibration.iterrows():
            key = (
                str(row["Target"]),
                str(row["Model"]),
                int(row["Horizon_Minutes"]),
                str(row["Interval_Level"]),
            )

            calibration_lookup[key] = row.to_dict()

    for keys, group in predictions.groupby(
        grouping_columns,
        sort=True,
    ):
        target, model, horizon = keys

        errors = group["Error"].to_numpy(
            dtype=float
        )

        errors = errors[
            np.isfinite(errors)
        ]

        if len(errors) == 0:
            continue

        absolute_errors = np.abs(errors)

        row: dict[str, Any] = {
            "Target": target,
            "Model": model,
            "Horizon_Minutes": int(horizon),
            "Prediction_Count": int(len(group)),
            "Mean_Absolute_Error": float(
                np.mean(absolute_errors)
            ),
            "Median_Absolute_Error": float(
                np.median(absolute_errors)
            ),
            "P90_Absolute_Error": float(
                np.quantile(absolute_errors, 0.90)
            ),
            "P95_Absolute_Error": float(
                np.quantile(absolute_errors, 0.95)
            ),
            "Error_Standard_Deviation": float(
                np.std(errors, ddof=1)
            ) if len(errors) > 1 else 0.0,
            "Mean_Signed_Error": float(
                np.mean(errors)
            ),
        }

        for level in INTERVAL_LEVELS:
            interval_name = (
                f"{int(level * 100)}%"
            )

            key = (
                str(target),
                str(model),
                int(horizon),
                interval_name,
            )

            summary = calibration_lookup.get(
                key,
                {},
            )

            suffix = interval_name.replace("%", "")

            row[
                f"Coverage_{suffix}pct"
            ] = summary.get(
                "Coverage_Percent"
            )

            row[
                f"Mean_Width_{suffix}pct"
            ] = summary.get(
                "Mean_Interval_Width"
            )

        records.append(row)

    result = pd.DataFrame(records)

    if not result.empty:
        result = result.sort_values(
            [
                "Target",
                "Horizon_Minutes",
                "Mean_Absolute_Error",
            ]
        ).reset_index(drop=True)

    return result


# ============================================================
# 10. HORIZON UNCERTAINTY SUMMARY
# ============================================================

def build_horizon_uncertainty_summary(
    model_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate model-level uncertainty by target and horizon.

    Multiple models are summarized separately through median
    and minimum/maximum metrics so one model does not silently
    represent the entire horizon.
    """
    if model_summary.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []

    grouping_columns = [
        "Target",
        "Horizon_Minutes",
    ]

    for keys, group in model_summary.groupby(
        grouping_columns,
        sort=True,
    ):
        target, horizon = keys

        record: dict[str, Any] = {
            "Target": target,
            "Horizon_Minutes": int(horizon),
            "Models_Evaluated": int(
                group["Model"].nunique()
            ),
            "Median_Model_MAE": float(
                group["Mean_Absolute_Error"].median()
            ),
            "Minimum_Model_MAE": float(
                group["Mean_Absolute_Error"].min()
            ),
            "Maximum_Model_MAE": float(
                group["Mean_Absolute_Error"].max()
            ),
            "Median_P90_Absolute_Error": float(
                group["P90_Absolute_Error"].median()
            ),
            "Median_P95_Absolute_Error": float(
                group["P95_Absolute_Error"].median()
            ),
            "Median_Error_Standard_Deviation": float(
                group["Error_Standard_Deviation"].median()
            ),
        }

        for level in INTERVAL_LEVELS:
            suffix = str(
                int(level * 100)
            )

            width_column = (
                f"Mean_Width_{suffix}pct"
            )

            coverage_column = (
                f"Coverage_{suffix}pct"
            )

            if width_column in group.columns:
                record[
                    f"Median_Width_{suffix}pct"
                ] = safe_float(
                    group[width_column].median()
                )

            if coverage_column in group.columns:
                record[
                    f"Median_Coverage_{suffix}pct"
                ] = safe_float(
                    group[coverage_column].median()
                )

        records.append(record)

    return pd.DataFrame(records)


# ============================================================
# 11. TARGET UNCERTAINTY SUMMARY
# ============================================================

def build_target_uncertainty_summary(
    model_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize error behavior for Sales and Redemption separately.
    """
    if model_summary.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []

    for target, group in model_summary.groupby(
        "Target",
        sort=True,
    ):
        records.append(
            {
                "Target": target,
                "Models_And_Horizons": int(len(group)),
                "Unique_Models": int(
                    group["Model"].nunique()
                ),
                "Unique_Horizons": int(
                    group["Horizon_Minutes"].nunique()
                ),
                "Median_MAE": safe_float(
                    group["Mean_Absolute_Error"].median()
                ),
                "Median_P90_Absolute_Error": safe_float(
                    group["P90_Absolute_Error"].median()
                ),
                "Median_P95_Absolute_Error": safe_float(
                    group["P95_Absolute_Error"].median()
                ),
                "Maximum_MAE": safe_float(
                    group["Mean_Absolute_Error"].max()
                ),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# 12. MODEL RANKING
# ============================================================

def build_uncertainty_ranking(
    model_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rank models within each target and horizon by historical MAE.

    This is an uncertainty/error ranking, not a claim that the
    lowest-MAE model will always be best in future periods.
    """
    if model_summary.empty:
        return pd.DataFrame()

    ranked = model_summary.copy()

    ranked["MAE_Rank"] = (
        ranked.groupby(
            ["Target", "Horizon_Minutes"]
        )["Mean_Absolute_Error"]
        .rank(
            method="dense",
            ascending=True,
        )
        .astype(int)
    )

    ranked["Error_Risk_Category"] = pd.cut(
        ranked["P90_Absolute_Error"],
        bins=[
            -np.inf,
            10,
            25,
            50,
            np.inf,
        ],
        labels=[
            "Lower Historical Error",
            "Moderate Historical Error",
            "Elevated Historical Error",
            "High Historical Error",
        ],
        include_lowest=True,
    ).astype(str)

    return ranked.sort_values(
        [
            "Target",
            "Horizon_Minutes",
            "MAE_Rank",
        ]
    ).reset_index(drop=True)


# ============================================================
# 13. COVERAGE QUALITY FLAGS
# ============================================================

def add_calibration_quality_flags(
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add descriptive coverage diagnostics.

    The tolerance is a reporting threshold, not a statistical
    guarantee. Small samples can produce unstable coverage.
    """
    if calibration.empty:
        return calibration.copy()

    result = calibration.copy()

    def classify(row: pd.Series) -> str:
        coverage = safe_float(
            row.get("Empirical_Coverage")
        )

        nominal = safe_float(
            row.get("Nominal_Coverage")
        )

        count = safe_float(
            row.get("Evaluation_Rows"),
            default=0.0,
        )

        sufficient = bool(
            row.get("Calibration_Sufficient", False)
        )

        if (
            coverage is None
            or nominal is None
            or count is None
            or count == 0
        ):
            return "Insufficient Evaluation Data"

        if not sufficient:
            return "Limited Calibration Sample"

        gap = coverage - nominal

        if gap < -0.10:
            return "Under-Coverage Observed"

        if gap > 0.10:
            return "Over-Coverage Observed"

        return "Coverage Within Reporting Tolerance"

    result["Calibration_Quality_Flag"] = result.apply(
        classify,
        axis=1,
    )

    result["Interpretation"] = (
        "Historical holdout coverage is descriptive and "
        "does not guarantee future interval coverage."
    )

    return result


# ============================================================
# 14. EXPORT REPORTS
# ============================================================

def save_dataframe(
    df: pd.DataFrame,
    destination: Path,
) -> None:
    """Write a DataFrame to CSV without a row index."""
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        destination,
        index=False,
    )

    logging.info(
        "Saved %s rows to %s",
        f"{len(df):,}",
        destination,
    )


def export_uncertainty_reports(
    intervals: pd.DataFrame,
    calibration: pd.DataFrame,
    error_distribution: pd.DataFrame,
    model_summary: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    target_summary: pd.DataFrame,
    ranking: pd.DataFrame,
) -> dict[str, str]:
    """
    Export the module's principal CSV reports.
    """
    output_map = {
        "prediction_intervals": (
            intervals,
            PREDICTION_INTERVALS_PATH,
        ),
        "interval_calibration": (
            calibration,
            CALIBRATION_PATH,
        ),
        "historical_error_distribution": (
            error_distribution,
            ERROR_DISTRIBUTION_PATH,
        ),
        "model_uncertainty": (
            model_summary,
            MODEL_UNCERTAINTY_PATH,
        ),
        "horizon_uncertainty": (
            horizon_summary,
            HORIZON_UNCERTAINTY_PATH,
        ),
        "target_uncertainty": (
            target_summary,
            TARGET_UNCERTAINTY_PATH,
        ),
        "uncertainty_ranking": (
            ranking,
            UNCERTAINTY_DIR / "uncertainty_model_ranking.csv",
        ),
    }

    exported: dict[str, str] = {}

    for name, (dataframe, destination) in output_map.items():
        save_dataframe(
            dataframe,
            destination,
        )

        exported[name] = str(
            destination.relative_to(BASE_DIR)
        )

    return exported


# ============================================================
# 15. PLOT: INTERVAL WIDTH BY HORIZON
# ============================================================

def plot_interval_width_by_horizon(
    calibration: pd.DataFrame,
) -> None:
    """
    Compare mean interval widths across forecast horizons.
    """
    if calibration.empty:
        logging.warning(
            "Skipping interval-width chart: no calibration records."
        )
        return

    required = {
        "Target",
        "Horizon_Minutes",
        "Interval_Level",
        "Mean_Interval_Width",
    }

    if not required.issubset(calibration.columns):
        logging.warning(
            "Skipping interval-width chart: required columns missing."
        )
        return

    data = calibration.dropna(
        subset=[
            "Mean_Interval_Width",
            "Horizon_Minutes",
        ]
    )

    if data.empty:
        return

    aggregated = (
        data.groupby(
            [
                "Target",
                "Horizon_Minutes",
                "Interval_Level",
            ],
            as_index=False,
        )["Mean_Interval_Width"]
        .median()
    )

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    for (
        target,
        interval_level,
    ), group in aggregated.groupby(
        ["Target", "Interval_Level"],
        sort=True,
    ):
        group = group.sort_values(
            "Horizon_Minutes"
        )

        ax.plot(
            group["Horizon_Minutes"],
            group["Mean_Interval_Width"],
            marker="o",
            linewidth=2,
            label=f"{target} — {interval_level}",
        )

    ax.set_title(
        "Historical Prediction Interval Width by Horizon"
    )

    ax.set_xlabel(
        "Forecast Horizon (minutes)"
    )

    ax.set_ylabel(
        "Median Model Mean Interval Width"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        fontsize=8,
        ncol=2,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_INTERVAL_WIDTH,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    logging.info(
        "Saved interval-width chart: %s",
        FIGURE_INTERVAL_WIDTH,
    )


# ============================================================
# 16. PLOT: EMPIRICAL COVERAGE
# ============================================================

def plot_empirical_coverage(
    calibration: pd.DataFrame,
) -> None:
    """
    Plot nominal interval coverage against observed holdout coverage.
    """
    if calibration.empty:
        return

    data = calibration.dropna(
        subset=[
            "Nominal_Coverage",
            "Empirical_Coverage",
        ]
    ).copy()

    if data.empty:
        return

    aggregated = (
        data.groupby(
            [
                "Target",
                "Horizon_Minutes",
                "Interval_Level",
                "Nominal_Coverage",
            ],
            as_index=False,
        )["Empirical_Coverage"]
        .median()
    )

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    nominal_values = sorted(
        aggregated["Nominal_Coverage"]
        .unique()
        .tolist()
    )

    for nominal in nominal_values:
        ax.axhline(
            y=nominal,
            linestyle="--",
            linewidth=1,
            label=(
                f"Nominal coverage: "
                f"{nominal:.0%}"
            ),
        )

    for (
        target,
        interval_level,
    ), group in aggregated.groupby(
        ["Target", "Interval_Level"],
        sort=True,
    ):
        group = group.sort_values(
            "Horizon_Minutes"
        )

        ax.plot(
            group["Horizon_Minutes"],
            group["Empirical_Coverage"],
            marker="o",
            linewidth=2,
            label=f"{target} — {interval_level}",
        )

    ax.set_title(
        "Nominal vs Empirical Prediction Interval Coverage"
    )

    ax.set_xlabel(
        "Forecast Horizon (minutes)"
    )

    ax.set_ylabel(
        "Empirical Coverage"
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        fontsize=8,
        ncol=2,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_COVERAGE,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    logging.info(
        "Saved coverage chart: %s",
        FIGURE_COVERAGE,
    )


# ============================================================
# 17. PLOT: HISTORICAL ERROR DISTRIBUTION
# ============================================================

def plot_error_distribution(
    predictions: pd.DataFrame,
) -> None:
    """
    Plot residual histograms for representative model groups.

    Sampling is applied only to chart data; report calculations
    continue to use the complete prediction table.
    """
    if predictions.empty:
        return

    groups = list(
        predictions.groupby(
            [
                "Target",
                "Model",
                "Horizon_Minutes",
            ],
            sort=True,
        )
    )

    if not groups:
        return

    selected_groups = groups[:6]

    number_of_plots = len(
        selected_groups
    )

    columns = 2

    rows = int(
        math.ceil(number_of_plots / columns)
    )

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            13,
            max(4, rows * 3.5),
        ),
        squeeze=False,
    )

    for index, (keys, group) in enumerate(
        selected_groups
    ):
        target, model, horizon = keys

        ax = axes[
            index // columns
        ][
            index % columns
        ]

        errors = group["Error"].dropna()

        if len(errors) > MAX_PLOT_POINTS:
            errors = errors.sample(
                n=MAX_PLOT_POINTS,
                random_state=RANDOM_STATE,
            )

        ax.hist(
            errors,
            bins=40,
            alpha=0.8,
        )

        ax.axvline(
            0,
            linestyle="--",
            linewidth=1.5,
        )

        ax.set_title(
            f"{target} | {model} | {horizon} min"
        )

        ax.set_xlabel(
            "Actual − Predicted"
        )

        ax.set_ylabel(
            "Frequency"
        )

        ax.grid(
            True,
            alpha=0.2,
        )

    for index in range(
        number_of_plots,
        rows * columns,
    ):
        axes[
            index // columns
        ][
            index % columns
        ].set_visible(False)

    fig.suptitle(
        "Historical Forecast Error Distributions",
        fontsize=14,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.96,
        ]
    )

    fig.savefig(
        FIGURE_ERROR_DISTRIBUTION,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    logging.info(
        "Saved error distribution chart: %s",
        FIGURE_ERROR_DISTRIBUTION,
    )


# ============================================================
# 18. PLOT: MODEL UNCERTAINTY COMPARISON
# ============================================================

def plot_model_uncertainty(
    model_summary: pd.DataFrame,
) -> None:
    """
    Compare historical MAE by model and horizon for each target.
    """
    if model_summary.empty:
        return

    targets = sorted(
        model_summary["Target"]
        .dropna()
        .unique()
        .tolist()
    )

    if not targets:
        return

    fig, axes = plt.subplots(
        len(targets),
        1,
        figsize=(
            12,
            max(4, len(targets) * 4),
        ),
        squeeze=False,
    )

    for index, target in enumerate(targets):
        ax = axes[index][0]

        data = model_summary.loc[
            model_summary["Target"] == target
        ].copy()

        pivot = data.pivot_table(
            index="Horizon_Minutes",
            columns="Model",
            values="Mean_Absolute_Error",
            aggfunc="median",
        )

        if pivot.empty:
            ax.set_visible(False)
            continue

        pivot = pivot.sort_index()

        pivot.plot(
            kind="bar",
            ax=ax,
        )

        ax.set_title(
            f"{target}: Historical Model Error by Horizon"
        )

        ax.set_xlabel(
            "Forecast Horizon (minutes)"
        )

        ax.set_ylabel(
            "Mean Absolute Error"
        )

        ax.grid(
            axis="y",
            alpha=0.3,
        )

        ax.legend(
            fontsize=8,
        )

    fig.tight_layout()

    fig.savefig(
        FIGURE_MODEL_UNCERTAINTY,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    logging.info(
        "Saved model uncertainty chart: %s",
        FIGURE_MODEL_UNCERTAINTY,
    )


# ============================================================
# 19. METHODOLOGY REPORT
# ============================================================

def build_methodology_report(
    predictions: pd.DataFrame,
    intervals: pd.DataFrame,
    calibration: pd.DataFrame,
) -> dict[str, Any]:
    """
    Record the assumptions and limitations of the uncertainty method.
    """
    return {
        "project": (
            "Short-Term Ferry Ticket Demand Forecasting "
            "and Predictive Decision Support System"
        ),
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "input_file": str(
            PREDICTIONS_PATH.relative_to(BASE_DIR)
        ),
        "input_rows": int(
            len(predictions)
        ),
        "interval_rows": int(
            len(intervals)
        ),
        "calibration_groups": int(
            len(calibration)
        ),
        "targets": sorted(
            predictions["Target"].unique().tolist()
        ),
        "models": sorted(
            predictions["Model"].unique().tolist()
        ),
        "horizons_minutes": sorted(
            predictions["Horizon_Minutes"]
            .unique()
            .tolist()
        ),
        "interval_levels": INTERVAL_LEVELS,
        "calibration_fraction": CALIBRATION_FRACTION,
        "calibration_strategy": (
            "Chronological within-group split: earlier residuals "
            "estimate empirical quantiles and later observations "
            "assess empirical coverage."
        ),
        "residual_definition": (
            "Error = Actual - Predicted. Positive values indicate "
            "underprediction; negative values indicate "
            "overprediction."
        ),
        "interval_method": (
            "Empirical residual-quantile intervals. Lower and upper "
            "bounds are the point prediction plus the corresponding "
            "historical residual quantiles."
        ),
        "uncertainty_interpretation": (
            "Intervals describe uncertainty estimated from historical "
            "forecast errors. They are not guaranteed to achieve their "
            "nominal coverage on future data."
        ),
        "limitations": [
            (
                "Historical residual behavior can change as demand "
                "patterns, seasonality, or operating conditions change."
            ),
            (
                "The chronological split is a holdout assessment, "
                "not a guarantee of future calibration."
            ),
            (
                "Coverage is reported only for the evaluation portion "
                "of each model/target/horizon group."
            ),
            (
                "Groups with limited calibration observations should "
                "be interpreted cautiously."
            ),
            (
                "The dataset supports ticket Sales and Redemption "
                "forecasting; it does not independently establish "
                "exact capacity, queue length, or boarding outcomes."
            ),
            (
                "Intervals are empirical ranges, not proof that the "
                "future value has a specific probability of occurring."
            ),
        ],
    }


# ============================================================
# 20. HIGH-LEVEL SUMMARY
# ============================================================

def build_summary_json(
    predictions: pd.DataFrame,
    intervals: pd.DataFrame,
    calibration: pd.DataFrame,
    error_distribution: pd.DataFrame,
    model_summary: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    target_summary: pd.DataFrame,
    ranking: pd.DataFrame,
    exported_files: dict[str, str],
) -> dict[str, Any]:
    """
    Build a compact summary for application and dashboard integration.
    """
    best_models: list[dict[str, Any]] = []

    if not ranking.empty:
        best_rows = ranking.loc[
            ranking["MAE_Rank"] == 1
        ]

        for _, row in best_rows.iterrows():
            best_models.append(
                {
                    "target": row["Target"],
                    "horizon_minutes": int(
                        row["Horizon_Minutes"]
                    ),
                    "model": row["Model"],
                    "mae": safe_float(
                        row["Mean_Absolute_Error"]
                    ),
                    "p90_absolute_error": safe_float(
                        row["P90_Absolute_Error"]
                    ),
                }
            )

    coverage_summary: list[dict[str, Any]] = []

    if not calibration.empty:
        for _, row in calibration.iterrows():
            coverage_summary.append(
                {
                    "target": row["Target"],
                    "model": row["Model"],
                    "horizon_minutes": int(
                        row["Horizon_Minutes"]
                    ),
                    "interval_level": row["Interval_Level"],
                    "nominal_coverage": safe_float(
                        row["Nominal_Coverage"]
                    ),
                    "empirical_coverage": safe_float(
                        row["Empirical_Coverage"]
                    ),
                    "coverage_percent": safe_float(
                        row["Coverage_Percent"]
                    ),
                    "mean_interval_width": safe_float(
                        row["Mean_Interval_Width"]
                    ),
                    "quality_flag": row.get(
                        "Calibration_Quality_Flag"
                    ),
                }
            )

    return {
        "project": (
            "Short-Term Ferry Ticket Demand Forecasting"
        ),
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": (
            "completed"
            if not predictions.empty
            else "no_data"
        ),
        "prediction_rows": int(
            len(predictions)
        ),
        "interval_rows": int(
            len(intervals)
        ),
        "error_distribution_groups": int(
            len(error_distribution)
        ),
        "model_uncertainty_groups": int(
            len(model_summary)
        ),
        "horizon_summary_rows": int(
            len(horizon_summary)
        ),
        "target_summary_rows": int(
            len(target_summary)
        ),
        "targets": sorted(
            predictions["Target"].unique().tolist()
        ),
        "models": sorted(
            predictions["Model"].unique().tolist()
        ),
        "horizons_minutes": sorted(
            predictions["Horizon_Minutes"]
            .unique()
            .tolist()
        ),
        "interval_levels": INTERVAL_LEVELS,
        "calibration_fraction": CALIBRATION_FRACTION,
        "best_historical_models_by_target_and_horizon": (
            best_models
        ),
        "interval_coverage": coverage_summary,
        "exported_files": exported_files,
        "important_note": (
            "Uncertainty ranges are estimated from historical errors. "
            "They are not guaranteed prediction coverage or "
            "certainty about future demand."
        ),
    }


# ============================================================
# 21. MAIN PIPELINE
# ============================================================

def run_uncertainty_analysis() -> dict[str, Any]:
    """
    Execute the complete uncertainty analysis pipeline.

    Steps:
        1. Prepare directories and logging.
        2. Load and validate predictions.
        3. Calculate historical residual distributions.
        4. Generate empirical prediction intervals.
        5. Calculate interval calibration diagnostics.
        6. Build model, horizon, and target summaries.
        7. Rank models using historical MAE.
        8. Export CSV reports.
        9. Generate diagnostic charts.
       10. Write JSON summary and methodology.
    """
    ensure_output_directories()
    setup_logging()

    logging.info(
        "=" * 72
    )

    logging.info(
        "FORECAST UNCERTAINTY ANALYSIS STARTED"
    )

    logging.info(
        "=" * 72
    )

    predictions = load_predictions()

    error_distribution = (
        build_error_distribution_report(
            predictions
        )
    )

    (
        intervals,
        calibration,
    ) = build_prediction_intervals(
        predictions
    )

    calibration = add_calibration_quality_flags(
        calibration
    )

    model_summary = (
        build_model_uncertainty_summary(
            predictions,
            calibration,
        )
    )

    horizon_summary = (
        build_horizon_uncertainty_summary(
            model_summary
        )
    )

    target_summary = (
        build_target_uncertainty_summary(
            model_summary
        )
    )

    ranking = build_uncertainty_ranking(
        model_summary
    )

    exported_files = export_uncertainty_reports(
        intervals=intervals,
        calibration=calibration,
        error_distribution=error_distribution,
        model_summary=model_summary,
        horizon_summary=horizon_summary,
        target_summary=target_summary,
        ranking=ranking,
    )

    # Charts are generated separately so a plotting failure does
    # not invalidate successfully calculated CSV reports.
    chart_functions = [
        (
            "interval width",
            lambda: plot_interval_width_by_horizon(
                calibration
            ),
        ),
        (
            "empirical coverage",
            lambda: plot_empirical_coverage(
                calibration
            ),
        ),
        (
            "error distribution",
            lambda: plot_error_distribution(
                predictions
            ),
        ),
        (
            "model uncertainty",
            lambda: plot_model_uncertainty(
                model_summary
            ),
        ),
    ]

    chart_errors: list[str] = []

    for chart_name, chart_function in chart_functions:
        try:
            chart_function()
        except Exception as exc:
            message = (
                f"{chart_name}: "
                f"{type(exc).__name__}: {exc}"
            )

            chart_errors.append(
                message
            )

            logging.exception(
                "Could not create %s chart.",
                chart_name,
            )

    methodology = build_methodology_report(
        predictions=predictions,
        intervals=intervals,
        calibration=calibration,
    )

    write_json(
        methodology,
        UNCERTAINTY_README_PATH,
    )

    summary = build_summary_json(
        predictions=predictions,
        intervals=intervals,
        calibration=calibration,
        error_distribution=error_distribution,
        model_summary=model_summary,
        horizon_summary=horizon_summary,
        target_summary=target_summary,
        ranking=ranking,
        exported_files=exported_files,
    )

    summary["chart_errors"] = chart_errors

    summary["charts"] = {
        "interval_width": str(
            FIGURE_INTERVAL_WIDTH.relative_to(BASE_DIR)
        ),
        "empirical_coverage": str(
            FIGURE_COVERAGE.relative_to(BASE_DIR)
        ),
        "error_distribution": str(
            FIGURE_ERROR_DISTRIBUTION.relative_to(BASE_DIR)
        ),
        "model_uncertainty": str(
            FIGURE_MODEL_UNCERTAINTY.relative_to(BASE_DIR)
        ),
    }

    write_json(
        summary,
        UNCERTAINTY_JSON_PATH,
    )

    logging.info(
        "Predictions processed: %s",
        f"{len(predictions):,}",
    )

    logging.info(
        "Intervals generated: %s",
        f"{len(intervals):,}",
    )

    logging.info(
        "Calibration groups: %s",
        f"{len(calibration):,}",
    )

    logging.info(
        "Model summary rows: %s",
        f"{len(model_summary):,}",
    )

    logging.info(
        "Uncertainty summary: %s",
        UNCERTAINTY_JSON_PATH,
    )

    logging.info(
        "FORECAST UNCERTAINTY ANALYSIS COMPLETED"
    )

    return summary


# ============================================================
# 22. COMMAND-LINE ENTRY POINT
# ============================================================

def main() -> int:
    """
    CLI entry point.

    Returns:
        0 for success.
        1 for a handled pipeline failure.
    """
    try:
        summary = run_uncertainty_analysis()

        print("\n" + "=" * 72)
        print("FORECAST UNCERTAINTY ANALYSIS COMPLETED")
        print("=" * 72)

        print(
            "Prediction rows:",
            f"{summary['prediction_rows']:,}",
        )

        print(
            "Prediction intervals:",
            f"{summary['interval_rows']:,}",
        )

        print(
            "Model uncertainty groups:",
            f"{summary['model_uncertainty_groups']:,}",
        )

        print(
            "Summary report:",
            UNCERTAINTY_JSON_PATH,
        )

        print(
            "Output directory:",
            UNCERTAINTY_DIR,
        )

        if summary.get("chart_errors"):
            print(
                "\nSome charts could not be generated:"
            )

            for error in summary["chart_errors"]:
                print(
                    " -",
                    error,
                )

        print(
            "\nNote: empirical intervals describe historical "
            "error behavior; they do not guarantee future coverage."
        )

        return 0

    except (
        FileNotFoundError,
        ValueError,
        pd.errors.ParserError,
        OSError,
    ) as exc:
        logging.exception(
            "Uncertainty analysis failed: %s",
            exc,
        )

        print(
            "\nERROR:",
            str(exc),
        )

        return 1

    except Exception as exc:
        logging.exception(
            "Unexpected uncertainty pipeline failure."
        )

        print(
            "\nUNEXPECTED ERROR:",
            f"{type(exc).__name__}: {exc}",
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
    
