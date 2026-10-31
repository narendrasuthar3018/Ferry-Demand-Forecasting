"""
PART 4: FERRY TICKET DEMAND FORECASTING ENGINE

Purpose:
    Prepare historical ticket demand data for forecasting.
    Validate timestamps and demand values.
    Aggregate ticket activity by day, week, or month.

Dataset:
    Toronto Island Ferry Tickets

Expected columns:
    Timestamp
    Sales Count
    Redemption Count

Important:
    Ticket activity is not the same as verified passenger
    boarding, queue length, or ferry capacity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


# ============================================================
# PART 4.1 — CONFIGURATION
# ============================================================

FORECAST_ENGINE_VERSION = "1.0.0"

FORECAST_REQUIRED_COLUMNS = (
    "Timestamp",
    "Sales Count",
    "Redemption Count",
)

FORECAST_DEMAND_COLUMNS = (
    "Sales Count",
    "Redemption Count",
)

FORECAST_DEFAULT_PERIODS = {
    "daily": "D",
    "weekly": "W",
    "monthly": "MS",
}


# ============================================================
# PART 4.2 — CUSTOM EXCEPTION
# ============================================================

class ForecastingDataError(ValueError):
    """Raised when the forecasting dataset is invalid."""


# ============================================================
# PART 4.3 — INPUT VALIDATION
# ============================================================

def di4_validate_input(
    dataframe: pd.DataFrame,
    required_columns: tuple[str, ...] = FORECAST_REQUIRED_COLUMNS,
) -> dict[str, Any]:
    """
    Validate the input DataFrame before forecasting.

    Returns:
        A dictionary containing validation status and details.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise ForecastingDataError(
            "Input must be a pandas DataFrame."
        )

    if dataframe.empty:
        raise ForecastingDataError(
            "The input dataset is empty."
        )

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ForecastingDataError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    return {
        "valid": True,
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "required_columns": list(required_columns),
        "missing_columns": [],
    }


# ============================================================
# PART 4.4 — PREPARE HISTORICAL DATA
# ============================================================

def di4_prepare_demand_data(
    dataframe: pd.DataFrame,
    timestamp_column: str = "Timestamp",
    demand_columns: tuple[str, ...] = FORECAST_DEMAND_COLUMNS,
    remove_negative_values: bool = True,
) -> pd.DataFrame:
    """
    Clean and prepare ticket activity data.

    Processing:
        1. Validate required columns.
        2. Copy the original DataFrame.
        3. Parse timestamps.
        4. Convert demand columns to numeric values.
        5. Handle invalid and negative demand values.
        6. Sort observations chronologically.

    Note:
        The original DataFrame is not modified.
    """

    required_columns = (
        timestamp_column,
        *demand_columns,
    )

    di4_validate_input(
        dataframe,
        required_columns=required_columns,
    )

    prepared = dataframe[
        list(required_columns)
    ].copy()

    # Convert timestamps into pandas datetime values.
    prepared[timestamp_column] = pd.to_datetime(
        prepared[timestamp_column],
        errors="coerce",
        utc=True,
    )

    # Remove rows that do not contain a valid timestamp.
    prepared = prepared.dropna(
        subset=[timestamp_column]
    ).copy()

    if prepared.empty:
        raise ForecastingDataError(
            "No valid timestamps remain after cleaning."
        )

    # Convert ticket counts to numeric values.
    for column in demand_columns:
        prepared[column] = pd.to_numeric(
            prepared[column],
            errors="coerce",
        )

        # Infinite values are treated as invalid.
        prepared[column] = prepared[column].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        # Negative ticket counts are invalid.
        if remove_negative_values:
            prepared.loc[
                prepared[column] < 0,
                column,
            ] = np.nan

    # Keep the missing values for transparent reporting.
    # Aggregation will ignore NaN values.
    prepared = prepared.sort_values(
        timestamp_column
    ).reset_index(drop=True)

    prepared = prepared.rename(
        columns={
            timestamp_column: "Forecast_Timestamp"
        }
    )

    return prepared


# ============================================================
# PART 4.5 — DATA QUALITY SUMMARY
# ============================================================

def di4_data_quality_summary(
    dataframe: pd.DataFrame,
    timestamp_column: str = "Timestamp",
    demand_columns: tuple[str, ...] = FORECAST_DEMAND_COLUMNS,
) -> dict[str, Any]:
    """
    Return a data quality summary without modifying the input.
    """

    required_columns = (
        timestamp_column,
        *demand_columns,
    )

    di4_validate_input(
        dataframe,
        required_columns=required_columns,
    )

    timestamp_values = pd.to_datetime(
        dataframe[timestamp_column],
        errors="coerce",
        utc=True,
    )

    summary: dict[str, Any] = {
        "total_rows": int(len(dataframe)),
        "valid_timestamp_rows": int(
            timestamp_values.notna().sum()
        ),
        "invalid_timestamp_rows": int(
            timestamp_values.isna().sum()
        ),
        "demand_columns": {},
    }

    for column in demand_columns:
        numeric_values = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        numeric_values = numeric_values.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        summary["demand_columns"][column] = {
            "missing_or_invalid": int(
                numeric_values.isna().sum()
            ),
            "negative_values": int(
                numeric_values.lt(0).sum()
            ),
            "valid_values": int(
                numeric_values.notna().sum()
            ),
        }

    return summary


# ============================================================
# PART 4.6 — AGGREGATE DEMAND
# ============================================================

def di4_aggregate_demand(
    dataframe: pd.DataFrame,
    frequency: str = "daily",
    demand_column: str = "Sales Count",
) -> pd.DataFrame:
    """
    Aggregate ticket demand by day, week, or month.

    Supported frequencies:
        daily
        weekly
        monthly

    Returns:
        A DataFrame with:
            Forecast_Date
            Actual_Demand
            Observation_Count
    """

    if frequency not in FORECAST_DEFAULT_PERIODS:
        raise ValueError(
            "Unsupported frequency. Choose daily, weekly, "
            "or monthly."
        )

    if demand_column not in FORECAST_DEMAND_COLUMNS:
        raise ValueError(
            f"Unsupported demand column: {demand_column}"
        )

    prepared = di4_prepare_demand_data(
        dataframe
    )

    # Select the requested demand measure.
    selected = prepared[
        ["Forecast_Timestamp", demand_column]
    ].copy()

    # Aggregate timestamps according to the chosen frequency.
    selected["Forecast_Date"] = (
        selected["Forecast_Timestamp"]
        .dt.tz_convert(None)
        .dt.to_period(
            {
                "daily": "D",
                "weekly": "W-SUN",
                "monthly": "M",
            }[frequency]
        )
        .dt.start_time
    )

    # Count valid observations and sum available demand.
    aggregated = (
        selected.groupby(
            "Forecast_Date",
            as_index=False,
        )
        .agg(
            Actual_Demand=(
                demand_column,
                lambda values: values.sum(min_count=1),
            ),
            Observation_Count=(
                demand_column,
                "count",
            ),
        )
        .sort_values("Forecast_Date")
        .reset_index(drop=True)
    )

    # Ensure the demand field is numeric.
    aggregated["Actual_Demand"] = pd.to_numeric(
        aggregated["Actual_Demand"],
        errors="coerce",
    )

    return aggregated


# ============================================================
# PART 4.7 — CREATE A CONTINUOUS TIME SERIES
# ============================================================

def di4_complete_time_series(
    aggregated_data: pd.DataFrame,
    frequency: str = "daily",
) -> pd.DataFrame:
    """
    Create a continuous daily, weekly, or monthly time series.

    Missing periods remain NaN instead of being silently
    interpreted as zero demand.

    This distinction matters because an absent record does
    not prove that zero tickets were sold or redeemed.
    """

    if frequency not in FORECAST_DEFAULT_PERIODS:
        raise ValueError(
            "Unsupported frequency. Choose daily, weekly, "
            "or monthly."
        )

    required = {
        "Forecast_Date",
        "Actual_Demand",
    }

    missing = required.difference(
        aggregated_data.columns
    )

    if missing:
        raise ForecastingDataError(
            "Missing aggregation columns: "
            + ", ".join(sorted(missing))
        )

    if aggregated_data.empty:
        raise ForecastingDataError(
            "Cannot complete an empty time series."
        )

    result = aggregated_data.copy()

    result["Forecast_Date"] = pd.to_datetime(
        result["Forecast_Date"],
        errors="coerce",
    )

    result = result.dropna(
        subset=["Forecast_Date"]
    )

    if result.empty:
        raise ForecastingDataError(
            "No valid dates exist in the aggregated data."
        )

    result = result.sort_values(
        "Forecast_Date"
    )

    # Combine duplicate dates before creating the date range.
    result = (
        result.groupby(
            "Forecast_Date",
            as_index=False,
        )
        .agg(
            Actual_Demand=(
                "Actual_Demand",
                lambda values: values.sum(min_count=1),
            )
        )
        .sort_values("Forecast_Date")
    )

    # Build a continuous index using the selected frequency.
    pandas_frequency = {
        "daily": "D",
        "weekly": "W-MON",
        "monthly": "MS",
    }[frequency]

    date_range = pd.date_range(
        start=result["Forecast_Date"].min(),
        end=result["Forecast_Date"].max(),
        freq=pandas_frequency,
    )

    result = (
        result.set_index("Forecast_Date")
        .reindex(date_range)
        .rename_axis("Forecast_Date")
        .reset_index()
    )

    # Record whether demand was actually observed.
    result["Has_Observed_Demand"] = (
        result["Actual_Demand"].notna()
    )

    return result


# ============================================================
# PART 4.8 — BUILD A FORECASTING DATASET
# ============================================================

def di4_build_forecasting_dataset(
    dataframe: pd.DataFrame,
    frequency: str = "daily",
    demand_column: str = "Sales Count",
) -> pd.DataFrame:
    """
    Prepare a complete time series for the forecasting models.

    This function combines:
        - Demand aggregation
        - Continuous date generation
        - Missing-period identification
    """

    aggregated = di4_aggregate_demand(
        dataframe=dataframe,
        frequency=frequency,
        demand_column=demand_column,
    )

    complete_series = di4_complete_time_series(
        aggregated_data=aggregated,
        frequency=frequency,
    )

    complete_series["Demand_Measure"] = demand_column
    complete_series["Frequency"] = frequency

    return complete_series

```python
# ============================================================
# PART 4.9 — FORECASTING CONFIGURATION
# ============================================================

FORECAST_MODEL_NAMES = (
    "naive",
    "moving_average",
    "seasonal_naive",
)


# ============================================================
# PART 4.10 — VALIDATE FORECASTING SERIES
# ============================================================

def di4_validate_forecasting_series(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate and standardize the time series before modelling.
    """

    required_columns = {
        "Forecast_Date",
        "Actual_Demand",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ForecastingDataError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    result = dataframe[
        ["Forecast_Date", "Actual_Demand"]
    ].copy()

    result["Forecast_Date"] = pd.to_datetime(
        result["Forecast_Date"],
        errors="coerce",
    )

    result["Actual_Demand"] = pd.to_numeric(
        result["Actual_Demand"],
        errors="coerce",
    )

    result["Actual_Demand"] = result[
        "Actual_Demand"
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    result = result.dropna(
        subset=["Forecast_Date"]
    )

    result = result.sort_values(
        "Forecast_Date"
    ).reset_index(drop=True)

    if result.empty:
        raise ForecastingDataError(
            "No valid forecasting observations were found."
        )

    if result["Forecast_Date"].duplicated().any():
        raise ForecastingDataError(
            "Duplicate dates detected. Aggregate the data first."
        )

    valid_demand = result["Actual_Demand"].dropna()

    if valid_demand.empty:
        raise ForecastingDataError(
            "No valid demand observations are available."
        )

    if (valid_demand < 0).any():
        raise ForecastingDataError(
            "Negative demand values must be cleaned first."
        )

    return result


# ============================================================
# PART 4.11 — NAIVE FORECAST
# ============================================================

def di4_naive_forecast(
    historical_values: pd.Series,
    horizon: int,
) -> np.ndarray:
    """
    Predict every future period using the latest observation.

    Example:
        Historical: 100, 110, 120
        Forecast:   120, 120, 120
    """

    if horizon < 1:
        raise ValueError(
            "Forecast horizon must be at least 1."
        )

    values = pd.to_numeric(
        historical_values,
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if values.empty:
        raise ForecastingDataError(
            "Naive forecasting requires valid historical data."
        )

    last_value = float(values.iloc[-1])

    return np.full(
        shape=horizon,
        fill_value=last_value,
        dtype=float,
    )


# ============================================================
# PART 4.12 — MOVING AVERAGE FORECAST
# ============================================================

def di4_moving_average_forecast(
    historical_values: pd.Series,
    horizon: int,
    window: int = 7,
) -> np.ndarray:
    """
    Predict future demand using the recent historical average.

    The forecast remains constant across the requested horizon.
    """

    if horizon < 1:
        raise ValueError(
            "Forecast horizon must be at least 1."
        )

    if window < 1:
        raise ValueError(
            "Moving-average window must be at least 1."
        )

    values = pd.to_numeric(
        historical_values,
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if values.empty:
        raise ForecastingDataError(
            "Moving average requires valid historical data."
        )

    recent_values = values.tail(window)

    average_demand = float(
        recent_values.mean()
    )

    return np.full(
        shape=horizon,
        fill_value=average_demand,
        dtype=float,
    )


# ============================================================
# PART 4.13 — SEASONAL NAIVE FORECAST
# ============================================================

def di4_seasonal_naive_forecast(
    historical_values: pd.Series,
    horizon: int,
    season_length: int = 7,
) -> np.ndarray:
    """
    Repeat the latest complete seasonal pattern.

    For daily demand, season_length=7 can represent
    a weekly pattern.

    For monthly demand, season_length=12 can represent
    an annual pattern, provided sufficient history exists.
    """

    if horizon < 1:
        raise ValueError(
            "Forecast horizon must be at least 1."
        )

    if season_length < 1:
        raise ValueError(
            "Season length must be at least 1."
        )

    values = pd.to_numeric(
        historical_values,
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(values) < season_length:
        raise ForecastingDataError(
            "Insufficient observations for seasonal forecasting. "
            f"Need at least {season_length} valid observations."
        )

    # Use the most recent complete set of seasonal observations.
    seasonal_pattern = values.tail(
        season_length
    ).to_numpy(dtype=float)

    repeated_pattern = np.resize(
        seasonal_pattern,
        horizon,
    )

    return repeated_pattern


# ============================================================
# PART 4.14 — FORECAST METRICS
# ============================================================

def di4_evaluate_forecast(
    actual_values,
    predicted_values,
) -> dict[str, Any]:
    """
    Calculate common forecasting error metrics.

    MAE:
        Mean Absolute Error.

    RMSE:
        Root Mean Squared Error.

    MAPE:
        Mean Absolute Percentage Error.
        Calculated only where actual demand is non-zero.

    WAPE:
        Weighted Absolute Percentage Error.
        Undefined when total absolute actual demand is zero.
    """

    actual = np.asarray(
        actual_values,
        dtype=float,
    ).reshape(-1)

    predicted = np.asarray(
        predicted_values,
        dtype=float,
    ).reshape(-1)

    if actual.size != predicted.size:
        raise ValueError(
            "Actual and predicted arrays must have equal lengths."
        )

    if actual.size == 0:
        raise ValueError(
            "Cannot evaluate an empty forecast."
        )

    valid_mask = (
        np.isfinite(actual)
        & np.isfinite(predicted)
    )

    actual = actual[valid_mask]
    predicted = predicted[valid_mask]

    if actual.size == 0:
        raise ValueError(
            "No valid actual-predicted pairs are available."
        )

    errors = actual - predicted

    absolute_errors = np.abs(errors)

    mae = float(
        np.mean(absolute_errors)
    )

    rmse = float(
        np.sqrt(np.mean(errors ** 2))
    )

    nonzero_mask = actual != 0

    if nonzero_mask.any():
        mape = float(
            np.mean(
                absolute_errors[nonzero_mask]
                / np.abs(actual[nonzero_mask])
            ) * 100
        )
    else:
        mape = np.nan

    total_actual = float(
        np.sum(np.abs(actual))
    )

    if total_actual > 0:
        wape = float(
            np.sum(absolute_errors)
            / total_actual
            * 100
        )
    else:
        wape = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_Percent": mape,
        "WAPE_Percent": wape,
        "Evaluated_Observations": int(actual.size),
    }


# ============================================================
# PART 4.15 — TIME-ORDERED TRAIN-TEST SPLIT
# ============================================================

def di4_time_ordered_split(
    dataframe: pd.DataFrame,
    test_size: int = 14,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split historical data chronologically.

    Earlier observations become training data.
    The latest observations become test data.
    """

    if test_size < 1:
        raise ValueError(
            "Test size must be at least 1."
        )

    series = di4_validate_forecasting_series(
        dataframe
    )

    # Missing demand observations are excluded from modelling.
    series = series.dropna(
        subset=["Actual_Demand"]
    ).reset_index(drop=True)

    if len(series) <= test_size:
        raise ForecastingDataError(
            "Not enough valid observations for the requested "
            "time-ordered train-test split."
        )

    split_index = len(series) - test_size

    train_data = series.iloc[
        :split_index
    ].copy().reset_index(drop=True)

    test_data = series.iloc[
        split_index:
    ].copy().reset_index(drop=True)

    return train_data, test_data


# ============================================================
# PART 4.16 — COMPARE BASELINE MODELS
# ============================================================

def di4_compare_baseline_models(
    dataframe: pd.DataFrame,
    test_size: int = 14,
    moving_average_window: int = 7,
    season_length: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare naive, moving-average, and seasonal-naive models.

    Returns:
        forecast_comparison:
            Actual demand and predictions for each model.

        metrics_table:
            Error metrics for each model.

    The test period is never used to create model predictions.
    """

    train_data, test_data = di4_time_ordered_split(
        dataframe=dataframe,
        test_size=test_size,
    )

    training_values = train_data[
        "Actual_Demand"
    ]

    actual_values = test_data[
        "Actual_Demand"
    ].to_numpy(dtype=float)

    horizon = len(test_data)

    predictions: dict[str, np.ndarray] = {
        "Naive": di4_naive_forecast(
            training_values,
            horizon,
        ),
        "Moving_Average": di4_moving_average_forecast(
            training_values,
            horizon,
            window=moving_average_window,
        ),
    }

    # If there is not enough history for the requested season,
    # seasonal-naive is reported as unavailable.
    try:
        predictions["Seasonal_Naive"] = (
            di4_seasonal_naive_forecast(
                training_values,
                horizon,
                season_length=season_length,
            )
        )
    except ForecastingDataError:
        predictions["Seasonal_Naive"] = np.full(
            horizon,
            np.nan,
            dtype=float,
        )

    forecast_comparison = pd.DataFrame(
        {
            "Forecast_Date": test_data[
                "Forecast_Date"
            ].to_numpy(),
            "Actual_Demand": actual_values,
            "Naive_Prediction": predictions["Naive"],
            "Moving_Average_Prediction": predictions[
                "Moving_Average"
            ],
            "Seasonal_Naive_Prediction": predictions[
                "Seasonal_Naive"
            ],
        }
    )

    metrics_rows = []

    model_column_map = {
        "Naive": "Naive_Prediction",
        "Moving_Average": "Moving_Average_Prediction",
        "Seasonal_Naive": "Seasonal_Naive_Prediction",
    }

    for model_name, prediction_column in (
        model_column_map.items()
    ):
        model_metrics = di4_evaluate_forecast(
            actual_values=forecast_comparison[
                "Actual_Demand"
            ],
            predicted_values=forecast_comparison[
                prediction_column
            ],
        )

        metrics_rows.append(
            {
                "Model": model_name,
                **model_metrics,
            }
        )

    metrics_table = pd.DataFrame(
        metrics_rows
    ).sort_values(
        "MAE",
        ascending=True,
        na_position="last",
    ).reset_index(drop=True)

    return forecast_comparison, metrics_table


# ============================================================
# PART 4.17 — SELECT THE BEST BASELINE MODEL
# ============================================================

def di4_select_best_baseline(
    metrics_table: pd.DataFrame,
    selection_metric: str = "MAE",
) -> str:
    """
    Select the model with the lowest valid error metric.

    This selection is based on historical validation results.
    """

    if metrics_table.empty:
        raise ForecastingDataError(
            "The model comparison table is empty."
        )

    if selection_metric not in metrics_table.columns:
        raise ValueError(
            f"Unknown selection metric: {selection_metric}"
        )

    valid_metrics = metrics_table[
        ["Model", selection_metric]
    ].copy()

    valid_metrics[selection_metric] = pd.to_numeric(
        valid_metrics[selection_metric],
        errors="coerce",
    )

    valid_metrics = valid_metrics.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna(
        subset=[selection_metric]
    )

    if valid_metrics.empty:
        raise ForecastingDataError(
            "No valid model metrics are available for selection."
        )

    best_row = valid_metrics.loc[
        valid_metrics[selection_metric].idxmin()
    ]

    return str(
        best_row["Model"]
    )

# =====================================================================
# PART 4.18 — START
# FUTURE FORECASTING, CONFIDENCE INTERVALS, REPORTING AND EXPORTS
# =====================================================================

import json
import logging
from datetime import datetime, timezone
from typing import Iterable


# =====================================================================
# PART 4.18.1 — LOGGING CONFIGURATION
# =====================================================================

DI4_LOGGER = logging.getLogger("ferry_forecasting")

if not DI4_LOGGER.handlers:
    _di4_handler = logging.StreamHandler()
    _di4_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
    )
    DI4_LOGGER.addHandler(_di4_handler)

DI4_LOGGER.setLevel(logging.INFO)


# =====================================================================
# PART 4.18.2 — FORECAST CONFIGURATION
# =====================================================================

DI4_SUPPORTED_FREQUENCIES = {
    "daily": "D",
    "weekly": "W-MON",
    "monthly": "MS",
}

DI4_DEFAULT_CONFIDENCE_LEVEL = 0.95

DI4_MIN_INTERVAL_OBSERVATIONS = 5

DI4_EXPORT_COLUMNS = [
    "Forecast_Date",
    "Forecast_Demand",
    "Lower_Bound",
    "Upper_Bound",
    "Forecast_Model",
    "Frequency",
    "Demand_Measure",
    "Confidence_Level",
]


# =====================================================================
# PART 4.18.3 — VALIDATE FORECAST HORIZON
# =====================================================================

def di4_validate_horizon(horizon: int) -> int:
    """Validate the number of future periods requested."""

    if isinstance(horizon, bool) or not isinstance(
        horizon, (int, np.integer)
    ):
        raise ValueError(
            "Forecast horizon must be a positive integer."
        )

    if horizon < 1:
        raise ValueError(
            "Forecast horizon must be at least 1."
        )

    if horizon > 3650:
        raise ValueError(
            "Forecast horizon cannot exceed 3650 periods."
        )

    return int(horizon)


# =====================================================================
# PART 4.18.4 — VALIDATE CONFIDENCE LEVEL
# =====================================================================

def di4_validate_confidence_level(
    confidence_level: float,
) -> float:
    """Validate a statistical confidence level."""

    if isinstance(confidence_level, bool):
        raise ValueError(
            "Confidence level must be numeric."
        )

    try:
        confidence_level = float(confidence_level)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Confidence level must be numeric."
        ) from exc

    if not 0 < confidence_level < 1:
        raise ValueError(
            "Confidence level must be between 0 and 1."
        )

    return confidence_level


# =====================================================================
# PART 4.18.5 — GENERATE FUTURE DATES
# =====================================================================

def di4_generate_future_dates(
    historical_dates: Iterable,
    horizon: int,
    frequency: str = "daily",
) -> pd.DatetimeIndex:
    """Generate dates immediately after the historical period."""

    horizon = di4_validate_horizon(horizon)

    if frequency not in DI4_SUPPORTED_FREQUENCIES:
        raise ValueError(
            "Frequency must be daily, weekly, or monthly."
        )

    dates = pd.to_datetime(
        pd.Series(list(historical_dates)),
        errors="coerce",
        utc=True,
    ).dropna()

    if dates.empty:
        raise ForecastingDataError(
            "Cannot generate future dates without valid history."
        )

    last_date = dates.max().tz_convert(None).normalize()

    pandas_frequency = DI4_SUPPORTED_FREQUENCIES[
        frequency
    ]

    if frequency == "daily":
        start_date = last_date + pd.Timedelta(days=1)

    elif frequency == "weekly":
        start_date = (
            last_date.to_period("W-MON").end_time.normalize()
            + pd.Timedelta(days=1)
        )

    else:
        start_date = (
            last_date.to_period("M").start_time
            + pd.offsets.MonthBegin(1)
        )

    future_dates = pd.date_range(
        start=start_date,
        periods=horizon,
        freq=pandas_frequency,
    )

    if len(future_dates) != horizon:
        raise ForecastingDataError(
            "Future date generation returned an unexpected result."
        )

    return future_dates


# =====================================================================
# PART 4.18.6 — ROBUST RESIDUAL SCALE
# =====================================================================

def di4_residual_scale(
    actual_values,
    predicted_values,
) -> float:
    """
    Estimate residual standard deviation.

    This is a simple uncertainty estimate, not a guarantee
    that future errors follow a normal distribution.
    """

    actual = np.asarray(
        actual_values,
        dtype=float,
    ).reshape(-1)

    predicted = np.asarray(
        predicted_values,
        dtype=float,
    ).reshape(-1)

    if actual.size != predicted.size:
        raise ValueError(
            "Actual and predicted values must have equal lengths."
        )

    valid_mask = (
        np.isfinite(actual)
        & np.isfinite(predicted)
    )

    residuals = actual[valid_mask] - predicted[valid_mask]

    if residuals.size < 2:
        return np.nan

    scale = float(np.std(residuals, ddof=1))

    if not np.isfinite(scale):
        return np.nan

    return scale


# =====================================================================
# PART 4.18.7 — EMPIRICAL ABSOLUTE RESIDUAL QUANTILE
# =====================================================================

def di4_empirical_error_quantile(
    actual_values,
    predicted_values,
    confidence_level: float = DI4_DEFAULT_CONFIDENCE_LEVEL,
) -> float:
    """
    Estimate an error radius from historical validation residuals.

    This interval is an empirical approximation. Its coverage
    is not guaranteed under distribution shift or strong
    temporal dependence.
    """

    confidence_level = di4_validate_confidence_level(
        confidence_level
    )

    actual = np.asarray(
        actual_values,
        dtype=float,
    ).reshape(-1)

    predicted = np.asarray(
        predicted_values,
        dtype=float,
    ).reshape(-1)

    if actual.size != predicted.size:
        raise ValueError(
            "Actual and predicted values must have equal lengths."
        )

    valid_mask = (
        np.isfinite(actual)
        & np.isfinite(predicted)
    )

    residuals = np.abs(
        actual[valid_mask] - predicted[valid_mask]
    )

    if residuals.size < DI4_MIN_INTERVAL_OBSERVATIONS:
        return np.nan

    quantile = min(
        1.0,
        confidence_level,
    )

    return float(
        np.quantile(
            residuals,
            quantile,
            method="higher",
        )
    )


# =====================================================================
# PART 4.18.8 — BUILD APPROXIMATE PREDICTION INTERVALS
# =====================================================================

def di4_build_prediction_intervals(
    predictions,
    residual_scale: float,
    confidence_level: float = DI4_DEFAULT_CONFIDENCE_LEVEL,
    empirical_error_radius: Optional[float] = None,
) -> pd.DataFrame:
    """
    Create approximate prediction bounds.

    Prefer the empirical error radius when enough validation
    residuals are available.

    Otherwise, use a simple normal approximation with a fixed
    residual scale. This fallback does not model increasing
    uncertainty over longer forecast horizons.
    """

    confidence_level = di4_validate_confidence_level(
        confidence_level
    )

    prediction_values = np.asarray(
        predictions,
        dtype=float,
    ).reshape(-1)

    if prediction_values.size == 0:
        raise ForecastingDataError(
            "Cannot create intervals for an empty forecast."
        )

    if not np.isfinite(prediction_values).all():
        raise ForecastingDataError(
            "Predictions contain invalid numeric values."
        )

    radius = np.nan

    if empirical_error_radius is not None:
        if np.isfinite(empirical_error_radius):
            if empirical_error_radius >= 0:
                radius = float(empirical_error_radius)

    if not np.isfinite(radius):
        if not np.isfinite(residual_scale) or residual_scale < 0:
            raise ForecastingDataError(
                "Valid residual information is required "
                "to estimate prediction intervals."
            )

        # Two-sided normal approximation.
        from statistics import NormalDist

        tail_probability = (
            1.0 + confidence_level
        ) / 2.0

        z_value = NormalDist().inv_cdf(
            tail_probability
        )

        radius = float(
            z_value * residual_scale
        )

    lower_bound = np.maximum(
        0.0,
        prediction_values - radius,
    )

    upper_bound = np.maximum(
        lower_bound,
        prediction_values + radius,
    )

    return pd.DataFrame(
        {
            "Forecast_Demand": prediction_values,
            "Lower_Bound": lower_bound,
            "Upper_Bound": upper_bound,
            "Confidence_Level": confidence_level,
            "Interval_Radius": radius,
        }
    )


# =====================================================================
# PART 4.18.9 — FIT A BASELINE MODEL
# =====================================================================

def di4_predict_with_model(
    historical_values: pd.Series,
    horizon: int,
    model_name: str,
    moving_average_window: int = 7,
    season_length: int = 7,
) -> np.ndarray:
    """Route prediction requests to supported baseline models."""

    horizon = di4_validate_horizon(horizon)

    if model_name == "naive":
        return di4_naive_forecast(
            historical_values,
            horizon,
        )

    if model_name == "moving_average":
        return di4_moving_average_forecast(
            historical_values,
            horizon,
            window=moving_average_window,
        )

    if model_name == "seasonal_naive":
        return di4_seasonal_naive_forecast(
            historical_values,
            horizon,
            season_length=season_length,
        )

    raise ValueError(
        f"Unsupported forecast model: {model_name}"
    )


# =====================================================================
# PART 4.18.10 — VALIDATE HISTORICAL FORECAST DATA
# =====================================================================

def di4_prepare_model_history(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare valid chronological observations for a model."""

    result = di4_validate_forecasting_series(
        dataframe
    )

    result = result.dropna(
        subset=["Actual_Demand"]
    ).copy()

    result = result.sort_values(
        "Forecast_Date"
    ).reset_index(drop=True)

    if len(result) < 2:
        raise ForecastingDataError(
            "At least two valid observations are required."
        )

    return result


# =====================================================================
# PART 4.18.11 — VALIDATION RESIDUALS FOR INTERVAL ESTIMATION
# =====================================================================

def di4_get_validation_residuals(
    historical_data: pd.DataFrame,
    validation_size: int = 14,
    model_name: str = "naive",
    moving_average_window: int = 7,
    season_length: int = 7,
) -> pd.DataFrame:
    """
    Generate validation predictions using only earlier history.

    The validation observations are not used to produce their
    own predictions.
    """

    history = di4_prepare_model_history(
        historical_data
    )

    if validation_size < 1:
        raise ValueError(
            "Validation size must be positive."
        )

    if len(history) <= validation_size:
        raise ForecastingDataError(
            "Insufficient history for validation residuals."
        )

    split_index = len(history) - validation_size

    training = history.iloc[
        :split_index
    ].copy()

    validation = history.iloc[
        split_index:
    ].copy()

    predictions = di4_predict_with_model(
        historical_values=training["Actual_Demand"],
        horizon=len(validation),
        model_name=model_name,
        moving_average_window=moving_average_window,
        season_length=season_length,
    )

    result = pd.DataFrame(
        {
            "Forecast_Date": validation[
                "Forecast_Date"
            ].to_numpy(),
            "Actual_Demand": validation[
                "Actual_Demand"
            ].to_numpy(dtype=float),
            "Validation_Prediction": predictions,
        }
    )

    result["Residual"] = (
        result["Actual_Demand"]
        - result["Validation_Prediction"]
    )

    result["Absolute_Error"] = result[
        "Residual"
    ].abs()

    return result


# =====================================================================
# PART 4.18.12 — FORECAST FUTURE DEMAND
# =====================================================================

def di4_forecast_future_demand(
    historical_data: pd.DataFrame,
    horizon: int = 14,
    frequency: str = "daily",
    demand_column: str = "Sales Count",
    model_name: str = "naive",
    confidence_level: float = DI4_DEFAULT_CONFIDENCE_LEVEL,
    validation_size: int = 14,
    moving_average_window: int = 7,
    season_length: int = 7,
) -> pd.DataFrame:
    """
    Generate future demand forecasts and approximate intervals.

    Expected input:
        Forecast_Date
        Actual_Demand

    The input should contain one aggregated observation
    per period.
    """

    horizon = di4_validate_horizon(
        horizon
    )

    confidence_level = di4_validate_confidence_level(
        confidence_level
    )

    if frequency not in DI4_SUPPORTED_FREQUENCIES:
        raise ValueError(
            "Frequency must be daily, weekly, or monthly."
        )

    history = di4_prepare_model_history(
        historical_data
    )

    # Generate predictions from all available history.
    predictions = di4_predict_with_model(
        historical_values=history["Actual_Demand"],
        horizon=horizon,
        model_name=model_name,
        moving_average_window=moving_average_window,
        season_length=season_length,
    )

    # Estimate uncertainty from a historical holdout.
    empirical_radius = np.nan
    residual_scale = np.nan

    try:
        validation = di4_get_validation_residuals(
            historical_data=history,
            validation_size=min(
                validation_size,
                max(1, len(history) // 3),
            ),
            model_name=model_name,
            moving_average_window=moving_average_window,
            season_length=season_length,
        )

        empirical_radius = di4_empirical_error_quantile(
            validation["Actual_Demand"],
            validation["Validation_Prediction"],
            confidence_level=confidence_level,
        )

        residual_scale = di4_residual_scale(
            validation["Actual_Demand"],
            validation["Validation_Prediction"],
        )

    except (ForecastingDataError, ValueError) as exc:
        DI4_LOGGER.warning(
            "Could not estimate validation uncertainty: %s",
            exc,
        )

    intervals = di4_build_prediction_intervals(
        predictions=predictions,
        residual_scale=residual_scale,
        confidence_level=confidence_level,
        empirical_error_radius=empirical_radius,
    )

    future_dates = di4_generate_future_dates(
        historical_dates=history["Forecast_Date"],
        horizon=horizon,
        frequency=frequency,
    )

    forecast = pd.DataFrame(
        {
            "Forecast_Date": future_dates,
            "Forecast_Demand": intervals[
                "Forecast_Demand"
            ].to_numpy(),
            "Lower_Bound": intervals[
                "Lower_Bound"
            ].to_numpy(),
            "Upper_Bound": intervals[
                "Upper_Bound"
            ].to_numpy(),
            "Forecast_Model": model_name,
            "Frequency": frequency,
            "Demand_Measure": demand_column,
            "Confidence_Level": confidence_level,
        }
    )

    forecast["Forecast_Demand"] = (
        forecast["Forecast_Demand"].clip(lower=0)
    )

    forecast["Lower_Bound"] = (
        forecast["Lower_Bound"].clip(lower=0)
    )

    forecast["Upper_Bound"] = (
        forecast["Upper_Bound"].clip(lower=0)
    )

    forecast["Forecast_Generated_At_UTC"] = (
        datetime.now(timezone.utc).isoformat()
    )

    return forecast


# =====================================================================
# PART 4.18.13 — FORECAST SUMMARY
# =====================================================================

def di4_build_forecast_summary(
    forecast: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize a generated forecast."""

    required_columns = {
        "Forecast_Date",
        "Forecast_Demand",
        "Lower_Bound",
        "Upper_Bound",
    }

    missing = required_columns.difference(
        forecast.columns
    )

    if missing:
        raise ForecastingDataError(
            "Forecast summary columns missing: "
            + ", ".join(sorted(missing))
        )

    if forecast.empty:
        raise ForecastingDataError(
            "Cannot summarize an empty forecast."
        )

    demand = pd.to_numeric(
        forecast["Forecast_Demand"],
        errors="coerce",
    )

    lower = pd.to_numeric(
        forecast["Lower_Bound"],
        errors="coerce",
    )

    upper = pd.to_numeric(
        forecast["Upper_Bound"],
        errors="coerce",
    )

    valid = (
        demand.notna()
        & lower.notna()
        & upper.notna()
    )

    if not valid.any():
        raise ForecastingDataError(
            "Forecast contains no valid numeric results."
        )

    demand = demand[valid]
    lower = lower[valid]
    upper = upper[valid]

    summary = {
        "forecast_periods": int(len(demand)),
        "forecast_total": float(demand.sum()),
        "forecast_mean": float(demand.mean()),
        "forecast_minimum": float(demand.min()),
        "forecast_maximum": float(demand.max()),
        "lower_bound_total": float(lower.sum()),
        "upper_bound_total": float(upper.sum()),
        "average_interval_width": float(
            (upper - lower).mean()
        ),
        "first_forecast_date": str(
            pd.to_datetime(
                forecast.loc[valid, "Forecast_Date"]
            ).min()
        ),
        "last_forecast_date": str(
            pd.to_datetime(
                forecast.loc[valid, "Forecast_Date"]
            ).max()
        ),
    }

    if "Forecast_Model" in forecast.columns:
        summary["forecast_model"] = str(
            forecast["Forecast_Model"].iloc[0]
        )

    if "Frequency" in forecast.columns:
        summary["frequency"] = str(
            forecast["Frequency"].iloc[0]
        )

    if "Demand_Measure" in forecast.columns:
        summary["demand_measure"] = str(
            forecast["Demand_Measure"].iloc[0]
        )

    return summary


# =====================================================================
# PART 4.18.14 — COMPARE HISTORICAL AND FORECAST PERIODS
# =====================================================================

def di4_build_forecast_timeline(
    historical_data: pd.DataFrame,
    forecast_data: pd.DataFrame,
) -> pd.DataFrame:
    """Combine historical observations and future predictions."""

    history = di4_prepare_model_history(
        historical_data
    )

    forecast_required = {
        "Forecast_Date",
        "Forecast_Demand",
    }

    missing = forecast_required.difference(
        forecast_data.columns
    )

    if missing:
        raise ForecastingDataError(
            "Forecast timeline columns missing: "
            + ", ".join(sorted(missing))
        )

    future = forecast_data.copy()

    future["Forecast_Date"] = pd.to_datetime(
        future["Forecast_Date"],
        errors="coerce",
    )

    future["Forecast_Demand"] = pd.to_numeric(
        future["Forecast_Demand"],
        errors="coerce",
    )

    future = future.dropna(
        subset=[
            "Forecast_Date",
            "Forecast_Demand",
        ]
    )

    history_result = pd.DataFrame(
        {
            "Forecast_Date": history[
                "Forecast_Date"
            ],
            "Actual_Demand": history[
                "Actual_Demand"
            ],
            "Forecast_Demand": np.nan,
            "Record_Type": "Historical",
        }
    )

    forecast_result = pd.DataFrame(
        {
            "Forecast_Date": future[
                "Forecast_Date"
            ],
            "Actual_Demand": np.nan,
            "Forecast_Demand": future[
                "Forecast_Demand"
            ],
            "Record_Type": "Forecast",
        }
    )

    combined = pd.concat(
        [
            history_result,
            forecast_result,
        ],
        ignore_index=True,
    )

    combined = combined.sort_values(
        "Forecast_Date"
    ).reset_index(drop=True)

    return combined


# =====================================================================
# PART 4.18.15 — EXPORT FORECAST CSV
# =====================================================================

def di4_export_forecast_csv(
    forecast: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Export forecast results to CSV."""

    if forecast.empty:
        raise ForecastingDataError(
            "Cannot export an empty forecast."
        )

    output_file = Path(
        output_path
    ).expanduser()

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecast.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    DI4_LOGGER.info(
        "Forecast CSV exported: %s",
        output_file,
    )

    return output_file.resolve()


# =====================================================================
# PART 4.18.16 — EXPORT FORECAST SUMMARY JSON
# =====================================================================

def di4_export_forecast_summary_json(
    forecast: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Export the forecast summary as JSON."""

    summary = di4_build_forecast_summary(
        forecast
    )

    output_file = Path(
        output_path
    ).expanduser()

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        )

    DI4_LOGGER.info(
        "Forecast summary exported: %s",
        output_file,
    )

    return output_file.resolve()


# =====================================================================
# PART 4.18.17 — EXPORT MODEL COMPARISON
# =====================================================================

def di4_export_model_metrics(
    metrics_table: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Export model evaluation metrics to CSV."""

    if metrics_table.empty:
        raise ForecastingDataError(
            "Cannot export an empty model metrics table."
        )

    output_file = Path(
        output_path
    ).expanduser()

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_table.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    DI4_LOGGER.info(
        "Model metrics exported: %s",
        output_file,
    )

    return output_file.resolve()


# =====================================================================
# PART 4.18.18 — CREATE A MODEL REPORT
# =====================================================================

def di4_build_model_report(
    historical_data: pd.DataFrame,
    test_size: int = 14,
    moving_average_window: int = 7,
    season_length: int = 7,
) -> dict[str, Any]:
    """
    Evaluate supported baseline models and return a report.

    Model selection uses the chronological holdout metrics.
    The report does not claim that the selected model will
    always perform best on future unseen data.
    """

    history = di4_prepare_model_history(
        historical_data
    )

    comparison, metrics = di4_compare_baseline_models(
        dataframe=history,
        test_size=test_size,
        moving_average_window=moving_average_window,
        season_length=season_length,
    )

    best_model = di4_select_best_baseline(
        metrics,
        selection_metric="MAE",
    )

    return {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "historical_observations": int(
            len(history)
        ),
        "test_observations": int(
            len(comparison)
        ),
        "best_baseline_model_by_mae": best_model,
        "metrics": metrics.to_dict(
            orient="records"
        ),
        "comparison": comparison.to_dict(
            orient="records"
        ),
    }


# =====================================================================
# PART 4.18.19 — EXPORT MODEL REPORT JSON
# =====================================================================

def di4_export_model_report_json(
    report: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Save the complete model report as JSON."""

    output_file = Path(
        output_path
    ).expanduser()

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
            allow_nan=False,
        )

    DI4_LOGGER.info(
        "Model report exported: %s",
        output_file,
    )

    return output_file.resolve()


# =====================================================================
# PART 4.18.20 — BUILD COMPLETE FORECAST PACKAGE
# =====================================================================

def di4_build_forecast_package(
    historical_data: pd.DataFrame,
    horizon: int = 14,
    frequency: str = "daily",
    demand_column: str = "Sales Count",
    model_name: str = "naive",
    confidence_level: float = DI4_DEFAULT_CONFIDENCE_LEVEL,
    validation_size: int = 14,
    moving_average_window: int = 7,
    season_length: int = 7,
) -> dict[str, Any]:
    """
    Build forecasts, validation results, and a summary.

    No files are written by this function.
    """

    history = di4_prepare_model_history(
        historical_data
    )

    forecast = di4_forecast_future_demand(
        historical_data=history,
        horizon=horizon,
        frequency=frequency,
        demand_column=demand_column,
        model_name=model_name,
        confidence_level=confidence_level,
        validation_size=validation_size,
        moving_average_window=moving_average_window,
        season_length=season_length,
    )

    timeline = di4_build_forecast_timeline(
        historical_data=history,
        forecast_data=forecast,
    )

    summary = di4_build_forecast_summary(
        forecast
    )

    return {
        "forecast": forecast,
        "timeline": timeline,
        "summary": summary,
        "metadata": {
            "engine_version": FORECAST_ENGINE_VERSION,
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "horizon": horizon,
            "frequency": frequency,
            "demand_column": demand_column,
            "model_name": model_name,
            "confidence_level": confidence_level,
        },
    }


# =====================================================================
# PART 4.18.21 — EXPORT COMPLETE FORECAST PACKAGE
# =====================================================================

def di4_export_forecast_package(
    package: dict[str, Any],
    output_directory: str | Path,
) -> dict[str, str]:
    """Export forecasts, timeline, and summary into separate files."""

    required_keys = {
        "forecast",
        "timeline",
        "summary",
        "metadata",
    }

    missing_keys = required_keys.difference(
        package
    )

    if missing_keys:
        raise ForecastingDataError(
            "Forecast package is incomplete: "
            + ", ".join(sorted(missing_keys))
        )

    output_dir = Path(
        output_directory
    ).expanduser()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecast_path = di4_export_forecast_csv(
        package["forecast"],
        output_dir / "future_demand_forecast.csv",
    )

    timeline_path = output_dir / "forecast_timeline.csv"

    package["timeline"].to_csv(
        timeline_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary_payload = {
        "summary": package["summary"],
        "metadata": package["metadata"],
    }

    summary_path = output_dir / "forecast_summary.json"

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary_payload,
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
            allow_nan=False,
        )

    DI4_LOGGER.info(
        "Forecast package exported to %s",
        output_dir.resolve(),
    )

    return {
        "forecast_csv": str(forecast_path),
        "timeline_csv": str(timeline_path.resolve()),
        "summary_json": str(summary_path.resolve()),
    }


# =====================================================================
# PART 4.18.22 — PUBLIC FUNCTION REGISTRY
# =====================================================================

DI4_FUNCTION_REGISTRY = {
    "validate_input": di4_validate_input,
    "prepare_demand_data": di4_prepare_demand_data,
    "data_quality_summary": di4_data_quality_summary,
    "aggregate_demand": di4_aggregate_demand,
    "complete_time_series": di4_complete_time_series,
    "build_forecasting_dataset": di4_build_forecasting_dataset,
    "naive_forecast": di4_naive_forecast,
    "moving_average_forecast": di4_moving_average_forecast,
    "seasonal_naive_forecast": di4_seasonal_naive_forecast,
    "evaluate_forecast": di4_evaluate_forecast,
    "time_ordered_split": di4_time_ordered_split,
    "compare_baseline_models": di4_compare_baseline_models,
    "select_best_baseline": di4_select_best_baseline,
    "generate_future_dates": di4_generate_future_dates,
    "build_prediction_intervals": di4_build_prediction_intervals,
    "forecast_future_demand": di4_forecast_future_demand,
    "build_forecast_summary": di4_build_forecast_summary,
    "build_forecast_timeline": di4_build_forecast_timeline,
    "build_model_report": di4_build_model_report,
    "build_forecast_package": di4_build_forecast_package,
    "export_forecast_csv": di4_export_forecast_csv,
    "export_model_metrics": di4_export_model_metrics,
    "export_model_report_json": di4_export_model_report_json,
    "export_forecast_package": di4_export_forecast_package,
}


# =====================================================================
# PART 4.18 — END
# FUTURE FORECASTING, CONFIDENCE INTERVALS, REPORTING AND EXPORTS
# =====================================================================
# =====================================================================
# PART 4.19 — START
# ADVANCED BACKTESTING, MODEL DIAGNOSTICS AND FORECAST QUALITY
# =====================================================================

from dataclasses import dataclass, asdict


# =====================================================================
# PART 4.19.1 — BACKTEST CONFIGURATION
# =====================================================================

DI4_BACKTEST_DEFAULT_FOLDS = 5
DI4_BACKTEST_MIN_TRAIN_SIZE = 28
DI4_BACKTEST_DEFAULT_HORIZON = 7

DI4_BACKTEST_METRICS = (
    "MAE",
    "RMSE",
    "MAPE_Percent",
    "WAPE_Percent",
)


# =====================================================================
# PART 4.19.2 — BACKTEST CONFIGURATION MODEL
# =====================================================================

@dataclass(frozen=True)
class DI4BacktestConfig:
    """Configuration for chronological rolling-origin evaluation."""

    horizon: int = DI4_BACKTEST_DEFAULT_HORIZON
    folds: int = DI4_BACKTEST_DEFAULT_FOLDS
    min_train_size: int = DI4_BACKTEST_MIN_TRAIN_SIZE
    moving_average_window: int = 7
    season_length: int = 7

    def validate(self) -> "DI4BacktestConfig":
        if self.horizon < 1:
            raise ValueError(
                "Backtest horizon must be positive."
            )

        if self.folds < 1:
            raise ValueError(
                "Number of backtest folds must be positive."
            )

        if self.min_train_size < 2:
            raise ValueError(
                "Minimum training size must be at least 2."
            )

        if self.moving_average_window < 1:
            raise ValueError(
                "Moving-average window must be positive."
            )

        if self.season_length < 1:
            raise ValueError(
                "Season length must be positive."
            )

        return self


# =====================================================================
# PART 4.19.3 — BUILD ROLLING BACKTEST WINDOWS
# =====================================================================

def di4_build_backtest_windows(
    observation_count: int,
    config: DI4BacktestConfig,
) -> list[tuple[int, int]]:
    """
    Build expanding-window backtest splits.

    Each pair represents:
        (training_end_index, validation_end_index)

    Training uses observations before training_end_index.
    Validation uses observations from training_end_index
    to validation_end_index - 1.
    """

    config.validate()

    if observation_count <= config.min_train_size:
        raise ForecastingDataError(
            "Not enough observations for backtesting."
        )

    possible_folds = (
        observation_count - config.min_train_size
    ) // config.horizon

    folds = min(
        config.folds,
        possible_folds,
    )

    if folds < 1:
        raise ForecastingDataError(
            "Insufficient history for even one complete "
            "backtest window."
        )

    first_validation_start = (
        observation_count - folds * config.horizon
    )

    if first_validation_start < config.min_train_size:
        raise ForecastingDataError(
            "Backtest configuration produces an invalid split."
        )

    windows = []

    for fold_index in range(folds):
        train_end = (
            first_validation_start
            + fold_index * config.horizon
        )

        validation_end = min(
            train_end + config.horizon,
            observation_count,
        )

        if train_end < config.min_train_size:
            continue

        if validation_end <= train_end:
            continue

        windows.append(
            (train_end, validation_end)
        )

    return windows


# =====================================================================
# PART 4.19.4 — GENERATE ONE MODEL'S BACKTEST PREDICTIONS
# =====================================================================

def di4_backtest_model_predictions(
    historical_data: pd.DataFrame,
    model_name: str,
    config: DI4BacktestConfig,
) -> pd.DataFrame:
    """
    Evaluate a model over multiple chronological validation windows.

    Each prediction is created using only data available before
    its corresponding validation period.
    """

    config.validate()

    history = di4_prepare_model_history(
        historical_data
    )

    windows = di4_build_backtest_windows(
        observation_count=len(history),
        config=config,
    )

    records = []

    for fold_number, (train_end, validation_end) in enumerate(
        windows,
        start=1,
    ):
        train = history.iloc[
            :train_end
        ].copy()

        validation = history.iloc[
            train_end:validation_end
        ].copy()

        predictions = di4_predict_with_model(
            historical_values=train["Actual_Demand"],
            horizon=len(validation),
            model_name=model_name,
            moving_average_window=(
                config.moving_average_window
            ),
            season_length=config.season_length,
        )

        for row_index, (_, row) in enumerate(
            validation.iterrows()
        ):
            prediction = float(
                predictions[row_index]
            )

            actual = float(
                row["Actual_Demand"]
            )

            records.append(
                {
                    "Fold": fold_number,
                    "Forecast_Date": row["Forecast_Date"],
                    "Actual_Demand": actual,
                    "Predicted_Demand": prediction,
                    "Residual": actual - prediction,
                    "Absolute_Error": abs(
                        actual - prediction
                    ),
                    "Model": model_name,
                    "Training_Observations": int(
                        len(train)
                    ),
                }
            )

    if not records:
        raise ForecastingDataError(
            "Backtesting did not produce any predictions."
        )

    return pd.DataFrame(records)


# =====================================================================
# PART 4.19.5 — BACKTEST ALL SUPPORTED BASELINE MODELS
# =====================================================================

def di4_backtest_all_models(
    historical_data: pd.DataFrame,
    config: Optional[DI4BacktestConfig] = None,
) -> pd.DataFrame:
    """
    Run rolling-origin backtesting for supported models.

    Seasonal-naive is skipped when the first training window
    does not contain enough observations for its seasonal period.
    """

    if config is None:
        config = DI4BacktestConfig()

    config.validate()

    history = di4_prepare_model_history(
        historical_data
    )

    result_frames = []

    for model_name in FORECAST_MODEL_NAMES:
        try:
            predictions = di4_backtest_model_predictions(
                historical_data=history,
                model_name=model_name,
                config=config,
            )

            result_frames.append(
                predictions
            )

        except ForecastingDataError as exc:
            DI4_LOGGER.warning(
                "Backtest skipped for %s: %s",
                model_name,
                exc,
            )

    if not result_frames:
        raise ForecastingDataError(
            "No supported model completed backtesting."
        )

    return pd.concat(
        result_frames,
        ignore_index=True,
    )


# =====================================================================
# PART 4.19.6 — CALCULATE FORECAST BIAS
# =====================================================================

def di4_calculate_forecast_bias(
    actual_values,
    predicted_values,
) -> dict[str, float]:
    """
    Calculate signed forecasting error.

    Positive residual:
        Actual demand was greater than predicted demand.

    Negative residual:
        Predicted demand exceeded actual demand.
    """

    actual = np.asarray(
        actual_values,
        dtype=float,
    ).reshape(-1)

    predicted = np.asarray(
        predicted_values,
        dtype=float,
    ).reshape(-1)

    if actual.size != predicted.size:
        raise ValueError(
            "Actual and predicted arrays must have equal lengths."
        )

    valid = (
        np.isfinite(actual)
        & np.isfinite(predicted)
    )

    actual = actual[valid]
    predicted = predicted[valid]

    if actual.size == 0:
        raise ForecastingDataError(
            "No valid observations are available for bias analysis."
        )

    residuals = actual - predicted

    mean_residual = float(
        np.mean(residuals)
    )

    mean_absolute_error = float(
        np.mean(np.abs(residuals))
    )

    total_actual = float(
        np.sum(np.abs(actual))
    )

    if total_actual > 0:
        normalized_bias_percent = float(
            np.sum(residuals)
            / total_actual
            * 100
        )
    else:
        normalized_bias_percent = np.nan

    if mean_residual > 0:
        bias_direction = "UNDER_FORECAST"
    elif mean_residual < 0:
        bias_direction = "OVER_FORECAST"
    else:
        bias_direction = "NO_MEAN_BIAS"

    return {
        "Mean_Residual": mean_residual,
        "Mean_Absolute_Error": mean_absolute_error,
        "Normalized_Bias_Percent": (
            normalized_bias_percent
        ),
        "Bias_Direction_Code": bias_direction,
    }


# =====================================================================
# PART 4.19.7 — MODEL METRICS BY BACKTEST FOLD
# =====================================================================

def di4_backtest_fold_metrics(
    backtest_results: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate error metrics for every model and fold."""

    required_columns = {
        "Fold",
        "Actual_Demand",
        "Predicted_Demand",
        "Model",
    }

    missing = required_columns.difference(
        backtest_results.columns
    )

    if missing:
        raise ForecastingDataError(
            "Backtest columns missing: "
            + ", ".join(sorted(missing))
        )

    records = []

    for (model_name, fold_number), group in (
        backtest_results.groupby(
            ["Model", "Fold"],
            sort=True,
        )
    ):
        metrics = di4_evaluate_forecast(
            actual_values=group["Actual_Demand"],
            predicted_values=group["Predicted_Demand"],
        )

        bias = di4_calculate_forecast_bias(
            actual_values=group["Actual_Demand"],
            predicted_values=group["Predicted_Demand"],
        )

        records.append(
            {
                "Model": model_name,
                "Fold": int(fold_number),
                **metrics,
                **bias,
            }
        )

    return pd.DataFrame(
        records
    )


# =====================================================================
# PART 4.19.8 — AGGREGATE BACKTEST METRICS
# =====================================================================

def di4_summarize_backtest_metrics(
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize model quality across validation folds.

    Mean and standard deviation of fold metrics are reported.
    They are descriptive statistics, not confidence intervals
    for future model performance.
    """

    required_columns = {
        "Model",
        "MAE",
        "RMSE",
        "MAPE_Percent",
        "WAPE_Percent",
    }

    missing = required_columns.difference(
        fold_metrics.columns
    )

    if missing:
        raise ForecastingDataError(
            "Fold metric columns missing: "
            + ", ".join(sorted(missing))
        )

    records = []

    for model_name, group in fold_metrics.groupby(
        "Model",
        sort=True,
    ):
        row = {
            "Model": model_name,
            "Fold_Count": int(len(group)),
        }

        for metric in DI4_BACKTEST_METRICS:
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            ).replace(
                [np.inf, -np.inf],
                np.nan,
            ).dropna()

            row[f"{metric}_Mean"] = (
                float(values.mean())
                if not values.empty
                else np.nan
            )

            row[f"{metric}_Std"] = (
                float(values.std(ddof=1))
                if len(values) > 1
                else np.nan
            )

        if "Normalized_Bias_Percent" in group.columns:
            bias_values = pd.to_numeric(
                group["Normalized_Bias_Percent"],
                errors="coerce",
            ).replace(
                [np.inf, -np.inf],
                np.nan,
            ).dropna()

            row["Normalized_Bias_Percent_Mean"] = (
                float(bias_values.mean())
                if not bias_values.empty
                else np.nan
            )

        records.append(
            row
        )

    result = pd.DataFrame(
        records
    )

    if "MAE_Mean" in result.columns:
        result = result.sort_values(
            "MAE_Mean",
            ascending=True,
            na_position="last",
        )

    return result.reset_index(
        drop=True
    )


# =====================================================================
# PART 4.19.9 — SELECT MODEL FROM BACKTEST RESULTS
# =====================================================================

def di4_select_model_from_backtest(
    summary_table: pd.DataFrame,
    metric_column: str = "MAE_Mean",
) -> str:
    """
    Select the baseline with the lowest average backtest error.

    This is a model-selection heuristic. It does not guarantee
    that the selected model will be best in the future.
    """

    if summary_table.empty:
        raise ForecastingDataError(
            "Backtest summary is empty."
        )

    if "Model" not in summary_table.columns:
        raise ForecastingDataError(
            "Backtest summary has no Model column."
        )

    if metric_column not in summary_table.columns:
        raise ForecastingDataError(
            f"Metric column not found: {metric_column}"
        )

    candidates = summary_table[
        ["Model", metric_column]
    ].copy()

    candidates[metric_column] = pd.to_numeric(
        candidates[metric_column],
        errors="coerce",
    )

    candidates = candidates.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna(
        subset=[metric_column]
    )

    if candidates.empty:
        raise ForecastingDataError(
            "No valid backtest metric is available for selection."
        )

    best = candidates.loc[
        candidates[metric_column].idxmin()
    ]

    return str(
        best["Model"]
    )


# =====================================================================
# PART 4.19.10 — FORECAST STABILITY DIAGNOSTICS
# =====================================================================

def di4_forecast_stability(
    forecast_values,
) -> dict[str, float]:
    """Measure variation within a forecast horizon."""

    values = np.asarray(
        forecast_values,
        dtype=float,
    ).reshape(-1)

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        raise ForecastingDataError(
            "No valid forecast values are available."
        )

    mean_value = float(
        np.mean(values)
    )

    std_value = float(
        np.std(values, ddof=0)
    )

    if mean_value != 0:
        coefficient_of_variation = float(
            std_value / abs(mean_value)
        )
    else:
        coefficient_of_variation = np.nan

    if values.size > 1:
        mean_absolute_step_change = float(
            np.mean(
                np.abs(np.diff(values))
            )
        )
    else:
        mean_absolute_step_change = 0.0

    return {
        "Forecast_Mean": mean_value,
        "Forecast_Std": std_value,
        "Forecast_Min": float(np.min(values)),
        "Forecast_Max": float(np.max(values)),
        "Coefficient_Of_Variation": (
            coefficient_of_variation
        ),
        "Mean_Absolute_Step_Change": (
            mean_absolute_step_change
        ),
    }


# =====================================================================
# PART 4.19.11 — DEMAND DISTRIBUTION DIAGNOSTICS
# =====================================================================

def di4_demand_distribution_summary(
    historical_data: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize historical demand distribution."""

    history = di4_prepare_model_history(
        historical_data
    )

    values = history[
        "Actual_Demand"
    ].astype(float)

    quantiles = values.quantile(
        [0.05, 0.25, 0.50, 0.75, 0.95]
    )

    return {
        "observations": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "standard_deviation": float(
            values.std(ddof=1)
        ) if len(values) > 1 else np.nan,
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "zero_observations": int(
            (values == 0).sum()
        ),
        "positive_observations": int(
            (values > 0).sum()
        ),
        "negative_observations": int(
            (values < 0).sum()
        ),
        "quantiles": {
            str(index): float(value)
            for index, value in quantiles.items()
        },
    }


# =====================================================================
# PART 4.19.12 — DETECT LARGE HISTORICAL ERRORS
# =====================================================================

def di4_detect_large_errors(
    backtest_results: pd.DataFrame,
    minimum_error: float = 0.0,
    error_quantile: float = 0.95,
) -> pd.DataFrame:
    """
    Flag unusually large absolute validation errors.

    The threshold is computed within each model using its
    backtest errors. Flags indicate unusual errors, not
    proof of a data problem.
    """

    if minimum_error < 0:
        raise ValueError(
            "Minimum error cannot be negative."
        )

    if not 0 < error_quantile < 1:
        raise ValueError(
            "Error quantile must be between 0 and 1."
        )

    required = {
        "Model",
        "Forecast_Date",
        "Actual_Demand",
        "Predicted_Demand",
    }

    missing = required.difference(
        backtest_results.columns
    )

    if missing:
        raise ForecastingDataError(
            "Backtest columns missing: "
            + ", ".join(sorted(missing))
        )

    result = backtest_results.copy()

    result["Absolute_Error"] = (
        pd.to_numeric(
            result["Actual_Demand"],
            errors="coerce",
        )
        - pd.to_numeric(
            result["Predicted_Demand"],
            errors="coerce",
        )
    ).abs()

    result["Error_Threshold"] = (
        result.groupby("Model")["Absolute_Error"]
        .transform(
            lambda values: max(
                minimum_error,
                float(
                    values.quantile(
                        error_quantile
                    )
                ),
            )
        )
    )

    result["Large_Error_Flag"] = (
        result["Absolute_Error"]
        > result["Error_Threshold"]
    )

    return result.loc[
        result["Large_Error_Flag"]
    ].reset_index(drop=True)


# =====================================================================
# PART 4.19.13 — COMPLETE BACKTEST REPORT
# =====================================================================

def di4_build_backtest_report(
    historical_data: pd.DataFrame,
    config: Optional[DI4BacktestConfig] = None,
) -> dict[str, Any]:
    """Build backtest predictions, metrics, and diagnostics."""

    if config is None:
        config = DI4BacktestConfig()

    config.validate()

    history = di4_prepare_model_history(
        historical_data
    )

    predictions = di4_backtest_all_models(
        historical_data=history,
        config=config,
    )

    fold_metrics = di4_backtest_fold_metrics(
        predictions
    )

    summary = di4_summarize_backtest_metrics(
        fold_metrics
    )

    best_model = di4_select_model_from_backtest(
        summary
    )

    large_errors = di4_detect_large_errors(
        predictions
    )

    distribution = di4_demand_distribution_summary(
        history
    )

    return {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "configuration": asdict(config),
        "observation_count": int(len(history)),
        "best_model_by_mean_mae": best_model,
        "predictions": predictions,
        "fold_metrics": fold_metrics,
        "model_summary": summary,
        "large_errors": large_errors,
        "demand_distribution": distribution,
    }


# =====================================================================
# PART 4.19.14 — EXPORT BACKTEST RESULTS
# =====================================================================

def di4_export_backtest_report(
    report: dict[str, Any],
    output_directory: str | Path,
) -> dict[str, str]:
    """Export backtest predictions and metrics as CSV files."""

    required_keys = {
        "predictions",
        "fold_metrics",
        "model_summary",
        "large_errors",
    }

    missing = required_keys.difference(
        report
    )

    if missing:
        raise ForecastingDataError(
            "Backtest report missing sections: "
            + ", ".join(sorted(missing))
        )

    output_dir = Path(
        output_directory
    ).expanduser()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {}

    export_items = {
        "predictions": "backtest_predictions.csv",
        "fold_metrics": "backtest_fold_metrics.csv",
        "model_summary": "backtest_model_summary.csv",
        "large_errors": "backtest_large_errors.csv",
    }

    for key, filename in export_items.items():
        dataframe = report[key]

        if not isinstance(dataframe, pd.DataFrame):
            raise ForecastingDataError(
                f"Backtest report section '{key}' must be a DataFrame."
            )

        path = output_dir / filename

        dataframe.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )

        outputs[key] = str(
            path.resolve()
        )

    metadata_path = output_dir / "backtest_metadata.json"

    metadata = {
        key: value
        for key, value in report.items()
        if not isinstance(value, pd.DataFrame)
    }

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
            allow_nan=False,
        )

    outputs["metadata"] = str(
        metadata_path.resolve()
    )

    return outputs


# =====================================================================
# PART 4.19.15 — BACKTEST FUNCTION REGISTRY
# =====================================================================

DI4_BACKTEST_FUNCTION_REGISTRY = {
    "build_windows": di4_build_backtest_windows,
    "model_predictions": di4_backtest_model_predictions,
    "backtest_all_models": di4_backtest_all_models,
    "calculate_bias": di4_calculate_forecast_bias,
    "fold_metrics": di4_backtest_fold_metrics,
    "summarize_metrics": di4_summarize_backtest_metrics,
    "select_model": di4_select_model_from_backtest,
    "forecast_stability": di4_forecast_stability,
    "demand_distribution": di4_demand_distribution_summary,
    "detect_large_errors": di4_detect_large_errors,
    "build_report": di4_build_backtest_report,
    "export_report": di4_export_backtest_report,
}


# =====================================================================
# PART 4.19 — END
# ADVANCED BACKTESTING, MODEL DIAGNOSTICS AND FORECAST QUALITY
# =====================================================================

# =====================================================================
# PART 4.20 — START
# AUTOMATED FORECAST PIPELINE AND FINAL REPORTING
# =====================================================================

from dataclasses import dataclass, field
from typing import Callable


# =====================================================================
# PART 4.20.1 — PIPELINE CONFIGURATION
# =====================================================================

@dataclass
class DI4PipelineConfig:
    """Configuration for the complete ferry demand forecast pipeline."""

    frequency: str = "daily"
    demand_column: str = "Sales Count"
    forecast_horizon: int = 14
    confidence_level: float = 0.95
    backtest_folds: int = 5
    backtest_horizon: int = 7
    minimum_training_observations: int = 28
    moving_average_window: int = 7
    season_length: int = 7
    output_directory: Optional[str] = None
    export_results: bool = True

    def validate(self) -> "DI4PipelineConfig":
        if self.frequency not in DI4_SUPPORTED_FREQUENCIES:
            raise ValueError(
                "Frequency must be daily, weekly, or monthly."
            )

        if self.demand_column not in FORECAST_DEMAND_COLUMNS:
            raise ValueError(
                f"Unsupported demand column: {self.demand_column}"
            )

        di4_validate_horizon(self.forecast_horizon)

        di4_validate_confidence_level(
            self.confidence_level
        )

        if self.backtest_folds < 1:
            raise ValueError(
                "Backtest folds must be positive."
            )

        if self.backtest_horizon < 1:
            raise ValueError(
                "Backtest horizon must be positive."
            )

        if self.minimum_training_observations < 2:
            raise ValueError(
                "Minimum training observations must be at least 2."
            )

        if self.moving_average_window < 1:
            raise ValueError(
                "Moving-average window must be positive."
            )

        if self.season_length < 1:
            raise ValueError(
                "Season length must be positive."
            )

        return self


# =====================================================================
# PART 4.20.2 — PIPELINE RESULT CONTAINER
# =====================================================================

@dataclass
class DI4PipelineResult:
    """Container holding the outputs of a forecast pipeline run."""

    status: str
    metadata: dict[str, Any] = field(default_factory=dict)
    quality_report: dict[str, Any] = field(default_factory=dict)
    forecast_summary: dict[str, Any] = field(default_factory=dict)
    selected_model: Optional[str] = None
    forecast: Optional[pd.DataFrame] = None
    timeline: Optional[pd.DataFrame] = None
    model_metrics: Optional[pd.DataFrame] = None
    backtest_predictions: Optional[pd.DataFrame] = None
    backtest_summary: Optional[pd.DataFrame] = None
    exports: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# =====================================================================
# PART 4.20.3 — INPUT QUALITY REPORT
# =====================================================================

def di4_pipeline_quality_report(
    raw_dataframe: pd.DataFrame,
    config: DI4PipelineConfig,
) -> dict[str, Any]:
    """Build a data quality report before forecasting."""

    config.validate()

    quality = di4_data_quality_summary(
        raw_dataframe,
        timestamp_column="Timestamp",
        demand_columns=FORECAST_DEMAND_COLUMNS,
    )

    prepared = di4_prepare_demand_data(
        raw_dataframe,
        timestamp_column="Timestamp",
        demand_columns=FORECAST_DEMAND_COLUMNS,
    )

    valid_timestamps = prepared[
        "Forecast_Timestamp"
    ].dropna()

    quality["valid_rows_after_timestamp_cleaning"] = int(
        len(prepared)
    )

    quality["demand_measure"] = config.demand_column

    if not valid_timestamps.empty:
        quality["first_timestamp"] = str(
            valid_timestamps.min()
        )
        quality["last_timestamp"] = str(
            valid_timestamps.max()
        )

    quality["negative_values_removed_or_marked"] = {
        column: int(
            (
                pd.to_numeric(
                    raw_dataframe[column],
                    errors="coerce",
                ) < 0
            ).sum()
        )
        for column in FORECAST_DEMAND_COLUMNS
    }

    return quality


# =====================================================================
# PART 4.20.4 — PREPARE PIPELINE HISTORY
# =====================================================================

def di4_pipeline_prepare_history(
    raw_dataframe: pd.DataFrame,
    config: DI4PipelineConfig,
) -> pd.DataFrame:
    """Convert raw ticket activity into a forecasting time series."""

    config.validate()

    quality = di4_validate_input(
        raw_dataframe,
        required_columns=FORECAST_REQUIRED_COLUMNS,
    )

    if not quality["valid"]:
        raise ForecastingDataError(
            "Input validation failed."
        )

    aggregated = di4_aggregate_demand(
        dataframe=raw_dataframe,
        frequency=config.frequency,
        demand_column=config.demand_column,
    )

    completed = di4_complete_time_series(
        aggregated_data=aggregated,
        frequency=config.frequency,
    )

    # Preserve missing periods as NaN.
    # Do not interpret missing records as zero ticket demand.
    completed["Demand_Measure"] = config.demand_column
    completed["Frequency"] = config.frequency

    return completed


# =====================================================================
# PART 4.20.5 — VALIDATE PIPELINE HISTORY LENGTH
# =====================================================================

def di4_pipeline_validate_history_length(
    history: pd.DataFrame,
    config: DI4PipelineConfig,
) -> dict[str, Any]:
    """Check whether enough historical observations exist."""

    valid_history = history.dropna(
        subset=["Actual_Demand"]
    )

    observation_count = len(valid_history)

    report = {
        "total_periods": int(len(history)),
        "valid_demand_periods": int(observation_count),
        "missing_demand_periods": int(
            history["Actual_Demand"].isna().sum()
        ),
        "minimum_required": int(
            config.minimum_training_observations
        ),
        "sufficient_history": (
            observation_count
            >= config.minimum_training_observations
        ),
    }

    if not report["sufficient_history"]:
        raise ForecastingDataError(
            "Insufficient valid historical observations. "
            f"Required at least "
            f"{config.minimum_training_observations}; "
            f"found {observation_count}."
        )

    return report


# =====================================================================
# PART 4.20.6 — BUILD BACKTEST CONFIGURATION
# =====================================================================

def di4_pipeline_backtest_config(
    config: DI4PipelineConfig,
) -> DI4BacktestConfig:
    """Translate pipeline settings into backtest settings."""

    config.validate()

    return DI4BacktestConfig(
        horizon=config.backtest_horizon,
        folds=config.backtest_folds,
        min_train_size=config.minimum_training_observations,
        moving_average_window=config.moving_average_window,
        season_length=config.season_length,
    ).validate()


# =====================================================================
# PART 4.20.7 — RUN MODEL BACKTESTING
# =====================================================================

def di4_pipeline_run_backtesting(
    history: pd.DataFrame,
    config: DI4PipelineConfig,
) -> dict[str, Any]:
    """Run rolling-origin validation for available baseline models."""

    backtest_config = di4_pipeline_backtest_config(
        config
    )

    predictions = di4_backtest_all_models(
        historical_data=history,
        config=backtest_config,
    )

    fold_metrics = di4_backtest_fold_metrics(
        predictions
    )

    summary = di4_summarize_backtest_metrics(
        fold_metrics
    )

    selected_model = di4_select_model_from_backtest(
        summary,
        metric_column="MAE_Mean",
    )

    return {
        "predictions": predictions,
        "fold_metrics": fold_metrics,
        "summary": summary,
        "selected_model": selected_model,
    }


# =====================================================================
# PART 4.20.8 — BUILD FORECAST USING SELECTED MODEL
# =====================================================================

def di4_pipeline_generate_forecast(
    history: pd.DataFrame,
    config: DI4PipelineConfig,
    selected_model: str,
) -> pd.DataFrame:
    """Generate future demand predictions with the selected model."""

    config.validate()

    forecast = di4_forecast_future_demand(
        historical_data=history,
        horizon=config.forecast_horizon,
        frequency=config.frequency,
        demand_column=config.demand_column,
        model_name=selected_model,
        confidence_level=config.confidence_level,
        validation_size=config.backtest_horizon,
        moving_average_window=config.moving_average_window,
        season_length=config.season_length,
    )

    return forecast


# =====================================================================
# PART 4.20.9 — CREATE FINAL PIPELINE SUMMARY
# =====================================================================

def di4_pipeline_summary(
    config: DI4PipelineConfig,
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    selected_model: str,
    model_metrics: pd.DataFrame,
) -> dict[str, Any]:
    """Combine forecast and model-selection metadata."""

    forecast_summary = di4_build_forecast_summary(
        forecast
    )

    return {
        "engine_version": FORECAST_ENGINE_VERSION,
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "frequency": config.frequency,
        "demand_measure": config.demand_column,
        "forecast_horizon": config.forecast_horizon,
        "selected_model": selected_model,
        "historical_periods": int(len(history)),
        "valid_historical_periods": int(
            history["Actual_Demand"].notna().sum()
        ),
        "forecast_summary": forecast_summary,
        "model_metrics": model_metrics.to_dict(
            orient="records"
        ),
        "interpretation": (
            "Forecasts estimate historical ticket activity patterns. "
            "They are not verified passenger counts, queue lengths, "
            "ferry capacity estimates, or guaranteed future outcomes."
        ),
    }


# =====================================================================
# PART 4.20.10 — SAFE EXPORT HELPERS
# =====================================================================

def di4_pipeline_export_json(
    payload: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Export a JSON-compatible report."""

    output_file = Path(
        output_path
    ).expanduser()

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
            allow_nan=False,
        )

    return output_file.resolve()


def di4_pipeline_export_csv(
    dataframe: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Export a non-empty DataFrame as CSV."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "CSV export requires a pandas DataFrame."
        )

    if dataframe.empty:
        raise ForecastingDataError(
            "Cannot export an empty DataFrame."
        )

    output_file = Path(
        output_path
    ).expanduser()

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    return output_file.resolve()


# =====================================================================
# PART 4.20.11 — EXPORT ALL PIPELINE RESULTS
# =====================================================================

def di4_pipeline_export_results(
    result: DI4PipelineResult,
    output_directory: str | Path,
) -> dict[str, str]:
    """Export available pipeline outputs."""

    output_dir = Path(
        output_directory
    ).expanduser()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    exports: dict[str, str] = {}

    dataframe_exports = {
        "forecast": result.forecast,
        "timeline": result.timeline,
        "model_metrics": result.model_metrics,
        "backtest_predictions": result.backtest_predictions,
        "backtest_summary": result.backtest_summary,
    }

    for name, dataframe in dataframe_exports.items():
        if (
            isinstance(dataframe, pd.DataFrame)
            and not dataframe.empty
        ):
            path = di4_pipeline_export_csv(
                dataframe,
                output_dir / f"{name}.csv",
            )

            exports[name] = str(path)

    summary_payload = {
        "status": result.status,
        "metadata": result.metadata,
        "quality_report": result.quality_report,
        "forecast_summary": result.forecast_summary,
        "selected_model": result.selected_model,
        "errors": result.errors,
        "warnings": result.warnings,
    }

    summary_path = di4_pipeline_export_json(
        summary_payload,
        output_dir / "pipeline_report.json",
    )

    exports["pipeline_report"] = str(
        summary_path
    )

    return exports


# =====================================================================
# PART 4.20.12 — RUN THE COMPLETE FORECAST PIPELINE
# =====================================================================

def di4_run_forecasting_pipeline(
    raw_dataframe: pd.DataFrame,
    config: Optional[DI4PipelineConfig] = None,
) -> DI4PipelineResult:
    """
    Execute the full forecasting workflow.

    Workflow:
        1. Validate configuration.
        2. Check raw data quality.
        3. Aggregate historical demand.
        4. Validate history length.
        5. Run rolling-origin backtesting.
        6. Select a baseline model by mean MAE.
        7. Generate future predictions.
        8. Build timeline and summary.
        9. Export results when requested.

    This function does not start a web application.
    """

    if config is None:
        config = DI4PipelineConfig()

    config.validate()

    result = DI4PipelineResult(
        status="STARTED",
    )

    result.metadata = {
        "engine_version": FORECAST_ENGINE_VERSION,
        "started_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "frequency": config.frequency,
        "demand_column": config.demand_column,
        "forecast_horizon": config.forecast_horizon,
    }

    try:
        result.quality_report = (
            di4_pipeline_quality_report(
                raw_dataframe,
                config,
            )
        )

        history = di4_pipeline_prepare_history(
            raw_dataframe,
            config,
        )

        history_quality = (
            di4_pipeline_validate_history_length(
                history,
                config,
            )
        )

        result.quality_report[
            "history_validation"
        ] = history_quality

        backtest_result = (
            di4_pipeline_run_backtesting(
                history,
                config,
            )
        )

        result.backtest_predictions = (
            backtest_result["predictions"]
        )

        result.model_metrics = (
            backtest_result["fold_metrics"]
        )

        result.backtest_summary = (
            backtest_result["summary"]
        )

        result.selected_model = (
            backtest_result["selected_model"]
        )

        forecast = di4_pipeline_generate_forecast(
            history=history,
            config=config,
            selected_model=result.selected_model,
        )

        result.forecast = forecast

        result.timeline = di4_build_forecast_timeline(
            history,
            forecast,
        )

        result.forecast_summary = (
            di4_pipeline_summary(
                config=config,
                history=history,
                forecast=forecast,
                selected_model=result.selected_model,
                model_metrics=result.backtest_summary,
            )
        )

        result.status = "SUCCESS"

    except Exception as exc:
        result.status = "FAILED"

        result.errors.append(
            f"{type(exc).__name__}: {exc}"
        )

        DI4_LOGGER.exception(
            "Forecasting pipeline failed."
        )

    if (
        result.status == "SUCCESS"
        and config.export_results
        and config.output_directory
    ):
        try:
            result.exports = (
                di4_pipeline_export_results(
                    result,
                    config.output_directory,
                )
            )

        except Exception as exc:
            result.warnings.append(
                "Forecast generation succeeded, but export failed: "
                f"{type(exc).__name__}: {exc}"
            )

            DI4_LOGGER.exception(
                "Forecast export failed."
            )

    result.metadata["finished_at_utc"] = (
        datetime.now(timezone.utc).isoformat()
    )

    result.metadata["final_status"] = result.status

    return result


# =====================================================================
# PART 4.20.13 — PIPELINE REGISTRY
# =====================================================================

DI4_PIPELINE_FUNCTION_REGISTRY = {
    "quality_report": di4_pipeline_quality_report,
    "prepare_history": di4_pipeline_prepare_history,
    "validate_history_length": di4_pipeline_validate_history_length,
    "build_backtest_config": di4_pipeline_backtest_config,
    "run_backtesting": di4_pipeline_run_backtesting,
    "generate_forecast": di4_pipeline_generate_forecast,
    "build_summary": di4_pipeline_summary,
    "export_json": di4_pipeline_export_json,
    "export_csv": di4_pipeline_export_csv,
    "export_results": di4_pipeline_export_results,
    "run_pipeline": di4_run_forecasting_pipeline,
}


# =====================================================================
# PART 4.20 — END
# AUTOMATED FORECAST PIPELINE AND FINAL REPORTING
# =====================================================================

# =====================================================================
# PART 4.21 — START
# RELIABILITY FIXES AND FORECASTING DATA CONTRACTS
# =====================================================================

import math
import logging
from collections.abc import Mapping
from datetime import date, datetime


# =====================================================================
# PART 4.21.1 — CENTRAL LOGGER
# =====================================================================

DI4_LOGGER = logging.getLogger("ferry_forecasting_engine")

if not DI4_LOGGER.handlers:
    DI4_LOGGER.addHandler(logging.NullHandler())


# =====================================================================
# PART 4.21.2 — SAFE CONVERSION OF NUMERIC VALUES
# =====================================================================

def di4_safe_float(
    value: Any,
    default: Optional[float] = None,
) -> Optional[float]:
    """Convert a value to a finite float or return a default."""

    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return default

    if not math.isfinite(converted):
        return default

    return converted


def di4_safe_integer(
    value: Any,
    default: Optional[int] = None,
) -> Optional[int]:
    """Convert a value to an integer when it is safely representable."""

    converted = di4_safe_float(value)

    if converted is None:
        return default

    if not converted.is_integer():
        return default

    return int(converted)


# =====================================================================
# PART 4.21.3 — RECURSIVE JSON SANITIZATION
# =====================================================================

def di4_json_safe_value(value: Any) -> Any:
    """
    Convert common pandas and NumPy values into JSON-safe Python values.

    Non-finite numbers become None rather than NaN or Infinity.
    """

    if value is None or value is pd.NA or value is pd.NaT:
        return None

    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    if isinstance(value, (np.integer, int)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        numeric_value = float(value)

        if not math.isfinite(numeric_value):
            return None

        return numeric_value

    if isinstance(value, (pd.Timestamp, datetime, date)):
        if pd.isna(value):
            return None

        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return [
            di4_json_safe_value(item)
            for item in value.tolist()
        ]

    if isinstance(value, pd.Series):
        return [
            di4_json_safe_value(item)
            for item in value.tolist()
        ]

    if isinstance(value, pd.DataFrame):
        records = value.to_dict(orient="records")

        return [
            di4_json_safe_value(record)
            for record in records
        ]

    if isinstance(value, Mapping):
        return {
            str(key): di4_json_safe_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            di4_json_safe_value(item)
            for item in value
        ]

    if isinstance(value, (str, bool)):
        return value

    if hasattr(value, "item"):
        try:
            return di4_json_safe_value(value.item())
        except (ValueError, TypeError, AttributeError):
            pass

    return str(value)


# =====================================================================
# PART 4.21.4 — REDEFINE SAFE JSON EXPORT
# =====================================================================

def di4_pipeline_export_json(
    payload: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Export JSON after converting NaN, Infinity and NumPy types."""

    if not isinstance(payload, dict):
        raise TypeError(
            "JSON export expects a dictionary."
        )

    output_file = Path(
        output_path
    ).expanduser()

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_payload = di4_json_safe_value(
        payload
    )

    temporary_file = output_file.with_suffix(
        output_file.suffix + ".tmp"
    )

    try:
        with temporary_file.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                safe_payload,
                file,
                indent=4,
                ensure_ascii=False,
                allow_nan=False,
            )

        temporary_file.replace(
            output_file
        )

    except Exception:
        if temporary_file.exists():
            temporary_file.unlink()

        raise

    return output_file.resolve()


# =====================================================================
# PART 4.21.5 — STANDARDIZE FREQUENCY VALIDATION
# =====================================================================

def di4_normalize_frequency(
    frequency: str,
) -> str:
    """Normalize accepted frequency names."""

    if not isinstance(frequency, str):
        raise ValueError(
            "Frequency must be a string."
        )

    normalized = frequency.strip().lower()

    aliases = {
        "d": "daily",
        "day": "daily",
        "daily": "daily",
        "w": "weekly",
        "week": "weekly",
        "weekly": "weekly",
        "m": "monthly",
        "month": "monthly",
        "monthly": "monthly",
    }

    if normalized not in aliases:
        raise ValueError(
            f"Unsupported frequency: {frequency}"
        )

    return aliases[normalized]


# =====================================================================
# PART 4.21.6 — WEEKLY DATE ALIGNMENT HELPER
# =====================================================================

def di4_week_start(
    values: pd.Series,
) -> pd.Series:
    """Map timestamps to Monday-start weekly periods."""

    timestamps = pd.to_datetime(
        values,
        errors="coerce",
    )

    return (
        timestamps.dt.normalize()
        - pd.to_timedelta(
            timestamps.dt.weekday,
            unit="D",
        )
    )


# =====================================================================
# PART 4.21.7 — REDEFINE FUTURE DATE GENERATION
# =====================================================================

def di4_generate_future_dates(
    historical_dates: pd.Series,
    horizon: int,
    frequency: str = "daily",
) -> pd.DatetimeIndex:
    """Generate future dates using consistent period boundaries."""

    horizon = di4_validate_horizon(
        horizon
    )

    frequency = di4_normalize_frequency(
        frequency
    )

    dates = pd.to_datetime(
        historical_dates,
        errors="coerce",
    )

    dates = dates.dropna()

    if dates.empty:
        raise ForecastingDataError(
            "Cannot generate future dates without valid history."
        )

    last_date = dates.max().normalize()

    if frequency == "daily":
        start_date = last_date + pd.Timedelta(days=1)

        return pd.date_range(
            start=start_date,
            periods=horizon,
            freq="D",
        )

    if frequency == "weekly":
        # Align the last historical date to Monday.
        last_monday = (
            last_date
            - pd.Timedelta(days=last_date.weekday())
        )

        first_future_monday = (
            last_monday
            + pd.Timedelta(days=7)
        )

        return pd.date_range(
            start=first_future_monday,
            periods=horizon,
            freq="7D",
        )

    # Monthly forecasts use the first day of each future month.
    current_month_start = last_date.replace(
        day=1
    )

    first_future_month = (
        current_month_start
        + pd.offsets.MonthBegin(1)
    )

    return pd.date_range(
        start=first_future_month,
        periods=horizon,
        freq="MS",
    )


# =====================================================================
# PART 4.21.8 — NORMALIZE AGGREGATED TIME SERIES
# =====================================================================

def di4_normalize_aggregated_dates(
    aggregated_data: pd.DataFrame,
    frequency: str = "daily",
) -> pd.DataFrame:
    """
    Normalize aggregated dates and validate demand values.

    Weekly timestamps are aligned to Monday.
    Monthly timestamps are aligned to month start.
    """

    frequency = di4_normalize_frequency(
        frequency
    )

    required = {
        "Forecast_Date",
        "Actual_Demand",
    }

    missing = required.difference(
        aggregated_data.columns
    )

    if missing:
        raise ForecastingDataError(
            f"Aggregated data is missing columns: "
            f"{sorted(missing)}"
        )

    result = aggregated_data.copy()

    result["Forecast_Date"] = pd.to_datetime(
        result["Forecast_Date"],
        errors="coerce",
    )

    result["Actual_Demand"] = pd.to_numeric(
        result["Actual_Demand"],
        errors="coerce",
    )

    result = result.dropna(
        subset=["Forecast_Date"]
    )

    result["Forecast_Date"] = (
        result["Forecast_Date"].dt.normalize()
    )

    if frequency == "weekly":
        result["Forecast_Date"] = (
            result["Forecast_Date"]
            - pd.to_timedelta(
                result["Forecast_Date"].dt.weekday,
                unit="D",
            )
        )

    elif frequency == "monthly":
        result["Forecast_Date"] = (
            result["Forecast_Date"]
            .dt.to_period("M")
            .dt.start_time
        )

    result = result.sort_values(
        "Forecast_Date"
    )

    if result["Forecast_Date"].duplicated().any():
        raise ForecastingDataError(
            "Multiple rows map to the same forecast period. "
            "Aggregate them before normalizing dates."
        )

    negative_mask = (
        result["Actual_Demand"].notna()
        & (result["Actual_Demand"] < 0)
    )

    if negative_mask.any():
        raise ForecastingDataError(
            "Negative demand values are not supported."
        )

    return result.reset_index(
        drop=True
    )


# =====================================================================
# PART 4.21.9 — REDEFINE COMPLETE TIME SERIES
# =====================================================================

def di4_complete_time_series(
    aggregated_data: pd.DataFrame,
    frequency: str = "daily",
) -> pd.DataFrame:
    """
    Create a continuous time series without fabricating demand.

    Missing periods remain NaN; they are not automatically set to zero.
    """

    frequency = di4_normalize_frequency(
        frequency
    )

    result = di4_normalize_aggregated_dates(
        aggregated_data,
        frequency=frequency,
    )

    if result.empty:
        raise ForecastingDataError(
            "No valid aggregated observations are available."
        )

    first_date = result["Forecast_Date"].min()
    last_date = result["Forecast_Date"].max()

    if frequency == "daily":
        complete_dates = pd.date_range(
            start=first_date,
            end=last_date,
            freq="D",
        )

    elif frequency == "weekly":
        complete_dates = pd.date_range(
            start=first_date,
            end=last_date,
            freq="7D",
        )

    else:
        complete_dates = pd.date_range(
            start=first_date,
            end=last_date,
            freq="MS",
        )

    result = (
        result.set_index("Forecast_Date")
        .reindex(complete_dates)
    )

    result.index.name = "Forecast_Date"

    result = result.reset_index()

    if "Observation_Count" in result.columns:
        result["Observation_Count"] = (
            result["Observation_Count"]
            .fillna(0)
            .astype("int64")
        )

    return result


# =====================================================================
# PART 4.21.10 — CHECK FORECAST OUTPUT CONTRACT
# =====================================================================

def di4_validate_forecast_output(
    forecast: pd.DataFrame,
    expected_horizon: Optional[int] = None,
) -> dict[str, Any]:
    """Validate forecast structure before reporting or export."""

    if not isinstance(forecast, pd.DataFrame):
        raise ForecastingDataError(
            "Forecast output must be a pandas DataFrame."
        )

    required_columns = {
        "Forecast_Date",
        "Forecast_Demand",
    }

    missing = required_columns.difference(
        forecast.columns
    )

    if missing:
        raise ForecastingDataError(
            f"Forecast output is missing columns: "
            f"{sorted(missing)}"
        )

    if forecast.empty:
        raise ForecastingDataError(
            "Forecast output is empty."
        )

    checked = forecast.copy()

    checked["Forecast_Date"] = pd.to_datetime(
        checked["Forecast_Date"],
        errors="coerce",
    )

    checked["Forecast_Demand"] = pd.to_numeric(
        checked["Forecast_Demand"],
        errors="coerce",
    )

    if checked["Forecast_Date"].isna().any():
        raise ForecastingDataError(
            "Forecast contains invalid dates."
        )

    if checked["Forecast_Demand"].isna().any():
        raise ForecastingDataError(
            "Forecast contains missing demand predictions."
        )

    if not np.isfinite(
        checked["Forecast_Demand"].to_numpy(
            dtype=float
        )
    ).all():
        raise ForecastingDataError(
            "Forecast contains infinite predictions."
        )

    if (checked["Forecast_Demand"] < 0).any():
        raise ForecastingDataError(
            "Forecast demand cannot be negative."
        )

    if checked["Forecast_Date"].duplicated().any():
        raise ForecastingDataError(
            "Forecast contains duplicate dates."
        )

    if not checked["Forecast_Date"].is_monotonic_increasing:
        raise ForecastingDataError(
            "Forecast dates must be sorted chronologically."
        )

    if (
        expected_horizon is not None
        and len(checked) != expected_horizon
    ):
        raise ForecastingDataError(
            f"Expected {expected_horizon} forecast rows, "
            f"received {len(checked)}."
        )

    return {
        "valid": True,
        "forecast_rows": int(len(checked)),
        "first_forecast_date": str(
            checked["Forecast_Date"].min()
        ),
        "last_forecast_date": str(
            checked["Forecast_Date"].max()
        ),
        "negative_predictions": 0,
        "duplicate_dates": 0,
    }


# =====================================================================
# PART 4.21.11 — CHECK PREDICTION INTERVALS
# =====================================================================

def di4_validate_prediction_intervals(
    forecast: pd.DataFrame,
) -> dict[str, Any]:
    """Validate lower and upper prediction bounds when present."""

    required = {
        "Forecast_Demand",
        "Lower_Bound",
        "Upper_Bound",
    }

    if not required.issubset(
        forecast.columns
    ):
        return {
            "interval_columns_available": False,
            "intervals_validated": False,
            "reason": (
                "One or more prediction interval columns are missing."
            ),
        }

    lower = pd.to_numeric(
        forecast["Lower_Bound"],
        errors="coerce",
    )

    predicted = pd.to_numeric(
        forecast["Forecast_Demand"],
        errors="coerce",
    )

    upper = pd.to_numeric(
        forecast["Upper_Bound"],
        errors="coerce",
    )

    finite_mask = (
        np.isfinite(lower.to_numpy(dtype=float))
        & np.isfinite(predicted.to_numpy(dtype=float))
        & np.isfinite(upper.to_numpy(dtype=float))
    )

    valid_order = (
        (lower <= predicted)
        & (predicted <= upper)
    )

    invalid_count = int(
        (~valid_order | ~finite_mask).sum()
    )

    return {
        "interval_columns_available": True,
        "intervals_validated": True,
        "interval_rows": int(len(forecast)),
        "invalid_interval_rows": invalid_count,
        "all_intervals_valid": invalid_count == 0,
    }


# =====================================================================
# PART 4.21.12 — FORECAST DIAGNOSTICS
# =====================================================================

def di4_build_forecast_diagnostics(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    expected_horizon: Optional[int] = None,
) -> dict[str, Any]:
    """Build structural diagnostics for a generated forecast."""

    if "Actual_Demand" not in history.columns:
        raise ForecastingDataError(
            "History must contain Actual_Demand."
        )

    historical_values = pd.to_numeric(
        history["Actual_Demand"],
        errors="coerce",
    )

    valid_values = historical_values.dropna()

    diagnostics = {
        "historical_periods": int(len(history)),
        "valid_historical_periods": int(
            valid_values.size
        ),
        "missing_historical_periods": int(
            historical_values.isna().sum()
        ),
        "historical_zero_demand_periods": int(
            (valid_values == 0).sum()
        ),
        "historical_negative_demand_periods": int(
            (valid_values < 0).sum()
        ),
        "forecast_validation": (
            di4_validate_forecast_output(
                forecast,
                expected_horizon=expected_horizon,
            )
        ),
        "prediction_interval_validation": (
            di4_validate_prediction_intervals(
                forecast
            )
        ),
    }

    if not valid_values.empty:
        diagnostics["historical_mean_demand"] = (
            di4_safe_float(
                valid_values.mean()
            )
        )

        diagnostics["historical_median_demand"] = (
            di4_safe_float(
                valid_values.median()
            )
        )

        diagnostics["historical_standard_deviation"] = (
            di4_safe_float(
                valid_values.std()
            )
        )

    return diagnostics


# =====================================================================
# PART 4.21.13 — DIAGNOSTIC REGISTRY
# =====================================================================

DI4_RELIABILITY_FUNCTION_REGISTRY = {
    "safe_float": di4_safe_float,
    "safe_integer": di4_safe_integer,
    "json_safe_value": di4_json_safe_value,
    "normalize_frequency": di4_normalize_frequency,
    "week_start": di4_week_start,
    "generate_future_dates": di4_generate_future_dates,
    "normalize_aggregated_dates": di4_normalize_aggregated_dates,
    "complete_time_series": di4_complete_time_series,
    "validate_forecast_output": di4_validate_forecast_output,
    "validate_prediction_intervals": di4_validate_prediction_intervals,
    "build_forecast_diagnostics": di4_build_forecast_diagnostics,
}


# =====================================================================
# PART 4.21 — END
# RELIABILITY FIXES AND FORECASTING DATA CONTRACTS
# =====================================================================

# =====================================================================
# PART 4.22 — START
# AUTOMATED TESTS AND INTEGRATION VALIDATION
# =====================================================================

import tempfile
import traceback
from dataclasses import dataclass, field


# =====================================================================
# PART 4.22.1 — TEST RESULT MODEL
# =====================================================================

@dataclass
class DI4TestResult:
    """Stores the result of an individual automated test."""

    name: str
    passed: bool
    message: str = ""
    duration_seconds: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


# =====================================================================
# PART 4.22.2 — TEST RUNNER
# =====================================================================

class DI4TestRunner:
    """Minimal test runner that collects failures without stopping."""

    def __init__(self) -> None:
        self.results: list[DI4TestResult] = []

    def run(
        self,
        name: str,
        test_function: Callable[[], Any],
    ) -> DI4TestResult:
        started = datetime.now(timezone.utc)

        try:
            details = test_function()

            if not isinstance(details, dict):
                details = {"result": str(details)}

            result = DI4TestResult(
                name=name,
                passed=True,
                message="Test passed.",
                details=details,
            )

        except Exception as exc:
            result = DI4TestResult(
                name=name,
                passed=False,
                message=f"{type(exc).__name__}: {exc}",
                details={
                    "traceback": traceback.format_exc(),
                },
            )

        finished = datetime.now(timezone.utc)

        result.duration_seconds = max(
            0.0,
            (finished - started).total_seconds(),
        )

        self.results.append(result)

        return result

    def summary(self) -> dict[str, Any]:
        total = len(self.results)
        passed = sum(
            result.passed
            for result in self.results
        )

        failed = total - passed

        return {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "success_rate_percent": (
                round(passed / total * 100, 2)
                if total
                else 0.0
            ),
            "all_passed": failed == 0 and total > 0,
            "results": [
                {
                    "name": result.name,
                    "passed": result.passed,
                    "message": result.message,
                    "duration_seconds": round(
                        result.duration_seconds,
                        6,
                    ),
                    "details": result.details,
                }
                for result in self.results
            ],
        }


# =====================================================================
# PART 4.22.3 — ASSERTION HELPERS
# =====================================================================

def di4_assert_true(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)


def di4_assert_equal(
    actual: Any,
    expected: Any,
    message: str = "",
) -> None:
    if actual != expected:
        raise AssertionError(
            message
            or f"Expected {expected!r}; received {actual!r}."
        )


def di4_assert_close(
    actual: float,
    expected: float,
    tolerance: float = 1e-6,
    message: str = "",
) -> None:
    if not math.isclose(
        float(actual),
        float(expected),
        rel_tol=tolerance,
        abs_tol=tolerance,
    ):
        raise AssertionError(
            message
            or f"Expected approximately {expected}; received {actual}."
        )


def di4_assert_columns(
    dataframe: pd.DataFrame,
    expected_columns: set[str],
) -> None:
    missing = expected_columns.difference(
        dataframe.columns
    )

    if missing:
        raise AssertionError(
            f"Missing expected columns: {sorted(missing)}"
        )


# =====================================================================
# PART 4.22.4 — SYNTHETIC TEST DATA
# =====================================================================

def di4_test_make_raw_data(
    periods: int = 90,
) -> pd.DataFrame:
    """Create deterministic synthetic ticket activity for tests."""

    if periods < 1:
        raise ValueError(
            "Periods must be positive."
        )

    dates = pd.date_range(
        start="2025-01-01",
        periods=periods,
        freq="D",
    )

    index = np.arange(periods)

    sales = (
        100
        + 15 * np.sin(index * 2 * np.pi / 7)
        + index * 0.2
    )

    redemptions = (
        70
        + 10 * np.sin(index * 2 * np.pi / 7)
        + index * 0.1
    )

    return pd.DataFrame(
        {
            "_id": np.arange(1, periods + 1),
            "Timestamp": dates,
            "Sales Count": np.maximum(
                0,
                np.round(sales),
            ).astype(int),
            "Redemption Count": np.maximum(
                0,
                np.round(redemptions),
            ).astype(int),
        }
    )


def di4_test_make_history(
    periods: int = 90,
    frequency: str = "daily",
) -> pd.DataFrame:
    """Create a forecasting history DataFrame."""

    raw = di4_test_make_raw_data(
        periods=periods
    )

    aggregated = di4_aggregate_demand(
        raw,
        frequency=frequency,
        demand_column="Sales Count",
    )

    return di4_complete_time_series(
        aggregated,
        frequency=frequency,
    )


# =====================================================================
# PART 4.22.5 — TEST CONFIGURATION VALIDATION
# =====================================================================

def di4_test_configuration_validation() -> dict[str, Any]:
    config = DI4PipelineConfig()

    config.validate()

    di4_assert_equal(
        config.frequency,
        "daily",
    )

    di4_assert_equal(
        config.forecast_horizon,
        14,
    )

    invalid_config = DI4PipelineConfig(
        frequency="yearly"
    )

    rejected = False

    try:
        invalid_config.validate()
    except ValueError:
        rejected = True

    di4_assert_true(
        rejected,
        "Unsupported frequency should be rejected.",
    )

    return {
        "default_configuration_valid": True,
        "unsupported_frequency_rejected": rejected,
    }


# =====================================================================
# PART 4.22.6 — TEST INPUT DATA VALIDATION
# =====================================================================

def di4_test_input_validation() -> dict[str, Any]:
    raw = di4_test_make_raw_data(
        periods=20
    )

    validation = di4_validate_input(
        raw,
        required_columns=FORECAST_REQUIRED_COLUMNS,
    )

    di4_assert_true(
        bool(validation["valid"]),
        "Valid synthetic data was rejected.",
    )

    missing_column_data = raw.drop(
        columns=["Sales Count"]
    )

    missing_column_rejected = False

    try:
        di4_validate_input(
            missing_column_data,
            required_columns=FORECAST_REQUIRED_COLUMNS,
        )
    except Exception:
        missing_column_rejected = True

    di4_assert_true(
        missing_column_rejected,
        "Missing required column should be rejected.",
    )

    return {
        "valid_data_accepted": True,
        "missing_required_column_rejected": True,
    }


# =====================================================================
# PART 4.22.7 — TEST DATA PREPARATION
# =====================================================================

def di4_test_data_preparation() -> dict[str, Any]:
    raw = di4_test_make_raw_data(
        periods=30
    )

    prepared = di4_prepare_demand_data(
        raw
    )

    di4_assert_columns(
        prepared,
        {
            "Forecast_Timestamp",
            "Sales Count",
            "Redemption Count",
        },
    )

    di4_assert_true(
        prepared["Forecast_Timestamp"].notna().all(),
        "Prepared timestamps contain invalid values.",
    )

    di4_assert_true(
        prepared["Forecast_Timestamp"].is_monotonic_increasing,
        "Prepared timestamps are not sorted.",
    )

    return {
        "input_rows": len(raw),
        "prepared_rows": len(prepared),
        "timestamps_sorted": True,
    }


# =====================================================================
# PART 4.22.8 — TEST AGGREGATION
# =====================================================================

def di4_test_daily_aggregation() -> dict[str, Any]:
    raw = di4_test_make_raw_data(
        periods=31
    )

    aggregated = di4_aggregate_demand(
        raw,
        frequency="daily",
        demand_column="Sales Count",
    )

    di4_assert_columns(
        aggregated,
        {
            "Forecast_Date",
            "Actual_Demand",
        },
    )

    di4_assert_equal(
        len(aggregated),
        31,
        "Daily aggregation returned an unexpected row count.",
    )

    di4_assert_true(
        (aggregated["Actual_Demand"] >= 0).all(),
        "Aggregated demand contains negative values.",
    )

    return {
        "daily_rows": len(aggregated),
        "minimum_demand": float(
            aggregated["Actual_Demand"].min()
        ),
        "maximum_demand": float(
            aggregated["Actual_Demand"].max()
        ),
    }


# =====================================================================
# PART 4.22.9 — TEST CONTINUOUS TIME SERIES
# =====================================================================

def di4_test_complete_time_series() -> dict[str, Any]:
    raw = di4_test_make_raw_data(
        periods=10
    )

    aggregated = di4_aggregate_demand(
        raw,
        frequency="daily",
        demand_column="Sales Count",
    )

    completed = di4_complete_time_series(
        aggregated,
        frequency="daily",
    )

    di4_assert_equal(
        len(completed),
        10,
    )

    di4_assert_true(
        completed["Forecast_Date"].is_monotonic_increasing,
        "Completed time series is not sorted.",
    )

    return {
        "completed_periods": len(completed),
        "dates_sorted": True,
    }


# =====================================================================
# PART 4.22.10 — TEST MISSING PERIOD HANDLING
# =====================================================================

def di4_test_missing_period_handling() -> dict[str, Any]:
    dates = pd.to_datetime(
        [
            "2025-01-01",
            "2025-01-03",
        ]
    )

    aggregated = pd.DataFrame(
        {
            "Forecast_Date": dates,
            "Actual_Demand": [100.0, 120.0],
            "Observation_Count": [1, 1],
        }
    )

    completed = di4_complete_time_series(
        aggregated,
        frequency="daily",
    )

    di4_assert_equal(
        len(completed),
        3,
    )

    missing_row = completed.loc[
        completed["Forecast_Date"]
        == pd.Timestamp("2025-01-02")
    ]

    di4_assert_equal(
        len(missing_row),
        1,
    )

    di4_assert_true(
        pd.isna(
            missing_row["Actual_Demand"].iloc[0]
        ),
        "Missing period should remain NaN, not become zero.",
    )

    return {
        "continuous_periods": len(completed),
        "missing_period_preserved": True,
    }


# =====================================================================
# PART 4.22.11 — TEST BASELINE MODELS
# =====================================================================

def di4_test_baseline_models() -> dict[str, Any]:
    values = np.array(
        [10, 20, 30, 40, 50],
        dtype=float,
    )

    naive = di4_naive_forecast(
        values,
        horizon=3,
    )

    moving_average = di4_moving_average_forecast(
        values,
        horizon=3,
        window=3,
    )

    seasonal = di4_seasonal_naive_forecast(
        np.arange(1, 15, dtype=float),
        horizon=4,
        season_length=7,
    )

    di4_assert_true(
        np.allclose(
            naive,
            [50, 50, 50],
        ),
        "Naive forecast returned unexpected values.",
    )

    di4_assert_true(
        np.allclose(
            moving_average,
            [40, 40, 40],
        ),
        "Moving-average forecast returned unexpected values.",
    )

    di4_assert_true(
        np.allclose(
            seasonal,
            [8, 9, 10, 11],
        ),
        "Seasonal-naive forecast returned unexpected values.",
    )

    return {
        "naive_forecast": naive.tolist(),
        "moving_average_forecast": moving_average.tolist(),
        "seasonal_naive_forecast": seasonal.tolist(),
    }


# =====================================================================
# PART 4.22.12 — TEST FORECAST METRICS
# =====================================================================

def di4_test_forecast_metrics() -> dict[str, Any]:
    actual = np.array(
        [100, 200, 300],
        dtype=float,
    )

    predicted = np.array(
        [110, 190, 310],
        dtype=float,
    )

    metrics = di4_evaluate_forecast(
        actual,
        predicted,
    )

    di4_assert_true(
        metrics["MAE"] >= 0,
        "MAE must not be negative.",
    )

    di4_assert_true(
        metrics["RMSE"] >= 0,
        "RMSE must not be negative.",
    )

    di4_assert_close(
        metrics["MAE"],
        10.0,
    )

    return {
        "MAE": float(metrics["MAE"]),
        "RMSE": float(metrics["RMSE"]),
        "WAPE": di4_safe_float(
            metrics.get("WAPE_Percent")
        ),
    }


# =====================================================================
# PART 4.22.13 — TEST FUTURE DATE GENERATION
# =====================================================================

def di4_test_future_dates() -> dict[str, Any]:
    history = pd.Series(
        pd.date_range(
            "2025-01-01",
            periods=30,
            freq="D",
        )
    )

    daily = di4_generate_future_dates(
        history,
        horizon=7,
        frequency="daily",
    )

    weekly = di4_generate_future_dates(
        history,
        horizon=4,
        frequency="weekly",
    )

    monthly = di4_generate_future_dates(
        history,
        horizon=3,
        frequency="monthly",
    )

    di4_assert_equal(
        len(daily),
        7,
    )

    di4_assert_equal(
        len(weekly),
        4,
    )

    di4_assert_equal(
        len(monthly),
        3,
    )

    di4_assert_true(
        (weekly.dayofweek == 0).all(),
        "Weekly forecast dates should align to Monday.",
    )

    di4_assert_true(
        (monthly.day == 1).all(),
        "Monthly forecast dates should align to month start.",
    )

    return {
        "daily_horizon": len(daily),
        "weekly_horizon": len(weekly),
        "monthly_horizon": len(monthly),
        "weekly_dates_monday_aligned": True,
        "monthly_dates_month_start_aligned": True,
    }


# =====================================================================
# PART 4.22.14 — TEST JSON SANITIZATION
# =====================================================================

def di4_test_json_sanitization() -> dict[str, Any]:
    payload = {
        "valid_number": np.float64(12.5),
        "integer": np.int64(7),
        "missing": np.nan,
        "positive_infinity": np.inf,
        "negative_infinity": -np.inf,
        "nested": {
            "items": [
                np.float64(2.5),
                pd.NaT,
            ]
        },
    }

    safe = di4_json_safe_value(
        payload
    )

    di4_assert_equal(
        safe["valid_number"],
        12.5,
    )

    di4_assert_equal(
        safe["integer"],
        7,
    )

    di4_assert_equal(
        safe["missing"],
        None,
    )

    di4_assert_equal(
        safe["positive_infinity"],
        None,
    )

    di4_assert_equal(
        safe["negative_infinity"],
        None,
    )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "test.json"

        di4_pipeline_export_json(
            payload,
            path,
        )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            restored = json.load(file)

    di4_assert_equal(
        restored["missing"],
        None,
    )

    return {
        "numpy_values_supported": True,
        "non_finite_numbers_sanitized": True,
        "json_round_trip_successful": True,
    }


# =====================================================================
# PART 4.22.15 — TEST FORECAST OUTPUT VALIDATION
# =====================================================================

def di4_test_forecast_output_validation() -> dict[str, Any]:
    forecast = pd.DataFrame(
        {
            "Forecast_Date": pd.date_range(
                "2025-04-01",
                periods=3,
                freq="D",
            ),
            "Forecast_Demand": [
                100.0,
                110.0,
                105.0,
            ],
        }
    )

    result = di4_validate_forecast_output(
        forecast,
        expected_horizon=3,
    )

    di4_assert_true(
        result["valid"],
        "Valid forecast output was rejected.",
    )

    invalid_forecast = forecast.copy()

    invalid_forecast.loc[
        0,
        "Forecast_Demand",
    ] = -1

    rejected = False

    try:
        di4_validate_forecast_output(
            invalid_forecast
        )
    except ForecastingDataError:
        rejected = True

    di4_assert_true(
        rejected,
        "Negative forecast prediction should be rejected.",
    )

    return {
        "valid_forecast_accepted": True,
        "negative_prediction_rejected": True,
    }


# =====================================================================
# PART 4.22.16 — TEST PREDICTION INTERVALS
# =====================================================================

def di4_test_prediction_intervals() -> dict[str, Any]:
    forecast = pd.DataFrame(
        {
            "Forecast_Demand": [100.0, 110.0],
            "Lower_Bound": [80.0, 90.0],
            "Upper_Bound": [120.0, 130.0],
        }
    )

    result = di4_validate_prediction_intervals(
        forecast
    )

    di4_assert_true(
        result["all_intervals_valid"],
        "Correct prediction intervals were rejected.",
    )

    invalid = forecast.copy()

    invalid.loc[
        0,
        "Lower_Bound",
    ] = 150.0

    invalid_result = di4_validate_prediction_intervals(
        invalid
    )

    di4_assert_true(
        not invalid_result["all_intervals_valid"],
        "Invalid interval ordering was not detected.",
    )

    return {
        "valid_intervals_accepted": True,
        "invalid_intervals_detected": True,
    }


# =====================================================================
# PART 4.22.17 — TEST BACKTEST WINDOW GENERATION
# =====================================================================

def di4_test_backtest_windows() -> dict[str, Any]:
    config = DI4BacktestConfig(
        horizon=7,
        folds=3,
        min_train_size=28,
    )

    windows = di4_build_backtest_windows(
        observation_count=70,
        config=config,
    )

    di4_assert_true(
        len(windows) > 0,
        "No backtest windows were generated.",
    )

    for window in windows:
        di4_assert_true(
            window["train_end"] <= window["test_start"],
            "Training and validation windows overlap incorrectly.",
        )

    return {
        "generated_windows": len(windows),
        "chronological_order_validated": True,
    }


# =====================================================================
# PART 4.22.18 — TEST PIPELINE CONFIGURATION AND HISTORY
# =====================================================================

def di4_test_pipeline_history() -> dict[str, Any]:
    raw = di4_test_make_raw_data(
        periods=90
    )

    config = DI4PipelineConfig(
        frequency="daily",
        demand_column="Sales Count",
        minimum_training_observations=28,
    )

    history = di4_pipeline_prepare_history(
        raw,
        config,
    )

    validation = di4_pipeline_validate_history_length(
        history,
        config,
    )

    di4_assert_true(
        validation["sufficient_history"],
        "Sufficient synthetic history was rejected.",
    )

    return {
        "historical_periods": len(history),
        "valid_periods": validation[
            "valid_demand_periods"
        ],
    }


# =====================================================================
# PART 4.22.19 — TEST RUNNER ENTRYPOINT
# =====================================================================

def di4_run_unit_tests() -> dict[str, Any]:
    """
    Run deterministic unit tests.

    These tests use synthetic data and temporary files.
    They do not load the real ferry dataset or start Streamlit.
    """

    runner = DI4TestRunner()

    test_cases = [
        (
            "configuration_validation",
            di4_test_configuration_validation,
        ),
        (
            "input_validation",
            di4_test_input_validation,
        ),
        (
            "data_preparation",
            di4_test_data_preparation,
        ),
        (
            "daily_aggregation",
            di4_test_daily_aggregation,
        ),
        (
            "complete_time_series",
            di4_test_complete_time_series,
        ),
        (
            "missing_period_handling",
            di4_test_missing_period_handling,
        ),
        (
            "baseline_models",
            di4_test_baseline_models,
        ),
        (
            "forecast_metrics",
            di4_test_forecast_metrics,
        ),
        (
            "future_dates",
            di4_test_future_dates,
        ),
        (
            "json_sanitization",
            di4_test_json_sanitization,
        ),
        (
            "forecast_output_validation",
            di4_test_forecast_output_validation,
        ),
        (
            "prediction_intervals",
            di4_test_prediction_intervals,
        ),
        (
            "backtest_windows",
            di4_test_backtest_windows,
        ),
        (
            "pipeline_history",
            di4_test_pipeline_history,
        ),
    ]

    for test_name, test_function in test_cases:
        runner.run(
            name=test_name,
            test_function=test_function,
        )

    report = runner.summary()

    DI4_LOGGER.info(
        "Forecast unit tests completed: %s passed, %s failed.",
        report["passed"],
        report["failed"],
    )

    return report


# =====================================================================
# PART 4.22.20 — TEST REPORT EXPORT
# =====================================================================

def di4_export_unit_test_report(
    report: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Save a test report as JSON."""

    required_keys = {
        "total_tests",
        "passed",
        "failed",
        "results",
    }

    missing = required_keys.difference(
        report.keys()
    )

    if missing:
        raise ValueError(
            f"Test report is missing keys: {sorted(missing)}"
        )

    return di4_pipeline_export_json(
        report,
        output_path,
    )


# =====================================================================
# PART 4.22.21 — TEST FUNCTION REGISTRY
# =====================================================================

DI4_TEST_FUNCTION_REGISTRY = {
    "configuration_validation": di4_test_configuration_validation,
    "input_validation": di4_test_input_validation,
    "data_preparation": di4_test_data_preparation,
    "daily_aggregation": di4_test_daily_aggregation,
    "complete_time_series": di4_test_complete_time_series,
    "missing_period_handling": di4_test_missing_period_handling,
    "baseline_models": di4_test_baseline_models,
    "forecast_metrics": di4_test_forecast_metrics,
    "future_dates": di4_test_future_dates,
    "json_sanitization": di4_test_json_sanitization,
    "forecast_output_validation": di4_test_forecast_output_validation,
    "prediction_intervals": di4_test_prediction_intervals,
    "backtest_windows": di4_test_backtest_windows,
    "pipeline_history": di4_test_pipeline_history,
    "run_all": di4_run_unit_tests,
    "export_report": di4_export_unit_test_report,
}


# =====================================================================
# PART 4.22 — END
# AUTOMATED TESTS AND INTEGRATION VALIDATION
# =====================================================================

# =====================================================================
# PART 4.23 — START
# FORECAST MONITORING, DRIFT DETECTION AND PERFORMANCE TRACKING
# =====================================================================

from dataclasses import dataclass, field


# =====================================================================
# PART 4.23.1 — MONITORING CONFIGURATION
# =====================================================================

@dataclass
class DI4MonitoringConfig:
    """Configuration for forecast performance monitoring."""

    demand_column: str = "Sales Count"
    timestamp_column: str = "Timestamp"
    rolling_window: int = 14
    drift_window: int = 14
    minimum_observations: int = 7
    performance_threshold_percent: float = 25.0
    drift_threshold: float = 0.25

    def validate(self) -> "DI4MonitoringConfig":
        if self.demand_column not in FORECAST_DEMAND_COLUMNS:
            raise ValueError(
                f"Unsupported demand column: {self.demand_column}"
            )

        if self.rolling_window < 2:
            raise ValueError(
                "Rolling window must be at least 2."
            )

        if self.drift_window < 2:
            raise ValueError(
                "Drift window must be at least 2."
            )

        if self.minimum_observations < 2:
            raise ValueError(
                "Minimum observations must be at least 2."
            )

        if self.performance_threshold_percent < 0:
            raise ValueError(
                "Performance threshold cannot be negative."
            )

        if not 0 < self.drift_threshold <= 1:
            raise ValueError(
                "Drift threshold must be in the interval (0, 1]."
            )

        return self


# =====================================================================
# PART 4.23.2 — NORMALIZE ACTUAL AND FORECAST DATA
# =====================================================================

def di4_monitor_prepare_actuals(
    actual_data: pd.DataFrame,
    timestamp_column: str = "Timestamp",
    demand_column: str = "Sales Count",
) -> pd.DataFrame:
    """Prepare observed demand for comparison with forecasts."""

    required = {
        timestamp_column,
        demand_column,
    }

    missing = required.difference(
        actual_data.columns
    )

    if missing:
        raise ForecastingDataError(
            f"Actual data is missing columns: {sorted(missing)}"
        )

    result = actual_data[
        [timestamp_column, demand_column]
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
        subset=[timestamp_column]
    )

    finite_mask = np.isfinite(
        result[demand_column].to_numpy(dtype=float)
    )

    result = result.loc[
        finite_mask
    ].copy()

    if (result[demand_column] < 0).any():
        raise ForecastingDataError(
            "Observed demand contains negative values."
        )

    result = result.rename(
        columns={
            timestamp_column: "Forecast_Date",
            demand_column: "Actual_Demand",
        }
    )

    result["Forecast_Date"] = (
        result["Forecast_Date"].dt.normalize()
    )

    return result.sort_values(
        "Forecast_Date"
    ).reset_index(drop=True)


def di4_monitor_prepare_forecast(
    forecast_data: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize the forecast frame for monitoring."""

    required = {
        "Forecast_Date",
        "Forecast_Demand",
    }

    missing = required.difference(
        forecast_data.columns
    )

    if missing:
        raise ForecastingDataError(
            f"Forecast is missing columns: {sorted(missing)}"
        )

    result = forecast_data[
        ["Forecast_Date", "Forecast_Demand"]
    ].copy()

    result["Forecast_Date"] = pd.to_datetime(
        result["Forecast_Date"],
        errors="coerce",
    )

    result["Forecast_Demand"] = pd.to_numeric(
        result["Forecast_Demand"],
        errors="coerce",
    )

    result = result.dropna(
        subset=["Forecast_Date", "Forecast_Demand"]
    )

    finite_mask = np.isfinite(
        result["Forecast_Demand"].to_numpy(dtype=float)
    )

    result = result.loc[
        finite_mask
    ].copy()

    if (result["Forecast_Demand"] < 0).any():
        raise ForecastingDataError(
            "Forecast contains negative predictions."
        )

    result["Forecast_Date"] = (
        result["Forecast_Date"].dt.normalize()
    )

    if result["Forecast_Date"].duplicated().any():
        raise ForecastingDataError(
            "Forecast contains duplicate dates."
        )

    return result.sort_values(
        "Forecast_Date"
    ).reset_index(drop=True)


# =====================================================================
# PART 4.23.3 — MATCH ACTUALS WITH FORECASTS
# =====================================================================

def di4_monitor_match_actuals(
    actual_data: pd.DataFrame,
    forecast_data: pd.DataFrame,
    timestamp_column: str = "Timestamp",
    demand_column: str = "Sales Count",
) -> pd.DataFrame:
    """Match actual demand and predictions by forecast date."""

    actuals = di4_monitor_prepare_actuals(
        actual_data,
        timestamp_column=timestamp_column,
        demand_column=demand_column,
    )

    forecasts = di4_monitor_prepare_forecast(
        forecast_data
    )

    # Multiple raw observations can occur on one day.
    # Aggregate them before comparing daily totals.
    actuals = (
        actuals.groupby(
            "Forecast_Date",
            as_index=False,
        )
        .agg(
            Actual_Demand=(
                "Actual_Demand",
                lambda values: values.sum(min_count=1),
            )
        )
    )

    matched = forecasts.merge(
        actuals,
        on="Forecast_Date",
        how="left",
        validate="one_to_one",
    )

    matched["Forecast_Error"] = (
        matched["Actual_Demand"]
        - matched["Forecast_Demand"]
    )

    matched["Absolute_Error"] = (
        matched["Forecast_Error"].abs()
    )

    matched["Squared_Error"] = (
        matched["Forecast_Error"] ** 2
    )

    matched["Absolute_Percentage_Error"] = np.where(
        matched["Actual_Demand"].notna()
        & (matched["Actual_Demand"] != 0),
        matched["Absolute_Error"]
        / matched["Actual_Demand"].abs()
        * 100,
        np.nan,
    )

    return matched


# =====================================================================
# PART 4.23.4 — PERFORMANCE METRICS
# =====================================================================

def di4_monitor_calculate_metrics(
    matched_data: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate performance metrics from matched observations."""

    required = {
        "Actual_Demand",
        "Forecast_Demand",
    }

    missing = required.difference(
        matched_data.columns
    )

    if missing:
        raise ForecastingDataError(
            f"Matched data is missing columns: {sorted(missing)}"
        )

    actual = pd.to_numeric(
        matched_data["Actual_Demand"],
        errors="coerce",
    ).to_numpy(dtype=float)

    predicted = pd.to_numeric(
        matched_data["Forecast_Demand"],
        errors="coerce",
    ).to_numpy(dtype=float)

    valid = (
        np.isfinite(actual)
        & np.isfinite(predicted)
    )

    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        return {
            "observation_count": 0,
            "MAE": None,
            "RMSE": None,
            "MAPE_Percent": None,
            "WAPE_Percent": None,
            "Mean_Error": None,
            "Bias_Direction": "UNAVAILABLE",
        }

    errors = actual - predicted
    absolute_errors = np.abs(errors)

    mae = float(
        np.mean(absolute_errors)
    )

    rmse = float(
        np.sqrt(np.mean(errors ** 2))
    )

    nonzero_actual = actual != 0

    if nonzero_actual.any():
        mape = float(
            np.mean(
                absolute_errors[nonzero_actual]
                / np.abs(actual[nonzero_actual])
            )
            * 100
        )
    else:
        mape = None

    actual_total = float(
        np.sum(np.abs(actual))
    )

    if actual_total > 0:
        wape = float(
            np.sum(absolute_errors)
            / actual_total
            * 100
        )
    else:
        wape = None

    mean_error = float(
        np.mean(errors)
    )

    if mean_error > 0:
        bias_direction = "UNDER_FORECASTING"
    elif mean_error < 0:
        bias_direction = "OVER_FORECASTING"
    else:
        bias_direction = "UNBIASED_ON_AVERAGE"

    return {
        "observation_count": int(len(actual)),
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_Percent": mape,
        "WAPE_Percent": wape,
        "Mean_Error": mean_error,
        "Bias_Direction": bias_direction,
    }


# =====================================================================
# PART 4.23.5 — ROLLING PERFORMANCE
# =====================================================================

def di4_monitor_rolling_performance(
    matched_data: pd.DataFrame,
    window: int = 14,
) -> pd.DataFrame:
    """Calculate rolling MAE and rolling forecast bias."""

    if window < 2:
        raise ValueError(
            "Window must be at least 2."
        )

    required = {
        "Forecast_Date",
        "Actual_Demand",
        "Forecast_Demand",
    }

    missing = required.difference(
        matched_data.columns
    )

    if missing:
        raise ForecastingDataError(
            f"Matched data is missing columns: {sorted(missing)}"
        )

    result = matched_data.copy()

    result["Forecast_Date"] = pd.to_datetime(
        result["Forecast_Date"],
        errors="coerce",
    )

    result = result.sort_values(
        "Forecast_Date"
    )

    result["Forecast_Error"] = (
        result["Actual_Demand"]
        - result["Forecast_Demand"]
    )

    result["Absolute_Error"] = (
        result["Forecast_Error"].abs()
    )

    result["Rolling_MAE"] = (
        result["Absolute_Error"]
        .rolling(
            window=window,
            min_periods=window,
        )
        .mean()
    )

    result["Rolling_Bias"] = (
        result["Forecast_Error"]
        .rolling(
            window=window,
            min_periods=window,
        )
        .mean()
    )

    return result.reset_index(
        drop=True
    )


# =====================================================================
# PART 4.23.6 — DEMAND DISTRIBUTION COMPARISON
# =====================================================================

def di4_monitor_distribution_drift(
    reference_values: pd.Series,
    current_values: pd.Series,
    minimum_observations: int = 7,
    threshold: float = 0.25,
) -> dict[str, Any]:
    """
    Compare two demand distributions using standardized mean change.

    This is a simple monitoring indicator, not a formal hypothesis test.
    """

    if minimum_observations < 2:
        raise ValueError(
            "Minimum observations must be at least 2."
        )

    if not 0 < threshold <= 1:
        raise ValueError(
            "Threshold must be in the interval (0, 1]."
        )

    reference = pd.to_numeric(
        reference_values,
        errors="coerce",
    ).to_numpy(dtype=float)

    current = pd.to_numeric(
        current_values,
        errors="coerce",
    ).to_numpy(dtype=float)

    reference = reference[
        np.isfinite(reference)
    ]

    current = current[
        np.isfinite(current)
    ]

    if (
        len(reference) < minimum_observations
        or len(current) < minimum_observations
    ):
        return {
            "status": "INSUFFICIENT_DATA",
            "reference_count": int(len(reference)),
            "current_count": int(len(current)),
            "standardized_mean_change": None,
            "drift_detected": None,
        }

    reference_mean = float(
        np.mean(reference)
    )

    current_mean = float(
        np.mean(current)
    )

    reference_std = float(
        np.std(reference, ddof=1)
    )

    current_std = float(
        np.std(current, ddof=1)
    )

    pooled_scale = float(
        np.sqrt(
            (
                reference_std ** 2
                + current_std ** 2
            )
            / 2
        )
    )

    if pooled_scale > 0:
        standardized_change = (
            current_mean - reference_mean
        ) / pooled_scale
    elif current_mean == reference_mean:
        standardized_change = 0.0
    else:
        standardized_change = None

    drift_detected = (
        abs(standardized_change) >= threshold
        if standardized_change is not None
        else True
    )

    return {
        "status": "COMPLETED",
        "reference_count": int(len(reference)),
        "current_count": int(len(current)),
        "reference_mean": reference_mean,
        "current_mean": current_mean,
        "reference_standard_deviation": reference_std,
        "current_standard_deviation": current_std,
        "standardized_mean_change": (
            float(standardized_change)
            if standardized_change is not None
            else None
        ),
        "threshold": float(threshold),
        "drift_detected": bool(drift_detected),
        "interpretation": (
            "A change indicator crossed the configured threshold."
            if drift_detected
            else "No threshold-crossing mean change was detected."
        ),
    }


# =====================================================================
# PART 4.23.7 — FORECAST PERFORMANCE ALERTS
# =====================================================================

def di4_monitor_performance_alerts(
    metrics: dict[str, Any],
    reference_mae: Optional[float] = None,
    threshold_percent: float = 25.0,
) -> list[dict[str, Any]]:
    """Create simple alerts from observed forecast performance."""

    if threshold_percent < 0:
        raise ValueError(
            "Threshold percentage cannot be negative."
        )

    alerts: list[dict[str, Any]] = []

    current_mae = di4_safe_float(
        metrics.get("MAE")
    )

    if current_mae is None:
        alerts.append(
            {
                "severity": "INFO",
                "code": "MAE_UNAVAILABLE",
                "message": (
                    "Forecast performance cannot be evaluated "
                    "without valid matched observations."
                ),
            }
        )

    elif reference_mae is not None:
        reference_mae = di4_safe_float(
            reference_mae
        )

        if reference_mae is None or reference_mae < 0:
            raise ValueError(
                "Reference MAE must be a finite non-negative number."
            )

        if reference_mae == 0:
            degradation_percent = (
                0.0
                if current_mae == 0
                else None
            )
        else:
            degradation_percent = (
                (current_mae - reference_mae)
                / reference_mae
                * 100
            )

        if degradation_percent is None:
            alerts.append(
                {
                    "severity": "WARNING",
                    "code": "ZERO_BASELINE_MAE",
                    "message": (
                        "Reference MAE is zero; percentage degradation "
                        "cannot be calculated meaningfully."
                    ),
                }
            )

        elif degradation_percent > threshold_percent:
            alerts.append(
                {
                    "severity": "WARNING",
                    "code": "FORECAST_ERROR_DEGRADATION",
                    "message": (
                        "Current MAE exceeds the reference MAE "
                        "by more than the configured threshold."
                    ),
                    "current_mae": current_mae,
                    "reference_mae": reference_mae,
                    "degradation_percent": degradation_percent,
                }
            )

    bias_direction = metrics.get(
        "Bias_Direction"
    )

    if bias_direction == "UNDER_FORECASTING":
        alerts.append(
            {
                "severity": "INFO",
                "code": "UNDER_FORECASTING",
                "message": (
                    "Actual demand is higher than predicted demand "
                    "on average in the evaluated observations."
                ),
            }
        )

    elif bias_direction == "OVER_FORECASTING":
        alerts.append(
            {
                "severity": "INFO",
                "code": "OVER_FORECASTING",
                "message": (
                    "Actual demand is lower than predicted demand "
                    "on average in the evaluated observations."
                ),
            }
        )

    return alerts


# =====================================================================
# PART 4.23.8 — BUILD MONITORING REPORT
# =====================================================================

def di4_build_monitoring_report(
    actual_data: pd.DataFrame,
    forecast_data: pd.DataFrame,
    config: Optional[DI4MonitoringConfig] = None,
    reference_values: Optional[pd.Series] = None,
    reference_mae: Optional[float] = None,
) -> dict[str, Any]:
    """Create a monitoring report for an existing forecast."""

    if config is None:
        config = DI4MonitoringConfig()

    config.validate()

    matched = di4_monitor_match_actuals(
        actual_data=actual_data,
        forecast_data=forecast_data,
        timestamp_column=config.timestamp_column,
        demand_column=config.demand_column,
    )

    metrics = di4_monitor_calculate_metrics(
        matched
    )

    rolling = di4_monitor_rolling_performance(
        matched,
        window=config.rolling_window,
    )

    alerts = di4_monitor_performance_alerts(
        metrics=metrics,
        reference_mae=reference_mae,
        threshold_percent=(
            config.performance_threshold_percent
        ),
    )

    drift_report = None

    if reference_values is not None:
        observed_values = matched[
            "Actual_Demand"
        ].dropna()

        drift_report = di4_monitor_distribution_drift(
            reference_values=reference_values,
            current_values=observed_values,
            minimum_observations=(
                config.minimum_observations
            ),
            threshold=config.drift_threshold,
        )

    return {
        "monitoring_version": "1.0.0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "demand_measure": config.demand_column,
        "metrics": metrics,
        "alerts": alerts,
        "distribution_drift": drift_report,
        "matched_observations": int(
            matched["Actual_Demand"].notna().sum()
        ),
        "unmatched_forecast_periods": int(
            matched["Actual_Demand"].isna().sum()
        ),
        "rolling_performance": rolling,
        "matched_data": matched,
    }


# =====================================================================
# PART 4.23.9 — EXPORT MONITORING REPORT
# =====================================================================

def di4_export_monitoring_report(
    report: dict[str, Any],
    output_directory: str | Path,
) -> dict[str, str]:
    """Export monitoring metrics, matched data and rolling metrics."""

    output_dir = Path(
        output_directory
    ).expanduser()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    matched = report.get(
        "matched_data"
    )

    rolling = report.get(
        "rolling_performance"
    )

    exports: dict[str, str] = {}

    if isinstance(matched, pd.DataFrame) and not matched.empty:
        path = di4_pipeline_export_csv(
            matched,
            output_dir / "monitoring_matched_data.csv",
        )

        exports["matched_data"] = str(path)

    if isinstance(rolling, pd.DataFrame) and not rolling.empty:
        path = di4_pipeline_export_csv(
            rolling,
            output_dir / "monitoring_rolling_performance.csv",
        )

        exports["rolling_performance"] = str(path)

    summary = {
        key: value
        for key, value in report.items()
        if key not in {
            "matched_data",
            "rolling_performance",
        }
    }

    summary_path = di4_pipeline_export_json(
        summary,
        output_dir / "monitoring_report.json",
    )

    exports["summary"] = str(
        summary_path
    )

    return exports


# =====================================================================
# PART 4.23.10 — MONITORING FUNCTION REGISTRY
# =====================================================================

DI4_MONITORING_FUNCTION_REGISTRY = {
    "prepare_actuals": di4_monitor_prepare_actuals,
    "prepare_forecast": di4_monitor_prepare_forecast,
    "match_actuals": di4_monitor_match_actuals,
    "calculate_metrics": di4_monitor_calculate_metrics,
    "rolling_performance": di4_monitor_rolling_performance,
    "distribution_drift": di4_monitor_distribution_drift,
    "performance_alerts": di4_monitor_performance_alerts,
    "build_report": di4_build_monitoring_report,
    "export_report": di4_export_monitoring_report,
}


# =====================================================================
# PART 4.23 — END
# FORECAST MONITORING, DRIFT DETECTION AND PERFORMANCE TRACKING
# =====================================================================

# =====================================================================
# PART 4.24 — START
# FINAL INTEGRATION AND END-TO-END VALIDATION FRAMEWORK
# =====================================================================

import platform
import sys
import inspect
from time import perf_counter


# =====================================================================
# PART 4.24.1 — INTEGRATION CONFIGURATION
# =====================================================================

@dataclass
class DI4IntegrationConfig:
    """Configuration for an integrated forecasting run."""

    pipeline: DI4PipelineConfig = field(
        default_factory=DI4PipelineConfig
    )

    monitoring: DI4MonitoringConfig = field(
        default_factory=DI4MonitoringConfig
    )

    run_unit_tests: bool = False
    fail_on_unit_test_failure: bool = False
    export_diagnostics: bool = True
    output_directory: Optional[str] = None

    def validate(self) -> "DI4IntegrationConfig":
        self.pipeline.validate()
        self.monitoring.validate()

        if (
            self.pipeline.demand_column
            != self.monitoring.demand_column
        ):
            raise ValueError(
                "Pipeline and monitoring demand columns must match."
            )

        return self


# =====================================================================
# PART 4.24.2 — INTEGRATION RESULT
# =====================================================================

@dataclass
class DI4IntegrationResult:
    """Collects the output of an integrated execution."""

    status: str = "NOT_STARTED"

    pipeline_result: Optional[DI4PipelineResult] = None

    monitoring_report: Optional[dict[str, Any]] = None

    unit_test_report: Optional[dict[str, Any]] = None

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )

    exports: dict[str, str] = field(
        default_factory=dict
    )

    errors: list[str] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    duration_seconds: float = 0.0


# =====================================================================
# PART 4.24.3 — ENVIRONMENT INFORMATION
# =====================================================================

def di4_get_environment_info() -> dict[str, Any]:
    """Return basic runtime information for diagnostics."""

    return {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "engine_version": FORECAST_ENGINE_VERSION,
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }


# =====================================================================
# PART 4.24.4 — STATIC API CONTRACT CHECK
# =====================================================================

def di4_check_function_contracts() -> dict[str, Any]:
    """
    Check that key integration functions exist and are callable.

    This checks API presence, not numerical correctness.
    """

    required_functions = {
        "input_validation": di4_validate_input,
        "prepare_data": di4_prepare_demand_data,
        "aggregate_demand": di4_aggregate_demand,
        "complete_series": di4_complete_time_series,
        "baseline_comparison": di4_compare_baseline_models,
        "forecast_generation": di4_forecast_future_demand,
        "forecast_validation": di4_validate_forecast_output,
        "pipeline": di4_run_forecasting_pipeline,
        "unit_tests": di4_run_unit_tests,
        "monitoring": di4_build_monitoring_report,
        "json_export": di4_pipeline_export_json,
    }

    checks = {}

    for name, function in required_functions.items():
        checks[name] = {
            "callable": callable(function),
            "signature_available": callable(
                getattr(function, "__signature__", None)
            ) or inspect.isfunction(function),
        }

    failed = [
        name
        for name, result in checks.items()
        if not result["callable"]
    ]

    return {
        "status": "PASS" if not failed else "FAIL",
        "function_count": len(required_functions),
        "checks": checks,
        "failed_functions": failed,
    }


# =====================================================================
# PART 4.24.5 — VERIFY DATASET CONTRACT
# =====================================================================

def di4_check_dataset_contract(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Validate the minimum raw-data schema and basic properties."""

    validation = di4_validate_input(
        dataframe,
        required_columns=FORECAST_REQUIRED_COLUMNS,
    )

    if not validation.get("valid", False):
        return {
            "status": "FAIL",
            "validation": di4_json_safe_value(
                validation
            ),
        }

    timestamps = pd.to_datetime(
        dataframe["Timestamp"],
        errors="coerce",
    )

    sales = pd.to_numeric(
        dataframe["Sales Count"],
        errors="coerce",
    )

    redemptions = pd.to_numeric(
        dataframe["Redemption Count"],
        errors="coerce",
    )

    report = {
        "status": "PASS",
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "invalid_timestamps": int(
            timestamps.isna().sum()
        ),
        "missing_sales_values": int(
            sales.isna().sum()
        ),
        "missing_redemption_values": int(
            redemptions.isna().sum()
        ),
        "duplicate_column_names": bool(
            dataframe.columns.duplicated().any()
        ),
    }

    if len(dataframe.columns) != len(
        set(dataframe.columns)
    ):
        report["status"] = "FAIL"

    return report


# =====================================================================
# PART 4.24.6 — FORECAST OUTPUT CONTRACT
# =====================================================================

def di4_check_pipeline_output_contract(
    pipeline_result: DI4PipelineResult,
) -> dict[str, Any]:
    """Check consistency of a completed pipeline result."""

    checks: dict[str, Any] = {
        "pipeline_successful": (
            pipeline_result.status == "SUCCESS"
        ),
        "forecast_available": (
            isinstance(
                pipeline_result.forecast,
                pd.DataFrame,
            )
            and not pipeline_result.forecast.empty
        ),
        "selected_model_available": bool(
            pipeline_result.selected_model
        ),
        "summary_available": bool(
            pipeline_result.forecast_summary
        ),
    }

    forecast_validation = None

    if checks["forecast_available"]:
        try:
            forecast_validation = (
                di4_validate_forecast_output(
                    pipeline_result.forecast
                )
            )

            checks["forecast_structure_valid"] = True

        except Exception as exc:
            checks["forecast_structure_valid"] = False
            checks["forecast_validation_error"] = (
                f"{type(exc).__name__}: {exc}"
            )

    else:
        checks["forecast_structure_valid"] = False

    failures = [
        name
        for name, passed in checks.items()
        if name.endswith("_available")
        or name.endswith("_successful")
        or name.endswith("_valid")
        if passed is False
    ]

    return {
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
        "forecast_validation": forecast_validation,
    }


# =====================================================================
# PART 4.24.7 — SAFE TEST EXECUTION
# =====================================================================

def di4_integration_run_tests() -> dict[str, Any]:
    """Run the engine unit-test suite and summarize failures."""

    started = perf_counter()

    report = di4_run_unit_tests()

    duration = perf_counter() - started

    return {
        "status": (
            "PASS"
            if report.get("all_passed", False)
            else "FAIL"
        ),
        "duration_seconds": round(duration, 4),
        "report": report,
    }


# =====================================================================
# PART 4.24.8 — COMBINE INTEGRATION DIAGNOSTICS
# =====================================================================

def di4_build_integration_diagnostics(
    raw_dataframe: pd.DataFrame,
    pipeline_result: Optional[DI4PipelineResult] = None,
) -> dict[str, Any]:
    """Combine environment, schema and API diagnostics."""

    diagnostics: dict[str, Any] = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "environment": di4_get_environment_info(),
        "api_contract": di4_check_function_contracts(),
        "dataset_contract": di4_check_dataset_contract(
            raw_dataframe
        ),
    }

    if pipeline_result is not None:
        diagnostics["pipeline_contract"] = (
            di4_check_pipeline_output_contract(
                pipeline_result
            )
        )

    return diagnostics


# =====================================================================
# PART 4.24.9 — EXPORT INTEGRATION DIAGNOSTICS
# =====================================================================

def di4_export_integration_diagnostics(
    diagnostics: dict[str, Any],
    output_directory: str | Path,
) -> Path:
    """Export diagnostics as a JSON report."""

    output_dir = Path(
        output_directory
    ).expanduser()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "integration_diagnostics.json"
    )

    return di4_pipeline_export_json(
        diagnostics,
        output_path,
    )


# =====================================================================
# PART 4.24.10 — END-TO-END INTEGRATION RUNNER
# =====================================================================

def di4_run_integrated_workflow(
    raw_dataframe: pd.DataFrame,
    config: Optional[DI4IntegrationConfig] = None,
    actual_data_for_monitoring: Optional[pd.DataFrame] = None,
    reference_values: Optional[pd.Series] = None,
    reference_mae: Optional[float] = None,
) -> DI4IntegrationResult:
    """
    Run forecasting and optional validation/monitoring.

    Monitoring runs only when actual observations are supplied.
    Unit tests are optional and use synthetic data.
    """

    started = perf_counter()

    if config is None:
        config = DI4IntegrationConfig()

    result = DI4IntegrationResult(
        status="RUNNING"
    )

    try:
        config.validate()

        result.diagnostics["environment"] = (
            di4_get_environment_info()
        )

        result.diagnostics["dataset_contract"] = (
            di4_check_dataset_contract(
                raw_dataframe
            )
        )

        if (
            result.diagnostics["dataset_contract"]["status"]
            != "PASS"
        ):
            raise ForecastingDataError(
                "Dataset contract validation failed."
            )

        if config.run_unit_tests:
            result.unit_test_report = (
                di4_integration_run_tests()
            )

            if (
                config.fail_on_unit_test_failure
                and result.unit_test_report["status"] != "PASS"
            ):
                raise ForecastingDataError(
                    "Unit tests failed; integration stopped."
                )

        result.pipeline_result = (
            di4_run_forecasting_pipeline(
                raw_dataframe=raw_dataframe,
                config=config.pipeline,
            )
        )

        if result.pipeline_result.status != "SUCCESS":
            raise ForecastingDataError(
                "Forecast pipeline failed: "
                + "; ".join(
                    result.pipeline_result.errors
                )
            )

        result.diagnostics["pipeline_contract"] = (
            di4_check_pipeline_output_contract(
                result.pipeline_result
            )
        )

        if (
            result.diagnostics["pipeline_contract"]["status"]
            != "PASS"
        ):
            raise ForecastingDataError(
                "Forecast output contract validation failed."
            )

        if actual_data_for_monitoring is not None:
            result.monitoring_report = (
                di4_build_monitoring_report(
                    actual_data=actual_data_for_monitoring,
                    forecast_data=(
                        result.pipeline_result.forecast
                    ),
                    config=config.monitoring,
                    reference_values=reference_values,
                    reference_mae=reference_mae,
                )
            )

        result.status = "SUCCESS"

    except Exception as exc:
        result.status = "FAILED"

        result.errors.append(
            f"{type(exc).__name__}: {exc}"
        )

        DI4_LOGGER.exception(
            "Integrated workflow failed."
        )

    finally:
        result.duration_seconds = round(
            perf_counter() - started,
            6,
        )

    if (
        result.status == "SUCCESS"
        and config.export_diagnostics
        and config.output_directory
    ):
        try:
            result.exports["diagnostics"] = str(
                di4_export_integration_diagnostics(
                    result.diagnostics,
                    config.output_directory,
                )
            )

            if result.monitoring_report is not None:
                monitoring_exports = (
                    di4_export_monitoring_report(
                        result.monitoring_report,
                        Path(config.output_directory)
                        / "monitoring",
                    )
                )

                result.exports.update(
                    {
                        f"monitoring_{key}": value
                        for key, value in monitoring_exports.items()
                    }
                )

        except Exception as exc:
            result.warnings.append(
                "Workflow succeeded, but diagnostic export failed: "
                f"{type(exc).__name__}: {exc}"
            )

    return result


# =====================================================================
# PART 4.24.11 — INTEGRATION FUNCTION REGISTRY
# =====================================================================

DI4_INTEGRATION_FUNCTION_REGISTRY = {
    "environment_info": di4_get_environment_info,
    "check_function_contracts": di4_check_function_contracts,
    "check_dataset_contract": di4_check_dataset_contract,
    "check_pipeline_output_contract": (
        di4_check_pipeline_output_contract
    ),
    "run_tests": di4_integration_run_tests,
    "build_diagnostics": di4_build_integration_diagnostics,
    "export_diagnostics": di4_export_integration_diagnostics,
    "run_integrated_workflow": di4_run_integrated_workflow,
}


# =====================================================================
# PART 4.24 — END
# FINAL INTEGRATION AND END-TO-END VALIDATION FRAMEWORK
# =====================================================================

# =====================================================================
# PART 4.25 — START
# FORECAST ERROR ANALYSIS AND MODEL DIAGNOSTICS
# =====================================================================

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd


# =====================================================================
# PART 4.25.1 — ERROR ANALYSIS CONFIGURATION
# =====================================================================

@dataclass
class DI4ErrorAnalysisConfig:
    """Configuration for forecast error analysis."""

    actual_column: str = "Actual_Demand"
    predicted_column: str = "Predicted_Demand"

    minimum_observations: int = 3

    large_error_quantile: float = 0.95

    relative_error_threshold_percent: float = 25.0

    zero_actual_policy: str = "exclude"

    def validate(self) -> "DI4ErrorAnalysisConfig":
        if self.minimum_observations < 1:
            raise ValueError(
                "minimum_observations must be positive."
            )

        if not 0 < self.large_error_quantile < 1:
            raise ValueError(
                "large_error_quantile must be between 0 and 1."
            )

        if self.relative_error_threshold_percent < 0:
            raise ValueError(
                "relative_error_threshold_percent cannot be negative."
            )

        if self.zero_actual_policy not in {
            "exclude",
            "zero_only",
        }:
            raise ValueError(
                "zero_actual_policy must be 'exclude' or 'zero_only'."
            )

        return self


# =====================================================================
# PART 4.25.2 — PREPARE ACTUAL AND PREDICTED VALUES
# =====================================================================

def di4_prepare_error_analysis_data(
    dataframe: pd.DataFrame,
    actual_column: str = "Actual_Demand",
    predicted_column: str = "Predicted_Demand",
) -> pd.DataFrame:
    """
    Prepare aligned actual and predicted values for error analysis.

    Negative demand values are not accepted.
    Rows containing missing or infinite values are excluded.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise ForecastingDataError(
            "Error analysis input must be a pandas DataFrame."
        )

    required_columns = [
        actual_column,
        predicted_column,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ForecastingDataError(
            "Missing error-analysis columns: "
            + ", ".join(missing_columns)
        )

    result = dataframe[
        required_columns
    ].copy()

    result[actual_column] = pd.to_numeric(
        result[actual_column],
        errors="coerce",
    )

    result[predicted_column] = pd.to_numeric(
        result[predicted_column],
        errors="coerce",
    )

    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    result = result.dropna(
        subset=required_columns
    ).reset_index(drop=True)

    if result.empty:
        raise ForecastingDataError(
            "No valid actual-predicted pairs remain."
        )

    if (result[actual_column] < 0).any():
        raise ForecastingDataError(
            "Actual demand cannot be negative."
        )

    if (result[predicted_column] < 0).any():
        raise ForecastingDataError(
            "Predicted demand cannot be negative."
        )

    result["Forecast_Error"] = (
        result[predicted_column]
        - result[actual_column]
    )

    result["Absolute_Error"] = (
        result["Forecast_Error"].abs()
    )

    result["Squared_Error"] = (
        result["Forecast_Error"] ** 2
    )

    return result


# =====================================================================
# PART 4.25.3 — CLASSIFY FORECAST DIRECTION
# =====================================================================

def di4_classify_forecast_direction(
    forecast_error: float,
    tolerance: float = 1e-9,
) -> str:
    """
    Positive error means over-forecasting.
    Negative error means under-forecasting.
    """

    error = float(forecast_error)

    if not np.isfinite(error):
        return "INVALID"

    if error > tolerance:
        return "OVER_FORECAST"

    if error < -tolerance:
        return "UNDER_FORECAST"

    return "ACCURATE"


# =====================================================================
# PART 4.25.4 — CALCULATE ERROR METRICS
# =====================================================================

def di4_calculate_error_metrics(
    dataframe: pd.DataFrame,
    actual_column: str = "Actual_Demand",
    predicted_column: str = "Predicted_Demand",
) -> dict[str, Any]:
    """Calculate standard forecast error metrics."""

    prepared = di4_prepare_error_analysis_data(
        dataframe=dataframe,
        actual_column=actual_column,
        predicted_column=predicted_column,
    )

    actual = prepared[
        actual_column
    ].to_numpy(dtype=float)

    predicted = prepared[
        predicted_column
    ].to_numpy(dtype=float)

    error = predicted - actual

    absolute_error = np.abs(error)

    squared_error = error ** 2

    nonzero_mask = actual != 0

    if nonzero_mask.any():
        mape = float(
            np.mean(
                absolute_error[nonzero_mask]
                / np.abs(actual[nonzero_mask])
            )
            * 100
        )
    else:
        mape = float("nan")

    total_actual = float(
        np.sum(np.abs(actual))
    )

    if total_actual > 0:
        wape = float(
            np.sum(absolute_error)
            / total_actual
            * 100
        )
    else:
        wape = float("nan")

    mean_error = float(
        np.mean(error)
    )

    over_count = int(
        np.sum(error > 1e-9)
    )

    under_count = int(
        np.sum(error < -1e-9)
    )

    accurate_count = int(
        len(error) - over_count - under_count
    )

    return {
        "Observation_Count": int(len(actual)),
        "MAE": float(np.mean(absolute_error)),
        "RMSE": float(np.sqrt(np.mean(squared_error))),
        "Mean_Error": mean_error,
        "MAPE_Percent": mape,
        "WAPE_Percent": wape,
        "Over_Forecast_Count": over_count,
        "Under_Forecast_Count": under_count,
        "Accurate_Count": accurate_count,
        "Over_Forecast_Percent": (
            over_count / len(error) * 100
        ),
        "Under_Forecast_Percent": (
            under_count / len(error) * 100
        ),
    }


# =====================================================================
# PART 4.25.5 — BUILD ROW-LEVEL ERROR DIAGNOSTICS
# =====================================================================

def di4_build_error_diagnostics(
    dataframe: pd.DataFrame,
    actual_column: str = "Actual_Demand",
    predicted_column: str = "Predicted_Demand",
    relative_error_threshold_percent: float = 25.0,
) -> pd.DataFrame:
    """Create a row-level diagnostic table."""

    if relative_error_threshold_percent < 0:
        raise ValueError(
            "Relative error threshold cannot be negative."
        )

    result = di4_prepare_error_analysis_data(
        dataframe=dataframe,
        actual_column=actual_column,
        predicted_column=predicted_column,
    )

    actual = result[
        actual_column
    ].to_numpy(dtype=float)

    error = result[
        "Forecast_Error"
    ].to_numpy(dtype=float)

    absolute_error = np.abs(error)

    relative_error = np.full(
        len(result),
        np.nan,
        dtype=float,
    )

    nonzero_actual = actual != 0

    relative_error[nonzero_actual] = (
        absolute_error[nonzero_actual]
        / np.abs(actual[nonzero_actual])
        * 100
    )

    result["Absolute_Percentage_Error"] = (
        relative_error
    )

    result["Forecast_Direction"] = [
        di4_classify_forecast_direction(value)
        for value in error
    ]

    result["Large_Relative_Error"] = (
        result["Absolute_Percentage_Error"]
        > relative_error_threshold_percent
    )

    result["Zero_Actual_Demand"] = (
        result[actual_column] == 0
    )

    return result


# =====================================================================
# PART 4.25.6 — ERROR DISTRIBUTION SUMMARY
# =====================================================================

def di4_error_distribution_summary(
    dataframe: pd.DataFrame,
    actual_column: str = "Actual_Demand",
    predicted_column: str = "Predicted_Demand",
) -> dict[str, Any]:
    """Summarize the distribution of forecast errors."""

    prepared = di4_prepare_error_analysis_data(
        dataframe=dataframe,
        actual_column=actual_column,
        predicted_column=predicted_column,
    )

    error = prepared[
        "Forecast_Error"
    ].to_numpy(dtype=float)

    absolute_error = np.abs(error)

    return {
        "Observation_Count": int(len(error)),
        "Error_Minimum": float(np.min(error)),
        "Error_Maximum": float(np.max(error)),
        "Error_Mean": float(np.mean(error)),
        "Error_Median": float(np.median(error)),
        "Error_Standard_Deviation": float(
            np.std(error, ddof=1)
        ) if len(error) > 1 else 0.0,
        "Absolute_Error_P50": float(
            np.quantile(absolute_error, 0.50)
        ),
        "Absolute_Error_P75": float(
            np.quantile(absolute_error, 0.75)
        ),
        "Absolute_Error_P90": float(
            np.quantile(absolute_error, 0.90)
        ),
        "Absolute_Error_P95": float(
            np.quantile(absolute_error, 0.95)
        ),
        "Absolute_Error_Maximum": float(
            np.max(absolute_error)
        ),
    }


# =====================================================================
# PART 4.25.7 — IDENTIFY HIGH-ERROR OBSERVATIONS
# =====================================================================

def di4_find_high_error_observations(
    dataframe: pd.DataFrame,
    actual_column: str = "Actual_Demand",
    predicted_column: str = "Predicted_Demand",
    error_quantile: float = 0.95,
) -> pd.DataFrame:
    """Identify observations with unusually large absolute errors."""

    if not 0 < error_quantile < 1:
        raise ValueError(
            "error_quantile must be between 0 and 1."
        )

    diagnostics = di4_build_error_diagnostics(
        dataframe=dataframe,
        actual_column=actual_column,
        predicted_column=predicted_column,
    )

    error_threshold = float(
        diagnostics["Absolute_Error"].quantile(
            error_quantile
        )
    )

    high_error = diagnostics.loc[
        diagnostics["Absolute_Error"] >= error_threshold
    ].copy()

    high_error["Error_Threshold"] = error_threshold

    high_error["Error_Quantile"] = error_quantile

    return high_error.reset_index(drop=True)


# =====================================================================
# PART 4.25.8 — ERROR ANALYSIS REPORT
# =====================================================================

def di4_build_error_analysis_report(
    dataframe: pd.DataFrame,
    actual_column: str = "Actual_Demand",
    predicted_column: str = "Predicted_Demand",
    error_quantile: float = 0.95,
    relative_error_threshold_percent: float = 25.0,
) -> dict[str, Any]:
    """Build a complete forecast error analysis report."""

    diagnostics = di4_build_error_diagnostics(
        dataframe=dataframe,
        actual_column=actual_column,
        predicted_column=predicted_column,
        relative_error_threshold_percent=(
            relative_error_threshold_percent
        ),
    )

    metrics = di4_calculate_error_metrics(
        diagnostics,
        actual_column=actual_column,
        predicted_column=predicted_column,
    )

    distribution = di4_error_distribution_summary(
        diagnostics,
        actual_column=actual_column,
        predicted_column=predicted_column,
    )

    high_errors = di4_find_high_error_observations(
        diagnostics,
        actual_column=actual_column,
        predicted_column=predicted_column,
        error_quantile=error_quantile,
    )

    return {
        "report_version": "1.0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "metrics": metrics,
        "error_distribution": distribution,
        "high_error_threshold_quantile": error_quantile,
        "high_error_count": int(len(high_errors)),
        "high_error_observations": high_errors,
        "row_level_diagnostics": diagnostics,
    }


# =====================================================================
# PART 4.25.9 — EXPORT ERROR ANALYSIS REPORT
# =====================================================================

def di4_export_error_analysis_report(
    report: dict[str, Any],
    output_directory: str | Path,
) -> dict[str, str]:
    """Export error metrics, row-level diagnostics and high errors."""

    output_dir = Path(
        output_directory
    ).expanduser()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    exports: dict[str, str] = {}

    diagnostics = report.get(
        "row_level_diagnostics"
    )

    high_errors = report.get(
        "high_error_observations"
    )

    if isinstance(diagnostics, pd.DataFrame):
        diagnostics_path = (
            output_dir
            / "forecast_error_diagnostics.csv"
        )

        diagnostics.to_csv(
            diagnostics_path,
            index=False,
        )

        exports["diagnostics_csv"] = str(
            diagnostics_path
        )

    if isinstance(high_errors, pd.DataFrame):
        high_errors_path = (
            output_dir
            / "high_forecast_errors.csv"
        )

        high_errors.to_csv(
            high_errors_path,
            index=False,
        )

        exports["high_errors_csv"] = str(
            high_errors_path
        )

    json_payload = {
        key: value
        for key, value in report.items()
        if not isinstance(value, pd.DataFrame)
    }

    json_path = (
        output_dir
        / "forecast_error_summary.json"
    )

    di4_pipeline_export_json(
        json_payload,
        json_path,
    )

    exports["summary_json"] = str(
        json_path
    )

    return exports


# =====================================================================
# PART 4.25.10 — ERROR ANALYSIS FUNCTION REGISTRY
# =====================================================================

DI4_ERROR_ANALYSIS_FUNCTION_REGISTRY = {
    "prepare_data": di4_prepare_error_analysis_data,
    "classify_direction": di4_classify_forecast_direction,
    "calculate_metrics": di4_calculate_error_metrics,
    "build_diagnostics": di4_build_error_diagnostics,
    "distribution_summary": di4_error_distribution_summary,
    "find_high_errors": di4_find_high_error_observations,
    "build_report": di4_build_error_analysis_report,
    "export_report": di4_export_error_analysis_report,
}


# =====================================================================
# PART 4.25 — END
# FORECAST ERROR ANALYSIS AND MODEL DIAGNOSTICS
# =====================================================================

# =====================================================================
# PART 4.26 — START
# FORECAST EXPLAINABILITY AND BUSINESS INSIGHTS
# =====================================================================

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd


# =====================================================================
# PART 4.26.1 — EXPLAINABILITY CONFIGURATION
# =====================================================================

@dataclass
class DI4ExplainabilityConfig:
    """Configuration for interpretable demand forecasting insights."""

    date_column: str = "Forecast_Date"
    demand_column: str = "Actual_Demand"

    forecast_column: str = "Forecast_Demand"
    lower_bound_column: str = "Lower_Bound"
    upper_bound_column: str = "Upper_Bound"

    recent_window: int = 7
    comparison_window: int = 7

    significant_change_percent: float = 10.0

    minimum_observations: int = 3

    def validate(self) -> "DI4ExplainabilityConfig":
        if self.recent_window < 1:
            raise ValueError(
                "recent_window must be positive."
            )

        if self.comparison_window < 1:
            raise ValueError(
                "comparison_window must be positive."
            )

        if self.minimum_observations < 2:
            raise ValueError(
                "minimum_observations must be at least 2."
            )

        if self.significant_change_percent < 0:
            raise ValueError(
                "significant_change_percent cannot be negative."
            )

        return self


# =====================================================================
# PART 4.26.2 — PREPARE HISTORICAL DEMAND
# =====================================================================

def di4_explain_prepare_history(
    historical_data: pd.DataFrame,
    date_column: str = "Forecast_Date",
    demand_column: str = "Actual_Demand",
) -> pd.DataFrame:
    """Prepare historical demand for explainability analysis."""

    if not isinstance(
        historical_data,
        pd.DataFrame,
    ):
        raise ForecastingDataError(
            "Historical data must be a DataFrame."
        )

    required_columns = [
        date_column,
        demand_column,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in historical_data.columns
    ]

    if missing_columns:
        raise ForecastingDataError(
            "Missing historical columns: "
            + ", ".join(missing_columns)
        )

    result = historical_data[
        required_columns
    ].copy()

    result[date_column] = pd.to_datetime(
        result[date_column],
        errors="coerce",
        utc=True,
    ).dt.tz_convert(None)

    result[demand_column] = pd.to_numeric(
        result[demand_column],
        errors="coerce",
    )

    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    result = result.dropna(
        subset=required_columns
    )

    if (result[demand_column] < 0).any():
        raise ForecastingDataError(
            "Historical demand cannot be negative."
        )

    if result[date_column].duplicated().any():
        raise ForecastingDataError(
            "Historical dates must be unique. "
            "Aggregate duplicate dates before analysis."
        )

    result = result.sort_values(
        date_column
    ).reset_index(drop=True)

    if result.empty:
        raise ForecastingDataError(
            "No valid historical observations remain."
        )

    return result


# =====================================================================
# PART 4.26.3 — PREPARE FORECAST VALUES
# =====================================================================

def di4_explain_prepare_forecast(
    forecast_data: pd.DataFrame,
    date_column: str = "Forecast_Date",
    forecast_column: str = "Forecast_Demand",
    lower_bound_column: str = "Lower_Bound",
    upper_bound_column: str = "Upper_Bound",
) -> pd.DataFrame:
    """Validate and normalize future forecast records."""

    if not isinstance(
        forecast_data,
        pd.DataFrame,
    ):
        raise ForecastingDataError(
            "Forecast data must be a DataFrame."
        )

    required_columns = [
        date_column,
        forecast_column,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in forecast_data.columns
    ]

    if missing_columns:
        raise ForecastingDataError(
            "Missing forecast columns: "
            + ", ".join(missing_columns)
        )

    result = forecast_data.copy()

    result[date_column] = pd.to_datetime(
        result[date_column],
        errors="coerce",
        utc=True,
    ).dt.tz_convert(None)

    result[forecast_column] = pd.to_numeric(
        result[forecast_column],
        errors="coerce",
    )

    numeric_columns = [
        forecast_column,
        lower_bound_column,
        upper_bound_column,
    ]

    for column in numeric_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    result = result.dropna(
        subset=required_columns
    )

    if result.empty:
        raise ForecastingDataError(
            "No valid forecast observations remain."
        )

    if (result[forecast_column] < 0).any():
        raise ForecastingDataError(
            "Forecast demand cannot be negative."
        )

    if result[date_column].duplicated().any():
        raise ForecastingDataError(
            "Forecast dates must be unique."
        )

    if (
        lower_bound_column in result.columns
        and upper_bound_column in result.columns
    ):
        bounds_available = result[
            [
                lower_bound_column,
                upper_bound_column,
            ]
        ].notna().all(axis=1)

        invalid_bounds = (
            bounds_available
            & (
                (
                    result[lower_bound_column]
                    > result[forecast_column]
                )
                | (
                    result[forecast_column]
                    > result[upper_bound_column]
                )
            )
        )

        if invalid_bounds.any():
            raise ForecastingDataError(
                "Forecast values must lie within valid bounds."
            )

    return result.sort_values(
        date_column
    ).reset_index(drop=True)


# =====================================================================
# PART 4.26.4 — HISTORICAL TREND ANALYSIS
# =====================================================================

def di4_explain_historical_trend(
    historical_data: pd.DataFrame,
    demand_column: str = "Actual_Demand",
    recent_window: int = 7,
    comparison_window: int = 7,
) -> dict[str, Any]:
    """Compare recent demand with the preceding time window."""

    if recent_window < 1 or comparison_window < 1:
        raise ValueError(
            "Window sizes must be positive."
        )

    if demand_column not in historical_data.columns:
        raise ForecastingDataError(
            f"Missing demand column: {demand_column}"
        )

    demand = pd.to_numeric(
        historical_data[demand_column],
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(demand) < recent_window + comparison_window:
        raise ForecastingDataError(
            "Insufficient observations for trend comparison."
        )

    recent_values = demand.iloc[
        -recent_window:
    ]

    comparison_values = demand.iloc[
        -(recent_window + comparison_window):-recent_window
    ]

    recent_mean = float(
        recent_values.mean()
    )

    comparison_mean = float(
        comparison_values.mean()
    )

    if comparison_mean == 0:
        change_percent = (
            0.0
            if recent_mean == 0
            else None
        )
    else:
        change_percent = float(
            (
                recent_mean - comparison_mean
            )
            / abs(comparison_mean)
            * 100
        )

    return {
        "recent_window_observations": int(
            len(recent_values)
        ),
        "comparison_window_observations": int(
            len(comparison_values)
        ),
        "recent_average_demand": recent_mean,
        "previous_average_demand": comparison_mean,
        "change_percent": change_percent,
        "change_direction": (
            "INCREASE"
            if recent_mean > comparison_mean
            else "DECREASE"
            if recent_mean < comparison_mean
            else "STABLE"
        ),
    }


# =====================================================================
# PART 4.26.5 — FORECAST UNCERTAINTY ANALYSIS
# =====================================================================

def di4_explain_forecast_uncertainty(
    forecast_data: pd.DataFrame,
    forecast_column: str = "Forecast_Demand",
    lower_bound_column: str = "Lower_Bound",
    upper_bound_column: str = "Upper_Bound",
) -> dict[str, Any]:
    """
    Summarize prediction interval widths.

    These bounds describe model uncertainty, not guaranteed limits.
    """

    required_columns = [
        forecast_column,
        lower_bound_column,
        upper_bound_column,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in forecast_data.columns
    ]

    if missing_columns:
        raise ForecastingDataError(
            "Uncertainty analysis requires: "
            + ", ".join(missing_columns)
        )

    frame = forecast_data[
        required_columns
    ].copy()

    for column in required_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if frame.empty:
        raise ForecastingDataError(
            "No valid prediction intervals are available."
        )

    if (
        frame[lower_bound_column]
        > frame[upper_bound_column]
    ).any():
        raise ForecastingDataError(
            "Lower bounds cannot exceed upper bounds."
        )

    interval_width = (
        frame[upper_bound_column]
        - frame[lower_bound_column]
    )

    forecast_values = frame[
        forecast_column
    ].to_numpy(dtype=float)

    widths = interval_width.to_numpy(
        dtype=float
    )

    relative_width = np.full(
        len(frame),
        np.nan,
        dtype=float,
    )

    positive_forecast = forecast_values > 0

    relative_width[positive_forecast] = (
        widths[positive_forecast]
        / forecast_values[positive_forecast]
        * 100
    )

    return {
        "forecast_observation_count": int(
            len(frame)
        ),
        "mean_interval_width": float(
            np.mean(widths)
        ),
        "median_interval_width": float(
            np.median(widths)
        ),
        "maximum_interval_width": float(
            np.max(widths)
        ),
        "mean_relative_interval_width_percent": (
            float(
                np.nanmean(relative_width)
            )
            if np.isfinite(relative_width).any()
            else None
        ),
        "interpretation": (
            "Wider intervals indicate greater forecast uncertainty."
        ),
    }


# =====================================================================
# PART 4.26.6 — FORECAST HORIZON COMPARISON
# =====================================================================

def di4_explain_horizon_profile(
    forecast_data: pd.DataFrame,
    forecast_column: str = "Forecast_Demand",
    date_column: str = "Forecast_Date",
    lower_bound_column: str = "Lower_Bound",
    upper_bound_column: str = "Upper_Bound",
) -> pd.DataFrame:
    """Show forecast levels and uncertainty by forecast date."""

    forecast = di4_explain_prepare_forecast(
        forecast_data=forecast_data,
        date_column=date_column,
        forecast_column=forecast_column,
        lower_bound_column=lower_bound_column,
        upper_bound_column=upper_bound_column,
    )

    result = forecast[
        [
            date_column,
            forecast_column,
        ]
    ].copy()

    result["Forecast_Step"] = (
        np.arange(len(result)) + 1
    )

    if lower_bound_column in forecast.columns:
        result[lower_bound_column] = (
            forecast[lower_bound_column].to_numpy()
        )

    if upper_bound_column in forecast.columns:
        result[upper_bound_column] = (
            forecast[upper_bound_column].to_numpy()
        )

    if (
        lower_bound_column in result.columns
        and upper_bound_column in result.columns
    ):
        result["Interval_Width"] = (
            result[upper_bound_column]
            - result[lower_bound_column]
        )

    return result


# =====================================================================
# PART 4.26.7 — BUSINESS INSIGHT GENERATOR
# =====================================================================

def di4_generate_business_insights(
    historical_data: pd.DataFrame,
    forecast_data: pd.DataFrame,
    config: Optional[DI4ExplainabilityConfig] = None,
) -> dict[str, Any]:
    """
    Generate transparent, data-driven demand insights.

    Insights are descriptive, not causal explanations.
    """

    if config is None:
        config = DI4ExplainabilityConfig()

    config.validate()

    history = di4_explain_prepare_history(
        historical_data=historical_data,
        date_column=config.date_column,
        demand_column=config.demand_column,
    )

    forecast = di4_explain_prepare_forecast(
        forecast_data=forecast_data,
        date_column=config.date_column,
        forecast_column=config.forecast_column,
        lower_bound_column=config.lower_bound_column,
        upper_bound_column=config.upper_bound_column,
    )

    if len(history) < config.minimum_observations:
        raise ForecastingDataError(
            "Not enough historical observations for insights."
        )

    insights: list[dict[str, Any]] = []

    trend = di4_explain_historical_trend(
        historical_data=history,
        demand_column=config.demand_column,
        recent_window=config.recent_window,
        comparison_window=config.comparison_window,
    ) if len(history) >= (
        config.recent_window
        + config.comparison_window
    ) else None

    historical_mean = float(
        history[config.demand_column].mean()
    )

    historical_median = float(
        history[config.demand_column].median()
    )

    forecast_mean = float(
        forecast[config.forecast_column].mean()
    )

    forecast_total = float(
        forecast[config.forecast_column].sum()
    )

    if historical_mean > 0:
        expected_change_percent = float(
            (
                forecast_mean - historical_mean
            )
            / historical_mean
            * 100
        )
    elif forecast_mean == 0:
        expected_change_percent = 0.0
    else:
        expected_change_percent = None

    if expected_change_percent is None:
        insights.append({
            "category": "BASELINE",
            "severity": "INFO",
            "title": "Historical baseline is zero",
            "message": (
                "Percentage comparison with historical average "
                "is undefined because historical demand is zero."
            ),
        })

    elif expected_change_percent >= (
        config.significant_change_percent
    ):
        insights.append({
            "category": "DEMAND_TREND",
            "severity": "ATTENTION",
            "title": "Forecast demand is higher",
            "message": (
                "The forecast average is "
                f"{expected_change_percent:.1f}% above "
                "the historical average."
            ),
        })

    elif expected_change_percent <= (
        -config.significant_change_percent
    ):
        insights.append({
            "category": "DEMAND_TREND",
            "severity": "ATTENTION",
            "title": "Forecast demand is lower",
            "message": (
                "The forecast average is "
                f"{abs(expected_change_percent):.1f}% below "
                "the historical average."
            ),
        })

    else:
        insights.append({
            "category": "DEMAND_TREND",
            "severity": "INFO",
            "title": "Forecast is near the historical baseline",
            "message": (
                "The forecast average is within the configured "
                "change threshold of the historical average."
            ),
        })

    if trend is not None:
        insights.append({
            "category": "HISTORICAL_TREND",
            "severity": "INFO",
            "title": "Recent historical demand comparison",
            "message": (
                f"Recent average demand is "
                f"{trend['recent_average_demand']:.2f}; "
                f"the previous-window average is "
                f"{trend['previous_average_demand']:.2f}."
            ),
        })

    if (
        config.lower_bound_column in forecast.columns
        and config.upper_bound_column in forecast.columns
    ):
        uncertainty = di4_explain_forecast_uncertainty(
            forecast_data=forecast,
            forecast_column=config.forecast_column,
            lower_bound_column=config.lower_bound_column,
            upper_bound_column=config.upper_bound_column,
        )

        insights.append({
            "category": "UNCERTAINTY",
            "severity": "INFO",
            "title": "Prediction uncertainty summarized",
            "message": (
                "Mean prediction interval width is "
                f"{uncertainty['mean_interval_width']:.2f}. "
                "Intervals are estimates, not guaranteed limits."
            ),
        })

    return {
        "report_version": "1.0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "historical_observation_count": int(
            len(history)
        ),
        "forecast_observation_count": int(
            len(forecast)
        ),
        "historical_average_demand": historical_mean,
        "historical_median_demand": historical_median,
        "forecast_average_demand": forecast_mean,
        "forecast_total_demand": forecast_total,
        "forecast_change_percent": expected_change_percent,
        "historical_trend": trend,
        "insights": insights,
        "disclaimer": (
            "Insights describe observed data and model outputs. "
            "They do not establish causal relationships or "
            "guarantee future demand."
        ),
    }


# =====================================================================
# PART 4.26.8 — EXPORT BUSINESS INSIGHTS
# =====================================================================

def di4_export_business_insights(
    insights_report: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Export business insights to a JSON file."""

    destination = Path(
        output_path
    ).expanduser()

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return di4_pipeline_export_json(
        insights_report,
        destination,
    )


# =====================================================================
# PART 4.26.9 — EXPLAINABILITY FUNCTION REGISTRY
# =====================================================================

DI4_EXPLAINABILITY_FUNCTION_REGISTRY = {
    "prepare_history": di4_explain_prepare_history,
    "prepare_forecast": di4_explain_prepare_forecast,
    "historical_trend": di4_explain_historical_trend,
    "forecast_uncertainty": di4_explain_forecast_uncertainty,
    "horizon_profile": di4_explain_horizon_profile,
    "business_insights": di4_generate_business_insights,
    "export_business_insights": di4_export_business_insights,
}


# =====================================================================
# PART 4.26 — END
# FORECAST EXPLAINABILITY AND BUSINESS INSIGHTS
# =====================================================================


# =====================================================================
# PART 4.27 — START
# FORECAST SCENARIO ANALYSIS AND WHAT-IF PLANNING
# =====================================================================


# =====================================================================
# PART 4.27.1 — SCENARIO CONFIGURATION
# =====================================================================

@dataclass
class DI4ScenarioConfig:
    """Configuration for demand what-if scenario analysis."""

    forecast_column: str = "Forecast_Demand"
    lower_bound_column: str = "Lower_Bound"
    upper_bound_column: str = "Upper_Bound"
    date_column: str = "Forecast_Date"

    demand_change_scenarios: dict[str, float] = field(
        default_factory=lambda: {
            "Downside": -0.15,
            "Baseline": 0.00,
            "Upside": 0.15,
        }
    )

    planning_buffer_percent: float = 10.0

    maximum_scenario_change_percent: float = 100.0

    def validate(self) -> "DI4ScenarioConfig":
        if not self.demand_change_scenarios:
            raise ValueError(
                "At least one demand scenario is required."
            )

        if "Baseline" not in self.demand_change_scenarios:
            raise ValueError(
                "A Baseline scenario is required."
            )

        if self.demand_change_scenarios["Baseline"] != 0:
            raise ValueError(
                "Baseline scenario change must be zero."
            )

        if self.planning_buffer_percent < 0:
            raise ValueError(
                "Planning buffer cannot be negative."
            )

        if self.maximum_scenario_change_percent < 0:
            raise ValueError(
                "Maximum scenario change cannot be negative."
            )

        maximum_change = (
            self.maximum_scenario_change_percent / 100
        )

        for name, change in self.demand_change_scenarios.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    "Scenario names must be non-empty strings."
                )

            if not isinstance(change, (int, float)):
                raise ValueError(
                    f"Scenario '{name}' must have a numeric change."
                )

            if not np.isfinite(change):
                raise ValueError(
                    f"Scenario '{name}' contains a non-finite change."
                )

            if abs(change) > maximum_change:
                raise ValueError(
                    f"Scenario '{name}' exceeds the permitted change."
                )

        return self


# =====================================================================
# PART 4.27.2 — VALIDATE FORECAST FOR SCENARIO ANALYSIS
# =====================================================================

def di4_scenario_prepare_forecast(
    forecast_data: pd.DataFrame,
    forecast_column: str = "Forecast_Demand",
    date_column: str = "Forecast_Date",
) -> pd.DataFrame:
    """Prepare forecast data for scenario calculations."""

    if not isinstance(forecast_data, pd.DataFrame):
        raise ForecastingDataError(
            "Scenario input must be a pandas DataFrame."
        )

    required_columns = [
        date_column,
        forecast_column,
    ]

    missing = [
        column
        for column in required_columns
        if column not in forecast_data.columns
    ]

    if missing:
        raise ForecastingDataError(
            "Missing scenario columns: "
            + ", ".join(missing)
        )

    result = forecast_data.copy()

    result[date_column] = pd.to_datetime(
        result[date_column],
        errors="coerce",
        utc=True,
    ).dt.tz_convert(None)

    result[forecast_column] = pd.to_numeric(
        result[forecast_column],
        errors="coerce",
    )

    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    result = result.dropna(
        subset=required_columns
    )

    if result.empty:
        raise ForecastingDataError(
            "No valid forecast observations are available."
        )

    if (result[forecast_column] < 0).any():
        raise ForecastingDataError(
            "Forecast demand cannot be negative."
        )

    if result[date_column].duplicated().any():
        raise ForecastingDataError(
            "Forecast dates must be unique."
        )

    return result.sort_values(
        date_column
    ).reset_index(drop=True)


# =====================================================================
# PART 4.27.3 — GENERATE DEMAND SCENARIOS
# =====================================================================

def di4_generate_demand_scenarios(
    forecast_data: pd.DataFrame,
    config: Optional[DI4ScenarioConfig] = None,
) -> pd.DataFrame:
    """
    Generate alternative demand scenarios.

    Scenario changes are illustrative assumptions applied to
    the baseline forecast, not independently trained predictions.
    """

    if config is None:
        config = DI4ScenarioConfig()

    config.validate()

    forecast = di4_scenario_prepare_forecast(
        forecast_data=forecast_data,
        forecast_column=config.forecast_column,
        date_column=config.date_column,
    )

    baseline = forecast[
        config.forecast_column
    ].to_numpy(dtype=float)

    scenario_frames = []

    for scenario_name, change_fraction in (
        config.demand_change_scenarios.items()
    ):
        scenario = forecast[
            [config.date_column]
        ].copy()

        scenario["Scenario"] = scenario_name

        scenario["Demand_Change_Percent"] = (
            change_fraction * 100
        )

        scenario["Baseline_Forecast"] = baseline

        scenario["Scenario_Demand"] = (
            baseline * (1 + change_fraction)
        )

        scenario["Scenario_Demand"] = (
            scenario["Scenario_Demand"].clip(lower=0)
        )

        scenario["Demand_Difference"] = (
            scenario["Scenario_Demand"]
            - scenario["Baseline_Forecast"]
        )

        scenario_frames.append(scenario)

    return pd.concat(
        scenario_frames,
        ignore_index=True,
    )


# =====================================================================
# PART 4.27.4 — SUMMARIZE SCENARIO OUTCOMES
# =====================================================================

def di4_summarize_demand_scenarios(
    scenario_data: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize demand and demand differences by scenario."""

    required_columns = [
        "Scenario",
        "Scenario_Demand",
        "Baseline_Forecast",
        "Demand_Difference",
    ]

    missing = [
        column
        for column in required_columns
        if column not in scenario_data.columns
    ]

    if missing:
        raise ForecastingDataError(
            "Scenario summary is missing columns: "
            + ", ".join(missing)
        )

    data = scenario_data.copy()

    for column in [
        "Scenario_Demand",
        "Baseline_Forecast",
        "Demand_Difference",
    ]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna(
        subset=required_columns
    )

    if data.empty:
        raise ForecastingDataError(
            "No valid scenario records remain."
        )

    summary = (
        data.groupby(
            "Scenario",
            sort=False,
        )
        .agg(
            Forecast_Periods=(
                "Scenario_Demand",
                "count",
            ),
            Total_Scenario_Demand=(
                "Scenario_Demand",
                "sum",
            ),
            Average_Scenario_Demand=(
                "Scenario_Demand",
                "mean",
            ),
            Minimum_Scenario_Demand=(
                "Scenario_Demand",
                "min",
            ),
            Maximum_Scenario_Demand=(
                "Scenario_Demand",
                "max",
            ),
            Total_Baseline_Demand=(
                "Baseline_Forecast",
                "sum",
            ),
            Total_Demand_Difference=(
                "Demand_Difference",
                "sum",
            ),
        )
        .reset_index()
    )

    baseline_totals = summary.loc[
        summary["Scenario"] == "Baseline",
        "Total_Baseline_Demand",
    ]

    if not baseline_totals.empty:
        baseline_total = float(
            baseline_totals.iloc[0]
        )

        if baseline_total > 0:
            summary["Total_Difference_Percent"] = (
                summary["Total_Demand_Difference"]
                / baseline_total
                * 100
            )
        else:
            summary["Total_Difference_Percent"] = np.nan

    return summary


# =====================================================================
# PART 4.27.5 — BUILD PLANNING BUFFER
# =====================================================================

def di4_build_demand_planning_buffer(
    forecast_data: pd.DataFrame,
    buffer_percent: float = 10.0,
    forecast_column: str = "Forecast_Demand",
    lower_bound_column: str = "Lower_Bound",
    upper_bound_column: str = "Upper_Bound",
    date_column: str = "Forecast_Date",
) -> pd.DataFrame:
    """
    Calculate a simple planning buffer above forecast demand.

    The buffer is a user-defined planning assumption, not a
    statistically calibrated safety-stock calculation.
    """

    if not np.isfinite(buffer_percent):
        raise ValueError(
            "Buffer percentage must be finite."
        )

    if buffer_percent < 0:
        raise ValueError(
            "Buffer percentage cannot be negative."
        )

    forecast = di4_scenario_prepare_forecast(
        forecast_data=forecast_data,
        forecast_column=forecast_column,
        date_column=date_column,
    )

    result = forecast.copy()

    result["Planning_Buffer_Percent"] = (
        buffer_percent
    )

    result["Planning_Buffer_Units"] = (
        result[forecast_column]
        * buffer_percent
        / 100
    )

    result["Buffered_Planning_Demand"] = (
        result[forecast_column]
        + result["Planning_Buffer_Units"]
    )

    if lower_bound_column in result.columns:
        result["Forecast_Lower_Bound"] = (
            result[lower_bound_column]
        )

    if upper_bound_column in result.columns:
        result["Forecast_Upper_Bound"] = (
            result[upper_bound_column]
        )

    return result


# =====================================================================
# PART 4.27.6 — COMPARE SCENARIO AGAINST PLANNING CAPACITY
# =====================================================================

def di4_compare_scenarios_with_capacity(
    scenario_data: pd.DataFrame,
    available_capacity: float,
) -> pd.DataFrame:
    """
    Compare scenario demand against a supplied capacity value.

    Capacity must use the same unit and period as scenario demand.
    """

    if not np.isfinite(available_capacity):
        raise ValueError(
            "Available capacity must be finite."
        )

    if available_capacity < 0:
        raise ValueError(
            "Available capacity cannot be negative."
        )

    required_columns = [
        "Scenario",
        "Scenario_Demand",
    ]

    missing = [
        column
        for column in required_columns
        if column not in scenario_data.columns
    ]

    if missing:
        raise ForecastingDataError(
            "Capacity comparison is missing columns: "
            + ", ".join(missing)
        )

    result = scenario_data.copy()

    result["Available_Capacity"] = (
        available_capacity
    )

    result["Capacity_Gap"] = (
        result["Scenario_Demand"]
        - available_capacity
    )

    result["Capacity_Status"] = np.select(
        [
            result["Capacity_Gap"] > 0,
            result["Capacity_Gap"] < 0,
        ],
        [
            "DEMAND_ABOVE_CAPACITY",
            "CAPACITY_WITHIN_LIMIT",
        ],
        default="AT_CAPACITY",
    )

    result["Capacity_Utilization_Percent"] = (
        np.where(
            available_capacity > 0,
            result["Scenario_Demand"]
            / available_capacity
            * 100,
            np.nan,
        )
    )

    return result


# =====================================================================
# PART 4.27.7 — BUILD COMPLETE SCENARIO REPORT
# =====================================================================

def di4_build_scenario_report(
    forecast_data: pd.DataFrame,
    config: Optional[DI4ScenarioConfig] = None,
    available_capacity: Optional[float] = None,
) -> dict[str, Any]:
    """Build scenario results, summaries and optional capacity analysis."""

    if config is None:
        config = DI4ScenarioConfig()

    config.validate()

    scenario_data = di4_generate_demand_scenarios(
        forecast_data=forecast_data,
        config=config,
    )

    summary = di4_summarize_demand_scenarios(
        scenario_data
    )

    capacity_comparison = None

    if available_capacity is not None:
        capacity_comparison = (
            di4_compare_scenarios_with_capacity(
                scenario_data=scenario_data,
                available_capacity=available_capacity,
            )
        )

    return {
        "report_version": "1.0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "scenario_count": int(
            scenario_data["Scenario"].nunique()
        ),
        "forecast_period_count": int(
            scenario_data["Forecast_Date"].nunique()
        ),
        "scenario_summary": summary,
        "scenario_details": scenario_data,
        "capacity_comparison": capacity_comparison,
        "assumptions": {
            "scenario_changes": (
                config.demand_change_scenarios
            ),
            "planning_buffer_percent": (
                config.planning_buffer_percent
            ),
            "scenario_values_are_indicative": True,
        },
        "limitations": [
            "Scenarios are mathematical adjustments to a baseline forecast.",
            "Scenario percentages are assumptions, not probability estimates.",
            "Capacity comparison is valid only when units and periods match.",
            "Planning buffers are not equivalent to optimized safety stock.",
        ],
    }


# =====================================================================
# PART 4.27.8 — EXPORT SCENARIO REPORT
# =====================================================================

def di4_export_scenario_report(
    report: dict[str, Any],
    output_directory: str | Path,
) -> dict[str, str]:
    """Export scenario detail, summary and optional capacity analysis."""

    output_dir = Path(
        output_directory
    ).expanduser()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    exports: dict[str, str] = {}

    scenario_details = report.get(
        "scenario_details"
    )

    if isinstance(scenario_details, pd.DataFrame):
        path = (
            output_dir
            / "demand_scenario_details.csv"
        )

        scenario_details.to_csv(
            path,
            index=False,
        )

        exports["scenario_details_csv"] = str(path)

    scenario_summary = report.get(
        "scenario_summary"
    )

    if isinstance(scenario_summary, pd.DataFrame):
        path = (
            output_dir
            / "demand_scenario_summary.csv"
        )

        scenario_summary.to_csv(
            path,
            index=False,
        )

        exports["scenario_summary_csv"] = str(path)

    capacity_comparison = report.get(
        "capacity_comparison"
    )

    if isinstance(capacity_comparison, pd.DataFrame):
        path = (
            output_dir
            / "scenario_capacity_comparison.csv"
        )

        capacity_comparison.to_csv(
            path,
            index=False,
        )

        exports["capacity_comparison_csv"] = str(path)

    json_payload = {
        key: value
        for key, value in report.items()
        if not isinstance(value, pd.DataFrame)
    }

    json_path = (
        output_dir
        / "demand_scenario_report.json"
    )

    di4_pipeline_export_json(
        json_payload,
        json_path,
    )

    exports["scenario_report_json"] = str(
        json_path
    )

    return exports


# =====================================================================
# PART 4.27.9 — SCENARIO FUNCTION REGISTRY
# =====================================================================

DI4_SCENARIO_FUNCTION_REGISTRY = {
    "prepare_forecast": di4_scenario_prepare_forecast,
    "generate_scenarios": di4_generate_demand_scenarios,
    "summarize_scenarios": di4_summarize_demand_scenarios,
    "planning_buffer": di4_build_demand_planning_buffer,
    "capacity_comparison": di4_compare_scenarios_with_capacity,
    "build_report": di4_build_scenario_report,
    "export_report": di4_export_scenario_report,
}


# =====================================================================
# PART 4.27 — END
# FORECAST SCENARIO ANALYSIS AND WHAT-IF PLANNING
# =====================================================================

# =====================================================================
# PART 4.28 — START
# SEASONAL PATTERN ANALYSIS AND DEMAND DECOMPOSITION
# =====================================================================


# =====================================================================
# PART 4.28.1 — SEASONAL ANALYSIS CONFIGURATION
# =====================================================================

@dataclass
class DI4SeasonalityConfig:
    """Configuration for seasonal demand analysis."""

    date_column: str = "Forecast_Date"
    demand_column: str = "Actual_Demand"

    minimum_observations: int = 14

    weekly_period: int = 7
    monthly_period: int = 30

    rolling_window: int = 7

    seasonal_strength_threshold: float = 0.30

    def validate(self) -> "DI4SeasonalityConfig":
        if self.minimum_observations < 3:
            raise ValueError(
                "minimum_observations must be at least 3."
            )

        if self.weekly_period < 2:
            raise ValueError(
                "weekly_period must be at least 2."
            )

        if self.monthly_period < 2:
            raise ValueError(
                "monthly_period must be at least 2."
            )

        if self.rolling_window < 2:
            raise ValueError(
                "rolling_window must be at least 2."
            )

        if not (
            0 <= self.seasonal_strength_threshold <= 1
        ):
            raise ValueError(
                "seasonal_strength_threshold must be between 0 and 1."
            )

        return self


# =====================================================================
# PART 4.28.2 — PREPARE CONTINUOUS DEMAND SERIES
# =====================================================================

def di4_seasonality_prepare_data(
    historical_data: pd.DataFrame,
    date_column: str = "Forecast_Date",
    demand_column: str = "Actual_Demand",
) -> pd.DataFrame:
    """
    Prepare a continuous daily series for seasonal analysis.

    Missing calendar dates are inserted with NaN demand.
    Missing demand is not automatically interpreted as zero.
    """

    history = di4_explain_prepare_history(
        historical_data=historical_data,
        date_column=date_column,
        demand_column=demand_column,
    )

    history[date_column] = (
        history[date_column].dt.normalize()
    )

    if history[date_column].duplicated().any():
        raise ForecastingDataError(
            "Duplicate calendar dates must be aggregated first."
        )

    history = history.sort_values(
        date_column
    ).reset_index(drop=True)

    complete_dates = pd.date_range(
        start=history[date_column].min(),
        end=history[date_column].max(),
        freq="D",
    )

    result = (
        history.set_index(date_column)
        .reindex(complete_dates)
        .rename_axis(date_column)
        .reset_index()
    )

    return result


# =====================================================================
# PART 4.28.3 — CALENDAR-BASED SEASONAL PROFILES
# =====================================================================

def di4_build_calendar_seasonal_profiles(
    historical_data: pd.DataFrame,
    date_column: str = "Forecast_Date",
    demand_column: str = "Actual_Demand",
) -> dict[str, pd.DataFrame]:
    """Build weekday, month and day-of-month demand profiles."""

    history = di4_seasonality_prepare_data(
        historical_data=historical_data,
        date_column=date_column,
        demand_column=demand_column,
    )

    valid = history.dropna(
        subset=[demand_column]
    ).copy()

    if valid.empty:
        raise ForecastingDataError(
            "No valid demand observations for seasonal profiles."
        )

    valid["Weekday_Number"] = (
        valid[date_column].dt.dayofweek
    )

    valid["Weekday_Name"] = (
        valid[date_column].dt.day_name()
    )

    valid["Month_Number"] = (
        valid[date_column].dt.month
    )

    valid["Month_Name"] = (
        valid[date_column].dt.month_name()
    )

    valid["Day_Of_Month"] = (
        valid[date_column].dt.day
    )

    weekday_profile = (
        valid.groupby(
            ["Weekday_Number", "Weekday_Name"],
            as_index=False,
        )
        .agg(
            Observation_Count=(demand_column, "count"),
            Average_Demand=(demand_column, "mean"),
            Median_Demand=(demand_column, "median"),
            Minimum_Demand=(demand_column, "min"),
            Maximum_Demand=(demand_column, "max"),
            Demand_Std=(demand_column, "std"),
        )
        .sort_values("Weekday_Number")
        .reset_index(drop=True)
    )

    month_profile = (
        valid.groupby(
            ["Month_Number", "Month_Name"],
            as_index=False,
        )
        .agg(
            Observation_Count=(demand_column, "count"),
            Average_Demand=(demand_column, "mean"),
            Median_Demand=(demand_column, "median"),
            Minimum_Demand=(demand_column, "min"),
            Maximum_Demand=(demand_column, "max"),
            Demand_Std=(demand_column, "std"),
        )
        .sort_values("Month_Number")
        .reset_index(drop=True)
    )

    day_profile = (
        valid.groupby(
            "Day_Of_Month",
            as_index=False,
        )
        .agg(
            Observation_Count=(demand_column, "count"),
            Average_Demand=(demand_column, "mean"),
            Median_Demand=(demand_column, "median"),
        )
        .sort_values("Day_Of_Month")
        .reset_index(drop=True)
    )

    return {
        "weekday_profile": weekday_profile,
        "month_profile": month_profile,
        "day_of_month_profile": day_profile,
    }


# =====================================================================
# PART 4.28.4 — WEEKLY SEASONALITY STRENGTH
# =====================================================================

def di4_measure_weekly_seasonality(
    historical_data: pd.DataFrame,
    demand_column: str = "Actual_Demand",
    period: int = 7,
    rolling_window: int = 7,
) -> dict[str, Any]:
    """
    Estimate weekly seasonal strength using a simple decomposition.

    This is a diagnostic estimate, not a formal statistical test.
    Missing observations remain missing during estimation.
    """

    if period < 2:
        raise ValueError(
            "Seasonal period must be at least 2."
        )

    if rolling_window < 2:
        raise ValueError(
            "Rolling window must be at least 2."
        )

    if demand_column not in historical_data.columns:
        raise ForecastingDataError(
            f"Missing demand column: {demand_column}"
        )

    values = pd.to_numeric(
        historical_data[demand_column],
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if values.notna().sum() < max(
        period * 2,
        rolling_window + 1,
    ):
        raise ForecastingDataError(
            "Insufficient observations for seasonal analysis."
        )

    trend = values.rolling(
        window=rolling_window,
        center=True,
        min_periods=max(2, rolling_window // 2),
    ).mean()

    detrended = values - trend

    valid = pd.DataFrame({
        "value": values,
        "trend": trend,
        "detrended": detrended,
    }).dropna()

    if valid.empty:
        raise ForecastingDataError(
            "No valid detrended observations remain."
        )

    valid["Season_Position"] = (
        np.arange(len(values))[valid.index] % period
    )

    seasonal_profile = (
        valid.groupby("Season_Position")["detrended"]
        .mean()
    )

    seasonal_component = (
        valid["Season_Position"]
        .map(seasonal_profile)
    )

    residual = (
        valid["detrended"]
        - seasonal_component
    )

    denominator = float(
        np.var(valid["value"].to_numpy(dtype=float))
    )

    numerator = float(
        np.var(seasonal_component.to_numpy(dtype=float))
    )

    seasonal_strength = (
        float(
            np.clip(
                numerator / denominator,
                0,
                1,
            )
        )
        if denominator > 0
        else 0.0
    )

    return {
        "observation_count": int(len(valid)),
        "seasonal_period": int(period),
        "seasonal_strength_estimate": seasonal_strength,
        "seasonal_component_std": float(
            seasonal_component.std(ddof=0)
        ),
        "residual_std": float(
            residual.std(ddof=0)
        ),
        "interpretation": (
            "Potential seasonal pattern detected."
            if seasonal_strength >= 0.30
            else "Weak seasonal pattern under this diagnostic."
        ),
        "method_note": (
            "Uses rolling-mean detrending and positional grouping. "
            "It is an exploratory diagnostic, not a formal test."
        ),
    }


# =====================================================================
# PART 4.28.5 — ROLLING TREND AND VOLATILITY
# =====================================================================

def di4_build_rolling_demand_diagnostics(
    historical_data: pd.DataFrame,
    date_column: str = "Forecast_Date",
    demand_column: str = "Actual_Demand",
    rolling_window: int = 7,
) -> pd.DataFrame:
    """Calculate rolling demand averages and variability."""

    if rolling_window < 2:
        raise ValueError(
            "rolling_window must be at least 2."
        )

    history = di4_seasonality_prepare_data(
        historical_data=historical_data,
        date_column=date_column,
        demand_column=demand_column,
    )

    demand = history[demand_column]

    result = history[
        [date_column, demand_column]
    ].copy()

    result["Rolling_Average"] = demand.rolling(
        window=rolling_window,
        min_periods=rolling_window,
    ).mean()

    result["Rolling_Std"] = demand.rolling(
        window=rolling_window,
        min_periods=rolling_window,
    ).std()

    result["Rolling_Minimum"] = demand.rolling(
        window=rolling_window,
        min_periods=rolling_window,
    ).min()

    result["Rolling_Maximum"] = demand.rolling(
        window=rolling_window,
        min_periods=rolling_window,
    ).max()

    result["Rolling_Range"] = (
        result["Rolling_Maximum"]
        - result["Rolling_Minimum"]
    )

    result["Rolling_Coefficient_Of_Variation"] = (
        result["Rolling_Std"]
        / result["Rolling_Average"].abs().replace(0, np.nan)
    )

    return result


# =====================================================================
# PART 4.28.6 — ADDITIVE DEMAND DECOMPOSITION
# =====================================================================

def di4_decompose_demand_series(
    historical_data: pd.DataFrame,
    demand_column: str = "Actual_Demand",
    period: int = 7,
    date_column: str = "Forecast_Date",
) -> pd.DataFrame:
    """
    Produce an approximate additive decomposition.

    Components:
        Observed = Trend + Seasonal + Residual

    The trend uses a centered rolling average. Seasonal effects
    are estimated by position within the requested period.
    """

    if period < 2:
        raise ValueError(
            "Decomposition period must be at least 2."
        )

    history = di4_seasonality_prepare_data(
        historical_data=historical_data,
        date_column=date_column,
        demand_column=demand_column,
    )

    if history[demand_column].notna().sum() < period * 2:
        raise ForecastingDataError(
            "At least two seasonal cycles are required."
        )

    result = history[
        [date_column, demand_column]
    ].copy()

    result["Trend_Component"] = (
        result[demand_column].rolling(
            window=period,
            center=True,
            min_periods=max(2, period // 2),
        ).mean()
    )

    result["Detrended_Demand"] = (
        result[demand_column]
        - result["Trend_Component"]
    )

    result["Season_Position"] = (
        np.arange(len(result)) % period
    )

    seasonal_profile = (
        result.groupby("Season_Position")[
            "Detrended_Demand"
        ]
        .mean()
    )

    result["Seasonal_Component"] = (
        result["Season_Position"].map(
            seasonal_profile
        )
    )

    # Center the estimated seasonal profile around zero.
    seasonal_mean = result[
        "Seasonal_Component"
    ].mean()

    result["Seasonal_Component"] = (
        result["Seasonal_Component"]
        - seasonal_mean
    )

    result["Residual_Component"] = (
        result[demand_column]
        - result["Trend_Component"]
        - result["Seasonal_Component"]
    )

    result["Decomposition_Period"] = period

    return result


# =====================================================================
# PART 4.28.7 — SEASONALITY REPORT
# =====================================================================

def di4_build_seasonality_report(
    historical_data: pd.DataFrame,
    config: Optional[DI4SeasonalityConfig] = None,
) -> dict[str, Any]:
    """Build a combined seasonal and trend diagnostic report."""

    if config is None:
        config = DI4SeasonalityConfig()

    config.validate()

    history = di4_seasonality_prepare_data(
        historical_data=historical_data,
        date_column=config.date_column,
        demand_column=config.demand_column,
    )

    valid_count = int(
        history[config.demand_column].notna().sum()
    )

    if valid_count < config.minimum_observations:
        raise ForecastingDataError(
            "Insufficient observations for seasonality report."
        )

    profiles = di4_build_calendar_seasonal_profiles(
        historical_data=history,
        date_column=config.date_column,
        demand_column=config.demand_column,
    )

    rolling = di4_build_rolling_demand_diagnostics(
        historical_data=history,
        date_column=config.date_column,
        demand_column=config.demand_column,
        rolling_window=config.rolling_window,
    )

    weekly_strength = None

    if valid_count >= max(
        config.weekly_period * 2,
        config.rolling_window + 1,
    ):
        weekly_strength = di4_measure_weekly_seasonality(
            historical_data=history,
            demand_column=config.demand_column,
            period=config.weekly_period,
            rolling_window=config.rolling_window,
        )

    decomposition = None

    if valid_count >= config.weekly_period * 2:
        decomposition = di4_decompose_demand_series(
            historical_data=history,
            demand_column=config.demand_column,
            period=config.weekly_period,
            date_column=config.date_column,
        )

    return {
        "report_version": "1.0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "observation_count": valid_count,
        "calendar_profiles": profiles,
        "rolling_diagnostics": rolling,
        "weekly_seasonality": weekly_strength,
        "decomposition": decomposition,
        "limitations": [
            "Seasonal profiles are descriptive, not causal.",
            "Missing demand is not automatically treated as zero.",
            "The decomposition is approximate and uses rolling averages.",
            "Weekly periodicity does not prove that demand repeats every week.",
            "Month-name comparisons may be affected by unequal sample sizes.",
        ],
    }


# =====================================================================
# PART 4.28.8 — EXPORT SEASONALITY REPORT
# =====================================================================

def di4_export_seasonality_report(
    report: dict[str, Any],
    output_directory: str | Path,
) -> dict[str, str]:
    """Export seasonal profiles, rolling diagnostics and decomposition."""

    output_dir = Path(
        output_directory
    ).expanduser()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    exports: dict[str, str] = {}

    profiles = report.get(
        "calendar_profiles",
        {},
    )

    if isinstance(profiles, dict):
        for name, dataframe in profiles.items():
            if isinstance(dataframe, pd.DataFrame):
                path = output_dir / f"{name}.csv"

                dataframe.to_csv(
                    path,
                    index=False,
                )

                exports[name] = str(path)

    rolling = report.get(
        "rolling_diagnostics"
    )

    if isinstance(rolling, pd.DataFrame):
        path = (
            output_dir
            / "rolling_demand_diagnostics.csv"
        )

        rolling.to_csv(
            path,
            index=False,
        )

        exports["rolling_diagnostics"] = str(path)

    decomposition = report.get(
        "decomposition"
    )

    if isinstance(decomposition, pd.DataFrame):
        path = (
            output_dir
            / "demand_decomposition.csv"
        )

        decomposition.to_csv(
            path,
            index=False,
        )

        exports["decomposition"] = str(path)

    json_payload = {}

    for key, value in report.items():
        if isinstance(value, pd.DataFrame):
            continue

        if isinstance(value, dict):
            json_payload[key] = {
                nested_key: nested_value
                for nested_key, nested_value in value.items()
                if not isinstance(
                    nested_value,
                    pd.DataFrame,
                )
            }
        else:
            json_payload[key] = value

    json_path = (
        output_dir
        / "seasonality_summary.json"
    )

    di4_pipeline_export_json(
        json_payload,
        json_path,
    )

    exports["summary_json"] = str(json_path)

    return exports


# =====================================================================
# PART 4.28.9 — SEASONALITY FUNCTION REGISTRY
# =====================================================================

DI4_SEASONALITY_FUNCTION_REGISTRY = {
    "prepare_data": di4_seasonality_prepare_data,
    "calendar_profiles": di4_build_calendar_seasonal_profiles,
    "weekly_seasonality": di4_measure_weekly_seasonality,
    "rolling_diagnostics": di4_build_rolling_demand_diagnostics,
    "decomposition": di4_decompose_demand_series,
    "build_report": di4_build_seasonality_report,
    "export_report": di4_export_seasonality_report,
}


# =====================================================================
# PART 4.28 — END
# SEASONAL PATTERN ANALYSIS AND DEMAND DECOMPOSITION
# =====================================================================

# PART 4.29 — START
# Forecast Model Calibration & Prediction Interval Evaluation

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class DI4CalibrationConfig:
    """
    Configuration for forecast calibration and interval evaluation.
    """

    confidence_levels: tuple = (0.80, 0.90, 0.95)
    minimum_observations: int = 10
    minimum_interval_width: float = 0.0
    target_coverage_tolerance: float = 0.05
    date_column: str = "Forecast_Date"
    actual_column: str = "Actual_Demand"
    predicted_column: str = "Predicted_Demand"
    lower_column: str = "Lower_Bound"
    upper_column: str = "Upper_Bound"

    def validate(self) -> None:
        if self.minimum_observations < 1:
            raise ValueError(
                "minimum_observations must be at least 1."
            )

        if self.minimum_interval_width < 0:
            raise ValueError(
                "minimum_interval_width cannot be negative."
            )

        if not 0 <= self.target_coverage_tolerance <= 1:
            raise ValueError(
                "target_coverage_tolerance must be between 0 and 1."
            )

        if not self.confidence_levels:
            raise ValueError(
                "At least one confidence level is required."
            )

        for level in self.confidence_levels:
            if not isinstance(level, (int, float)):
                raise ValueError(
                    "Confidence levels must be numeric."
                )

            if not 0 < level < 1:
                raise ValueError(
                    "Each confidence level must be between 0 and 1."
                )


def di4_calibration_prepare_data(
    dataframe: pd.DataFrame,
    config: Optional[DI4CalibrationConfig] = None,
) -> pd.DataFrame:
    """
    Validate and prepare actuals, predictions, and optional interval bounds.
    """

    config = config or DI4CalibrationConfig()
    config.validate()

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")

    required = [
        config.actual_column,
        config.predicted_column,
    ]

    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"Missing calibration columns: {missing}"
        )

    result = dataframe.copy()

    if config.date_column in result.columns:
        result[config.date_column] = pd.to_datetime(
            result[config.date_column],
            errors="coerce",
            utc=True,
        )

    numeric_columns = [
        config.actual_column,
        config.predicted_column,
        config.lower_column,
        config.upper_column,
    ]

    for column in numeric_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

            result[column] = result[column].replace(
                [np.inf, -np.inf],
                np.nan,
            )

    result = result.dropna(
        subset=[
            config.actual_column,
            config.predicted_column,
        ]
    ).copy()

    result = result[
        (result[config.actual_column] >= 0)
        & (result[config.predicted_column] >= 0)
    ].copy()

    if config.date_column in result.columns:
        result = result.dropna(
            subset=[config.date_column]
        )

        result = result.sort_values(
            config.date_column
        )

    return result.reset_index(drop=True)


def di4_calibration_interval_metrics(
    dataframe: pd.DataFrame,
    confidence_level: float = 0.95,
    config: Optional[DI4CalibrationConfig] = None,
) -> dict:
    """
    Evaluate prediction interval coverage and width.

    This function evaluates existing lower and upper bounds.
    It does not assume that intervals are statistically calibrated.
    """

    config = config or DI4CalibrationConfig()
    config.validate()

    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must be between 0 and 1."
        )

    required = [
        config.actual_column,
        config.lower_column,
        config.upper_column,
    ]

    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:
        return {
            "confidence_level": float(confidence_level),
            "status": "unavailable",
            "reason": f"Missing columns: {missing}",
            "observations": 0,
            "empirical_coverage": None,
            "target_coverage": float(confidence_level),
            "coverage_gap": None,
            "mean_interval_width": None,
            "median_interval_width": None,
            "mean_interval_score": None,
        }

    data = dataframe[required].copy()

    for column in required:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        data[column] = data[column].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    data = data.dropna(subset=required)

    data = data[
        (data[config.actual_column] >= 0)
        & (data[config.lower_column] <= data[config.upper_column])
    ].copy()

    if len(data) < config.minimum_observations:
        return {
            "confidence_level": float(confidence_level),
            "status": "insufficient_data",
            "reason": (
                "Not enough valid observations to evaluate "
                "prediction intervals."
            ),
            "observations": int(len(data)),
            "empirical_coverage": None,
            "target_coverage": float(confidence_level),
            "coverage_gap": None,
            "mean_interval_width": None,
            "median_interval_width": None,
            "mean_interval_score": None,
        }

    actual = data[config.actual_column].to_numpy(dtype=float)
    lower = data[config.lower_column].to_numpy(dtype=float)
    upper = data[config.upper_column].to_numpy(dtype=float)

    covered = (actual >= lower) & (actual <= upper)

    coverage = float(np.mean(covered))
    widths = upper - lower

    alpha = 1.0 - float(confidence_level)

    below = actual < lower
    above = actual > upper

    interval_score = (
        widths
        + (2.0 / alpha) * (lower - actual) * below
        + (2.0 / alpha) * (actual - upper) * above
    )

    coverage_gap = coverage - float(confidence_level)

    if abs(coverage_gap) <= config.target_coverage_tolerance:
        calibration_status = "within_tolerance"
    elif coverage_gap < 0:
        calibration_status = "under_coverage"
    else:
        calibration_status = "over_coverage"

    return {
        "confidence_level": float(confidence_level),
        "status": calibration_status,
        "reason": None,
        "observations": int(len(data)),
        "empirical_coverage": coverage,
        "target_coverage": float(confidence_level),
        "coverage_gap": float(coverage_gap),
        "mean_interval_width": float(np.mean(widths)),
        "median_interval_width": float(np.median(widths)),
        "mean_interval_score": float(np.mean(interval_score)),
        "covered_observations": int(np.sum(covered)),
        "missed_observations": int(np.sum(~covered)),
        "below_lower_bound": int(np.sum(below)),
        "above_upper_bound": int(np.sum(above)),
    }


def di4_calibration_residual_summary(
    dataframe: pd.DataFrame,
    config: Optional[DI4CalibrationConfig] = None,
) -> dict:
    """
    Summarize forecast errors and directional bias.
    """

    config = config or DI4CalibrationConfig()
    config.validate()

    data = di4_calibration_prepare_data(
        dataframe,
        config=config,
    )

    if data.empty:
        return {
            "status": "no_valid_data",
            "observations": 0,
        }

    actual = data[config.actual_column].to_numpy(dtype=float)
    predicted = data[config.predicted_column].to_numpy(dtype=float)

    residuals = actual - predicted
    absolute_errors = np.abs(residuals)

    nonzero_actual = actual != 0

    if np.any(nonzero_actual):
        mape = float(
            np.mean(
                absolute_errors[nonzero_actual]
                / actual[nonzero_actual]
            )
            * 100.0
        )
    else:
        mape = None

    denominator = float(np.sum(np.abs(actual)))

    wape = (
        float(np.sum(absolute_errors) / denominator * 100.0)
        if denominator > 0
        else None
    )

    return {
        "status": "success",
        "observations": int(len(data)),
        "mean_actual": float(np.mean(actual)),
        "mean_prediction": float(np.mean(predicted)),
        "mean_error": float(np.mean(residuals)),
        "mean_absolute_error": float(np.mean(absolute_errors)),
        "root_mean_squared_error": float(
            np.sqrt(np.mean(residuals ** 2))
        ),
        "mean_absolute_percentage_error": mape,
        "weighted_absolute_percentage_error": wape,
        "underforecast_count": int(np.sum(residuals > 0)),
        "overforecast_count": int(np.sum(residuals < 0)),
        "exact_prediction_count": int(np.sum(residuals == 0)),
    }


def di4_calibration_compare_levels(
    dataframe: pd.DataFrame,
    config: Optional[DI4CalibrationConfig] = None,
) -> pd.DataFrame:
    """
    Compare empirical coverage and interval width across confidence levels.

    The caller must supply bounds for each confidence level separately,
    using columns such as Lower_Bound_80 and Upper_Bound_80.
    """

    config = config or DI4CalibrationConfig()
    config.validate()

    rows = []

    for level in config.confidence_levels:
        suffix = str(int(round(float(level) * 100)))

        lower_column = f"Lower_Bound_{suffix}"
        upper_column = f"Upper_Bound_{suffix}"

        if (
            lower_column not in dataframe.columns
            or upper_column not in dataframe.columns
        ):
            rows.append(
                {
                    "confidence_level": float(level),
                    "status": "unavailable",
                    "reason": (
                        f"Required columns: {lower_column}, "
                        f"{upper_column}"
                    ),
                    "observations": 0,
                    "empirical_coverage": np.nan,
                    "target_coverage": float(level),
                    "coverage_gap": np.nan,
                    "mean_interval_width": np.nan,
                    "median_interval_width": np.nan,
                    "mean_interval_score": np.nan,
                }
            )
            continue

        level_config = DI4CalibrationConfig(
            confidence_levels=(float(level),),
            minimum_observations=config.minimum_observations,
            minimum_interval_width=config.minimum_interval_width,
            target_coverage_tolerance=config.target_coverage_tolerance,
            date_column=config.date_column,
            actual_column=config.actual_column,
            predicted_column=config.predicted_column,
            lower_column=lower_column,
            upper_column=upper_column,
        )

        metrics = di4_calibration_interval_metrics(
            dataframe,
            confidence_level=float(level),
            config=level_config,
        )

        rows.append(metrics)

    return pd.DataFrame(rows)


def di4_calibration_build_report(
    dataframe: pd.DataFrame,
    config: Optional[DI4CalibrationConfig] = None,
) -> dict:
    """
    Build a combined calibration and prediction interval report.
    """

    config = config or DI4CalibrationConfig()
    config.validate()

    prepared = di4_calibration_prepare_data(
        dataframe,
        config=config,
    )

    residual_summary = di4_calibration_residual_summary(
        prepared,
        config=config,
    )

    interval_metrics = di4_calibration_interval_metrics(
        prepared,
        confidence_level=0.95,
        config=config,
    )

    level_comparison = di4_calibration_compare_levels(
        dataframe,
        config=config,
    )

    return {
        "report_version": "1.0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "observations": int(len(prepared)),
        "residual_summary": residual_summary,
        "default_interval_metrics": interval_metrics,
        "confidence_level_comparison": level_comparison,
        "interpretation": {
            "coverage_note": (
                "Empirical coverage is the fraction of observed actual "
                "values contained within the supplied prediction intervals."
            ),
            "calibration_note": (
                "Coverage close to the target is a useful diagnostic, "
                "but does not guarantee future calibration."
            ),
            "data_note": (
                "Results depend on the supplied holdout observations and "
                "prediction bounds. They are not a guarantee of future demand."
            ),
        },
    }


def di4_calibration_export_report(
    report: dict,
    output_path: str | Path,
) -> Path:
    """
    Export the calibration report as JSON.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "report_version": report.get("report_version", "1.0"),
        "generated_at_utc": report.get("generated_at_utc"),
        "observations": report.get("observations", 0),
        "residual_summary": report.get("residual_summary", {}),
        "default_interval_metrics": report.get(
            "default_interval_metrics",
            {},
        ),
        "confidence_level_comparison": (
            report.get(
                "confidence_level_comparison",
                pd.DataFrame(),
            )
            .to_dict(orient="records")
            if isinstance(
                report.get("confidence_level_comparison"),
                pd.DataFrame,
            )
            else report.get(
                "confidence_level_comparison",
                [],
            )
        ),
        "interpretation": report.get("interpretation", {}),
    }

    return di4_pipeline_export_json(
        payload,
        output_path,
    )


DI4_CALIBRATION_FUNCTION_REGISTRY = {
    "prepare_data": di4_calibration_prepare_data,
    "interval_metrics": di4_calibration_interval_metrics,
    "residual_summary": di4_calibration_residual_summary,
    "compare_levels": di4_calibration_compare_levels,
    "build_report": di4_calibration_build_report,
    "export_report": di4_calibration_export_report,
}

# PART 4.29 — END

# PART 4.30 — START
# Monitoring Frequency Fix & Improvements

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DI4MonitoringFrequencyConfig:
    """
    Configuration for frequency-aware forecast monitoring.
    """

    frequency: str = "daily"
    timestamp_column: str = "Timestamp"
    actual_column: str = "Sales Count"
    forecast_date_column: str = "Forecast_Date"
    forecast_column: str = "Forecast_Demand"
    rolling_window: int = 14
    minimum_observations: int = 3

    def validate(self) -> None:
        if self.frequency not in {
            "daily",
            "weekly",
            "monthly",
        }:
            raise ValueError(
                "frequency must be daily, weekly, or monthly."
            )

        if self.rolling_window < 1:
            raise ValueError(
                "rolling_window must be at least 1."
            )

        if self.minimum_observations < 1:
            raise ValueError(
                "minimum_observations must be at least 1."
            )


def di4_monitor_frequency_rule(
    frequency: str,
) -> str:
    """
    Return the canonical pandas frequency rule.
    """

    mapping = {
        "daily": "D",
        "weekly": "W-MON",
        "monthly": "MS",
    }

    if frequency not in mapping:
        raise ValueError(
            "Unsupported frequency. Use daily, weekly, or monthly."
        )

    return mapping[frequency]


def di4_monitor_normalize_period_dates(
    dates: pd.Series,
    frequency: str,
) -> pd.Series:
    """
    Normalize timestamps to daily, Monday-start weekly,
    or month-start monitoring periods.
    """

    rule = di4_monitor_frequency_rule(frequency)

    parsed = pd.to_datetime(
        dates,
        errors="coerce",
        utc=True,
    )

    # Use timezone-naive UTC dates consistently for period matching.
    parsed = parsed.dt.tz_convert(None)

    if rule == "D":
        return parsed.dt.normalize()

    if rule == "W-MON":
        return (
            parsed.dt.normalize()
            - pd.to_timedelta(
                parsed.dt.weekday,
                unit="D",
            )
        )

    return parsed.dt.to_period("M").dt.to_timestamp()


def di4_monitor_prepare_actuals_by_frequency(
    dataframe: pd.DataFrame,
    config: Optional[DI4MonitoringFrequencyConfig] = None,
) -> pd.DataFrame:
    """
    Aggregate actual demand using the configured monitoring frequency.

    The output contains:
        Forecast_Date
        Actual_Demand
        Observation_Count
    """

    config = config or DI4MonitoringFrequencyConfig()
    config.validate()

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "dataframe must be a pandas DataFrame."
        )

    required = [
        config.timestamp_column,
        config.actual_column,
    ]

    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"Missing actual-demand columns: {missing}"
        )

    data = dataframe[
        required
    ].copy()

    data[config.timestamp_column] = pd.to_datetime(
        data[config.timestamp_column],
        errors="coerce",
        utc=True,
    )

    data[config.actual_column] = pd.to_numeric(
        data[config.actual_column],
        errors="coerce",
    )

    data[config.actual_column] = data[
        config.actual_column
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    data = data.dropna(
        subset=[
            config.timestamp_column,
            config.actual_column,
        ]
    )

    data = data[
        data[config.actual_column] >= 0
    ].copy()

    if data.empty:
        return pd.DataFrame(
            columns=[
                "Forecast_Date",
                "Actual_Demand",
                "Observation_Count",
            ]
        )

    data["Forecast_Date"] = (
        di4_monitor_normalize_period_dates(
            data[config.timestamp_column],
            config.frequency,
        )
    )

    result = (
        data.groupby(
            "Forecast_Date",
            as_index=False,
        )
        .agg(
            Actual_Demand=(
                config.actual_column,
                "sum",
            ),
            Observation_Count=(
                config.actual_column,
                "count",
            ),
        )
        .sort_values("Forecast_Date")
        .reset_index(drop=True)
    )

    return result


def di4_monitor_prepare_forecasts_by_frequency(
    dataframe: pd.DataFrame,
    config: Optional[DI4MonitoringFrequencyConfig] = None,
) -> pd.DataFrame:
    """
    Prepare forecasts and normalize their period dates.
    """

    config = config or DI4MonitoringFrequencyConfig()
    config.validate()

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "dataframe must be a pandas DataFrame."
        )

    required = [
        config.forecast_date_column,
        config.forecast_column,
    ]

    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"Missing forecast columns: {missing}"
        )

    data = dataframe[
        required
    ].copy()

    data[config.forecast_date_column] = pd.to_datetime(
        data[config.forecast_date_column],
        errors="coerce",
        utc=True,
    )

    data[config.forecast_column] = pd.to_numeric(
        data[config.forecast_column],
        errors="coerce",
    )

    data[config.forecast_column] = data[
        config.forecast_column
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    data = data.dropna(
        subset=required
    ).copy()

    data = data[
        data[config.forecast_column] >= 0
    ].copy()

    if data.empty:
        return pd.DataFrame(
            columns=[
                "Forecast_Date",
                "Forecast_Demand",
            ]
        )

    data["Forecast_Date"] = (
        di4_monitor_normalize_period_dates(
            data[config.forecast_date_column],
            config.frequency,
        )
    )

    data = data.rename(
        columns={
            config.forecast_column: "Forecast_Demand",
        }
    )

    # Multiple forecasts for the same normalized period are ambiguous.
    # Do not silently sum them or choose one arbitrarily.
    duplicates = data["Forecast_Date"].duplicated(
        keep=False
    )

    if duplicates.any():
        duplicate_dates = (
            data.loc[duplicates, "Forecast_Date"]
            .dt.strftime("%Y-%m-%d")
            .unique()
            .tolist()
        )

        raise ValueError(
            "Multiple forecasts map to the same monitoring period. "
            f"Resolve duplicate forecast periods first: {duplicate_dates[:10]}"
        )

    return (
        data[
            [
                "Forecast_Date",
                "Forecast_Demand",
            ]
        ]
        .sort_values("Forecast_Date")
        .reset_index(drop=True)
    )


def di4_monitor_match_frequency_periods(
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Match actual and forecast records on normalized period dates.
    """

    actual_required = [
        "Forecast_Date",
        "Actual_Demand",
    ]

    forecast_required = [
        "Forecast_Date",
        "Forecast_Demand",
    ]

    for column in actual_required:
        if column not in actuals.columns:
            raise ValueError(
                f"Actual data missing column: {column}"
            )

    for column in forecast_required:
        if column not in forecasts.columns:
            raise ValueError(
                f"Forecast data missing column: {column}"
            )

    actual_data = actuals[
        actual_required
    ].copy()

    forecast_data = forecasts[
        forecast_required
    ].copy()

    actual_data["Forecast_Date"] = pd.to_datetime(
        actual_data["Forecast_Date"],
        errors="coerce",
    )

    forecast_data["Forecast_Date"] = pd.to_datetime(
        forecast_data["Forecast_Date"],
        errors="coerce",
    )

    actual_data = actual_data.dropna(
        subset=["Forecast_Date"]
    )

    forecast_data = forecast_data.dropna(
        subset=["Forecast_Date"]
    )

    if actual_data["Forecast_Date"].duplicated().any():
        raise ValueError(
            "Actual data contains duplicate monitoring periods."
        )

    if forecast_data["Forecast_Date"].duplicated().any():
        raise ValueError(
            "Forecast data contains duplicate monitoring periods."
        )

    matched = actual_data.merge(
        forecast_data,
        on="Forecast_Date",
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    matched["Monitoring_Status"] = matched[
        "_merge"
    ].map(
        {
            "both": "matched",
            "left_only": "actual_only",
            "right_only": "forecast_only",
        }
    )

    matched = matched.drop(
        columns=["_merge"]
    )

    matched["Forecast_Error"] = (
        matched["Actual_Demand"]
        - matched["Forecast_Demand"]
    )

    matched["Absolute_Error"] = (
        matched["Forecast_Error"].abs()
    )

    matched["Squared_Error"] = (
        matched["Forecast_Error"] ** 2
    )

    return (
        matched.sort_values("Forecast_Date")
        .reset_index(drop=True)
    )


def di4_monitor_frequency_metrics(
    matched_data: pd.DataFrame,
    minimum_observations: int = 3,
) -> dict:
    """
    Calculate performance metrics only for matched periods.
    """

    if minimum_observations < 1:
        raise ValueError(
            "minimum_observations must be at least 1."
        )

    required = [
        "Forecast_Date",
        "Actual_Demand",
        "Forecast_Demand",
    ]

    missing = [
        column
        for column in required
        if column not in matched_data.columns
    ]

    if missing:
        raise ValueError(
            f"Matched data missing columns: {missing}"
        )

    data = matched_data[
        required
    ].copy()

    data["Actual_Demand"] = pd.to_numeric(
        data["Actual_Demand"],
        errors="coerce",
    )

    data["Forecast_Demand"] = pd.to_numeric(
        data["Forecast_Demand"],
        errors="coerce",
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    data = data.dropna(
        subset=[
            "Actual_Demand",
            "Forecast_Demand",
        ]
    )

    data = data[
        (data["Actual_Demand"] >= 0)
        & (data["Forecast_Demand"] >= 0)
    ].copy()

    count = len(data)

    if count < minimum_observations:
        return {
            "status": "insufficient_data",
            "observations": int(count),
            "mae": None,
            "rmse": None,
            "mape_percent": None,
            "wape_percent": None,
            "mean_error": None,
            "matched_periods": int(count),
        }

    actual = data["Actual_Demand"].to_numpy(
        dtype=float
    )

    predicted = data["Forecast_Demand"].to_numpy(
        dtype=float
    )

    errors = actual - predicted
    absolute_errors = np.abs(errors)

    nonzero_actual = actual != 0

    mape = (
        float(
            np.mean(
                absolute_errors[nonzero_actual]
                / actual[nonzero_actual]
            )
            * 100.0
        )
        if np.any(nonzero_actual)
        else None
    )

    actual_total = float(np.sum(np.abs(actual)))

    wape = (
        float(
            np.sum(absolute_errors)
            / actual_total
            * 100.0
        )
        if actual_total > 0
        else None
    )

    return {
        "status": "success",
        "observations": int(count),
        "matched_periods": int(count),
        "mae": float(np.mean(absolute_errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mape_percent": mape,
        "wape_percent": wape,
        "mean_error": float(np.mean(errors)),
        "underforecast_periods": int(np.sum(errors > 0)),
        "overforecast_periods": int(np.sum(errors < 0)),
        "exact_periods": int(np.sum(errors == 0)),
    }


def di4_monitor_build_frequency_report(
    raw_actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    config: Optional[DI4MonitoringFrequencyConfig] = None,
) -> dict:
    """
    Build a monitoring report using a consistent time frequency.
    """

    config = config or DI4MonitoringFrequencyConfig()
    config.validate()

    actual_data = di4_monitor_prepare_actuals_by_frequency(
        raw_actuals,
        config=config,
    )

    forecast_data = di4_monitor_prepare_forecasts_by_frequency(
        forecasts,
        config=config,
    )

    matched = di4_monitor_match_frequency_periods(
        actual_data,
        forecast_data,
    )

    metrics = di4_monitor_frequency_metrics(
        matched,
        minimum_observations=config.minimum_observations,
    )

    status_counts = (
        matched["Monitoring_Status"]
        .value_counts()
        .to_dict()
        if not matched.empty
        else {}
    )

    return {
        "report_version": "1.0",
        "frequency": config.frequency,
        "actual_periods": int(len(actual_data)),
        "forecast_periods": int(len(forecast_data)),
        "matched_periods": int(
            (matched["Monitoring_Status"] == "matched").sum()
        ) if not matched.empty else 0,
        "period_status_counts": {
            str(key): int(value)
            for key, value in status_counts.items()
        },
        "metrics": metrics,
        "actuals": actual_data,
        "forecasts": forecast_data,
        "matched_periods_data": matched,
        "interpretation": (
            "Performance metrics use only periods with both actual "
            "and forecast values. Actual-only and forecast-only periods "
            "are reported separately and excluded from error metrics."
        ),
    }


DI4_MONITORING_FREQUENCY_FUNCTION_REGISTRY = {
    "frequency_rule": di4_monitor_frequency_rule,
    "normalize_period_dates": di4_monitor_normalize_period_dates,
    "prepare_actuals": di4_monitor_prepare_actuals_by_frequency,
    "prepare_forecasts": di4_monitor_prepare_forecasts_by_frequency,
    "match_periods": di4_monitor_match_frequency_periods,
    "calculate_metrics": di4_monitor_frequency_metrics,
    "build_report": di4_monitor_build_frequency_report,
}

# PART 4.30 — END

# PART 4.31 — START
# Performance Optimization & Memory Efficiency

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Optional

import gc
import logging

import numpy as np
import pandas as pd


DI4_PERFORMANCE_LOGGER = logging.getLogger(
    "ferry_forecasting.performance"
)


@dataclass
class DI4PerformanceConfig:
    """
    Configuration for memory-efficient data processing.
    """

    chunk_size: int = 25000
    target_memory_reduction_percent: float = 15.0
    optimize_numeric_columns: bool = True
    optimize_category_columns: bool = True
    collect_garbage_after_processing: bool = False

    def validate(self) -> None:
        if (
            not isinstance(self.chunk_size, int)
            or isinstance(self.chunk_size, bool)
            or self.chunk_size < 1
        ):
            raise ValueError(
                "chunk_size must be a positive integer."
            )

        if not (
            0 <= self.target_memory_reduction_percent <= 100
        ):
            raise ValueError(
                "target_memory_reduction_percent must be between 0 and 100."
            )


def di4_measure_dataframe_memory(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Measure DataFrame memory usage in bytes and MiB.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "dataframe must be a pandas DataFrame."
        )

    memory_by_column = dataframe.memory_usage(
        index=True,
        deep=True,
    )

    total_bytes = int(memory_by_column.sum())

    return {
        "rows": int(len(dataframe)),
        "columns": int(len(dataframe.columns)),
        "memory_bytes": total_bytes,
        "memory_mib": float(
            total_bytes / (1024 ** 2)
        ),
        "memory_by_column_mib": {
            str(column): float(value / (1024 ** 2))
            for column, value in memory_by_column.iloc[1:].items()
        },
    }


def di4_optimize_dataframe_memory(
    dataframe: pd.DataFrame,
    config: Optional[DI4PerformanceConfig] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Reduce DataFrame memory usage without intentionally changing
    numeric values or converting missing values into zero.

    Numeric downcasting is conservative:
    integer columns remain integers and floating-point columns
    are converted only when pandas identifies a smaller suitable dtype.
    """

    config = config or DI4PerformanceConfig()
    config.validate()

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "dataframe must be a pandas DataFrame."
        )

    optimized = dataframe.copy()

    before = di4_measure_dataframe_memory(
        optimized
    )

    if config.optimize_numeric_columns:
        for column in optimized.columns:
            series = optimized[column]

            if pd.api.types.is_bool_dtype(series.dtype):
                continue

            if pd.api.types.is_integer_dtype(series.dtype):
                optimized[column] = pd.to_numeric(
                    series,
                    downcast="integer",
                )

            elif pd.api.types.is_float_dtype(series.dtype):
                optimized[column] = pd.to_numeric(
                    series,
                    downcast="float",
                )

    if config.optimize_category_columns:
        for column in optimized.columns:
            series = optimized[column]

            if (
                series.dtype == object
                or isinstance(
                    series.dtype,
                    pd.StringDtype,
                )
            ):
                non_null_count = int(
                    series.notna().sum()
                )

                if non_null_count == 0:
                    continue

                unique_count = int(
                    series.nunique(dropna=True)
                )

                cardinality_ratio = (
                    unique_count / non_null_count
                )

                # Category conversion is useful mainly when repeated
                # labels dominate the column.
                if (
                    unique_count > 0
                    and cardinality_ratio <= 0.20
                ):
                    try:
                        optimized[column] = (
                            series.astype("category")
                        )
                    except (TypeError, ValueError):
                        DI4_PERFORMANCE_LOGGER.warning(
                            "Could not optimize category column %s.",
                            column,
                        )

    after = di4_measure_dataframe_memory(
        optimized
    )

    before_bytes = before["memory_bytes"]
    after_bytes = after["memory_bytes"]

    reduction_percent = (
        (before_bytes - after_bytes)
        / before_bytes
        * 100.0
        if before_bytes > 0
        else 0.0
    )

    report = {
        "status": "success",
        "before": before,
        "after": after,
        "memory_saved_bytes": int(
            before_bytes - after_bytes
        ),
        "memory_saved_mib": float(
            (before_bytes - after_bytes) / (1024 ** 2)
        ),
        "memory_reduction_percent": float(
            reduction_percent
        ),
        "target_reduction_percent": float(
            config.target_memory_reduction_percent
        ),
        "target_achieved": bool(
            reduction_percent
            >= config.target_memory_reduction_percent
        ),
        "dtype_changes": {
            str(column): {
                "before": str(dataframe[column].dtype),
                "after": str(optimized[column].dtype),
            }
            for column in dataframe.columns
            if dataframe[column].dtype
            != optimized[column].dtype
        },
    }

    return optimized, report


def di4_read_csv_optimized(
    file_path: str | Path,
    usecols: Optional[list[str]] = None,
    parse_dates: Optional[list[str]] = None,
    optimize_memory: bool = True,
    config: Optional[DI4PerformanceConfig] = None,
    **read_csv_kwargs: Any,
) -> tuple[pd.DataFrame, dict]:
    """
    Read a CSV file with optional column selection and memory optimization.

    Additional pandas read_csv options may be passed as keyword arguments.
    """

    config = config or DI4PerformanceConfig()
    config.validate()

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"CSV file does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"CSV path is not a file: {path}"
        )

    started = perf_counter()

    dataframe = pd.read_csv(
        path,
        usecols=usecols,
        parse_dates=parse_dates,
        **read_csv_kwargs,
    )

    read_duration = perf_counter() - started

    if optimize_memory:
        dataframe, memory_report = (
            di4_optimize_dataframe_memory(
                dataframe,
                config=config,
            )
        )
    else:
        memory_report = di4_measure_dataframe_memory(
            dataframe
        )

    report = {
        "file_path": str(path.resolve()),
        "file_size_bytes": int(path.stat().st_size),
        "read_duration_seconds": float(
            read_duration
        ),
        "dataframe_memory": memory_report,
        "rows_loaded": int(len(dataframe)),
        "columns_loaded": int(len(dataframe.columns)),
    }

    return dataframe, report


def di4_read_csv_in_chunks(
    file_path: str | Path,
    process_chunk: Callable[[pd.DataFrame], Any],
    config: Optional[DI4PerformanceConfig] = None,
    usecols: Optional[list[str]] = None,
    **read_csv_kwargs: Any,
) -> dict:
    """
    Process CSV chunks sequentially.

    The callback receives one DataFrame chunk at a time.
    Callback results are counted, not retained in memory.

    The callback is responsible for persisting or aggregating results
    when those results need to be used after processing.
    """

    config = config or DI4PerformanceConfig()
    config.validate()

    if not callable(process_chunk):
        raise TypeError(
            "process_chunk must be callable."
        )

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"CSV file does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"CSV path is not a file: {path}"
        )

    started = perf_counter()

    chunks_processed = 0
    rows_processed = 0

    reader = pd.read_csv(
        path,
        chunksize=config.chunk_size,
        usecols=usecols,
        **read_csv_kwargs,
    )

    for chunk in reader:
        process_chunk(chunk)

        chunks_processed += 1
        rows_processed += len(chunk)

        del chunk

        if config.collect_garbage_after_processing:
            gc.collect()

    duration = perf_counter() - started

    return {
        "status": "success",
        "file_path": str(path.resolve()),
        "chunks_processed": int(chunks_processed),
        "rows_processed": int(rows_processed),
        "chunk_size": int(config.chunk_size),
        "duration_seconds": float(duration),
        "rows_per_second": float(
            rows_processed / duration
        ) if duration > 0 else None,
    }


def di4_measure_processing_time(
    function: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> dict:
    """
    Execute a callable and report elapsed time.

    The original function result is returned unchanged inside the report.
    Exceptions are recorded and re-raised rather than hidden.
    """

    if not callable(function):
        raise TypeError(
            "function must be callable."
        )

    started = perf_counter()

    try:
        result = function(
            *args,
            **kwargs,
        )

        duration = perf_counter() - started

        return {
            "status": "success",
            "function_name": getattr(
                function,
                "__name__",
                type(function).__name__,
            ),
            "duration_seconds": float(
                duration
            ),
            "result": result,
        }

    except Exception:
        duration = perf_counter() - started

        DI4_PERFORMANCE_LOGGER.exception(
            "Function execution failed after %.4f seconds.",
            duration,
        )

        raise


def di4_build_performance_report(
    dataframes: Optional[dict[str, pd.DataFrame]] = None,
    timings: Optional[list[dict]] = None,
) -> dict:
    """
    Build a lightweight performance report from existing measurements.
    """

    dataframes = dataframes or {}
    timings = timings or []

    dataframe_reports = {}

    for name, dataframe in dataframes.items():
        if not isinstance(dataframe, pd.DataFrame):
            dataframe_reports[str(name)] = {
                "status": "skipped",
                "reason": "Object is not a pandas DataFrame.",
            }
            continue

        dataframe_reports[str(name)] = (
            di4_measure_dataframe_memory(
                dataframe
            )
        )

    normalized_timings = []

    for timing in timings:
        if not isinstance(timing, dict):
            continue

        duration = timing.get(
            "duration_seconds"
        )

        try:
            duration = float(duration)
        except (TypeError, ValueError):
            duration = None

        normalized_timings.append(
            {
                "function_name": str(
                    timing.get(
                        "function_name",
                        "unknown",
                    )
                ),
                "duration_seconds": (
                    duration
                    if duration is not None
                    and np.isfinite(duration)
                    and duration >= 0
                    else None
                ),
                "status": str(
                    timing.get(
                        "status",
                        "unknown",
                    )
                ),
            }
        )

    return {
        "report_version": "1.0",
        "dataframe_memory": dataframe_reports,
        "timings": normalized_timings,
        "total_dataframes_measured": int(
            len(dataframe_reports)
        ),
        "total_timing_records": int(
            len(normalized_timings)
        ),
        "notes": [
            (
                "Memory figures describe DataFrame storage, "
                "not total process memory."
            ),
            (
                "Chunk processing reduces the need to load an entire "
                "CSV into memory, but callback logic can still consume memory."
            ),
            (
                "Performance depends on hardware, storage, data types, "
                "and the operations being performed."
            ),
        ],
    }


DI4_PERFORMANCE_FUNCTION_REGISTRY = {
    "measure_dataframe_memory": di4_measure_dataframe_memory,
    "optimize_dataframe_memory": di4_optimize_dataframe_memory,
    "read_csv_optimized": di4_read_csv_optimized,
    "read_csv_in_chunks": di4_read_csv_in_chunks,
    "measure_processing_time": di4_measure_processing_time,
    "build_performance_report": di4_build_performance_report,
}

# PART 4.31 — END

# PART 4.32 — START
# Error Handling & Edge Cases

import json
import logging
import math
import os
import tempfile

from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


DI4_ERROR_LOGGER = logging.getLogger(
    "ferry_forecasting.errors"
)


class DI4InputError(ValueError):
    """Raised when forecasting input is invalid."""


class DI4ConfigurationError(ValueError):
    """Raised when forecasting configuration is invalid."""


class DI4ProcessingError(RuntimeError):
    """Raised when a forecasting operation cannot be completed."""


class DI4ExportError(RuntimeError):
    """Raised when a forecasting artifact cannot be exported."""


def di4_error_validate_dataframe(
    dataframe: pd.DataFrame,
    required_columns: Optional[list[str]] = None,
    allow_empty: bool = False,
    allow_duplicate_columns: bool = False,
) -> dict:
    """
    Validate a DataFrame's basic structure.

    This function does not modify the supplied DataFrame.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise DI4InputError(
            "Input must be a pandas DataFrame."
        )

    if (
        not allow_duplicate_columns
        and dataframe.columns.duplicated().any()
    ):
        duplicates = (
            dataframe.columns[
                dataframe.columns.duplicated()
            ]
            .astype(str)
            .tolist()
        )

        raise DI4InputError(
            f"Duplicate column names detected: {duplicates}"
        )

    required_columns = required_columns or []

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise DI4InputError(
            f"Required columns are missing: {missing_columns}"
        )

    if dataframe.empty and not allow_empty:
        raise DI4InputError(
            "The input DataFrame is empty."
        )

    return {
        "status": "valid",
        "rows": int(len(dataframe)),
        "columns": int(len(dataframe.columns)),
        "missing_required_columns": [],
        "duplicate_columns": [],
        "empty": bool(dataframe.empty),
    }


def di4_error_validate_numeric_series(
    series: pd.Series,
    name: str = "value",
    allow_missing: bool = False,
    allow_negative: bool = True,
) -> dict:
    """
    Validate numeric values without silently replacing invalid entries.
    """

    if not isinstance(series, pd.Series):
        raise DI4InputError(
            f"{name} must be a pandas Series."
        )

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    numeric_array = numeric.to_numpy(
        dtype=float,
        na_value=np.nan,
    )

    missing_count = int(
        pd.isna(numeric_array).sum()
    )

    infinite_count = int(
        np.isinf(numeric_array).sum()
    )

    finite_values = numeric_array[
        np.isfinite(numeric_array)
    ]

    negative_count = int(
        np.sum(finite_values < 0)
    )

    if not allow_missing and missing_count:
        raise DI4InputError(
            f"{name} contains {missing_count} missing or "
            "non-numeric values."
        )

    if infinite_count:
        raise DI4InputError(
            f"{name} contains {infinite_count} infinite values."
        )

    if not allow_negative and negative_count:
        raise DI4InputError(
            f"{name} contains {negative_count} negative values."
        )

    return {
        "status": "valid",
        "name": name,
        "total_values": int(len(series)),
        "missing_or_non_numeric": missing_count,
        "infinite_values": infinite_count,
        "negative_values": negative_count,
        "finite_values": int(len(finite_values)),
        "minimum": (
            float(np.min(finite_values))
            if len(finite_values)
            else None
        ),
        "maximum": (
            float(np.max(finite_values))
            if len(finite_values)
            else None
        ),
    }


def di4_error_validate_dates(
    dates: pd.Series,
    name: str = "date",
    allow_missing: bool = False,
    require_unique: bool = False,
) -> dict:
    """
    Validate date values and optionally require unique timestamps.
    """

    if not isinstance(dates, pd.Series):
        raise DI4InputError(
            f"{name} must be a pandas Series."
        )

    parsed = pd.to_datetime(
        dates,
        errors="coerce",
        utc=True,
    )

    invalid_count = int(
        parsed.isna().sum()
    )

    if invalid_count and not allow_missing:
        raise DI4InputError(
            f"{name} contains {invalid_count} invalid or missing dates."
        )

    valid_dates = parsed.dropna()

    duplicate_count = int(
        valid_dates.duplicated().sum()
    )

    if require_unique and duplicate_count:
        raise DI4InputError(
            f"{name} contains {duplicate_count} duplicate timestamps."
        )

    return {
        "status": "valid",
        "name": name,
        "total_values": int(len(dates)),
        "invalid_dates": invalid_count,
        "duplicate_dates": duplicate_count,
        "unique_valid_dates": int(valid_dates.nunique()),
        "minimum_date": (
            valid_dates.min().isoformat()
            if not valid_dates.empty
            else None
        ),
        "maximum_date": (
            valid_dates.max().isoformat()
            if not valid_dates.empty
            else None
        ),
    }


def di4_error_check_forecast_horizon(
    horizon: Any,
    maximum_horizon: int = 365,
) -> int:
    """
    Validate forecast horizon and return a regular Python integer.
    """

    if (
        not isinstance(horizon, (int, np.integer))
        or isinstance(horizon, (bool, np.bool_))
    ):
        raise DI4ConfigurationError(
            "Forecast horizon must be an integer."
        )

    if (
        not isinstance(maximum_horizon, (int, np.integer))
        or isinstance(maximum_horizon, (bool, np.bool_))
        or maximum_horizon < 1
    ):
        raise DI4ConfigurationError(
            "maximum_horizon must be a positive integer."
        )

    if horizon < 1:
        raise DI4ConfigurationError(
            "Forecast horizon must be at least 1."
        )

    if horizon > maximum_horizon:
        raise DI4ConfigurationError(
            f"Forecast horizon cannot exceed {maximum_horizon}."
        )

    return int(horizon)


def di4_error_check_frequency(
    frequency: str,
) -> str:
    """
    Return a validated canonical frequency name.
    """

    if not isinstance(frequency, str):
        raise DI4ConfigurationError(
            "Frequency must be a string."
        )

    normalized = frequency.strip().lower()

    aliases = {
        "d": "daily",
        "day": "daily",
        "daily": "daily",
        "w": "weekly",
        "week": "weekly",
        "weekly": "weekly",
        "m": "monthly",
        "ms": "monthly",
        "month": "monthly",
        "monthly": "monthly",
    }

    if normalized not in aliases:
        raise DI4ConfigurationError(
            "Unsupported frequency. Use daily, weekly, or monthly."
        )

    return aliases[normalized]


def di4_error_check_forecast_bounds(
    lower_bounds: Any,
    upper_bounds: Any,
) -> dict:
    """
    Validate paired lower and upper prediction bounds.
    """

    lower = np.asarray(
        lower_bounds,
        dtype=float,
    )

    upper = np.asarray(
        upper_bounds,
        dtype=float,
    )

    if lower.ndim != 1 or upper.ndim != 1:
        raise DI4InputError(
            "Prediction bounds must be one-dimensional."
        )

    if lower.size != upper.size:
        raise DI4InputError(
            "Lower and upper bounds must have equal lengths."
        )

    if not np.isfinite(lower).all():
        raise DI4InputError(
            "Lower bounds contain missing or infinite values."
        )

    if not np.isfinite(upper).all():
        raise DI4InputError(
            "Upper bounds contain missing or infinite values."
        )

    if np.any(lower > upper):
        raise DI4InputError(
            "Some lower bounds exceed their upper bounds."
        )

    return {
        "status": "valid",
        "observations": int(lower.size),
        "minimum_interval_width": (
            float(np.min(upper - lower))
            if lower.size
            else None
        ),
        "maximum_interval_width": (
            float(np.max(upper - lower))
            if lower.size
            else None
        ),
    }


def di4_error_json_safe(
    value: Any,
) -> Any:
    """
    Convert common Python, NumPy, and pandas values into JSON-safe values.
    """

    if value is None or value is pd.NA or value is pd.NaT:
        return None

    if isinstance(value, dict):
        return {
            str(key): di4_error_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            di4_error_json_safe(item)
            for item in value
        ]

    if isinstance(value, pd.DataFrame):
        return [
            di4_error_json_safe(row)
            for row in value.to_dict(orient="records")
        ]

    if isinstance(value, pd.Series):
        return [
            di4_error_json_safe(item)
            for item in value.tolist()
        ]

    if isinstance(value, np.ndarray):
        return di4_error_json_safe(
            value.tolist()
        )

    if isinstance(value, np.generic):
        return di4_error_json_safe(
            value.item()
        )

    if isinstance(value, (pd.Timestamp, datetime, date)):
        if pd.isna(value):
            return None

        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)

    if isinstance(value, (str, bool)):
        return value

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return str(value)


def di4_error_atomic_json_export(
    payload: Any,
    output_path: str | Path,
) -> Path:
    """
    Write JSON atomically so a failed write does not leave a partial target.

    The temporary file is created in the target directory to keep the
    final replacement on the same filesystem when possible.
    """

    target = Path(output_path)

    try:
        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        safe_payload = di4_error_json_safe(
            payload
        )

        serialized = json.dumps(
            safe_payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(target.parent),
            text=True,
        )

        temporary_path = Path(
            temporary_name
        )

        try:
            with os.fdopen(
                file_descriptor,
                "w",
                encoding="utf-8",
            ) as file:
                file.write(serialized)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())

            temporary_path.replace(target)

        except Exception:
            try:
                temporary_path.unlink(
                    missing_ok=True
                )
            except OSError:
                DI4_ERROR_LOGGER.warning(
                    "Could not remove temporary export file: %s",
                    temporary_path,
                )

            raise

        return target

    except Exception as exc:
        DI4_ERROR_LOGGER.exception(
            "JSON export failed for %s.",
            target,
        )

        raise DI4ExportError(
            f"Could not export JSON to {target}: {exc}"
        ) from exc


def di4_error_run_safely(
    operation: Any,
    *args: Any,
    operation_name: Optional[str] = None,
    **kwargs: Any,
) -> dict:
    """
    Run an operation and return a structured success/error result.

    This helper is intended for optional workflows where callers explicitly
    want a result object instead of an exception. It does not replace
    validation at the source of the error.
    """

    if not callable(operation):
        raise TypeError(
            "operation must be callable."
        )

    name = operation_name or getattr(
        operation,
        "__name__",
        "unnamed_operation",
    )

    try:
        result = operation(
            *args,
            **kwargs,
        )

        return {
            "status": "success",
            "operation": name,
            "result": result,
            "error_type": None,
            "error_message": None,
        }

    except Exception as exc:
        DI4_ERROR_LOGGER.exception(
            "Optional operation '%s' failed.",
            name,
        )

        return {
            "status": "error",
            "operation": name,
            "result": None,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


def di4_error_build_health_summary(
    checks: dict[str, Any],
) -> dict:
    """
    Combine validation checks into a compact health summary.

    Each check may be a boolean or a dictionary containing a status.
    """

    if not isinstance(checks, dict):
        raise TypeError(
            "checks must be a dictionary."
        )

    normalized_checks = {}

    for name, result in checks.items():
        if isinstance(result, bool):
            passed = result

        elif isinstance(result, dict):
            status = str(
                result.get("status", "")
            ).strip().lower()

            passed = status in {
                "success",
                "valid",
                "passed",
                "healthy",
                "ok",
            }

        else:
            passed = False

        normalized_checks[str(name)] = {
            "passed": bool(passed),
            "result": di4_error_json_safe(result),
        }

    total = len(normalized_checks)

    passed_count = sum(
        item["passed"]
        for item in normalized_checks.values()
    )

    if passed_count == total:
        overall_status = "healthy"
    elif passed_count == 0:
        overall_status = "unhealthy"
    else:
        overall_status = "degraded"

    return {
        "status": overall_status,
        "total_checks": int(total),
        "passed_checks": int(passed_count),
        "failed_checks": int(total - passed_count),
        "checks": normalized_checks,
    }


DI4_ERROR_HANDLING_FUNCTION_REGISTRY = {
    "validate_dataframe": di4_error_validate_dataframe,
    "validate_numeric_series": di4_error_validate_numeric_series,
    "validate_dates": di4_error_validate_dates,
    "check_forecast_horizon": di4_error_check_forecast_horizon,
    "check_frequency": di4_error_check_frequency,
    "check_forecast_bounds": di4_error_check_forecast_bounds,
    "json_safe": di4_error_json_safe,
    "atomic_json_export": di4_error_atomic_json_export,
    "run_safely": di4_error_run_safely,
    "build_health_summary": di4_error_build_health_summary,
}

# PART 4.32 — END

# PART 4.33 — START
# Final Integration & Compatibility Layer

import inspect
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


DI4_INTEGRATION_REGISTRIES = {
    "core": "DI4_FUNCTION_REGISTRY",
    "backtesting": "DI4_BACKTEST_FUNCTION_REGISTRY",
    "pipeline": "DI4_PIPELINE_FUNCTION_REGISTRY",
    "reliability": "DI4_RELIABILITY_FUNCTION_REGISTRY",
    "testing": "DI4_TEST_FUNCTION_REGISTRY",
    "monitoring": "DI4_MONITORING_FUNCTION_REGISTRY",
    "integration": "DI4_INTEGRATION_FUNCTION_REGISTRY",
    "error_analysis": "DI4_ERROR_ANALYSIS_FUNCTION_REGISTRY",
    "explainability": "DI4_EXPLAINABILITY_FUNCTION_REGISTRY",
    "scenario": "DI4_SCENARIO_FUNCTION_REGISTRY",
    "seasonality": "DI4_SEASONALITY_FUNCTION_REGISTRY",
    "calibration": "DI4_CALIBRATION_FUNCTION_REGISTRY",
    "monitoring_frequency": "DI4_MONITORING_FREQUENCY_FUNCTION_REGISTRY",
    "performance": "DI4_PERFORMANCE_FUNCTION_REGISTRY",
    "error_handling": "DI4_ERROR_HANDLING_FUNCTION_REGISTRY",
}


def di4_integration_environment() -> dict:
    """
    Collect runtime and library version information.
    """

    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "engine_version": str(
            globals().get(
                "FORECAST_ENGINE_VERSION",
                "unknown",
            )
        ),
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }


def di4_integration_collect_registries() -> dict:
    """
    Collect available function registries without importing other files.
    """

    collected = {}
    missing = []

    for registry_name, global_name in (
        DI4_INTEGRATION_REGISTRIES.items()
    ):
        registry = globals().get(global_name)

        if registry is None:
            missing.append(
                {
                    "registry_name": registry_name,
                    "global_name": global_name,
                }
            )
            continue

        if not isinstance(registry, dict):
            collected[registry_name] = {
                "status": "invalid_registry",
                "global_name": global_name,
                "functions": [],
            }
            continue

        collected[registry_name] = {
            "status": "available",
            "global_name": global_name,
            "functions": sorted(
                str(key)
                for key in registry
            ),
            "function_count": int(
                len(registry)
            ),
        }

    return {
        "available_registries": collected,
        "missing_registries": missing,
        "available_count": int(
            len(collected)
        ),
        "missing_count": int(
            len(missing)
        ),
    }


def di4_integration_check_registry_functions() -> dict:
    """
    Check whether registered values are callable.

    This checks references only; it does not execute the functions.
    """

    results = []

    for registry_name, global_name in (
        DI4_INTEGRATION_REGISTRIES.items()
    ):
        registry = globals().get(global_name)

        if not isinstance(registry, dict):
            results.append(
                {
                    "registry": registry_name,
                    "status": "unavailable",
                    "checked": 0,
                    "invalid_functions": [],
                }
            )
            continue

        invalid = []

        for function_name, function in registry.items():
            if not callable(function):
                invalid.append(
                    str(function_name)
                )

        results.append(
            {
                "registry": registry_name,
                "status": (
                    "healthy"
                    if not invalid
                    else "invalid_functions"
                ),
                "checked": int(
                    len(registry)
                ),
                "invalid_functions": invalid,
            }
        )

    total_checked = sum(
        item["checked"]
        for item in results
    )

    total_invalid = sum(
        len(item["invalid_functions"])
        for item in results
    )

    return {
        "status": (
            "healthy"
            if total_invalid == 0
            and all(
                item["status"] != "unavailable"
                for item in results
            )
            else "issues_detected"
        ),
        "total_functions_checked": int(
            total_checked
        ),
        "total_invalid_functions": int(
            total_invalid
        ),
        "registries": results,
    }


def di4_integration_check_required_symbols() -> dict:
    """
    Check whether important project functions and classes exist.

    Missing symbols are reported rather than causing import-time failures.
    """

    required_symbols = [
        "di4_validate_input",
        "di4_prepare_demand_data",
        "di4_aggregate_demand",
        "di4_complete_time_series",
        "di4_validate_forecasting_series",
        "di4_naive_forecast",
        "di4_moving_average_forecast",
        "di4_seasonal_naive_forecast",
        "di4_evaluate_forecast",
        "di4_compare_baseline_models",
        "di4_forecast_future_demand",
        "di4_run_forecasting_pipeline",
        "di4_run_unit_tests",
        "di4_run_integrated_workflow",
        "di4_build_error_analysis_report",
        "di4_generate_business_insights",
        "di4_build_scenario_report",
        "di4_build_seasonality_report",
        "di4_calibration_build_report",
        "di4_monitor_build_frequency_report",
        "di4_optimize_dataframe_memory",
        "di4_error_atomic_json_export",
    ]

    missing = [
        symbol
        for symbol in required_symbols
        if not callable(
            globals().get(symbol)
        )
    ]

    available = [
        symbol
        for symbol in required_symbols
        if callable(
            globals().get(symbol)
        )
    ]

    return {
        "status": (
            "healthy"
            if not missing
            else "missing_symbols"
        ),
        "required_symbol_count": int(
            len(required_symbols)
        ),
        "available_symbol_count": int(
            len(available)
        ),
        "missing_symbols": missing,
        "available_symbols": available,
    }


def di4_integration_check_configurations() -> dict:
    """
    Check that expected configuration classes are defined.

    This does not instantiate configurations or assume that all classes
    expose a validate() method.
    """

    configuration_names = [
        "DI4BacktestConfig",
        "DI4PipelineConfig",
        "DI4MonitoringConfig",
        "DI4ErrorAnalysisConfig",
        "DI4ExplainabilityConfig",
        "DI4ScenarioConfig",
        "DI4SeasonalityConfig",
        "DI4CalibrationConfig",
        "DI4MonitoringFrequencyConfig",
        "DI4PerformanceConfig",
    ]

    results = []

    for name in configuration_names:
        config_class = globals().get(name)

        exists = (
            inspect.isclass(config_class)
        )

        validate_available = (
            exists
            and callable(
                getattr(
                    config_class,
                    "validate",
                    None,
                )
            )
        )

        results.append(
            {
                "name": name,
                "class_available": bool(exists),
                "validate_method_available": bool(
                    validate_available
                ),
            }
        )

    missing = [
        item["name"]
        for item in results
        if not item["class_available"]
    ]

    return {
        "status": (
            "healthy"
            if not missing
            else "missing_configurations"
        ),
        "configuration_count": int(
            len(results)
        ),
        "available_count": int(
            sum(
                item["class_available"]
                for item in results
            )
        ),
        "missing_configurations": missing,
        "details": results,
    }


def di4_integration_check_public_api() -> dict:
    """
    Check that the main pipeline has an inspectable public signature.
    """

    function = globals().get(
        "di4_run_forecasting_pipeline"
    )

    if not callable(function):
        return {
            "status": "unavailable",
            "function": "di4_run_forecasting_pipeline",
            "parameters": [],
        }

    try:
        signature = inspect.signature(
            function
        )

        parameters = [
            {
                "name": parameter.name,
                "kind": str(parameter.kind),
                "required": (
                    parameter.default
                    is inspect.Parameter.empty
                ),
            }
            for parameter in signature.parameters.values()
        ]

        return {
            "status": "available",
            "function": "di4_run_forecasting_pipeline",
            "signature": str(signature),
            "parameters": parameters,
        }

    except (TypeError, ValueError) as exc:
        return {
            "status": "signature_unavailable",
            "function": "di4_run_forecasting_pipeline",
            "error": str(exc),
            "parameters": [],
        }


def di4_integration_build_compatibility_report() -> dict:
    """
    Build a consolidated static compatibility report.

    No dataset is loaded, no forecast is generated, and no unit tests
    are executed by this function.
    """

    environment = di4_integration_environment()

    registries = di4_integration_collect_registries()

    registry_functions = (
        di4_integration_check_registry_functions()
    )

    symbols = (
        di4_integration_check_required_symbols()
    )

    configurations = (
        di4_integration_check_configurations()
    )

    public_api = (
        di4_integration_check_public_api()
    )

    checks = {
        "registry_functions": registry_functions,
        "required_symbols": symbols,
        "configurations": configurations,
        "public_api": public_api,
    }

    issues = []

    if registry_functions["status"] != "healthy":
        issues.append(
            "Some function registries are missing or contain invalid entries."
        )

    if symbols["status"] != "healthy":
        issues.append(
            "Some required public functions are missing."
        )

    if configurations["status"] != "healthy":
        issues.append(
            "Some configuration classes are missing."
        )

    if public_api["status"] != "available":
        issues.append(
            "The main pipeline public API is unavailable."
        )

    return {
        "report_version": "1.0",
        "report_type": "static_compatibility",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": (
            "potential_issues_detected"
            if issues
            else "static_checks_passed"
        ),
        "environment": environment,
        "registry_inventory": registries,
        "checks": checks,
        "issues": issues,
        "limitations": [
            (
                "Static compatibility checks do not prove that every "
                "function executes correctly."
            ),
            (
                "Dataset processing, model outputs, exports, and "
                "end-to-end integration require separate runtime tests."
            ),
            (
                "A missing optional registry may indicate that its "
                "corresponding part has not yet been pasted."
            ),
        ],
    }


def di4_integration_export_compatibility_report(
    report: dict,
    output_path: str | Path,
) -> Path:
    """
    Export the static compatibility report using the safe JSON exporter.
    """

    exporter = globals().get(
        "di4_error_atomic_json_export"
    )

    if callable(exporter):
        return exporter(
            report,
            output_path,
        )

    fallback_exporter = globals().get(
        "di4_pipeline_export_json"
    )

    if callable(fallback_exporter):
        return fallback_exporter(
            report,
            output_path,
        )

    raise RuntimeError(
        "No compatible JSON exporter is available."
    )


DI4_FINAL_INTEGRATION_FUNCTION_REGISTRY = {
    "environment": di4_integration_environment,
    "collect_registries": di4_integration_collect_registries,
    "check_registry_functions": di4_integration_check_registry_functions,
    "check_required_symbols": di4_integration_check_required_symbols,
    "check_configurations": di4_integration_check_configurations,
    "check_public_api": di4_integration_check_public_api,
    "build_compatibility_report": di4_integration_build_compatibility_report,
    "export_compatibility_report": di4_integration_export_compatibility_report,
}

# PART 4.33 — END
# PART 4.34 — START
# Final Diagnostics & Readiness Report

import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


DI4_READINESS_LOGGER = logging.getLogger(
    "ferry_forecasting.readiness"
)


DI4_READINESS_REQUIRED_FUNCTIONS = [
    "di4_validate_input",
    "di4_prepare_demand_data",
    "di4_aggregate_demand",
    "di4_complete_time_series",
    "di4_validate_forecasting_series",
    "di4_naive_forecast",
    "di4_moving_average_forecast",
    "di4_seasonal_naive_forecast",
    "di4_evaluate_forecast",
    "di4_compare_baseline_models",
    "di4_forecast_future_demand",
    "di4_run_forecasting_pipeline",
    "di4_run_unit_tests",
    "di4_run_integrated_workflow",
    "di4_build_error_analysis_report",
    "di4_generate_business_insights",
    "di4_build_scenario_report",
    "di4_build_seasonality_report",
    "di4_calibration_build_report",
    "di4_monitor_build_frequency_report",
    "di4_optimize_dataframe_memory",
    "di4_error_atomic_json_export",
]


def di4_readiness_check_python_environment() -> dict:
    """
    Check the Python runtime and core scientific libraries.
    """

    library_versions = {}

    for library_name in ("numpy", "pandas"):
        try:
            module = sys.modules.get(library_name)

            if module is None:
                library_versions[library_name] = {
                    "available": False,
                    "version": None,
                }
            else:
                library_versions[library_name] = {
                    "available": True,
                    "version": str(
                        getattr(module, "__version__", "unknown")
                    ),
                }

        except Exception as exc:
            library_versions[library_name] = {
                "available": False,
                "version": None,
                "error": str(exc),
            }

    all_available = all(
        item["available"]
        for item in library_versions.values()
    )

    return {
        "status": "passed" if all_available else "failed",
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "libraries": library_versions,
    }


def di4_readiness_check_required_functions() -> dict:
    """
    Check the availability of required forecasting functions.
    """

    available = []
    missing = []

    for function_name in DI4_READINESS_REQUIRED_FUNCTIONS:
        function = globals().get(function_name)

        if callable(function):
            available.append(function_name)
        else:
            missing.append(function_name)

    return {
        "status": "passed" if not missing else "failed",
        "required_count": int(
            len(DI4_READINESS_REQUIRED_FUNCTIONS)
        ),
        "available_count": int(len(available)),
        "missing_count": int(len(missing)),
        "available_functions": available,
        "missing_functions": missing,
    }


def di4_readiness_check_registries() -> dict:
    """
    Check the availability and basic structure of module registries.
    """

    registry_names = [
        "DI4_FUNCTION_REGISTRY",
        "DI4_BACKTEST_FUNCTION_REGISTRY",
        "DI4_PIPELINE_FUNCTION_REGISTRY",
        "DI4_RELIABILITY_FUNCTION_REGISTRY",
        "DI4_TEST_FUNCTION_REGISTRY",
        "DI4_MONITORING_FUNCTION_REGISTRY",
        "DI4_INTEGRATION_FUNCTION_REGISTRY",
        "DI4_ERROR_ANALYSIS_FUNCTION_REGISTRY",
        "DI4_EXPLAINABILITY_FUNCTION_REGISTRY",
        "DI4_SCENARIO_FUNCTION_REGISTRY",
        "DI4_SEASONALITY_FUNCTION_REGISTRY",
        "DI4_CALIBRATION_FUNCTION_REGISTRY",
        "DI4_MONITORING_FREQUENCY_FUNCTION_REGISTRY",
        "DI4_PERFORMANCE_FUNCTION_REGISTRY",
        "DI4_ERROR_HANDLING_FUNCTION_REGISTRY",
        "DI4_FINAL_INTEGRATION_FUNCTION_REGISTRY",
    ]

    available = []
    missing = []
    invalid = []

    for name in registry_names:
        registry = globals().get(name)

        if registry is None:
            missing.append(name)
            continue

        if not isinstance(registry, dict):
            invalid.append(name)
            continue

        non_callable = [
            str(key)
            for key, value in registry.items()
            if not callable(value)
        ]

        if non_callable:
            invalid.append(
                {
                    "registry": name,
                    "non_callable_entries": non_callable,
                }
            )
        else:
            available.append(
                {
                    "name": name,
                    "function_count": int(len(registry)),
                }
            )

    passed = not missing and not invalid

    return {
        "status": "passed" if passed else "issues_detected",
        "expected_registry_count": int(len(registry_names)),
        "available_registry_count": int(len(available)),
        "missing_registries": missing,
        "invalid_registries": invalid,
        "available_registries": available,
    }


def di4_readiness_check_configurations() -> dict:
    """
    Check availability of the project's main configuration classes.
    """

    configuration_names = [
        "DI4BacktestConfig",
        "DI4PipelineConfig",
        "DI4MonitoringConfig",
        "DI4ErrorAnalysisConfig",
        "DI4ExplainabilityConfig",
        "DI4ScenarioConfig",
        "DI4SeasonalityConfig",
        "DI4CalibrationConfig",
        "DI4MonitoringFrequencyConfig",
        "DI4PerformanceConfig",
    ]

    available = []
    missing = []

    for name in configuration_names:
        config_class = globals().get(name)

        if isinstance(config_class, type):
            available.append(name)
        else:
            missing.append(name)

    return {
        "status": "passed" if not missing else "issues_detected",
        "expected_count": int(len(configuration_names)),
        "available_count": int(len(available)),
        "missing_count": int(len(missing)),
        "available_configurations": available,
        "missing_configurations": missing,
    }


def di4_readiness_check_unit_test_availability() -> dict:
    """
    Check whether the unit-test runner and test-report exporter exist.

    This function does not execute tests.
    """

    test_runner = globals().get(
        "di4_run_unit_tests"
    )

    test_exporter = globals().get(
        "di4_export_unit_test_report"
    )

    return {
        "status": (
            "available"
            if callable(test_runner)
            else "unavailable"
        ),
        "test_runner_available": bool(
            callable(test_runner)
        ),
        "test_report_exporter_available": bool(
            callable(test_exporter)
        ),
        "tests_executed": False,
    }


def di4_readiness_check_dataset(
    dataframe: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Check dataset readiness without modifying the input.

    When no DataFrame is supplied, the result reports that dataset
    readiness was not evaluated.
    """

    if dataframe is None:
        return {
            "status": "not_checked",
            "reason": (
                "No DataFrame was supplied to the readiness checker."
            ),
            "rows": None,
            "columns": None,
        }

    if not isinstance(dataframe, pd.DataFrame):
        return {
            "status": "failed",
            "reason": (
                "Dataset must be a pandas DataFrame."
            ),
            "rows": None,
            "columns": None,
        }

    required_columns = [
        "Timestamp",
        "Sales Count",
        "Redemption Count",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    duplicate_columns = (
        dataframe.columns[
            dataframe.columns.duplicated()
        ]
        .astype(str)
        .tolist()
    )

    timestamp_valid_count = None

    if "Timestamp" in dataframe.columns:
        parsed_dates = pd.to_datetime(
            dataframe["Timestamp"],
            errors="coerce",
            utc=True,
        )

        timestamp_valid_count = int(
            parsed_dates.notna().sum()
        )

    demand_quality = {}

    for column in ("Sales Count", "Redemption Count"):
        if column not in dataframe.columns:
            demand_quality[column] = {
                "available": False,
            }
            continue

        values = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        numeric_values = values.to_numpy(
            dtype=float,
            na_value=np.nan,
        )

        finite = np.isfinite(numeric_values)

        demand_quality[column] = {
            "available": True,
            "numeric_count": int(finite.sum()),
            "invalid_or_missing_count": int(
                (~finite).sum()
            ),
            "negative_count": int(
                np.sum(
                    numeric_values[finite] < 0
                )
            ),
        }

    if dataframe.empty:
        status = "failed"
        reason = "Dataset is empty."
    elif missing_columns or duplicate_columns:
        status = "failed"
        reason = (
            "Required columns are missing or column names are duplicated."
        )
    elif timestamp_valid_count == 0:
        status = "failed"
        reason = "No valid timestamps were found."
    else:
        status = "inspected"
        reason = (
            "Basic dataset checks completed. Detailed data cleaning "
            "and forecasting suitability still require validation."
        )

    return {
        "status": status,
        "reason": reason,
        "rows": int(len(dataframe)),
        "columns": int(len(dataframe.columns)),
        "missing_required_columns": missing_columns,
        "duplicate_columns": duplicate_columns,
        "valid_timestamp_count": timestamp_valid_count,
        "demand_quality": demand_quality,
        "dataset_modified": False,
    }


def di4_readiness_build_report(
    dataframe: Optional[pd.DataFrame] = None,
    unit_test_report: Optional[dict] = None,
) -> dict:
    """
    Build the final forecasting-engine readiness report.

    This function performs structural checks only.
    It does not automatically execute unit tests or the forecast pipeline.
    """

    environment = (
        di4_readiness_check_python_environment()
    )

    functions = (
        di4_readiness_check_required_functions()
    )

    registries = (
        di4_readiness_check_registries()
    )

    configurations = (
        di4_readiness_check_configurations()
    )

    test_availability = (
        di4_readiness_check_unit_test_availability()
    )

    dataset = (
        di4_readiness_check_dataset(
            dataframe
        )
    )

    checks = {
        "environment": environment["status"] == "passed",
        "required_functions": functions["status"] == "passed",
        "registries": registries["status"] == "passed",
        "configurations": configurations["status"] == "passed",
    }

    issues = []

    for name, passed in checks.items():
        if not passed:
            issues.append(
                f"Readiness check failed: {name}"
            )

    if dataset["status"] == "failed":
        issues.append(
            "Dataset readiness checks failed."
        )

    test_status = "not_run"

    if isinstance(unit_test_report, dict):
        supplied_status = str(
            unit_test_report.get(
                "status",
                "unknown",
            )
        ).lower()

        if supplied_status in {
            "passed",
            "success",
            "healthy",
        }:
            test_status = "passed"

        elif supplied_status in {
            "failed",
            "error",
            "unhealthy",
        }:
            test_status = "failed"
            issues.append(
                "The supplied unit-test report indicates failures."
            )
        else:
            test_status = supplied_status

    if issues:
        overall_status = "issues_detected"
    elif dataset["status"] == "not_checked":
        overall_status = "structurally_ready_dataset_not_checked"
    elif test_status != "passed":
        overall_status = "structurally_ready_tests_not_confirmed"
    else:
        overall_status = "ready_for_review"

    return {
        "report_version": "1.0",
        "report_type": "forecasting_engine_readiness",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "overall_status": overall_status,
        "checks": checks,
        "environment": environment,
        "functions": functions,
        "registries": registries,
        "configurations": configurations,
        "unit_test_availability": test_availability,
        "unit_test_status": test_status,
        "dataset": dataset,
        "issues": issues,
        "limitations": [
            (
                "Structural readiness is not proof of successful execution."
            ),
            (
                "Unit tests and end-to-end integration must be run separately."
            ),
            (
                "Dataset readiness requires the actual dataset to be supplied."
            ),
            (
                "Forecast accuracy must be evaluated on chronologically "
                "separated validation data."
            ),
            (
                "Historical ticket sales and redemptions do not, by "
                "themselves, establish actual ferry boarding counts."
            ),
        ],
    }


def di4_readiness_export_report(
    report: dict,
    output_path: str | Path,
) -> Path:
    """
    Export the readiness report using the safest available JSON exporter.
    """

    exporter = globals().get(
        "di4_error_atomic_json_export"
    )

    if not callable(exporter):
        exporter = globals().get(
            "di4_pipeline_export_json"
        )

    if not callable(exporter):
        raise RuntimeError(
            "No compatible JSON export function is available."
        )

    return exporter(
        report,
        output_path,
    )


DI4_READINESS_FUNCTION_REGISTRY = {
    "check_environment": di4_readiness_check_python_environment,
    "check_functions": di4_readiness_check_required_functions,
    "check_registries": di4_readiness_check_registries,
    "check_configurations": di4_readiness_check_configurations,
    "check_unit_tests": di4_readiness_check_unit_test_availability,
    "check_dataset": di4_readiness_check_dataset,
    "build_report": di4_readiness_build_report,
    "export_report": di4_readiness_export_report,
}

# PART 4.34 — END
