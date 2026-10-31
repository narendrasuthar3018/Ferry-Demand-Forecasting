from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = (
    BASE_DIR
    / "outputs"
    / "forecasts"
    / "observed_time_series.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs"
FORECAST_DIR = OUTPUT_DIR / "forecasts"
REPORT_DIR = OUTPUT_DIR / "reports"

TARGET_COLUMNS = ["Sales Count", "Redemption Count"]

HORIZONS_MINUTES = [15, 30, 60, 120]

# Evaluation is chronological, never randomly shuffled.
TRAIN_FRACTION = 0.80

# A target observation must be close to the requested horizon.
# For example, a 15-minute forecast cannot silently use a
# target observation several hours after the forecast origin.
HORIZON_TOLERANCE_MINUTES = 7.5

# Limit the number of forecast origins to keep baseline
# evaluation practical on a large dataset.
MAX_EVALUATION_ORIGINS = 30000


def setup_logging():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                REPORT_DIR / "baseline_models.log",
                encoding="utf-8",
            ),
        ],
        force=True,
    )


def load_data():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Prepared data not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required = ["Timestamp", *TARGET_COLUMNS]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Required columns are missing: {missing}"
        )

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce",
    )

    for target in TARGET_COLUMNS:
        df[target] = pd.to_numeric(
            df[target],
            errors="coerce",
        )

    df = df.dropna(subset=required)
    df = df.sort_values("Timestamp").reset_index(drop=True)

    if df["Timestamp"].duplicated().any():
        raise ValueError(
            "Duplicate timestamps must be resolved before evaluation."
        )

    if (df[TARGET_COLUMNS] < 0).any().any():
        raise ValueError("Negative demand values detected.")

    if len(df) < 100:
        raise ValueError(
            "Insufficient observations for chronological evaluation."
        )

    logging.info("Loaded %s observations.", f"{len(df):,}")

    return df


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    valid = (
        np.isfinite(actual)
        & np.isfinite(predicted)
    )

    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        return {
            "MAE": None,
            "RMSE": None,
            "WAPE_Percent": None,
            "Mean_Bias": None,
            "Evaluated_Records": 0,
        }

    errors = predicted - actual

    absolute_error = np.abs(errors)

    denominator = np.abs(actual).sum()

    wape = (
        float(absolute_error.sum() / denominator * 100)
        if denominator > 0
        else None
    )

    return {
        "MAE": float(absolute_error.mean()),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "WAPE_Percent": wape,
        "Mean_Bias": float(errors.mean()),
        "Evaluated_Records": int(len(actual)),
    }


def build_origin_target_pairs(df, horizon_minutes):
    """
    Match each forecast origin to an observed target near the
    requested elapsed-time horizon.

    Each target timestamp is used at most once per horizon.
    Matching is chronological and one-to-one.
    """

    timestamps = df["Timestamp"].to_numpy(
        dtype="datetime64[ns]"
    )

    n = len(df)

    target_indices = np.full(n, -1, dtype=np.int64)

    tolerance = pd.Timedelta(
        minutes=HORIZON_TOLERANCE_MINUTES
    ).to_timedelta64()

    horizon = pd.Timedelta(
        minutes=horizon_minutes
    ).to_timedelta64()

    target_position = 0

    for origin_position in range(n):
        desired = timestamps[origin_position] + horizon

        while (
            target_position < n
            and timestamps[target_position] < desired - tolerance
        ):
            target_position += 1

        if target_position >= n:
            break

        # A forecast must predict a future observation.
        candidate = target_position

        if candidate <= origin_position:
            continue

        difference = abs(
            timestamps[candidate] - desired
        )

        if difference <= tolerance:
            target_indices[origin_position] = candidate
            target_position = candidate + 1

    return target_indices


def choose_evaluation_origins(df):
    """
    Create an expanding chronological split.
    The test period occurs strictly after the training period.
    """

    n = len(df)

    split_index = int(n * TRAIN_FRACTION)

    if split_index <= 0 or split_index >= n:
        raise ValueError("Invalid chronological split.")

    test_origins = np.arange(split_index, n)

    if len(test_origins) > MAX_EVALUATION_ORIGINS:
        positions = np.linspace(
            0,
            len(test_origins) - 1,
            MAX_EVALUATION_ORIGINS,
            dtype=int,
        )

        test_origins = test_origins[positions]

    return split_index, test_origins


def naive_predictions(values, origins):
    """
    Last-observation baseline.
    Uses only values available at the forecast origin.
    """

    return values[origins]


def moving_average_predictions(values, origins, window):
    """
    Average of the last `window` observed values, including
    the current observation available at the forecast origin.
    """

    predictions = np.full(
        len(origins),
        np.nan,
        dtype=float,
    )

    for i, origin in enumerate(origins):
        start = origin - window + 1

        if start < 0:
            continue

        history = values[start:origin + 1]

        if len(history) == window:
            predictions[i] = float(np.mean(history))

    return predictions


def seasonal_naive_predictions(
    values,
    timestamps,
    origins,
    seasonal_minutes,
):
    """
    Find an observed value close to one seasonal cycle before
    the forecast origin. Missing historical matches remain NaN.
    """

    predictions = np.full(
        len(origins),
        np.nan,
        dtype=float,
    )

    tolerance = pd.Timedelta(
        minutes=HORIZON_TOLERANCE_MINUTES
    ).to_timedelta64()

    seasonal_delta = pd.Timedelta(
        minutes=seasonal_minutes
    ).to_timedelta64()

    for i, origin in enumerate(origins):
        desired = timestamps[origin] - seasonal_delta

        left = np.searchsorted(
            timestamps,
            desired - tolerance,
            side="left",
        )

        right = np.searchsorted(
            timestamps,
            desired + tolerance,
            side="right",
        )

        candidates = np.arange(left, right)

        candidates = candidates[candidates < origin]

        if len(candidates) == 0:
            continue

        differences = np.abs(
            timestamps[candidates] - desired
        )

        best = candidates[np.argmin(differences)]

        predictions[i] = values[best]

    return predictions


def main():
    setup_logging()

    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 65)
    logging.info("BASELINE FORECASTING ENGINE")
    logging.info("=" * 65)

    df = load_data()

    timestamps = df["Timestamp"].to_numpy(
        dtype="datetime64[ns]"
    )

    split_index, evaluation_origins = (
        choose_evaluation_origins(df)
    )

    logging.info(
        "Chronological training records: %s",
        f"{split_index:,}",
    )

    logging.info(
        "Candidate evaluation origins: %s",
        f"{len(evaluation_origins):,}",
    )

    results = []
    prediction_records = []

    # Baselines use historical observations only.
    model_specs = [
        ("Naive_Last_Observation", "naive", None),
        ("Moving_Average_4", "moving_average", 4),
        ("Moving_Average_12", "moving_average", 12),
        ("Moving_Average_24", "moving_average", 24),
        ("Seasonal_Naive_24_Hours", "seasonal", 1440),
        ("Seasonal_Naive_7_Days", "seasonal", 10080),
    ]

    for horizon in HORIZONS_MINUTES:
        logging.info(
            "Evaluating horizon: %s minutes",
            horizon,
        )

        target_indices = build_origin_target_pairs(
            df,
            horizon,
        )

        valid_origins = evaluation_origins[
            target_indices[evaluation_origins] >= 0
        ]

        valid_targets = target_indices[valid_origins]

        # Keep only pairs whose target lies in the held-out
        # chronological test period.
        test_mask = valid_targets >= split_index

        valid_origins = valid_origins[test_mask]
        valid_targets = valid_targets[test_mask]

        if len(valid_origins) == 0:
            logging.warning(
                "No eligible test pairs for %s-minute horizon.",
                horizon,
            )
            continue

        for target in TARGET_COLUMNS:
            values = df[target].to_numpy(dtype=float)

            actual = values[valid_targets]

            for model_name, model_type, parameter in model_specs:
                if model_type == "naive":
                    predicted = naive_predictions(
                        values,
                        valid_origins,
                    )

                elif model_type == "moving_average":
                    predicted = moving_average_predictions(
                        values,
                        valid_origins,
                        parameter,
                    )

                elif model_type == "seasonal":
                    predicted = seasonal_naive_predictions(
                        values,
                        timestamps,
                        valid_origins,
                        parameter,
                    )

                else:
                    continue

                metrics = calculate_metrics(
                    actual,
                    predicted,
                )

                results.append(
                    {
                        "Horizon_Minutes": horizon,
                        "Target": target,
                        "Model": model_name,
                        **metrics,
                    }
                )

                for i in range(len(valid_origins)):
                    prediction_records.append(
                        {
                            "Horizon_Minutes": horizon,
                            "Target": target,
                            "Model": model_name,
                            "Forecast_Origin": str(
                                df["Timestamp"].iloc[
                                    valid_origins[i]
                                ]
                            ),
                            "Target_Timestamp": str(
                                df["Timestamp"].iloc[
                                    valid_targets[i]
                                ]
                            ),
                            "Actual": float(actual[i]),
                            "Predicted": (
                                float(predicted[i])
                                if np.isfinite(predicted[i])
                                else np.nan
                            ),
                            "Absolute_Error": (
                                float(abs(actual[i] - predicted[i]))
                                if np.isfinite(predicted[i])
                                else np.nan
                            ),
                        }
                    )

    if not results:
        raise RuntimeError(
            "No baseline results were generated. "
            "Review timestamp coverage and horizon matching tolerance."
        )

    results_df = pd.DataFrame(results)

    results_path = (
        REPORT_DIR / "baseline_model_comparison.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    predictions_df = pd.DataFrame(
        prediction_records
    )

    predictions_path = (
        FORECAST_DIR / "baseline_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    summary = {
        "input_records": int(len(df)),
        "training_records": int(split_index),
        "test_records": int(len(df) - split_index),
        "candidate_test_origins": int(len(evaluation_origins)),
        "horizons_minutes": HORIZONS_MINUTES,
        "targets": TARGET_COLUMNS,
        "evaluation_strategy": (
            "Chronological holdout; no random shuffle."
        ),
        "horizon_matching_tolerance_minutes": (
            HORIZON_TOLERANCE_MINUTES
        ),
        "forecast_origin_count_by_horizon": {},
        "models": [item[0] for item in model_specs],
        "metrics": ["MAE", "RMSE", "WAPE_Percent", "Mean_Bias"],
        "limitations": [
            (
                "This module establishes baseline benchmarks; "
                "it does not establish that any model is production-ready."
            ),
            (
                "Forecast targets are matched to observed timestamps "
                "within a defined elapsed-time tolerance."
            ),
            (
                "Seasonal baselines can have missing predictions when "
                "a matching historical observation is unavailable."
            ),
        ],
    }

    for horizon in HORIZONS_MINUTES:
        target_indices = build_origin_target_pairs(
            df,
            horizon,
        )

        eligible = evaluation_origins[
            target_indices[evaluation_origins] >= split_index
        ]

        summary["forecast_origin_count_by_horizon"][
            str(horizon)
        ] = int(len(eligible))

    with open(
        REPORT_DIR / "baseline_model_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    logging.info("-" * 65)
    logging.info("BASELINE EVALUATION COMPLETED")
    logging.info(
        "Comparison report: %s",
        results_path,
    )
    logging.info(
        "Prediction records: %s",
        predictions_path,
    )
    logging.info(
        "Summary report: %s",
        REPORT_DIR / "baseline_model_summary.json",
    )

    print("\nBASELINE MODEL COMPARISON")
    print("=" * 90)

    display_columns = [
        "Horizon_Minutes",
        "Target",
        "Model",
        "MAE",
        "RMSE",
        "WAPE_Percent",
        "Mean_Bias",
        "Evaluated_Records",
    ]

    print(
        results_df[display_columns]
        .round(3)
        .to_string(index=False)
    )

    logging.info("=" * 65)


if __name__ == "__main__":
    main()
