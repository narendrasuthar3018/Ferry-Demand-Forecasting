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
    / "engineered_features.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "forecasts"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

TARGET_COLUMNS = ["Sales Count", "Redemption Count"]

HORIZONS_MINUTES = [15, 30, 60, 120]

HORIZON_TOLERANCE_MINUTES = 7.5


def setup_logging():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                REPORT_DIR / "forecasting_dataset.log",
                encoding="utf-8",
            ),
        ],
        force=True,
    )


def load_features():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required = ["Timestamp", *TARGET_COLUMNS]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["Timestamp", *TARGET_COLUMNS]
    )

    df = df.sort_values("Timestamp").reset_index(drop=True)

    if df["Timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps detected.")

    logging.info("Loaded feature records: %s", f"{len(df):,}")

    return df


def build_horizon_dataset(df, horizon_minutes):
    timestamps = df["Timestamp"].to_numpy(
        dtype="datetime64[ns]"
    )

    n = len(df)

    target_indices = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    tolerance = pd.Timedelta(
        minutes=HORIZON_TOLERANCE_MINUTES
    ).to_timedelta64()

    horizon = pd.Timedelta(
        minutes=horizon_minutes
    ).to_timedelta64()

    # For each origin, select the nearest future observation
    # within the allowed elapsed-time tolerance.
    for origin in range(n):
        desired = timestamps[origin] + horizon

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

        candidates = candidates[candidates > origin]

        if len(candidates) == 0:
            continue

        differences = np.abs(
            timestamps[candidates] - desired
        )

        target_indices[origin] = candidates[
            np.argmin(differences)
        ]

    valid_origins = np.flatnonzero(
        target_indices >= 0
    )

    valid_targets = target_indices[valid_origins]

    # Exclude pairs where the target timestamp is not truly
    # later than the forecast origin.
    valid_mask = valid_targets > valid_origins

    valid_origins = valid_origins[valid_mask]
    valid_targets = valid_targets[valid_mask]

    # Select only features that are available at forecast time.
    # Exclude raw targets, future target columns, and features
    # directly calculated from the current target.
    excluded_columns = {
        "Timestamp",
        "Date",
        "Day_Name",
        "Observed_Record",
        "Sales Count",
        "Redemption Count",
        "Sales_Minus_Redemption",
        "Sales_Redemption_Ratio",
        "Combined_Observed_Count",
        "Sales_Rolling_Mean_4",
        "Redemption_Rolling_Mean_4",
    }

    # Keep calendar/time features and historical features only.
    candidate_features = [
        column
        for column in df.columns
        if column not in excluded_columns
        and not column.endswith("_Historical_Anomaly_Score")
        and "_Change_From_Previous_Record" not in column
        and "_Percentage_Change" not in column
    ]

    feature_frame = df[candidate_features].copy()

    # Day_Name and Date are excluded above; all model inputs
    # must be numeric.
    feature_frame = feature_frame.select_dtypes(
        include=[np.number]
    )

    # Use origin-time features to predict future targets.
    model_data = feature_frame.iloc[
        valid_origins
    ].reset_index(drop=True)

    model_data.insert(
        0,
        "Forecast_Origin",
        df["Timestamp"].iloc[valid_origins].astype(str).to_numpy(),
    )

    model_data["Target_Timestamp"] = (
        df["Timestamp"].iloc[valid_targets].astype(str).to_numpy()
    )

    actual_elapsed_minutes = (
        df["Timestamp"].iloc[valid_targets].reset_index(drop=True)
        - df["Timestamp"].iloc[valid_origins].reset_index(drop=True)
    ).dt.total_seconds() / 60

    model_data["Actual_Horizon_Minutes"] = (
        actual_elapsed_minutes.to_numpy()
    )

    model_data["Horizon_Minutes"] = horizon_minutes

    for target in TARGET_COLUMNS:
        model_data[f"Future_{target}"] = (
            df[target].iloc[valid_targets].to_numpy()
        )

    model_data = model_data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Keep missing lag/rolling values for now; the training
    # pipeline will handle them using training-only imputers.
    output_path = (
        OUTPUT_DIR
        / f"forecasting_dataset_{horizon_minutes}m.csv"
    )

    model_data.to_csv(
        output_path,
        index=False,
    )

    report = {
        "horizon_minutes": horizon_minutes,
        "input_records": int(len(df)),
        "valid_forecast_pairs": int(len(model_data)),
        "feature_count": int(
            len(feature_frame.columns)
        ),
        "target_columns": [
            f"Future_{target}" for target in TARGET_COLUMNS
        ],
        "output_file": str(output_path),
        "timestamp_tolerance_minutes": (
            HORIZON_TOLERANCE_MINUTES
        ),
        "feature_policy": (
            "Only numeric features available at forecast origin "
            "are retained. Current raw targets are excluded."
        ),
    }

    return model_data, report


def main():
    setup_logging()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 65)
    logging.info("FORECASTING DATASET BUILDER")
    logging.info("=" * 65)

    df = load_features()

    reports = []

    for horizon in HORIZONS_MINUTES:
        logging.info(
            "Building %s-minute forecasting dataset.",
            horizon,
        )

        dataset, report = build_horizon_dataset(
            df,
            horizon,
        )

        reports.append(report)

        logging.info(
            "Valid pairs: %s",
            f"{len(dataset):,}",
        )

        logging.info(
            "Feature count: %s",
            report["feature_count"],
        )

    report_path = (
        REPORT_DIR / "forecasting_dataset_summary.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(reports, file, indent=4)

    logging.info(
        "Summary report saved: %s",
        report_path,
    )

    logging.info("FORECASTING DATASET BUILD COMPLETED")
    logging.info("=" * 65)


if __name__ == "__main__":
    main()
