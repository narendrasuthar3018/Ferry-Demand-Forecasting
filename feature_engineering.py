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

ROLLING_WINDOWS = [4, 8, 12, 24]
LAG_INTERVALS_MINUTES = [15, 30, 60, 120]

TIMESTAMP_COLUMN = "Timestamp"


def setup_logging():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                REPORT_DIR / "feature_engineering.log",
                encoding="utf-8",
            ),
        ],
        force=True,
    )


def load_observed_series():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Prepared time series not found: {INPUT_PATH}. "
            "Run time_series_preparation.py first."
        )

    logging.info("Loading prepared observed time series.")

    df = pd.read_csv(INPUT_PATH)

    if TIMESTAMP_COLUMN not in df.columns:
        raise ValueError(
            f"Missing timestamp column: {TIMESTAMP_COLUMN}"
        )

    missing_targets = [
        column
        for column in TARGET_COLUMNS
        if column not in df.columns
    ]

    if missing_targets:
        raise ValueError(
            f"Missing target columns: {missing_targets}"
        )

    df[TIMESTAMP_COLUMN] = pd.to_datetime(
        df[TIMESTAMP_COLUMN],
        errors="coerce",
    )

    for column in TARGET_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[TIMESTAMP_COLUMN, *TARGET_COLUMNS]
    )

    df = df.sort_values(
        TIMESTAMP_COLUMN
    ).reset_index(drop=True)

    if df[TIMESTAMP_COLUMN].duplicated().any():
        raise ValueError(
            "Duplicate timestamps detected. "
            "Resolve duplicate records before feature engineering."
        )

    if (df[TARGET_COLUMNS] < 0).any().any():
        raise ValueError(
            "Negative target values detected."
        )

    logging.info("Loaded records: %s", f"{len(df):,}")

    return df


def add_calendar_features(df):
    logging.info("Creating calendar and cyclical time features.")

    timestamp = df[TIMESTAMP_COLUMN]

    df["Year"] = timestamp.dt.year
    df["Month"] = timestamp.dt.month
    df["Quarter"] = timestamp.dt.quarter
    df["Day"] = timestamp.dt.day
    df["Day_Of_Week"] = timestamp.dt.dayofweek
    df["Day_Name"] = timestamp.dt.day_name()
    df["Hour"] = timestamp.dt.hour
    df["Minute"] = timestamp.dt.minute
    df["Date"] = timestamp.dt.date.astype(str)

    df["Is_Weekend"] = (
        timestamp.dt.dayofweek >= 5
    ).astype("int8")

    df["Is_Monday"] = (
        timestamp.dt.dayofweek == 0
    ).astype("int8")

    df["Is_Friday"] = (
        timestamp.dt.dayofweek == 4
    ).astype("int8")

    minute_of_day = (
        timestamp.dt.hour * 60
        + timestamp.dt.minute
    )

    day_of_week = timestamp.dt.dayofweek

    df["Time_Of_Day_Sin"] = np.sin(
        2 * np.pi * minute_of_day / 1440
    )

    df["Time_Of_Day_Cos"] = np.cos(
        2 * np.pi * minute_of_day / 1440
    )

    df["Day_Of_Week_Sin"] = np.sin(
        2 * np.pi * day_of_week / 7
    )

    df["Day_Of_Week_Cos"] = np.cos(
        2 * np.pi * day_of_week / 7
    )

    df["Month_Sin"] = np.sin(
        2 * np.pi * (timestamp.dt.month - 1) / 12
    )

    df["Month_Cos"] = np.cos(
        2 * np.pi * (timestamp.dt.month - 1) / 12
    )

    return df


def add_elapsed_time_features(df):
    logging.info("Creating elapsed-time and observation-gap features.")

    timestamp = df[TIMESTAMP_COLUMN]

    elapsed = timestamp.diff().dt.total_seconds() / 60

    df["Minutes_Since_Previous_Record"] = elapsed

    df["Previous_Gap_15_Minutes"] = (
        elapsed == 15
    ).astype("int8")

    df["Previous_Gap_30_Minutes"] = (
        elapsed == 30
    ).astype("int8")

    df["Previous_Gap_60_Minutes"] = (
        elapsed == 60
    ).astype("int8")

    df["Previous_Gap_Over_60_Minutes"] = (
        elapsed > 60
    ).astype("int8")

    df["Elapsed_Hours_From_Start"] = (
        timestamp - timestamp.min()
    ).dt.total_seconds() / 3600

    return df


def add_row_lag_features(df):
    logging.info("Creating observed-row lag features.")

    for target in TARGET_COLUMNS:
        for lag in [1, 2, 4, 8, 12, 24]:
            df[f"{target}_Observed_Lag_{lag}"] = (
                df[target].shift(lag)
            )

    return df


def add_elapsed_time_lag_features(df):
    logging.info("Creating elapsed-time-based lag features.")

    indexed = df.set_index(TIMESTAMP_COLUMN)

    for target in TARGET_COLUMNS:
        for minutes in LAG_INTERVALS_MINUTES:
            lookup_index = (
                indexed.index
                - pd.Timedelta(minutes=minutes)
            )

            previous_values = indexed[target].reindex(
                lookup_index
            )

            df[
                f"{target}_Exact_Time_Lag_{minutes}m"
            ] = previous_values.to_numpy()

    return df


def add_rolling_features(df):
    logging.info("Creating historical rolling statistics.")

    for target in TARGET_COLUMNS:
        historical = df[target].shift(1)

        for window in ROLLING_WINDOWS:
            rolling = historical.rolling(
                window=window,
                min_periods=window,
            )

            df[
                f"{target}_Rolling_Mean_{window}"
            ] = rolling.mean()

            df[
                f"{target}_Rolling_Std_{window}"
            ] = rolling.std()

            df[
                f"{target}_Rolling_Min_{window}"
            ] = rolling.min()

            df[
                f"{target}_Rolling_Max_{window}"
            ] = rolling.max()

            df[
                f"{target}_Rolling_Median_{window}"
            ] = rolling.median()

    return df


def add_demand_change_features(df):
    logging.info("Creating historical demand-change features.")

    for target in TARGET_COLUMNS:
        previous = df[target].shift(1)

        df[
            f"{target}_Change_From_Previous_Record"
        ] = df[target] - previous

        df[
            f"{target}_Percentage_Change"
        ] = (
            (df[target] - previous)
            / previous.replace(0, np.nan)
        ) * 100

        df[
            f"{target}_Previous_4_Record_Mean"
        ] = df[target].shift(1).rolling(
            window=4,
            min_periods=4,
        ).mean()

        df[
            f"{target}_Previous_12_Record_Mean"
        ] = df[target].shift(1).rolling(
            window=12,
            min_periods=12,
        ).mean()

        df[
            f"{target}_Demand_Momentum"
        ] = (
            df[f"{target}_Previous_4_Record_Mean"]
            - df[f"{target}_Previous_12_Record_Mean"]
        )

        baseline = df[
            f"{target}_Rolling_Mean_12"
        ]

        variation = df[
            f"{target}_Rolling_Std_12"
        ]

        df[
            f"{target}_Historical_Anomaly_Score"
        ] = (
            (df[target] - baseline)
            / variation.replace(0, np.nan)
        )

    return df


def add_cross_target_features(df):
    logging.info("Creating sales-redemption relationship features.")

    df["Sales_Minus_Redemption"] = (
        df["Sales Count"] - df["Redemption Count"]
    )

    df["Sales_Redemption_Ratio"] = (
        df["Sales Count"]
        / df["Redemption Count"].replace(0, np.nan)
    )

    df["Combined_Observed_Count"] = (
        df["Sales Count"] + df["Redemption Count"]
    )

    df["Sales_Rolling_Mean_4"] = (
        df["Sales Count"].shift(1).rolling(
            4,
            min_periods=4,
        ).mean()
    )

    df["Redemption_Rolling_Mean_4"] = (
        df["Redemption Count"].shift(1).rolling(
            4,
            min_periods=4,
        ).mean()
    )

    return df


def create_feature_quality_report(df, feature_columns):
    logging.info("Generating feature-quality report.")

    report = {
        "input_file": str(INPUT_PATH),
        "output_records": int(len(df)),
        "total_columns": int(len(df.columns)),
        "feature_count_including_targets": int(
            len(feature_columns)
        ),
        "start_timestamp": str(df[TIMESTAMP_COLUMN].min()),
        "end_timestamp": str(df[TIMESTAMP_COLUMN].max()),
        "target_columns": TARGET_COLUMNS,
        "feature_columns": feature_columns,
        "missing_values_by_column": {
            column: int(count)
            for column, count in (
                df.isna().sum().items()
            )
            if count > 0
        },
        "infinite_values_by_column": {},
        "missing_data_policy": (
            "Lag and rolling features may be missing where "
            "historical observations are unavailable. "
            "These values are not automatically replaced with zero."
        ),
        "leakage_policy": (
            "Rolling statistics use shifted historical values. "
            "Current target values must not be included as model inputs "
            "when predicting that same target."
        ),
        "timestamp_policy": (
            "Irregular timestamps are preserved. Row lags represent "
            "previous observations, not guaranteed 15-minute intervals."
        ),
    }

    numeric = df.select_dtypes(include=[np.number])

    for column in numeric.columns:
        values = numeric[column].to_numpy(dtype=float)

        infinite_count = int(
            np.isinf(values).sum()
        )

        if infinite_count:
            report["infinite_values_by_column"][column] = (
                infinite_count
            )

    report_path = (
        REPORT_DIR / "feature_engineering_quality.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=4)

    return report


def main():
    setup_logging()

    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 65)
    logging.info("ADVANCED FEATURE ENGINEERING")
    logging.info("=" * 65)

    df = load_observed_series()

    df = add_calendar_features(df)
    df = add_elapsed_time_features(df)
    df = add_row_lag_features(df)
    df = add_elapsed_time_lag_features(df)
    df = add_rolling_features(df)
    df = add_demand_change_features(df)
    df = add_cross_target_features(df)

    # Remove infinities without inventing replacement demand values.
    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    feature_columns = [
        column
        for column in df.columns
        if column != TIMESTAMP_COLUMN
    ]

    output_path = (
        FORECAST_DIR / "engineered_features.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    report = create_feature_quality_report(
        df,
        feature_columns,
    )

    logging.info("-" * 65)
    logging.info("FEATURE ENGINEERING COMPLETED")
    logging.info("Records: %s", f"{len(df):,}")
    logging.info("Total columns: %s", len(df.columns))
    logging.info(
        "Generated feature columns: %s",
        len(feature_columns) - len(TARGET_COLUMNS),
    )
    logging.info(
        "Columns with missing values: %s",
        len(report["missing_values_by_column"]),
    )
    logging.info(
        "Output file: %s",
        output_path,
    )
    logging.info(
        "Quality report: %s",
        REPORT_DIR / "feature_engineering_quality.json",
    )
    logging.info("=" * 65)


if __name__ == "__main__":
    main()
