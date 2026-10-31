from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Toronto Island Ferry Tickets (1).csv"
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
PREPARED_DIR = OUTPUT_DIR / "forecasts"

TIMESTAMP_COLUMN = "Timestamp"
TARGET_COLUMNS = ["Sales Count", "Redemption Count"]

EXPECTED_INTERVAL = "15min"


def setup_logging():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                REPORT_DIR / "time_series_preparation.log",
                encoding="utf-8",
            ),
        ],
        force=True,
    )


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    logging.info("Loading raw ferry ticket dataset.")

    df = pd.read_csv(DATA_PATH)

    required_columns = [TIMESTAMP_COLUMN, *TARGET_COLUMNS]
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {missing_columns}"
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

    original_rows = len(df)

    df = df.dropna(
        subset=[TIMESTAMP_COLUMN, *TARGET_COLUMNS]
    ).copy()

    invalid_rows_removed = original_rows - len(df)

    negative_counts = {
        column: int((df[column] < 0).sum())
        for column in TARGET_COLUMNS
    }

    if any(negative_counts.values()):
        raise ValueError(
            "Negative ticket counts found. "
            f"Investigate before proceeding: {negative_counts}"
        )

    duplicate_timestamps = int(
        df[TIMESTAMP_COLUMN].duplicated().sum()
    )

    if duplicate_timestamps:
        raise ValueError(
            f"Found {duplicate_timestamps} duplicate timestamps. "
            "Resolve them before preparing the time series."
        )

    df = df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)

    logging.info("Valid rows loaded: %s", f"{len(df):,}")
    logging.info(
        "Invalid rows removed: %s",
        f"{invalid_rows_removed:,}",
    )

    return df, invalid_rows_removed


def analyze_time_gaps(df):
    logging.info("Analyzing timestamp intervals and gaps.")

    timestamps = df[TIMESTAMP_COLUMN]
    differences = timestamps.diff().dropna()

    expected_delta = pd.Timedelta(EXPECTED_INTERVAL)

    exact_interval_count = int(
        (differences == expected_delta).sum()
    )

    longer_gap_count = int(
        (differences > expected_delta).sum()
    )

    shorter_interval_count = int(
        (differences < expected_delta).sum()
    )

    gap_report = {
        "record_count": int(len(df)),
        "start_timestamp": str(timestamps.min()),
        "end_timestamp": str(timestamps.max()),
        "unique_timestamps": int(timestamps.nunique()),
        "expected_interval": EXPECTED_INTERVAL,
        "exact_15_minute_intervals": exact_interval_count,
        "intervals_longer_than_15_minutes": longer_gap_count,
        "intervals_shorter_than_15_minutes": shorter_interval_count,
        "median_interval_minutes": (
            float(differences.median().total_seconds() / 60)
            if not differences.empty
            else None
        ),
        "largest_interval_minutes": (
            float(differences.max().total_seconds() / 60)
            if not differences.empty
            else None
        ),
    }

    interval_counts = (
        differences.value_counts()
        .sort_values(ascending=False)
        .head(30)
    )

    interval_counts.index = interval_counts.index.astype(str)

    gap_report["most_common_intervals"] = {
        str(interval): int(count)
        for interval, count in interval_counts.items()
    }

    with open(
        REPORT_DIR / "timestamp_gap_report.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(gap_report, file, indent=4)

    pd.DataFrame(
        {
            TIMESTAMP_COLUMN: timestamps,
            "Minutes_Since_Previous_Record": (
                timestamps.diff().dt.total_seconds() / 60
            ),
        }
    ).to_csv(
        REPORT_DIR / "timestamp_interval_details.csv",
        index=False,
    )

    logging.info(
        "Exact 15-minute intervals: %s",
        f"{exact_interval_count:,}",
    )
    logging.info(
        "Longer intervals: %s",
        f"{longer_gap_count:,}",
    )
    logging.info(
        "Shorter intervals: %s",
        f"{shorter_interval_count:,}",
    )

    return gap_report


def prepare_observed_series(df):
    logging.info("Preparing chronological observed time series.")

    series = df[
        [TIMESTAMP_COLUMN, *TARGET_COLUMNS]
    ].copy()

    series = series.set_index(TIMESTAMP_COLUMN)
    series = series.sort_index()

    if not series.index.is_monotonic_increasing:
        raise ValueError(
            "Timestamps are not in chronological order."
        )

    if series.index.has_duplicates:
        raise ValueError(
            "Duplicate timestamps remain after cleaning."
        )

    # Preserve only observed records.
    # Do not fill missing timestamps with zero demand.
    series["Observed_Record"] = True

    for column in TARGET_COLUMNS:
        series[f"{column}_Rolling_Mean_4"] = (
            series[column]
            .rolling(window=4, min_periods=4)
            .mean()
        )

        series[f"{column}_Rolling_Mean_12"] = (
            series[column]
            .rolling(window=12, min_periods=12)
            .mean()
        )

        series[f"{column}_Rolling_Std_12"] = (
            series[column]
            .rolling(window=12, min_periods=12)
            .std()
        )

    series = series.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    output_path = PREPARED_DIR / "observed_time_series.csv"

    series.to_csv(output_path, index=True)

    logging.info(
        "Observed time series saved: %s",
        output_path,
    )

    return series


def create_time_quality_summary(df, series, gap_report):
    logging.info("Creating time-series quality summary.")

    summary = {
        "dataset_name": DATA_PATH.name,
        "raw_dataset_rows": int(
            len(pd.read_csv(DATA_PATH, usecols=[TIMESTAMP_COLUMN]))
        ),
        "prepared_observed_rows": int(len(series)),
        "prepared_columns": list(series.columns),
        "start_timestamp": str(series.index.min()),
        "end_timestamp": str(series.index.max()),
        "missing_values_in_source_targets": {
            column: int(df[column].isna().sum())
            for column in TARGET_COLUMNS
        },
        "negative_values_in_source_targets": {
            column: int((df[column] < 0).sum())
            for column in TARGET_COLUMNS
        },
        "timestamp_quality": gap_report,
        "missing_intervals_policy": (
            "Missing timestamps are preserved as gaps. "
            "They are not automatically interpreted as zero demand."
        ),
        "forecasting_note": (
            "Forecast horizons must be defined using elapsed time, "
            "not merely the next row, because timestamp intervals vary."
        ),
    }

    with open(
        REPORT_DIR / "time_series_quality_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    return summary


def main():
    setup_logging()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 65)
    logging.info("FERRY TIME-SERIES PREPARATION")
    logging.info("=" * 65)

    df, invalid_rows_removed = load_dataset()

    gap_report = analyze_time_gaps(df)

    series = prepare_observed_series(df)

    summary = create_time_quality_summary(
        df,
        series,
        gap_report,
    )

    summary["invalid_rows_removed"] = invalid_rows_removed

    with open(
        REPORT_DIR / "time_series_quality_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    logging.info("-" * 65)
    logging.info("PREPARATION COMPLETED")
    logging.info("Observed records: %s", f"{len(series):,}")
    logging.info("Prepared columns: %s", len(series.columns))
    logging.info(
        "Saved data: %s",
        PREPARED_DIR / "observed_time_series.csv",
    )
    logging.info(
        "Saved quality report: %s",
        REPORT_DIR / "time_series_quality_summary.json",
    )
    logging.info("=" * 65)


if __name__ == "__main__":
    main()
