
"""
Short-Term Ferry Ticket Demand Forecasting
===========================================
Trains and compares multiple regression models for:
    - 15-minute forecasting
    - 30-minute forecasting
    - 60-minute forecasting
    - 120-minute forecasting

Targets:
    - Future sales count
    - Future redemption count

Validation:
    - Chronological holdout
    - Temporal leakage safeguards
    - MAE, RMSE, WAPE, bias, and R2

Outputs:
    outputs/reports/model_comparison.csv
    outputs/reports/model_training_summary.json
    outputs/forecasts/model_predictions.csv
    models/*.joblib
"""

from __future__ import annotations

import json
import logging
import re
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "outputs" / "forecasts"
REPORT_DIR = BASE_DIR / "outputs" / "reports"
MODEL_DIR = BASE_DIR / "models"

HORIZONS = [15, 30, 60, 120]

TARGETS = {
    "Sales": [
        "Future_Sales Count",
        "Future_Sales_Count",
        "Future Sales Count",
    ],
    "Redemption": [
        "Future_Redemption Count",
        "Future_Redemption_Count",
        "Future Redemption Count",
    ],
}

TRAIN_FRACTION = 0.80
RANDOM_STATE = 42

# Random Forest is trained on a representative subset to control
# memory and runtime on a large dataset.
RF_MAX_TRAIN_ROWS = 60_000
RF_TREES = 80

# Enable or disable expensive models without modifying the workflow.
ENABLE_RANDOM_FOREST = True
ENABLE_HIST_GRADIENT_BOOSTING = True

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("ferry_forecasting")

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)


# ============================================================
# 2. DATA LOADING AND COLUMN IDENTIFICATION
# ============================================================

def load_forecasting_dataset(horizon: int) -> pd.DataFrame:
    """Load one horizon-specific forecasting dataset."""

    path = DATA_DIR / f"forecasting_dataset_{horizon}m.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Forecasting dataset not found: {path}\n"
            "Run forecasting_dataset.py first."
        )

    logger.info("Loading dataset: %s", path.name)

    df = pd.read_csv(path, low_memory=False)

    if df.empty:
        raise ValueError(f"Dataset is empty: {path.name}")

    # The prepared forecasting datasets use Forecast_Origin
    # rather than the original raw-data column name Timestamp.
    if "Forecast_Origin" not in df.columns:
        raise ValueError(
            f"'Forecast_Origin' column is missing from {path.name}."
        )

    if "Target_Timestamp" not in df.columns:
        raise ValueError(
            f"'Target_Timestamp' column is missing from {path.name}; "
            "future-target leakage cannot be checked safely."
        )

    df["Forecast_Origin"] = pd.to_datetime(
        df["Forecast_Origin"],
        errors="coerce",
    )

    df["Target_Timestamp"] = pd.to_datetime(
        df["Target_Timestamp"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["Forecast_Origin", "Target_Timestamp"]
    ).copy()

    # Preserve compatibility with the rest of the existing script.
    df["Timestamp"] = df["Forecast_Origin"]

    df = df.sort_values("Timestamp").reset_index(drop=True)

    logger.info(
        "Loaded %s rows and %s columns.",
        f"{len(df):,}",
        len(df.columns),
    )

    return df


def find_target_column(
    df: pd.DataFrame,
    target_type: str,
) -> str:
    """Find the target column without assuming one exact spelling."""

    candidates = TARGETS[target_type]

    for column in candidates:
        if column in df.columns:
            return column

    # Fallback for harmless differences in spacing or underscores.
    normalized = {
        re.sub(r"[^a-z0-9]", "", str(col).lower()): col
        for col in df.columns
    }

    if target_type == "Sales":
        possible_keys = [
            "futuresalescount",
            "futuresales",
        ]
    else:
        possible_keys = [
            "futureredemptioncount",
            "futureredemption",
        ]

    for key in possible_keys:
        if key in normalized:
            return normalized[key]

    raise KeyError(
        f"Could not identify the future {target_type} target.\n"
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# 3. LEAKAGE-SAFE FEATURE SELECTION
# ============================================================

def select_features(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Select numeric features available at forecast origin."""

    excluded_exact = {
        "timestamp",
        "date",
        "observed_record",
        "sales count",
        "redemption count",
        "sales",
        "redemption",
        "sales_count",
        "redemption_count",
        "_id",
    }

    # These features depend on current-row demand values.
    # They are excluded to prevent accidental target leakage.
    blocked_fragments = (
        "change_from_previous_record",
        "percentage_change",
        "pct_change",
        "historical_anomaly_score",
        "sales_minus_redemption",
        "sales_redemption_ratio",
        "combined_observed_count",
    )

    feature_columns = []

    for column in df.columns:
        name = str(column)
        lower = name.lower().strip()

        if column == target_column:
            continue

        if lower in excluded_exact:
            continue

        if lower.startswith((
            "future_",
            "target_",
            "actual_",
        )):
            continue

        if "target_timestamp" in lower:
            continue

        if "future_timestamp" in lower:
            continue

        if lower in {
            "target_time",
            "target_datetime",
            "forecast_target_time",
        }:
            continue

        if any(
            fragment in lower
            for fragment in blocked_fragments
        ):
            continue

        if not pd.api.types.is_numeric_dtype(df[column]):
            continue

        feature_columns.append(column)

    if not feature_columns:
        raise ValueError(
            "No usable numeric features were found."
        )

    X = df[feature_columns].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    usable_columns = [
        column
        for column in X.columns
        if X[column].notna().any()
    ]

    X = X[usable_columns]

    if X.empty or not usable_columns:
        raise ValueError(
            "No usable features remain after cleaning."
        )

    logger.info(
        "Selected %s numeric features.",
        len(usable_columns),
    )

    return X, usable_columns


# ============================================================
# 4. METRICS
# ============================================================

def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict:
    """Calculate robust regression metrics."""

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    valid = np.isfinite(actual) & np.isfinite(predicted)

    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        raise ValueError("No valid predictions available for metrics.")

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    denominator = np.abs(actual).sum()

    # WAPE remains defined when individual actual values are zero,
    # provided total absolute actual demand is positive.
    wape = (
        100.0 * np.abs(actual - predicted).sum() / denominator
        if denominator > 0
        else np.nan
    )

    bias = float(np.mean(predicted - actual))

    if len(actual) >= 2 and np.var(actual) > 0:
        r2 = r2_score(actual, predicted)
    else:
        r2 = np.nan

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "WAPE_Percent": float(wape),
        "Bias": bias,
        "R2": float(r2),
        "Evaluation_Rows": int(len(actual)),
    }


# ============================================================
# 5. MODEL DEFINITIONS
# ============================================================

def build_models() -> dict:
    """Create the candidate model pipelines."""

    models = {
        "Ridge Regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
    }

    if ENABLE_RANDOM_FOREST:
        models["Random Forest"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=RF_TREES,
                        max_depth=20,
                        min_samples_leaf=3,
                        max_features=0.8,
                        max_samples=0.8,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

    if ENABLE_HIST_GRADIENT_BOOSTING:
        models["Hist Gradient Boosting"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=120,
                        learning_rate=0.08,
                        max_leaf_nodes=31,
                        l2_regularization=1.0,
                        early_stopping=True,
                        validation_fraction=0.1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

    return models


# ============================================================
# 6. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

def chronological_split(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split by forecast-origin time.

    If a target timestamp is available, training examples whose
    target reaches into the test period are purged.
    """

    n_rows = len(df)

    if n_rows < 100:
        raise ValueError(
            f"Not enough rows for reliable evaluation: {n_rows}"
        )

    split_index = int(n_rows * TRAIN_FRACTION)
    split_index = min(max(split_index, 1), n_rows - 1)

    test_start_time = df.iloc[split_index]["Timestamp"]

    train_mask = np.arange(n_rows) < split_index
    test_mask = np.arange(n_rows) >= split_index

    # A training origin may be before the cutoff while its future
    # label occurs after the cutoff. Remove such rows if the
    # forecasting dataset contains a usable target timestamp.
    possible_target_time_columns = [
        column
        for column in df.columns
        if (
            "target_timestamp" in str(column).lower()
            or "future_timestamp" in str(column).lower()
            or str(column).lower() in {
                "target_time",
                "target_datetime",
                "forecast_target_time",
            }
        )
    ]

    if possible_target_time_columns:
        target_time_column = possible_target_time_columns[0]

        target_times = pd.to_datetime(
            df[target_time_column],
            errors="coerce",
        )

        valid_target_time = target_times.notna().to_numpy()

        purge_mask = (
            valid_target_time
            & (target_times >= test_start_time).to_numpy()
        )

        train_mask &= ~purge_mask

        logger.info(
            "Purged %s training rows whose target timestamps "
            "overlap the test period.",
            int(purge_mask.sum()),
        )

    train_indices = np.flatnonzero(train_mask)
    test_indices = np.flatnonzero(test_mask)

    if len(train_indices) == 0 or len(test_indices) == 0:
        raise ValueError(
            "Chronological split produced an empty train or test set."
        )

    logger.info(
        "Train rows: %s | Test rows: %s | Test begins: %s",
        f"{len(train_indices):,}",
        f"{len(test_indices):,}",
        test_start_time,
    )

    return train_indices, test_indices


# ============================================================
# 7. TRAIN ONE TARGET FOR ONE HORIZON
# ============================================================

def train_target(
    df: pd.DataFrame,
    horizon: int,
    target_type: str,
    target_column: str,
) -> tuple[list[dict], list[pd.DataFrame]]:
    """Train and evaluate all models for one target and horizon."""

    logger.info("-" * 70)
    logger.info(
        "HORIZON: %s minutes | TARGET: %s",
        horizon,
        target_type,
    )
    logger.info("-" * 70)

    X, feature_columns = select_features(
        df,
        target_column,
    )

    y = pd.to_numeric(
        df[target_column],
        errors="coerce",
    )

    valid_target = y.notna() & np.isfinite(y)

    working_df = df.loc[valid_target].reset_index(drop=True)
    X = X.loc[valid_target].reset_index(drop=True)
    y = y.loc[valid_target].reset_index(drop=True)

    if len(working_df) < 100:
        raise ValueError(
            f"Insufficient valid target rows for {horizon}m {target_type}."
        )

    train_indices, test_indices = chronological_split(
        working_df,
        target_column,
    )

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    timestamps_test = working_df.iloc[test_indices]["Timestamp"]

    model_results = []
    prediction_frames = []

    candidate_models = build_models()

    for model_name, model_template in candidate_models.items():
        logger.info("Training model: %s", model_name)

        started = time.perf_counter()

        model = clone(model_template)

        # Limit Random Forest training size to avoid excessive
        # memory usage while keeping the latest training period
        # represented in the training sample.
        if model_name == "Random Forest" and len(X_train) > RF_MAX_TRAIN_ROWS:
            sample_indices = np.linspace(
                0,
                len(X_train) - 1,
                num=RF_MAX_TRAIN_ROWS,
                dtype=int,
            )

            X_fit = X_train.iloc[sample_indices]
            y_fit = y_train.iloc[sample_indices]

            logger.info(
                "Random Forest training subset: %s rows.",
                f"{len(X_fit):,}",
            )
        else:
            X_fit = X_train
            y_fit = y_train

        model.fit(X_fit, y_fit)

        predictions = model.predict(X_test)

        elapsed_seconds = time.perf_counter() - started

        metrics = calculate_metrics(
            y_test.to_numpy(),
            predictions,
        )

        result = {
            "Horizon_Minutes": horizon,
            "Target": target_type,
            "Target_Column": target_column,
            "Model": model_name,
            "Train_Rows": int(len(X_fit)),
            "Available_Train_Rows": int(len(X_train)),
            "Test_Rows": int(len(X_test)),
            "Feature_Count": int(len(feature_columns)),
            "Training_Seconds": round(elapsed_seconds, 3),
            "Train_Start": str(
                working_df.iloc[train_indices[0]]["Timestamp"]
            ),
            "Train_End": str(
                working_df.iloc[train_indices[-1]]["Timestamp"]
            ),
            "Test_Start": str(timestamps_test.iloc[0]),
            "Test_End": str(timestamps_test.iloc[-1]),
            **metrics,
        }

        model_results.append(result)

        prediction_frames.append(
            pd.DataFrame(
                {
                    "Timestamp": timestamps_test.to_numpy(),
                    "Horizon_Minutes": horizon,
                    "Target": target_type,
                    "Model": model_name,
                    "Actual": y_test.to_numpy(),
                    "Predicted": predictions,
                    "Error": predictions - y_test.to_numpy(),
                    "Absolute_Error": np.abs(
                        predictions - y_test.to_numpy()
                    ),
                }
            )
        )

        # Save trained model with the exact feature ordering.
        safe_target = re.sub(
            r"[^A-Za-z0-9]+",
            "_",
            target_type,
        )

        safe_model = re.sub(
            r"[^A-Za-z0-9]+",
            "_",
            model_name,
        )

        model_path = MODEL_DIR / (
            f"{safe_target}_{horizon}m_{safe_model}.joblib"
        )

        joblib.dump(
            {
                "model": model,
                "feature_columns": feature_columns,
                "horizon_minutes": horizon,
                "target_type": target_type,
                "target_column": target_column,
                "model_name": model_name,
                "train_end": str(
                    working_df.iloc[train_indices[-1]]["Timestamp"]
                ),
            },
            model_path,
            compress=3,
        )

        logger.info(
            "%s | MAE: %.3f | RMSE: %.3f | WAPE: %.2f%% | "
            "Bias: %.3f | R2: %.4f | Time: %.1fs",
            model_name,
            metrics["MAE"],
            metrics["RMSE"],
            metrics["WAPE_Percent"],
            metrics["Bias"],
            metrics["R2"],
            elapsed_seconds,
        )

    return model_results, prediction_frames


# ============================================================
# 8. MAIN TRAINING WORKFLOW
# ============================================================

def main() -> None:
    logger.info("=" * 76)
    logger.info("FERRY DEMAND FORECASTING — MODEL TRAINING")
    logger.info("=" * 76)

    all_results = []
    all_predictions = []
    failures = []

    for horizon in HORIZONS:
        try:
            df = load_forecasting_dataset(horizon)

            for target_type in TARGETS:
                try:
                    target_column = find_target_column(
                        df,
                        target_type,
                    )

                    results, predictions = train_target(
                        df=df,
                        horizon=horizon,
                        target_type=target_type,
                        target_column=target_column,
                    )

                    all_results.extend(results)
                    all_predictions.extend(predictions)

                except Exception as exc:
                    logger.exception(
                        "Target training failed: %s | %sm",
                        target_type,
                        horizon,
                    )

                    failures.append(
                        {
                            "Horizon_Minutes": horizon,
                            "Target": target_type,
                            "Error": str(exc),
                        }
                    )

        except Exception as exc:
            logger.exception(
                "Could not process horizon: %sm",
                horizon,
            )

            failures.append(
                {
                    "Horizon_Minutes": horizon,
                    "Target": "ALL",
                    "Error": str(exc),
                }
            )

    if all_results:
        results_df = pd.DataFrame(all_results)

        results_df = results_df.sort_values(
            ["Horizon_Minutes", "Target", "WAPE_Percent"],
            ascending=[True, True, True],
        ).reset_index(drop=True)

        comparison_path = (
            REPORT_DIR / "model_comparison.csv"
        )

        results_df.to_csv(
            comparison_path,
            index=False,
        )

        logger.info(
            "Model comparison saved: %s",
            comparison_path,
        )

    if all_predictions:
        predictions_df = pd.concat(
            all_predictions,
            ignore_index=True,
        )

        predictions_path = (
            DATA_DIR / "model_predictions.csv"
        )

        predictions_df.to_csv(
            predictions_path,
            index=False,
        )

        logger.info(
            "Predictions saved: %s",
            predictions_path,
        )

    summary = {
        "project": "Short-Term Ferry Ticket Demand Forecasting",
        "training_completed_at": pd.Timestamp.now().isoformat(),
        "horizons_minutes": HORIZONS,
        "train_fraction": TRAIN_FRACTION,
        "random_state": RANDOM_STATE,
        "random_forest_max_training_rows": RF_MAX_TRAIN_ROWS,
        "successful_model_runs": len(all_results),
        "failed_runs": failures,
        "metrics": [
            "MAE",
            "RMSE",
            "WAPE_Percent",
            "Bias",
            "R2",
        ],
        "validation": (
            "Chronological holdout. Target-time purging is applied "
            "when a target timestamp column exists in the dataset."
        ),
    }

    summary_path = (
        REPORT_DIR / "model_training_summary.json"
    )

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

    logger.info("Summary saved: %s", summary_path)

    logger.info("=" * 76)
    logger.info("MODEL TRAINING FINISHED")
    logger.info("Successful model runs: %s", len(all_results))
    logger.info("Failed runs: %s", len(failures))
    logger.info("=" * 76)

    if failures:
        logger.warning(
            "Some model runs failed. Check the logs and "
            "model_training_summary.json before interpreting results."
        )

    if not all_results:
        raise RuntimeError(
            "No models completed successfully. Review the error logs."
        )


if __name__ == "__main__":
    main()

