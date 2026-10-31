from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
FORECAST_DIR = OUTPUT_DIR / "forecasts"
REPORT_DIR = OUTPUT_DIR / "reports"
FIGURE_DIR = OUTPUT_DIR / "figures"

PREDICTIONS_PATH = FORECAST_DIR / "model_predictions.csv"
COMPARISON_PATH = REPORT_DIR / "model_comparison.csv"

TARGETS = ["Sales", "Redemption"]
HORIZONS = [15, 30, 60, 120]


def setup_logging():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_predictions():
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_PATH}. "
            "Run forecasting_models.py first."
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    required = {
        "Horizon_Minutes",
        "Target",
        "Model",
        "Actual",
        "Predicted",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Predictions file is missing columns: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    for column in ["Horizon_Minutes", "Actual", "Predicted"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna(
        subset=[
            "Horizon_Minutes",
            "Target",
            "Model",
            "Actual",
            "Predicted",
        ]
    ).copy()

    if df.empty:
        raise ValueError(
            "No valid prediction records remain after cleaning."
        )

    df["Horizon_Minutes"] = (
        df["Horizon_Minutes"].astype(int)
    )

    df["Absolute_Error"] = (
        df["Actual"] - df["Predicted"]
    ).abs()

    df["Squared_Error"] = (
        df["Actual"] - df["Predicted"]
    ) ** 2

    df["Signed_Error"] = (
        df["Predicted"] - df["Actual"]
    )

    logging.info(
        "Loaded %s prediction records.",
        f"{len(df):,}",
    )

    return df


def calculate_metrics(group):
    actual = group["Actual"].to_numpy(dtype=float)
    predicted = group["Predicted"].to_numpy(dtype=float)

    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        return {
            "Evaluation_Rows": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "WAPE_Percent": np.nan,
            "Bias": np.nan,
            "R2": np.nan,
        }

    errors = predicted - actual

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    denominator = float(np.abs(actual).sum())

    wape = (
        float(100 * np.abs(errors).sum() / denominator)
        if denominator > 0
        else np.nan
    )

    total_variation = float(
        np.sum((actual - np.mean(actual)) ** 2)
    )

    r2 = (
        float(
            1 - np.sum(errors ** 2) / total_variation
        )
        if total_variation > 0
        else np.nan
    )

    return {
        "Evaluation_Rows": int(len(actual)),
        "MAE": mae,
        "RMSE": rmse,
        "WAPE_Percent": wape,
        "Bias": float(np.mean(errors)),
        "R2": r2,
    }


def build_evaluation_table(df):
    rows = []

    grouping = ["Horizon_Minutes", "Target", "Model"]

    for keys, group in df.groupby(
        grouping,
        sort=True,
        dropna=False,
    ):
        horizon, target, model = keys

        metrics = calculate_metrics(group)

        rows.append({
            "Horizon_Minutes": int(horizon),
            "Target": str(target),
            "Model": str(model),
            **metrics,
        })

    result = pd.DataFrame(rows)

    if result.empty:
        raise ValueError(
            "Unable to calculate evaluation metrics."
        )

    result = result.sort_values(
        ["Horizon_Minutes", "Target", "WAPE_Percent"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)

    result["Rank_By_WAPE"] = (
        result.groupby(
            ["Horizon_Minutes", "Target"]
        )["WAPE_Percent"]
        .rank(method="min", ascending=True)
    )

    return result


def build_best_model_table(metrics):
    valid = metrics.dropna(
        subset=["WAPE_Percent"]
    ).copy()

    if valid.empty:
        return pd.DataFrame(
            columns=[
                "Horizon_Minutes",
                "Target",
                "Best_Model",
                "MAE",
                "RMSE",
                "WAPE_Percent",
                "R2",
            ]
        )

    valid = valid.sort_values(
        ["Horizon_Minutes", "Target", "WAPE_Percent"]
    )

    best = valid.groupby(
        ["Horizon_Minutes", "Target"],
        as_index=False,
    ).first()

    best = best.rename(
        columns={"Model": "Best_Model"}
    )

    columns = [
        "Horizon_Minutes",
        "Target",
        "Best_Model",
        "Evaluation_Rows",
        "MAE",
        "RMSE",
        "WAPE_Percent",
        "Bias",
        "R2",
    ]

    return best[columns]


def create_model_comparison_chart(metrics):
    plot_data = metrics.dropna(
        subset=["WAPE_Percent"]
    ).copy()

    if plot_data.empty:
        logging.warning(
            "No valid WAPE values available for plotting."
        )
        return

    for target in sorted(plot_data["Target"].unique()):
        subset = plot_data[
            plot_data["Target"] == target
        ]

        fig, ax = plt.subplots(figsize=(11, 6))

        for model in sorted(subset["Model"].unique()):
            model_data = subset[
                subset["Model"] == model
            ].sort_values("Horizon_Minutes")

            ax.plot(
                model_data["Horizon_Minutes"],
                model_data["WAPE_Percent"],
                marker="o",
                label=model,
            )

        ax.set_title(
            f"{target} Forecasting Error by Horizon"
        )
        ax.set_xlabel("Forecast Horizon (minutes)")
        ax.set_ylabel("WAPE (%)")
        ax.set_xticks(HORIZONS)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        safe_target = (
            "".join(
                character if character.isalnum() else "_"
                for character in target
            )
        )

        output_path = (
            FIGURE_DIR
            / f"model_evaluation_{safe_target.lower()}.png"
        )

        fig.savefig(output_path, dpi=160)
        plt.close(fig)

        logging.info(
            "Saved chart: %s",
            output_path,
        )


def create_actual_vs_predicted_chart(df, best_models):
    for _, row in best_models.iterrows():
        horizon = int(row["Horizon_Minutes"])
        target = str(row["Target"])
        model = str(row["Best_Model"])

        subset = df[
            (df["Horizon_Minutes"] == horizon)
            & (df["Target"] == target)
            & (df["Model"] == model)
        ].copy()

        if subset.empty:
            continue

        # Use a deterministic sample for a readable chart.
        if len(subset) > 2000:
            subset = subset.iloc[
                np.linspace(
                    0,
                    len(subset) - 1,
                    2000,
                    dtype=int,
                )
            ]

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(
            np.arange(len(subset)),
            subset["Actual"].to_numpy(),
            label="Actual",
            linewidth=1,
        )

        ax.plot(
            np.arange(len(subset)),
            subset["Predicted"].to_numpy(),
            label="Predicted",
            linewidth=1,
        )

        ax.set_title(
            f"{target}: Actual vs Predicted "
            f"({horizon} min, {model})"
        )
        ax.set_xlabel("Chronological Test Samples")
        ax.set_ylabel("Ticket Count")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        safe_target = (
            "".join(
                character if character.isalnum() else "_"
                for character in target
            ).lower()
        )

        output_path = (
            FIGURE_DIR
            / f"actual_vs_predicted_{safe_target}_{horizon}m.png"
        )

        fig.savefig(output_path, dpi=160)
        plt.close(fig)


def save_reports(df, metrics, best_models):
    metrics_path = REPORT_DIR / "model_evaluation_metrics.csv"
    best_path = REPORT_DIR / "best_model_by_horizon.csv"
    summary_path = REPORT_DIR / "model_evaluation_summary.json"

    metrics.to_csv(metrics_path, index=False)
    best_models.to_csv(best_path, index=False)

    summary = {
        "project": "Short-Term Ferry Ticket Demand Forecasting",
        "prediction_records_evaluated": int(len(df)),
        "models_evaluated": sorted(
            df["Model"].astype(str).unique().tolist()
        ),
        "targets_evaluated": sorted(
            df["Target"].astype(str).unique().tolist()
        ),
        "horizons_evaluated_minutes": sorted(
            df["Horizon_Minutes"].astype(int).unique().tolist()
        ),
        "metric_definitions": {
            "MAE": "Mean Absolute Error; lower is better.",
            "RMSE": "Root Mean Squared Error; lower is better.",
            "WAPE_Percent": (
                "Weighted Absolute Percentage Error; lower is better. "
                "Undefined when total absolute actual demand is zero."
            ),
            "Bias": (
                "Mean predicted minus actual demand. "
                "Positive means overprediction."
            ),
            "R2": (
                "Coefficient of determination; higher is generally better. "
                "Can be negative for poor predictions."
            ),
        },
        "selection_policy": (
            "Best model per horizon and target is selected by the lowest "
            "WAPE among the supplied prediction records."
        ),
        "limitations": [
            "This evaluation summarizes existing predictions; it does not retrain models.",
            "Comparisons are fair only when models use the same test timestamps and rows.",
            "Best model selection is based on the available evaluation results.",
            "Forecast accuracy does not establish ferry capacity or queue length.",
        ],
        "best_models": best_models.replace(
            {np.nan: None}
        ).to_dict(orient="records"),
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            allow_nan=False,
            default=str,
        )

    logging.info("Saved evaluation metrics: %s", metrics_path)
    logging.info("Saved best-model report: %s", best_path)
    logging.info("Saved evaluation summary: %s", summary_path)


def main():
    setup_logging()

    logging.info("=" * 70)
    logging.info("FERRY FORECAST MODEL EVALUATION")
    logging.info("=" * 70)

    df = load_predictions()

    metrics = build_evaluation_table(df)
    best_models = build_best_model_table(metrics)

    save_reports(
        df,
        metrics,
        best_models,
    )

    create_model_comparison_chart(metrics)
    create_actual_vs_predicted_chart(df, best_models)

    print("\\nMODEL EVALUATION SUMMARY")
    print("=" * 70)

    if best_models.empty:
        print("No best-model results could be calculated.")
    else:
        print(
            best_models[
                [
                    "Horizon_Minutes",
                    "Target",
                    "Best_Model",
                    "MAE",
                    "RMSE",
                    "WAPE_Percent",
                    "R2",
                ]
            ].to_string(index=False)
        )

    logging.info("=" * 70)
    logging.info("MODEL EVALUATION COMPLETED")
    logging.info("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception(
            "Model evaluation failed."
        )
        raise
