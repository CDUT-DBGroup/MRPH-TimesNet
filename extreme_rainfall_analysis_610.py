"""
Extreme rainfall event analysis for MRPH-TimesNet and comparison models.

This script focuses on:
1. Top 10% rainfall event definition on the test split
2. Event-window and peak-window performance
3. Flood-period error distribution
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.metrics import MAE, MSE, NSE, RMSE


OUTPUT_DIR = Path("results/dl_result/extreme_rainfall_analysis")
DEFAULT_DATA_PATH = Path("dataset/water_timeseries_610.csv")
DEFAULT_SUMMARY_CSV = Path("results/dl_result/res_all/dl_comparison_610_with_mrph_seed67_ranking_by_nse.csv")
DATE_COL = "date"
RAINFALL_COL = "daily_rainfall"
INFLOW_COL = "water_inflow"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extreme rainfall event analysis.")
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH), help="dataset csv path")
    parser.add_argument("--summary-csv", type=str, default=str(DEFAULT_SUMMARY_CSV), help="model summary csv path")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="output directory")
    parser.add_argument("--event-quantile", type=float, default=0.9, help="rainfall threshold quantile on test split")
    parser.add_argument("--merge-gap", type=int, default=1, help="merge trigger days within this gap")
    parser.add_argument("--response-window", type=int, default=30, help="response window after each extreme event")
    parser.add_argument("--peak-window-radius", type=int, default=1, help="peak window radius around true peak day")
    return parser.parse_args()


def style_for_publication() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "figure.titlesize": 13,
            "axes.linewidth": 0.8,
        }
    )


def save_figure(fig: plt.Figure, path: Path, dpi: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if dpi is None:
        fig.savefig(str(path), bbox_inches="tight")
    else:
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")


def load_dataset(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df


def get_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    num_test = int(len(df) * 0.1)
    test_start = len(df) - num_test
    test_df = df.iloc[test_start:].reset_index(drop=True).copy()
    return test_df, test_start


def compute_target_scaling(df: pd.DataFrame, target_col: str) -> tuple[float, float]:
    num_train = int(len(df) * 0.8)
    train_target = df.iloc[:num_train][target_col].to_numpy(dtype=float)
    mean = float(train_target.mean())
    scale = float(train_target.std(ddof=0))
    return mean, scale if scale > 1e-12 else 1.0


def inverse_target_transform(values: np.ndarray, mean: float, scale: float) -> np.ndarray:
    return np.asarray(values, dtype=float) * scale + mean


def load_model_predictions(summary_csv: Path) -> pd.DataFrame:
    summary_df = pd.read_csv(summary_csv)
    summary_df = summary_df[summary_df["status"] == "Success"].copy()
    return summary_df


def flatten_prediction(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def read_model_arrays(summary_df: pd.DataFrame, test_len: int) -> dict[str, dict[str, np.ndarray]]:
    model_outputs: dict[str, dict[str, np.ndarray]] = {}
    for _, row in summary_df.iterrows():
        model_name = str(row["model"])
        model_dir = Path(str(row["metrics_path"])).parent
        pred = flatten_prediction(np.load(model_dir / "pred.npy"))
        true = flatten_prediction(np.load(model_dir / "true.npy"))
        if len(pred) != test_len or len(true) != test_len:
            raise ValueError(f"{model_name} length mismatch: pred={len(pred)}, true={len(true)}, test={test_len}")
        model_outputs[model_name] = {"pred": pred, "true": true, "model_dir": str(model_dir)}
    return model_outputs


def detect_extreme_events(
    test_df: pd.DataFrame,
    quantile: float,
    merge_gap: int,
    response_window: int,
    peak_window_radius: int,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    threshold = float(test_df[RAINFALL_COL].quantile(quantile))
    trigger_indices = np.where(test_df[RAINFALL_COL].to_numpy() >= threshold)[0].tolist()
    if not trigger_indices:
        raise ValueError("No extreme rainfall events were found on the test split.")

    merged_events: list[tuple[int, int]] = []
    start = trigger_indices[0]
    end = trigger_indices[0]
    for idx in trigger_indices[1:]:
        if idx - end <= merge_gap:
            end = idx
        else:
            merged_events.append((start, end))
            start = idx
            end = idx
    merged_events.append((start, end))

    trigger_mask = np.zeros(len(test_df), dtype=bool)
    trigger_mask[trigger_indices] = True
    response_mask = np.zeros(len(test_df), dtype=bool)
    peak_mask = np.zeros(len(test_df), dtype=bool)
    event_rows = []

    inflow_values = test_df[INFLOW_COL].to_numpy(dtype=float)
    rainfall_values = test_df[RAINFALL_COL].to_numpy(dtype=float)
    for event_id, (trigger_start, trigger_end) in enumerate(merged_events, start=1):
        response_start = trigger_start
        response_end = min(len(test_df) - 1, trigger_end + response_window)
        response_mask[response_start : response_end + 1] = True

        local_true = inflow_values[response_start : response_end + 1]
        peak_local_idx = int(np.argmax(local_true))
        true_peak_idx = response_start + peak_local_idx
        peak_window_start = max(response_start, true_peak_idx - peak_window_radius)
        peak_window_end = min(response_end, true_peak_idx + peak_window_radius)
        peak_mask[peak_window_start : peak_window_end + 1] = True

        event_rows.append(
            {
                "event_id": event_id,
                "trigger_start_idx": trigger_start,
                "trigger_end_idx": trigger_end,
                "trigger_start_date": test_df.iloc[trigger_start][DATE_COL].strftime("%Y-%m-%d"),
                "trigger_end_date": test_df.iloc[trigger_end][DATE_COL].strftime("%Y-%m-%d"),
                "trigger_duration_days": trigger_end - trigger_start + 1,
                "response_start_idx": response_start,
                "response_end_idx": response_end,
                "response_start_date": test_df.iloc[response_start][DATE_COL].strftime("%Y-%m-%d"),
                "response_end_date": test_df.iloc[response_end][DATE_COL].strftime("%Y-%m-%d"),
                "response_duration_days": response_end - response_start + 1,
                "peak_rainfall": float(rainfall_values[trigger_start : trigger_end + 1].max()),
                "total_rainfall": float(rainfall_values[trigger_start : trigger_end + 1].sum()),
                "true_peak_idx": true_peak_idx,
                "true_peak_date": test_df.iloc[true_peak_idx][DATE_COL].strftime("%Y-%m-%d"),
                "true_peak_inflow_raw": float(inflow_values[true_peak_idx]),
                "peak_window_start_idx": peak_window_start,
                "peak_window_end_idx": peak_window_end,
            }
        )

    sample_flags = test_df[[DATE_COL, RAINFALL_COL, INFLOW_COL]].copy()
    sample_flags.insert(0, "test_sample_idx", np.arange(len(test_df)))
    sample_flags["is_top10_rainfall_day"] = trigger_mask
    sample_flags["is_event_window"] = response_mask
    sample_flags["is_peak_window"] = peak_mask
    return pd.DataFrame(event_rows), sample_flags, threshold


def compute_basic_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    return {
        "nse": float(NSE(pred, true)),
        "mse": float(MSE(pred, true)),
        "rmse": float(RMSE(pred, true)),
        "mae": float(MAE(pred, true)),
    }


def summarize_event_models(
    model_outputs: dict[str, dict[str, np.ndarray]],
    event_df: pd.DataFrame,
    sample_flags: pd.DataFrame,
    target_mean: float,
    target_scale: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event_indices = sample_flags.index[sample_flags["is_event_window"]].to_numpy()
    peak_indices = sample_flags.index[sample_flags["is_peak_window"]].to_numpy()

    summary_rows = []
    flood_rows = []
    peak_detail_rows = []

    for model_name, outputs in model_outputs.items():
        pred_std = outputs["pred"]
        true_std = outputs["true"]
        pred_raw = inverse_target_transform(pred_std, target_mean, target_scale)
        true_raw = inverse_target_transform(true_std, target_mean, target_scale)

        event_metrics = compute_basic_metrics(pred_std[event_indices], true_std[event_indices])
        peak_metrics = compute_basic_metrics(pred_std[peak_indices], true_std[peak_indices])

        peak_mag_errors = []
        peak_timing_errors = []
        for _, event in event_df.iterrows():
            start = int(event["response_start_idx"])
            end = int(event["response_end_idx"])
            true_window = true_raw[start : end + 1]
            pred_window = pred_raw[start : end + 1]
            true_peak_rel = int(np.argmax(true_window))
            pred_peak_rel = int(np.argmax(pred_window))
            true_peak_val = float(true_window[true_peak_rel])
            pred_peak_val = float(pred_window[pred_peak_rel])
            timing_error = pred_peak_rel - true_peak_rel
            mag_error_pct = 100.0 * (pred_peak_val - true_peak_val) / true_peak_val if abs(true_peak_val) > 1e-12 else np.nan

            peak_mag_errors.append(mag_error_pct)
            peak_timing_errors.append(timing_error)
            peak_detail_rows.append(
                {
                    "event_id": int(event["event_id"]),
                    "model": model_name,
                    "true_peak_date": event["true_peak_date"],
                    "pred_peak_date": sample_flags.iloc[start + pred_peak_rel][DATE_COL].strftime("%Y-%m-%d"),
                    "true_peak_inflow_raw": true_peak_val,
                    "pred_peak_inflow_raw": pred_peak_val,
                    "peak_magnitude_error_pct": float(mag_error_pct),
                    "peak_timing_error_days": int(timing_error),
                }
            )

        flood_residual_std = pred_std[event_indices] - true_std[event_indices]
        flood_rows.append(
            {
                "model": model_name,
                "event_window_sample_count": int(len(event_indices)),
                "median_error_std": float(np.median(flood_residual_std)),
                "iqr_error_std": float(np.percentile(flood_residual_std, 75) - np.percentile(flood_residual_std, 25)),
                "p90_abs_error_std": float(np.percentile(np.abs(flood_residual_std), 90)),
                "overestimation_rate": float(np.mean(flood_residual_std > 0)),
                "underestimation_rate": float(np.mean(flood_residual_std < 0)),
            }
        )

        summary_rows.append(
            {
                "model": model_name,
                "event_window_sample_count": int(len(event_indices)),
                "peak_window_sample_count": int(len(peak_indices)),
                "event_window_nse": event_metrics["nse"],
                "event_window_rmse": event_metrics["rmse"],
                "event_window_mae": event_metrics["mae"],
                "peak_inflow_nse": peak_metrics["nse"],
                "peak_window_rmse": peak_metrics["rmse"],
                "peak_window_mae": peak_metrics["mae"],
                "mean_peak_magnitude_error_pct": float(np.nanmean(peak_mag_errors)),
                "median_peak_magnitude_error_pct": float(np.nanmedian(peak_mag_errors)),
                "mean_abs_peak_timing_error_days": float(np.mean(np.abs(peak_timing_errors))),
                "mean_peak_timing_error_days": float(np.mean(peak_timing_errors)),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(by="event_window_nse", ascending=False).reset_index(drop=True)
    flood_df = pd.DataFrame(flood_rows).set_index("model").loc[summary_df["model"]].reset_index()
    peak_detail_df = pd.DataFrame(peak_detail_rows)
    return summary_df, flood_df, peak_detail_df


def plot_event_timeline(output_dir: Path, test_df: pd.DataFrame, sample_flags: pd.DataFrame, threshold: float) -> None:
    style_for_publication()
    x = test_df[DATE_COL].to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 5.8), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
    ax1, ax2 = axes

    ax1.bar(x, test_df[RAINFALL_COL].to_numpy(), color="#4C78A8", width=1.0, label="Daily rainfall")
    ax1.axhline(threshold, color="#d55e00", linestyle="--", linewidth=1.2, label="Top 10% threshold")
    ax1.scatter(
        test_df.loc[sample_flags["is_top10_rainfall_day"], DATE_COL],
        test_df.loc[sample_flags["is_top10_rainfall_day"], RAINFALL_COL],
        color="#b22222",
        s=20,
        zorder=5,
        label="Trigger day",
    )
    ax1.set_ylabel("Rainfall")
    ax1.legend(frameon=False, ncol=3, loc="upper right")
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

    ax2.plot(x, test_df[INFLOW_COL].to_numpy(), color="#1f4e79", linewidth=1.6, label="Observed inflow")
    in_event = sample_flags["is_event_window"].to_numpy()
    start = None
    for idx, flag in enumerate(in_event):
        if flag and start is None:
            start = idx
        if start is not None and (idx == len(in_event) - 1 or not in_event[idx + 1]):
            end = idx
            ax2.axvspan(x[start], x[end], alpha=0.18, color="#fdd0a2")
            start = None
    ax2.scatter(
        test_df.loc[sample_flags["is_peak_window"], DATE_COL],
        test_df.loc[sample_flags["is_peak_window"], INFLOW_COL],
        color="#d55e00",
        s=18,
        zorder=5,
        label="Peak window",
    )
    ax2.set_ylabel("Water inflow")
    ax2.set_xlabel("Date")
    ax2.legend(frameon=False, loc="upper right")
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

    fig.suptitle("Top 10% Rainfall Events and Corresponding Response Windows", y=1.01)
    fig.tight_layout()
    save_figure(fig, output_dir / "extreme_event_timeline.png", dpi=350)
    save_figure(fig, output_dir / "extreme_event_timeline.pdf")
    plt.close(fig)


def plot_event_performance(output_dir: Path, summary_df: pd.DataFrame) -> None:
    style_for_publication()
    display_df = summary_df.copy().iloc[::-1]
    models = display_df["model"].tolist()
    colors = ["#d55e00" if model == "MRPH-TimesNet" else "#4C78A8" for model in models]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), sharey=True)
    metric_specs = [
        ("event_window_nse", "(a) Event-window NSE"),
        ("peak_inflow_nse", "(b) Peak inflow NSE"),
    ]
    for ax, (column, title) in zip(axes, metric_specs):
        ax.barh(models, display_df[column].to_numpy(), color=colors, alpha=0.88)
        ax.set_title(title, pad=8)
        ax.set_xlabel("NSE")
        ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.35)

    fig.suptitle("Model Performance Under Extreme Rainfall Conditions", y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir / "extreme_event_performance.png", dpi=350)
    save_figure(fig, output_dir / "extreme_event_performance.pdf")
    plt.close(fig)


def plot_flood_error_distribution(output_dir: Path, summary_df: pd.DataFrame, flood_df: pd.DataFrame, model_outputs: dict[str, dict[str, np.ndarray]], sample_flags: pd.DataFrame) -> None:
    style_for_publication()
    model_order = summary_df["model"].tolist()
    event_indices = sample_flags.index[sample_flags["is_event_window"]].to_numpy()
    residuals = [model_outputs[model]["pred"][event_indices] - model_outputs[model]["true"][event_indices] for model in model_order]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    ax1, ax2 = axes
    bp = ax1.boxplot(residuals, patch_artist=True, vert=True, tick_labels=model_order, showfliers=False)
    for patch, model in zip(bp["boxes"], model_order):
        patch.set_facecolor("#d55e00" if model == "MRPH-TimesNet" else "#9ecae1")
        patch.set_alpha(0.85)
    ax1.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_title("(a) Flood-period residual distribution", pad=8)
    ax1.set_ylabel("Residual in standardized space")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)

    display_df = flood_df.iloc[::-1]
    colors = ["#d55e00" if model == "MRPH-TimesNet" else "#4C78A8" for model in display_df["model"]]
    ax2.barh(display_df["model"], display_df["p90_abs_error_std"], color=colors, alpha=0.88)
    ax2.set_title("(b) Tail error during floods", pad=8)
    ax2.set_xlabel("90th percentile absolute error")
    ax2.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.35)

    fig.suptitle("Error Distribution During Flood-Response Periods", y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir / "flood_error_distribution.png", dpi=350)
    save_figure(fig, output_dir / "flood_error_distribution.pdf")
    plt.close(fig)


def run_analysis() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(Path(args.data_path))
    test_df, test_start = get_test_split(df)
    target_mean, target_scale = compute_target_scaling(df, INFLOW_COL)

    summary_df = load_model_predictions(Path(args.summary_csv))
    model_outputs = read_model_arrays(summary_df, len(test_df))

    # Save explicit sample-date alignment so event subsets are reproducible.
    sample_alignment = test_df[[DATE_COL, RAINFALL_COL, INFLOW_COL]].copy()
    sample_alignment.insert(0, "global_row_idx", np.arange(test_start, len(df)))
    sample_alignment.insert(1, "test_sample_idx", np.arange(len(test_df)))
    sample_alignment.to_csv(output_dir / "test_sample_alignment.csv", index=False)

    event_df, sample_flags, threshold = detect_extreme_events(
        test_df=test_df,
        quantile=args.event_quantile,
        merge_gap=args.merge_gap,
        response_window=args.response_window,
        peak_window_radius=args.peak_window_radius,
    )
    event_df.to_csv(output_dir / "extreme_event_catalog.csv", index=False)
    sample_flags.to_csv(output_dir / "extreme_event_sample_flags.csv", index=False)

    summary_metrics_df, flood_df, peak_detail_df = summarize_event_models(
        model_outputs=model_outputs,
        event_df=event_df,
        sample_flags=sample_flags,
        target_mean=target_mean,
        target_scale=target_scale,
    )
    summary_metrics_df.to_csv(output_dir / "event_metrics_summary.csv", index=False)
    flood_df.to_csv(output_dir / "flood_error_distribution_summary.csv", index=False)
    peak_detail_df.to_csv(output_dir / "peak_event_details_by_model.csv", index=False)

    plot_event_timeline(output_dir, test_df, sample_flags, threshold)
    plot_event_performance(output_dir, summary_metrics_df)
    plot_flood_error_distribution(output_dir, summary_metrics_df, flood_df, model_outputs, sample_flags)

    report = {
        "data_path": str(Path(args.data_path)),
        "summary_csv": str(Path(args.summary_csv)),
        "event_quantile": args.event_quantile,
        "event_threshold": threshold,
        "merge_gap_days": args.merge_gap,
        "response_window_days": args.response_window,
        "peak_window_radius_days": args.peak_window_radius,
        "test_sample_count": int(len(test_df)),
        "event_count": int(len(event_df)),
        "event_window_sample_count": int(sample_flags["is_event_window"].sum()),
        "peak_window_sample_count": int(sample_flags["is_peak_window"].sum()),
    }
    (output_dir / "extreme_event_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved extreme rainfall analysis results to: {output_dir}")
    print(f"Top 10% rainfall threshold on test split: {threshold:.4f}")
    print(f"Extreme event count: {len(event_df)}")
    print(summary_metrics_df.to_string(index=False))
    print(flood_df.to_string(index=False))


if __name__ == "__main__":
    run_analysis()
