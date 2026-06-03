"""
Physical consistency verification for MRPH-TimesNet.

This script integrates three parts:
1. Parameter stability / statistical interval analysis across many checkpoints
2. Sensitivity ranking based on existing OFAT sensitivity outputs
3. Physical-response consistency analysis for PCHTE-derived hydrological signals
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from models import MRPH_TimesNet


OUTPUT_DIR = Path("results/dl_result/physical_consistency_analysis")
DEFAULT_DATA_PATH = Path("dataset/water_timeseries_610.csv")
DEFAULT_SEED_CSV = Path("results/dl_result/res_all/mrph_seed_search_mse_lt_010.csv")
DEFAULT_SENSITIVITY_DIR = Path("results/dl_result/res_all/sensitivity_analysis")
DEFAULT_HYDRO_DIR = Path("results/dl_result/hydro_mechanism_analysis")
DEFAULT_MAIN_CHECKPOINT = Path(
    "checkpoints/water_MRPH_TimesNet_610_MRPH_TimesNet_sl24_pl1_Comparison_Exp_0/checkpoint.pth"
)

DATE_COL = "date"
RAINFALL_COL = "daily_rainfall"
WATER_LEVEL_COL = "avg_water_level"
INFLOW_COL = "water_inflow"
MSRLA_SCALES = [3, 7, 15, 30]
PARAMETER_ORDER = ["alpha", "beta", "gamma", "phys_gate", "msrla_gate"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Physical consistency verification for MRPH-TimesNet.")
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH), help="dataset csv path")
    parser.add_argument("--seed-csv", type=str, default=str(DEFAULT_SEED_CSV), help="seed search summary csv")
    parser.add_argument("--sensitivity-dir", type=str, default=str(DEFAULT_SENSITIVITY_DIR), help="sensitivity output directory")
    parser.add_argument("--hydro-dir", type=str, default=str(DEFAULT_HYDRO_DIR), help="hydro mechanism output directory")
    parser.add_argument("--main-checkpoint", type=str, default=str(DEFAULT_MAIN_CHECKPOINT), help="main MRPH checkpoint")
    parser.add_argument("--checkpoints-root", type=str, default="checkpoints", help="checkpoint root directory")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="output directory")
    parser.add_argument("--max-lag", type=int, default=30, help="maximum lag for consistency analysis")
    parser.add_argument("--event-quantile", type=float, default=0.9, help="rainfall quantile for event extraction")
    parser.add_argument("--merge-gap", type=int, default=1, help="merge extreme events within this gap")
    parser.add_argument("--baseline-window", type=int, default=3, help="pre-event baseline window")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
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


def normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if all(key.startswith("module.") for key in state_dict):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def build_model_args() -> argparse.Namespace:
    return argparse.Namespace(
        task_name="long_term_forecast",
        model="MRPH_TimesNet",
        data="custom",
        root_path="./dataset/",
        data_path="water_timeseries_610.csv",
        features="MS",
        target="water_inflow",
        freq="d",
        checkpoints="./checkpoints/",
        seq_len=24,
        label_len=1,
        pred_len=1,
        seasonal_patterns="Monthly",
        inverse=False,
        expand=2,
        d_conv=4,
        top_k=5,
        num_kernels=6,
        enc_in=7,
        dec_in=7,
        c_out=1,
        d_model=32,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=32,
        moving_avg=25,
        factor=3,
        distil=True,
        dropout=0.1,
        embed="timeF",
        activation="gelu",
        output_attention=False,
        use_gpu=torch.cuda.is_available(),
        use_multi_gpu=False,
        gpu=0,
    )


def load_model(checkpoint_path: Path, gpu: int) -> torch.nn.Module:
    model_args = build_model_args()
    model_args.gpu = gpu
    device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    model = MRPH_TimesNet.Model(model_args).float().to(device)
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(normalize_state_dict(state_dict))
    model.eval()
    return model


def extract_parameter_record(model: torch.nn.Module, seed: int, checkpoint_path: Path) -> dict[str, float | int | str]:
    return {
        "seed": int(seed),
        "checkpoint_path": str(checkpoint_path),
        "alpha": float(model.pchte.alpha.item()),
        "beta": float(model.pchte.beta.item()),
        "gamma": float(model.pchte.gamma.item()),
        "phys_gate": float(model.phys_gate.item()),
        "msrla_gate": float(model.msrla_gate.item()),
    }


def collect_checkpoint_paths(seed_csv: Path, checkpoints_root: Path, main_checkpoint: Path) -> list[tuple[int, Path]]:
    seed_df = pd.read_csv(seed_csv)
    seeds = [int(seed) for seed in seed_df["seed"].tolist()]
    checkpoint_pairs: list[tuple[int, Path]] = []
    for seed in seeds:
        seed_ckpt = checkpoints_root / f"water_MRPH_TimesNet_610_MRPH_TimesNet_sl24_pl1_SeedSearch_{seed}_0" / "checkpoint.pth"
        if seed_ckpt.exists():
            checkpoint_pairs.append((seed, seed_ckpt))
        elif seed == 67 and main_checkpoint.exists():
            checkpoint_pairs.append((seed, main_checkpoint))
    if not checkpoint_pairs and main_checkpoint.exists():
        checkpoint_pairs.append((67, main_checkpoint))
    return checkpoint_pairs


def summarize_parameter_intervals(parameter_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for parameter in PARAMETER_ORDER:
        values = parameter_df[parameter].to_numpy(dtype=float)
        summary_rows.append(
            {
                "parameter": parameter,
                "count": int(len(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "median": float(np.median(values)),
                "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "ci95_low": float(np.percentile(values, 2.5)),
                "ci95_high": float(np.percentile(values, 97.5)),
            }
        )

    beta_values = parameter_df["beta"].to_numpy(dtype=float)
    characteristic_lag = 1.0 / np.maximum(beta_values, 1e-8)
    half_life = np.log(2.0) / np.maximum(beta_values, 1e-8)
    summary_rows.extend(
        [
            {
                "parameter": "beta_characteristic_lag",
                "count": int(len(characteristic_lag)),
                "mean": float(np.mean(characteristic_lag)),
                "std": float(np.std(characteristic_lag, ddof=0)),
                "median": float(np.median(characteristic_lag)),
                "iqr": float(np.percentile(characteristic_lag, 75) - np.percentile(characteristic_lag, 25)),
                "min": float(np.min(characteristic_lag)),
                "max": float(np.max(characteristic_lag)),
                "ci95_low": float(np.percentile(characteristic_lag, 2.5)),
                "ci95_high": float(np.percentile(characteristic_lag, 97.5)),
            },
            {
                "parameter": "beta_half_life",
                "count": int(len(half_life)),
                "mean": float(np.mean(half_life)),
                "std": float(np.std(half_life, ddof=0)),
                "median": float(np.median(half_life)),
                "iqr": float(np.percentile(half_life, 75) - np.percentile(half_life, 25)),
                "min": float(np.min(half_life)),
                "max": float(np.max(half_life)),
                "ci95_low": float(np.percentile(half_life, 2.5)),
                "ci95_high": float(np.percentile(half_life, 97.5)),
            },
        ]
    )
    return pd.DataFrame(summary_rows)


def plot_parameter_intervals(output_dir: Path, parameter_df: pd.DataFrame) -> None:
    style_for_publication()
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    groups = [
        (["alpha", "beta", "gamma"], "(a) Physical parameter intervals"),
        (["phys_gate", "msrla_gate"], "(b) Fusion gate intervals"),
    ]
    for ax, (cols, title) in zip(axes, groups):
        data = [parameter_df[col].to_numpy(dtype=float) for col in cols]
        bp = ax.boxplot(data, patch_artist=True, tick_labels=cols, showfliers=False)
        for patch, col in zip(bp["boxes"], cols):
            patch.set_facecolor("#d55e00" if col in {"alpha", "beta", "gamma"} else "#4C78A8")
            patch.set_alpha(0.82)
        ax.set_title(title, pad=8)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    fig.suptitle("Statistical Interval Analysis of Learned Physical Parameters", y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir / "physical_parameter_intervals.png", dpi=350)
    save_figure(fig, output_dir / "physical_parameter_intervals.pdf")
    plt.close(fig)


def plot_beta_timescales(output_dir: Path, parameter_df: pd.DataFrame) -> None:
    style_for_publication()
    beta_values = parameter_df["beta"].to_numpy(dtype=float)
    characteristic_lag = 1.0 / np.maximum(beta_values, 1e-8)
    half_life = np.log(2.0) / np.maximum(beta_values, 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    for ax, values, title, color in zip(
        axes,
        [characteristic_lag, half_life],
        ["(a) Characteristic lag from beta", "(b) Half-life from beta"],
        ["#1f4e79", "#d55e00"],
    ):
        ax.hist(values, bins=min(10, max(5, len(values) // 2)), color=color, alpha=0.82, edgecolor="white")
        ax.axvline(np.median(values), linestyle="--", linewidth=1.1, color="black")
        ax.set_title(title, pad=8)
        ax.set_xlabel("Days")
        ax.set_ylabel("Checkpoint count")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    fig.suptitle("Derived Hydrological Time Scales from the Learned Decay Parameter", y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir / "beta_timescale_distribution.png", dpi=350)
    save_figure(fig, output_dir / "beta_timescale_distribution.pdf")
    plt.close(fig)


def compute_sensitivity_ranking(sensitivity_csv: Path, sensitivity_summary_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_df = pd.read_csv(sensitivity_csv)
    summary_df = pd.read_csv(sensitivity_summary_csv)

    ranking_rows = []
    for parameter in PARAMETER_ORDER:
        subset = detail_df[detail_df["parameter"] == parameter].sort_values("value").reset_index(drop=True)
        baseline_idx = int(subset.index[subset["is_baseline"] == 1][0])
        baseline = subset.iloc[baseline_idx]
        low_idx = max(0, baseline_idx - 1)
        high_idx = min(len(subset) - 1, baseline_idx + 1)
        low = subset.iloc[low_idx]
        high = subset.iloc[high_idx]

        delta_value = float(high["value"] - low["value"])
        if abs(delta_value) <= 1e-12:
            local_nse_slope = 0.0
            local_rmse_slope = 0.0
        else:
            local_nse_slope = float((high["test_nse"] - low["test_nse"]) / delta_value)
            local_rmse_slope = float((high["test_rmse"] - low["test_rmse"]) / delta_value)

        normalized_index = abs(local_nse_slope * float(baseline["value"]) / max(abs(float(baseline["test_nse"])), 1e-8))
        summary_row = summary_df.loc[summary_df["parameter"] == parameter].iloc[0]
        ranking_rows.append(
            {
                "parameter": parameter,
                "group": summary_row["group"],
                "module": summary_row["module"],
                "baseline_value": float(baseline["value"]),
                "baseline_test_nse": float(baseline["test_nse"]),
                "test_nse_range": float(summary_row["test_nse_range"]),
                "local_nse_slope": local_nse_slope,
                "local_rmse_slope": local_rmse_slope,
                "normalized_sensitivity_index": float(normalized_index),
            }
        )

    ranking_df = pd.DataFrame(ranking_rows).sort_values(by="normalized_sensitivity_index", ascending=False).reset_index(drop=True)
    return detail_df, ranking_df


def plot_sensitivity_ranking(output_dir: Path, ranking_df: pd.DataFrame) -> None:
    style_for_publication()
    display_df = ranking_df.iloc[::-1]
    colors = ["#d55e00" if group == "physical" else "#4C78A8" for group in display_df["group"]]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    ax1, ax2 = axes

    ax1.barh(display_df["parameter"], display_df["normalized_sensitivity_index"], color=colors, alpha=0.86)
    ax1.set_title("(a) Normalized sensitivity ranking", pad=8)
    ax1.set_xlabel("Normalized sensitivity index")
    ax1.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.35)

    ax2.barh(display_df["parameter"], display_df["test_nse_range"], color=colors, alpha=0.86)
    ax2.set_title("(b) NSE fluctuation range", pad=8)
    ax2.set_xlabel("Test NSE range")
    ax2.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.35)

    fig.suptitle("Parameter Sensitivity Verification of MRPH-TimesNet", y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir / "physical_sensitivity_ranking.png", dpi=350)
    save_figure(fig, output_dir / "physical_sensitivity_ranking.pdf")
    plt.close(fig)


def standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    std = values.std(ddof=0)
    if std <= 1e-12:
        return values * 0.0
    return (values - values.mean()) / std


def compute_ccf(x: np.ndarray, y: np.ndarray, max_lag: int) -> np.ndarray:
    x = standardize(x)
    y = standardize(y)
    corrs = []
    for lag in range(max_lag + 1):
        if lag == 0:
            x_lag = x
            y_lag = y
        else:
            x_lag = x[:-lag]
            y_lag = y[lag:]
        corrs.append(float(np.corrcoef(x_lag, y_lag)[0, 1]))
    return np.asarray(corrs)


def summarize_ccf(label: str, x: np.ndarray, y: np.ndarray, max_lag: int) -> dict[str, float | str]:
    lags = np.arange(0, max_lag + 1)
    corrs = compute_ccf(x, y, max_lag)
    best_idx = int(np.nanargmax(corrs))
    return {
        "pair": label,
        "best_lag_days": int(lags[best_idx]),
        "peak_correlation": float(corrs[best_idx]),
    }


def detect_rainfall_events(rainfall: np.ndarray, quantile: float, merge_gap: int) -> tuple[list[tuple[int, int]], float]:
    rainfall = np.asarray(rainfall, dtype=float)
    positive = rainfall[rainfall > 0]
    threshold = float(np.quantile(positive, quantile)) if len(positive) else 0.0
    indices = np.where(rainfall >= threshold)[0].tolist()
    if not indices:
        return [], threshold

    events: list[tuple[int, int]] = []
    start = indices[0]
    end = indices[0]
    for idx in indices[1:]:
        if idx - end <= merge_gap:
            end = idx
        else:
            events.append((start, end))
            start = idx
            end = idx
    events.append((start, end))
    return events, threshold


def response_peak(series: np.ndarray, event_start: int, event_end: int, max_lag: int, baseline_window: int) -> tuple[int, float]:
    baseline_end = max(0, event_start - 1)
    baseline_start = max(0, baseline_end - baseline_window + 1)
    if baseline_end >= baseline_start:
        baseline = float(np.mean(series[baseline_start : baseline_end + 1]))
    else:
        baseline = float(series[event_start])
    window_end = min(len(series) - 1, event_end + max_lag)
    response = series[event_end : window_end + 1] - baseline
    best_idx = int(np.argmax(response))
    return best_idx, float(response[best_idx])


def compute_pchte_signals(df: pd.DataFrame, params: dict[str, float], max_lag: int) -> pd.DataFrame:
    rainfall = df[RAINFALL_COL].to_numpy(dtype=float)
    water_level = df[WATER_LEVEL_COL].to_numpy(dtype=float)
    tau = np.arange(max_lag, dtype=float)
    kernel = params["alpha"] * np.exp(-params["beta"] * tau)
    infiltration = np.convolve(rainfall, kernel, mode="full")[: len(rainfall)]
    delta_h = np.zeros_like(water_level)
    delta_h[1:] = water_level[1:] - water_level[:-1]
    q_phys = params["gamma"] * infiltration + (1.0 - params["gamma"]) * delta_h
    return pd.DataFrame(
        {
            DATE_COL: df[DATE_COL],
            RAINFALL_COL: rainfall,
            WATER_LEVEL_COL: water_level,
            INFLOW_COL: df[INFLOW_COL].to_numpy(dtype=float),
            "I_t": infiltration,
            "delta_H_t": delta_h,
            "Q_phys_t": q_phys,
        }
    )


def build_physical_response_summary(
    signal_df: pd.DataFrame,
    hydro_dir: Path,
    max_lag: int,
    event_quantile: float,
    merge_gap: int,
    baseline_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    inflow = signal_df[INFLOW_COL].to_numpy(dtype=float)
    response_rows = [
        summarize_ccf("Rainfall -> Inflow", signal_df[RAINFALL_COL].to_numpy(dtype=float), inflow, max_lag),
        summarize_ccf("Infiltration I(t) -> Inflow", signal_df["I_t"].to_numpy(dtype=float), inflow, max_lag),
        summarize_ccf("delta_H(t) -> Inflow", signal_df["delta_H_t"].to_numpy(dtype=float), inflow, max_lag),
        summarize_ccf("Q_phys(t) -> Inflow", signal_df["Q_phys_t"].to_numpy(dtype=float), inflow, max_lag),
    ]
    response_df = pd.DataFrame(response_rows)

    hydro_ccf = hydro_dir / "ccf_summary.csv"
    hydro_lag = hydro_dir / "lag_distribution_summary.csv"
    empirical_peak_lag = float("nan")
    empirical_event_median = float("nan")
    if hydro_ccf.exists():
        hydro_ccf_df = pd.read_csv(hydro_ccf)
        row = hydro_ccf_df.loc[hydro_ccf_df["pair"] == "Rainfall -> Inflow"]
        if not row.empty:
            empirical_peak_lag = float(row.iloc[0]["best_lag_days"])
    if hydro_lag.exists():
        hydro_lag_df = pd.read_csv(hydro_lag)
        row = hydro_lag_df.loc[hydro_lag_df["target"] == "Inflow"]
        if not row.empty:
            empirical_event_median = float(row.iloc[0]["median_lag_days"])

    events, threshold = detect_rainfall_events(signal_df[RAINFALL_COL].to_numpy(dtype=float), event_quantile, merge_gap)
    event_rows = []
    for event_id, (start, end) in enumerate(events, start=1):
        inflow_lag, inflow_peak = response_peak(inflow, start, end, max_lag, baseline_window)
        qphys_lag, qphys_peak = response_peak(signal_df["Q_phys_t"].to_numpy(dtype=float), start, end, max_lag, baseline_window)
        event_rows.append(
            {
                "event_id": event_id,
                "start_date": signal_df.iloc[start][DATE_COL].strftime("%Y-%m-%d"),
                "end_date": signal_df.iloc[end][DATE_COL].strftime("%Y-%m-%d"),
                "rainfall_peak": float(signal_df.iloc[start : end + 1][RAINFALL_COL].max()),
                "true_inflow_peak_lag_days": int(inflow_lag),
                "q_phys_peak_lag_days": int(qphys_lag),
                "peak_lag_gap_days": int(qphys_lag - inflow_lag),
                "true_inflow_peak_response": float(inflow_peak),
                "q_phys_peak_response": float(qphys_peak),
            }
        )
    event_alignment_df = pd.DataFrame(event_rows)
    aggregate = {
        "event_threshold_rainfall": threshold,
        "event_count": int(len(event_alignment_df)),
        "mean_abs_qphys_peak_gap_days": float(np.mean(np.abs(event_alignment_df["peak_lag_gap_days"]))) if len(event_alignment_df) else float("nan"),
        "median_abs_qphys_peak_gap_days": float(np.median(np.abs(event_alignment_df["peak_lag_gap_days"]))) if len(event_alignment_df) else float("nan"),
        "empirical_rainfall_inflow_peak_lag_days": empirical_peak_lag,
        "empirical_event_median_inflow_lag_days": empirical_event_median,
    }
    return response_df, event_alignment_df, aggregate


def plot_physical_response(output_dir: Path, signal_df: pd.DataFrame, response_df: pd.DataFrame, params: dict[str, float], aggregate: dict[str, float]) -> None:
    style_for_publication()
    peak_event_idx = int(np.argmax(signal_df[RAINFALL_COL].to_numpy(dtype=float)))
    start = max(0, peak_event_idx - 5)
    end = min(len(signal_df) - 1, peak_event_idx + 35)
    window = signal_df.iloc[start : end + 1].copy()
    x = window[DATE_COL].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    ax1, ax2 = axes
    ax1.plot(x, standardize(window[INFLOW_COL].to_numpy(dtype=float)), linewidth=1.8, color="#1f4e79", label="Observed inflow")
    ax1.plot(x, standardize(window["Q_phys_t"].to_numpy(dtype=float)), linewidth=1.7, color="#d55e00", label="Q_phys")
    ax1.plot(x, standardize(window["I_t"].to_numpy(dtype=float)), linewidth=1.4, color="#2ca02c", label="I(t)")
    ax1.plot(x, standardize(window["delta_H_t"].to_numpy(dtype=float)), linewidth=1.4, color="#9467bd", label="delta_H(t)")
    ax1.set_title("(a) Physical-response traces around a major rainfall event", pad=8)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Standardized amplitude")
    ax1.legend(frameon=False, loc="upper right")
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

    labels = [
        "1/beta",
        "ln(2)/beta",
        "Rain->Inflow\nCCF lag",
        "Event median\nlag",
        "Qphys->Inflow\nCCF lag",
    ]
    values = [
        1.0 / max(params["beta"], 1e-8),
        math.log(2.0) / max(params["beta"], 1e-8),
        float(aggregate["empirical_rainfall_inflow_peak_lag_days"]),
        float(aggregate["empirical_event_median_inflow_lag_days"]),
        float(response_df.loc[response_df["pair"] == "Q_phys(t) -> Inflow", "best_lag_days"].iloc[0]),
    ]
    ax2.bar(labels, values, color=["#d55e00", "#fdae6b", "#4C78A8", "#9ecae1", "#2ca02c"], alpha=0.86)
    ax2.set_title("(b) Learned time scales vs empirical lag evidence", pad=8)
    ax2.set_ylabel("Days")
    ax2.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)

    fig.suptitle("Physical-Response Consistency of the Learned MRPH Branch", y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir / "physical_response_consistency.png", dpi=350)
    save_figure(fig, output_dir / "physical_response_consistency.pdf")
    plt.close(fig)


def run_analysis() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_csv = Path(args.seed_csv)
    checkpoints_root = Path(args.checkpoints_root)
    main_checkpoint = Path(args.main_checkpoint)
    sensitivity_dir = Path(args.sensitivity_dir)
    hydro_dir = Path(args.hydro_dir)
    data_path = Path(args.data_path)

    checkpoint_pairs = collect_checkpoint_paths(seed_csv, checkpoints_root, main_checkpoint)
    if not checkpoint_pairs:
        raise FileNotFoundError("No valid MRPH checkpoints were found for interval analysis.")

    parameter_rows = []
    for seed, checkpoint_path in checkpoint_pairs:
        model = load_model(checkpoint_path, args.gpu)
        parameter_rows.append(extract_parameter_record(model, seed, checkpoint_path))

    parameter_df = pd.DataFrame(parameter_rows).sort_values(by="seed").reset_index(drop=True)
    parameter_df["beta_characteristic_lag"] = 1.0 / np.maximum(parameter_df["beta"].to_numpy(dtype=float), 1e-8)
    parameter_df["beta_half_life"] = np.log(2.0) / np.maximum(parameter_df["beta"].to_numpy(dtype=float), 1e-8)
    parameter_df.to_csv(output_dir / "parameter_seed_values.csv", index=False)

    interval_df = summarize_parameter_intervals(parameter_df)
    interval_df.to_csv(output_dir / "parameter_interval_summary.csv", index=False)
    plot_parameter_intervals(output_dir, parameter_df)
    plot_beta_timescales(output_dir, parameter_df)

    sensitivity_csv = sensitivity_dir / "mrph_parameter_sensitivity.csv"
    sensitivity_summary_csv = sensitivity_dir / "mrph_parameter_sensitivity_summary.csv"
    _, ranking_df = compute_sensitivity_ranking(sensitivity_csv, sensitivity_summary_csv)
    ranking_df.to_csv(output_dir / "parameter_sensitivity_ranking.csv", index=False)
    plot_sensitivity_ranking(output_dir, ranking_df)

    main_model = load_model(main_checkpoint, args.gpu)
    main_params = extract_parameter_record(main_model, 67, main_checkpoint)
    signal_df = compute_pchte_signals(load_dataset(data_path), main_params, args.max_lag)
    signal_df.to_csv(output_dir / "physical_response_series.csv", index=False)

    response_df, event_alignment_df, aggregate = build_physical_response_summary(
        signal_df=signal_df,
        hydro_dir=hydro_dir,
        max_lag=args.max_lag,
        event_quantile=args.event_quantile,
        merge_gap=args.merge_gap,
        baseline_window=args.baseline_window,
    )
    response_df.to_csv(output_dir / "physical_response_summary.csv", index=False)
    event_alignment_df.to_csv(output_dir / "physical_response_event_alignment.csv", index=False)
    plot_physical_response(output_dir, signal_df, response_df, main_params, aggregate)

    report = {
        "checkpoint_count_for_interval_analysis": int(len(parameter_df)),
        "seeds_used": [int(seed) for seed in parameter_df["seed"].tolist()],
        "main_checkpoint": str(main_checkpoint),
        "max_lag_days": args.max_lag,
        "event_quantile": args.event_quantile,
        "parameter_interval_summary_path": str(output_dir / "parameter_interval_summary.csv"),
        "parameter_sensitivity_ranking_path": str(output_dir / "parameter_sensitivity_ranking.csv"),
        "physical_response_summary_path": str(output_dir / "physical_response_summary.csv"),
        "event_alignment_summary": aggregate,
    }
    (output_dir / "physical_consistency_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved physical consistency results to: {output_dir}")
    print(interval_df.to_string(index=False))
    print(ranking_df.to_string(index=False))
    print(response_df.to_string(index=False))
    print(pd.DataFrame([aggregate]).to_string(index=False))


if __name__ == "__main__":
    run_analysis()
