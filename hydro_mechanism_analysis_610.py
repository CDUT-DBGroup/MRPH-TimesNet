"""
Water-mechanism support experiments for MRPH-TimesNet.

Outputs:
1. Cross-correlation analysis among rainfall, water level, and inflow
2. Event-level lag distribution analysis
3. Seasonal lag-shift analysis
4. Mechanism alignment between empirical lag evidence and MRPH structure
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from models import MRPH_TimesNet
from run_mrph_timesnet import set_random_seed


OUTPUT_DIR = Path("results/dl_result/hydro_mechanism_analysis")
DEFAULT_DATA_PATH = Path("dataset/water_timeseries_610.csv")
DEFAULT_CHECKPOINT = Path(
    "checkpoints/water_MRPH_TimesNet_610_MRPH_TimesNet_sl24_pl1_Comparison_Exp_0/checkpoint.pth"
)
RAINFALL_COL = "daily_rainfall"
WATER_LEVEL_COL = "avg_water_level"
INFLOW_COL = "water_inflow"
DATE_COL = "date"
MSRLA_SCALES = [3, 7, 15, 30]


def style_for_publication() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
            "axes.linewidth": 0.8,
        }
    )


def save_figure(fig: plt.Figure, path: Path, dpi: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if dpi is None:
        fig.savefig(str(path), bbox_inches="tight")
    else:
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hydro-mechanism support experiments.")
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH), help="input csv path")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="MRPH checkpoint path")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="output directory")
    parser.add_argument("--max-lag", type=int, default=30, help="maximum lag in days")
    parser.add_argument("--event-quantile", type=float, default=0.9, help="rainfall quantile for event selection")
    parser.add_argument("--merge-gap", type=int, default=1, help="merge rainfall events within this many days")
    parser.add_argument("--baseline-window", type=int, default=3, help="pre-event baseline window")
    parser.add_argument("--num-permutations", type=int, default=300, help="block permutation count")
    parser.add_argument("--block-size", type=int, default=7, help="block size for permutation significance")
    parser.add_argument("--seed", type=int, default=67, help="random seed")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    return parser.parse_args()


def load_dataset(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df["season"] = df[DATE_COL].dt.month.map(season_from_month)
    return df


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
        if len(x_lag) < 3:
            corrs.append(np.nan)
        else:
            corrs.append(float(np.corrcoef(x_lag, y_lag)[0, 1]))
    return np.asarray(corrs)


def block_permute(values: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    n = len(values)
    block_size = min(max(1, block_size), n)
    blocks = []
    while len(blocks) * block_size < n:
        start = int(rng.integers(0, max(1, n - block_size + 1)))
        blocks.append(values[start : start + block_size])
    return np.concatenate(blocks, axis=0)[:n]


def compute_significance_band(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int,
    num_permutations: int,
    block_size: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    simulated = []
    for _ in range(num_permutations):
        y_perm = block_permute(np.asarray(y, dtype=float), block_size, rng)
        simulated.append(np.abs(compute_ccf(x, y_perm, max_lag)))
    return np.percentile(np.vstack(simulated), 97.5, axis=0)


def summarize_ccf(
    pair_name: str,
    lags: np.ndarray,
    corrs: np.ndarray,
    band: np.ndarray | None = None,
) -> dict[str, object]:
    best_idx = int(np.nanargmax(corrs))
    summary = {
        "pair": pair_name,
        "peak_correlation": float(corrs[best_idx]),
        "best_lag_days": int(lags[best_idx]),
    }
    if band is not None:
        significant_lags = lags[corrs > band]
        sig_window = (
            f"{int(significant_lags.min())}-{int(significant_lags.max())}"
            if len(significant_lags) > 0
            else "none"
        )
        summary["significant_lag_window"] = sig_window
        summary["best_lag_is_significant"] = bool(corrs[best_idx] > band[best_idx])
    return summary


def detect_rainfall_events(
    rainfall: pd.Series,
    quantile: float,
    merge_gap: int,
) -> tuple[list[tuple[int, int]], float]:
    positive = rainfall[rainfall > 0]
    threshold = float(positive.quantile(quantile)) if len(positive) else 0.0
    event_indices = np.where(rainfall.to_numpy() >= threshold)[0].tolist()
    if not event_indices:
        return [], threshold

    events: list[tuple[int, int]] = []
    start = event_indices[0]
    end = event_indices[0]
    for idx in event_indices[1:]:
        if idx - end <= merge_gap:
            end = idx
        else:
            events.append((start, end))
            start = idx
            end = idx
    events.append((start, end))
    return events, threshold


def event_response_lag(
    series: np.ndarray,
    event_start: int,
    event_end: int,
    max_lag: int,
    baseline_window: int,
) -> tuple[int, float]:
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


def summarize_lag_distribution(target_name: str, lags: np.ndarray) -> dict[str, object]:
    lags = np.asarray(lags, dtype=float)
    q25 = float(np.percentile(lags, 25))
    q75 = float(np.percentile(lags, 75))
    counts = np.bincount(lags.astype(int))
    peak_lag = int(np.argmax(counts))
    return {
        "target": target_name,
        "event_count": int(len(lags)),
        "mean_lag_days": float(np.mean(lags)),
        "median_lag_days": float(np.median(lags)),
        "iqr_days": float(q75 - q25),
        "dominant_peak_lag_days": peak_lag,
        "dominant_peak_share": float(counts[peak_lag] / max(1, counts.sum())),
    }


def season_from_month(month: int) -> str:
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    if month in (9, 10, 11):
        return "Autumn"
    return "Winter"


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


def normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if all(key.startswith("module.") for key in state_dict):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def load_mrph_parameters(checkpoint_path: Path, gpu: int) -> dict[str, float]:
    model_args = build_model_args()
    model_args.gpu = gpu
    device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    model = MRPH_TimesNet.Model(model_args).float().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(normalize_state_dict(state_dict))
    model.eval()

    alpha = float(model.pchte.alpha.item())
    beta = float(model.pchte.beta.item())
    gamma = float(model.pchte.gamma.item())
    phys_gate = float(model.phys_gate.item())
    msrla_gate = float(model.msrla_gate.item())
    return {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "phys_gate": phys_gate,
        "msrla_gate": msrla_gate,
    }


def plot_ccf_analysis(output_dir: Path, lags: np.ndarray, results: list[dict[str, object]]) -> None:
    style_for_publication()
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.2), sharex=True, sharey=True)
    panel_labels = ["(a)", "(b)", "(c)"]
    for ax, result, panel in zip(axes, results, panel_labels):
        corrs = np.asarray(result["corrs"])
        band = np.asarray(result["band"])
        summary = result["summary"]
        best_lag = int(summary["best_lag_days"])
        peak_corr = float(summary["peak_correlation"])
        ax.plot(lags, corrs, color="#1f4e79", linewidth=1.7)
        ax.fill_between(lags, -band, band, color="#d9d9d9", alpha=0.5)
        ax.scatter([best_lag], [peak_corr], color="#d55e00", s=36, zorder=5)
        ax.set_title(f"{panel} {result['pair']}", pad=8)
        ax.set_xlabel("Lag (days)")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    axes[0].set_ylabel("Correlation coefficient")
    fig.suptitle("Cross-Correlation Analysis of Hydro-Meteorological Responses", y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir / "hydro_ccf_analysis.png", dpi=350)
    save_figure(fig, output_dir / "hydro_ccf_analysis.pdf")
    plt.close(fig)


def plot_lag_distribution(output_dir: Path, event_df: pd.DataFrame, max_lag: int) -> None:
    style_for_publication()
    bins = np.arange(-0.5, max_lag + 1.5, 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), sharey=True)
    targets = [
        ("inflow_lag_days", "Inflow response lag"),
        ("water_level_lag_days", "Water-level response lag"),
    ]
    for idx, (column, title) in enumerate(targets):
        ax = axes[idx]
        ax.hist(event_df[column], bins=bins, color="#4C78A8", alpha=0.82, edgecolor="white")
        ax.set_title(f"({chr(97 + idx)}) {title}", pad=8)
        ax.set_xlabel("Lag (days)")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    axes[0].set_ylabel("Event count")
    fig.suptitle("Event-Level Lag Distribution Curves", y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir / "hydro_lag_distribution.png", dpi=350)
    save_figure(fig, output_dir / "hydro_lag_distribution.pdf")
    plt.close(fig)


def plot_seasonal_lag_shift(
    output_dir: Path,
    lags: np.ndarray,
    seasonal_results: list[dict[str, object]],
) -> None:
    style_for_publication()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2))
    ax1, ax2 = axes
    for result in seasonal_results:
        ax1.plot(lags, result["corrs"], linewidth=1.6, label=result["season"])
    ax1.set_title("(a) Seasonal CCF", pad=8)
    ax1.set_xlabel("Lag (days)")
    ax1.set_ylabel("Correlation coefficient")
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax1.legend(frameon=False)

    seasons = [result["season"] for result in seasonal_results]
    best_lags = [result["summary"]["best_lag_days"] for result in seasonal_results]
    peak_corrs = [result["summary"]["peak_correlation"] for result in seasonal_results]
    ax2.plot(seasons, best_lags, marker="o", linewidth=1.6, color="#d55e00")
    for season, lag, corr in zip(seasons, best_lags, peak_corrs):
        ax2.text(season, lag + 0.4, f"r={corr:.2f}", ha="center", va="bottom", fontsize=8)
    ax2.set_title("(b) Seasonal lag shift", pad=8)
    ax2.set_xlabel("Season")
    ax2.set_ylabel("Best lag (days)")
    ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

    fig.suptitle("Seasonal Lag Shift of Rainfall-Inflow Response", y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir / "hydro_seasonal_lag_shift.png", dpi=350)
    save_figure(fig, output_dir / "hydro_seasonal_lag_shift.pdf")
    plt.close(fig)


def plot_mechanism_alignment(
    output_dir: Path,
    lags: np.ndarray,
    ccf_result: dict[str, object],
    event_df: pd.DataFrame,
    params: dict[str, float],
) -> None:
    style_for_publication()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2))
    kernel = params["alpha"] * np.exp(-params["beta"] * lags)
    kernel_norm = kernel / max(kernel.max(), 1e-8)
    ccf_positive = np.maximum(np.asarray(ccf_result["corrs"]), 0.0)
    ccf_norm = ccf_positive / max(ccf_positive.max(), 1e-8)

    axes[0].plot(lags, ccf_norm, color="#1f4e79", linewidth=1.8, label="Empirical CCF (normalized)")
    axes[0].plot(lags, kernel_norm, color="#d55e00", linewidth=1.8, label="PCHTE kernel (normalized)")
    axes[0].set_title("(a) Empirical lag vs PCHTE kernel", pad=8)
    axes[0].set_xlabel("Lag (days)")
    axes[0].set_ylabel("Normalized amplitude")
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    axes[0].legend(frameon=False)

    axes[1].hist(
        event_df["inflow_lag_days"],
        bins=np.arange(-0.5, lags.max() + 1.5, 1.0),
        color="#4C78A8",
        alpha=0.82,
        edgecolor="white",
    )
    top_y = axes[1].get_ylim()[1]
    for scale in MSRLA_SCALES:
        axes[1].axvline(scale, linestyle="--", linewidth=1.0, color="black", alpha=0.7)
        axes[1].text(scale, top_y * 0.96, f"{scale}d", ha="center", va="top", fontsize=8)
    axes[1].set_title("(b) Event lags vs MSRLA scales", pad=8)
    axes[1].set_xlabel("Lag (days)")
    axes[1].set_ylabel("Event count")
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

    fig.suptitle("Mechanism Alignment Between Hydro Evidence and MRPH Modules", y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir / "hydro_mechanism_alignment.png", dpi=350)
    save_figure(fig, output_dir / "hydro_mechanism_alignment.pdf")
    plt.close(fig)


def run_analysis() -> None:
    args = parse_args()
    set_random_seed(args.seed, deterministic=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(Path(args.data_path))
    lags = np.arange(0, args.max_lag + 1)

    ccf_pairs = [
        ("Rainfall -> Inflow", df[RAINFALL_COL].to_numpy(), df[INFLOW_COL].to_numpy()),
        ("Rainfall -> Water level", df[RAINFALL_COL].to_numpy(), df[WATER_LEVEL_COL].to_numpy()),
        ("Water level -> Inflow", df[WATER_LEVEL_COL].to_numpy(), df[INFLOW_COL].to_numpy()),
    ]
    ccf_results = []
    ccf_summary_rows = []
    for idx, (pair_name, x, y) in enumerate(ccf_pairs):
        corrs = compute_ccf(x, y, args.max_lag)
        band = compute_significance_band(
            x=x,
            y=y,
            max_lag=args.max_lag,
            num_permutations=args.num_permutations,
            block_size=args.block_size,
            seed=args.seed + idx,
        )
        summary = summarize_ccf(pair_name, lags, corrs, band)
        ccf_results.append({"pair": pair_name, "corrs": corrs, "band": band, "summary": summary})
        ccf_summary_rows.append(summary)
    ccf_summary_df = pd.DataFrame(ccf_summary_rows)
    ccf_summary_df.to_csv(output_dir / "ccf_summary.csv", index=False)
    plot_ccf_analysis(output_dir, lags, ccf_results)

    events, threshold = detect_rainfall_events(df[RAINFALL_COL], args.event_quantile, args.merge_gap)
    inflow = df[INFLOW_COL].to_numpy()
    water_level = df[WATER_LEVEL_COL].to_numpy()
    rainfall = df[RAINFALL_COL].to_numpy()
    event_rows = []
    for event_id, (start_idx, end_idx) in enumerate(events, start=1):
        inflow_lag, inflow_response = event_response_lag(
            series=inflow,
            event_start=start_idx,
            event_end=end_idx,
            max_lag=args.max_lag,
            baseline_window=args.baseline_window,
        )
        wl_lag, wl_response = event_response_lag(
            series=water_level,
            event_start=start_idx,
            event_end=end_idx,
            max_lag=args.max_lag,
            baseline_window=args.baseline_window,
        )
        event_rows.append(
            {
                "event_id": event_id,
                "start_date": df.iloc[start_idx][DATE_COL].strftime("%Y-%m-%d"),
                "end_date": df.iloc[end_idx][DATE_COL].strftime("%Y-%m-%d"),
                "season": df.iloc[start_idx]["season"],
                "event_duration_days": end_idx - start_idx + 1,
                "peak_rainfall": float(rainfall[start_idx : end_idx + 1].max()),
                "total_rainfall": float(rainfall[start_idx : end_idx + 1].sum()),
                "inflow_lag_days": int(inflow_lag),
                "inflow_peak_response": float(inflow_response),
                "water_level_lag_days": int(wl_lag),
                "water_level_peak_response": float(wl_response),
            }
        )
    event_df = pd.DataFrame(event_rows)
    event_df.to_csv(output_dir / "event_lag_details.csv", index=False)

    lag_summary_rows = [
        summarize_lag_distribution("Inflow", event_df["inflow_lag_days"].to_numpy()),
        summarize_lag_distribution("Water level", event_df["water_level_lag_days"].to_numpy()),
    ]
    lag_summary_df = pd.DataFrame(lag_summary_rows)
    lag_summary_df.insert(0, "event_threshold_quantile", args.event_quantile)
    lag_summary_df.insert(1, "event_threshold_rainfall", threshold)
    lag_summary_df.to_csv(output_dir / "lag_distribution_summary.csv", index=False)
    plot_lag_distribution(output_dir, event_df, args.max_lag)

    seasonal_results = []
    seasonal_rows = []
    for season in ["Spring", "Summer", "Autumn", "Winter"]:
        season_df = df[df["season"] == season].reset_index(drop=True)
        if len(season_df) <= args.max_lag + 5:
            continue
        corrs = compute_ccf(
            season_df[RAINFALL_COL].to_numpy(),
            season_df[INFLOW_COL].to_numpy(),
            args.max_lag,
        )
        summary = summarize_ccf(f"{season} rainfall -> inflow", lags, corrs)
        summary["season"] = season
        summary["sample_days"] = int(len(season_df))
        seasonal_results.append({"season": season, "corrs": corrs, "summary": summary})
        seasonal_rows.append(summary)
    seasonal_summary_df = pd.DataFrame(seasonal_rows)
    seasonal_summary_df.to_csv(output_dir / "seasonal_lag_shift_summary.csv", index=False)
    plot_seasonal_lag_shift(output_dir, lags, seasonal_results)

    params = load_mrph_parameters(Path(args.checkpoint), args.gpu)
    rainfall_inflow_summary = ccf_results[0]["summary"]
    inflow_lag_summary = lag_summary_df.loc[lag_summary_df["target"] == "Inflow"].iloc[0]
    mechanism_summary = {
        "alpha": params["alpha"],
        "beta": params["beta"],
        "gamma": params["gamma"],
        "phys_gate": params["phys_gate"],
        "msrla_gate": params["msrla_gate"],
        "pchte_characteristic_lag_days": float(1.0 / max(params["beta"], 1e-8)),
        "pchte_half_life_days": float(np.log(2.0) / max(params["beta"], 1e-8)),
        "empirical_ccf_peak_lag_days": int(rainfall_inflow_summary["best_lag_days"]),
        "empirical_ccf_peak_correlation": float(rainfall_inflow_summary["peak_correlation"]),
        "event_median_inflow_lag_days": float(inflow_lag_summary["median_lag_days"]),
        "event_mean_inflow_lag_days": float(inflow_lag_summary["mean_lag_days"]),
        "msrla_scales_days": "/".join(str(scale) for scale in MSRLA_SCALES),
    }
    pd.DataFrame([mechanism_summary]).to_csv(output_dir / "mechanism_alignment_summary.csv", index=False)
    plot_mechanism_alignment(output_dir, lags, ccf_results[0], event_df, params)

    report = {
        "seed": args.seed,
        "data_path": str(Path(args.data_path)),
        "checkpoint": str(Path(args.checkpoint)),
        "max_lag_days": args.max_lag,
        "event_quantile": args.event_quantile,
        "event_rainfall_threshold": threshold,
        "num_events": int(len(event_df)),
        "ccf_summary": ccf_summary_rows,
        "lag_distribution_summary": lag_summary_rows,
        "seasonal_lag_shift_summary": seasonal_rows,
        "mechanism_alignment": mechanism_summary,
    }
    (output_dir / "hydro_mechanism_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved hydro-mechanism results to: {output_dir}")
    print(ccf_summary_df.to_string(index=False))
    print(lag_summary_df.to_string(index=False))
    print(pd.DataFrame(seasonal_rows).to_string(index=False))
    print(pd.DataFrame([mechanism_summary]).to_string(index=False))


if __name__ == "__main__":
    run_analysis()
