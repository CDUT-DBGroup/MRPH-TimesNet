"""
Uncertainty analysis experiment

Main scheme:
1. Ensemble (5-10 models)

Supplementary scheme:
2. Monte Carlo Dropout

Core outputs:
- 90% / 95% prediction interval
- PICP (coverage rate)
- PINAW
- ACE
- Winkler score
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

from data_provider.data_factory import data_provider
from models import MRPH_TimesNet
from run_mrph_timesnet import set_random_seed
from utils.metrics import MAE, NSE, RMSE

OUTPUT_DIR = Path("results/dl_result/uncertainty_analysis")
DEFAULT_MC_CHECKPOINT = Path(
    "checkpoints/water_MRPH_TimesNet_610_MRPH_TimesNet_sl24_pl1_Comparison_Exp_0/checkpoint.pth"
)
DEFAULT_ENSEMBLE_SEEDS = [67, 95, 170, 249, 285]
DEFAULT_ENSEMBLE_PATTERN = (
    "checkpoints/water_MRPH_TimesNet_610_MRPH_TimesNet_sl24_pl1_SeedSearch_{seed}_0/checkpoint.pth"
)
DATE_COL = "date"
TARGET_COL = "water_inflow"


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
    parser = argparse.ArgumentParser(description="Uncertainty analysis for MRPH-TimesNet.")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="output directory")
    parser.add_argument("--root_path", type=str, default="./dataset/", help="dataset root path")
    parser.add_argument("--data_path", type=str, default="water_timeseries_610.csv", help="dataset file")
    parser.add_argument("--target", type=str, default=TARGET_COL, help="target column")
    parser.add_argument("--seq_len", type=int, default=24, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=1, help="decoder label length")
    parser.add_argument("--pred_len", type=int, default=1, help="prediction length")
    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
    parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
    parser.add_argument("--c_out", type=int, default=1, help="output size")
    parser.add_argument("--d_model", type=int, default=32, help="model dimension")
    parser.add_argument("--n_heads", type=int, default=8, help="attention heads")
    parser.add_argument("--e_layers", type=int, default=2, help="encoder layers")
    parser.add_argument("--d_layers", type=int, default=1, help="decoder layers")
    parser.add_argument("--d_ff", type=int, default=32, help="feed-forward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument("--top_k", type=int, default=5, help="top-k periods")
    parser.add_argument("--num_kernels", type=int, default=6, help="inception kernels")
    parser.add_argument("--moving_avg", type=int, default=25, help="moving average window")
    parser.add_argument("--factor", type=int, default=3, help="attention factor")
    parser.add_argument("--expand", type=int, default=2, help="mamba expand factor")
    parser.add_argument("--d_conv", type=int, default=4, help="mamba conv kernel")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="data loader workers")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument("--base-seed", type=int, default=2178, help="base reproducibility seed")

    parser.add_argument("--run-ensemble", action="store_true", default=True, help="run ensemble uncertainty")
    parser.add_argument("--run-mc-dropout", action="store_true", default=True, help="run MC dropout uncertainty")
    parser.add_argument("--disable-ensemble", action="store_false", dest="run_ensemble")
    parser.add_argument("--disable-mc-dropout", action="store_false", dest="run_mc_dropout")

    parser.add_argument(
        "--mc-checkpoint",
        type=str,
        default=str(DEFAULT_MC_CHECKPOINT),
        help="checkpoint used for MC Dropout",
    )
    parser.add_argument("--mc-samples", type=int, default=100, help="number of MC dropout forward passes")

    parser.add_argument(
        "--ensemble-checkpoints",
        type=str,
        nargs="*",
        default=None,
        help="explicit ensemble checkpoint paths",
    )
    parser.add_argument(
        "--ensemble-seeds",
        type=int,
        nargs="*",
        default=DEFAULT_ENSEMBLE_SEEDS,
        help="ensemble seeds used with checkpoint pattern",
    )
    parser.add_argument(
        "--ensemble-checkpoint-pattern",
        type=str,
        default=DEFAULT_ENSEMBLE_PATTERN,
        help="checkpoint pattern containing {seed}",
    )
    parser.add_argument(
        "--nominal-levels",
        type=float,
        nargs="*",
        default=[0.90, 0.95],
        help="nominal interval levels, e.g. 0.9 0.95",
    )
    parser.add_argument(
        "--apply-conformal-calibration",
        action="store_true",
        default=True,
        help="apply split conformal calibration using the validation split",
    )
    parser.add_argument(
        "--disable-conformal-calibration",
        action="store_false",
        dest="apply_conformal_calibration",
    )
    return parser.parse_args()


def build_model_args(cli_args: argparse.Namespace, seed: int) -> argparse.Namespace:
    use_gpu = torch.cuda.is_available()
    return argparse.Namespace(
        task_name="long_term_forecast",
        is_training=0,
        model_id="water_MRPH_TimesNet_610",
        model="MRPH_TimesNet",
        data="custom",
        root_path=cli_args.root_path,
        data_path=cli_args.data_path,
        features="MS",
        target=cli_args.target,
        freq="d",
        checkpoints="./checkpoints/",
        seq_len=cli_args.seq_len,
        label_len=cli_args.label_len,
        pred_len=cli_args.pred_len,
        seasonal_patterns="Monthly",
        inverse=False,
        expand=cli_args.expand,
        d_conv=cli_args.d_conv,
        top_k=cli_args.top_k,
        num_kernels=cli_args.num_kernels,
        enc_in=cli_args.enc_in,
        dec_in=cli_args.dec_in,
        c_out=cli_args.c_out,
        d_model=cli_args.d_model,
        n_heads=cli_args.n_heads,
        e_layers=cli_args.e_layers,
        d_layers=cli_args.d_layers,
        d_ff=cli_args.d_ff,
        moving_avg=cli_args.moving_avg,
        factor=cli_args.factor,
        distil=True,
        dropout=cli_args.dropout,
        embed="timeF",
        activation="gelu",
        output_attention=False,
        num_workers=cli_args.num_workers,
        itr=1,
        train_epochs=150,
        batch_size=cli_args.batch_size,
        patience=10,
        learning_rate=0.001,
        des="Uncertainty",
        loss="MSE",
        lradj="type1",
        use_amp=False,
        use_gpu=use_gpu,
        gpu=cli_args.gpu,
        use_multi_gpu=False,
        devices="0,1,2,3",
        augmentation_ratio=0,
        seed=seed,
        deterministic=True,
        jitter=False,
    )


def get_device(args: argparse.Namespace) -> torch.device:
    if args.use_gpu:
        return torch.device(f"cuda:{args.gpu}")
    return torch.device("cpu")


def load_model(args: argparse.Namespace, checkpoint_path: Path) -> torch.nn.Module:
    device = get_device(args)
    model = MRPH_TimesNet.Model(args).float().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_split_dates(data_path: Path, seq_len: int, pred_len: int, flag: str) -> pd.Series:
    df = pd.read_csv(data_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    total_len = len(df)
    num_train = int(total_len * 0.8)
    num_test = int(total_len * 0.1)
    num_vali = total_len - num_train - num_test
    border1s = [0, num_train - seq_len, total_len - num_test - seq_len]
    border2s = [num_train, num_train + num_vali, total_len]
    flag_to_idx = {"train": 0, "val": 1, "test": 2}
    border1 = border1s[flag_to_idx[flag]]
    border2 = border2s[flag_to_idx[flag]]
    target_start = border1 + seq_len + pred_len - 1
    target_end = total_len
    if flag != "test":
        target_end = border2
    return df.iloc[target_start:target_end][DATE_COL].reset_index(drop=True)


def align_dates(dates: pd.Series, target_length: int) -> pd.Series:
    if len(dates) == target_length:
        return dates.reset_index(drop=True)
    return pd.Series(np.arange(target_length), name="time_step")


def get_loader(args: argparse.Namespace, flag: str):
    _, data_loader = data_provider(args, flag)
    return data_loader


def predict_on_loader(
    model: torch.nn.Module,
    data_loader,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    preds = []
    trues = []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float()
            dec_inp = torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1).float().to(device)
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == "MS" else 0
            outputs = outputs[:, -args.pred_len :, f_dim:]
            batch_y = batch_y[:, -args.pred_len :, f_dim:]

            preds.append(outputs.detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0).reshape(-1)
    trues = np.concatenate(trues, axis=0).reshape(-1)
    return preds, trues


def enable_mc_dropout(model: torch.nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def resolve_ensemble_checkpoints(cli_args: argparse.Namespace) -> list[Path]:
    if cli_args.ensemble_checkpoints:
        checkpoints = [Path(path) for path in cli_args.ensemble_checkpoints]
    else:
        checkpoints = [
            Path(cli_args.ensemble_checkpoint_pattern.format(seed=seed))
            for seed in cli_args.ensemble_seeds
        ]
    missing = [str(path) for path in checkpoints if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The following ensemble checkpoints were not found:\n" + "\n".join(missing)
        )
    return checkpoints


def compute_point_metrics(mean_pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    return {
        "nse": float(NSE(mean_pred, true)),
        "rmse": float(RMSE(mean_pred, true)),
        "mae": float(MAE(mean_pred, true)),
        "r2": float(r2_score(true.reshape(-1), mean_pred.reshape(-1))),
    }


def interval_bounds(samples: np.ndarray, nominal_level: float) -> tuple[np.ndarray, np.ndarray]:
    alpha = 1.0 - nominal_level
    lower = np.quantile(samples, alpha / 2.0, axis=0)
    upper = np.quantile(samples, 1.0 - alpha / 2.0, axis=0)
    return lower, upper


def calculate_picp(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def calculate_pinaw(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    width = upper - lower
    y_range = float(np.max(y_true) - np.min(y_true))
    y_range = max(y_range, 1e-8)
    return float(np.mean(width) / y_range)


def calculate_ace(picp: float, nominal_level: float) -> float:
    return float(picp - nominal_level)


def calculate_winkler_score(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, nominal_level: float) -> float:
    alpha = 1.0 - nominal_level
    score = upper - lower
    below = y_true < lower
    above = y_true > upper
    score = score.copy()
    score[below] += 2.0 / alpha * (lower[below] - y_true[below])
    score[above] += 2.0 / alpha * (y_true[above] - upper[above])
    return float(np.mean(score))


def build_interval_map(samples: np.ndarray, nominal_levels: list[float]) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    interval_map = {}
    for nominal_level in nominal_levels:
        interval_map[nominal_level] = interval_bounds(samples, nominal_level)
    return interval_map


def evaluate_interval_map(
    method_name: str,
    members_or_samples: int,
    mean_pred: np.ndarray,
    y_true: np.ndarray,
    interval_map: dict[float, tuple[np.ndarray, np.ndarray]],
    nominal_levels: list[float],
    interval_type: str,
    calibration_offsets: dict[float, float] | None = None,
) -> list[dict[str, float]]:
    point_metrics = compute_point_metrics(mean_pred, y_true)
    rows = []
    for nominal_level in nominal_levels:
        lower, upper = interval_map[nominal_level]
        picp = calculate_picp(y_true, lower, upper)
        pinaw = calculate_pinaw(y_true, lower, upper)
        ace = calculate_ace(picp, nominal_level)
        winkler = calculate_winkler_score(y_true, lower, upper, nominal_level)
        rows.append(
            {
                "method": method_name,
                "interval_type": interval_type,
                "members_or_samples": members_or_samples,
                "nominal_level": nominal_level,
                "picp": picp,
                "pinaw": pinaw,
                "ace": ace,
                "winkler": winkler,
                "calibration_offset": 0.0 if calibration_offsets is None else calibration_offsets[nominal_level],
                "nse": point_metrics["nse"],
                "rmse": point_metrics["rmse"],
                "mae": point_metrics["mae"],
                "r2": point_metrics["r2"],
            }
        )
    return rows


def conformal_quantile(scores: np.ndarray, nominal_level: float) -> float:
    scores = np.sort(np.asarray(scores).reshape(-1))
    if scores.size == 0:
        return 0.0
    rank = int(np.ceil((scores.size + 1) * nominal_level)) - 1
    rank = min(max(rank, 0), scores.size - 1)
    return float(scores[rank])


def compute_conformal_offsets(
    y_val: np.ndarray,
    val_interval_map: dict[float, tuple[np.ndarray, np.ndarray]],
    nominal_levels: list[float],
) -> dict[float, float]:
    offsets = {}
    for nominal_level in nominal_levels:
        lower, upper = val_interval_map[nominal_level]
        scores = np.maximum.reduce([lower - y_val, y_val - upper, np.zeros_like(y_val)])
        offsets[nominal_level] = conformal_quantile(scores, nominal_level)
    return offsets


def apply_offsets_to_intervals(
    interval_map: dict[float, tuple[np.ndarray, np.ndarray]],
    offsets: dict[float, float],
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    calibrated = {}
    for nominal_level, (lower, upper) in interval_map.items():
        offset = offsets[nominal_level]
        calibrated[nominal_level] = (lower - offset, upper + offset)
    return calibrated


def save_interval_details(
    method_name: str,
    interval_type: str,
    output_dir: Path,
    dates: pd.Series,
    y_true: np.ndarray,
    mean_pred: np.ndarray,
    interval_map: dict[float, tuple[np.ndarray, np.ndarray]],
) -> None:
    aligned_dates = align_dates(dates, len(y_true))
    df = pd.DataFrame(
        {
            "date": aligned_dates.astype(str),
            "true": y_true,
            "mean_pred": mean_pred,
        }
    )
    for nominal_level, (lower, upper) in interval_map.items():
        pct = int(round(nominal_level * 100))
        df[f"lower_{pct}"] = lower
        df[f"upper_{pct}"] = upper
    output_name = f"{method_name.lower().replace(' ', '_')}_{interval_type.lower().replace(' ', '_')}_interval_details.csv"
    df.to_csv(output_dir / output_name, index=False)


def plot_interval_figure(
    method_name: str,
    interval_type: str,
    output_dir: Path,
    dates: pd.Series,
    y_true: np.ndarray,
    mean_pred: np.ndarray,
    lower_95: np.ndarray,
    upper_95: np.ndarray,
    summary_row: dict[str, float],
) -> None:
    style_for_publication()
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    x = align_dates(dates, len(y_true))
    x_values = x.to_numpy() if isinstance(x, pd.Series) else np.asarray(x)
    ax.fill_between(x, lower_95, upper_95, color="#c6dbef", alpha=0.55, label="95% PI")
    ax.plot(x_values, y_true, color="#1f4e79", linewidth=1.7, label="Observed")
    ax.plot(x_values, mean_pred, color="#d55e00", linewidth=1.5, label="Predicted mean")
    ax.set_title(
        f"{method_name} ({interval_type}): 95% Prediction Interval\n"
        f"PICP={summary_row['picp']:.3f}, PINAW={summary_row['pinaw']:.3f}, NSE={summary_row['nse']:.3f}"
    )
    ax.set_ylabel("Inflow")
    if np.issubdtype(np.asarray(x).dtype, np.datetime64):
        ax.set_xlabel("Date")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=25)
    else:
        ax.set_xlabel("Time step")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    fig.tight_layout()
    prefix = f"{method_name.lower().replace(' ', '_')}_{interval_type.lower().replace(' ', '_')}"
    save_figure(fig, output_dir / f"{prefix}_95pi.png", dpi=350)
    save_figure(fig, output_dir / f"{prefix}_95pi.pdf")
    plt.close(fig)


def plot_calibration_figure(output_dir: Path, summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return
    style_for_publication()
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for (method_name, interval_type), subset in summary_df.groupby(["method", "interval_type"]):
        subset = subset.sort_values("nominal_level")
        ax.plot(
            subset["nominal_level"],
            subset["picp"],
            marker="o",
            linewidth=1.8,
            label=f"{method_name} ({interval_type})",
        )
    reference = np.linspace(0.0, 1.0, 100)
    ax.plot(reference, reference, linestyle="--", color="black", linewidth=1.0, label="Ideal calibration")
    ax.set_xlabel("Nominal interval level")
    ax.set_ylabel("Empirical coverage (PICP)")
    ax.set_title("Prediction-Interval Calibration")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir / "uncertainty_calibration.png", dpi=350)
    save_figure(fig, output_dir / "uncertainty_calibration.pdf")
    plt.close(fig)


def run_ensemble_analysis(cli_args: argparse.Namespace, output_dir: Path, dates: pd.Series) -> list[dict[str, float]]:
    checkpoints = resolve_ensemble_checkpoints(cli_args)
    val_member_preds = []
    test_member_preds = []
    y_val = None
    y_test = None

    for idx, checkpoint_path in enumerate(checkpoints):
        seed = cli_args.ensemble_seeds[idx] if idx < len(cli_args.ensemble_seeds) else cli_args.base_seed + idx
        model_args = build_model_args(cli_args, seed=seed)
        device = get_device(model_args)
        set_random_seed(seed, deterministic=model_args.deterministic)
        model = load_model(model_args, checkpoint_path)
        val_loader = get_loader(model_args, "val")
        test_loader = get_loader(model_args, "test")
        val_preds, val_trues = predict_on_loader(model, val_loader, model_args, device)
        test_preds, test_trues = predict_on_loader(model, test_loader, model_args, device)
        val_member_preds.append(val_preds)
        test_member_preds.append(test_preds)
        if y_val is None:
            y_val = val_trues
        if y_test is None:
            y_test = test_trues

    val_member_preds = np.stack(val_member_preds, axis=0)
    test_member_preds = np.stack(test_member_preds, axis=0)
    method_name = f"Ensemble-{len(checkpoints)}"
    mean_pred = np.mean(test_member_preds, axis=0)
    raw_test_interval_map = build_interval_map(test_member_preds, cli_args.nominal_levels)
    rows = evaluate_interval_map(
        method_name=method_name,
        members_or_samples=len(checkpoints),
        mean_pred=mean_pred,
        y_true=y_test,
        interval_map=raw_test_interval_map,
        nominal_levels=cli_args.nominal_levels,
        interval_type="raw",
    )
    save_interval_details("ensemble", "raw", output_dir, dates, y_test, mean_pred, raw_test_interval_map)
    preferred_level = 0.95 if 0.95 in raw_test_interval_map else max(raw_test_interval_map)
    summary_row = next(
        row
        for row in rows
        if row["interval_type"] == "raw" and abs(row["nominal_level"] - preferred_level) < 1e-9
    )
    lower, upper = raw_test_interval_map[preferred_level]
    plot_interval_figure("Ensemble", "raw", output_dir, dates, y_test, mean_pred, lower, upper, summary_row)

    if cli_args.apply_conformal_calibration:
        raw_val_interval_map = build_interval_map(val_member_preds, cli_args.nominal_levels)
        offsets = compute_conformal_offsets(y_val, raw_val_interval_map, cli_args.nominal_levels)
        calibrated_test_map = apply_offsets_to_intervals(raw_test_interval_map, offsets)
        rows.extend(
            evaluate_interval_map(
                method_name=method_name,
                members_or_samples=len(checkpoints),
                mean_pred=mean_pred,
                y_true=y_test,
                interval_map=calibrated_test_map,
                nominal_levels=cli_args.nominal_levels,
                interval_type="conformal",
                calibration_offsets=offsets,
            )
        )
        save_interval_details("ensemble", "conformal", output_dir, dates, y_test, mean_pred, calibrated_test_map)
        conformal_row = next(
            row
            for row in rows
            if row["interval_type"] == "conformal" and abs(row["nominal_level"] - preferred_level) < 1e-9
        )
        lower, upper = calibrated_test_map[preferred_level]
        plot_interval_figure("Ensemble", "conformal", output_dir, dates, y_test, mean_pred, lower, upper, conformal_row)
    return rows


def run_mc_dropout_analysis(cli_args: argparse.Namespace, output_dir: Path, dates: pd.Series) -> list[dict[str, float]]:
    checkpoint_path = Path(cli_args.mc_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"MC Dropout checkpoint not found: {checkpoint_path}")

    model_args = build_model_args(cli_args, seed=cli_args.base_seed)
    device = get_device(model_args)
    set_random_seed(cli_args.base_seed, deterministic=model_args.deterministic)
    model = load_model(model_args, checkpoint_path)
    enable_mc_dropout(model)

    val_sample_preds = []
    test_sample_preds = []
    y_val = None
    y_test = None
    val_loader = get_loader(model_args, "val")
    test_loader = get_loader(model_args, "test")
    for sample_idx in range(cli_args.mc_samples):
        val_preds, val_trues = predict_on_loader(model, val_loader, model_args, device)
        test_preds, test_trues = predict_on_loader(model, test_loader, model_args, device)
        val_sample_preds.append(val_preds)
        test_sample_preds.append(test_preds)
        if y_val is None:
            y_val = val_trues
        if y_test is None:
            y_test = test_trues

    val_sample_preds = np.stack(val_sample_preds, axis=0)
    test_sample_preds = np.stack(test_sample_preds, axis=0)
    method_name = f"MC Dropout-{cli_args.mc_samples}"
    mean_pred = np.mean(test_sample_preds, axis=0)
    raw_test_interval_map = build_interval_map(test_sample_preds, cli_args.nominal_levels)
    rows = evaluate_interval_map(
        method_name=method_name,
        members_or_samples=cli_args.mc_samples,
        mean_pred=mean_pred,
        y_true=y_test,
        interval_map=raw_test_interval_map,
        nominal_levels=cli_args.nominal_levels,
        interval_type="raw",
    )
    save_interval_details("mc_dropout", "raw", output_dir, dates, y_test, mean_pred, raw_test_interval_map)
    preferred_level = 0.95 if 0.95 in raw_test_interval_map else max(raw_test_interval_map)
    summary_row = next(
        row
        for row in rows
        if row["interval_type"] == "raw" and abs(row["nominal_level"] - preferred_level) < 1e-9
    )
    lower, upper = raw_test_interval_map[preferred_level]
    plot_interval_figure("MC Dropout", "raw", output_dir, dates, y_test, mean_pred, lower, upper, summary_row)

    if cli_args.apply_conformal_calibration:
        raw_val_interval_map = build_interval_map(val_sample_preds, cli_args.nominal_levels)
        offsets = compute_conformal_offsets(y_val, raw_val_interval_map, cli_args.nominal_levels)
        calibrated_test_map = apply_offsets_to_intervals(raw_test_interval_map, offsets)
        rows.extend(
            evaluate_interval_map(
                method_name=method_name,
                members_or_samples=cli_args.mc_samples,
                mean_pred=mean_pred,
                y_true=y_test,
                interval_map=calibrated_test_map,
                nominal_levels=cli_args.nominal_levels,
                interval_type="conformal",
                calibration_offsets=offsets,
            )
        )
        save_interval_details("mc_dropout", "conformal", output_dir, dates, y_test, mean_pred, calibrated_test_map)
        conformal_row = next(
            row
            for row in rows
            if row["interval_type"] == "conformal" and abs(row["nominal_level"] - preferred_level) < 1e-9
        )
        lower, upper = calibrated_test_map[preferred_level]
        plot_interval_figure("MC Dropout", "conformal", output_dir, dates, y_test, mean_pred, lower, upper, conformal_row)
    return rows


def write_experiment_note(cli_args: argparse.Namespace, output_dir: Path, summary_df: pd.DataFrame) -> None:
    note = {
        "dataset": cli_args.data_path,
        "methods_run": summary_df["method"].unique().tolist(),
        "nominal_levels": cli_args.nominal_levels,
        "ensemble_seeds": cli_args.ensemble_seeds,
        "ensemble_checkpoint_pattern": cli_args.ensemble_checkpoint_pattern,
        "mc_checkpoint": cli_args.mc_checkpoint,
        "mc_samples": cli_args.mc_samples,
        "apply_conformal_calibration": cli_args.apply_conformal_calibration,
        "recommended_primary_method": "Ensemble + conformal calibration",
        "recommended_primary_metrics": ["PICP", "PINAW", "ACE", "Winkler", "NSE", "RMSE"],
    }
    with (output_dir / "experiment_setup.json").open("w", encoding="utf-8") as f:
        json.dump(note, f, ensure_ascii=False, indent=2)


def run_uncertainty_analysis() -> pd.DataFrame:
    cli_args = parse_args()
    output_dir = Path(cli_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dates = load_split_dates(
        Path(cli_args.root_path) / cli_args.data_path,
        cli_args.seq_len,
        cli_args.pred_len,
        flag="test",
    )

    all_rows = []
    if cli_args.run_ensemble:
        print("=== Running Ensemble uncertainty analysis ===")
        all_rows.extend(run_ensemble_analysis(cli_args, output_dir, dates))
    if cli_args.run_mc_dropout:
        print("=== Running MC Dropout uncertainty analysis ===")
        all_rows.extend(run_mc_dropout_analysis(cli_args, output_dir, dates))

    summary_df = pd.DataFrame(all_rows)
    if summary_df.empty:
        raise ValueError("No uncertainty method was executed. Please enable ensemble and/or MC dropout.")
    summary_df = summary_df.sort_values(by=["method", "nominal_level"]).reset_index(drop=True)
    summary_df.to_csv(output_dir / "uncertainty_summary.csv", index=False)
    plot_calibration_figure(output_dir, summary_df)
    write_experiment_note(cli_args, output_dir, summary_df)

    print(f"\nUncertainty analysis finished. Results saved to: {output_dir}")
    return summary_df


if __name__ == "__main__":
    run_uncertainty_analysis()
