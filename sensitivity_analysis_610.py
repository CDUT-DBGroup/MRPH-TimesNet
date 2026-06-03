"""
Parameter sensitivity analysis based on trained MRPH-TimesNet checkpoints.
Evaluates physical parameters alpha / beta / gamma with one-factor-at-a-time sweeps.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

from data_provider.data_factory import data_provider
from models import MRPH_TimesNet
from run_mrph_timesnet import set_random_seed
from utils.metrics import metric

OUTPUT_DIR = Path("results/dl_result/res_all/sensitivity_analysis")
DEFAULT_CHECKPOINT = Path(
    "checkpoints/water_MRPH_TimesNet_610_MRPH_TimesNet_sl24_pl1_Comparison_Exp_0/checkpoint.pth"
)
PHYSICAL_PARAMETERS = ["alpha", "beta", "gamma"]
GATE_PARAMETERS = ["phys_gate", "msrla_gate"]
PARAMETER_SPECS = {
    "alpha": {
        "group": "physical",
        "module": "PCHTE",
        "display": r"$\alpha$",
        "title": "Infiltration coefficient",
        "description": "Controls the rainfall-to-infiltration strength in the physical embedding branch.",
        "transform": "sigmoid",
        "clip": (0.02, 0.98),
    },
    "beta": {
        "group": "physical",
        "module": "PCHTE",
        "display": r"$\beta$",
        "title": "Lag-decay coefficient",
        "description": "Controls how fast the lagged rainfall response decays over time.",
        "transform": "softplus",
        "clip": (0.01, None),
    },
    "gamma": {
        "group": "physical",
        "module": "PCHTE",
        "display": r"$\gamma$",
        "title": "Response-mixing coefficient",
        "description": "Balances infiltration response and water-level-change response in the physical branch.",
        "transform": "sigmoid",
        "clip": (0.02, 0.98),
    },
    "phys_gate": {
        "group": "gate",
        "module": "Fusion Gate",
        "display": r"$g_{\mathrm{phys}}$",
        "title": "Physical branch fusion gate",
        "description": "Controls how strongly the PCHTE physical embedding is fused into the backbone features.",
        "transform": "sigmoid",
        "clip": (0.0, 1.0),
    },
    "msrla_gate": {
        "group": "gate",
        "module": "Fusion Gate",
        "display": r"$g_{\mathrm{msrla}}$",
        "title": "Lag-attention fusion gate",
        "description": "Controls how strongly the MSRLA multi-scale lag feature branch is fused into the backbone features.",
        "transform": "sigmoid",
        "clip": (0.0, 1.0),
    },
}


def calculate_nse(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred).reshape(-1)
    true = np.asarray(true).reshape(-1)
    denominator = np.sum((true - np.mean(true)) ** 2)
    if denominator <= 1e-12:
        return float("nan")
    return float(1 - np.sum((true - pred) ** 2) / denominator)


def inverse_sigmoid(value: float) -> float:
    value = float(np.clip(value, 1e-6, 1 - 1e-6))
    return float(np.log(value / (1 - value)))


def inverse_softplus(value: float) -> float:
    value = float(max(value, 1e-6))
    return float(np.log(np.expm1(value)))


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


def safe_to_csv(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        fallback_path = path.with_name(f"{path.stem}_latest{path.suffix}")
        df.to_csv(fallback_path, index=False)
        print(f"File is in use; wrote to fallback file: {fallback_path}")
        return fallback_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-factor-at-a-time sensitivity analysis for MRPH-TimesNet physical and gate parameters."
    )
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="trained checkpoint path")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="directory for csv and figures")
    parser.add_argument("--root_path", type=str, default="./dataset/", help="dataset root path")
    parser.add_argument("--data_path", type=str, default="water_timeseries_610.csv", help="dataset file")
    parser.add_argument("--target", type=str, default="water_inflow", help="target column")
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
    parser.add_argument("--seed", type=int, default=67, help="reproducibility seed")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument("--num-points", type=int, default=7, help="number of sweep points per parameter")
    parser.add_argument(
        "--relative-span",
        type=float,
        default=0.6,
        help="relative sweep span around the learned value, e.g. 0.6 -> [0.4x, 1.6x]",
    )
    return parser.parse_args()


def build_model_args(cli_args: argparse.Namespace) -> argparse.Namespace:
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
        des="Sensitivity",
        loss="MSE",
        lradj="type1",
        use_amp=False,
        use_gpu=use_gpu,
        gpu=cli_args.gpu,
        use_multi_gpu=False,
        devices="0,1,2,3",
        augmentation_ratio=0,
        seed=cli_args.seed,
        deterministic=True,
        jitter=False,
    )


def get_device(args: argparse.Namespace) -> torch.device:
    if args.use_gpu:
        return torch.device(f"cuda:{args.gpu}")
    return torch.device("cpu")


def load_trained_model(args: argparse.Namespace, checkpoint_path: Path) -> torch.nn.Module:
    device = get_device(args)
    model = MRPH_TimesNet.Model(args).float().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def get_parameter_value(model: torch.nn.Module, parameter_name: str) -> float:
    base_model = unwrap_model(model)
    if parameter_name in PHYSICAL_PARAMETERS:
        return float(getattr(base_model.pchte, parameter_name).item())
    if parameter_name in GATE_PARAMETERS:
        return float(getattr(base_model, parameter_name).item())
    raise ValueError(f"Unsupported parameter: {parameter_name}")


def set_parameter_value(model: torch.nn.Module, parameter_name: str, value: float) -> float:
    base_model = unwrap_model(model)
    transform = PARAMETER_SPECS[parameter_name]["transform"]
    if transform == "sigmoid":
        clipped_value = float(np.clip(value, 1e-4, 1 - 1e-4))
        raw_value = inverse_sigmoid(clipped_value)
    elif transform == "softplus":
        clipped_value = float(max(value, 1e-4))
        raw_value = inverse_softplus(clipped_value)
    else:
        raise ValueError(f"Unsupported parameter: {parameter_name}")

    if parameter_name in PHYSICAL_PARAMETERS:
        getattr(base_model.pchte, f"{parameter_name}_raw").data.fill_(raw_value)
    elif parameter_name in GATE_PARAMETERS:
        getattr(base_model, f"{parameter_name}_raw").data.fill_(raw_value)
    else:
        raise ValueError(f"Unsupported parameter: {parameter_name}")
    return get_parameter_value(model, parameter_name)


def make_parameter_grid(base_value: float, parameter_name: str, num_points: int, relative_span: float) -> list[float]:
    low_ratio = max(0.1, 1.0 - relative_span)
    high_ratio = 1.0 + relative_span
    ratios = np.linspace(low_ratio, high_ratio, num_points)
    values = base_value * ratios

    clip_low, clip_high = PARAMETER_SPECS[parameter_name]["clip"]
    values = np.clip(values, clip_low, clip_high)

    values = np.unique(np.round(values, 6))
    if not np.isclose(values, base_value, atol=1e-6).any():
        values = np.sort(np.append(values, round(base_value, 6)))
    return [float(v) for v in values]


def evaluate_on_loader(model: torch.nn.Module, data_loader, args: argparse.Namespace, device: torch.device) -> dict[str, float]:
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

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    mae, mse, rmse, mape, mspe = metric(preds, trues)
    r2 = r2_score(trues.reshape(-1), preds.reshape(-1))
    nse = calculate_nse(preds, trues)
    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "mape": float(mape),
        "mspe": float(mspe),
        "r2": float(r2),
        "nse": float(nse),
    }


def evaluate_parameter_setting(
    model_args: argparse.Namespace,
    checkpoint_path: Path,
    parameter_name: str,
    parameter_value: float,
) -> dict[str, float]:
    set_random_seed(model_args.seed, deterministic=model_args.deterministic)
    device = get_device(model_args)
    model = load_trained_model(model_args, checkpoint_path)
    actual_value = set_parameter_value(model, parameter_name, parameter_value)

    _, test_loader = data_provider(model_args, "test")
    test_metrics = evaluate_on_loader(model, test_loader, model_args, device)

    result = {
        "parameter": parameter_name,
        "parameter_group": PARAMETER_SPECS[parameter_name]["group"],
        "parameter_module": PARAMETER_SPECS[parameter_name]["module"],
        "parameter_title": PARAMETER_SPECS[parameter_name]["title"],
        "value": actual_value,
        "alpha": get_parameter_value(model, "alpha"),
        "beta": get_parameter_value(model, "beta"),
        "gamma": get_parameter_value(model, "gamma"),
        "phys_gate": get_parameter_value(model, "phys_gate"),
        "msrla_gate": get_parameter_value(model, "msrla_gate"),
    }
    for key, value in test_metrics.items():
        result[f"test_{key}"] = value
    return result


def plot_sensitivity(
    results_df: pd.DataFrame,
    output_dir: Path,
    parameter_names: list[str],
    output_stem: str,
    title: str,
) -> None:
    style_for_publication()
    subset_df = results_df[results_df["parameter"].isin(parameter_names)].copy()
    all_nse = subset_df["test_nse"]
    y_min = float(all_nse.min())
    y_max = float(all_nse.max())
    y_pad = max((y_max - y_min) * 0.12, 0.00005)

    n_params = len(parameter_names)
    fig, axes = plt.subplots(1, n_params, figsize=(4.6 * n_params, 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    panel_labels = [f"({chr(97 + idx)})" for idx in range(n_params)]

    for idx, parameter_name in enumerate(parameter_names):
        ax = axes[idx]
        subset = subset_df[subset_df["parameter"] == parameter_name].sort_values("value")
        baseline_row = subset.loc[(subset["is_baseline"] == 1)].iloc[0]
        best_row = subset.loc[subset["test_nse"].idxmax()]
        display = PARAMETER_SPECS[parameter_name]["display"]

        ax.plot(
            subset["value"],
            subset["test_nse"],
            marker="o",
            linewidth=1.8,
            color="#d62728",
            label="Test NSE",
        )
        ax.axvline(
            baseline_row["value"],
            linestyle="--",
            linewidth=1.0,
            color="black",
            alpha=0.8,
        )
        ax.scatter(
            [baseline_row["value"]],
            [baseline_row["test_nse"]],
            marker="*",
            s=140,
            color="#ffbf00",
            edgecolor="black",
            linewidth=0.5,
            zorder=5,
        )
        ax.scatter(
            [best_row["value"]],
            [best_row["test_nse"]],
            marker="D",
            s=52,
            color="#2ca02c",
            edgecolor="black",
            linewidth=0.5,
            zorder=6,
        )

        ax.set_title(f"{panel_labels[idx]} {display} Sensitivity", pad=8)
        ax.set_xlabel(f"{display} value")
        if idx == 0:
            ax.set_ylabel("NSE")
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
        ax.text(
            baseline_row["value"],
            y_max + y_pad * 0.15,
            "Learned",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.annotate(
            f"Best={best_row['value']:.3f}\nNSE={best_row['test_nse']:.4f}",
            xy=(best_row["value"], best_row["test_nse"]),
            xytext=(8, -24),
            textcoords="offset points",
            fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.9),
            arrowprops=dict(arrowstyle="->", lw=0.7, color="0.3"),
        )

    handles, labels = axes[0].get_legend_handles_labels()
    handles.extend(
        [
            plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#ffbf00", markeredgecolor="black", markersize=10, label="Learned value"),
            plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#2ca02c", markeredgecolor="black", markersize=7, label="Best test NSE"),
        ]
    )
    labels.extend(["Learned value", "Best test NSE"])
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(title, y=1.10)
    fig.tight_layout()
    fig.savefig(output_dir / f"{output_stem}.png", dpi=350, bbox_inches="tight")
    fig.savefig(output_dir / f"{output_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def run_sensitivity_analysis() -> pd.DataFrame:
    cli_args = parse_args()
    checkpoint_path = Path(cli_args.checkpoint)
    output_dir = Path(cli_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model_args = build_model_args(cli_args)
    base_model = load_trained_model(model_args, checkpoint_path)
    all_parameter_names = PHYSICAL_PARAMETERS + GATE_PARAMETERS
    baseline_values = {
        parameter_name: get_parameter_value(base_model, parameter_name)
        for parameter_name in all_parameter_names
    }

    print("=== MRPH-TimesNet Parameter Sensitivity Analysis ===")
    print("Using real checkpoints for one-factor-at-a-time evaluation without surrogate models or noise simulation.")
    print(
        "Learned physical parameter baseline: "
        f"alpha={baseline_values['alpha']:.6f}, "
        f"beta={baseline_values['beta']:.6f}, "
        f"gamma={baseline_values['gamma']:.6f}"
    )
    print(
        "Learned gate parameter baseline: "
        f"phys_gate={baseline_values['phys_gate']:.6f}, "
        f"msrla_gate={baseline_values['msrla_gate']:.6f}"
    )

    parameter_catalog_rows = []
    for parameter_name in all_parameter_names:
        spec = PARAMETER_SPECS[parameter_name]
        parameter_catalog_rows.append(
            {
                "parameter": parameter_name,
                "group": spec["group"],
                "module": spec["module"],
                "title": spec["title"],
                "description": spec["description"],
                "learned_value": baseline_values[parameter_name],
            }
        )
    catalog_csv_path = safe_to_csv(
        pd.DataFrame(parameter_catalog_rows),
        output_dir / "mrph_parameter_catalog.csv",
    )

    all_rows = []
    for parameter_name in all_parameter_names:
        sweep_values = make_parameter_grid(
            baseline_values[parameter_name],
            parameter_name,
            num_points=cli_args.num_points,
            relative_span=cli_args.relative_span,
        )
        print(f"\nAnalyzing {parameter_name}, sweep values: {sweep_values}")

        for value in sweep_values:
            row = evaluate_parameter_setting(model_args, checkpoint_path, parameter_name, value)
            row["baseline_value"] = baseline_values[parameter_name]
            row["relative_to_baseline"] = row["value"] / baseline_values[parameter_name]
            row["is_baseline"] = int(np.isclose(row["value"], baseline_values[parameter_name], atol=1e-6))
            all_rows.append(row)
            print(
                f"  {parameter_name}={row['value']:.6f} | "
                f"test_nse={row['test_nse']:.6f} | "
                f"test_rmse={row['test_rmse']:.6f} | "
                f"test_r2={row['test_r2']:.6f}"
            )

    results_df = pd.DataFrame(all_rows)
    results_df = results_df.sort_values(by=["parameter_group", "parameter", "value"]).reset_index(drop=True)
    results_csv_path = safe_to_csv(results_df, output_dir / "mrph_parameter_sensitivity.csv")

    summary_rows = []
    for parameter_name in all_parameter_names:
        subset = results_df[results_df["parameter"] == parameter_name]
        best_row = subset.loc[subset["test_nse"].idxmax()]
        baseline_row = subset.loc[subset["is_baseline"] == 1].iloc[0]
        spec = PARAMETER_SPECS[parameter_name]
        summary_rows.append(
            {
                "parameter": parameter_name,
                "group": spec["group"],
                "module": spec["module"],
                "title": spec["title"],
                "description": spec["description"],
                "baseline_value": baseline_row["value"],
                "scan_min": subset["value"].min(),
                "scan_max": subset["value"].max(),
                "baseline_test_nse": baseline_row["test_nse"],
                "baseline_test_mse": baseline_row["test_mse"],
                "baseline_test_rmse": baseline_row["test_rmse"],
                "best_value_in_sweep": best_row["value"],
                "best_test_nse_in_sweep": best_row["test_nse"],
                "best_test_mse_in_sweep": best_row["test_mse"],
                "best_test_rmse_in_sweep": best_row["test_rmse"],
                "test_nse_range": subset["test_nse"].max() - subset["test_nse"].min(),
            }
        )

    summary_csv_path = safe_to_csv(
        pd.DataFrame(summary_rows), output_dir / "mrph_parameter_sensitivity_summary.csv"
    )
    plot_sensitivity(
        results_df,
        output_dir,
        PHYSICAL_PARAMETERS,
        "mrph_physical_parameter_sensitivity",
        "One-Factor-at-a-Time Sensitivity of MRPH-TimesNet Physical Parameters",
    )
    plot_sensitivity(
        results_df,
        output_dir,
        GATE_PARAMETERS,
        "mrph_gate_parameter_sensitivity",
        "One-Factor-at-a-Time Sensitivity of MRPH-TimesNet Fusion Gates",
    )

    print(f"\nSensitivity analysis completed. Results saved to: {output_dir}")
    print(f"Parameter catalog: {catalog_csv_path}")
    print(f"Detailed results table: {results_csv_path}")
    print(f"Summary table: {summary_csv_path}")
    return results_df


if __name__ == "__main__":
    run_sensitivity_analysis()
