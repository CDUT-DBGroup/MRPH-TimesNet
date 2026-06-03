"""
绘制测试集真实值与预测值的多子图拟合图。

功能说明：
- 从 `results/dl_result/res_all/dl_comparison_610_summary.csv` 读取各模型结果目录。
- 读取每个模型测试阶段生成的 `pred.npy` 和 `true.npy`。
- 将测试集上的真实涌水量与预测涌水量画到一张总图中，每个模型对应一个子图。
- 输出图片保存到 `results/dl_result/dl_comparison_610_test_fit_subplots.png`。
"""

import argparse
import math
import os
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_SUMMARY = Path("results/dl_result/res_all/dl_comparison_610_summary.csv")
DATA_PATH = Path("dataset/water_timeseries_610.csv")
OUTPUT_PATH = Path("results/dl_result/dl_comparison_610_test_fit_subplots.png")

SEQ_LEN = 24
PRED_LEN = 1
TARGET_COL = "water_inflow"
DATE_COL = "date"


def load_test_dates(data_path: Path, seq_len: int, pred_len: int) -> pd.Series:
    df = pd.read_csv(data_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    total_len = len(df)
    num_test = int(total_len * 0.1)
    border1 = total_len - num_test - seq_len
    target_start = border1 + seq_len
    target_end = total_len

    return df.iloc[target_start:target_end][DATE_COL].reset_index(drop=True)


def style_for_publication() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "figure.titlesize": 15,
            "axes.linewidth": 0.8,
            "savefig.dpi": 300,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot model fit subplots from summary CSV.")
    parser.add_argument("--summary", type=str, default=str(RESULTS_SUMMARY), help="summary csv path")
    parser.add_argument("--data-path", type=str, default=str(DATA_PATH), help="source dataset path")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH), help="output image path")
    parser.add_argument("--max-cols", type=int, default=3, help="maximum subplot columns")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    style_for_publication()
    summary_path = Path(args.summary)
    data_path = Path(args.data_path)
    output_path = Path(args.output)

    summary_df = pd.read_csv(summary_path)
    summary_df["nse"] = pd.to_numeric(summary_df["nse"], errors="coerce")
    summary_df["rmse"] = pd.to_numeric(summary_df["rmse"], errors="coerce")
    success_df = summary_df[summary_df["status"] == "Success"].sort_values(
        by="nse", ascending=False, na_position="last"
    )
    failed_df = summary_df[summary_df["status"] != "Success"]
    summary_df = pd.concat([success_df, failed_df], ignore_index=True)
    test_dates = load_test_dates(data_path, SEQ_LEN, PRED_LEN)

    n_models = len(summary_df)
    ncols = min(args.max_cols, max(1, n_models))
    nrows = math.ceil(n_models / ncols)

    series_cache = []
    global_min = None
    global_max = None
    for _, row in summary_df.iterrows():
        metrics_path = Path(str(row["metrics_path"]).replace("\\", os.sep))
        model_dir = metrics_path.parent
        pred_path = model_dir / "pred.npy"
        true_path = model_dir / "true.npy"

        cached = {"preds": None, "trues": None, "plot_len": 0}
        if row["status"] == "Success" and pred_path.exists() and true_path.exists():
            preds = np.load(pred_path).reshape(-1)
            trues = np.load(true_path).reshape(-1)
            plot_len = min(len(test_dates), len(preds), len(trues))
            preds = preds[:plot_len]
            trues = trues[:plot_len]
            cached = {"preds": preds, "trues": trues, "plot_len": plot_len}

            local_min = min(float(np.min(preds)), float(np.min(trues)))
            local_max = max(float(np.max(preds)), float(np.max(trues)))
            global_min = local_min if global_min is None else min(global_min, local_min)
            global_max = local_max if global_max is None else max(global_max, local_max)

        series_cache.append(cached)

    if global_min is None or global_max is None:
        global_min, global_max = 0.0, 1.0
    y_pad = max((global_max - global_min) * 0.08, 0.02)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(15.5, 3.6 * nrows),
        sharex=True,
        sharey=True,
    )
    axes = np.array(axes).reshape(-1)

    for idx, row in summary_df.iterrows():
        ax = axes[idx]
        model_name = row["model"]
        status = row["status"]
        nse = row.get("nse", np.nan)
        rmse = row.get("rmse", np.nan)

        rank_tag = f"({chr(97 + idx)})"
        metric_text = (
            f"NSE={float(nse):.4f} | RMSE={float(rmse):.4f}"
            if pd.notna(nse) and pd.notna(rmse)
            else "NSE/RMSE=N/A"
        )
        ax.set_title(f"{rank_tag} {model_name}\n{metric_text}", pad=8)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=25, labelbottom=True)
        ax.tick_params(axis="y")
        ax.set_ylim(global_min - y_pad, global_max + y_pad)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

        if idx % ncols == 0:
            ax.set_ylabel("Inflow")
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Date")

        cached = series_cache[idx]
        if status != "Success" or cached["plot_len"] == 0:
            ax.text(0.5, 0.5, f"{model_name}\nFailed", ha="center", va="center", fontsize=11)
            continue

        dates = test_dates.iloc[: cached["plot_len"]]
        ax.plot(dates, cached["trues"], label="Observed", linewidth=1.7, color="#1f4e79")
        ax.plot(dates, cached["preds"], label="Predicted", linewidth=1.5, color="#d55e00")

    for idx in range(n_models, len(axes)):
        fig.delaxes(axes[idx])

    handles = [
        plt.Line2D([0], [0], color="#1f4e79", linewidth=1.7, label="Observed"),
        plt.Line2D([0], [0], color="#d55e00", linewidth=1.5, label="Predicted"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.985))
    dataset_name = data_path.stem.replace("water_timeseries_", "").replace("_", " ")
    title_suffix = dataset_name if dataset_name else data_path.stem
    fig.suptitle(f"Test-Set Fit Comparison on the {title_suffix} Dataset", y=0.995)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.91, bottom=0.08, hspace=0.42, wspace=0.16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved subplot figure to: {output_path}")


if __name__ == "__main__":
    main()
