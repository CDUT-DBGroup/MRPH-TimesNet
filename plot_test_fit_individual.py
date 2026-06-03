"""
逐个模型绘制测试集真实值与预测值的拟合图。

功能说明：
- 从 `results/dl_result/dl_models_1step_summary.csv` 读取各模型结果目录。
- 读取每个模型测试阶段生成的 `pred.npy` 和 `true.npy`。
- 为每个模型单独生成一张测试集拟合图。
- 所有图片保存到 `results/dl_result/dl_models_test_fit_individual/` 目录。
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_SUMMARY = Path("results/dl_result/dl_models_1step_summary.csv")
DATA_PATH = Path("dataset/water_timeseries.csv")
OUTPUT_DIR = Path("results/dl_result/dl_models_test_fit_individual")

SEQ_LEN = 24
PRED_LEN = 1
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


def safe_filename(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")


def main() -> None:
    summary_df = pd.read_csv(RESULTS_SUMMARY)
    test_dates = load_test_dates(DATA_PATH, SEQ_LEN, PRED_LEN)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for _, row in summary_df.iterrows():
        model_name = row["model"]
        status = row["status"]
        metrics_path = Path(str(row["metrics_path"]))
        model_dir = metrics_path.parent
        pred_path = model_dir / "pred.npy"
        true_path = model_dir / "true.npy"
        output_path = OUTPUT_DIR / f"{safe_filename(model_name)}_test_fit.png"

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.set_title(f"{model_name} - Test Set Fit", fontsize=13)
        ax.set_xlabel("Date")
        ax.set_ylabel("water_inflow")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=30)

        if status != "Success" or not pred_path.exists() or not true_path.exists():
            ax.text(0.5, 0.5, f"{model_name}\nFailed", ha="center", va="center", fontsize=14)
            ax.grid(True, alpha=0.3)
        else:
            preds = np.load(pred_path).reshape(-1)
            trues = np.load(true_path).reshape(-1)
            plot_len = min(len(test_dates), len(preds), len(trues))
            dates = test_dates.iloc[:plot_len]

            ax.plot(dates, trues[:plot_len], label="True", linewidth=1.8, color="#1f77b4")
            ax.plot(dates, preds[:plot_len], label="Pred", linewidth=1.6, color="#ff7f0e")
            ax.grid(True, alpha=0.3)
            ax.legend()

        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
