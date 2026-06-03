"""
Deep learning model comparison script for TimesNet-related baselines.
"""
import glob
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from utils.metrics import NSE

DL_RESULTS_DIR = Path("results") / "dl_result"
DL_SUMMARY_DIR = DL_RESULTS_DIR / "res_all"

# Available deep learning models
MODEL_SPECS = [
    {"run_name": "TimesNet", "display_name": "TimesNet"},
    {"run_name": "DLinear", "display_name": "DLinear"},
    {"run_name": "Transformer", "display_name": "Transformer"},
    {"run_name": "Informer", "display_name": "Informer"},
    {"run_name": "Autoformer", "display_name": "Autoformer"},
    {"run_name": "FEDformer", "display_name": "FEDformer"},
    {"run_name": "LSTM", "display_name": "LSTM"},
    {"run_name": "CNN_LSTM", "display_name": "CNN-LSTM"},
]

# Dataset configuration
DATASETS = [
    ('water_timeseries_610.csv', '610'),
]

def infer_feature_dims(root_path, data_path, target, features):
    """Infer enc/dec/c_out from current csv schema."""
    csv_path = os.path.join(root_path, data_path)
    df_head = pd.read_csv(csv_path, nrows=1)
    value_cols = [c for c in df_head.columns if c != "date"]
    if features == "S":
        return 1, 1, 1
    if target not in value_cols:
        raise ValueError(f"target '{target}' not found in {csv_path}")
    enc_in = len(value_cols)
    dec_in = enc_in
    c_out = 1 if features == "MS" else enc_in
    return enc_in, dec_in, c_out


def print_dataset_time_spans(root_path, data_path, seq_len):
    csv_path = os.path.join(root_path, data_path)
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])

    total_rows = len(df)
    num_train = int(total_rows * 0.8)
    num_test = int(total_rows * 0.1)
    num_vali = total_rows - num_train - num_test

    border1s = [0, num_train - seq_len, total_rows - num_test - seq_len]
    border2s = [num_train, num_train + num_vali, total_rows]

    split_names = ["train", "val", "test"]
    print("  Data time span (8:1:1):")
    for split_name, b1, b2 in zip(split_names, border1s, border2s):
        split_dates = df.iloc[b1:b2]["date"]
        print(
            f"    - {split_name}: rows={len(split_dates)}, "
            f"start={split_dates.iloc[0].strftime('%Y-%m-%d')}, "
            f"end={split_dates.iloc[-1].strftime('%Y-%m-%d')}"
        )


def find_latest_metrics_path(run_name, dataset_label):
    pattern = os.path.join(
        str(DL_RESULTS_DIR),
        "test_results",
        f"*water_{run_name}_{dataset_label}_{run_name}_custom_ftMS_sl24_ll1_pl1_dm32_nh8_el2_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_Comparison_Exp_0",
        "metrics.npy",
    )
    matches = glob.glob(pattern)
    if not matches:
        return ""
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


def run_model(model_spec, data_path, dataset_label):
    """Run a single model."""
    run_name = model_spec["run_name"]
    display_name = model_spec["display_name"]
    print(f"\nRunning model: {display_name} on {dataset_label}")
    root_path = "./dataset/"
    features = "MS"
    target = "water_inflow"
    seq_len = "24"
    enc_in, dec_in, c_out = infer_feature_dims(root_path, data_path, target, features)
    print(f"  Inferred dimensions: enc_in={enc_in}, dec_in={dec_in}, c_out={c_out}")
    print_dataset_time_spans(root_path, data_path, int(seq_len))

    cmd = [
        sys.executable, "-u", "run.py",
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--root_path", root_path,
        "--data_path", data_path,
        "--model_id", f"water_{run_name}_{dataset_label}",
        "--model", run_name,
        "--data", "custom",
        "--features", features,
        "--target", target,
        "--seq_len", seq_len,
        "--label_len", "1",
        "--pred_len", "1",
        "--enc_in", str(enc_in),
        "--dec_in", str(dec_in),
        "--c_out", str(c_out),
        "--d_model", "32",
        "--n_heads", "8",
        "--e_layers", "2",
        "--d_layers", "1",
        "--d_ff", "32",
        "--moving_avg", "25",
        "--factor", "3",
        "--dropout", "0.1",
        "--embed", "timeF",
        "--freq", "d",
        "--batch_size", "32",
        "--learning_rate", "0.001",
        "--num_workers", "0",
        "--itr", "1",
        "--train_epochs", "150",
        "--patience", "10",
        "--seed", "67",
        "--des", "Comparison_Exp",
        "--gpu", "0"
    ]
    
    # Inherit parent stdout/stderr so epoch logs stream in real time.
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  Command failed with return code: {result.returncode}")
    return result.returncode == 0

def main():
    print("=" * 60)
    print("Starting deep learning model comparison experiment")
    print("=" * 60)
    
    results = {}
    metrics_rows = []
    summary_dir = DL_SUMMARY_DIR
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    for data_path, dataset_label in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_label}")
        print(f"{'='*60}")

        results[dataset_label] = {}
        
        for model_spec in MODEL_SPECS:
            run_name = model_spec["run_name"]
            display_name = model_spec["display_name"]
            success = run_model(model_spec, data_path, dataset_label)
            results[dataset_label][display_name] = "Success" if success else "Failed"
            print(f"  {display_name}: {'Success' if success else 'Failed'}")

            metrics_path = find_latest_metrics_path(run_name, dataset_label) if success else ""
            row = {
                "dataset": dataset_label,
                "model": display_name,
                "status": "Success" if success else "Failed",
                "nse": np.nan,
                "mse": np.nan,
                "mae": np.nan,
                "rmse": np.nan,
                "mape": np.nan,
                "mspe": np.nan,
                "metrics_path": metrics_path,
            }
            if metrics_path and os.path.exists(metrics_path):
                metrics = np.load(metrics_path)
                mae, mse, rmse, mape, mspe = metrics
                model_dir = Path(metrics_path).parent
                pred_path = model_dir / "pred.npy"
                true_path = model_dir / "true.npy"
                nse = np.nan
                if pred_path.exists() and true_path.exists():
                    preds = np.load(pred_path)
                    trues = np.load(true_path)
                    nse = float(NSE(preds, trues))
                row.update({
                    "nse": nse,
                    "mse": float(mse),
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "mape": float(mape),
                    "mspe": float(mspe),
                })
            metrics_rows.append(row)
    
    # Save result summary
    results_df = pd.DataFrame(results).T
    results_df.to_csv(summary_dir / 'dl_models_status.csv')

    column_order = ["dataset", "model", "status", "nse", "mse", "mae", "rmse", "mape", "mspe", "metrics_path"]
    metrics_df = pd.DataFrame(metrics_rows)[column_order]
    metrics_df.to_csv(summary_dir / 'dl_comparison_610_summary.csv', index=False)
    success_df = metrics_df[metrics_df["status"] == "Success"].sort_values(by="nse", ascending=False)
    success_df.to_csv(summary_dir / 'dl_comparison_610_ranking_by_nse.csv', index=False)
    
    print("\n" + "=" * 60)
    print("Deep learning model comparison completed")
    print("=" * 60)

if __name__ == '__main__':
    main()
