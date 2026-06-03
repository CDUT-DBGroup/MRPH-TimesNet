"""
Physical consistency verification for RF670.

Reuse the RF610 workflow while switching all defaults to the RF670 experiment.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import physical_consistency_analysis_610 as base

OUTPUT_DIR_670 = Path("results/dl_result/physical_consistency_analysis_670")
DEFAULT_DATA_PATH_670 = Path("dataset/water_timeseries_670.csv")
DEFAULT_SEED_CSV_670 = Path("results/dl_result/res_all/mrph_seed_search_670_mse_lt_010.csv")
DEFAULT_SENSITIVITY_DIR_670 = Path("results/dl_result/res_all/sensitivity_analysis_670")
DEFAULT_HYDRO_DIR_670 = Path("results/dl_result/hydro_mechanism_analysis_670")
DEFAULT_MAIN_CHECKPOINT_670 = Path(
    "checkpoints/water_MRPH_TimesNet_670_MRPH_TimesNet_sl24_pl1_Comparison_Exp_0/checkpoint.pth"
)

base.OUTPUT_DIR = OUTPUT_DIR_670
base.DEFAULT_DATA_PATH = DEFAULT_DATA_PATH_670
base.DEFAULT_SEED_CSV = DEFAULT_SEED_CSV_670
base.DEFAULT_SENSITIVITY_DIR = DEFAULT_SENSITIVITY_DIR_670
base.DEFAULT_HYDRO_DIR = DEFAULT_HYDRO_DIR_670
base.DEFAULT_MAIN_CHECKPOINT = DEFAULT_MAIN_CHECKPOINT_670

_ORIGINAL_PARSE_ARGS = base.parse_args


def parse_args():
    args = _ORIGINAL_PARSE_ARGS()
    if args.data_path == "dataset/water_timeseries_610.csv":
        args.data_path = str(DEFAULT_DATA_PATH_670)
    if args.seed_csv == "results/dl_result/res_all/mrph_seed_search_mse_lt_010.csv":
        args.seed_csv = str(DEFAULT_SEED_CSV_670)
    if args.sensitivity_dir == "results/dl_result/res_all/sensitivity_analysis":
        args.sensitivity_dir = str(DEFAULT_SENSITIVITY_DIR_670)
    if args.hydro_dir == "results/dl_result/hydro_mechanism_analysis":
        args.hydro_dir = str(DEFAULT_HYDRO_DIR_670)
    if args.main_checkpoint == "checkpoints/water_MRPH_TimesNet_610_MRPH_TimesNet_sl24_pl1_Comparison_Exp_0/checkpoint.pth":
        args.main_checkpoint = str(DEFAULT_MAIN_CHECKPOINT_670)
    if args.output_dir == "results/dl_result/physical_consistency_analysis":
        args.output_dir = str(OUTPUT_DIR_670)
    return args


def build_model_args():
    args = base.argparse.Namespace(
        task_name="long_term_forecast",
        model="MRPH_TimesNet",
        data="custom",
        root_path="./dataset/",
        data_path="water_timeseries_670.csv",
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
        use_gpu=base.torch.cuda.is_available(),
        use_multi_gpu=False,
        gpu=0,
    )
    return args


def collect_checkpoint_paths(seed_csv: Path, checkpoints_root: Path, main_checkpoint: Path):
    seed_df = pd.read_csv(seed_csv)
    seeds = [int(seed) for seed in seed_df["seed"].tolist()]
    checkpoint_pairs: list[tuple[int, Path]] = []
    for seed in seeds:
        seed_ckpt = checkpoints_root / f"water_MRPH_TimesNet_670_MRPH_TimesNet_sl24_pl1_SeedSearch_{seed}_0" / "checkpoint.pth"
        if seed_ckpt.exists():
            checkpoint_pairs.append((seed, seed_ckpt))
        elif seed == 488 and main_checkpoint.exists():
            checkpoint_pairs.append((seed, main_checkpoint))
    if not checkpoint_pairs and main_checkpoint.exists():
        checkpoint_pairs.append((488, main_checkpoint))
    return checkpoint_pairs


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
        model = base.load_model(checkpoint_path, args.gpu)
        parameter_rows.append(base.extract_parameter_record(model, seed, checkpoint_path))

    parameter_df = pd.DataFrame(parameter_rows).sort_values(by="seed").reset_index(drop=True)
    parameter_df["beta_characteristic_lag"] = 1.0 / base.np.maximum(parameter_df["beta"].to_numpy(dtype=float), 1e-8)
    parameter_df["beta_half_life"] = base.np.log(2.0) / base.np.maximum(parameter_df["beta"].to_numpy(dtype=float), 1e-8)
    parameter_df.to_csv(output_dir / "parameter_seed_values.csv", index=False)

    interval_df = base.summarize_parameter_intervals(parameter_df)
    interval_df.to_csv(output_dir / "parameter_interval_summary.csv", index=False)
    base.plot_parameter_intervals(output_dir, parameter_df)
    base.plot_beta_timescales(output_dir, parameter_df)

    sensitivity_csv = sensitivity_dir / "mrph_parameter_sensitivity_670.csv"
    sensitivity_summary_csv = sensitivity_dir / "mrph_parameter_sensitivity_summary_670.csv"
    _, ranking_df = base.compute_sensitivity_ranking(sensitivity_csv, sensitivity_summary_csv)
    ranking_df.to_csv(output_dir / "parameter_sensitivity_ranking.csv", index=False)
    base.plot_sensitivity_ranking(output_dir, ranking_df)

    main_model = base.load_model(main_checkpoint, args.gpu)
    main_params = base.extract_parameter_record(main_model, 488, main_checkpoint)
    signal_df = base.compute_pchte_signals(base.load_dataset(data_path), main_params, args.max_lag)
    signal_df.to_csv(output_dir / "physical_response_series.csv", index=False)

    response_df, event_alignment_df, aggregate = base.build_physical_response_summary(
        signal_df=signal_df,
        hydro_dir=hydro_dir,
        max_lag=args.max_lag,
        event_quantile=args.event_quantile,
        merge_gap=args.merge_gap,
        baseline_window=args.baseline_window,
    )
    response_df.to_csv(output_dir / "physical_response_summary.csv", index=False)
    event_alignment_df.to_csv(output_dir / "physical_response_event_alignment.csv", index=False)
    base.plot_physical_response(output_dir, signal_df, response_df, main_params, aggregate)

    report = {
        "checkpoint_count_for_interval_analysis": int(len(parameter_df)),
        "seeds_used": [int(seed) for seed in parameter_df["seed"].tolist()],
        "main_checkpoint": str(main_checkpoint),
        "main_seed": 488,
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
