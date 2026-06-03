"""
RF670 hydrological mechanism support experiment.

Reuses the main workflow from hydro_mechanism_analysis_610.py.
Only the default dataset, checkpoint, random seed, and output directory are changed.
"""
from pathlib import Path
import sys

import run_mrph_timesnet_670 as mrph_runner

sys.modules["run_mrph_timesnet"] = mrph_runner

import hydro_mechanism_analysis_610 as base

DEFAULT_OUTPUT_DIR_670 = Path("results/dl_result/hydro_mechanism_analysis_670")
DEFAULT_DATA_PATH_670 = Path("dataset/water_timeseries_670.csv")
DEFAULT_CHECKPOINT_670 = Path(
    "checkpoints/water_MRPH_TimesNet_670_MRPH_TimesNet_sl24_pl1_Comparison_Exp_0/checkpoint.pth"
)

base.OUTPUT_DIR = DEFAULT_OUTPUT_DIR_670
base.DEFAULT_DATA_PATH = DEFAULT_DATA_PATH_670
base.DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINT_670

_ORIGINAL_PARSE_ARGS = base.parse_args


def parse_args():
    args = _ORIGINAL_PARSE_ARGS()
    if args.data_path == "dataset/water_timeseries_610.csv":
        args.data_path = str(DEFAULT_DATA_PATH_670)
    if args.checkpoint == "checkpoints/water_MRPH_TimesNet_610_MRPH_TimesNet_sl24_pl1_Comparison_Exp_0/checkpoint.pth":
        args.checkpoint = str(DEFAULT_CHECKPOINT_670)
    if args.output_dir == "results/dl_result/hydro_mechanism_analysis":
        args.output_dir = str(DEFAULT_OUTPUT_DIR_670)
    if args.seed == 67:
        args.seed = 488
    return args


def build_model_args() -> object:
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


base.parse_args = parse_args
base.build_model_args = build_model_args


if __name__ == "__main__":
    base.run_analysis()
