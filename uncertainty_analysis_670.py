"""
RF670 uncertainty analysis experiment.

Reuses the main workflow from uncertainty_analysis_610.py.
Only the default dataset, checkpoints, ensemble seeds, and random seed configuration are changed.
"""
from pathlib import Path
import sys

import run_mrph_timesnet_670 as mrph_runner

# Redirect set_random_seed to the 670 runner before reusing the 610 analysis script.
sys.modules["run_mrph_timesnet"] = mrph_runner

import uncertainty_analysis_610 as base

DEFAULT_OUTPUT_DIR_670 = Path("results/dl_result/uncertainty_analysis_670")
DEFAULT_MC_CHECKPOINT_670 = Path(
    "checkpoints/water_MRPH_TimesNet_670_MRPH_TimesNet_sl24_pl1_Comparison_Exp_0/checkpoint.pth"
)
DEFAULT_ENSEMBLE_SEEDS_670 = [216, 237, 309, 488, 666]
DEFAULT_ENSEMBLE_PATTERN_670 = (
    "checkpoints/water_MRPH_TimesNet_670_MRPH_TimesNet_sl24_pl1_SeedSearch_{seed}_0/checkpoint.pth"
)

base.OUTPUT_DIR = DEFAULT_OUTPUT_DIR_670
base.DEFAULT_MC_CHECKPOINT = DEFAULT_MC_CHECKPOINT_670
base.DEFAULT_ENSEMBLE_SEEDS = DEFAULT_ENSEMBLE_SEEDS_670
base.DEFAULT_ENSEMBLE_PATTERN = DEFAULT_ENSEMBLE_PATTERN_670

_ORIGINAL_PARSE_ARGS = base.parse_args
_ORIGINAL_BUILD_MODEL_ARGS = base.build_model_args


def parse_args():
    cli_args = _ORIGINAL_PARSE_ARGS()

    if cli_args.output_dir == "results/dl_result/uncertainty_analysis":
        cli_args.output_dir = str(DEFAULT_OUTPUT_DIR_670)
    if cli_args.data_path == "water_timeseries_610.csv":
        cli_args.data_path = "water_timeseries_670.csv"
    if cli_args.base_seed == 2178:
        cli_args.base_seed = 488
    if str(cli_args.mc_checkpoint) == str(
        Path("checkpoints/water_MRPH_TimesNet_610_MRPH_TimesNet_sl24_pl1_Comparison_Exp_0/checkpoint.pth")
    ):
        cli_args.mc_checkpoint = str(DEFAULT_MC_CHECKPOINT_670)
    if cli_args.ensemble_seeds == [67, 95, 170, 249, 285]:
        cli_args.ensemble_seeds = list(DEFAULT_ENSEMBLE_SEEDS_670)
    if (
        cli_args.ensemble_checkpoint_pattern
        == "checkpoints/water_MRPH_TimesNet_610_MRPH_TimesNet_sl24_pl1_SeedSearch_{seed}_0/checkpoint.pth"
    ):
        cli_args.ensemble_checkpoint_pattern = DEFAULT_ENSEMBLE_PATTERN_670

    return cli_args


def build_model_args(cli_args, seed: int):
    args = _ORIGINAL_BUILD_MODEL_ARGS(cli_args, seed)
    args.model_id = "water_MRPH_TimesNet_670"
    args.data_path = cli_args.data_path
    return args


base.parse_args = parse_args
base.build_model_args = build_model_args


if __name__ == "__main__":
    base.run_uncertainty_analysis()
