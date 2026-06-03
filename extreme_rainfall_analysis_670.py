"""
Extreme rainfall event analysis for RF670.

Reuse the RF610 analysis workflow while switching defaults to the RF670 dataset.
"""
from pathlib import Path

import extreme_rainfall_analysis_610 as base

DEFAULT_OUTPUT_DIR_670 = Path("results/dl_result/extreme_rainfall_analysis_670")
DEFAULT_DATA_PATH_670 = Path("dataset/water_timeseries_670.csv")
DEFAULT_SUMMARY_CSV_670 = Path("results/dl_result/res_all/dl_comparison_670_with_mrph_ranking_by_nse.csv")
DEFAULT_SEED_670 = 488

base.OUTPUT_DIR = DEFAULT_OUTPUT_DIR_670
base.DEFAULT_DATA_PATH = DEFAULT_DATA_PATH_670
base.DEFAULT_SUMMARY_CSV = DEFAULT_SUMMARY_CSV_670

_ORIGINAL_PARSE_ARGS = base.parse_args


def parse_args():
    args = _ORIGINAL_PARSE_ARGS()
    if args.data_path == "dataset/water_timeseries_610.csv":
        args.data_path = str(DEFAULT_DATA_PATH_670)
    if args.summary_csv == "results/dl_result/res_all/dl_comparison_610_with_mrph_seed67_ranking_by_nse.csv":
        args.summary_csv = str(DEFAULT_SUMMARY_CSV_670)
    if args.output_dir == "results/dl_result/extreme_rainfall_analysis":
        args.output_dir = str(DEFAULT_OUTPUT_DIR_670)
    args.seed = DEFAULT_SEED_670
    return args


base.parse_args = parse_args


if __name__ == "__main__":
    base.run_analysis()
