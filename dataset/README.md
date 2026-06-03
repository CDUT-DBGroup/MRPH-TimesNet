# Dataset Notice

The original rainfall-inflow datasets used in this study are not included in this repository because they contain confidential underground mine groundwater monitoring data. The public repository only provides the code and the required data format.

## Dataset Access

The datasets may be made available for academic verification upon reasonable request and with the author's permission. Researchers who need access to the original data can contact the author by email:

```text
2024051108@stu.cdut.edu.cn
```

Data sharing is subject to institutional policies, confidentiality requirements, and approval from the data owner.

## Required File Names

To reproduce the experiments with your own data or with authorized data, place the CSV files in this directory using the following names:

```text
dataset/water_timeseries_610.csv
dataset/water_timeseries_670.csv
```

The released scripts read these file names by default. If different file names are used, the corresponding script arguments or default paths should be modified.

## Data Format

Each dataset should be a chronological multivariate time series CSV file. Each row represents one time step, and each column represents either the timestamp, a rainfall-related predictor, a water-level predictor, or the inflow target.

The expected column structure is:

```text
date
daily_rainfall
cum_rainfall_3d
cum_rainfall_7d
cum_rainfall_15d
cum_rainfall_30d
avg_water_level
water_inflow
```

## Column Description

| Column | Role | Description |
| --- | --- | --- |
| `date` | Time index | Observation date or timestamp. It should be parseable by `pandas.to_datetime`. |
| `daily_rainfall` | Input feature | Rainfall observed at the current time step. |
| `cum_rainfall_3d` | Input feature | Cumulative rainfall over the previous 3 days. |
| `cum_rainfall_7d` | Input feature | Cumulative rainfall over the previous 7 days. |
| `cum_rainfall_15d` | Input feature | Cumulative rainfall over the previous 15 days. |
| `cum_rainfall_30d` | Input feature | Cumulative rainfall over the previous 30 days. |
| `avg_water_level` | Input feature | Average groundwater or water-level observation used as a hydrological state variable. |
| `water_inflow` | Target | Mine groundwater inflow value to be predicted. |

## Formatting Requirements

- The file should be saved in CSV format.
- The first row should contain column names.
- The `date` column should be sorted in ascending chronological order.
- Missing values should be processed before training.
- The target column must be named `water_inflow`.
- The released experiments use multivariate input and single-output prediction.
- The released scripts use an 8:1:1 chronological split for training, validation, and testing.

## Example Schema

The following example only shows the required structure. The values are placeholders and are not the original data:

```csv
date,daily_rainfall,cum_rainfall_3d,cum_rainfall_7d,cum_rainfall_15d,cum_rainfall_30d,avg_water_level,water_inflow
2020-01-01,0.0,1.2,5.6,12.4,25.3,101.5,35.2
2020-01-02,2.4,3.6,6.1,13.0,26.7,101.8,36.0
2020-01-03,0.8,3.2,6.9,13.8,27.1,102.0,36.5
```


