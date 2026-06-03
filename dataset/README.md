# Dataset Notice

The original rainfall-inflow datasets used in this study are not included in this repository because they contain confidential underground mine groundwater monitoring data.

To reproduce the experiments, please prepare your own rainfall-inflow time series data with the same file names and column format:

```text
dataset/water_timeseries_610.csv
dataset/water_timeseries_670.csv
```

Each CSV file should contain a `date` column, rainfall-related input variables, water level variables, and the target variable `water_inflow`.

The expected column structure used by the released scripts is:

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

Notes:

- `date` should be a valid timestamp or date string.
- `water_inflow` is the prediction target.
- Other columns are used as input variables for multivariate forecasting.
- The data should be sorted in chronological order.
- The released scripts use an 8:1:1 chronological split for training, validation, and testing.

If the original data are required for academic verification, please contact the corresponding author or data owner subject to institutional data-sharing policies.

