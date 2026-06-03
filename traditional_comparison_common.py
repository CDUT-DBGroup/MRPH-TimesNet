import argparse
import json
import math
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from utils.metrics import NSE
import xgboost as xgb

warnings.filterwarnings("ignore")

DATE_COL = "date"
TARGET_COL = "water_inflow"

OBS_COLOR = "#1f4e79"
PRED_COLOR = "#d55e00"


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
            "axes.linewidth": 0.8,
            "savefig.dpi": 350,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output_path: Path, dpi: int = 350) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")


@dataclass
class DatasetBundle:
    df: pd.DataFrame
    feature_cols: list[str]
    train_X: np.ndarray
    val_X: np.ndarray
    test_X: np.ndarray
    train_y: np.ndarray
    val_y: np.ndarray
    test_y: np.ndarray
    train_dates: pd.Series
    val_dates: pd.Series
    test_dates: pd.Series
    train_exog: np.ndarray
    val_exog: np.ndarray
    test_exog: np.ndarray
    sarimax_exog_cols: list[str]
    rain_feature_name: str
    train_rain_X: np.ndarray
    val_rain_X: np.ndarray
    test_rain_X: np.ndarray
    target_mean: float
    target_scale: float
    train_series: np.ndarray
    val_series: np.ndarray
    test_series: np.ndarray


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=float).reshape(-1)
    true = np.asarray(true, dtype=float).reshape(-1)
    mse = float(mean_squared_error(true, pred))
    mae = float(mean_absolute_error(true, pred))
    rmse = float(np.sqrt(mse))
    nse = float(NSE(pred, true))
    denom = np.maximum(np.abs(true), 1e-8)
    mape = float(np.mean(np.abs((pred - true) / denom)))
    return {
        "nse": nse,
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }


def build_supervised_split(
    features_scaled: np.ndarray,
    target_values: np.ndarray,
    dates: pd.Series,
    feature_rows: np.ndarray,
    target_indices: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, pd.Series, np.ndarray]:
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    date_list: list[pd.Timestamp] = []
    exog_list: list[np.ndarray] = []
    for target_idx in target_indices:
        X_list.append(features_scaled[target_idx - seq_len:target_idx].reshape(-1))
        y_list.append(float(target_values[target_idx]))
        date_list.append(dates.iloc[target_idx])
        exog_list.append(feature_rows[target_idx])
    return (
        np.asarray(X_list, dtype=float),
        np.asarray(y_list, dtype=float),
        pd.Series(date_list),
        np.asarray(exog_list, dtype=float),
    )


def build_univariate_window_split(
    series: np.ndarray,
    target_indices: np.ndarray,
    seq_len: int,
) -> np.ndarray:
    X_list: list[np.ndarray] = []
    for target_idx in target_indices:
        X_list.append(np.asarray(series[target_idx - seq_len:target_idx], dtype=float))
    return np.asarray(X_list, dtype=float)


def load_dataset_bundle(data_path: Path, seq_len: int) -> DatasetBundle:
    df = pd.read_csv(data_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    feature_cols = [c for c in df.columns if c not in {DATE_COL, TARGET_COL}]

    total_len = len(df)
    num_train = int(total_len * 0.8)
    num_test = int(total_len * 0.1)
    num_val = total_len - num_train - num_test

    feature_values = df[feature_cols].to_numpy(dtype=float)
    target_values_raw = df[TARGET_COL].to_numpy(dtype=float)

    scaler = StandardScaler()
    scaler.fit(feature_values[:num_train])
    features_scaled = scaler.transform(feature_values)

    target_mean = float(np.mean(target_values_raw[:num_train]))
    target_scale = float(np.std(target_values_raw[:num_train], ddof=0))
    if abs(target_scale) < 1e-12:
        target_scale = 1.0
    target_values = (target_values_raw - target_mean) / target_scale

    train_indices = np.arange(seq_len, num_train)
    val_indices = np.arange(num_train, num_train + num_val)
    test_indices = np.arange(total_len - num_test, total_len)

    preferred_sarimax_cols = [c for c in ["daily_rainfall", "cum_rainfall_7d", "avg_water_level"] if c in feature_cols]
    sarimax_exog_cols = preferred_sarimax_cols if preferred_sarimax_cols else feature_cols[: min(3, len(feature_cols))]
    sarimax_exog_idx = [feature_cols.index(c) for c in sarimax_exog_cols]
    sarimax_exog_values = features_scaled[:, sarimax_exog_idx]
    rain_feature_name = "daily_rainfall" if "daily_rainfall" in feature_cols else feature_cols[0]
    rain_idx = feature_cols.index(rain_feature_name)
    rain_series_scaled = features_scaled[:, rain_idx]

    train_X, train_y, train_dates, _ = build_supervised_split(
        features_scaled, target_values, df[DATE_COL], sarimax_exog_values, train_indices, seq_len
    )
    val_X, val_y, val_dates, _ = build_supervised_split(
        features_scaled, target_values, df[DATE_COL], sarimax_exog_values, val_indices, seq_len
    )
    test_X, test_y, test_dates, _ = build_supervised_split(
        features_scaled, target_values, df[DATE_COL], sarimax_exog_values, test_indices, seq_len
    )
    train_rain_X = build_univariate_window_split(rain_series_scaled, train_indices, seq_len)
    val_rain_X = build_univariate_window_split(rain_series_scaled, val_indices, seq_len)
    test_rain_X = build_univariate_window_split(rain_series_scaled, test_indices, seq_len)

    train_exog = sarimax_exog_values[:num_train]
    val_exog = sarimax_exog_values[num_train:num_train + num_val]
    test_exog = sarimax_exog_values[total_len - num_test:]

    return DatasetBundle(
        df=df,
        feature_cols=feature_cols,
        train_X=train_X,
        val_X=val_X,
        test_X=test_X,
        train_y=train_y,
        val_y=val_y,
        test_y=test_y,
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
        train_exog=train_exog,
        val_exog=val_exog,
        test_exog=test_exog,
        sarimax_exog_cols=sarimax_exog_cols,
        rain_feature_name=rain_feature_name,
        train_rain_X=train_rain_X,
        val_rain_X=val_rain_X,
        test_rain_X=test_rain_X,
        target_mean=target_mean,
        target_scale=target_scale,
        train_series=target_values[:num_train],
        val_series=target_values[num_train:num_train + num_val],
        test_series=target_values[total_len - num_test:],
    )


def walk_forward_arima(history: np.ndarray, future: np.ndarray, order: tuple[int, int, int]) -> np.ndarray:
    preds: list[float] = []
    hist = list(np.asarray(history, dtype=float))
    for actual in np.asarray(future, dtype=float):
        model = ARIMA(hist, order=order, enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit()
        forecast = float(fitted.forecast(steps=1)[0])
        preds.append(forecast)
        hist.append(float(actual))
    return np.asarray(preds, dtype=float)


def fit_once_arima_forecast(
    history: np.ndarray,
    history_exog: np.ndarray,
    future_exog: np.ndarray,
    order: tuple[int, int, int],
) -> np.ndarray:
    model = ARIMA(
        np.asarray(history, dtype=float),
        exog=np.asarray(history_exog, dtype=float),
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit()
    pred = fitted.forecast(steps=len(future_exog), exog=np.asarray(future_exog, dtype=float))
    return np.asarray(pred, dtype=float).reshape(-1)


def walk_forward_sarimax(
    history: np.ndarray,
    history_exog: np.ndarray,
    future: np.ndarray,
    future_exog: np.ndarray,
    order: tuple[int, int, int],
) -> np.ndarray:
    preds: list[float] = []
    hist = list(np.asarray(history, dtype=float))
    exog_hist = np.asarray(history_exog, dtype=float).copy()
    for actual, exog_row in zip(np.asarray(future, dtype=float), np.asarray(future_exog, dtype=float)):
        model = SARIMAX(
            hist,
            exog=exog_hist,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
            trend="c",
        )
        fitted = model.fit(disp=False)
        forecast = float(fitted.forecast(steps=1, exog=np.asarray(exog_row, dtype=float).reshape(1, -1))[0])
        preds.append(forecast)
        hist.append(float(actual))
        exog_hist = np.vstack([exog_hist, exog_row])
    return np.asarray(preds, dtype=float)


def fit_once_sarimax_forecast(
    history: np.ndarray,
    history_exog: np.ndarray,
    future_exog: np.ndarray,
    order: tuple[int, int, int],
) -> np.ndarray:
    model = SARIMAX(
        np.asarray(history, dtype=float),
        exog=np.asarray(history_exog, dtype=float),
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
        trend="c",
    )
    fitted = model.fit(disp=False)
    pred = fitted.forecast(steps=len(future_exog), exog=np.asarray(future_exog, dtype=float))
    return np.asarray(pred, dtype=float).reshape(-1)


def gm11_predict_next(series: np.ndarray) -> float:
    x0 = np.asarray(series, dtype=float).reshape(-1)
    if len(x0) < 4:
        raise ValueError("GM(1,1) requires at least 4 observations.")
    x1 = np.cumsum(x0)
    z1 = 0.5 * (x1[1:] + x1[:-1])
    B = np.column_stack((-z1, np.ones(len(z1))))
    Y = x0[1:].reshape(-1, 1)
    a, b = np.linalg.lstsq(B, Y, rcond=None)[0].reshape(-1)
    if abs(a) < 1e-8:
        return float(x0[-1])
    k = len(x0) + 1
    x1_next = (x0[0] - b / a) * math.exp(-a * (k - 1)) + b / a
    x1_prev = (x0[0] - b / a) * math.exp(-a * (k - 2)) + b / a
    return float(x1_next - x1_prev)


def walk_forward_gm11(history: np.ndarray, future: np.ndarray, window_size: int) -> np.ndarray:
    preds: list[float] = []
    hist = list(np.asarray(history, dtype=float))
    for actual in np.asarray(future, dtype=float):
        window = np.asarray(hist[-window_size:], dtype=float)
        if len(window) < 4:
            window = np.asarray(hist, dtype=float)
        preds.append(gm11_predict_next(window))
        hist.append(float(actual))
    return np.asarray(preds, dtype=float)


def select_best_svr(bundle: DatasetBundle, seed: int) -> tuple[SVR, dict[str, float]]:
    best_model = None
    best_params = None
    best_mse = float("inf")
    grid = [
        {"C": 1.0, "epsilon": 0.05, "gamma": "scale"},
        {"C": 5.0, "epsilon": 0.05, "gamma": "scale"},
        {"C": 10.0, "epsilon": 0.05, "gamma": "scale"},
        {"C": 10.0, "epsilon": 0.02, "gamma": "scale"},
        {"C": 10.0, "epsilon": 0.05, "gamma": "auto"},
    ]
    for params in grid:
        model = SVR(kernel="rbf", **params)
        model.fit(bundle.train_X, bundle.train_y)
        val_pred = model.predict(bundle.val_X)
        mse = mean_squared_error(bundle.val_y, val_pred)
        if mse < best_mse:
            best_mse = float(mse)
            best_model = model
            best_params = params
    assert best_model is not None and best_params is not None
    final_model = SVR(kernel="rbf", **best_params)
    final_model.fit(
        np.vstack([bundle.train_X, bundle.val_X]),
        np.concatenate([bundle.train_y, bundle.val_y]),
    )
    return final_model, {"best_val_mse": best_mse, **best_params, "seed": seed}


def select_best_xgboost(bundle: DatasetBundle, seed: int) -> tuple[xgb.XGBRegressor, dict[str, float]]:
    best_params = None
    best_mse = float("inf")
    grid = [
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8},
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1, "subsample": 1.0},
    ]
    for params in grid:
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=1,
            **params,
        )
        model.fit(bundle.train_X, bundle.train_y)
        val_pred = model.predict(bundle.val_X)
        mse = mean_squared_error(bundle.val_y, val_pred)
        if mse < best_mse:
            best_mse = float(mse)
            best_params = params
    assert best_params is not None
    final_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=1,
        **best_params,
    )
    final_model.fit(
        np.vstack([bundle.train_X, bundle.val_X]),
        np.concatenate([bundle.train_y, bundle.val_y]),
    )
    return final_model, {"best_val_mse": best_mse, **best_params, "seed": seed}


def select_best_random_forest(bundle: DatasetBundle, seed: int) -> tuple[RandomForestRegressor, dict[str, float]]:
    best_params = None
    best_mse = float("inf")
    grid = [
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 1},
        {"n_estimators": 500, "max_depth": 8, "min_samples_leaf": 2},
        {"n_estimators": 500, "max_depth": 12, "min_samples_leaf": 2},
    ]
    for params in grid:
        model = RandomForestRegressor(random_state=seed, n_jobs=-1, **params)
        model.fit(bundle.train_X, bundle.train_y)
        val_pred = model.predict(bundle.val_X)
        mse = mean_squared_error(bundle.val_y, val_pred)
        if mse < best_mse:
            best_mse = float(mse)
            best_params = params
    assert best_params is not None
    final_model = RandomForestRegressor(random_state=seed, n_jobs=-1, **best_params)
    final_model.fit(
        np.vstack([bundle.train_X, bundle.val_X]),
        np.concatenate([bundle.train_y, bundle.val_y]),
    )
    return final_model, {"best_val_mse": best_mse, **best_params, "seed": seed}


def select_best_unit_hydrograph(bundle: DatasetBundle, seed: int) -> tuple[LinearRegression, dict[str, float]]:
    best_model = None
    best_window = None
    best_mse = float("inf")
    candidate_windows = [7, 15, 24, 30]
    for window in candidate_windows:
        if window > bundle.train_rain_X.shape[1]:
            continue
        train_X = bundle.train_rain_X[:, -window:]
        val_X = bundle.val_rain_X[:, -window:]
        model = LinearRegression(positive=True)
        model.fit(train_X, bundle.train_y)
        val_pred = model.predict(val_X)
        mse = mean_squared_error(bundle.val_y, val_pred)
        if mse < best_mse:
            best_mse = float(mse)
            best_window = window
            best_model = model
    assert best_model is not None and best_window is not None
    final_model = LinearRegression(positive=True)
    final_model.fit(
        np.vstack([bundle.train_rain_X[:, -best_window:], bundle.val_rain_X[:, -best_window:]]),
        np.concatenate([bundle.train_y, bundle.val_y]),
    )
    metadata = {
        "best_val_mse": best_mse,
        "window_size": best_window,
        "seed": seed,
        "rain_feature": bundle.rain_feature_name,
        "unit_hydrograph_weights": final_model.coef_.tolist(),
    }
    return final_model, metadata


def run_linear_regression(bundle: DatasetBundle, seed: int) -> dict:
    model = LinearRegression()
    model.fit(bundle.train_X, bundle.train_y)
    val_pred = model.predict(bundle.val_X)
    final_model = LinearRegression()
    final_model.fit(
        np.vstack([bundle.train_X, bundle.val_X]),
        np.concatenate([bundle.train_y, bundle.val_y]),
    )
    test_pred = final_model.predict(bundle.test_X)
    return {
        "model_name": "LinearRegression",
        "val_pred": val_pred,
        "test_pred": test_pred,
        "metadata": {"seed": seed},
    }


def run_svr(bundle: DatasetBundle, seed: int) -> dict:
    model, metadata = select_best_svr(bundle, seed)
    val_model = SVR(kernel="rbf", C=metadata["C"], epsilon=metadata["epsilon"], gamma=metadata["gamma"])
    val_model.fit(bundle.train_X, bundle.train_y)
    val_pred = val_model.predict(bundle.val_X)
    test_pred = model.predict(bundle.test_X)
    return {
        "model_name": "SVR",
        "val_pred": val_pred,
        "test_pred": test_pred,
        "metadata": metadata,
    }


def run_xgboost(bundle: DatasetBundle, seed: int) -> dict:
    model, metadata = select_best_xgboost(bundle, seed)
    val_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=1,
        n_estimators=metadata["n_estimators"],
        max_depth=metadata["max_depth"],
        learning_rate=metadata["learning_rate"],
        subsample=metadata["subsample"],
    )
    val_model.fit(bundle.train_X, bundle.train_y)
    val_pred = val_model.predict(bundle.val_X)
    test_pred = model.predict(bundle.test_X)
    return {
        "model_name": "XGBoost",
        "val_pred": val_pred,
        "test_pred": test_pred,
        "metadata": metadata,
    }


def run_random_forest(bundle: DatasetBundle, seed: int) -> dict:
    model, metadata = select_best_random_forest(bundle, seed)
    val_model = RandomForestRegressor(
        random_state=seed,
        n_jobs=-1,
        n_estimators=metadata["n_estimators"],
        max_depth=metadata["max_depth"],
        min_samples_leaf=metadata["min_samples_leaf"],
    )
    val_model.fit(bundle.train_X, bundle.train_y)
    val_pred = val_model.predict(bundle.val_X)
    test_pred = model.predict(bundle.test_X)
    return {
        "model_name": "RandomForest",
        "val_pred": val_pred,
        "test_pred": test_pred,
        "metadata": metadata,
    }


def run_unit_hydrograph(bundle: DatasetBundle, seed: int) -> dict:
    model, metadata = select_best_unit_hydrograph(bundle, seed)
    window = int(metadata["window_size"])
    val_model = LinearRegression(positive=True)
    val_model.fit(bundle.train_rain_X[:, -window:], bundle.train_y)
    val_pred = val_model.predict(bundle.val_rain_X[:, -window:])
    test_pred = model.predict(bundle.test_rain_X[:, -window:])
    return {
        "model_name": "UnitHydrograph",
        "val_pred": val_pred,
        "test_pred": test_pred,
        "metadata": metadata,
    }


def run_arima(bundle: DatasetBundle, seed: int) -> dict:
    candidate_orders = [(1, 0, 0), (2, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 1), (3, 1, 1)]
    best_order = None
    best_val_pred = None
    best_mse = float("inf")
    for order in candidate_orders:
        try:
            val_pred = fit_once_arima_forecast(
                bundle.train_series,
                bundle.train_exog,
                bundle.val_exog,
                order,
            )
            mse = mean_squared_error(bundle.val_y, val_pred)
            if mse < best_mse:
                best_mse = float(mse)
                best_order = order
                best_val_pred = val_pred
        except Exception:
            continue
    if best_order is None or best_val_pred is None:
        raise RuntimeError("ARIMA failed for all candidate orders.")
    history = np.concatenate([bundle.train_series, bundle.val_series])
    history_exog = np.vstack([bundle.train_exog, bundle.val_exog])
    test_pred = fit_once_arima_forecast(
        history,
        history_exog,
        bundle.test_exog,
        best_order,
    )
    return {
        "model_name": "ARIMA",
        "val_pred": best_val_pred,
        "test_pred": test_pred,
        "metadata": {
            "seed": seed,
            "order": best_order,
            "best_val_mse": best_mse,
            "exog_cols": bundle.sarimax_exog_cols,
        },
    }


def run_sarimax(bundle: DatasetBundle, seed: int) -> dict:
    candidate_orders = [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 1)]
    best_order = None
    best_val_pred = None
    best_mse = float("inf")
    for order in candidate_orders:
        try:
            val_pred = fit_once_sarimax_forecast(
                bundle.train_series,
                bundle.train_exog,
                bundle.val_exog,
                order,
            )
            mse = mean_squared_error(bundle.val_y, val_pred)
            if mse < best_mse:
                best_mse = float(mse)
                best_order = order
                best_val_pred = val_pred
        except Exception:
            continue
    if best_order is None or best_val_pred is None:
        raise RuntimeError("SARIMAX failed for all candidate orders.")
    history = np.concatenate([bundle.train_series, bundle.val_series])
    history_exog = np.vstack([bundle.train_exog, bundle.val_exog])
    test_pred = fit_once_sarimax_forecast(
        history,
        history_exog,
        bundle.test_exog,
        best_order,
    )
    return {
        "model_name": "SARIMAX",
        "val_pred": best_val_pred,
        "test_pred": test_pred,
        "metadata": {
            "seed": seed,
            "order": best_order,
            "best_val_mse": best_mse,
            "exog_cols": bundle.sarimax_exog_cols,
        },
    }


def run_gm11(bundle: DatasetBundle, seed: int) -> dict:
    candidate_windows = [6, 8, 12, 24, 30]
    best_window = None
    best_val_pred = None
    best_mse = float("inf")
    for window_size in candidate_windows:
        try:
            val_pred = walk_forward_gm11(bundle.train_series, bundle.val_series, window_size)
            mse = mean_squared_error(bundle.val_y, val_pred)
            if mse < best_mse:
                best_mse = float(mse)
                best_window = window_size
                best_val_pred = val_pred
        except Exception:
            continue
    if best_window is None or best_val_pred is None:
        raise RuntimeError("GM11 failed for all candidate windows.")
    history = np.concatenate([bundle.train_series, bundle.val_series])
    test_pred = walk_forward_gm11(history, bundle.test_series, best_window)
    return {
        "model_name": "GM11",
        "val_pred": best_val_pred,
        "test_pred": test_pred,
        "metadata": {"seed": seed, "window_size": best_window, "best_val_mse": best_mse},
    }


def save_model_outputs(
    model_dir: Path,
    model_name: str,
    val_dates: pd.Series,
    val_true: np.ndarray,
    val_pred: np.ndarray,
    test_dates: pd.Series,
    test_true: np.ndarray,
    test_pred: np.ndarray,
    target_mean: float,
    target_scale: float,
    metadata: dict,
) -> dict[str, float]:
    model_dir.mkdir(parents=True, exist_ok=True)
    apply_publication_style()
    val_metrics = compute_metrics(val_pred, val_true)
    test_metrics = compute_metrics(test_pred, test_true)

    val_true_inverse = np.asarray(val_true, dtype=float) * target_scale + target_mean
    val_pred_inverse = np.asarray(val_pred, dtype=float) * target_scale + target_mean
    test_true_inverse = np.asarray(test_true, dtype=float) * target_scale + target_mean
    test_pred_inverse = np.asarray(test_pred, dtype=float) * target_scale + target_mean

    pd.DataFrame(
        {
            "date": val_dates.dt.strftime("%Y-%m-%d"),
            "true_value": val_true,
            "pred_value": val_pred,
        }
    ).to_csv(model_dir / "val_predictions.csv", index=False)

    pd.DataFrame(
        {
            "date": test_dates.dt.strftime("%Y-%m-%d"),
            "true_value": test_true,
            "pred_value": test_pred,
        }
    ).to_csv(model_dir / "test_predictions.csv", index=False)

    pd.DataFrame(
        {
            "date": val_dates.dt.strftime("%Y-%m-%d"),
            "true_value": val_true_inverse,
            "pred_value": val_pred_inverse,
        }
    ).to_csv(model_dir / "val_predictions_inverse.csv", index=False)

    pd.DataFrame(
        {
            "date": test_dates.dt.strftime("%Y-%m-%d"),
            "true_value": test_true_inverse,
            "pred_value": test_pred_inverse,
        }
    ).to_csv(model_dir / "test_predictions_inverse.csv", index=False)

    np.save(model_dir / "pred.npy", np.asarray(test_pred, dtype=float).reshape(-1, 1, 1))
    np.save(model_dir / "true.npy", np.asarray(test_true, dtype=float).reshape(-1, 1, 1))
    np.save(
        model_dir / "metrics.npy",
        np.asarray(
            [
                test_metrics["mae"],
                test_metrics["mse"],
                test_metrics["rmse"],
                test_metrics["mape"],
                np.nan,
            ],
            dtype=float,
        ),
    )

    with (model_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_name,
                "validation": val_metrics,
                "test": test_metrics,
                "metadata": {
                    **metadata,
                    "value_space": "standardized",
                    "target_mean": target_mean,
                    "target_scale": target_scale,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=250)
    ax.plot(test_dates, test_true, label="Observed", color=OBS_COLOR, linewidth=1.8)
    ax.plot(test_dates, test_pred, label="Predicted", color=PRED_COLOR, linewidth=1.6)
    ax.set_title(f"{model_name} Test Fit")
    ax.set_xlabel("Date")
    ax.set_ylabel("Standardized Inflow")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.28)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=25)
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, model_dir / "test_fit.png", dpi=300)
    plt.close(fig)

    return test_metrics


def plot_summary_figure(summary_df: pd.DataFrame, output_path: Path, dates: pd.Series) -> None:
    apply_publication_style()
    success_df = summary_df.sort_values(by="nse", ascending=False, na_position="last").reset_index(drop=True)
    n_models = len(success_df)
    ncols = min(3, max(1, n_models))
    nrows = math.ceil(n_models / ncols)

    global_min = float("inf")
    global_max = float("-inf")
    series_cache: list[tuple[np.ndarray, np.ndarray]] = []
    non_gm_min = float("inf")
    non_gm_max = float("-inf")
    for _, row in success_df.iterrows():
        model_dir = Path(str(row["model_dir"]))
        pred = np.load(model_dir / "pred.npy").reshape(-1)
        true = np.load(model_dir / "true.npy").reshape(-1)
        series_cache.append((pred, true))
        global_min = min(global_min, float(np.min(pred)), float(np.min(true)))
        global_max = max(global_max, float(np.max(pred)), float(np.max(true)))
        if str(row["model"]) != "GM11":
            non_gm_min = min(non_gm_min, float(np.min(pred)), float(np.min(true)))
            non_gm_max = max(non_gm_max, float(np.max(pred)), float(np.max(true)))

    if np.isfinite(non_gm_min) and np.isfinite(non_gm_max):
        global_min, global_max = non_gm_min, non_gm_max
    y_pad = max((global_max - global_min) * 0.08, 0.02)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16.0, 4.0 * nrows), sharex=True, sharey=False)
    axes = np.array(axes).reshape(-1)
    for idx, (_, row) in enumerate(success_df.iterrows()):
        ax = axes[idx]
        pred, true = series_cache[idx]
        rank_tag = f"({chr(97 + idx)})"
        gm_suffix = " [ind. y-axis]" if str(row["model"]) == "GM11" else ""
        ax.set_title(
            f"{rank_tag} {row['model']}{gm_suffix}\nNSE={row['nse']:.4f} | RMSE={row['rmse']:.4f}",
            pad=9,
        )
        ax.plot(dates.iloc[: len(true)], true, color=OBS_COLOR, linewidth=1.8)
        ax.plot(dates.iloc[: len(pred)], pred, color=PRED_COLOR, linewidth=1.55)
        if str(row["model"]) == "GM11":
            local_min = min(float(np.min(pred)), float(np.min(true)))
            local_max = max(float(np.max(pred)), float(np.max(true)))
            local_pad = max((local_max - local_min) * 0.08, 0.02)
            ax.set_ylim(local_min - local_pad, local_max + local_pad)
            ax.text(
                0.98,
                0.06,
                "Independent y-axis",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.5,
                color="#666666",
            )
        else:
            ax.set_ylim(global_min - y_pad, global_max + y_pad)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=25, labelbottom=True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx % ncols == 0:
            ax.set_ylabel("Standardized Inflow")
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Date")

    for idx in range(n_models, len(axes)):
        fig.delaxes(axes[idx])

    handles = [
        plt.Line2D([0], [0], color=OBS_COLOR, linewidth=1.8, label="Observed"),
        plt.Line2D([0], [0], color=PRED_COLOR, linewidth=1.55, label="Predicted"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.975))
    fig.suptitle(f"Traditional Method Fit Comparison on Dataset {output_path.parent.name}", y=0.992)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.855, bottom=0.08, hspace=0.5, wspace=0.18)
    save_figure(fig, output_path, dpi=350)
    plt.close(fig)


def append_mrph_result(summary_rows: list[dict], project_root: Path, dataset_label: str) -> None:
    mrph_dir = (
        project_root
        / "results"
        / "dl_result"
        / "test_results"
        / f"water_MRPH_TimesNet_{dataset_label}_MRPH_TimesNet_sl24_pl1_Comparison_Exp_0"
    )
    pred_path = mrph_dir / "pred.npy"
    true_path = mrph_dir / "true.npy"
    if not pred_path.exists() or not true_path.exists():
        return

    pred = np.load(pred_path).reshape(-1)
    true = np.load(true_path).reshape(-1)
    metrics = compute_metrics(pred, true)
    summary_rows.append(
        {
            "dataset": dataset_label,
            "model": "MRPH-TimesNet",
            "nse": metrics["nse"],
            "mse": metrics["mse"],
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "mape": metrics["mape"],
            "model_dir": str(mrph_dir),
        }
    )


def parse_args(default_dataset: str, default_seed: int) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Traditional method comparison for mine inflow forecasting.")
    parser.add_argument("--data-path", type=str, default=default_dataset, help="dataset csv path under ./dataset")
    parser.add_argument("--dataset-label", type=str, default=Path(default_dataset).stem.split("_")[-1], help="dataset label")
    parser.add_argument("--seed", type=int, default=default_seed, help="random seed")
    parser.add_argument("--seq-len", type=int, default=24, help="history window length")
    parser.add_argument(
        "--models",
        type=str,
        default="LinearRegression,ARIMA,SARIMAX,GM11,SVR,XGBoost,RandomForest,UnitHydrograph",
        help="comma-separated model names",
    )
    return parser.parse_args()


def run_traditional_comparison(default_dataset: str, default_seed: int) -> None:
    args = parse_args(default_dataset, default_seed)
    set_random_seed(args.seed)

    project_root = Path(__file__).resolve().parent
    data_path = project_root / "dataset" / args.data_path
    output_root = project_root / "results" / "trandition_results" / args.dataset_label
    output_root.mkdir(parents=True, exist_ok=True)

    bundle = load_dataset_bundle(data_path, args.seq_len)
    model_order = [m.strip() for m in args.models.split(",") if m.strip()]

    runners = {
        "LinearRegression": run_linear_regression,
        "ARIMA": run_arima,
        "SARIMAX": run_sarimax,
        "GM11": run_gm11,
        "SVR": run_svr,
        "XGBoost": run_xgboost,
        "RandomForest": run_random_forest,
        "UnitHydrograph": run_unit_hydrograph,
    }

    summary_rows = []
    for model_name in model_order:
        if model_name not in runners:
            raise ValueError(f"Unsupported model: {model_name}")
        print(f"\n===== Running {model_name} on {args.dataset_label} =====")
        result = runners[model_name](bundle, args.seed)
        model_dir = output_root / model_name
        test_metrics = save_model_outputs(
            model_dir=model_dir,
            model_name=model_name,
            val_dates=bundle.val_dates,
            val_true=bundle.val_y,
            val_pred=result["val_pred"],
            test_dates=bundle.test_dates,
            test_true=bundle.test_y,
            test_pred=result["test_pred"],
            target_mean=bundle.target_mean,
            target_scale=bundle.target_scale,
            metadata=result["metadata"],
        )
        summary_rows.append(
            {
                "dataset": args.dataset_label,
                "model": model_name,
                "nse": test_metrics["nse"],
                "mse": test_metrics["mse"],
                "mae": test_metrics["mae"],
                "rmse": test_metrics["rmse"],
                "mape": test_metrics["mape"],
                "model_dir": str(model_dir),
            }
        )
        print(
            f"{model_name}: NSE={test_metrics['nse']:.4f}, "
            f"RMSE={test_metrics['rmse']:.4f}, MSE={test_metrics['mse']:.4f}"
        )

    append_mrph_result(summary_rows, project_root, args.dataset_label)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_root / f"traditional_comparison_{args.dataset_label}_summary.csv", index=False)
    summary_df.sort_values(by="nse", ascending=False).to_csv(
        output_root / f"traditional_comparison_{args.dataset_label}_ranking_by_nse.csv",
        index=False,
    )
    plot_summary_figure(
        summary_df=summary_df,
        output_path=output_root / f"traditional_comparison_{args.dataset_label}_test_fit_subplots.png",
        dates=bundle.test_dates,
    )
    print(f"\nSaved results to: {output_root}")

