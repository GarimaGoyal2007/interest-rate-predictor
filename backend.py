"""Backend for the Stochastic Interest Rate Modelling dashboard.

This module adapts the user's Colab Vasicek workflow to the Streamlit app.
Keep this file in the same folder as app.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TENORS = [
    "3 Mo", "6 Mo", "1 Yr", "2 Yr", "5 Yr",
    "7 Yr", "10 Yr", "20 Yr", "30 Yr",
]
DATE_CANDIDATES = ["Date", "date", "DATE"]
TRADING_DAYS_PER_YEAR = 252
DT = 1.0 / TRADING_DAYS_PER_YEAR


def _prepare_data(dataframe: pd.DataFrame):
    """Validate and clean the uploaded yield-curve CSV."""
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        raise ValueError("The uploaded CSV is empty or is not a valid table.")

    df = dataframe.copy()
    df.columns = df.columns.astype(str).str.strip()

    date_column = next((c for c in DATE_CANDIDATES if c in df.columns), None)
    if date_column is None:
        raise ValueError(
            "Date column not found. The CSV must contain a column named Date."
        )

    missing = [tenor for tenor in TENORS if tenor not in df.columns]
    if missing:
        raise ValueError(
            "The CSV is missing required maturity columns: "
            + ", ".join(missing)
        )

    clean = df[[date_column] + TENORS].copy()
    clean = clean.rename(columns={date_column: "Date"})
    clean["Date"] = pd.to_datetime(clean["Date"], errors="coerce")

    for tenor in TENORS:
        clean[tenor] = pd.to_numeric(clean[tenor], errors="coerce")

    clean = clean.dropna(subset=["Date"])
    clean = clean.sort_values("Date")
    clean = clean.drop_duplicates(subset=["Date"], keep="last")
    clean = clean.reset_index(drop=True)

    clean[TENORS] = clean[TENORS].interpolate(
        method="linear", limit_direction="both"
    )
    clean[TENORS] = clean[TENORS].ffill().bfill()

    if clean[TENORS].isna().any().any():
        raise ValueError(
            "Some maturity columns contain no usable numeric observations."
        )
    if len(clean) < 32:
        raise ValueError(
            "At least 32 dated observations are needed for an 80/20 "
            "train/test split and Vasicek calibration."
        )

    # Match the notebook: identify unusual observations for diagnostics.
    # The notebook reports these values; it does not remove or winsorize them.
    outlier_rows = []
    for tenor in TENORS:
        series = clean[tenor]
        rolling_median = series.rolling(window=30, min_periods=10).median()
        deviation = (series - rolling_median).abs()
        rolling_mad = deviation.rolling(window=30, min_periods=10).median()
        robust_scale = (1.4826 * rolling_mad).replace(0, np.nan)
        robust_score = (series - rolling_median) / robust_scale
        flagged = robust_score.abs() > 4.5

        for idx in clean.index[flagged.fillna(False)]:
            outlier_rows.append({
                "Date": clean.loc[idx, "Date"],
                "Tenor": tenor,
                "Yield": clean.loc[idx, tenor],
                "Robust_Score": robust_score.loc[idx],
            })

    outlier_report = pd.DataFrame(
        outlier_rows,
        columns=["Date", "Tenor", "Yield", "Robust_Score"],
    )
    return clean, outlier_report


def _calibrate_vasicek(rate_values, dt=DT):
    """Estimate Vasicek a, b and sigma by maximum likelihood."""
    rates = np.asarray(rate_values, dtype=float)
    rates = rates[np.isfinite(rates)]

    if len(rates) < 30:
        raise ValueError("At least 30 training observations are required.")

    previous_rates = rates[:-1]
    next_rates = rates[1:]
    sample_std = np.std(rates)

    if sample_std < 1e-10:
        raise ValueError("A maturity series is nearly constant; calibration failed.")

    def negative_log_likelihood(params):
        log_a, b, log_sigma = params
        a = np.exp(log_a)
        sigma = np.exp(log_sigma)
        phi = np.exp(-a * dt)
        expected_rate = b + (previous_rates - b) * phi
        variance = sigma**2 * (1 - phi**2) / (2 * a)
        variance = max(variance, 1e-14)
        errors = next_rates - expected_rate
        return 0.5 * np.sum(
            np.log(2 * np.pi * variance) + errors**2 / variance
        )

    initial_sigma = max(np.std(np.diff(rates)) / np.sqrt(dt), 1e-5)
    initial_params = [np.log(0.5), np.mean(rates), np.log(initial_sigma)]

    result = minimize(
        negative_log_likelihood,
        initial_params,
        method="L-BFGS-B",
        bounds=[
            (np.log(1e-5), np.log(20)),
            (
                np.min(rates) - 10 * sample_std,
                np.max(rates) + 10 * sample_std,
            ),
            (np.log(1e-8), np.log(20)),
        ],
    )
    if not result.success:
        raise ValueError("Vasicek calibration failed: " + str(result.message))

    return float(np.exp(result.x[0])), float(result.x[1]), float(np.exp(result.x[2]))


def _predict_next_rate(current_rate, a, b, dt=DT):
    return b + (current_rate - b) * np.exp(-a * dt)


def _simulate_vasicek(
    initial_rate, a, b, sigma, days, simulations, dt=DT, rng=None
):
    """Generate exact one-step Vasicek transitions for the requested horizon."""
    if rng is None:
        rng = np.random.default_rng(42)

    paths = np.zeros((days + 1, simulations), dtype=float)
    paths[0, :] = initial_rate
    phi = np.exp(-a * dt)
    step_std = sigma * np.sqrt((1 - phi**2) / (2 * a))

    for day in range(1, days + 1):
        random_values = rng.normal(0, 1, simulations)
        paths[day, :] = (
            b + (paths[day - 1, :] - b) * phi + step_std * random_values
        )
    return paths


def run_model(dataframe: pd.DataFrame, horizon: int, n_paths: int) -> dict:
    """Run cleaning, Vasicek calibration, holdout evaluation and simulation.

    Returns the keys expected by the dashboard:
      metrics, predictions, simulation
    It also returns supporting tables for a richer dashboard.
    """
    horizon = int(horizon)
    n_paths = int(n_paths)
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least 1 day.")
    if n_paths < 1:
        raise ValueError("Number of simulation paths must be at least 1.")

    cleaned_data, outlier_report = _prepare_data(dataframe)

    split_index = int(len(cleaned_data) * 0.80)
    train_data = cleaned_data.iloc[:split_index].copy().reset_index(drop=True)
    test_data = cleaned_data.iloc[split_index:].copy().reset_index(drop=True)

    if len(train_data) < 30 or len(test_data) < 2:
        raise ValueError("Not enough training or testing observations.")

    parameters = {}
    prediction_rows = []
    evaluation_rows = []
    rng = np.random.default_rng(42)
    simulation_frames = []

    for tenor in TENORS:
        a, b, sigma = _calibrate_vasicek(train_data[tenor].to_numpy())
        parameters[tenor] = {"a": a, "b": b, "sigma": sigma}

        actual = test_data[tenor].to_numpy(dtype=float)
        previous_rates = np.concatenate(
            ([train_data[tenor].iloc[-1]], actual[:-1])
        )
        predicted = np.array([
            _predict_next_rate(rate, a, b) for rate in previous_rates
        ])

        for date, actual_value, predicted_value in zip(
            test_data["Date"], actual, predicted
        ):
            prediction_rows.append({
                "Date": date,
                "Tenor": tenor,
                "Actual": actual_value,
                "Predicted": predicted_value,
            })

        evaluation_rows.append({
            "Tenor": tenor,
            "MAE": mean_absolute_error(actual, predicted),
            "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
            "R2": r2_score(actual, predicted),
        })

        paths = _simulate_vasicek(
            initial_rate=float(cleaned_data[tenor].iloc[-1]),
            a=a,
            b=b,
            sigma=sigma,
            days=horizon,
            simulations=n_paths,
            rng=rng,
        )

        simulation_frames.append(pd.DataFrame({
            "Future_Trading_Day": np.arange(horizon + 1),
            "Tenor": tenor,
            "Mean_Prediction": np.mean(paths, axis=1),
            "Lower_95_Percentile": np.percentile(paths, 2.5, axis=1),
            "Upper_95_Percentile": np.percentile(paths, 97.5, axis=1),
        }))

    predictions = pd.DataFrame(prediction_rows)
    evaluation_table = pd.DataFrame(evaluation_rows)
    simulation = pd.concat(simulation_frames, ignore_index=True)

    # Overall metrics across every tested date and all nine maturity series.
    metrics = {
        "MAE": float(mean_absolute_error(
            predictions["Actual"], predictions["Predicted"]
        )),
        "RMSE": float(np.sqrt(mean_squared_error(
            predictions["Actual"], predictions["Predicted"]
        ))),
        "R2": float(r2_score(
            predictions["Actual"], predictions["Predicted"]
        )),
    }

    parameter_table = pd.DataFrame.from_dict(parameters, orient="index")
    parameter_table.index.name = "Tenor"

    return {
        "metrics": metrics,
        "predictions": predictions,
        "simulation": simulation,
        "parameters": parameter_table.reset_index(),
        "evaluation": evaluation_table,
        "cleaned_data": cleaned_data,
        "outlier_report": outlier_report,
    }
