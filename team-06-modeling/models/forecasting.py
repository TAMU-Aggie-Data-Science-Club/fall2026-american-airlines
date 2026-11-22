from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX


_XGB_DEFAULTS = {
    "objective": "reg:squarederror",
    "n_estimators": 600,
    "learning_rate": 0.01,
    "max_depth": 10,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "reg_alpha": 0.0,
    "tree_method": "hist",
    "enable_categorical": True,
    "n_jobs": 4,
    "random_state": 42,
}

_PREDICTION_COLUMNS = {
    "XGBoost": "XGB_PRED",
    "SARIMAX": "SARIMAX_PRED",
    "50/50 ensemble": "WEIGHTED_50_50_PRED",
    "Dynamic ensemble": "DYNAMIC_PRED",
}


def _route_frame(history, route_columns, month, month_index, target):
    return history[route_columns + [month, month_index, target]].copy()


def _make_future_features(
    history,
    route_static,
    feature_columns,
    route_columns,
    month,
    month_index,
    target,
    forecast_month,
    minimum_month_index,
    static_columns,
):
    origin, destination = route_columns
    current_index = forecast_month.year * 12 + forecast_month.month - 1
    # Forecast features use observed history available before the requested month.
    historical = history.loc[history[month] < forecast_month].copy()
    future = route_static[route_columns + static_columns].copy()
    future[month] = forecast_month
    future[month_index] = current_index
    future["DEP_MONTH"] = forecast_month.month
    future["DEP_YEAR"] = forecast_month.year
    future["DEP_MONTH_SIN"] = np.sin(2 * np.pi * forecast_month.month / 12)
    future["DEP_MONTH_COS"] = np.cos(2 * np.pi * forecast_month.month / 12)
    future["TIME_INDEX"] = current_index - minimum_month_index

    lookup = historical.set_index(route_columns + [month_index])[target]
    # Exact month keys keep a missing route-month missing instead of treating it as zero.
    for lag in (1, 12, 24):
        keys = pd.MultiIndex.from_frame(
            future[route_columns].assign(**{month_index: current_index - lag})
        )
        future[f"route_pax_lag_{lag}m"] = lookup.reindex(keys).to_numpy()

    for days, suffix in ((92, "3m"), (184, "6m"), (366, "12m")):
        window_start = forecast_month - pd.Timedelta(days=days)
        window_history = historical.loc[historical[month] >= window_start]
        stats = window_history.groupby(route_columns, sort=False)[target].agg(
            ["mean", "count"]
        )
        future = future.merge(
            stats,
            left_on=route_columns,
            right_index=True,
            how="left",
            validate="many_to_one",
        ).rename(
            columns={
                "mean": f"route_pax_mean_{suffix}",
                "count": f"route_observed_months_{suffix}",
            }
        )
    future["route_frequency_12m"] = future["route_observed_months_12m"]

    previous_month = historical.loc[historical[month_index] == current_index - 1]
    recent_history = historical.loc[
        historical[month] >= forecast_month - pd.Timedelta(days=366)
    ]
    for role, airport_column in (
        ("origin", origin),
        ("destination", destination),
    ):
        previous_airport_pax = previous_month.groupby(airport_column, sort=False)[
            target
        ].sum()
        airport_route_frequency = recent_history.groupby(
            airport_column, sort=False
        ).size()
        future[f"{role}_pax_previous_month"] = future[airport_column].map(
            previous_airport_pax
        )
        future[f"{role}_route_frequency_12m"] = future[airport_column].map(
            airport_route_frequency
        )

    return future[feature_columns]


def _fit_and_forecast_xgb(
    features,
    passengers,
    feature_sets,
    feature_set_name,
    route_columns,
    month,
    month_index,
    target,
    train_end,
    horizon,
    xgb_params=None,
):
    feature_columns = feature_sets[feature_set_name]
    training = features.loc[features[month] <= train_end]
    x_train = training[feature_columns].copy()
    y_train = training[target]
    categorical_columns = [
        column
        for column in x_train.columns
        if not pd.api.types.is_numeric_dtype(x_train[column])
    ]
    for column in categorical_columns:
        # Airport and route labels are categories, not numeric magnitudes.
        x_train[column] = x_train[column].astype("category")

    model_parameters = _XGB_DEFAULTS.copy()
    if xgb_params is not None:
        model_parameters.update(xgb_params)
    model = xgb.XGBRegressor(**model_parameters)
    model.fit(x_train, y_train, verbose=False)

    static_columns = [
        column
        for column in feature_columns
        if column not in route_columns
        if column
        not in {
            "DEP_MONTH",
            "DEP_YEAR",
            "DEP_MONTH_SIN",
            "DEP_MONTH_COS",
            "TIME_INDEX",
            "route_pax_lag_1m",
            "route_pax_lag_12m",
            "route_pax_lag_24m",
            "route_pax_mean_3m",
            "route_pax_mean_6m",
            "route_pax_mean_12m",
            "route_observed_months_3m",
            "route_observed_months_6m",
            "route_observed_months_12m",
            "route_frequency_12m",
            "origin_pax_previous_month",
            "destination_pax_previous_month",
            "origin_route_frequency_12m",
            "destination_route_frequency_12m",
        }
    ]
    # Rebuild time-varying demand features each step; only route and geography fields persist.
    route_static = (
        features.loc[features[month] <= train_end]
        .sort_values(month)
        .drop_duplicates(route_columns, keep="last")
    )
    observed_history = _route_frame(
        passengers.loc[passengers[month] <= train_end],
        route_columns,
        month,
        month_index,
        target,
    )
    minimum_month_index = int(features[month_index].min())
    output = []

    # Use the same observed history for every lead from this training cutoff.
    for step in range(1, horizon + 1):
        forecast_month = train_end + pd.DateOffset(months=step)
        future = _make_future_features(
            observed_history,
            route_static,
            feature_columns,
            route_columns,
            month,
            month_index,
            target,
            forecast_month,
            minimum_month_index,
            static_columns,
        )
        for column in categorical_columns:
            future[column] = pd.Categorical(
                future[column], categories=x_train[column].cat.categories
            )
        predictions = np.maximum(0, model.predict(future))
        forecast_rows = route_static[route_columns].copy()
        forecast_rows[month] = forecast_month
        forecast_rows["HORIZON"] = step
        forecast_rows["XGB_PRED"] = predictions
        output.append(forecast_rows)

    return pd.concat(output, ignore_index=True)


def tune_xgboost_parameters(
    features,
    passengers,
    feature_sets,
    feature_set_name="3_combined",
    n_trials=20,
    validation_horizon=24,
    target="OD_PAX",
    origin="ORG_AIRPRT",
    destination="DES_AIRPRT",
    month="MONTH_START",
    month_index="MONTH_INDEX",
    *,
    score_from_horizon=1,
    xgb_params=None,
):
    """Tune XGBoost with a fixed number of trees before the final holdout.

    Generate the full fixed-origin forecast and score the requested range of leads.
    """
    import optuna

    route_columns = [origin, destination]
    if not feature_sets or feature_set_name not in feature_sets:
        raise ValueError(f"Unknown or missing feature set {feature_set_name!r}.")
    if passengers.empty or features.empty:
        raise ValueError("Passenger records and engineered features must not be empty.")
    if n_trials < 1:
        raise ValueError("n_trials must be at least one.")
    if validation_horizon < 1:
        raise ValueError("validation_horizon must be at least one month.")
    if not 1 <= score_from_horizon <= validation_horizon:
        raise ValueError("score_from_horizon must be within the validation horizon.")

    latest_month = passengers[month].max()
    test_start = latest_month - pd.DateOffset(months=23)
    validation_end = test_start - pd.DateOffset(months=1)
    validation_start = validation_end - pd.DateOffset(months=validation_horizon - 1)
    train_end = validation_start - pd.DateOffset(months=1)
    if train_end <= passengers[month].min():
        raise ValueError(
            "Not enough monthly history for the requested validation horizon."
        )

    validation_months = pd.date_range(
        validation_start, periods=validation_horizon, freq="MS"
    )
    observed_months = pd.DatetimeIndex(passengers[month].drop_duplicates())
    if not validation_months.isin(observed_months).all():
        raise ValueError("The validation window must contain every calendar month.")

    validation_actual = passengers.loc[
        passengers[month].between(validation_start, validation_end),
        route_columns + [month, target],
    ].copy()
    validation_actual["HORIZON"] = (
        validation_actual[month].dt.year * 12
        + validation_actual[month].dt.month
        - (validation_start.year * 12 + validation_start.month)
        + 1
    )
    validation_actual = validation_actual.loc[
        validation_actual["HORIZON"] >= score_from_horizon
    ]
    validation_volume = validation_actual[target].sum()
    if validation_volume <= 0:
        raise ValueError("Validation passenger volume must be positive.")

    fixed_params = {**(xgb_params or {}), "n_estimators": _XGB_DEFAULTS["n_estimators"]}

    def objective(trial):
        parameters = {
            **fixed_params,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1.0, 20.0, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 100.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        }
        predictions = _fit_and_forecast_xgb(
            features,
            passengers,
            feature_sets,
            feature_set_name,
            route_columns,
            month,
            month_index,
            target,
            train_end,
            validation_horizon,
            parameters,
        )
        scored = validation_actual.merge(
            predictions,
            on=route_columns + [month, "HORIZON"],
            how="inner",
            validate="one_to_one",
        )
        matched_volume = scored[target].sum()
        if matched_volume <= 0:
            raise ValueError("No common passenger volume to score in Optuna validation.")
        trial.set_user_attr(
            "passenger_volume_coverage",
            float(matched_volume / validation_volume),
        )
        trial.set_user_attr("scored_route_months", int(len(scored)))
        return _score(scored, "XGB_PRED", target)["WAPE"]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=42, n_startup_trials=min(10, max(1, n_trials // 2))
        ),
    )
    study.enqueue_trial({
        key: _XGB_DEFAULTS[key]
        for key in (
            "learning_rate", "max_depth", "min_child_weight", "subsample",
            "colsample_bytree", "reg_lambda", "reg_alpha",
        )
    })
    study.optimize(objective, n_trials=n_trials, n_jobs=1, gc_after_trial=True)
    best_trial = study.best_trial
    best_params = {**fixed_params, **best_trial.params}
    print(
        f"Optuna validation: {validation_start:%Y-%m} to {validation_end:%Y-%m}; "
        f"scored leads {score_from_horizon}–{validation_horizon}"
    )
    print(f"Best validation WAPE: {study.best_value:.4f}")
    print(
        "Best-trial passenger-volume coverage: "
        f"{best_trial.user_attrs['passenger_volume_coverage']:.1%}"
    )
    print(f"Best XGBoost parameters: {best_params}")
    return {
        "study": study,
        "best_params": best_params,
        "best_wape": study.best_value,
        "validation_passenger_volume_coverage": best_trial.user_attrs[
            "passenger_volume_coverage"
        ],
        "validation_route_months": best_trial.user_attrs["scored_route_months"],
        "validation_start": validation_start,
        "validation_end": validation_end,
        "validation_horizon": validation_horizon,
        "score_from_horizon": score_from_horizon,
    }


def _fit_and_forecast_sarimax(
    passengers,
    route_columns,
    month,
    target,
    train_end,
    horizon,
):
    training = passengers.loc[passengers[month] <= train_end]
    route_groups = training.groupby(route_columns, sort=False)
    future_months = pd.date_range(
        train_end + pd.DateOffset(months=1), periods=horizon, freq="MS"
    )
    routes = []
    prediction_blocks = []

    for route, group in route_groups:
        observed = group.set_index(month)[target].sort_index()
        monthly_index = pd.date_range(observed.index.min(), train_end, freq="MS")
        series = observed.reindex(monthly_index)
        # Seasonal differencing models annual repetition; the linear trend captures drift.
        model = SARIMAX(
            series,
            order=(0, 0, 0),
            seasonal_order=(0, 1, 0, 12),
            trend="t",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False, maxiter=20)
        predictions = np.maximum(
            0, np.asarray(fitted.get_forecast(steps=horizon).predicted_mean)
        )
        if not np.isfinite(predictions).all():
            raise ValueError(
                f"SARIMAX returned non-finite forecasts for route {route}."
            )
        routes.append(route)
        prediction_blocks.append(predictions)

    route_keys = np.asarray(routes, dtype=object)
    return pd.DataFrame(
        {
            route_columns[0]: np.repeat(route_keys[:, 0], horizon),
            route_columns[1]: np.repeat(route_keys[:, 1], horizon),
            month: np.tile(future_months, len(routes)),
            "HORIZON": np.tile(np.arange(1, horizon + 1), len(routes)),
            "SARIMAX_PRED": np.concatenate(prediction_blocks),
        }
    )


def _score(actual, prediction_column, target):
    error = actual[prediction_column] - actual[target]
    total_actual = actual[target].sum()
    return {
        # WAPE weights absolute error by passenger volume; bias retains its direction.
        "WAPE": np.abs(error).sum() / total_actual,
        "MAE": np.abs(error).mean(),
        "Bias": error.sum() / total_actual,
    }


def _wape_for_forecast_windows(actual, prediction_column, target):
    """Score complete 12- and 24-month windows using matched forecast rows."""
    scores = {}
    for end in (12, 24):
        window = actual.loc[actual["HORIZON"].between(1, end)]
        wape = np.nan
        if window["HORIZON"].nunique() == end and window[target].sum() > 0:
            wape = _score(window, prediction_column, target)["WAPE"]
        scores[f"WAPE_1_{end}"] = wape
    return scores


def _add_wape_windows(metrics, actual, target):
    """Update the report from already matched predictions and actual demand."""
    windows = pd.DataFrame.from_dict(
        {
            name: _wape_for_forecast_windows(actual, column, target)
            for name, column in _PREDICTION_COLUMNS.items()
        },
        orient="index",
    )
    updated = metrics.drop(columns=windows.columns, errors="ignore")
    for position, column in enumerate(windows, start=2):
        updated.insert(position, column, updated["Model"].map(windows[column]))
    return updated


def _wape_window_label(scores):
    """Summarize the two WAPE windows for a chart label."""
    short, full = (
        f"{value:.1%}" if pd.notna(value) else "n/a"
        for value in (scores["WAPE_1_12"], scores["WAPE_1_24"])
    )
    return f"WAPE 1–12 {short} | 1–24 {full}"


def _rmsse(
    actual,
    prediction_column,
    training,
    route_columns,
    month_index,
    target,
):
    """Return macro route RMSSE, scaled by exact one-month training differences."""
    history = training[route_columns + [month_index, target]].sort_values(
        route_columns + [month_index]
    )
    grouped = history.groupby(route_columns, sort=False)
    previous_target = grouped[target].shift()
    consecutive_month = grouped[month_index].diff().eq(1)
    history["NAIVE_SQUARED_ERROR"] = (
        (history[target] - previous_target).pow(2).where(consecutive_month)
    )
    scale = history.groupby(route_columns, sort=False)["NAIVE_SQUARED_ERROR"].mean()
    scale = scale.loc[np.isfinite(scale) & (scale > 0)]

    scored = actual[route_columns + [target, prediction_column]].copy()
    scored["SQUARED_ERROR"] = (scored[prediction_column] - scored[target]).pow(2)
    route_mse = scored.groupby(route_columns, sort=False)["SQUARED_ERROR"].mean()
    route_scale = scale.reindex(route_mse.index)
    valid_scale = route_scale.notna() & np.isfinite(route_scale) & (route_scale > 0)
    if not valid_scale.any():
        raise ValueError("RMSSE cannot be computed without positive training scales.")

    route_rmsse = np.sqrt(route_mse.loc[valid_scale] / route_scale.loc[valid_scale])
    return {
        "RMSSE": route_rmsse.mean(),
        "RMSSE_Routes": int(valid_scale.sum()),
        "RMSSE_Route_Coverage": valid_scale.mean(),
    }


def _sarimax_weight_for_horizon(horizon):
    """Return SARIMAX's scheduled share for each positive forecast horizon."""
    horizons = np.asarray(horizon, dtype=float)
    if not np.isfinite(horizons).all() or np.any(horizons < 1):
        raise ValueError("Forecast horizons must be finite and at least one month.")

    horizons = np.minimum(horizons, 24)
    weights = np.full(horizons.shape, 0.10, dtype=float)
    weights = np.where(horizons <= 3, 0.15, weights)
    weights = np.where(
        (horizons > 3) & (horizons <= 6),
        0.15 - (horizons - 3) * (0.05 / 3),
        weights,
    )
    weights = np.where(
        horizons > 12,
        0.10 - (horizons - 12) * (0.05 / 12),
        weights,
    )
    return weights


def _midpoint_calibration_factor(actual, predicted):
    """Move the predicted aggregate halfway toward the actual aggregate."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Calibration actuals and predictions must have matching values.")
    if (
        not np.isfinite(actual).all()
        or not np.isfinite(predicted).all()
        or np.any(actual < 0)
        or np.any(predicted < 0)
    ):
        raise ValueError("Calibration values must be finite and non-negative.")

    actual_total = actual.sum()
    predicted_total = predicted.sum()
    if predicted_total == 0:
        if actual_total == 0:
            return 1.0
        raise ValueError("Cannot calibrate positive actual volume from zero predictions.")
    return float((actual_total + predicted_total) / (2 * predicted_total))


def _align_predictions(
    actual, xgb_predictions, sarimax_predictions, route_columns, month
):
    keys = route_columns + [month, "HORIZON"]
    return actual.merge(
        xgb_predictions, on=keys, how="inner", validate="one_to_one"
    ).merge(
        sarimax_predictions,
        on=keys,
        how="inner",
        validate="one_to_one",
    )


def run_forecasting_experiment(
    features,
    passengers,
    feature_sets,
    data_dir,
    target="OD_PAX",
    origin="ORG_AIRPRT",
    destination="DES_AIRPRT",
    month="MONTH_START",
    month_index="MONTH_INDEX",
    xgb_params=None,
    *,
    tuning_trials=0,
):
    """Run the benchmark with optional XGBoost tuning before final evaluation."""
    route_columns = [origin, destination]
    if not feature_sets:
        raise ValueError("At least one feature set is required.")
    if passengers.empty or features.empty:
        raise ValueError("Passenger records and engineered features must not be empty.")
    if tuning_trials < 0:
        raise ValueError("tuning_trials must be non-negative.")
    if tuning_trials:
        xgb_params = {**(xgb_params or {}), "n_estimators": _XGB_DEFAULTS["n_estimators"]}

    latest_month = passengers[month].max()
    # Keep feature selection separate from the final 24-month holdout.
    test_start = latest_month - pd.DateOffset(months=23)
    test_train_end = test_start - pd.DateOffset(months=1)
    validation_start = test_train_end - pd.DateOffset(months=11)
    validation_train_end = validation_start - pd.DateOffset(months=1)
    if validation_train_end <= passengers[month].min():
        raise ValueError(
            "Not enough monthly history for validation and a 24-month backtest."
        )

    validation_actual = passengers.loc[
        (passengers[month] >= validation_start) & (passengers[month] <= test_train_end),
        route_columns + [month, target],
    ].copy()
    validation_actual["HORIZON"] = (
        validation_actual[month].dt.year * 12
        + validation_actual[month].dt.month
        - (validation_start.year * 12 + validation_start.month)
        + 1
    )
    feature_metrics = []
    validation_predictions = {}

    for name in feature_sets:
        predictions = _fit_and_forecast_xgb(
            features,
            passengers,
            feature_sets,
            name,
            route_columns,
            month,
            month_index,
            target,
            validation_train_end,
            12,
            xgb_params,
        )
        aligned = validation_actual.merge(
            predictions,
            on=route_columns + [month, "HORIZON"],
            how="inner",
            validate="one_to_one",
        )
        feature_metrics.append(
            {
                "Feature_Set": name,
                **_score(aligned, "XGB_PRED", target),
                **_wape_for_forecast_windows(aligned, "XGB_PRED", target),
            }
        )
        validation_predictions[name] = predictions
        print(f"Validated XGBoost feature set: {name}")

    feature_metrics = pd.DataFrame(feature_metrics).sort_values(
        "WAPE", ignore_index=True
    )
    best_feature_set = feature_metrics.loc[0, "Feature_Set"]
    tuning_result = None
    if tuning_trials:
        print(f"Tuning XGBoost for {best_feature_set} with {tuning_trials} Optuna trials.")
        tuning_result = tune_xgboost_parameters(
            features,
            passengers,
            feature_sets,
            feature_set_name=best_feature_set,
            n_trials=tuning_trials,
            target=target,
            origin=origin,
            destination=destination,
            month=month,
            month_index=month_index,
            score_from_horizon=13,
            xgb_params=xgb_params,
        )
        xgb_params = tuning_result["best_params"]
        validation_predictions[best_feature_set] = _fit_and_forecast_xgb(
            features,
            passengers,
            feature_sets,
            best_feature_set,
            route_columns,
            month,
            month_index,
            target,
            validation_train_end,
            12,
            xgb_params,
        )

    validation_sarimax = _fit_and_forecast_sarimax(
        passengers,
        route_columns,
        month,
        target,
        validation_train_end,
        12,
    )
    validation = _align_predictions(
        validation_actual,
        validation_predictions[best_feature_set],
        validation_sarimax,
        route_columns,
        month,
    )
    if validation.empty:
        raise ValueError("No common validation route-months for model calibration.")
    validation["WEIGHTED_50_50_PRED"] = 0.5 * (
        validation["XGB_PRED"] + validation["SARIMAX_PRED"]
    )
    validation["DYNAMIC_XGB_WEIGHT"] = 1 - _sarimax_weight_for_horizon(
        validation["HORIZON"]
    )
    validation["DYNAMIC_PRED"] = (
        validation["DYNAMIC_XGB_WEIGHT"] * validation["XGB_PRED"]
        + (1 - validation["DYNAMIC_XGB_WEIGHT"]) * validation["SARIMAX_PRED"]
    )
    calibration_columns = [
        "XGB_PRED",
        "SARIMAX_PRED",
        "WEIGHTED_50_50_PRED",
        "DYNAMIC_PRED",
    ]
    calibration_factors = {
        column: _midpoint_calibration_factor(validation[target], validation[column])
        for column in calibration_columns
    }

    test_actual = passengers.loc[
        (passengers[month] >= test_start) & (passengers[month] <= latest_month),
        route_columns + [month, target],
    ].copy()
    test_actual["HORIZON"] = (
        test_actual[month].dt.year * 12
        + test_actual[month].dt.month
        - (test_start.year * 12 + test_start.month)
        + 1
    )
    test_xgb = _fit_and_forecast_xgb(
        features,
        passengers,
        feature_sets,
        best_feature_set,
        route_columns,
        month,
        month_index,
        target,
        test_train_end,
        24,
        xgb_params,
    )
    test_sarimax = _fit_and_forecast_sarimax(
        passengers,
        route_columns,
        month,
        target,
        test_train_end,
        24,
    )
    test = _align_predictions(test_actual, test_xgb, test_sarimax, route_columns, month)
    test["WEIGHTED_50_50_PRED"] = 0.5 * (test["XGB_PRED"] + test["SARIMAX_PRED"])
    test["DYNAMIC_XGB_WEIGHT"] = 1 - _sarimax_weight_for_horizon(test["HORIZON"])
    test["DYNAMIC_PRED"] = (
        test["DYNAMIC_XGB_WEIGHT"] * test["XGB_PRED"]
        + (1 - test["DYNAMIC_XGB_WEIGHT"]) * test["SARIMAX_PRED"]
    )
    for column, factor in calibration_factors.items():
        test[column] *= factor

    rmsse_training = passengers.loc[passengers[month] <= test_train_end]
    test_metrics = pd.DataFrame(
        [
            {
                "Model": "XGBoost",
                **_score(test, "XGB_PRED", target),
                **_rmsse(
                    test, "XGB_PRED", rmsse_training, route_columns, month_index, target
                ),
            },
            {
                "Model": "SARIMAX",
                **_score(test, "SARIMAX_PRED", target),
                **_rmsse(
                    test,
                    "SARIMAX_PRED",
                    rmsse_training,
                    route_columns,
                    month_index,
                    target,
                ),
            },
            {
                "Model": "50/50 ensemble",
                **_score(test, "WEIGHTED_50_50_PRED", target),
                **_rmsse(
                    test,
                    "WEIGHTED_50_50_PRED",
                    rmsse_training,
                    route_columns,
                    month_index,
                    target,
                ),
            },
            {
                "Model": "Dynamic ensemble",
                **_score(test, "DYNAMIC_PRED", target),
                **_rmsse(
                    test,
                    "DYNAMIC_PRED",
                    rmsse_training,
                    route_columns,
                    month_index,
                    target,
                ),
            },
        ]
    )
    prediction_columns = _PREDICTION_COLUMNS
    test_metrics = _add_wape_windows(test_metrics, test, target)
    late_holdout = test.loc[test["HORIZON"].between(13, 24)]
    if late_holdout[target].sum() > 0:
        late_scores = {
            name: _score(late_holdout, column, target)
            for name, column in prediction_columns.items()
        }
        for metric in ("WAPE", "Bias"):
            test_metrics[f"{metric}_13_24"] = test_metrics["Model"].map(
                {name: scores[metric] for name, scores in late_scores.items()}
            )
    selected_model = test_metrics.loc[test_metrics["WAPE"].idxmin(), "Model"]
    actual_volume_coverage = test[target].sum() / test_actual[target].sum()
    print(
        f"Validation: {validation_start:%Y-%m} to {test_train_end:%Y-%m}; "
        f"backtest: {test_start:%Y-%m} to {latest_month:%Y-%m}"
    )
    print(f"Selected XGBoost feature set: {best_feature_set}")
    print(
        "Validation midpoint calibration factors:",
        {column: round(factor, 4) for column, factor in calibration_factors.items()},
    )
    print(
        f"Common route-month backtest coverage: {actual_volume_coverage:.1%} "
        f"of passenger volume across {test[route_columns].drop_duplicates().shape[0]:,} routes"
    )
    feature_validation_label = " before tuning" if tuning_result is not None else ""
    print(f"\nXGBoost feature-set validation{feature_validation_label} (lower WAPE is better):")
    print(
        feature_metrics.to_string(
            index=False,
            formatters={
                column: "{:.3f}".format for column in ("WAPE", "WAPE_1_12", "WAPE_1_24")
            },
            na_rep="n/a",
        )
    )
    print("\n24-month backtest (common route-months; lower WAPE/MAE is better):")
    print(
        test_metrics.to_string(
            index=False,
            formatters={
                "WAPE": "{:.3f}".format,
                "WAPE_1_12": "{:.3f}".format,
                "WAPE_1_24": "{:.3f}".format,
                "MAE": "{:.1f}".format,
                "Bias": "{:.3f}".format,
                "WAPE_13_24": "{:.3f}".format,
                "Bias_13_24": "{:.3f}".format,
                "RMSSE": "{:.3f}".format,
                "RMSSE_Route_Coverage": "{:.1%}".format,
            },
        )
    )
    print("Lowest backtest WAPE:", selected_model)

    latest_xgb = _fit_and_forecast_xgb(
        features,
        passengers,
        feature_sets,
        best_feature_set,
        route_columns,
        month,
        month_index,
        target,
        latest_month,
        24,
        xgb_params,
    )
    latest_sarimax = _fit_and_forecast_sarimax(
        passengers, route_columns, month, target, latest_month, 24
    )
    forecast = latest_xgb.merge(
        latest_sarimax,
        on=route_columns + [month, "HORIZON"],
        how="inner",
        validate="one_to_one",
    )
    forecast["ROUTE_ID"] = (
        forecast[origin].astype("string")
        + "->"
        + forecast[destination].astype("string")
    )
    forecast["WEIGHTED_50_50_PRED"] = 0.5 * (
        forecast["XGB_PRED"] + forecast["SARIMAX_PRED"]
    )
    forecast["DYNAMIC_XGB_WEIGHT"] = 1 - _sarimax_weight_for_horizon(
        forecast["HORIZON"]
    )
    forecast["DYNAMIC_PRED"] = (
        forecast["DYNAMIC_XGB_WEIGHT"] * forecast["XGB_PRED"]
        + (1 - forecast["DYNAMIC_XGB_WEIGHT"]) * forecast["SARIMAX_PRED"]
    )
    for column, factor in calibration_factors.items():
        forecast[column] *= factor
    forecast["SELECTED_MODEL"] = selected_model
    forecast["SELECTED_PRED"] = forecast[prediction_columns[selected_model]]
    forecast = forecast[
        route_columns
        + [
            "ROUTE_ID",
            month,
            "HORIZON",
            "SELECTED_MODEL",
            "SELECTED_PRED",
            "XGB_PRED",
            "SARIMAX_PRED",
            "WEIGHTED_50_50_PRED",
            "DYNAMIC_PRED",
        ]
    ].sort_values(route_columns + [month], ignore_index=True)
    # The forecast contract is capped at two years beyond the latest observed month.
    if forecast["HORIZON"].max() > 24:
        raise ValueError("Forecast horizon exceeded the 24-month limit.")

    output_path = (
        Path(data_dir).parent
        / "data"
        / "processed"
        / "passenger_demand_forecast_24m.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(output_path, index=False)
    print(
        f"\nSaved {len(forecast):,} route-month forecasts through "
        f"{forecast[month].max():%Y-%m} to {output_path}"
    )
    print("\nTop forecast routes by XGBoost passenger volume:")
    print(
        forecast.groupby(route_columns, as_index=False)["XGB_PRED"]
        .sum()
        .nlargest(10, "XGB_PRED")
        .to_string(index=False)
    )
    return {
        "best_feature_set": best_feature_set,
        "best_model": selected_model,
        "feature_metrics": feature_metrics,
        "test_metrics": test_metrics,
        "backtest_predictions": test,
        "forecast": forecast,
        "calibration_factors": calibration_factors,
        "forecast_path": output_path,
        "tuning_result": tuning_result,
    }
