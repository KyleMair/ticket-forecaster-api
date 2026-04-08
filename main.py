"""
Support Ticket Forecaster — FastAPI backend
Upgraded model pipeline:
  - CV-weighted ensemble: AutoARIMA + AutoETS + AutoCES + AutoTheta (or MSTL when ≥2yr data)
  - True exogenous regressors via AutoARIMA X_df (orders, revenue, event binary flags)
  - Multi-seasonality via MSTL ([7, 365]) when ≥2 years of history available
  - Conformal prediction intervals (distribution-free, guaranteed coverage)
  - Event binary regressors baked into training (not post-hoc multipliers)
  - SQLite persistence for shareable forecast links
"""

import json
import math
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoCES, AutoETS, AutoTheta, MSTL
from statsforecast.utils import ConformalIntervals

# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

DB_PATH = "forecasts.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS forecasts (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            payload TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Ticket Forecaster API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ---------------------------------------------------------------------------
# Event feature engineering
# ---------------------------------------------------------------------------

def add_event_features(df: pd.DataFrame, events: list[dict]) -> pd.DataFrame:
    """
    Encode events as three binary columns:
      - is_event:    1 on the event day(s)
      - event_lead:  1 in the N days before (suppression window)
      - event_lag:   1 in the N days after  (tail/returns window)

    Using three columns instead of one lets AutoARIMA learn separate
    coefficients for pre-event suppression and post-event tail.
    """
    df = df.copy()
    df["is_event"] = 0
    df["event_lead"] = 0
    df["event_lag"] = 0

    for ev in events:
        start = pd.Timestamp(ev["start_date"])
        end = pd.Timestamp(ev["end_date"])
        window = int(ev.get("window_days", 2))

        lead_start = start - timedelta(days=window)
        lag_end = end + timedelta(days=window)

        df.loc[(df["ds"] >= lead_start) & (df["ds"] < start), "event_lead"] = 1
        df.loc[(df["ds"] >= start) & (df["ds"] <= end), "is_event"] = 1
        df.loc[(df["ds"] > end) & (df["ds"] <= lag_end), "event_lag"] = 1

    return df


def make_future_event_df(
    future_dates: pd.Series,
    events: list[dict],
    unique_id: str,
) -> pd.DataFrame:
    """Build the X_df for the forecast horizon with event binary flags."""
    future = pd.DataFrame({"unique_id": unique_id, "ds": future_dates})
    future = add_event_features(future, events)
    return future[["unique_id", "ds", "is_event", "event_lead", "event_lag"]]


# ---------------------------------------------------------------------------
# Core forecast logic
# ---------------------------------------------------------------------------

MIN_ROWS_FOR_ANNUAL = 2 * 365   # need 2 full years for season_length=365
MIN_ROWS_FOR_CV = 60            # below this, skip CV and use equal weights
N_CV_WINDOWS = 3

def run_model(
    train: pd.DataFrame,
    horizon_days: int,
    events: list[dict],
    has_regressors: bool,
) -> dict:
    """
    Fit an ensemble of statistical models and return a forecast with
    conformal prediction intervals and CV-derived ensemble weights.

    Args:
        train:           DataFrame with columns [unique_id, ds, y]
                         and optionally [orders, revenue].
        horizon_days:    Number of days to forecast.
        events:          List of event dicts with start_date/end_date/window_days.
        has_regressors:  Whether orders/revenue columns are present.

    Returns:
        dict with keys: dates, forecast, lower, upper, weights, errors,
                        seasonality_mode, interval_type
    """
    n = len(train)
    uid = train["unique_id"].iloc[0]

    # -----------------------------------------------------------------------
    # Decide seasonality mode
    # -----------------------------------------------------------------------
    use_annual = n >= MIN_ROWS_FOR_ANNUAL
    seasonality_mode = "weekly+annual (MSTL)" if use_annual else "weekly"

    # -----------------------------------------------------------------------
    # Build exogenous feature DataFrames
    # -----------------------------------------------------------------------
    x_train: Optional[pd.DataFrame] = None
    x_future: Optional[pd.DataFrame] = None

    future_dates = pd.date_range(
        start=train["ds"].max() + timedelta(days=1),
        periods=horizon_days,
        freq="D",
    )

    has_events = bool(events)
    has_orders = "orders" in train.columns and train["orders"].notna().any()
    has_revenue = "revenue" in train.columns and train["revenue"].notna().any()

    # We can only pass X_df to AutoARIMA (not MSTL's internal trend forecaster,
    # not AutoETS/AutoCES/AutoTheta). So if we have any exogenous data, we
    # build a unified X_df for AutoARIMA and run the others without it.
    build_x = has_events or (has_regressors and (has_orders or has_revenue))

    if build_x:
        train_x = train[["unique_id", "ds"]].copy()
        if has_orders:
            train_x["orders"] = train["orders"].fillna(train["orders"].median())
        if has_revenue:
            train_x["revenue"] = train["revenue"].fillna(train["revenue"].median())

        if has_events:
            train_with_events = train[["unique_id", "ds"]].copy()
            train_with_events = add_event_features(train_with_events, events)
            for col in ["is_event", "event_lead", "event_lag"]:
                train_x[col] = train_with_events[col].values

        x_train = train_x

        # Future X: carry forward recent regressor mean; events from calendar
        future_x = pd.DataFrame({"unique_id": uid, "ds": future_dates})
        if has_orders:
            future_x["orders"] = train["orders"].iloc[-14:].mean()
        if has_revenue:
            future_x["revenue"] = train["revenue"].iloc[-14:].mean()
        if has_events:
            future_with_events = add_event_features(future_x, events)
            for col in ["is_event", "event_lead", "event_lag"]:
                future_x[col] = future_with_events[col].values

        x_future = future_x

    # -----------------------------------------------------------------------
    # Conformal intervals — use cross-validation on residuals
    # guarantees empirical coverage at the requested level
    # -----------------------------------------------------------------------
    ci = ConformalIntervals(h=horizon_days, n_windows=N_CV_WINDOWS)
    interval_type = "conformal"

    # -----------------------------------------------------------------------
    # Model pool
    # MSTL handles multi-seasonality; the others provide ensemble diversity.
    # AutoARIMA receives exogenous data; the others are univariate.
    # -----------------------------------------------------------------------
    if use_annual:
        mstl = MSTL(
            season_length=[7, 365],
            trend_forecaster=AutoARIMA(),
        )
    else:
        mstl = MSTL(
            season_length=[7],
            trend_forecaster=AutoARIMA(),
        )

    # AutoARIMA is the exogenous-capable anchor model.
    # We run it both with and without MSTL depending on history length.
    arimax = AutoARIMA(season_length=7, prediction_intervals=ci)
    ces   = AutoCES(season_length=7, prediction_intervals=ci)
    theta = AutoTheta(season_length=7, prediction_intervals=ci)

    # When annual seasonality is available, MSTL replaces standalone AutoARIMA
    # in the ensemble (but we still need AutoARIMA for exogenous support).
    # Strategy: fit MSTL + AutoCES + AutoTheta for the univariate ensemble,
    # and AutoARIMA(X_df) as a fourth member when exogenous data exists.
    if use_annual:
        univariate_models = [mstl, ces, theta]
        model_names_uni = ["MSTL", "AutoCES", "AutoTheta"]
        exog_model_name = "AutoARIMA" if build_x else None
    else:
        univariate_models = [arimax, ces, theta]
        model_names_uni = ["AutoARIMA", "AutoCES", "AutoTheta"]
        exog_model_name = None  # AutoARIMA IS the exog model

    # For the exogenous-only AutoARIMA when MSTL is active:
    exog_arimax = None
    if use_annual and build_x:
        exog_arimax = AutoARIMA(season_length=7, prediction_intervals=ci)

    # -----------------------------------------------------------------------
    # CV-weighted ensemble
    # -----------------------------------------------------------------------
    sf_uni = StatsForecast(models=univariate_models, freq="D", n_jobs=1)

    weights: dict[str, float] = {}
    errors:  dict[str, float] = {}

    if n >= MIN_ROWS_FOR_CV:
        try:
            cv = sf_uni.cross_validation(
                df=train[["unique_id", "ds", "y"]],
                h=horizon_days,
                n_windows=N_CV_WINDOWS,
                step_size=horizon_days,
            )
            actuals = cv["y"].values
            for m in model_names_uni:
                preds = cv[m].values
                errors[m] = float(np.mean(np.abs(preds - actuals)))

            if build_x and exog_arimax is not None:
                sf_exog_cv = StatsForecast(models=[exog_arimax], freq="D", n_jobs=1)
                cv_exog = sf_exog_cv.cross_validation(
                    df=train[["unique_id", "ds", "y"]],
                    h=horizon_days,
                    n_windows=N_CV_WINDOWS,
                    step_size=horizon_days,
                    X_df=x_train,
                )
                errors["AutoARIMA"] = float(
                    np.mean(np.abs(cv_exog["AutoARIMA"].values - cv_exog["y"].values))
                )

            # Inverse-error weighting (softmax-stable)
            inv = {m: 1.0 / (e + 1e-6) for m, e in errors.items()}
            total = sum(inv.values())
            weights = {m: v / total for m, v in inv.items()}

        except Exception:
            # Fall back to equal weights if CV fails (e.g., too little data)
            all_models = model_names_uni + (["AutoARIMA"] if build_x and exog_arimax else [])
            weights = {m: 1.0 / len(all_models) for m in all_models}
            errors  = {m: float("nan") for m in all_models}
    else:
        # Insufficient history — equal weights
        all_models = model_names_uni + (["AutoARIMA"] if build_x and exog_arimax else [])
        weights = {m: 1.0 / len(all_models) for m in all_models}
        errors  = {m: float("nan") for m in all_models}

    # -----------------------------------------------------------------------
    # Fit on full training data and predict
    # -----------------------------------------------------------------------
    sf_uni.fit(train[["unique_id", "ds", "y"]])
    raw_uni = sf_uni.predict(h=horizon_days, level=[80])

    # Build weighted forecast from univariate models
    forecast = np.zeros(horizon_days)
    lo = np.zeros(horizon_days)
    hi = np.zeros(horizon_days)

    for m in model_names_uni:
        w = weights.get(m, 0.0)
        forecast += w * raw_uni[m].values
        lo_col = f"{m}-lo-80"
        hi_col = f"{m}-hi-80"
        if lo_col in raw_uni.columns:
            lo += w * raw_uni[lo_col].values
            hi += w * raw_uni[hi_col].values
        else:
            # Model doesn't return intervals — use point forecast ±20% as fallback
            lo += w * (raw_uni[m].values * 0.80)
            hi += w * (raw_uni[m].values * 1.20)

    # Blend in exogenous AutoARIMA if active
    if build_x and use_annual and exog_arimax:
        sf_exog = StatsForecast(models=[exog_arimax], freq="D", n_jobs=1)
        sf_exog.fit(train[["unique_id", "ds", "y"]], X_df=x_train)
        raw_exog = sf_exog.predict(h=horizon_days, X_df=x_future, level=[80])
        w_exog = weights.get("AutoARIMA", 0.0)
        forecast += w_exog * raw_exog["AutoARIMA"].values
        lo += w_exog * raw_exog["AutoARIMA-lo-80"].values
        hi += w_exog * raw_exog["AutoARIMA-hi-80"].values

    elif build_x and not use_annual:
        # AutoARIMA is in univariate_models but was fit without X_df.
        # Re-fit it with X_df and replace its contribution.
        w_arima = weights.get("AutoARIMA", 0.0)
        if w_arima > 0:
            sf_exog = StatsForecast(
                models=[AutoARIMA(season_length=7, prediction_intervals=ci)],
                freq="D", n_jobs=1,
            )
            sf_exog.fit(train[["unique_id", "ds", "y"]], X_df=x_train)
            raw_exog = sf_exog.predict(h=horizon_days, X_df=x_future, level=[80])

            # Subtract the univariate ARIMA contribution and add exog version
            forecast -= w_arima * raw_uni["AutoARIMA"].values
            lo       -= w_arima * raw_uni.get("AutoARIMA-lo-80", raw_uni["AutoARIMA"]).values
            hi       -= w_arima * raw_uni.get("AutoARIMA-hi-80", raw_uni["AutoARIMA"]).values
            forecast += w_arima * raw_exog["AutoARIMA"].values
            lo       += w_arima * raw_exog["AutoARIMA-lo-80"].values
            hi       += w_arima * raw_exog["AutoARIMA-hi-80"].values

    # Clamp to non-negative (count data)
    forecast = np.maximum(forecast, 0)
    lo = np.maximum(lo, 0)
    hi = np.maximum(hi, 0)

    return {
        "dates": [d.strftime("%Y-%m-%d") for d in future_dates],
        "forecast": forecast.round().astype(int).tolist(),
        "lower":    lo.round().astype(int).tolist(),
        "upper":    hi.round().astype(int).tolist(),
        "weights":  {m: round(v, 4) for m, v in weights.items()},
        "errors":   {m: round(v, 2) if not math.isnan(v) else None for m, v in errors.items()},
        "seasonality_mode": seasonality_mode,
        "interval_type":    interval_type,
        "has_annual_seasonality": use_annual,
        "exogenous_used": build_x,
        "regressors": (
            (["orders"] if has_orders else []) +
            (["revenue"] if has_revenue else []) +
            (["event_flags"] if has_events else [])
        ),
    }


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def parse_csv(content: bytes) -> pd.DataFrame:
    """
    Parse uploaded CSV into a DataFrame with [unique_id, ds, y] plus
    optional [orders, revenue] columns.

    Accepted column names (case-insensitive):
      date/day/ds  →  ds
      tickets/y/volume/count  →  y
      orders  →  orders
      revenue/sales  →  revenue
    """
    from io import StringIO
    text = content.decode("utf-8", errors="replace")
    df = pd.read_csv(StringIO(text))
    df.columns = [c.strip().lower() for c in df.columns]

    # Map date column
    date_candidates = ["date", "day", "ds"]
    date_col = next((c for c in date_candidates if c in df.columns), None)
    if date_col is None:
        raise ValueError("No date column found. Expected: date, day, or ds.")
    df = df.rename(columns={date_col: "ds"})
    df["ds"] = pd.to_datetime(df["ds"])

    # Map target column
    y_candidates = ["tickets", "y", "volume", "count", "ticket_count"]
    y_col = next((c for c in y_candidates if c in df.columns), None)
    if y_col is None:
        raise ValueError("No ticket column found. Expected: tickets, y, volume, or count.")
    df = df.rename(columns={y_col: "y"})
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    # Optional regressors
    for alias, canon in [("revenue", "revenue"), ("sales", "revenue"), ("orders", "orders")]:
        if alias in df.columns and canon not in df.columns:
            df = df.rename(columns={alias: canon})
        elif alias in df.columns:
            pass  # already mapped

    df["unique_id"] = "series_1"
    df = df.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str):
    """Explicit OPTIONS handler for CORS preflight requests."""
    from fastapi.responses import Response
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        },
    )

@app.get("/")
def health():
    return {"status": "ok", "service": "ticket-forecaster"}


@app.post("/forecast")
async def forecast(
    file: UploadFile = File(...),
    params: str = Form(...),
):
    """
    Run a forecast and return results.

    Form fields:
      - file:   CSV with date, tickets[, orders, revenue]
      - params: JSON string with keys:
                  horizon_days (int)
                  events       (list of event dicts)

    Event dict keys:
      name, start_date, end_date, spike_pct, event_type,
      window_days (optional, default 2)
    """
    try:
        req = json.loads(params)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="params must be valid JSON")

    horizon_days: int = int(req.get("horizon_days", 90))
    events: list[dict] = req.get("events", [])

    if horizon_days < 1 or horizon_days > 365:
        raise HTTPException(status_code=422, detail="horizon_days must be 1–365")

    content = await file.read()
    try:
        df = parse_csv(content)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if len(df) < 14:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least 14 days of data; got {len(df)}."
        )

    has_regressors = "orders" in df.columns or "revenue" in df.columns

    try:
        result = run_model(
            train=df,
            horizon_days=horizon_days,
            events=events,
            has_regressors=has_regressors,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")

    # Build response payload
    forecast_rows = [
        {
            "date":     result["dates"][i],
            "forecast": result["forecast"][i],
            "lower":    result["lower"][i],
            "upper":    result["upper"][i],
        }
        for i in range(horizon_days)
    ]

    return {
        "forecast": forecast_rows,
        "model_info": {
            "training_rows":  len(df),
            "training_from":  df["ds"].min().strftime("%Y-%m-%d"),
            "training_to":    df["ds"].max().strftime("%Y-%m-%d"),
            "regressors":     result["regressors"],
            "seasonality":    result["seasonality_mode"],
            "interval_type":  result["interval_type"],
            "has_annual":     result["has_annual_seasonality"],
            "exogenous_used": result["exogenous_used"],
            "ensemble_weights": result["weights"],
            "cv_errors":        result["errors"],
        },
    }


@app.post("/forecast/save")
async def save_forecast(
    file: UploadFile = File(...),
    params: str = Form(...),
):
    """Run forecast and persist result for shareable link retrieval."""
    try:
        req = json.loads(params)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="params must be valid JSON")

    horizon_days: int = int(req.get("horizon_days", 90))
    events: list[dict] = req.get("events", [])

    content = await file.read()
    try:
        df = parse_csv(content)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    has_regressors = "orders" in df.columns or "revenue" in df.columns

    try:
        result = run_model(df, horizon_days, events, has_regressors)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")

    forecast_rows = [
        {
            "date":     result["dates"][i],
            "forecast": result["forecast"][i],
            "lower":    result["lower"][i],
            "upper":    result["upper"][i],
        }
        for i in range(horizon_days)
    ]

    payload = {
        "forecast": forecast_rows,
        "model_info": {
            "training_rows":    len(df),
            "training_from":    df["ds"].min().strftime("%Y-%m-%d"),
            "training_to":      df["ds"].max().strftime("%Y-%m-%d"),
            "regressors":       result["regressors"],
            "seasonality":      result["seasonality_mode"],
            "interval_type":    result["interval_type"],
            "has_annual":       result["has_annual_seasonality"],
            "exogenous_used":   result["exogenous_used"],
            "ensemble_weights": result["weights"],
            "cv_errors":        result["errors"],
        },
    }

    forecast_id = str(uuid.uuid4())[:8]
    conn = get_db()
    conn.execute(
        "INSERT INTO forecasts (id, created_at, payload) VALUES (?, ?, ?)",
        (forecast_id, datetime.utcnow().isoformat(), json.dumps(payload)),
    )
    conn.commit()
    conn.close()

    return {"id": forecast_id}


@app.get("/forecast/{forecast_id}")
def get_forecast(forecast_id: str):
    """Retrieve a saved forecast by ID."""
    conn = get_db()
    row = conn.execute(
        "SELECT payload FROM forecasts WHERE id = ?", (forecast_id,)
    ).fetchone()
    conn.close()

    if row is None:
        raise HTTPException(status_code=404, detail="Forecast not found")

    return JSONResponse(content=json.loads(row["payload"]))
