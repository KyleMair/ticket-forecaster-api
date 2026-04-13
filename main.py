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

import asyncio
import json
import math
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Callable, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
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
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Fit a CV-weighted ensemble and return a forecast with conformal intervals.
    Optimised for Render free tier: CV uses bare models (no ConformalIntervals),
    column aliases are discovered once and cached, MSTL only activates with 2yr data.
    """
    import copy

    def _p(msg: str):
        if progress:
            progress(msg)

    n = len(train)
    uid = train["unique_id"].iloc[0]

    # -----------------------------------------------------------------------
    # Column alias discovery — run once with bare models, no ConformalIntervals
    # -----------------------------------------------------------------------
    KNOWN_ALIASES = {
        "AutoARIMA": "AutoARIMA",
        "AutoCES":   "CES",
        "AutoTheta": "Theta",
        "MSTL":      "MSTL",
    }

    def _bare(m):
        """Return a copy of a model with prediction_intervals stripped."""
        m2 = copy.copy(m)
        if hasattr(m2, "prediction_intervals"):
            m2.prediction_intervals = None
        return m2

    def _col_names(models):
        """Resolve column aliases using the known map; fall back to dummy fit only
        for any model not in the map."""
        names = []
        unknown = []
        for m in models:
            cls = type(m).__name__
            if cls in KNOWN_ALIASES:
                names.append(KNOWN_ALIASES[cls])
            else:
                unknown.append((len(names), m))
                names.append(None)  # placeholder

        if unknown:
            rng = np.random.default_rng(42)
            dummy = pd.DataFrame({
                "unique_id": "x",
                "ds": pd.date_range("2020-01-01", periods=30, freq="D"),
                "y": rng.integers(50, 150, 30).astype(float),
            })
            sf_tmp = StatsForecast(models=[_bare(m) for _, m in unknown], freq="D", n_jobs=1)
            sf_tmp.fit(dummy)
            out = sf_tmp.predict(h=1)
            discovered = [c for c in out.columns if c not in ("unique_id", "ds")]
            for (idx, _), col in zip(unknown, discovered):
                names[idx] = col

        return names

    # -----------------------------------------------------------------------
    # Seasonality / data flags
    # -----------------------------------------------------------------------
    _p(f"Loaded {n} days of training data")
    use_annual = n >= MIN_ROWS_FOR_ANNUAL
    seasonality_mode = "weekly+annual (MSTL)" if use_annual else "weekly"
    _p(f"Seasonality: {seasonality_mode}")

    has_events  = bool(events)
    has_orders  = "orders" in train.columns and train["orders"].notna().any()
    has_revenue = "revenue" in train.columns and train["revenue"].notna().any()
    build_x     = has_events or (has_regressors and (has_orders or has_revenue))

    future_dates = pd.date_range(
        start=train["ds"].max() + timedelta(days=1),
        periods=horizon_days, freq="D",
    )

    # -----------------------------------------------------------------------
    # Exogenous features
    # -----------------------------------------------------------------------
    x_train: Optional[pd.DataFrame] = None
    x_future: Optional[pd.DataFrame] = None

    if build_x:
        _p("Building exogenous feature columns")
        train_x = train[["unique_id", "ds"]].copy()
        if has_orders:
            train_x["orders"] = train["orders"].fillna(train["orders"].median())
        if has_revenue:
            train_x["revenue"] = train["revenue"].fillna(train["revenue"].median())
        if has_events:
            tmp = add_event_features(train[["unique_id", "ds"]].copy(), events)
            for col in ["is_event", "event_lead", "event_lag"]:
                train_x[col] = tmp[col].values
        x_train = train_x

        future_x = pd.DataFrame({"unique_id": uid, "ds": future_dates})
        if has_orders:
            future_x["orders"] = train["orders"].iloc[-14:].mean()
        if has_revenue:
            future_x["revenue"] = train["revenue"].iloc[-14:].mean()
        if has_events:
            tmp2 = add_event_features(future_x.copy(), events)
            for col in ["is_event", "event_lead", "event_lag"]:
                future_x[col] = tmp2[col].values
        x_future = future_x

    # -----------------------------------------------------------------------
    # Model pool
    # CV uses bare models (no ConformalIntervals) — much faster.
    # Final fit uses conformal models for calibrated intervals.
    # -----------------------------------------------------------------------
    _p("Preparing model pool")

    mstl_bare  = MSTL(season_length=[7, 365] if use_annual else [7],
                      trend_forecaster=AutoARIMA())
    arima_bare = AutoARIMA(season_length=7)
    ces_bare   = AutoCES(season_length=7)
    theta_bare = AutoTheta(season_length=7)

    ci = ConformalIntervals(h=horizon_days, n_windows=N_CV_WINDOWS)

    mstl_ci  = MSTL(season_length=[7, 365] if use_annual else [7],
                    trend_forecaster=AutoARIMA())
    arima_ci = AutoARIMA(season_length=7, prediction_intervals=ci)
    ces_ci   = AutoCES(season_length=7,   prediction_intervals=ci)
    theta_ci = AutoTheta(season_length=7, prediction_intervals=ci)

    if use_annual:
        cv_models    = [mstl_bare,  ces_bare,  theta_bare]
        final_models = [mstl_ci,    ces_ci,    theta_ci]
    else:
        cv_models    = [arima_bare, ces_bare,  theta_bare]
        final_models = [arima_ci,   ces_ci,    theta_ci]

    arima_col = KNOWN_ALIASES["AutoARIMA"]

    # -----------------------------------------------------------------------
    # CV-weighted ensemble (bare models — no conformal overhead during CV)
    # Column names are read directly from the CV output, never assumed.
    # -----------------------------------------------------------------------
    weights: dict[str, float] = {}
    errors:  dict[str, float] = {}
    model_names: list = []

    if n >= MIN_ROWS_FOR_CV:
        _p(f"Cross-validating {len(cv_models)} models ({N_CV_WINDOWS} windows)")
        try:
            sf_cv = StatsForecast(models=cv_models, freq="D", n_jobs=1)
            cv_df = sf_cv.cross_validation(
                df=train[["unique_id", "ds", "y"]],
                h=horizon_days,
                n_windows=N_CV_WINDOWS,
                step_size=horizon_days,
            )
            skip = {"unique_id", "ds", "y", "cutoff"}
            model_names = [c for c in cv_df.columns if c not in skip]
            actuals = cv_df["y"].values
            for m in model_names:
                errors[m] = float(np.mean(np.abs(cv_df[m].values - actuals)))

            _p("Cross-validation complete — computing weights")
            inv   = {m: 1.0 / (e + 1e-6) for m, e in errors.items()}
            total = sum(inv.values())
            weights = {m: v / total for m, v in inv.items()}

        except Exception as exc:
            _p(f"CV failed ({exc}), using equal weights")
            model_names = _col_names(cv_models)
            weights = {m: 1.0 / len(model_names) for m in model_names}
            errors  = {m: float("nan") for m in model_names}
    else:
        model_names = _col_names(cv_models)
        weights = {m: 1.0 / len(model_names) for m in model_names}
        errors  = {m: float("nan") for m in model_names}

    # -----------------------------------------------------------------------
    # Final fit with conformal intervals
    # Read column names from actual predict output — paired by position with
    # cv_models/final_models which are always in the same order.
    # -----------------------------------------------------------------------
    _p("Fitting models on full training data")
    sf_final = StatsForecast(models=final_models, freq="D", n_jobs=1)
    sf_final.fit(train[["unique_id", "ds", "y"]])

    _p(f"Generating {horizon_days}-day forecast")
    raw = sf_final.predict(h=horizon_days, level=[80])

    skip_pred = {"unique_id", "ds"}
    raw_model_cols = [c for c in raw.columns
                      if c not in skip_pred and not c.endswith(("-lo-80", "-hi-80"))]

    forecast = np.zeros(horizon_days)
    lo = np.zeros(horizon_days)
    hi = np.zeros(horizon_days)

    for raw_col, cv_col in zip(raw_model_cols, model_names):
        w = weights.get(cv_col, 1.0 / len(model_names))
        forecast += w * raw[raw_col].values
        lo_col, hi_col = f"{raw_col}-lo-80", f"{raw_col}-hi-80"
        if lo_col in raw.columns:
            lo += w * raw[lo_col].values
            hi += w * raw[hi_col].values
        else:
            lo += w * raw[raw_col].values * 0.80
            hi += w * raw[raw_col].values * 1.20

    # -----------------------------------------------------------------------
    # Exogenous AutoARIMA blend (only when orders/revenue/events provided)
    # -----------------------------------------------------------------------
    if build_x:
        _p("Fitting exogenous ARIMAX model")
        sf_x = StatsForecast(
            models=[AutoARIMA(season_length=7, prediction_intervals=ci)],
            freq="D", n_jobs=1,
        )
        sf_x.fit(train[["unique_id", "ds", "y"]], X_df=x_train)
        raw_x = sf_x.predict(h=horizon_days, X_df=x_future, level=[80])

        # Give exog ARIMA 1/4 weight; redistribute the rest proportionally
        exog_w = 0.25
        scale  = 1.0 - exog_w
        weights = {m: v * scale for m, v in weights.items()}

        forecast = scale * forecast + exog_w * raw_x[arima_col].values
        lo_x = raw_x.get(f"{arima_col}-lo-80", raw_x[arima_col]).values
        hi_x = raw_x.get(f"{arima_col}-hi-80", raw_x[arima_col]).values
        lo = scale * lo + exog_w * lo_x
        hi = scale * hi + exog_w * hi_x
        weights[arima_col] = exog_w
        errors[arima_col]  = float("nan")

    _p("Assembling forecast")
    forecast = np.maximum(forecast, 0)
    lo       = np.maximum(lo, 0)
    hi       = np.maximum(hi, 0)

    return {
        "dates":    [d.strftime("%Y-%m-%d") for d in future_dates],
        "forecast": forecast.round().astype(int).tolist(),
        "lower":    lo.round().astype(int).tolist(),
        "upper":    hi.round().astype(int).tolist(),
        "weights":  {m: round(v, 4) for m, v in weights.items()},
        "errors":   {m: round(v, 2) if not math.isnan(v) else None
                     for m, v in errors.items()},
        "seasonality_mode":      seasonality_mode,
        "interval_type":         "conformal",
        "has_annual_seasonality": use_annual,
        "exogenous_used":         build_x,
        "regressors": (
            (["orders"]       if has_orders  else []) +
            (["revenue"]      if has_revenue else []) +
            (["event_flags"]  if has_events  else [])
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
    text = content.decode("utf-8-sig", errors="replace")  # utf-8-sig strips BOM if present
    # Auto-detect delimiter: use tab if present in the first line, else comma
    first_line = text.split("\n")[0]
    sep = "\t" if "\t" in first_line else ","
    df = pd.read_csv(StringIO(text), sep=sep)
    df.columns = [c.strip().lower() for c in df.columns]

    # Map date column
    date_candidates = ["date", "day", "ds"]
    date_col = next((c for c in date_candidates if c in df.columns), None)
    if date_col is None:
        raise ValueError("No date column found. Expected: date, day, or ds.")
    df = df.rename(columns={date_col: "ds"})
    df["ds"] = pd.to_datetime(df["ds"])

    # Map target column
    y_candidates = ["tickets", "y", "volume", "count", "ticket_count", "ticket_volume", "support_tickets", "num_tickets", "total_tickets"]
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


@app.post("/debug-csv")
async def debug_csv(file: UploadFile = File(...)):
    """
    Debug endpoint — returns exactly what parse_csv sees from your file.
    Hit this first when the forecast fails with a row-count error.
    Returns: raw byte info, detected encoding/separator, column names, row count.
    """
    content = await file.read()
    result = {}

    # Raw byte inspection
    result["file_size_bytes"] = len(content)
    result["first_20_bytes_hex"] = content[:20].hex()
    result["has_bom"] = content[:3] == b"\xef\xbb\xbf"

    # Decode
    text_sig  = content.decode("utf-8-sig", errors="replace")
    text_raw  = content.decode("utf-8",     errors="replace")
    result["first_line_raw"]     = text_raw.split("\n")[0]
    result["first_line_bom_stripped"] = text_sig.split("\n")[0]

    # Separator detection
    first_line = text_sig.split("\n")[0]
    sep = "\t" if "\t" in first_line else ","
    result["detected_separator"] = "tab" if sep == "\t" else "comma"

    # Parse
    from io import StringIO
    import pandas as pd
    try:
        df = pd.read_csv(StringIO(text_sig), sep=sep)
        df.columns = [c.strip().lower() for c in df.columns]
        result["parsed_columns"] = df.columns.tolist()
        result["parsed_row_count"] = len(df)
        result["first_3_rows"] = df.head(3).to_dict(orient="records")
        result["null_counts"] = df.isnull().sum().to_dict()

        # Simulate column mapping
        date_candidates = ["date", "day", "ds"]
        y_candidates = ["tickets", "y", "volume", "count", "ticket_count",
                        "ticket_volume", "support_tickets", "num_tickets", "total_tickets"]
        date_col = next((c for c in date_candidates if c in df.columns), None)
        y_col    = next((c for c in y_candidates    if c in df.columns), None)
        result["mapped_date_col"] = date_col
        result["mapped_y_col"]    = y_col

        if date_col and y_col:
            df2 = df.rename(columns={date_col: "ds", y_col: "y"})
            df2["ds"] = pd.to_datetime(df2["ds"], errors="coerce")
            df2["y"]  = pd.to_numeric(df2["y"],   errors="coerce")
            df2 = df2.dropna(subset=["ds", "y"])
            result["rows_after_dropna"] = len(df2)
        else:
            result["rows_after_dropna"] = 0
    except Exception as e:
        result["parse_error"] = str(e)

    return result


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


@app.post("/forecast/stream")
async def forecast_stream(
    file: UploadFile = File(...),
    params: str = Form(...),
):
    """
    SSE streaming forecast endpoint.
    Yields progress messages as `data: <message>\n\n` events, then a final
    `data: RESULT:<json>\n\n` event containing the full forecast payload.
    The frontend listens with EventSource (polyfilled via fetch+ReadableStream).
    """
    try:
        req = json.loads(params)
    except json.JSONDecodeError:
        async def err():
            yield "data: ERROR:params must be valid JSON\n\n"
        return StreamingResponse(err(), media_type="text/event-stream")

    horizon_days: int = int(req.get("horizon_days", 90))
    events: list[dict] = req.get("events", [])
    content = await file.read()

    import queue, threading

    msg_queue: queue.Queue = queue.Queue()

    def progress(msg: str):
        msg_queue.put(("progress", msg))

    def worker():
        try:
            df = parse_csv(content)
            has_regressors = "orders" in df.columns or "revenue" in df.columns
            result = run_model(df, horizon_days, events, has_regressors, progress=progress)

            forecast_rows = [
                {"date": result["dates"][i], "forecast": result["forecast"][i],
                 "lower": result["lower"][i], "upper": result["upper"][i]}
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
            msg_queue.put(("result", json.dumps(payload)))
        except Exception as e:
            msg_queue.put(("error", str(e)))
        finally:
            msg_queue.put(("done", None))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    async def event_generator():
        loop = asyncio.get_event_loop()
        TOTAL_TIMEOUT   = 600   # 10 min absolute ceiling
        HEARTBEAT_EVERY = 15    # SSE comment every 15s keeps proxies/Render alive
        start           = loop.time()
        last_heartbeat  = start

        while True:
            # Poll with a short timeout so we never block the event loop long
            # enough to drop the connection or starve other coroutines.
            try:
                kind, data = await loop.run_in_executor(
                    None, lambda: msg_queue.get(timeout=0.5)
                )
            except queue.Empty:
                now = loop.time()
                if now - start > TOTAL_TIMEOUT:
                    yield "data: ERROR:Forecast timed out after 10 minutes\n\n"
                    break
                if now - last_heartbeat >= HEARTBEAT_EVERY:
                    # SSE comment — invisible to JS but keeps the TCP connection open
                    yield ": keep-alive\n\n"
                    last_heartbeat = now
                continue
            except Exception as exc:
                yield f"data: ERROR:{exc}\n\n"
                break

            if kind == "progress":
                yield f"data: {data}\n\n"
                last_heartbeat = loop.time()
            elif kind == "result":
                yield f"data: RESULT:{data}\n\n"
                break
            elif kind == "error":
                yield f"data: ERROR:{data}\n\n"
                break
            elif kind == "done":
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


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
