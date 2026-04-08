from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.make_holidays import make_holidays_df
import io
import json
from datetime import date, timedelta, datetime

app = FastAPI(title="Ticket Forecaster API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.kylemair.com",
        "https://kylemair.com",
        "http://localhost",
        "http://127.0.0.1",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class EventItem(BaseModel):
    name: str
    start_date: str          # YYYY-MM-DD
    end_date: str            # YYYY-MM-DD (same as start for single-day)
    spike_pct: float         # e.g. 40 = +40%, -20 = -20%


class ForecastRequest(BaseModel):
    horizon_days: int = 90
    events: Optional[list[EventItem]] = []


class ForecastPoint(BaseModel):
    date: str
    forecast: float
    lower: float
    upper: float
    event: Optional[str] = None


class ForecastResponse(BaseModel):
    forecast: list[ForecastPoint]
    model_info: dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BFCM_DATES = {
    "Thanksgiving":          {"ds": "2025-11-27", "lower_window": 0, "upper_window": 0},
    "Black Friday":          {"ds": "2025-11-28", "lower_window": 0, "upper_window": 0},
    "Small Business Sat":    {"ds": "2025-11-29", "lower_window": 0, "upper_window": 0},
    "Cyber Monday":          {"ds": "2025-12-01", "lower_window": 0, "upper_window": 0},
    # 2026
    "Thanksgiving_26":       {"ds": "2026-11-26", "lower_window": 0, "upper_window": 0},
    "Black Friday_26":       {"ds": "2026-11-27", "lower_window": 0, "upper_window": 0},
    "Small Business Sat_26": {"ds": "2026-11-28", "lower_window": 0, "upper_window": 0},
    "Cyber Monday_26":       {"ds": "2026-11-30", "lower_window": 0, "upper_window": 0},
}


def build_holidays_df() -> pd.DataFrame:
    rows = [
        {"holiday": k, "ds": pd.Timestamp(v["ds"]),
         "lower_window": v["lower_window"], "upper_window": v["upper_window"]}
        for k, v in BFCM_DATES.items()
    ]
    return pd.DataFrame(rows)


def parse_csv(contents: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(contents))
    df.columns = [c.strip().lower() for c in df.columns]

    # Require at minimum date + tickets
    required = {"date", "tickets"}
    if not required.issubset(set(df.columns)):
        raise HTTPException(
            status_code=422,
            detail=f"CSV must contain columns: date, tickets. Got: {list(df.columns)}"
        )

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    for col in ["tickets", "orders", "revenue"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "tickets"])
    return df


def build_future_regressors(
    future: pd.DataFrame,
    history: pd.DataFrame,
    has_orders: bool,
    has_revenue: bool,
    events: list[EventItem],
) -> pd.DataFrame:
    """Fill regressor columns for future dates using trailing medians."""
    future = future.copy()
    future["ds_date"] = future["ds"].dt.date

    if has_orders:
        # Use median orders by day-of-week as a simple forward fill
        history["dow"] = history["ds"].dt.dayofweek
        dow_orders = history.groupby("dow")["orders"].median()
        future["dow"] = future["ds"].dt.dayofweek
        future["orders"] = future["dow"].map(dow_orders)
        future["orders"] = future["orders"].fillna(history["orders"].median())

    if has_revenue:
        history["dow"] = history["ds"].dt.dayofweek
        dow_rev = history.groupby("dow")["revenue"].median()
        future["dow"] = future["ds"].dt.dayofweek
        future["revenue"] = future["dow"].map(dow_rev)
        future["revenue"] = future["revenue"].fillna(history["revenue"].median())

    # Apply event spikes as a regressor-like additive column
    future["event_boost"] = 0.0
    future["event_name"] = ""

    for ev in events:
        start = pd.Timestamp(ev.start_date)
        end = pd.Timestamp(ev.end_date)
        mask = (future["ds"] >= start) & (future["ds"] <= end)
        future.loc[mask, "event_boost"] += ev.spike_pct / 100.0
        future.loc[mask, "event_name"] = ev.name

    return future


# ---------------------------------------------------------------------------
# Forecast endpoint
# ---------------------------------------------------------------------------

@app.post("/forecast", response_model=ForecastResponse)
async def forecast(
    file: UploadFile = File(...),
    params: str = "{}",
):
    # Parse params JSON string (sent as form field alongside the file)
    try:
        params_dict = json.loads(params)
        req = ForecastRequest(**params_dict)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid params: {e}")

    contents = await file.read()
    df = parse_csv(contents)

    has_orders = "orders" in df.columns and df["orders"].notna().sum() > 10
    has_revenue = "revenue" in df.columns and df["revenue"].notna().sum() > 10

    # Build Prophet training dataframe
    train = pd.DataFrame({"ds": df["date"], "y": df["tickets"]})

    if has_orders:
        train["orders"] = df["orders"].values
    if has_revenue:
        train["revenue"] = df["revenue"].values

    # Normalise regressors to avoid scale issues
    orders_mean = train["orders"].mean() if has_orders else 1
    revenue_mean = train["revenue"].mean() if has_revenue else 1

    if has_orders:
        train["orders"] = train["orders"] / orders_mean
    if has_revenue:
        train["revenue"] = train["revenue"] / revenue_mean

    # Fit Prophet
    holidays = build_holidays_df()

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=holidays,
        seasonality_mode="multiplicative",
        interval_width=0.80,
        changepoint_prior_scale=0.05,
    )

    if has_orders:
        m.add_regressor("orders", mode="multiplicative")
    if has_revenue:
        m.add_regressor("revenue", mode="multiplicative")

    m.fit(train)

    # Build future dataframe
    future = m.make_future_dataframe(periods=req.horizon_days)
    future_only = future[future["ds"] > train["ds"].max()].copy()

    # Fill regressors for future dates
    if has_orders:
        dow_orders_norm = train.groupby(train["ds"].dt.dayofweek)["orders"].median()
        future_only["orders"] = future_only["ds"].dt.dayofweek.map(dow_orders_norm)
        future_only["orders"] = future_only["orders"].fillna(train["orders"].median())

    if has_revenue:
        dow_rev_norm = train.groupby(train["ds"].dt.dayofweek)["revenue"].median()
        future_only["revenue"] = future_only["ds"].dt.dayofweek.map(dow_rev_norm)
        future_only["revenue"] = future_only["revenue"].fillna(train["revenue"].median())

    # Merge back
    for col in (["orders"] if has_orders else []) + (["revenue"] if has_revenue else []):
        future[col] = np.nan
        future.loc[future["ds"] <= train["ds"].max(), col] = train[col].values
        future.loc[future["ds"] > train["ds"].max(), col] = future_only[col].values

    forecast_df = m.predict(future)

    # Extract future portion only
    result_df = forecast_df[forecast_df["ds"] > train["ds"].max()].copy()
    result_df = result_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    result_df["yhat"] = result_df["yhat"].clip(lower=0).round(0)
    result_df["yhat_lower"] = result_df["yhat_lower"].clip(lower=0).round(0)
    result_df["yhat_upper"] = result_df["yhat_upper"].clip(lower=0).round(0)

    # Apply custom event boosts on top of Prophet forecast
    event_names = {}
    for ev in req.events:
        start = pd.Timestamp(ev.start_date)
        end = pd.Timestamp(ev.end_date)
        mask = (result_df["ds"] >= start) & (result_df["ds"] <= end)
        boost = 1 + (ev.spike_pct / 100.0)
        result_df.loc[mask, "yhat"] = (result_df.loc[mask, "yhat"] * boost).round(0)
        result_df.loc[mask, "yhat_lower"] = (result_df.loc[mask, "yhat_lower"] * boost).round(0)
        result_df.loc[mask, "yhat_upper"] = (result_df.loc[mask, "yhat_upper"] * boost).round(0)
        for idx in result_df[mask].index:
            event_names[idx] = ev.name

    points = []
    for idx, row in result_df.iterrows():
        points.append(ForecastPoint(
            date=row["ds"].strftime("%Y-%m-%d"),
            forecast=int(row["yhat"]),
            lower=int(row["yhat_lower"]),
            upper=int(row["yhat_upper"]),
            event=event_names.get(idx),
        ))

    model_info = {
        "training_rows": len(train),
        "training_from": train["ds"].min().strftime("%Y-%m-%d"),
        "training_to": train["ds"].max().strftime("%Y-%m-%d"),
        "regressors": (["orders"] if has_orders else []) + (["revenue (AUD)"] if has_revenue else []),
        "horizon_days": req.horizon_days,
    }

    return ForecastResponse(forecast=points, model_info=model_info)


@app.get("/health")
def health():
    return {"status": "ok"}
