from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS
import io, json, sqlite3, uuid, warnings
from datetime import datetime

warnings.filterwarnings('ignore')

app = FastAPI(title="Ticket Forecaster API")

DB_PATH = '/tmp/forecasts.db'


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS forecasts (
        id TEXT PRIMARY KEY,
        created_at TEXT,
        model_info TEXT,
        forecast TEXT
    )''')
    conn.commit()
    return conn


@app.middleware("http")
async def add_cors_headers(request, call_next):
    if request.method == "OPTIONS":
        from starlette.responses import Response
        response = Response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class EventItem(BaseModel):
    name: str
    start_date: str
    end_date: str
    spike_pct: float


class ForecastRequest(BaseModel):
    horizon_days: int = 90
    events: Optional[list[EventItem]] = []


BFCM_DATES = {
    '11-27': ('Thanksgiving', 1.40),
    '11-28': ('Black Friday', 1.19),
    '11-29': ('Small Business Saturday', 1.08),
    '12-01': ('Cyber Monday', 1.15),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_csv(contents: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(contents))
    df.columns = [c.strip().lower() for c in df.columns]
    if 'date' not in df.columns or 'tickets' not in df.columns:
        raise HTTPException(status_code=422,
            detail=f"CSV must have 'date' and 'tickets' columns. Got: {list(df.columns)}")
    df['date'] = pd.to_datetime(df['date'], dayfirst=False)
    df = df.sort_values('date').reset_index(drop=True)
    for col in ['tickets', 'orders', 'revenue']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(subset=['date', 'tickets'])


def run_model(df: pd.DataFrame, req: ForecastRequest):
    has_orders = 'orders' in df.columns and df['orders'].notna().sum() > 10
    has_revenue = 'revenue' in df.columns and df['revenue'].notna().sum() > 10

    if has_orders and has_revenue:
        y = (df['tickets']/df['tickets'].mean()*0.60
           + df['orders']/df['orders'].mean()*0.25
           + df['revenue']/df['revenue'].mean()*0.15) * df['tickets'].mean()
    elif has_orders:
        y = (df['tickets']/df['tickets'].mean()*0.70
           + df['orders']/df['orders'].mean()*0.30) * df['tickets'].mean()
    else:
        y = df['tickets'].astype(float)

    train = pd.DataFrame({'unique_id': 'tickets', 'ds': df['date'], 'y': y.values})

    sf = StatsForecast(
        models=[AutoARIMA(season_length=7), AutoETS(season_length=7)],
        freq='D', n_jobs=1
    )
    sf.fit(train)
    raw = sf.predict(h=req.horizon_days, level=[80]).reset_index(drop=True)

    result = pd.DataFrame({
        'ds':       raw['ds'],
        'forecast': ((raw['AutoARIMA'] + raw['AutoETS']) / 2).clip(lower=0).round(0),
        'lower':    ((raw['AutoARIMA-lo-80'] + raw['AutoETS-lo-80']) / 2).clip(lower=0).round(0),
        'upper':    ((raw['AutoARIMA-hi-80'] + raw['AutoETS-hi-80']) / 2).clip(lower=0).round(0),
    })

    event_names = {}
    for idx, row in result.iterrows():
        mmdd = pd.Timestamp(row['ds']).strftime('%m-%d')
        if mmdd in BFCM_DATES:
            label, mult = BFCM_DATES[mmdd]
            result.at[idx, 'forecast'] = round(row['forecast'] * mult)
            result.at[idx, 'lower']    = round(row['lower'] * mult)
            result.at[idx, 'upper']    = round(row['upper'] * mult)
            event_names[idx] = label
        for ev in req.events:
            if pd.Timestamp(ev.start_date) <= pd.Timestamp(row['ds']) <= pd.Timestamp(ev.end_date):
                mult = 1 + ev.spike_pct / 100.0
                result.at[idx, 'forecast'] = round(result.at[idx, 'forecast'] * mult)
                result.at[idx, 'lower']    = round(result.at[idx, 'lower'] * mult)
                result.at[idx, 'upper']    = round(result.at[idx, 'upper'] * mult)
                event_names[idx] = ev.name

    points = [
        {'date': pd.Timestamp(r['ds']).strftime('%Y-%m-%d'),
         'forecast': int(r['forecast']),
         'lower':    int(r['lower']),
         'upper':    int(r['upper']),
         'event':    event_names.get(i)}
        for i, r in result.iterrows()
    ]

    model_info = {
        'training_rows': len(train),
        'training_from': train['ds'].min().strftime('%Y-%m-%d'),
        'training_to':   train['ds'].max().strftime('%Y-%m-%d'),
        'model':         'AutoARIMA + AutoETS ensemble',
        'regressors':    (['orders'] if has_orders else []) + (['revenue (AUD)'] if has_revenue else []),
        'horizon_days':  req.horizon_days,
    }

    return points, model_info


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/forecast")
async def forecast(file: UploadFile = File(...), params: str = "{}"):
    try:
        req = ForecastRequest(**json.loads(params))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid params: {e}")

    df = parse_csv(await file.read())
    points, model_info = run_model(df, req)

    return JSONResponse(content={'forecast': points, 'model_info': model_info})


@app.post("/forecast/save")
async def save_forecast(file: UploadFile = File(...), params: str = "{}"):
    try:
        req = ForecastRequest(**json.loads(params))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid params: {e}")

    df = parse_csv(await file.read())
    points, model_info = run_model(df, req)

    forecast_id = uuid.uuid4().hex[:8]
    conn = get_db()
    conn.execute(
        "INSERT INTO forecasts (id, created_at, model_info, forecast) VALUES (?, ?, ?, ?)",
        (forecast_id, datetime.utcnow().isoformat(), json.dumps(model_info), json.dumps(points))
    )
    conn.commit()
    conn.close()

    return JSONResponse(content={
        'id': forecast_id,
        'forecast': points,
        'model_info': model_info
    })


@app.get("/forecast/{forecast_id}")
async def get_forecast(forecast_id: str):
    conn = get_db()
    row = conn.execute(
        "SELECT id, created_at, model_info, forecast FROM forecasts WHERE id = ?",
        (forecast_id,)
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"Forecast '{forecast_id}' not found.")

    return JSONResponse(content={
        'id':         row[0],
        'created_at': row[1],
        'model_info': json.loads(row[2]),
        'forecast':   json.loads(row[3]),
    })


@app.get("/health")
def health():
    return {"status": "ok"}
