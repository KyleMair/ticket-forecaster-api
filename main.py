from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
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


class EventItem(BaseModel):
    name: str
    start_date: str
    end_date: str
    spike_pct: float
    event_type: str = 'custom'


class ForecastRequest(BaseModel):
    horizon_days: int = 90
    events: List[EventItem] = []


BFCM_DATES = {
    '11-27': ('Thanksgiving', 1.40),
    '11-28': ('Black Friday', 1.19),
    '11-29': ('Small Business Saturday', 1.08),
    '12-01': ('Cyber Monday', 1.15),
}


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


def apply_events(result: pd.DataFrame, events: List[EventItem], bfcm: dict):
    event_names = {}
    for idx in range(len(result)):
        ds = pd.Timestamp(result.loc[idx, 'ds'])
        mmdd = ds.strftime('%m-%d')

        # BFCM
        if mmdd in bfcm:
            label, mult = bfcm[mmdd]
            result.loc[idx, 'forecast'] = round(float(result.loc[idx, 'forecast']) * mult)
            result.loc[idx, 'lower']    = round(float(result.loc[idx, 'lower']) * mult)
            result.loc[idx, 'upper']    = round(float(result.loc[idx, 'upper']) * mult)
            event_names[idx] = (label, 'bfcm')

        # Custom events
        for ev in events:
            ev_start = pd.Timestamp(ev.start_date)
            ev_end   = pd.Timestamp(ev.end_date)
            if ev_start <= ds <= ev_end:
                mult = 1.0 + float(ev.spike_pct) / 100.0
                result.loc[idx, 'forecast'] = round(float(result.loc[idx, 'forecast']) * mult)
                result.loc[idx, 'lower']    = round(float(result.loc[idx, 'lower']) * mult)
                result.loc[idx, 'upper']    = round(float(result.loc[idx, 'upper']) * mult)
                event_names[idx] = (ev.name, ev.event_type)

    return result, event_names


def run_model(df: pd.DataFrame, req: ForecastRequest):
    has_orders  = 'orders'  in df.columns and df['orders'].notna().sum() > 10
    has_revenue = 'revenue' in df.columns and df['revenue'].notna().sum() > 10

    if has_orders and has_revenue:
        y = (df['tickets']  / df['tickets'].mean()  * 0.60
           + df['orders']   / df['orders'].mean()   * 0.25
           + df['revenue']  / df['revenue'].mean()  * 0.15) * df['tickets'].mean()
    elif has_orders:
        y = (df['tickets'] / df['tickets'].mean() * 0.70
           + df['orders']  / df['orders'].mean()  * 0.30) * df['tickets'].mean()
    else:
        y = df['tickets'].astype(float)

    train = pd.DataFrame({
        'unique_id': 'tickets',
        'ds': df['date'],
        'y': y.values.astype(float)
    })

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
    }).reset_index(drop=True)

    result, event_names = apply_events(result, req.events, BFCM_DATES)

    points = []
    for idx in range(len(result)):
        ev = event_names.get(idx)
        points.append({
            'date':       pd.Timestamp(result.loc[idx, 'ds']).strftime('%Y-%m-%d'),
            'forecast':   int(result.loc[idx, 'forecast']),
            'lower':      int(result.loc[idx, 'lower']),
            'upper':      int(result.loc[idx, 'upper']),
            'event':      ev[0] if ev else None,
            'event_type': ev[1] if ev else None,
        })

    model_info = {
        'training_rows': len(train),
        'training_from': train['ds'].min().strftime('%Y-%m-%d'),
        'training_to':   train['ds'].max().strftime('%Y-%m-%d'),
        'model':         'AutoARIMA + AutoETS ensemble',
        'regressors':    (['orders'] if has_orders else []) + (['revenue (AUD)'] if has_revenue else []),
        'horizon_days':  req.horizon_days,
    }

    return points, model_info


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
    return JSONResponse(content={'id': forecast_id, 'forecast': points, 'model_info': model_info})


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
