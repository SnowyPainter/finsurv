# build_pipeline.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# -----------------------
# Config
# -----------------------
DATA_ROOT = Path("data")
RAW_DIR = DATA_ROOT / "raw"
SILVER_DIR = DATA_ROOT / "silver"
GOLD_DIR = DATA_ROOT / "gold"

HORIZON = 30
UPPER = 0.10
LOWER = -0.10

# Technical indicator params
MA_WINDOWS = [5, 10, 20, 50, 100, 200]
VOL_WINDOWS = [5, 10, 20, 60]
RSI_WINDOW = 14
BB_WINDOW = 20
BB_K = 2.0
ATR_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

MIN_PRICE = 0.5  # optional filter to avoid extreme penny noise

ID_COLS = ["ticker", "Date"]
LABEL_COLS = ["duration", "event"]
DROP_COLS = [
    "start_price",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Dividends",
    "Stock Splits",
]


# -----------------------
# Helpers: indicators
# -----------------------
def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = set(ID_COLS + LABEL_COLS)
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def write_feature_manifest(columns: list[str]) -> None:
    manifest_path = DATA_ROOT / "available_features.json"
    text_path = DATA_ROOT / "available_features.txt"

    payload = {
        "feature_count": len(columns),
        "features": columns,
        "scaling": "apply train-fit scaling at experiment time (load_dataset.py)",
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    text_path.write_text("\n".join(columns) + "\n", encoding="utf-8")


# -----------------------
# 3) GOLD: features + merge labels
# -----------------------
def add_features_per_ticker(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("Date").copy()

    o = g["Open"]
    h = g["High"]
    l = g["Low"]
    c = g["Close"]
    v = g["Volume"].fillna(0)

    # Basic returns
    g["ret_1"] = c.pct_change(1)
    g["logret_1"] = np.log(c).diff()

    # Multi-horizon returns / vol
    for w in [5, 10, 20, 60]:
        g[f"ret_{w}"] = c.pct_change(w)
        g[f"vol_{w}"] = c.pct_change().rolling(w).std()

    # Moving averages & distance
    for w in MA_WINDOWS:
        ma = c.rolling(w).mean()
        g[f"sma_{w}"] = ma
        g[f"dist_sma_{w}"] = (c / ma) - 1.0

    # Volume features
    for w in [5, 10, 20, 60]:
        g[f"volm_mean_{w}"] = v.rolling(w).mean()
        g[f"volm_z_{w}"] = (v - v.rolling(w).mean()) / v.rolling(w).std()

    # Price range / candle
    g["hl_range"] = (h - l) / c.replace(0, np.nan)
    g["oc_return"] = (c - o) / o.replace(0, np.nan)

    # ATR
    g[f"atr_{ATR_WINDOW}"] = atr(h, l, c, ATR_WINDOW)
    g[f"atrp_{ATR_WINDOW}"] = g[f"atr_{ATR_WINDOW}"] / c.replace(0, np.nan)  # ATR%

    # RSI
    g[f"rsi_{RSI_WINDOW}"] = rsi(c, RSI_WINDOW)

    # Bollinger Bands
    mid = c.rolling(BB_WINDOW).mean()
    sd = c.rolling(BB_WINDOW).std()
    upper = mid + BB_K * sd
    lower = mid - BB_K * sd
    g[f"bb_mid_{BB_WINDOW}"] = mid
    g[f"bb_width_{BB_WINDOW}"] = (upper - lower) / mid.replace(0, np.nan)
    g[f"bb_pos_{BB_WINDOW}"] = (c - lower) / (upper - lower).replace(0, np.nan)

    # MACD
    ema_fast = _ema(c, MACD_FAST)
    ema_slow = _ema(c, MACD_SLOW)
    macd_line = ema_fast - ema_slow
    macd_signal = _ema(macd_line, MACD_SIGNAL)
    g["macd"] = macd_line
    g["macd_signal"] = macd_signal
    g["macd_hist"] = macd_line - macd_signal

    # OBV
    g["obv"] = obv(c, v)
    g["obv_ret_5"] = g["obv"].pct_change(5)

    # Simple momentum/trend proxies
    g["mom_10_20"] = g["ret_10"] - g["ret_20"]
    g["trend_20"] = g["dist_sma_20"]

    return g


def build_gold(raw_path: Path, silver_path: Path) -> Path:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(raw_path)
    df = df.sort_values(["ticker", "Date"])

    # Add features per ticker
    tqdm.pandas(desc="Feature engineering by ticker")
    feat = df.groupby("ticker", group_keys=False, sort=False).progress_apply(add_features_per_ticker)

    # Merge with labels
    silver = pd.read_parquet(silver_path)

    gold = silver.merge(
        feat,
        on=["ticker", "Date"],
        how="left",
    )

    # Remove duplicated labels on the same observation key.
    before_rows = len(gold)
    gold = gold.drop_duplicates(subset=["ticker", "Date"], keep="first").reset_index(drop=True)
    removed_dupes = before_rows - len(gold)

    # Remove redundant raw/label-like columns.
    existing_drop = [c for c in DROP_COLS if c in gold.columns]
    if existing_drop:
        gold = gold.drop(columns=existing_drop)

    # Drop NA rows produced by rolling windows etc.
    gold = gold.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    # Drop constant/invalid columns and write available feature list.
    gold = gold.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    feature_cols = infer_feature_columns(gold)
    write_feature_manifest(feature_cols)

    out_path = GOLD_DIR / "survival_dataset.parquet"
    gold.to_parquet(out_path, index=False)
    if removed_dupes > 0:
        print(f"[GOLD] removed duplicate labels: {removed_dupes:,}")
    print(f"[GOLD] feature manifest: {DATA_ROOT / 'available_features.json'}")
    print(f"[GOLD] saved: {out_path} rows={len(gold):,}")
    return out_path


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    raw_path = RAW_DIR / "all_stocks.parquet"
    silver_path = SILVER_DIR / "survival_windows.parquet"
    gold_path = build_gold(raw_path, silver_path)
