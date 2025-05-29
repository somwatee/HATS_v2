"""
features.py

คำนวณ feature หลัก (MSS, RSI) จาก DataFrame ของแท่งเทียน
"""


import pandas as pd
import numpy as np


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    รับ DataFrame ของแท่งเทียน → คืน DataFrame ที่เพิ่มคอลัมน์ features

    Args:
        df (pd.DataFrame): คอลัมน์ ['time','open','high','low','close',…]
    Returns:
        pd.DataFrame: เพิ่มคอลัมน์ ['mss', 'rsi']
    """
    df = df.copy().reset_index(drop=True)

    # RSI (Wilder’s smoothing)
    period = 14
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # MSS (simple trend shift)
    mss = [None] * len(df)
    for i in range(3, len(df)):
        h = df["high"]
        if h.iloc[i - 3] < h.iloc[i - 2] < h.iloc[i - 1] < h.iloc[i]:
            mss[i] = 1
        elif h.iloc[i - 3] > h.iloc[i - 2] > h.iloc[i - 1] > h.iloc[i]:
            mss[i] = -1
        else:
            mss[i] = 0
    df["mss"] = pd.Series(mss, dtype=object)

    # --- Fair Value Gap (FVG) ---
    df["fvg_bull"] = 0
    df["fvg_bear"] = 0
    for i in range(2, len(df)):
        low_i = df.at[i, "low"]
        high_i2 = df.at[i - 2, "high"]
        high_i = df.at[i, "high"]
        low_i2 = df.at[i - 2, "low"]

        if low_i > high_i2:
            df.at[i, "fvg_bull"] = 1
        elif high_i < low_i2:
            df.at[i, "fvg_bear"] = 1

    # --- Exponential Moving Average (EMA) ---
    ema_period = 14
    df["ema"] = df["close"].ewm(span=ema_period, adjust=False).mean()

    # --- Fibonacci Levels ---
    fib_period = 5
    fibo_382 = [None] * len(df)
    fibo_5 = [None] * len(df)
    fibo_618 = [None] * len(df)
    for i in range(fib_period - 1, len(df)):
        window_high = df["high"].iloc[i - (fib_period - 1) : i + 1]
        window_low = df["low"].iloc[i - (fib_period - 1) : i + 1]
        max_h = window_high.max()
        min_l = window_low.min()
        diff = max_h - min_l
        fibo_382[i] = min_l + 0.382 * diff
        fibo_5[i] = min_l + 0.5 * diff
        fibo_618[i] = min_l + 0.618 * diff
    df["fibo_382"] = pd.Series(fibo_382, dtype=object)
    df["fibo_5"] = pd.Series(fibo_5, dtype=object)
    df["fibo_618"] = pd.Series(fibo_618, dtype=object)

    # --- Average True Range (ATR) ---
    atr_period = 14
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / atr_period, adjust=False).mean()

    # --- ADX (Average Directional Index) ---
    adx_period = 14
    # Directional Movement
    up_move = df['high'].diff()
    down_move = df['low'].shift(1) - df['low']
    dm_plus = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    dm_minus = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # Wilder’s smoothing for DM and TR
    sm_tr = tr.ewm(alpha=1/atr_period, adjust=False).mean()
    sm_dm_plus = dm_plus.ewm(alpha=1/adx_period, adjust=False).mean()
    sm_dm_minus = dm_minus.ewm(alpha=1/adx_period, adjust=False).mean()

    # Directional Indicators
    plus_di = 100 * (sm_dm_plus / sm_tr)
    minus_di = 100 * (sm_dm_minus / sm_tr)

    # DX and ADX
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    df['adx'] = dx.ewm(alpha=1/adx_period, adjust=False).mean()

    # --- Volume Imbalance ---
    vol_up = df['tick_volume'].where(df['close'] > df['open'], 0.0)
    vol_down = df['tick_volume'].where(df['open'] > df['close'], 0.0)
    raw = (vol_up - vol_down) / df['tick_volume']
    imbalance = raw.replace([np.inf, -np.inf], pd.NA).fillna(0.0).astype(float)
    df['volume_imbalance'] = imbalance

    return df
