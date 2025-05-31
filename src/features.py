# src/features.py

"""
features.py

คำนวณฟีเจอร์ต่าง ๆ จากข้อมูลแท่งเทียน (M1) สำหรับ Hybrid AI Trading EA

ฟีเจอร์เดิม: MSS, FVG, RSI, EMA, Fibonacci, ATR, ADX, Volume imbalance ฯลฯ
ฟีเจอร์ใหม่: VWAP, Higher-Timeframe Signals, Bollinger Bands, FVG Bull/Bear
"""

import pandas as pd
import numpy as np


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    รับ DataFrame มีคอลัมน์ขั้นต่ำ ['time','open','high','low','close','tick_volume']
    ต้องแปลง df['time'] เป็น datetime ก่อนเรียกฟังก์ชันนี้

    คืน DataFrame ที่เพิ่มคอลัมน์ฟีเจอร์ต่าง ๆ ได้แก่:
      - mss (Momentum Simple Spread)
      - fvg (Fair Value Gap)
      - fvg_bull, fvg_bear
      - rsi
      - ema
      - fibo_382, fibo_5, fibo_618
      - atr
      - adx
      - volume_imbalance
      - vwap
      - sma_h4, sma_d1 (Higher-Timeframe SMA)
      - bb_upper, bb_lower (Bollinger Bands)
    """
    df = df.copy()

    # 1) แปลง time เป็น datetime (ถ้ายังไม่ใช่)
    if not np.issubdtype(df["time"].dtype, np.datetime64):
        df["time"] = pd.to_datetime(df["time"])

    # ฟีเจอร์เดิม
    df["mss"] = _compute_mss(df)
    df["fvg"] = _compute_fvg(df)

    # ฟีเจอร์ FVG bull/bear
    df["fvg_bull"] = _compute_fvg_bull(df)
    df["fvg_bear"] = _compute_fvg_bear(df)

    df["rsi"] = _compute_rsi(df["close"], window=14)

    # เปลี่ยนชื่อคอลัมน์เป็น "ema" ให้ตรงกับ unit test
    df["ema"] = df["close"].ewm(span=20, adjust=False).mean()

    # คำนวณ Fibonacci แล้วใส่ลง DataFrame
    fibo_vals = _compute_fibonacci_levels(df, period=5)
    df["fibo_382"] = fibo_vals["fibo_382"]
    df["fibo_5"] = fibo_vals["fibo_5"]
    df["fibo_618"] = fibo_vals["fibo_618"]

    df["atr"] = _compute_atr(df, window=14)
    adx_vals = _compute_adx(df, window=14)
    df["adx"] = adx_vals["adx"]
    df["volume_imbalance"] = _compute_volume_imbalance(df)

    # ฟีเจอร์ใหม่
    df["vwap"] = _compute_intraday_vwap(df)
    df["sma_h4"] = _compute_htf_sma(df, span=50, freq="4h")
    df["sma_d1"] = _compute_htf_sma(df, span=20, freq="1d")
    bb = _compute_bollinger_bands(df["close"], window=20, num_std=2)
    df["bb_upper"] = bb["upper"]
    df["bb_lower"] = bb["lower"]

    return df


# ========== ฟังก์ชันย่อยปรับปรุง ==========
def _compute_mss(df: pd.DataFrame) -> pd.Series:
    """
    Momentum Simple Spread: max difference of highs over window size 4
    คืนค่า None สำหรับ 3 แถวแรก (dtype object)
    """
    highs = df["high"].values
    result = []
    for i in range(len(highs)):
        if i < 3:
            result.append(None)
        else:
            window_vals = highs[i - 3 : i + 1]
            diffs = np.diff(window_vals)
            if len(diffs) == 0:
                result.append(None)
            else:
                result.append(int(np.max(diffs)))
    return pd.Series(result, index=df.index, dtype=object)


def _compute_fvg(df: pd.DataFrame) -> pd.Series:
    """
    Fair Value Gap (เบื้องต้นใช้ placeholder)
    """
    return df["high"] - df["low"].shift(1)


def _compute_fvg_bull(df: pd.DataFrame) -> pd.Series:
    """
    Bullish Fair Value Gap: ถ้า low[i] > high[i-2] → คืน 1, มิฉะนั้น 0
    สำหรับ i<2 → คืน 0
    """
    highs = df["high"].values
    lows = df["low"].values
    result = []
    for i in range(len(lows)):
        if i < 2:
            result.append(0)
        else:
            result.append(1 if lows[i] > highs[i - 2] else 0)
    return pd.Series(result, index=df.index, dtype=object)


def _compute_fvg_bear(df: pd.DataFrame) -> pd.Series:
    """
    Bearish Fair Value Gap: ถ้า high[i] < low[i-2] → คืน 1, มิฉะนั้น 0
    สำหรับ i<2 → คืน 0
    """
    highs = df["high"].values
    lows = df["low"].values
    result = []
    for i in range(len(highs)):
        if i < 2:
            result.append(0)
        else:
            result.append(1 if highs[i] < lows[i - 2] else 0)
    return pd.Series(result, index=df.index, dtype=object)


def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI) โดยใช้ EMA กำลังสองฝั่ง
    """
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(span=window, adjust=False).mean()
    roll_down = down.ewm(span=window, adjust=False).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))


def _compute_fibonacci_levels(
    df: pd.DataFrame, period: int = 5
) -> dict[str, pd.Series]:
    """
    คำนวณ Fibonacci levels (0.382, 0.5, 0.618) จากข้อมูล high/low ย้อนกลับ period แท่ง

    - คืนค่า None สำหรับ index < period-1
    - สำหรับ index >= period-1 ใช้ window [i-period+1 ... i] (รวมตัวปัจจุบัน)
    """
    highs = df["high"].values
    lows = df["low"].values
    fib_382 = []
    fib_5 = []
    fib_618 = []
    n = len(df)
    for i in range(n):
        if i < period - 1:
            fib_382.append(None)
            fib_5.append(None)
            fib_618.append(None)
        else:
            start = i - period + 1
            end = i + 1  # รวมตำแหน่ง i
            window_high_max = highs[start:end].max()
            window_low_min = lows[start:end].min()
            diff = window_high_max - window_low_min
            fib_382.append(window_low_min + 0.382 * diff)
            fib_5.append(window_low_min + 0.5 * diff)
            fib_618.append(window_low_min + 0.618 * diff)

    return {
        "fibo_382": pd.Series(fib_382, index=df.index, dtype=object),
        "fibo_5": pd.Series(fib_5, index=df.index, dtype=object),
        "fibo_618": pd.Series(fib_618, index=df.index, dtype=object),
    }


def _compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Average True Range (ATR) ด้วยวิธี Rolling Average
    """
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=1).mean()


def _compute_adx(df: pd.DataFrame, window: int = 14) -> dict[str, pd.Series]:
    """
    Average Directional Index (ADX) กับ +DI, -DI (แต่คืนแค่อินเด็กซ์ "adx")
    """
    up_move = df["high"].diff()
    down_move = df["low"].shift(1) - df["low"]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=window, min_periods=1).mean()
    plus_di = 100 * (
        pd.Series(plus_dm).rolling(window=window, min_periods=1).mean() / atr
    )
    minus_di = 100 * (
        pd.Series(minus_dm).rolling(window=window, min_periods=1).mean() / atr
    )
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=window, min_periods=1).mean()
    return {"adx": adx}


def _compute_volume_imbalance(df: pd.DataFrame) -> pd.Series:
    """
    Volume Imbalance ตามทิศทางราคา:
    - idx 0 ให้ถือเป็น "ขึ้น" → 1.0
    - idx > 0:
        * ถ้า close[i] > close[i-1] →  1.0
        * ถ้า close[i] < close[i-1] → -1.0
        * ถ้า close[i] == close[i-1] → 0.0
    """
    closes = df["close"].values
    result = []
    for i in range(len(closes)):
        if i == 0:
            # แถวแรกถือว่าราคา "ขึ้น"
            result.append(1.0)
        else:
            if closes[i] > closes[i - 1]:
                result.append(1.0)
            elif closes[i] < closes[i - 1]:
                result.append(-1.0)
            else:
                result.append(0.0)
    return pd.Series(result, index=df.index, dtype=float)


# ========== ฟังก์ชันย่อยสำหรับฟีเจอร์ใหม่ ==========
def _compute_intraday_vwap(df: pd.DataFrame) -> pd.Series:
    """
    VWAP (Volume-Weighted Average Price) แยกเป็นรายวัน (date)
    """
    df = df.copy()
    df["date_only"] = df["time"].dt.date
    df["pv"] = df["close"] * df["tick_volume"]
    df["cum_pv"] = df.groupby("date_only")["pv"].cumsum()
    df["cum_vol"] = df.groupby("date_only")["tick_volume"].cumsum()
    vwap = df["cum_pv"] / df["cum_vol"]
    return vwap.ffill().fillna(0.0)


def _compute_htf_sma(df: pd.DataFrame, span: int, freq: str) -> pd.Series:
    """
    Higher-Timeframe SMA:
    - รีแซมเปิลจาก df["time"] ตาม freq (เช่น "4H" หรือ "1D")
    - คำนวณ SMA ด้วย window=span บนข้อมูลรีแซมเปิล
    - หลังจบสร้างค่ากลับมาเติมให้ตรง index เดิม
    """
    tmp = df.set_index("time")[["close"]]
    htf = tmp["close"].resample(freq).last().ffill()
    sma_htf = htf.rolling(window=span, min_periods=1).mean()
    sma_htf_aligned = sma_htf.reindex(df["time"], method="ffill").values
    return pd.Series(sma_htf_aligned, index=df.index)


def _compute_bollinger_bands(
    series: pd.Series, window: int = 20, num_std: float = 2.0
) -> dict[str, pd.Series]:
    """
    Bollinger Bands:
    - MA = rolling mean(window)
    - STD = rolling std(window)
    - upper = MA + num_std * STD
    - lower = MA - num_std * STD
    """
    ma = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std().fillna(0.0)
    upper = ma + num_std * std
    lower = ma - num_std * std
    return {"upper": upper, "lower": lower}
