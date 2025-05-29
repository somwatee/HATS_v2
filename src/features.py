"""
features.py

คำนวณ feature หลัก (MSS, RSI) จาก DataFrame ของแท่งเทียน
"""

import pandas as pd

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
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MSS (simple trend shift)
    mss = [None] * len(df)
    for i in range(3, len(df)):
        h = df['high']
        if h.iloc[i-3] < h.iloc[i-2] < h.iloc[i-1] < h.iloc[i]:
            mss[i] = 1
        elif h.iloc[i-3] > h.iloc[i-2] > h.iloc[i-1] > h.iloc[i]:
            mss[i] = -1
        else:
            mss[i] = 0
    
    # กำหนด dtype เป็น object เพื่อเก็บ None และ ints ได้ตรงตาม test
    df['mss'] = pd.Series(mss, dtype=object)
    
# ตรงท้ายไฟล์ compute_features() ก่อน return df

    # --- Fair Value Gap (FVG) ---
    # สร้างคอลัมน์ default = 0
    df['fvg_bull'] = 0
    df['fvg_bear'] = 0
    # เริ่มจาก index 2 เป็นต้นไป (ต้องมีข้อมูล i-2)
    for i in range(2, len(df)):
        low_i = df.at[i, 'low']
        high_i2 = df.at[i-2, 'high']
        high_i = df.at[i, 'high']
        low_i2 = df.at[i-2, 'low']

        # Bullish FVG
        if low_i > high_i2:
            df.at[i, 'fvg_bull'] = 1
        # Bearish FVG
        elif high_i < low_i2:
            df.at[i, 'fvg_bear'] = 1

    # กลับไป return df ด้านล่าง (อย่าลืมว่าต้องเพิ่มสองคอลัมน์นี้เข้าไปใน output)

    # --- Exponential Moving Average (EMA) ---
    # กำหนด period สำหรับ EMA
    ema_period = 14
    # คำนวณ EMA จากราคาปิด
    df['ema'] = df['close'].ewm(span=ema_period, adjust=False).mean()


    return df

