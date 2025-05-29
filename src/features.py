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
    
    return df
