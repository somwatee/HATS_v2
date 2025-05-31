"""
fetch_candles.py

เชื่อมต่อ MT5 ตาม config แล้วดึงแท่งเทียนล่าสุดคืนค่าเป็น pandas.DataFrame
"""

import pandas as pd
import MetaTrader5 as mt5
import yaml
from pathlib import Path

# โหลด config.yaml (ระบุ encoding เพื่อรองรับ Unicode)
_cfg_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
with open(_cfg_path, "r", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)

# Mapping timeframe string → MetaTrader5 constant
_TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


def fetch_candles(symbol: str, timeframe: str, n: int) -> pd.DataFrame:
    """
    ดึง n แท่งเทียนล่าสุดของสัญลักษณ์และ timeframe ที่กำหนด

    Args:
        symbol (str): เช่น "XAUUSD"
        timeframe (str): เช่น "M1", "H1"
        n (int): จำนวนแท่งที่ต้องการดึง
    Returns:
        pandas.DataFrame:
            คอลัมน์ [
                'time',
                'open',
                'high',
                'low',
                'close',
                'tick_volume'
            ]
    """
    # กรณี n ไม่บวก หรือ timeframe ไม่รองรับ → คืน DataFrame เปล่าพร้อมคอลัมน์
    cols = ["time", "open", "high", "low", "close", "tick_volume"]
    if n <= 0 or timeframe not in _TIMEFRAME_MAP:
        return pd.DataFrame(columns=cols)

    # เชื่อมต่อ MT5
    mt5_cfg = _cfg.get("mt5", {})
    ok = mt5.initialize(
        path=mt5_cfg.get("terminal_path"),
        login=mt5_cfg.get("login"),
        password=mt5_cfg.get("password"),
        server=mt5_cfg.get("server"),
        timeout=mt5_cfg.get("timeout", 5000),
    )
    if not ok:
        return pd.DataFrame(columns=cols)

    # ดึงข้อมูลแท่งเทียน
    tf_const = _TIMEFRAME_MAP[timeframe]
    rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, n)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        return pd.DataFrame(columns=cols)

    # สร้าง DataFrame และแปลง time
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    # กรองเฉพาะคอลัมน์ที่ต้องการ
    df = df.loc[:, cols]
    return df
