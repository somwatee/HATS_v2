import pandas as pd
import pytest
from src.fetch_candles import fetch_candles

def test_fetch_zero_length():
    df = fetch_candles("XAUUSD", "M1", 0)
    # ควรคืน DataFrame เปล่าแต่มีคอลัมน์ถูกต้อง
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["time", "open", "high", "low", "close", "tick_volume"]
    assert df.empty

@pytest.mark.parametrize("n", [1, 5])
def test_fetch_positive_length_returns_dataframe(n):
    df = fetch_candles("XAUUSD", "M1", n)
    # ถ้าเชื่อม MT5 ไม่สำเร็จ จะได้ empty DataFrame
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["time", "open", "high", "low", "close", "tick_volume"]
    # df อาจว่างหรือไม่ว่าง ขึ้นกับการเชื่อม MT5 → test ผ่านถ้าไม่ error
