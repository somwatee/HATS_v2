import pandas as pd
import pytest
from src.features import compute_features

def test_compute_features_columns():
    # สร้าง DataFrame ตัวอย่าง
    data = {
        'time': pd.date_range(start='2025-01-01', periods=20, freq='min'),
        'open': [1.0 + i*0.1 for i in range(20)],
        'high': [1.1 + i*0.1 for i in range(20)],
        'low': [0.9 + i*0.1 for i in range(20)],
        'close': [1.0 + i*0.1 for i in range(20)],
        'tick_volume': [100 + i for i in range(20)],
    }
    df = pd.DataFrame(data)
    result = compute_features(df)

    # ควรมีคอลัมน์ rsi และ mss
    assert 'rsi' in result.columns
    assert 'mss' in result.columns
    # ความยาว DataFrame ไม่เปลี่ยน
    assert len(result) == len(df)

def test_rsi_values_range():
    # สร้าง DataFrame ที่ close คงที่ → rsi ควรเป็น 0.0 (หลังค่าดีฟอลต์)
    data = {
        'time': pd.date_range(start='2025-01-01', periods=15, freq='min'),
        'open': [1.0]*15,
        'high': [1.0]*15,
        'low': [1.0]*15,
        'close': [1.0]*15,
        'tick_volume': [100]*15,
    }
    df = pd.DataFrame(data)
    result = compute_features(df)
    # ทุกค่า rsi ที่ไม่ใช่ NaN ต้องเป็น 0.0
    assert all(result['rsi'].dropna() == 0.0)

def test_mss_detection():
    # high เรียงเพิ่ม → mss = 1 จาก index 3 ขึ้นไป
    highs = [1, 2, 3, 4, 5, 6]
    data = {
        'time': pd.date_range(start='2025-01-01', periods=len(highs), freq='min'),
        'open': highs,
        'high': highs,
        'low': highs,
        'close': highs,
        'tick_volume': [100]*len(highs),
    }
    df = pd.DataFrame(data)
    result = compute_features(df)
    # index 0-2 คือ None, index 3+ คือ 1
    expected = [None, None, None] + [1]*(len(highs)-3)
    assert result['mss'].tolist() == expected
