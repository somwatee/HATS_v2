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

def test_fvg_detection():
    # กรณี Bullish FVG ที่ index=2 (low[2] > high[0])
    data = {
        'time': pd.date_range('2025-01-01', periods=4, freq='min'),
        'open': [1,1,1,1],
        'high': [1,1,1,1],
        'low':  [1,1,3,1],  # low[2]=3 > high[0]=1 → fvg_bull=1 ที่ index 2
        'close':[1,1,1,1],
        'tick_volume': [10,10,10,10],
    }
    df = pd.DataFrame(data)
    result = compute_features(df)
    # ตรวจคอลัมน์ใหม่
    assert 'fvg_bull' in result.columns
    assert 'fvg_bear' in result.columns
    # Check values
    expected_bull = [0, 0, 1, 0]
    expected_bear = [0, 0, 0, 0]
    assert result['fvg_bull'].tolist() == expected_bull
    assert result['fvg_bear'].tolist() == expected_bear

    # กรณี Bearish FVG ที่ index=2 (high[2] < low[0])
    data2 = {
        'time': pd.date_range('2025-01-01', periods=4, freq='min'),
        'open': [2,2,2,2],
        'high': [2,2,1,2],  # high[2]=1 < low[0]=2 → fvg_bear=1 ที่ index 2
        'low':  [2,2,2,2],
        'close':[2,2,2,2],
        'tick_volume': [10,10,10,10],
    }
    df2 = pd.DataFrame(data2)
    result2 = compute_features(df2)
    expected_bull2 = [0, 0, 0, 0]
    expected_bear2 = [0, 0, 1, 0]
    assert result2['fvg_bull'].tolist() == expected_bull2
    assert result2['fvg_bear'].tolist() == expected_bear2


def test_ema_column_and_length():
    import pandas as pd
    from src.features import compute_features

    # สร้าง DataFrame ตัวอย่าง
    data = {
        'time': pd.date_range('2025-01-01', periods=5, freq='min'),
        'open': [1, 2, 3, 4, 5],
        'high': [1, 2, 3, 4, 5],
        'low':  [1, 2, 3, 4, 5],
        'close':[1, 2, 3, 4, 5],
        'tick_volume': [10]*5,
    }
    df = pd.DataFrame(data)
    result = compute_features(df)

    # เช็คว่าเพิ่มคอลัมน์ ema และความยาวไม่เปลี่ยน
    assert 'ema' in result.columns
    assert len(result) == len(df)

def test_ema_increasing_behavior():
    import pandas as pd
    from src.features import compute_features

    # สร้าง DataFrame ที่ close เพิ่มขึ้นทุกแท่ง
    data = {
        'time': pd.date_range('2025-01-01', periods=5, freq='min'),
        'open': [1, 2, 3, 4, 5],
        'high': [1, 2, 3, 4, 5],
        'low':  [1, 2, 3, 4, 5],
        'close':[1, 2, 3, 4, 5],
        'tick_volume': [10]*5,
    }
    df = pd.DataFrame(data)
    result = compute_features(df)

    # EMA ควรเป็นลำดับเพิ่มขึ้นเช่นกัน
    ema_vals = result['ema'].tolist()
    # ทุกคู่สองตำแหน่งถัดไป ต้อง ema[i] <= ema[i+1]
    assert all(ema_vals[i] <= ema_vals[i+1] for i in range(len(ema_vals)-1))


def test_fibonacci_levels_columns_and_default_none():
    import pandas as pd
    from src.features import compute_features

    data = {
        'time': pd.date_range('2025-01-01', periods=7, freq='min'),
        'open': list(range(7)),
        'high': list(range(7)),
        'low':  list(range(7)),
        'close':list(range(7)),
        'tick_volume': [10]*7,
    }
    df = pd.DataFrame(data)
    result = compute_features(df)

    # ตรวจคอลัมน์
    for col in ['fibo_382', 'fibo_5', 'fibo_618']:
        assert col in result.columns

    # ก่อน index 4 (fib_period-1) ต้องเป็น None
    assert all(val is None for val in result['fibo_382'][:4])

def test_fibonacci_levels_values_correct():
    import pandas as pd
    import pytest
    from src.features import compute_features

    data = {
        'time': pd.date_range('2025-01-01', periods=5, freq='min'),
        'open': list(range(5)),
        'high': list(range(5)),  # 0,1,2,3,4
        'low':  list(range(5)),
        'close':list(range(5)),
        'tick_volume': [10]*5,
    }
    df = pd.DataFrame(data)
    result = compute_features(df)

    # ที่ index=4: max_h=4,min_l=0,diff=4
    # fibo382=1.528, fibo5=2.0, fibo618=2.472
    assert pytest.approx(1.528, rel=1e-3) == result.at[4, 'fibo_382']
    assert pytest.approx(2.0,   rel=1e-3) == result.at[4, 'fibo_5']
    assert pytest.approx(2.472, rel=1e-3) == result.at[4, 'fibo_618']
