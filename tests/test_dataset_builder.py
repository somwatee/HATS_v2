import pandas as pd
import pytest
from src.dataset_builder import build_dataset

def test_build_dataset_signature_and_length():
    # สร้าง features_df ตัวอย่าง
    data = {
        'time': pd.date_range('2025-01-01', periods=3, freq='min'),
        'feature1': [1,2,3],
        'label': ['Buy','Sell','NoTrade']
    }
    features_df = pd.DataFrame(data)
    result = build_dataset(features_df)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ['time','feature1','label']
    assert len(result) == 3

def test_build_dataset_with_trade_log_merge():
    # สร้าง features_df
    times = pd.date_range('2025-01-01', periods=3, freq='min')
    features_df = pd.DataFrame({
        'time': times,
        'feature1': [1,2,3],
        'label': ['Buy','Sell','NoTrade']
    })
    # สร้าง trade_log_df
    log = pd.DataFrame({
        'timestamp': [times[0], times[2]],
        'side': ['Buy','Sell'],
        'pnl': [0.5, -0.2]
    })
    result = build_dataset(features_df, log)
    # คอลัมน์ควรรวม 'timestamp','side','pnl'
    for col in ['timestamp','side','pnl']:
        assert col in result.columns
    # ตรวจค่าที่ merge: row index1 ไม่มี log → NaN
    assert pd.isna(result.loc[1,'side'])
    assert result.loc[0,'side'] == 'Buy'
    assert result.loc[2,'pnl'] == -0.2
