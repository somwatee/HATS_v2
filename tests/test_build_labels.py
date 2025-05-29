import pandas as pd
import pytest
from src.build_labels import build_labels

def test_build_labels_signature_and_length():
    # สร้าง DataFrame minimal
    df = pd.DataFrame({
        "high":  [1,2,3,4,5],
        "low":   [0,1,2,3,4],
        "close": [1,2,3,4,5],
    })
    result = build_labels(df, horizon=2)
    # ต้องคืน DataFrame พร้อมคอลัมน์ label และความยาวเท่าเดิม
    assert isinstance(result, pd.DataFrame)
    assert "label" in result.columns
    assert len(result) == len(df)

def test_build_labels_buy_sell_no_trade():
    # สร้างกรณีทดสอบ
    # index 0: future_high(1,2)>1 → Buy
    # index 1: future_high(2,3)>2 → Buy
    # index 2: future_low(1,2+? )<3? low window=(2,3) fl=2→fl<3 → Sell
    # index 3: future_high window has only index4 high=5>4→Buy
    # index 4: ไม่มีข้างหน้า → NoTrade
    data = {
        "high":  [1,2,3,4,5],
        "low":   [1,2,3,4,5],
        "close": [1,2,3,4,5],
    }
    df = pd.DataFrame(data)
    # horizon=2
    result = build_labels(df, horizon=2)
    # ทั้ง index 0–3 ควรเป็น Buy, index 4 ไม่มีข้างหน้า → NoTrade
    expected = ["Buy","Buy","Buy","Buy","NoTrade"]
    assert result["label"].tolist() == expected

def test_build_labels_default_horizon_from_config(tmp_path, monkeypatch):
    # ทดสอบใช้ horizon ค่า default จาก config
    # สร้างไฟล์ config.yaml ชั่วคราว
    cfg = tmp_path / "config.yaml"
    cfg.write_text("label_horizon: 3\n", encoding="utf-8")
    # Monkey-patch path
    import src.build_labels as bl
    monkeypatch.setattr(bl, "_cfg_path", cfg)
    # Reload module
    import importlib
    importlib.reload(bl)

    df = pd.DataFrame({
        "high":  [1,2,3,4],
        "low":   [1,2,3,4],
        "close": [1,2,3,4],
    })
    res = bl.build_labels(df)
    # horizon=3 → index 0-2 มีข้างหน้า ใช้ logic, index 3 ไม่มีข้างหน้า→NoTrade
    assert res["label"].iloc[-1] == "NoTrade"
