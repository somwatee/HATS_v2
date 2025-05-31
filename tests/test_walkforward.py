import pandas as pd
import os
import pytest
from src.walkforward import walk_forward


def test_walk_forward_signature_and_output(tmp_path, monkeypatch):
    # เตรียม dataset สั้นๆ
    df = pd.DataFrame(
        {
            "feature": list(range(10)),
            "label": ["A"] * 10,  # label คงที่ → accuracy = 1.0 ทุก split
        }
    )
    data_file = tmp_path / "dataset.csv"
    df.to_csv(data_file, index=False)

    results_file = tmp_path / "results.csv"

    # Monkeypatch XGBClassifier.fit/predict เพื่อเลี่ยง compute จริง
    import xgboost as xgb

    monkeypatch.setattr(xgb.XGBClassifier, "fit", lambda self, X, y: None)
    monkeypatch.setattr(xgb.XGBClassifier, "predict", lambda self, X: [0] * len(X))

    # รัน walk-forward
    walk_forward(str(data_file), str(results_file), n_splits=2)

    # ตรวจว่าผลลัพธ์เป็นไฟล์ CSV
    assert os.path.exists(str(results_file))

    res = pd.read_csv(results_file)
    # ควรมีสองแถว (n_splits=2) และสองคอลัมน์
    assert list(res.columns) == ["split", "accuracy"]
    assert len(res) == 2
    # accuracy ควรเป็น 1.0
    assert all(res["accuracy"] == 1.0)
