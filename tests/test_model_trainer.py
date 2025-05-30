import pandas as pd
import tempfile
import os
import pytest
from src.model_trainer import train_model
import xgboost as xgb

def test_train_model_signature(tmp_path, monkeypatch):
    # จับ dummy dataset ลง temp file
    df = pd.DataFrame({
        'feature1': [1,2,3,4],
        'feature2': [0.1,0.2,0.3,0.4],
        'label':    ['Buy','Sell','Buy','NoTrade']
    })
    dataset_file = tmp_path / "dataset.csv"
    df.to_csv(dataset_file, index=False)

    # สร้าง output paths
    model_file = tmp_path / "model.json"
    report_file = tmp_path / "report.txt"
    imp_file = tmp_path / "importance.csv"

    # Monkey-patch any actual training to avoid heavy compute
    monkeypatch.setattr(xgb.XGBClassifier, "fit", lambda self, X, y: None)

    # เรียกฟังก์ชัน ไม่ควร error
    train_model(
        str(dataset_file),
        str(model_file),
        report_path=str(report_file),
        importance_path=str(imp_file)
    )

    # ตรวจว่าไฟล์ output ถูกสร้าง (หรือถูกสร้างด้วย pass อาจไม่ขึ้น error)
    assert os.path.exists(str(model_file))
    assert os.path.exists(str(report_file))
    assert os.path.exists(str(imp_file))
