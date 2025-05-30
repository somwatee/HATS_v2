import pandas as pd
import tempfile
import os
import pytest
from src.model_trainer import train_model

def test_train_model_signature(tmp_path, monkeypatch):
    # สร้างไฟล์ dataset ชั่วคราว
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

    # Monkey-patch fit เพื่อไม่ให้รันจริงหนัก
    import xgboost as xgb
    monkeypatch.setattr(xgb.XGBClassifier, "fit", lambda self, X, y: None)

    # เรียก train_model ไม่ควร error
    train_model(
        str(dataset_file),
        str(model_file),
        report_path=str(report_file),
        importance_path=str(imp_file),
        test_size=0.5,
        random_state=0
    )

    # ตรวจว่าไฟล์ output ถูกสร้าง
    assert os.path.exists(str(model_file))
    assert os.path.exists(str(report_file))
    assert os.path.exists(str(imp_file))

def test_report_and_importance_content(tmp_path):
    # สร้างไฟล์ dataset ชุดเล็กเพื่อทดสอบรายงานจริง
    df = pd.DataFrame({
        'feature1': [1,2,3,4,5],
        'label':    ['Buy','Sell','Buy','Sell','NoTrade']
    })
    data_file = tmp_path / "data.csv"
    df.to_csv(data_file, index=False)

    report_file = tmp_path / "report.txt"
    imp_file = tmp_path / "importance.csv"
    model_file = tmp_path / "model.json"

    # เรียก train_model
    train_model(
        str(data_file),
        str(model_file),
        report_path=str(report_file),
        importance_path=str(imp_file),
        test_size=0.4,
        random_state=0
    )

    # เช็กเนื้อหา report.txt เริ่มต้นด้วย 'precision'
    text = report_file.read_text(encoding="utf-8")
    assert text.strip().startswith("precision")

    # เช็กว่า importance.csv มีสองคอลัมน์ feature,importance
    df_imp = pd.read_csv(imp_file)
    assert list(df_imp.columns) == ["feature", "importance"]
