# src/walkforward.py

"""
walkforward.py

ทำ rolling walk-forward backtest บน dataset แล้วบันทึก metrics ของแต่ละ split
"""

import pandas as pd
import xgboost as xgb
import yaml
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit

# โหลด config (อ่านจำนวน splits จาก config ถ้ามี)
_cfg_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
with open(_cfg_path, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)
_n_splits = _cfg.get("walkforward_splits", 5)


def walk_forward(
    dataset_path: str,
    results_path: str,
    n_splits: int = None,
) -> None:
    """
    รัน walk-forward backtest

    Args:
        dataset_path (str): path ไปยัง data/dataset.csv
        results_path (str): path สำหรับบันทึกผล walkforward_results.csv
        n_splits (int, optional): จำนวน splits ถ้าไม่กำหนดจะอ่านจาก config

    Returns:
        None
    """
    # โหลด dataset
    df = pd.read_csv(dataset_path)

    # เตรียม features: drop 'label' และ 'time' (ถ้ามี)
    drop_cols = ["label"]
    if "time" in df.columns:
        drop_cols.append("time")
    X = df.drop(columns=drop_cols)

    # เตรียม target
    y = LabelEncoder().fit_transform(df["label"].astype(str))

    # กำหนดจำนวน splits
    splits = n_splits or _n_splits
    tscv = TimeSeriesSplit(n_splits=splits)

    results = []
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        model = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric="mlogloss",
        )
        try:
            # train & predict
            model.fit(X.iloc[train_idx], y[train_idx])
            y_pred = model.predict(X.iloc[test_idx])
            acc = accuracy_score(y[test_idx], y_pred)
        except Exception:
            # ในกรณี fit/predict ผิดพลาด ให้เก็บ accuracy เป็น NaN
            acc = float("nan")

        results.append({"split": i, "accuracy": acc})

    # สร้างโฟลเดอร์ผลลัพธ์ถ้ายังไม่มี
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    # บันทึกผลเป็น CSV
    pd.DataFrame(results).to_csv(results_path, index=False)
