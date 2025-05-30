"""
walkforward.py

ทำ sliding walk-forward backtest บน dataset แล้วบันทึก metrics ของแต่ละ split
รองรับทั้ง TimeSeriesSplit (ถ้ามี n_splits) และ sliding-window
ใช้ oversampling + class weights ช่วยบาลานซ์ข้อมูลในแต่ละ window
"""

import pandas as pd
import xgboost as xgb
import yaml
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from imblearn.over_sampling import RandomOverSampler

# โหลด config
_cfg_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
with open(_cfg_path, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

WINDOW_SIZE = _cfg["walk_forward"]["window_size"]
STEP_SIZE = _cfg["walk_forward"]["step_size"]


def _balanced_fit_predict(X_train, y_train, X_test):
    """
    ทำ oversampling บน X_train, y_train แล้วเทรน XGBoost ด้วย class weight
    """
    # 1) Oversampling
    ros = RandomOverSampler(random_state=0)
    X_res, y_res = ros.fit_resample(X_train, y_train)

    # 2) คำนวณ class weights (ratio ของ negative/positive)
    # งานนี้มีหลายคลาส: weight สำหรับแต่ละ class = median_freq / freq(class)
    freqs = pd.Series(y_res).value_counts().to_dict()
    median = pd.Series(list(freqs.values())).median()
    class_weights = {cls: median/count for cls, count in freqs.items()}

    # 3) สร้างตัวแปร weight ตรงตาม sequence y_res
    sample_weight = pd.Series(y_res).map(class_weights).to_numpy()

    # 4) เทรนโมเดล
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
    model.fit(X_res, y_res, sample_weight=sample_weight)

    # 5) ทำนายบน X_test
    return model.predict(X_test)


def walk_forward(
    dataset_path: str,
    results_path: str,
    n_splits: int = None,  # ถ้าไม่ None ให้ใช้ TimeSeriesSplit
) -> None:
    """
    รัน walk-forward backtest

    ถ้า n_splits ไม่เป็น None: ใช้ TimeSeriesSplit(n_splits)
    ถ้า n_splits เป็น None: ใช้ sliding-window ตาม config

    Args:
        dataset_path (str)
        results_path (str)
        n_splits (int, optional)

    Returns:
        None
    """
    df = pd.read_csv(dataset_path)
    # เตรียม features & target
    drop_cols = ["label"]
    if "time" in df.columns:
        drop_cols.append("time")
    X = df.drop(columns=drop_cols)
    y = pd.Series(LabelEncoder().fit_transform(df["label"].astype(str)))

    results = []

    if n_splits is not None:
        # ใช้ TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if y_train.nunique() < 2:
                # ถ้ามีคลาสเดียว
                cls = y_train.iloc[0]
                y_pred = [cls] * len(y_test)
            else:
                # oversample + class weight
                try:
                    y_pred = _balanced_fit_predict(X_train, y_train, X_test)
                except Exception:
                    y_pred = [y_train.mode()[0]] * len(y_test)

            acc = accuracy_score(y_test, y_pred)
            results.append({"split": i, "accuracy": acc})

    else:
        # sliding-window
        max_start = len(df) - WINDOW_SIZE - STEP_SIZE + 1
        split = 0
        i = 0
        while i < max_start:
            train_slice = slice(i, i + WINDOW_SIZE)
            test_slice = slice(i + WINDOW_SIZE,
                               i + WINDOW_SIZE + STEP_SIZE)
            X_train, X_test = X.iloc[train_slice], X.iloc[test_slice]
            y_train, y_test = y.iloc[train_slice], y.iloc[test_slice]

            if y_train.nunique() < 2:
                cls = y_train.iloc[0]
                y_pred = [cls] * len(y_test)
            else:
                try:
                    y_pred = _balanced_fit_predict(X_train, y_train, X_test)
                except Exception:
                    y_pred = [y_train.mode()[0]] * len(y_test)

            acc = accuracy_score(y_test, y_pred)
            results.append({"split": split, "accuracy": acc})

            i += STEP_SIZE
            split += 1

    # บันทึกผล
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(results_path, index=False)
