"""
walkforward.py

ทำ walk-forward backtest บน dataset แล้วบันทึก metrics ของแต่ละ split
รองรับทั้ง TimeSeriesSplit และ sliding-window
ใช้พารามิเตอร์ XGBoost จาก config/xgboost_params
"""

import pandas as pd
import xgboost as xgb
import yaml
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit

# โหลด config
_cfg_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
with open(_cfg_path, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

# อ่าน walk-forward settings
WINDOW_SIZE = _cfg["walk_forward"]["window_size"]
STEP_SIZE = _cfg["walk_forward"]["step_size"]
# อ่าน hyperparams สำหรับ XGBoost
xgb_params = _cfg.get("xgboost_params", {})


def _make_model():
    """สร้าง XGBClassifier จาก config['xgboost_params']"""
    return xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        n_estimators=xgb_params.get("n_estimators", 100),
        max_depth=xgb_params.get("max_depth", 3),
        learning_rate=xgb_params.get("learning_rate", 0.1),
        subsample=xgb_params.get("subsample", 1.0),
        colsample_bytree=xgb_params.get("colsample_bytree", 1.0),
    )


def walk_forward(
    dataset_path: str,
    results_path: str,
    n_splits: int = None,
) -> None:
    df = pd.read_csv(dataset_path)
    drop_cols = ["label"]
    if "time" in df.columns:
        drop_cols.append("time")
    X = df.drop(columns=drop_cols)
    y = pd.Series(LabelEncoder().fit_transform(df["label"].astype(str)))

    results = []

    if n_splits is not None:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if y_train.nunique() < 2:
                cls = y_train.iloc[0]
                y_pred = [cls] * len(y_test)
            else:
                try:
                    model = _make_model()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                except Exception:
                    y_pred = [y_train.mode()[0]] * len(y_test)

            acc = accuracy_score(y_test, y_pred)
            results.append({"split": i, "accuracy": acc})

    else:
        max_start = len(df) - WINDOW_SIZE - STEP_SIZE + 1
        split = 0
        idx = 0
        while idx < max_start:
            train_slice = slice(idx, idx + WINDOW_SIZE)
            test_slice = slice(idx + WINDOW_SIZE, idx + WINDOW_SIZE + STEP_SIZE)
            X_train, X_test = X.iloc[train_slice], X.iloc[test_slice]
            y_train, y_test = y.iloc[train_slice], y.iloc[test_slice]

            if y_train.nunique() < 2:
                cls = y_train.iloc[0]
                y_pred = [cls] * len(y_test)
            else:
                try:
                    model = _make_model()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                except Exception:
                    y_pred = [y_train.mode()[0]] * len(y_test)

            acc = accuracy_score(y_test, y_pred)
            results.append({"split": split, "accuracy": acc})

            idx += STEP_SIZE
            split += 1

    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(results_path, index=False)
