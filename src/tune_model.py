"""
tune_model.py

หา hyper-parameters ที่ดีที่สุดสำหรับ XGBoost ด้วย GridSearchCV
โดยเพิ่ม scale_pos_weight เพื่อรับมือข้อมูลไม่สมดุล
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# โหลด config
_cfg_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
with open(_cfg_path, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)


def tune_model(
    dataset_path: str,
    results_path: str = "models/hparam_results.csv",
    random_state: int = 42,
) -> None:
    df = pd.read_csv(dataset_path)
    X = df.drop(columns=["label"])
    y = df["label"].astype(str)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=random_state, stratify=y_enc
    )

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "scale_pos_weight": [1, 5, 10],  # ปรับตามสัดส่วน Buy/Sell ใน train
    }

    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=random_state,
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print("Best hyperparameters:", best_params)

    # เก็บผลลัพธ์ทั้งหมดเป็น CSV
    df_results = pd.DataFrame(grid_search.cv_results_)
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(results_path, index=False)


if __name__ == "__main__":
    tune_model(
        dataset_path=_cfg["dataset_path"],
        results_path="models/hparam_results.csv",
        random_state=_cfg.get("random_state", 42),
    )
