import pandas as pd
import xgboost as xgb
import yaml
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit

# โหลด config
_cfg_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
with open(_cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# พารามิเตอร์ XGBoost (ใน config หรือ hard-coded ชั่วคราว)
params = {
    "objective":     "multi:softprob",
    "num_class":     3,
    "eval_metric":   "mlogloss",
    "max_depth":     cfg.get("xgb_max_depth", 4),
    "eta":           cfg.get("xgb_eta", 0.05),
    "subsample":     cfg.get("xgb_subsample", 0.8),
    "colsample_bytree": cfg.get("xgb_colsample_bytree", 0.8),
    "random_state":  42,
}

def train_walkforward(
    dataset_path: str,
    model_output: str,
    results_output: str = "models/walkforward_classification_report.txt",
):
    df = pd.read_csv(dataset_path)
    # เตรียม features (สมมติชื่อคอลัมน์ตามที่ระบุ)
    feature_cols = [
        "mss_bullish", "mss_bearish",
        "fvg_bullish", "fvg_bearish", "fib_in_zone",
        "rsi", "ema9", "ema21",
        "atr", "adx",
        "vol_imbalance",
        # VWAP_diff = open - vwap
        # สมมติเจนคอลัมน์ “vwap_diff” ใน data_with_features.py
        "vwap_diff",
        "ema50_h4", "ema200_h4", "rsi_h4",
        # หมายเหตุ: เก็บ diff ของ BB บนลากเป็นฟีเจอร์
        "bb_upper_diff",  # e.g. close - bb_upper
        "bb_lower_diff",  # e.g. close - bb_lower
    ]
    X = df[feature_cols]
    # แปลง label: Buy=1, Sell=-1, NoTrade=0 → ต้องแมปเป็น [0,1,2] ก่อนส่งให้ XGBoost
    df["label_num"] = df["label"].map({"Buy": 1, "Sell": -1, "NoTrade": 0})
    # สร้าง mapping multi-class: we want 0→NoTrade, 1→Buy, 2→Sell
    # จาก label_num: ต้องเปลี่ยน -1 → 2
    df["label_code"] = df["label_num"].replace({-1: 2})
    y = df["label_code"]

    # ใช้ TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    all_reports = []
    fold = 0

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Scale pos weight: วัดจากสัดส่วนคลาสในชุด train
        counts = y_train.value_counts().to_dict()
        # counts ตัวอย่างเช่น {0: 8000, 1: 950, 2: 50}
        # scale_pos_weight = (sum of 2 other classes) / size_of_class_1  (เฉพาะ binary)
        # แต่ multi-class ให้ใช้ default หรือปรับเองได้ภายหลัง
        # ตัวอย่างง่าย ๆ ไม่ตั้ง scale_pos_weight

        clf = xgb.XGBClassifier(**params)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        report = classification_report(
            y_test, y_pred, labels=[1, 2, 0],
            target_names=["Buy", "Sell", "NoTrade"]
        )
        header = f"=== Fold {fold} ===\n"
        all_reports.append(header + report)
        fold += 1

    # บันทึก classification report ทั้งหมด
    Path(results_output).parent.mkdir(parents=True, exist_ok=True)
    with open(results_output, "w", encoding="utf-8") as f:
        f.write("\n".join(all_reports))

    # เทรนบนข้อมูลทั้งหมด แล้วเซฟโมเดล JSON
    clf_final = xgb.XGBClassifier(**params)
    clf_final.fit(X, y)
    Path(model_output).parent.mkdir(parents=True, exist_ok=True)
    clf_final.save_model(model_output)


if __name__ == "__main__":
    train_walkforward(
        "data/with_labels.csv",
        "models/xgb_hybrid_trading.json",
        "models/walkforward_classification_report.txt",
    )
