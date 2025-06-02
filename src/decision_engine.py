import xgboost as xgb
import pandas as pd

class DecisionEngine:
    def __init__(self, model_path: str):
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)

    def predict_signal(self, features: dict):
        """
        features: dict รูปแบบ { "mss_bullish": 0/1, "mss_bearish": 0/1, ..., "bb_lower_diff": float }
        ต้องแน่ใจว่าคีย์ตรงกับ feature_cols ใช้ตอน train
        """
        df = pd.DataFrame([features])
        # เรียงคอลัมน์ให้ตรงกัน
        feature_cols = [
            "mss_bullish", "mss_bearish",
            "fvg_bullish", "fvg_bearish", "fib_in_zone",
            "rsi", "ema9", "ema21",
            "atr", "adx",
            "vol_imbalance", "vwap_diff",
            "ema50_h4", "ema200_h4", "rsi_h4",
            "bb_upper_diff", "bb_lower_diff",
        ]
        X = df[feature_cols]

        pred_code = self.model.predict(X)[0]  # คืนค่า 0/1/2
        proba = self.model.predict_proba(X)[0]  # [prob_noTrade, prob_buy, prob_sell] ตามลำดับ
        # แปลงรหัสกลับเป็นชื่อสัญญาณ
        code_to_label = {0: "NoTrade", 1: "Buy", 2: "Sell"}
        label = code_to_label.get(pred_code, "NoTrade")
        confidence = max(proba)

        return label, confidence

# ใช้งานตัวอย่าง
if __name__ == "__main__":
    engine = DecisionEngine("models/xgb_hybrid_trading.json")
    sample_features = {
        "mss_bullish": 1,
        "mss_bearish": 0,
        "fvg_bullish": 1,
        "fvg_bearish": 0,
        "fib_in_zone": True,
        "rsi": 25.3,
        "ema9": 1800.5,
        "ema21": 1798.2,
        "atr": 12.5,
        "adx": 30.0,
        "vol_imbalance": 0.25,
        "vwap_diff": 1800.2 - 1799.0,      # แค่ตัวอย่าง
        "ema50_h4": 1802.0,
        "ema200_h4": 1795.0,
        "rsi_h4": 60.0,
        "bb_upper_diff": 1803.0 - 1800.5,
        "bb_lower_diff": 1795.0 - 1800.5,
    }
    sig, conf = engine.predict_signal(sample_features)
    print(f"Signal = {sig}, confidence = {conf:.3f}")
