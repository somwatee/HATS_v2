# run_phase1.py

import pandas as pd
import yaml
from pathlib import Path
from src.features import compute_features
from src.build_labels import build_labels
from src.dataset_builder import build_dataset


def main():
    # 1) โหลด config
    _cfg_path = Path(__file__).resolve().parent / "config" / "config.yaml"
    with open(_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2) ขั้นตอน Phase 1: Fetch → Features → Labels → Dataset
    # (สมมติ data/historical.csv มีอยู่แล้ว)
    df_candles = pd.read_csv(cfg["historical_data_path"])
    if not pd.api.types.is_datetime64_any_dtype(df_candles["time"]):
        df_candles["time"] = pd.to_datetime(df_candles["time"])

    # 2.1) Compute features
    feat_df = compute_features(df_candles)

    # 2.2) Build labels
    df_labels = build_labels(feat_df, horizon=cfg.get("label_horizon", None))

    # 2.3) Build dataset (คืน DataFrame ด้วย)
    # --- ตรงนี้เรียก build_dataset โดยไม่ต้องส่ง trade_log_df (None) ---
    df_dataset = build_dataset(df_labels)

    print("Phase 1 pipeline completed.")


if __name__ == "__main__":
    main()
