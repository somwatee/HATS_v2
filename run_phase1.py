# run_phase1.py

from pathlib import Path
import yaml
import pandas as pd

from src.fetch_candles    import fetch_candles
from src.features         import compute_features
from src.build_labels     import build_labels
from src.dataset_builder  import build_dataset

def main():
    # Load config
    cfg = yaml.safe_load(open("config/config.yaml", encoding="utf-8"))

    symbol    = cfg["symbol"]
    timeframe = cfg["timeframe"]
    horizon   = cfg.get("label_horizon", 5)
    N         = cfg.get("fetch_candles_n", 500)

    # เตรียมโฟลเดอร์ data
    Path("data").mkdir(exist_ok=True)

    # 1) Fetch
    df_hist = fetch_candles(symbol, timeframe, N)
    df_hist.to_csv("data/historical.csv", index=False)

    # 2) Features
    df_feat = compute_features(df_hist)
    df_feat.to_csv("data/data_with_features.csv", index=False)

    # 3) Labels
    df_labels = build_labels(df_feat, horizon=horizon)
    df_labels.to_csv("data/with_labels.csv", index=False)

    # 4) Dataset (ไม่มี trade log ที่แท้จริงใน CI)
    df_dataset = build_dataset(df_labels, trade_log_df=None)
    df_dataset.to_csv("data/dataset.csv", index=False)

    print("Phase 1 pipeline completed.")

if __name__ == "__main__":
    main()
