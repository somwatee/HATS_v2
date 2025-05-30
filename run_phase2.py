# run_phase2.py

import yaml
from pathlib import Path

from src.model_trainer import train_model
from src.walkforward    import walk_forward

def main():
    # โหลด config
    cfg = yaml.safe_load(open("config/config.yaml", encoding="utf-8"))

    # Paths
    ds_path       = cfg.get("dataset_path", "data/dataset.csv")
    model_path    = cfg.get("model_path",   "models/xgboost_model.json")
    report_path   = cfg.get("report_path",  "models/classification_report.txt")
    imp_path      = cfg.get("importance_path", "models/feature_importance.csv")
    wf_results    = cfg.get("walkforward_path", "models/walkforward_results.csv")

    # 1) Train model
    train_model(
        ds_path,
        model_path,
        report_path=report_path,
        importance_path=imp_path,
        test_size=cfg.get("test_size", 0.2),
        random_state=cfg.get("random_state", 42),
    )

    # 2) Run walk-forward backtest
    walk_forward(
        ds_path,
        wf_results,
    )

    print("Phase 2 pipeline completed.")

if __name__ == "__main__":
    main()
