import pandas as pd

paths = {
    "Model (JSON)":         "models/xgboost_model.json",
    "Classification Report": "models/classification_report.txt",
    "Feature Importance":    "models/feature_importance.csv",
    "Walk-forward Results":  "models/walkforward_results.csv",
}

for name, path in paths.items():
    print(f"--- {name} ({path}) ---")
    try:
        if path.endswith(".csv"):
            df = pd.read_csv(path)
            print("Shape:", df.shape)
            print("Columns:", list(df.columns))
            print(df.head(), "\n")
        elif path.endswith(".txt"):
            with open(path, encoding="utf-8") as f:
                txt = "".join([next(f) for _ in range(5)])
            print("First lines:\n", txt, "\n")
        elif path.endswith(".json"):
            # แค่บอกว่ามีไฟล์ และขนาดโดยประมาณ
            import os
            size = os.path.getsize(path)
            print(f"Exists, size = {size} bytes\n")
    except Exception as e:
        print("Error reading:", e, "\n")
