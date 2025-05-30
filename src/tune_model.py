import pandas as pd
import yaml
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder

# 1) โหลด config (เพื่ออ่าน paths)
_cfg_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
with open(_cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# 2) โหลด dataset
df = pd.read_csv(cfg["dataset_path"])
X = df.drop(columns=["label"] + (["time"] if "time" in df.columns else []))
y = LabelEncoder().fit_transform(df["label"].astype(str))

# 3) แบ่ง train/test (ลอง stratify ก่อน ถ้า error ให้ non-stratified)
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
except ValueError:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

# 4) กำหนด parameter grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

# 5) สร้าง GridSearchCV
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42,
)
grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=2,
)

# 6) รัน tuning
grid.fit(X_train, y_train)

# 7) บันทึกผล
print("Best hyperparameters:", grid.best_params_)
results_df = pd.DataFrame(grid.cv_results_)
output_path = Path("models/hparam_results.csv")
output_path.parent.mkdir(exist_ok=True)
results_df.to_csv(output_path, index=False)
print(f"Saved CV results to {output_path}")
