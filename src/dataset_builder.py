# src/dataset_builder.py

"""
dataset_builder.py

โหลด DataFrame ฟีเจอร์ + label → สร้าง dataset พร้อม merge trade log ถ้ามี
"""

import pandas as pd
from pathlib import Path


def build_dataset(
    features_input,
    trade_log_df=None,
    output_path: str = None,
    horizon: int = None,
):
    """
    ถ้า features_input เป็น DataFrame → ใช้โดยตรง, ถ้าเป็น path → อ่าน CSV
    trade_log_df: ถ้าเป็น DataFrame → ใช้โดยตรง, ถ้าเป็น path → อ่าน CSV
    คืน DataFrame result

    Args:
        features_input: DataFrame หรือ path (CSV) ของฟีเจอร์+label
        trade_log_df:    DataFrame หรือ path (CSV) ของ trade log (optional)
        output_path:     ถ้าไม่ None จะเขียนผลเป็น CSV ที่ path นี้
        horizon:         บางทีอาจเอาไว้ส่งต่อ (ไม่บังคับใช้ที่นี่)
    Returns:
        feat_df: DataFrame สุดท้าย (หลัง merge log)
    """
    # 1) อ่าน features
    if isinstance(features_input, pd.DataFrame):
        feat_df = features_input.copy()
    else:
        feat_df = pd.read_csv(features_input)
        if "time" in feat_df.columns and not pd.api.types.is_datetime64_any_dtype(
            feat_df["time"]
        ):
            feat_df["time"] = pd.to_datetime(feat_df["time"])

    # 2) Merge trade log (ถ้ามี)
    if trade_log_df is not None:
        if isinstance(trade_log_df, pd.DataFrame):
            df_trade = trade_log_df.copy()
        else:
            df_trade = pd.read_csv(trade_log_df)
        if (
            "timestamp" in df_trade.columns
            and not pd.api.types.is_datetime64_any_dtype(df_trade["timestamp"])
        ):
            df_trade["timestamp"] = pd.to_datetime(df_trade["timestamp"])

        # เชื่อมทั้งสอง DataFrame บน time<->timestamp
        feat_df = feat_df.merge(
            df_trade, left_on="time", right_on="timestamp", how="left"
        )

    # 3) ถ้ามี output_path ให้เขียนออกเป็น CSV ด้วย
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        feat_df.to_csv(output_path, index=False)

    return feat_df


if __name__ == "__main__":
    import yaml

    # โหลด config เพื่ออ่าน path
    _cfg_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(_cfg_path, "r", encoding="utf-8") as f:
        _cfg = yaml.safe_load(f)

    build_dataset(
        features_input=_cfg["historical_data_path"],
        trade_log_df=_cfg.get("trade_log_path", None),
        output_path=_cfg["dataset_path"],
        horizon=_cfg.get("label_horizon", None),
    )
