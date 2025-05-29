"""
build_labels.py

สร้าง label Buy/Sell/NoTrade จากราคาล่วงหน้า horizon N แท่ง
"""

import pandas as pd
import yaml

from pathlib import Path

# โหลด config.yaml เพื่ออ่าน horizon ถ้ามี (กำหนด default ไว้เป็น 5)
_cfg_path = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "config.yaml"
)
with open(_cfg_path, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)
_horizon = _cfg.get("label_horizon", 5)


def build_labels(df: pd.DataFrame, horizon: int = None) -> pd.DataFrame:
    """
    คำนวณ label จาก future price horizon

    Args:
        df (pd.DataFrame): ข้อมูล features + ราคาปัจจุบัน
            ต้องมีคอลัมน์ ['high','low','close']
        horizon (int, optional): จำนวนแท่งข้างหน้า
            ถ้าไม่กำหนดจะอ่านจาก config (default 5)

    Returns:
        pd.DataFrame: เพิ่มคอลัมน์ 'label'
            (ค่าหนึ่งใน ['Buy','Sell','NoTrade'])
    """
    df = df.copy().reset_index(drop=True)
    h = horizon if horizon is not None else _horizon
    n = len(df)

    # เตรียมคอลัมน์ default
    labels = ["NoTrade"] * n

    # สำหรับแต่ละแท่ง หา High/Low ใน horizon ถัดไป
    future_high = (
        df["high"]
        .shift(-1)
        .rolling(window=h, min_periods=1)
        .max()
    )
    future_low = (
        df["low"]
        .shift(-1)
        .rolling(window=h, min_periods=1)
        .min()
    )
    current_close = df["close"]

    for i in range(n):
        if i + 1 >= n:
            # ไม่มีข้อมูลข้างหน้า
            labels[i] = "NoTrade"
            continue

        fh = future_high.iloc[i]
        fl = future_low.iloc[i]
        cc = current_close.iloc[i]

        # ถ้า future_high > current_close → Buy
        if fh > cc:
            labels[i] = "Buy"
        # ถ้า future_low < current_close → Sell
        elif fl < cc:
            labels[i] = "Sell"
        else:
            labels[i] = "NoTrade"

    df["label"] = pd.Series(labels, dtype=object)
    return df
