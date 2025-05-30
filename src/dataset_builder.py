"""
dataset_builder.py

รวม features และ trade log (ถ้ามี) เป็น dataset สำหรับเทรนโมเดล
"""

import pandas as pd


def build_dataset(
    features_df: pd.DataFrame,
    trade_log_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    รวมข้อมูล features กับ log การเทรดจริง (ถ้ามี)

    Args:
        features_df (pd.DataFrame): DataFrame ที่มีฟีเจอร์และคอลัมน์ 'label'
        trade_log_df (pd.DataFrame, optional): DataFrame log การเทรดจริง
            ควรมีคอลัมน์ 'timestamp' ชนิด datetime

    Returns:
        pd.DataFrame: DataFrame รวมข้อมูลทั้งหมด
    """
    df = features_df.copy()

    if trade_log_df is not None:
        trade_log = trade_log_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(
            trade_log["timestamp"]
        ):
            trade_log["timestamp"] = pd.to_datetime(
                trade_log["timestamp"]
            )
        df = df.merge(
            trade_log,
            left_on="time",
            right_on="timestamp",
            how="left",
        )

    return df
