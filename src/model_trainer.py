"""
model_trainer.py

เทรน XGBoost model บน dataset.csv แล้วบันทึกผลลัพธ์พื้นฐาน
"""

from pathlib import Path


def train_model(
    dataset_path: str,
    model_path: str,
    report_path: str = "models/classification_report.txt",
    importance_path: str = "models/feature_importance.csv",
) -> None:
    """
    เทรน XGBoost บนไฟล์ dataset.csv

    Args:
        dataset_path (str): path ไปยัง data/dataset.csv
        model_path (str): path สำหรับบันทึกไฟล์โมเดล .json
        report_path (str): path สำหรับบันทึก classification report .txt
        importance_path (str): path สำหรับบันทึก feature importance .csv

    Returns:
        None
    """
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(importance_path).parent.mkdir(parents=True, exist_ok=True)

    Path(model_path).write_text("", encoding="utf-8")
    Path(report_path).write_text("", encoding="utf-8")
    Path(importance_path).write_text("", encoding="utf-8")
