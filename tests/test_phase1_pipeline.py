import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd

def test_phase1_pipeline_creates_all_files(tmp_path, monkeypatch):
    # 1) เตรียมเบสโฟลเดอร์โปรเจกต์จริง
    base = Path(__file__).resolve().parents[1]

    # 2) คัดลอกโครงโปรเจกต์ไป tmp_path
    #    - run_phase1.py
    shutil.copy(base / "run_phase1.py", tmp_path / "run_phase1.py")
    #    - โฟลเดอร์ src/
    shutil.copytree(base / "src", tmp_path / "src")
    #    - โฟลเดอร์ config/
    shutil.copytree(base / "config", tmp_path / "config")

    # เปลี่ยน working dir ไปที่ tmp_path
    monkeypatch.chdir(tmp_path)

    # 3) รัน pipeline
    result = subprocess.run(
        ["python", "run_phase1.py"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Pipeline failed:\n{result.stdout}\n{result.stderr}"

    # 4) ตรวจไฟล์ที่สร้าง
    expected_files = [
        "data/historical.csv",
        "data/data_with_features.csv",
        "data/with_labels.csv",
        "data/dataset.csv"
    ]
    for fname in expected_files:
        assert (tmp_path / fname).exists(), f"{fname} not created"

    # 5) อ่านเป็น DataFrame ได้
    for fname in expected_files:
        df = pd.read_csv(tmp_path / fname)
        assert not df.empty
