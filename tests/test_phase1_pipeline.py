# tests/test_phase1_pipeline.py

import os
import pandas as pd
import subprocess
import pytest

@pytest.fixture(autouse=True)
def setup_ci_env(tmp_path, monkeypatch):
    # ย้าย working directory ไปที่ tmp_path สำหรับ CI
    monkeypatch.chdir(tmp_path)
    # สร้างโฟลเดอร์ config และเขียน config.yaml จำลอง
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text(
        "\n".join([
            "symbol: 'XAUUSD'",
            "timeframe: 'M1'",
            "label_horizon: 3",
            "fetch_candles_n: 5",
            "mt5:",
            "  terminal_path: ''",
            "  server: ''",
            "  login: 0",
            "  password: ''",
            "  timeout: 1000",
            "telegram:",
            "  bot_token: ''",
            "  chat_id: ''",
            "online_learning:",
            "  enabled: false"
        ]),
        encoding="utf-8"
    )
    # ทำให้โฟลเดอร์ src และ run_phase1.py ถูกมองเห็นจาก tmp_path
    # โดย copy ไฟล์จาก root project ไปที่ tmp_path/src และ tmp_path/run_phase1.py
    # ใน CI runner ของ GitHub Actions ไม่จำเป็นต้อง copy เพราะ checkout จะวางโค้ดลง workspace ตรง tmp_path อยู่แล้ว

def test_phase1_pipeline_creates_all_files():
    # รัน script
    result = subprocess.run(
        ["python", "run_phase1.py"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Pipeline failed:\n{result.stdout}\n{result.stderr}"

    # ตรวจว่าไฟล์ทุกตัวถูกสร้างใน data/
    expected = [
        "data/historical.csv",
        "data/data_with_features.csv",
        "data/with_labels.csv",
        "data/dataset.csv",
    ]
    for fname in expected:
        assert os.path.exists(fname), f"{fname} not created"
        # ตรวจว่าอ่านได้ไม่ว่าง
        df = pd.read_csv(fname)
        assert not df.empty
