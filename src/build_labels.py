import pandas as pd
import yaml
from pathlib import Path
from typing import Union


# โหลด config
_cfg_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
with open(_cfg_path, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

# ค่าพารามิเตอร์ default จาก config
DEFAULT_HORIZON = _cfg.get("label_horizon", 5)
DEFAULT_K_ATR = _cfg.get("label_atr_multiplier", 0.5)


def build_labels(
    data: Union[pd.DataFrame, str],
    horizon: int = None,
    k_atr: float = None
) -> pd.DataFrame:
    """
    สร้างคอลัมน์ 'label' ใน DataFrame โดย:
    - ถ้า data เป็น DataFrame อยู่แล้ว ให้ใช้ตรง ๆ
    - ถ้า data เป็น path (str) ให้อ่าน CSV เข้ามา

    Args:
        data (pd.DataFrame or str): 
            - ถ้าเป็น DataFrame: ใช้ DataFrame นั้น
            - ถ้าเป็น str: มองเป็น path ไปยังไฟล์ CSV แล้ว pd.read_csv
        horizon (int, optional): lookahead horizon (จำนวนแท่งถัดไป) 
            ถ้าไม่ระบุ จะใช้ DEFAULT_HORIZON จาก config
        k_atr (float, optional): สัดส่วน ATR สำหรับ threshold 
            ถ้าไม่ระบุ จะใช้ DEFAULT_K_ATR จาก config

    Returns:
        pd.DataFrame: DataFrame เดิม พร้อมคอลัมน์ 'label'
                      (แต่ยังไม่ได้บันทึกไฟล์)
    """
    # 1) เตรียม DataFrame
    if isinstance(data, pd.DataFrame):
        df = data.copy().reset_index(drop=True)
    else:
        df = pd.read_csv(str(data))

    n = len(df)
    H = horizon if horizon is not None else DEFAULT_HORIZON
    k = k_atr if k_atr is not None else DEFAULT_K_ATR

    # ต้องมีคอลัมน์พื้นฐาน: ['open','high','low','atr', 'ema50_h4','ema200_h4',
    # 'rsi_h4','vwap','bb_upper','bb_lower','atr_ma','mss_bullish','mss_bearish',
    # 'fvg_bullish','fvg_bearish','fib_in_zone','rsi','adx','vol_imbalance']
    # Tests มักจะให้ minimal columns ['high','low','close'], ดังนั้นเงื่อนไขขั้นลึกต้องตรวจก่อนใช้งาน
    labels = ["NoTrade"] * n

    for t in range(n):
        # ถ้าไม่มีแท่งข้างหน้าเพียงพอ
        if t + H >= n:
            labels[t] = "NoTrade"
            continue

        price_open = df.at[t, "open"] if "open" in df.columns else df.at[t, "close"]
        atr_t = df.at[t, "atr"] if "atr" in df.columns else 0.0
        threshold = k * atr_t

        future_high = df["high"].iloc[t + 1 : t + 1 + H].max()
        future_low = df["low"].iloc[t + 1 : t + 1 + H].min()

        # 1) Base label จาก ATR-break (หรือจาก high/low ถ้าไม่มี ATR)
        if atr_t > 0:
            is_buy_base = (future_high - price_open) >= threshold
            is_sell_base = (price_open - future_low) >= threshold
        else:
            # ถ้าไม่มี ATR ก็ใช้ราคาเปรียบเทียบแบบง่าย
            is_buy_base = future_high > price_open
            is_sell_base = future_low < price_open

        if is_buy_base:
            base_label = "Buy"
        elif is_sell_base:
            base_label = "Sell"
        else:
            labels[t] = "NoTrade"
            continue

        # 2) HTF + VWAP bias (ถ้ามีคอลัมน์)
        if base_label == "Buy":
            # ตรวจ EMA50_H4 vs EMA200_H4 + RSI_H4 + VWAP bias
            if all(col in df.columns for col in ["ema50_h4", "ema200_h4", "rsi_h4", "vwap"]):
                ema50_h4 = df.at[t, "ema50_h4"]
                ema200_h4 = df.at[t, "ema200_h4"]
                rsi_h4 = df.at[t, "rsi_h4"]
                vwap = df.at[t, "vwap"]
                tol = 0.1 * atr_t

                cond_htf = (ema50_h4 > ema200_h4) and (rsi_h4 > 50)
                cond_vwap = (price_open > vwap + tol)
                if not (cond_htf and cond_vwap):
                    labels[t] = "NoTrade"
                    continue
            # ถ้าไม่มีคอลัมน์ HTF/VWAP ให้ข้ามกรองส่วนนี้

        else:  # base_label == "Sell"
            if all(col in df.columns for col in ["ema50_h4", "ema200_h4", "rsi_h4", "vwap"]):
                ema50_h4 = df.at[t, "ema50_h4"]
                ema200_h4 = df.at[t, "ema200_h4"]
                rsi_h4 = df.at[t, "rsi_h4"]
                vwap = df.at[t, "vwap"]
                tol = 0.1 * atr_t

                cond_htf = (ema50_h4 < ema200_h4) and (rsi_h4 < 50)
                cond_vwap = (price_open < vwap - tol)
                if not (cond_htf and cond_vwap):
                    labels[t] = "NoTrade"
                    continue

        # 3) Bollinger Bands + ATR_MA (ถ้ามีคอลัมน์)
        if all(col in df.columns for col in ["bb_upper", "bb_lower", "atr_ma"]):
            bb_upper = df.at[t, "bb_upper"]
            bb_lower = df.at[t, "bb_lower"]
            atr_ma = df.at[t, "atr_ma"]

            if base_label == "Buy":
                if not (
                    (price_open <= bb_lower and atr_t < atr_ma)  # reversal ที่ BB_lower
                    or (price_open >= bb_upper and atr_t > atr_ma)  # breakout ที่ BB_upper
                ):
                    labels[t] = "NoTrade"
                    continue
            else:  # Sell
                if not (
                    (price_open >= bb_upper and atr_t < atr_ma)
                    or (price_open <= bb_lower and atr_t > atr_ma)
                ):
                    labels[t] = "NoTrade"
                    continue

        # 4) MSS/FVG + Fibonacci + RSI/ADX + Volume Imbalance (ถ้ามีคอลัมน์)
        if all(
            col in df.columns
            for col in [
                "mss_bullish",
                "mss_bearish",
                "fvg_bullish",
                "fvg_bearish",
                "fib_in_zone",
                "rsi",
                "adx",
                "vol_imbalance",
            ]
        ):
            mss_b = df.at[t, "mss_bullish"]
            mss_br = df.at[t, "mss_bearish"]
            fvg_b = df.at[t, "fvg_bullish"]
            fvg_br = df.at[t, "fvg_bearish"]
            fib_in_zone = df.at[t, "fib_in_zone"]
            rsi_t = df.at[t, "rsi"]
            adx_t = df.at[t, "adx"]
            vol_imb = df.at[t, "vol_imbalance"]

            if base_label == "Buy":
                cond1 = bool(mss_b) or (bool(fvg_b) and bool(fib_in_zone))
                cond2 = (rsi_t < 30 and adx_t > 25) or (vol_imb > 0.2)
                if not (cond1 and cond2):
                    labels[t] = "NoTrade"
                    continue

            else:  # Sell
                cond1 = bool(mss_br) or (bool(fvg_br) and bool(fib_in_zone))
                cond2 = (rsi_t > 70 and adx_t > 25) or (vol_imb < -0.2)
                if not (cond1 and cond2):
                    labels[t] = "NoTrade"
                    continue

        # หากผ่านทุกเงื่อนไข จึงตั้ง label ตาม base_label
        labels[t] = base_label

    df["label"] = labels
    return df


def save_labeled_csv(
    input_path: str,
    output_path: str = "data/with_labels.csv",
    horizon: int = None,
    k_atr: float = None,
):
    """
    อ่านไฟล์ CSV จาก input_path → สร้าง label → เขียนออกไปยัง output_path
    """
    df_in = pd.read_csv(input_path)
    df_out = build_labels(df_in, horizon=horizon, k_atr=k_atr)
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    df_out.to_csv(output_path, index=False)


if __name__ == "__main__":
    # เมื่อรันตรง ๆ: ใช้ไฟล์ data_with_features.csv → ออก data/with_labels.csv
    save_labeled_csv("data/data_with_features.csv", "data/with_labels.csv")
