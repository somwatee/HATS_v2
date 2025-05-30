# Data Schema ของ Phase 1

## historical.csv
| Column      | Type     | คำอธิบาย                |
| ----------- | -------- | ----------------------- |
| time        | datetime | เวลาแท่งเทียน           |
| open        | float    | ราคาเปิด                |
| high        | float    | ราคาสูงสุด              |
| low         | float    | ราคาต่ำสุด             |
| close       | float    | ราคาปิด                |
| tick_volume | int      | Tick volume             |

## data_with_features.csv
| Column           | Type       | คำอธิบาย                           |
| ---------------- | ---------- | ---------------------------------- |
| time, open, …    | ตาม `historical.csv`                |
| rsi              | float      | Relative Strength Index            |
| mss              | object     | Market Structure Shift (1/-1/0)     |
| fvg_bull         | int        | Fair Value Gap bullish flag        |
| fvg_bear         | int        | Fair Value Gap bearish flag        |
| ema              | float      | Exponential Moving Average         |
| fibo_382, fibo_5, fibo_618 | object | ระดับ Fibonacci (None หรือ float)|
| atr              | float      | Average True Range                 |
| adx              | float      | Average Directional Index          |
| volume_imbalance | float      | (vol_up – vol_down)/volume         |

## with_labels.csv
| Column | Type    | คำอธิบาย                 |
| ------ | ------- | ------------------------ |
| ทุกคอลัมน์จาก `data_with_features.csv` | — | — |
| label  | object  | Buy / Sell / NoTrade     |

## dataset.csv
| Column   | Type    | คำอธิบาย                       |
| -------- | ------- | ------------------------------ |
| ทุกคอลัมน์จาก `with_labels.csv`    | — | — |
| timestamp, side, pnl, … (ถ้ามี log จริง) | ตามชนิด log | ข้อมูล log การเทรดจริง (optional) |

