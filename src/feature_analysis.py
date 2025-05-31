import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List, Tuple


def analyze_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    pnl_col: str = "pnl",
) -> Tuple[pd.DataFrame, Figure, pd.DataFrame, Figure]:
    """
    วิเคราะห์ความสัมพันธ์ระหว่างฟีเจอร์กับผลลัพธ์การเทรด
    คืนทั้ง:
      1) stats_pnl_df: DataFrame สองคอลัมน์ ["feature",
      "correlation"] สำหรับ corr(feature, pnl)
      2) fig_pnl: Heatmap Figure ของ correlation matrix (features + pnl)
      3) stats_win_df: DataFrame สองคอลัมน์ ["feature",
      "corr_win_flag"] สำหรับ corr(feature, win_flag)
      4) fig_win: Bar chart Figure ของ correlation with win_flag

    Args:
        df: DataFrame ที่มีคอลัมน์ feature_cols และ pnl_col
        feature_cols: รายชื่อคอลัมน์ฟีเจอร์ (ตัวเลข)
        pnl_col: ชื่อคอลัมน์ผลลัพธ์ PnL (default "pnl")

    Returns:
        Tuple of (stats_pnl_df, fig_pnl, stats_win_df, fig_win)
    """
    # ตรวจว่าคอลัมน์ครบ
    missing = set(feature_cols + [pnl_col]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {missing}")

    # Subset for correlation
    sub = df[feature_cols + [pnl_col]].astype(float)
    corr = sub.corr()

    # stats for pnl
    stats_pnl_df = (
        corr[pnl_col]
        .drop(index=pnl_col)
        .reset_index()
        .rename(columns={"index": "feature", pnl_col: "correlation"})
    )

    # Heatmap for corr matrix
    fig_pnl = plt.figure(figsize=(8, 6))
    ax1 = fig_pnl.add_subplot(1, 1, 1)
    im = ax1.imshow(corr, vmin=-1, vmax=1)
    labels = feature_cols + [pnl_col]
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels)
    ax1.set_title("Correlation Matrix (features vs PnL)")
    fig_pnl.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    fig_pnl.tight_layout()

    # Compute win_flag
    df_w = df[feature_cols + [pnl_col]].copy()
    df_w["win_flag"] = (df_w[pnl_col] > 0).astype(int)

    # Correlation with win_flag
    corr_w = df_w.corr()
    stats_win_df = (
        corr_w["win_flag"]
        .drop(index="win_flag")
        .reset_index()
        .rename(columns={"index": "feature", "win_flag": "corr_win_flag"})
    )

    # Bar chart for win_flag correlation
    fig_win = plt.figure(figsize=(8, 4))
    ax2 = fig_win.add_subplot(1, 1, 1)
    ax2.bar(stats_win_df["feature"], stats_win_df["corr_win_flag"])
    ax2.set_ylabel("Correlation with Win Flag")
    ax2.set_xticks(range(len(stats_win_df["feature"])))
    ax2.set_xticklabels(stats_win_df["feature"], rotation=45, ha="right")
    ax2.set_title("Feature vs Win Rate Correlation")
    fig_win.tight_layout()

    return stats_pnl_df, fig_pnl, stats_win_df, fig_win


# Example usage (not run on import):
# df_trade = pd.read_csv("data/real_trade_log.csv")
# feature_columns = ["rsi", "atr", "adx", "tick_volume", "fibo_382"]
# stats_pnl, fig_pnl, stats_win, fig_win = analyze_features(df_trade, feature_columns)
# fig_pnl.savefig("models/feature_pnl_heatmap.png")
# fig_win.savefig("models/feature_win_corr.png")
