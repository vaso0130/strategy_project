import pandas as pd
from config import STOCK_SYMBOL

def generate_monthly_report(
    daily_df: pd.DataFrame,
    trade_log_df: pd.DataFrame,
    strategy_name: str,
    initial_capital: int = 1_000_000
):
    """
    根據每日資產與交易紀錄，輸出：
    1. 每月績效報表（xlsx）
    2. 每筆交易 CSV（符合 XQ 回測格式）

    Args:
        daily_df: 含 ['date', 'capital'] 的 DataFrame
        trade_log_df: 含 ['entry_date', 'exit_date', 'entry_price', 'exit_price', 'side', 'pnl'] 的 DataFrame
        strategy_name: 策略名稱（如 "LSTMStrategy"）
        initial_capital: 初始資金
    """

    # 自動推導檔名前綴
    symbol_code = STOCK_SYMBOL.split('.')[0]
    output_prefix = f"{symbol_code}_{strategy_name}"

    # 日期標準化
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    if trade_log_df is None or trade_log_df.empty:
        trade_log_df = pd.DataFrame(
            columns=["entry_date", "exit_date", "entry_price", "exit_price", "side", "pnl"]
        )

    trade_log_df["entry_date"] = pd.to_datetime(trade_log_df["entry_date"])
    trade_log_df["exit_date"] = pd.to_datetime(trade_log_df["exit_date"])

    # === 每月資產績效 ===
    daily_df["month"] = daily_df["date"].dt.to_period("M")
    perf = daily_df.groupby("month").agg(
        start_capital=("capital", lambda x: x.iloc[0]),
        end_capital=("capital", lambda x: x.iloc[-1])
    )
    perf["月報酬率"] = (perf["end_capital"] - perf["start_capital"]) / perf["start_capital"]
    perf["累積報酬率"] = (perf["end_capital"] / daily_df["capital"].iloc[0]) - 1
    perf["月份"] = perf.index.astype(str)
    perf["資產金額"] = perf["累積報酬率"] * initial_capital + initial_capital

    # === 每月交易統計 ===
    trade_log_df["是否獲利"] = trade_log_df["pnl"] > 0
    trade_log_df["月份"] = trade_log_df["exit_date"].dt.to_period("M").astype(str)

    stats = trade_log_df.groupby("月份").agg(
        當月交易次數=("pnl", "count"),
        當月勝率=("是否獲利", lambda x: round(100 * x.mean(), 2) if len(x) > 0 else 0)
    ).reset_index()

    # 合併績效與交易統計
    report = pd.merge(perf.reset_index(drop=True), stats, on="月份", how="left")
    report["當月交易次數"] = report["當月交易次數"].fillna(0).astype(int)
    report["當月勝率"] = report["當月勝率"].fillna(0)

    # === 總體統計 ===
    total_trades = len(trade_log_df)
    win_rate = 100 * trade_log_df["是否獲利"].mean() if total_trades > 0 else 0
    final_return = report["累積報酬率"].iloc[-1] if not report.empty else 0
    summary = pd.DataFrame({
        "指標": ["總交易次數", "總勝率 (%)", "總報酬率 (%)"],
        "數值": [total_trades, round(win_rate, 2), round(final_return * 100, 2)]
    })

    # === 輸出 Excel：使用 ExcelWriter 正確寫入多工作表 ===
    excel_path = f"{output_prefix}_monthly_report.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        report.to_excel(writer, sheet_name="每月績效", index=False)
        summary.to_excel(writer, sheet_name="總體統計", index=False)

    # === 輸出 XQ 回測格式 CSV ===
    if not trade_log_df.empty:
        xq_df = trade_log_df.copy()
        xq_df["Date"] = xq_df["entry_date"].dt.strftime('%Y%m%d')
        xq_df["Action"] = xq_df["side"].map({"long": "Buy", "short": "Short"}).fillna("Buy")
        xq_df["Symbol"] = STOCK_SYMBOL
        xq_df["Price"] = xq_df["entry_price"]
        xq_df["Quantity"] = 1000
        xq_csv = xq_df[["Date", "Action", "Symbol", "Price", "Quantity"]]
        xq_csv.to_csv(f"{output_prefix}_trades.csv", index=False, encoding='utf-8-sig')

    return report, summary
