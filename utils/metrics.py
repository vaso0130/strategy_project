import pandas as pd
from config import STOCK_SYMBOL

def _convert_trades_to_paired_format(trade_log_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a list of individual trade actions (from TradeSimulator)
    into a DataFrame of paired (entry/exit) trades.

    Args:
        trade_log_df: DataFrame from TradeSimulator.get_trade_log_df()
                      Expected columns: ['Date', 'Action', 'Symbol', 'Price', 'Quantity', 'PNL', 'TradeID']
                      'Date' is in 'YYYYMMDD' string format.
                      'Action' can be 'Buy', 'Sell', 'Short', 'Cover', or include suffixes like '_SL', '_TP'.

    Returns:
        A DataFrame with columns:
        ['entry_date', 'exit_date', 'entry_price', 'exit_price', 'side', 'pnl', 'entry_trade_id', 'exit_trade_id']
    """
    if trade_log_df.empty:
        return pd.DataFrame(columns=[
            'entry_date', 'exit_date', 'entry_price', 'exit_price', 'side', 'pnl',
            'entry_trade_id', 'exit_trade_id'
        ])

    # Ensure correct dtypes from simulator
    trade_log_df['Date'] = pd.to_datetime(trade_log_df['Date'], format='%Y%m%d')
    trade_log_df['Price'] = pd.to_numeric(trade_log_df['Price'], errors='coerce')
    trade_log_df['Quantity'] = pd.to_numeric(trade_log_df['Quantity'], errors='coerce').fillna(0).astype(int)
    trade_log_df['PNL'] = pd.to_numeric(trade_log_df['PNL'], errors='coerce').fillna(0.0)
    trade_log_df['TradeID'] = pd.to_numeric(trade_log_df['TradeID'], errors='coerce').fillna(0).astype(int)


    paired_trades = []
    active_trades = {}  # Stores entry trades by their original TradeID

    # Sort by date and then by TradeID to process in order
    # This is important if multiple actions can happen on the same day for the same original trade
    # (e.g. partial close, then another partial close - though current simulator logic might not do this for a single TradeID)
    trade_log_df = trade_log_df.sort_values(by=['Date', 'TradeID'])

    for _, row in trade_log_df.iterrows():
        action = row['Action']
        trade_id = row['TradeID'] # This is the ID of the specific action (buy, sell, etc.)

        # Determine the base action and if it's an entry or exit
        is_entry = False
        is_exit = False
        side = None

        if action.startswith('Buy') or action.startswith('OptBuy') or action.startswith('ForcedBuy'):
            is_entry = True
            side = 'long'
        elif action.startswith('Short') or action.startswith('OptShort') or action.startswith('ForcedShort'):
            is_entry = True
            side = 'short'
        elif action.startswith('Sell') or action.startswith('OptSell') or action.startswith('ForcedSell'):
            is_exit = True
            side = 'long' # Exiting a long
        elif action.startswith('Cover') or action.startswith('OptCover') or action.startswith('ForcedCover'):
            is_exit = True
            side = 'short' # Exiting a short

        if is_entry:
            # Store the entry trade. If a TradeID is reused for entry (should not happen with current simulator),
            # this would overwrite. The TradeID from simulator is unique per buy/sell/short/cover action.
            # For pairing, we need to link exits back to their *original* entry.
            # The current trade_log_df doesn't directly link exits to the TradeID of the entry.
            # It uses self.current_trade_id or self.current_forced_trade_id internally in simulator.
            # The PNL is calculated at the point of exit in the simulator.

            # This simplified pairing assumes that any 'Buy' or 'Short' opens a new position,
            # and any 'Sell' or 'Cover' closes the *currently open* position of that type.
            # It relies on the PNL already calculated by the simulator for that closing action.

            # We need a way to identify the *original* entry that an exit corresponds to.
            # The simulator's `current_trade_id` (for regular) and `current_forced_trade_id` (for forced)
            # are set on entry and cleared on exit. The `TradeID` in the log is unique per action.

            # Let's assume for now that the PNL recorded with an exit action ('Sell', 'Cover')
            # is the PNL for the entire position that was just closed.
            # We will try to find a preceding 'Buy' or 'Short' that has not yet been closed.

            # A better approach for pairing would be if the simulator's log explicitly linked
            # an exit TradeID to an entry TradeID. Since it doesn't, we make assumptions.

            # If this is an entry, we record it.
            # We need to handle the case where a new entry happens before a previous one is closed
            # (e.g. averaging in, though the current simulator might not fully support complex averaging PNL tracking for this report)

            # For simplicity in this converter, we'll assume a LIFO-like or "current active trade" model.
            # The `active_trades` will hold the *most recent* unclosed entry for 'long' or 'short'.
            if side:
                active_trades[side] = {
                    'entry_date': row['Date'],
                    'entry_price': row['Price'],
                    'entry_quantity': row['Quantity'],
                    'entry_trade_id': trade_id,
                    'side': side
                }

        elif is_exit:
            if side and side in active_trades:
                entry_details = active_trades.pop(side) # Remove the active trade as it's now closed
                
                # The PNL in the row is for this specific exit action.
                # If the simulator calculates PNL for the full position closure, this is correct.
                paired_trades.append({
                    'entry_date': entry_details['entry_date'],
                    'exit_date': row['Date'],
                    'entry_price': entry_details['entry_price'],
                    'exit_price': row['Price'],
                    'side': side,
                    'pnl': row['PNL'], # PNL from the simulator's exit record
                    'entry_trade_id': entry_details['entry_trade_id'],
                    'exit_trade_id': trade_id
                })
            # else:
                # This is an exit without a corresponding active entry in our simple model.
                # This might happen if data is partial or logic is more complex (e.g. multiple open positions of same side).
                # For now, we ignore such exits if no matching entry was found.
                # print(f"Warning: Exit trade {trade_id} on {row['Date']} for side {side} has no matching entry in active_trades.")


    return pd.DataFrame(paired_trades)


def generate_monthly_report(
    daily_df: pd.DataFrame,
    raw_trade_log_df: pd.DataFrame, # Changed parameter name
    strategy_name: str,
    initial_capital: int = 1_000_000
):
    """
    根據每日資產與交易紀錄，輸出：
    1. 每月績效報表（xlsx）
    2. 每筆交易 CSV（符合 XQ 回測格式）

    Args:
        daily_df: 含 ['date', 'capital'] 的 DataFrame
        raw_trade_log_df: Simulator's trade log, columns like ['Date', 'Action', 'PNL', 'TradeID', etc.]
        strategy_name: 策略名稱（如 "LSTMStrategy"）
        initial_capital: 初始資金
    """
    # Convert raw trade log to paired format
    paired_trade_log_df = _convert_trades_to_paired_format(raw_trade_log_df)

    # 自動推導檔名前綴
    symbol_code = STOCK_SYMBOL.split('.')[0] if isinstance(STOCK_SYMBOL, str) else "STOCK"
    output_prefix = f"{symbol_code}_{strategy_name}"

    # 日期標準化
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    
    # Use the converted paired_trade_log_df
    trade_log_df_for_report = paired_trade_log_df.copy() # Use the new name for clarity

    # Ensure trade_log_df_for_report has the necessary columns even if empty (after conversion)
    expected_paired_cols = ["entry_date", "exit_date", "entry_price", "exit_price", "side", "pnl"]
    if trade_log_df_for_report.empty:
        trade_log_df_for_report = pd.DataFrame(columns=expected_paired_cols)
    else:
        for col in expected_paired_cols:
            if col not in trade_log_df_for_report.columns:
                # This shouldn't happen if _convert_trades_to_paired_format is correct
                trade_log_df_for_report[col] = None

    if not trade_log_df_for_report.empty:
        trade_log_df_for_report["entry_date"] = pd.to_datetime(trade_log_df_for_report["entry_date"], errors='coerce')
        trade_log_df_for_report["exit_date"] = pd.to_datetime(trade_log_df_for_report["exit_date"], errors='coerce')


    # === 每月資產績效 ===
    daily_df["month"] = daily_df["date"].dt.to_period("M")
    daily_df["capital"] = pd.to_numeric(daily_df["capital"], errors='coerce')
    
    if daily_df.empty or daily_df['capital'].isna().all():
        empty_report_cols = ["月份", "start_capital", "end_capital", "月報酬率", "累積報酬率", "資產金額", "當月交易次數", "當月勝率"]
        empty_summary_cols = ["指標", "數值"]
        return pd.DataFrame(columns=empty_report_cols), pd.DataFrame(columns=empty_summary_cols)

    perf = daily_df.groupby("month").agg(
        start_capital=("capital", lambda x: x.dropna().iloc[0] if not x.dropna().empty else 0),
        end_capital=("capital", lambda x: x.dropna().iloc[-1] if not x.dropna().empty else 0)
    )
    perf["月報酬率"] = (perf["end_capital"] - perf["start_capital"]) / perf["start_capital"].replace(0, pd.NA) # Avoid division by zero, result in NA
    
    first_valid_capital = daily_df["capital"].dropna().iloc[0] if not daily_df["capital"].dropna().empty else pd.NA
    if pd.notna(first_valid_capital) and first_valid_capital != 0:
        perf["累積報酬率"] = (perf["end_capital"] / first_valid_capital) - 1
    else:
        perf["累積報酬率"] = pd.NA 

    perf["月份"] = perf.index.astype(str)
    perf["資產金額"] = (1 + perf["累積報酬率"].fillna(0)) * initial_capital # Use fillna(0) for calculation if NA

    # === 每月交易統計 ===
    # Ensure 'pnl' is numeric and 'exit_date' is datetime for trade_log_df
    if not trade_log_df_for_report.empty and 'pnl' in trade_log_df_for_report.columns:
        trade_log_df_for_report["pnl"] = pd.to_numeric(trade_log_df_for_report["pnl"], errors='coerce')
        trade_log_df_for_report["是否獲利"] = trade_log_df_for_report["pnl"] > 0
        if 'exit_date' in trade_log_df_for_report.columns and pd.api.types.is_datetime64_any_dtype(trade_log_df_for_report['exit_date']):
             trade_log_df_for_report["月份"] = trade_log_df_for_report["exit_date"].dt.to_period("M").astype(str)
        else:
            trade_log_df_for_report["月份"] = pd.NaT # Assign NaT if exit_date is not suitable

        stats = trade_log_df_for_report.groupby("月份").agg(
            當月交易次數=("pnl", "count"), # Counts non-NA pnl values
            當月勝率=("是否獲利", lambda x: round(100 * x.mean(), 2) if len(x) > 0 and x.count() > 0 else 0) # Ensure there are valid boolean values to mean
        ).reset_index()
    else:
        stats = pd.DataFrame(columns=["月份", "當月交易次數", "當月勝率"])


    # 合併績效與交易統計
    if not perf.empty:
        report = pd.merge(perf.reset_index(drop=True), stats, on="月份", how="left")
    else: # If perf is empty, report should also be empty with correct columns
        report = pd.DataFrame(columns=["月份", "start_capital", "end_capital", "月報酬率", "累積報酬率", "資產金額", "當月交易次數", "當月勝率"])

    report["當月交易次數"] = report["當月交易次數"].fillna(0).astype(int)
    report["當月勝率"] = report["當月勝率"].fillna(0).astype(float)
    report["月報酬率"] = report["月報酬率"].fillna(0).astype(float) # Fill NA returns with 0
    report["累積報酬率"] = report["累積報酬率"].fillna(0).astype(float)


    # === 總體統計 ===
    total_trades = 0
    win_rate = 0.0
    if not trade_log_df_for_report.empty and 'pnl' in trade_log_df_for_report.columns and not trade_log_df_for_report['pnl'].isna().all():
        total_trades = trade_log_df_for_report['pnl'].count() # Count non-NA trades
        if total_trades > 0 and '是否獲利' in trade_log_df_for_report.columns:
            win_rate = 100 * trade_log_df_for_report["是否獲利"].mean() if trade_log_df_for_report["是否獲利"].count() > 0 else 0.0

    final_return = report["累積報酬率"].iloc[-1] if not report.empty and not report["累積報酬率"].isna().all() else 0.0
    summary = pd.DataFrame({
        "指標": ["總交易次數", "總勝率 (%)", "總報酬率 (%)"],
        "數值": [total_trades, round(win_rate, 2), round(final_return * 100, 2)]
    })

    # === 輸出 Excel：使用 ExcelWriter 正確寫入多工作表 ===
    excel_path = f"{output_prefix}_monthly_report.xlsx"
    # --- 新增：寫入前清理 DataFrame 型別與 NaN ---
    def _clean_df_for_excel(df):
        for col in df.columns:
            if df[col].dtype == 'O':
                df[col] = df[col].astype(str)
            if df[col].isnull().any():
                if df[col].dtype.kind in 'fi':
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna('')
        return df
    report = _clean_df_for_excel(report)
    summary = _clean_df_for_excel(summary)
    # --- End 清理 ---
    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            report.to_excel(writer, sheet_name="每月績效", index=False)
            summary.to_excel(writer, sheet_name="總體統計", index=False)
    except Exception as e:
        print(f"Error writing Excel file {excel_path}: {e}")
        print("report dtypes:", report.dtypes)
        print("summary dtypes:", summary.dtypes)
        print("report head:\n", report.head())
        print("summary head:\n", summary.head())


    # === 輸出 XQ 回測格式 CSV ===
    # This part used 'entry_date', 'side', 'entry_price' from the paired log.
    if not trade_log_df_for_report.empty and all(col in trade_log_df_for_report.columns for col in ["entry_date", "side", "entry_price"]):
        xq_df = trade_log_df_for_report.dropna(subset=["entry_date", "side", "entry_price"]).copy()
        if not xq_df.empty:
            xq_df["Date"] = xq_df["entry_date"].dt.strftime('%Y%m%d')
            xq_df["Action"] = xq_df["side"].map({"long": "Buy", "short": "Short"}).fillna("Buy")
            xq_df["Symbol"] = symbol_code
            xq_df["Price"] = xq_df["entry_price"]
            xq_df["Quantity"] = 1000 
            xq_csv = xq_df[["Date", "Action", "Symbol", "Price", "Quantity"]]
            try:
                xq_csv.to_csv(f"{output_prefix}_trades.csv", index=False, encoding='utf-8-sig')
            except Exception as e:
                print(f"Error writing CSV file {output_prefix}_trades.csv: {e}")

    return report, summary

def calculate_performance_metrics(daily_df: pd.DataFrame | None, 
                                  trade_log_df: pd.DataFrame | None, 
                                  initial_capital_for_return_calc: float | None = None):
    """
    Calculates overall performance metrics.

    Args:
        daily_df: DataFrame with ['date', 'capital'], sorted by date. Can be None.
        trade_log_df: DataFrame with ['pnl']. Can be None.
        initial_capital_for_return_calc: Optional. If provided, used as the base for total return.
                                         If None, the first value in daily_df['capital'] is used.

    Returns:
        A dictionary with performance metrics:
        - total_trades: int
        - win_rate_percentage: float
        - total_return_percentage: float
    """
    if trade_log_df is None:
        trade_log_df = pd.DataFrame(columns=['pnl'])
    
    # Ensure 'pnl' column exists, if not, add it with NaNs
    if 'pnl' not in trade_log_df.columns:
        trade_log_df['pnl'] = pd.NA

    # Convert 'pnl' to numeric, coercing errors. This handles strings or other types.
    trade_log_df['pnl'] = pd.to_numeric(trade_log_df['pnl'], errors='coerce')
    
    # Calculate total_trades based on non-NaN 'pnl' entries
    total_trades = trade_log_df['pnl'].count()

    win_rate_percentage = 0.0
    if total_trades > 0:
        # 'pnl' is already numeric here
        wins = trade_log_df[trade_log_df["pnl"] > 0]
        win_rate_percentage = round(100 * (len(wins) / total_trades), 2)
    
    total_return_percentage = 0.0
    if daily_df is not None and not daily_df.empty and 'capital' in daily_df.columns:
        # Ensure 'capital' column is numeric and handle potential NaNs
        daily_df['capital'] = pd.to_numeric(daily_df['capital'], errors='coerce')
        valid_capital_df = daily_df.dropna(subset=['capital'])

        if not valid_capital_df.empty:
            base_capital = initial_capital_for_return_calc
            if base_capital is None:
                base_capital = valid_capital_df["capital"].iloc[0]

            final_capital = valid_capital_df["capital"].iloc[-1]

            if base_capital is not None and base_capital != 0: # Check base_capital is not zero
                total_return_percentage = round(100 * ((final_capital / base_capital) - 1), 2)
            elif base_capital == 0 and final_capital > 0:
                total_return_percentage = float('inf') 
            # If base_capital is 0 and final_capital is 0, or base_capital is None, return remains 0.0
        
    return {
        "total_trades": total_trades,
        "win_rate_percentage": win_rate_percentage,
        "total_return_percentage": total_return_percentage,
    }
