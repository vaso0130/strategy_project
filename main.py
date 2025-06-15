import pandas as pd
import inspect
from config import * # Make sure this imports all new config variables
from data_loader import load_price_data
from lstm_model import LSTMPredictor
from market_regime import calculate_market_regime
from llm_decision_engine import select_strategy_and_params
from trade_simulator import TradeSimulator
from utils.metrics import generate_monthly_report, calculate_performance_metrics
import strategies # Import the strategies module
from strategies import TrendStrategy, RangeStrategy, BreakoutStrategy, VolumePriceStrategy
from optimizer import StrategyOptimizer
from datetime import timedelta, datetime # Ensure datetime is imported
import logging # Added import for logging
import os # For file operations if needed for trade log

# --- DataFetcher class ---
# This class is used to fetch and preprocess data for the strategies and LSTM
# It wraps the full_df DataFrame and provides methods to get data for specific dates or ranges
class DataFetcher:
    def __init__(self, df_full):
        self.df_full = df_full.copy()
        self.df_full['date'] = pd.to_datetime(self.df_full['date']).dt.date
        self.df_full.set_index('date', inplace=True)
        # Ensure necessary columns like 'MA', 'RSI' are pre-calculated if needed by strategies directly from this df
        if 'Close' in self.df_full.columns:
            if 'MA' not in self.df_full.columns:
                self.df_full['MA'] = self.df_full['Close'].rolling(window=20).mean()
        else:
            print("Warning: 'Close' column not in DataFetcher's DataFrame. Some features might not be calculated.")

    def get_day_data(self, date_str):
        # date_str should be 'YYYY-MM-DD'
        dt = pd.to_datetime(date_str).date()
        if dt in self.df_full.index:
            return self.df_full.loc[[dt]].reset_index() # Return as DataFrame
        return pd.DataFrame()

    def get_historical_data(self, end_date_dt_param, num_days):
        # end_date_dt_param can be a date object or Timestamp (exclusive)
        # Convert Timestamp to date for comparison with index
        if isinstance(end_date_dt_param, pd.Timestamp):
            effective_end_date = end_date_dt_param.date()
        else: # Assuming it's already a date object
            effective_end_date = end_date_dt_param

        # Returns data up to, but not including, effective_end_date
        mask = self.df_full.index < effective_end_date
        historical = self.df_full[mask].tail(num_days)
        return historical.reset_index()

    def get_data_for_strategy(self, end_date_dt_param, lookback_period):
        # end_date_dt_param is the date *before* the decision day, or the last day of historical data needed.
        # Convert Timestamp to date for comparison with index
        if isinstance(end_date_dt_param, pd.Timestamp):
            effective_end_date = end_date_dt_param.date()
        else: # Assuming it's already a date object
            effective_end_date = end_date_dt_param
            
        # Strategies usually need data up to and including this effective_end_date.
        mask = self.df_full.index <= effective_end_date
        data = self.df_full[mask].tail(lookback_period)
        return data.reset_index() # Return as DataFrame

    def get_recent_price_data_str(self, end_date_dt, num_days):
        # Data up to and including end_date_dt for LLM context
        data = self.get_data_for_strategy(end_date_dt, num_days)
        if not data.empty:
            cols_for_llm = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] if col in data.columns]
            return data[cols_for_llm].to_string()
        return "No recent price data available."

    def get_market_regime(self, end_date_dt, window=30):
        # Simplified regime detection, replace with actual `calculate_market_regime` logic
        data_for_regime = self.get_data_for_strategy(end_date_dt, window + 5) # Get a bit more for stability
        if len(data_for_regime) >= window:
            # return calculate_market_regime(data_for_regime) # Assuming this function exists and takes a DataFrame
            return "trend" # Placeholder
        return "unknown"

def calculate_yearly_pnl(trade_log_df, year):
    if trade_log_df.empty or 'Date' not in trade_log_df.columns or 'PNL' not in trade_log_df.columns:
        return 0.0
    
    # Ensure 'Date' is datetime like
    try:
        # Attempt to convert, handling potential mixed formats or errors
        log_df_copy = trade_log_df.copy()
        log_df_copy['Date'] = pd.to_datetime(log_df_copy['Date'], errors='coerce')
        # Drop rows where date conversion failed
        log_df_copy.dropna(subset=['Date'], inplace=True)
        
        yearly_trades = log_df_copy[log_df_copy['Date'].dt.year == year]
        return yearly_trades['PNL'].sum()
    except Exception as e:
        print(f"Error in calculate_yearly_pnl during date conversion or sum: {e}")
        return 0.0

def main(): # Consolidate all execution logic into this main function
    # === 載入資料 ===
    required_trading_days = LSTM_TRAIN_WINDOW + LSTM_LOOKBACK_DAYS
    estimated_calendar_days = int(required_trading_days * 1.5) + 5
    real_start = (
        pd.to_datetime(START_DATE)
        - timedelta(days=estimated_calendar_days)
    ).strftime("%Y-%m-%d")
    print(f"[資訊] `real_start` 日期計算為: {real_start} (需要 {required_trading_days} 交易日, 估算 {estimated_calendar_days} 日曆日)")
    full_df = load_price_data(symbol=STOCK_SYMBOL, start_date=real_start, end_date=END_DATE)

    # 將欄位名稱重命名為大寫版本，應在 full_df 載入後立即進行
    # 以確保所有後續衍生的 DataFrame (包括 lstm_initial_train_df 和 df) 都使用一致的欄位名
    print("[DEBUG] full_df.columns before rename:", full_df.columns)
    print("[DEBUG] full_df.head() before rename:\n", full_df.head())
    full_df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close", # MA 和 RSI 將使用此欄位
        "adjclose": "Close", # Yahoo 有時會是 adjclose
        "Adj Close": "Close" # 若有空格也加上
    }, inplace=True)
    print("[DEBUG] full_df.columns after rename:", full_df.columns)
    print("[DEBUG] full_df.head() after rename:\n", full_df.head())
    # 驗證 'Close' 是否存在
    if 'Close' not in full_df.columns:
        print("[警告] full_df 在重命名後仍然缺少 'Close' 欄位。請檢查 load_price_data 的回傳。")
        # 可以考慮引發錯誤或採取其他處理

    # --- 資料欄位重複全面清理 ---
    # 載入資料、重命名欄位後，立即去除重複欄位，確保 downstream 不會出錯
    full_df = full_df.loc[:, ~full_df.columns.duplicated()]
    print(f"[DEBUG] full_df 欄位: {full_df.columns.tolist()}")

    # === 初始化元件 ===
    # 使用 config.py 中的 LSTM_EPOCHS 和 LSTM_LEARNING_RATE
    lstm = LSTMPredictor(
        lookback_days=LSTM_LOOKBACK_DAYS,
        predict_days=LSTM_PREDICT_DAYS,
        epochs=LSTM_EPOCHS,
        lr=LSTM_LEARNING_RATE
    )

    # === LSTM 預先訓練 ===
    # 確保 'date' 欄位是 datetime 物件以便比較
    full_df['date'] = pd.to_datetime(full_df['date']).dt.date

    # 分割訓練資料和回測資料
    backtest_start_date_dt = pd.to_datetime(START_DATE).date()

    # 用於 LSTM 初始訓練的數據 (從 real_start 到 START_DATE 前一天)
    lstm_initial_train_df = full_df[full_df['date'] < backtest_start_date_dt].copy()

    initial_train_successful = False
    initial_train_end_date = None

    if len(lstm_initial_train_df) >= LSTM_TRAIN_WINDOW + LSTM_LOOKBACK_DAYS:
        print(f"[資訊] 開始 LSTM 初始訓練，使用 {len(lstm_initial_train_df)} 筆從 {lstm_initial_train_df['date'].min()} 到 {lstm_initial_train_df['date'].max()} 的資料...")
        # LSTMPredictor.train 期望的 DataFrame 包含 'date' 和 'Close'
        lstm.train(lstm_initial_train_df[['date', 'Close']]) # lstm.is_trained 會在此方法內部設定
        if lstm.is_trained:
            print("[資訊] LSTM 初始訓練完成。")
            initial_train_successful = True
            initial_train_end_date = lstm_initial_train_df['date'].max()
        else:
            print("[警告] LSTM 初始訓練未成功（可能因內部數據長度不足等原因）。lstm.is_trained 為 False。")
    else:
        print(f"[警告] LSTM 初始訓練資料不足 ({len(lstm_initial_train_df)} 筆)，至少需要 {LSTM_TRAIN_WINDOW + LSTM_LOOKBACK_DAYS} 筆。將在回測過程中逐步訓練。")

    # 用於回測的數據 (從 START_DATE 到 END_DATE)
    df = full_df[full_df['date'] >= backtest_start_date_dt].copy().reset_index(drop=True)
    df = df.loc[:, ~df.columns.duplicated()]
    print(f"[DEBUG] df 欄位: {df.columns.tolist()}")

    if df.empty:
        raise ValueError(f"回測期間 ({START_DATE} 至 {END_DATE}) 無可用數據，請檢查日期設定或資料來源。")
    print(f"[資訊] 回測資料期間：{df['date'].min()} 至 {df['date'].max()}，共 {len(df)} 筆。")

    # === 新增：將 LSTM 預測結果寫入 df 供所有策略參考 ===
    if initial_train_successful:
        lookback = lstm.lookback if hasattr(lstm, 'lookback') else LSTM_LOOKBACK_DAYS
        if len(df) >= lookback:
            # 只保留一個 'Close' 欄位，避免重複
            lstm_input = df.loc[:, ~df.columns.duplicated()][['Close']].tail(lookback)
            print(f"[DEBUG] LSTM predict input shape: {lstm_input.shape}, columns: {lstm_input.columns}")
            lstm_pred_signal = lstm.predict(lstm_input)
            print(f"[DEBUG] LSTM predict output: {lstm_pred_signal}")
            df['LSTM_PREDICTION'] = lstm_pred_signal  # 全部填同一個信號
        else:
            print(f"[警告] df 資料長度 {len(df)} 小於 lookback {lookback}，無法進行 LSTM 預測。")
            df['LSTM_PREDICTION'] = 0
    else:
        df['LSTM_PREDICTION'] = 0  # 若無預測則預設為 0

    print(f"[DEBUG] df['LSTM_PREDICTION'] 前 5 筆: {df['LSTM_PREDICTION'].head().tolist()}")
    # 若有 select_strategy_and_params 或策略物件，請在傳遞時加上 debug print
    # 例如：
    # print(f"[DEBUG] 傳遞給 LLM 的 LSTM_PREDICTION: {df['LSTM_PREDICTION'].iloc[-1]}")
    # print(f"[DEBUG] 傳遞給 {strategy_name} 的 LSTM_PREDICTION: {df['LSTM_PREDICTION'].iloc[-1]}")

    simulator = TradeSimulator(
        initial_capital=INITIAL_CAPITAL,
        stop_loss=STOP_LOSS_THRESHOLD,
        allow_short=ALLOW_SHORT_SELLING,
        stock_symbol=STOCK_SYMBOL,
        short_qty_cap=SHORT_QTY_CAP,
        # 新增強制交易參數
        enable_forced_trading=ENABLE_FORCED_TRADING,
        forced_trade_take_profit_pct=FORCED_TRADE_TAKE_PROFIT_PCT,
        forced_trade_stop_loss_pct=FORCED_TRADE_STOP_LOSS_PCT,
        forced_trade_use_trailing_stop=FORCED_TRADE_USE_TRAILING_STOP,
        forced_trade_capital_allocation=FORCED_TRADE_CAPITAL_ALLOCATION,
        # 新增交易單位與精度參數
        trade_unit=TRADE_UNIT,
        price_precision_rules=PRICE_PRECISION_RULES
    )

    strategy_classes = {
        "TrendStrategy": TrendStrategy, # 注意：沒有 ()
        "RangeStrategy": RangeStrategy,
        "BreakoutStrategy": BreakoutStrategy,
        "VolumePriceStrategy": VolumePriceStrategy
    }

    df['date'] = pd.to_datetime(df['date']).dt.date # Ensure it's date objects
    df = df.sort_values("date").reset_index(drop=True)

    # 補上策略可能使用的欄位名稱與指標
    # 確保模擬器需要的欄位名稱存在且大小寫正確
    # 將欄位名稱重命名為大寫版本，假設 load_price_data 提供小寫版本
    # df.rename(columns={ # MOVED EARLIER to operate on full_df
    #     "open": "Open",
    #     "high": "High",
    #     "low": "Low",
    #     "close": "Close", # MA 和 RSI 將使用此欄位
    #     "volume": "Volume"
    # }, inplace=True)

    # 確保必要的欄位存在，以防重命名未發生 (例如，如果 load_price_data 已返回大寫名稱)
    # 如果 'Close' 等欄位不存在，以下 MA 和 RSI 計算會出錯
    # 這裡假設 'Close' 欄位在重命名後或原本就存在

    # 注意：TrendStrategy 目前直接使用此處計算的 MA(20)
    if 'Close' in df.columns:
        df["MA"] = df["Close"].rolling(window=20).mean()
    else:
        print("[警告] df 中缺少 'Close' 欄位，無法計算 MA。")

    def compute_rsi(series, window=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        if loss.eq(0).any(): # Avoid division by zero
             rs = pd.Series([float('inf')] * len(loss)) # or some other large number / specific handling
             rs[loss != 0] = gain[loss != 0] / loss[loss != 0]
        else:
            rs = gain / loss
        return 100 - (100 / (1 + rs))

    if 'Close' in df.columns:
        df["RSI"] = compute_rsi(df["Close"])
    else:
        print("[警告] df 中缺少 'Close' 欄位，無法計算 RSI。")
    
    print("DEBUG: df after initial processing and feature engineering:")
    print(df.head(3))
    print(df.isnull().sum())

    # Initialize the main loop\'s lstm_trained flag based on pre-training result
    lstm_trained = lstm.is_trained # 正確初始化 lstm_trained

    # === 新增：策略目標持續性相關變數 ===
    current_active_strategy_name = None
    current_active_strategy_params = None
    current_strategy_start_date = None
    # 新增：追蹤連續選同一策略次數
    strategy_repeat_count = 0
    last_llm_selected_strategy = None
    MAX_REPEAT_STRATEGY = 3  # 連續同策略最大次數
    # === END 新增 ===

    # === 策略參數優化 ===
    from config import PARAM_GRIDS
    if ENABLE_STRATEGY_OPTIMIZATION:
        print("[資訊] 啟用策略參數優化")
        param_grids = PARAM_GRIDS.copy()
    else:
        param_grids = {}

    eval_map = {
        "win_rate": lambda m: m.get("win_rate_percentage", 0.0), # 修正鍵名並增加 .get 以防萬一
        "total_return": lambda m: m.get("total_return_percentage", 0.0), # 修正鍵名並增加 .get
        "sharpe": lambda m: m.get("sharpe", 0.0), # 增加 .get
    }
    evaluator = eval_map.get(STRATEGY_EVALUATOR, lambda m: m.get("sharpe", 0.0))
    # Correct pre_start_for_opt to use data before START_DATE from full_df
    pre_start_for_opt = full_df[full_df["date"] < backtest_start_date_dt].copy()
    print(f"[資訊] 用於策略優化的 pre_start_for_opt 資料期間: {pre_start_for_opt['date'].min() if not pre_start_for_opt.empty else 'N/A'} 至 {pre_start_for_opt['date'].max() if not pre_start_for_opt.empty else 'N/A'}, 共 {len(pre_start_for_opt)} 筆")


    # optimized_strategy_classes = {} # 此變數目前未實際用於改變策略行為
    for name, strat_class in strategy_classes.items():
        grid = param_grids.get(name)
        if grid:
            opt = StrategyOptimizer(strat_class, grid, evaluator)
            min_days_for_opt = max(max(grid.get("window", [0])) if grid.get("window") else 0, 20)
            if not pre_start_for_opt.empty and len(pre_start_for_opt) >= min_days_for_opt:
                print(f"[資訊] 開始優化 {name}，使用 {len(pre_start_for_opt)} 筆數據。所需最小數據: {min_days_for_opt}")
                result = opt.optimize(pre_start_for_opt.copy())
                best_params = result.get("best_params") or {}
                print(f"[優化] {name} 最佳參數: {best_params}")
                # 僅將 best_params 傳給 LLM 參考，不再自動寫入 param_grids
            else:
                print(f"[優化警告] {name} 的 pre_start_for_opt 數據不足 ({len(pre_start_for_opt)} 筆, 需要 {min_days_for_opt} 筆) 或 grid 配置不當，跳過優化。")

    print(f"DEBUG: strategy_classes after potential optimization: {strategy_classes}")
    print(f"DEBUG: param_grids used for LLM: {param_grids}")

    # 只將 optimizer_best_params 傳給 LLM 參考，不再作為主流程參數來源
    optimizer_best_params = {name: best_params if 'best_params' in locals() else {} for name in strategy_classes}

    # 初始化用於收集結果的列表
    trade_logs = []
    daily_results_df = pd.DataFrame(columns=['date', 'capital']) # Initialize daily_results_df

    # === 主流程 ===
    current_day_dt = pd.to_datetime(START_DATE).date() # Ensure current_day_dt is a date object
    days_since_last_trade = 0

    # --- Portfolio Protection and Capital Allocation Variables ---
    portfolio_protection_triggered_this_year = False
    current_year_pnl_tracker = 0.0 
    protected_profit_level = 0.0
    effective_capital_factor_limit = 1.0 
    halt_all_new_trades_flag = False 
    last_processed_year_for_protection = None
    
    # Load all historical trades if a log file exists, to calculate previous years' PNL
    all_trades_history_df = pd.DataFrame()
    trade_log_path = f"g:/final/產治結果/{STOCK_SYMBOL.replace('.TW', '')}trade_log.csv" # Construct path
    if os.path.exists(trade_log_path):
        try:
            all_trades_history_df = pd.read_csv(trade_log_path)
            if 'Date' in all_trades_history_df.columns:
                 all_trades_history_df['Date'] = pd.to_datetime(all_trades_history_df['Date'], errors='coerce')
            print(f"Successfully loaded historical trade log from {trade_log_path}")
        except Exception as e:
            print(f"Error loading historical trade log from {trade_log_path}: {e}")
    # --- End Portfolio Protection Init ---

    if initial_train_successful and initial_train_end_date is not None:
        retrain_day = initial_train_end_date
        print(f"DEBUG: LSTM 預訓練成功，上次訓練日設為 {retrain_day}")
    else:
        # If no successful pre-training, set retrain_day to ensure training happens early in the loop
        retrain_day = current_day_dt - timedelta(days=LSTM_RETRAIN_INTERVAL + 1)
        print(f"DEBUG: LSTM 未進行預訓練或預訓練失敗，下次訓練將盡早於 {current_day_dt} 之後開始。")


    prev_strategy_logging_name = None # 用於日誌記錄的變數，以避免重複打印相同的策略

    # 新增：策略切換紀錄列表
    strategy_switch_log = []

    while current_day_dt <= pd.to_datetime(END_DATE).date(): # Use current_day_dt
        current_year = current_day_dt.year
        trade_occurred_today = False 
        signal_action_str = None 
        forced_trade_action = None 
        llm_suggested_capital_factor_for_trade = None # Initialize for the day

        # --- Yearly Portfolio Protection Logic ---
        if ENABLE_PORTFOLIO_PROTECTION and current_year != last_processed_year_for_protection:
            print(f"--- Processing Portfolio Protection for year: {current_year} ---")
            portfolio_protection_triggered_this_year = False # Reset trigger for new year
            halt_all_new_trades_flag = False # Reset halt flag for new year
            effective_capital_factor_limit = 1.0 # Reset limit for new year
            current_year_pnl_tracker = 0.0 # Reset PNL tracker for the new year
            
            if current_year > PORTFOLIO_START_YEAR:
                previous_year_to_check = current_year - 1
                # Calculate PNL for previous_year_to_check using all_trades_history_df
                # This assumes all_trades_history_df contains up-to-date PNL from previous simulations/runs
                # Or, if trade_log is built incrementally, it should be used.
                # For simplicity, using simulator's log if available, else history.
                current_sim_trade_log_df = simulator.get_trade_log_df()
                
                # Prioritize current simulation's log for previous year's PNL if it spans multiple years
                # otherwise, use the loaded historical log.
                df_for_prev_year_pnl = current_sim_trade_log_df if not current_sim_trade_log_df.empty and current_sim_trade_log_df['Date'].dt.year.min() < current_year else all_trades_history_df
                
                prev_year_pnl_value = calculate_yearly_pnl(df_for_prev_year_pnl, previous_year_to_check)
                
                if prev_year_pnl_value > 0:
                    protected_profit_level = prev_year_pnl_value * PORTFOLIO_PROFIT_PROTECTION_THRESHOLD
                    print(f"Year {current_year}: Protected profit level set to {protected_profit_level:.2f} (based on {previous_year_to_check} PNL: {prev_year_pnl_value:.2f})")
                else:
                    protected_profit_level = 0 # No profit from previous year to protect
                    print(f"Year {current_year}: No profit to protect from {previous_year_to_check} (PNL: {prev_year_pnl_value:.2f})")
            else:
                protected_profit_level = 0 # Not past the start year for protection
                print(f"Year {current_year}: Portfolio protection not active (current year <= PORTFOLIO_START_YEAR).")
            last_processed_year_for_protection = current_year
        # --- End Yearly Portfolio Protection Logic ---

        print(f"\\nDEBUG: ----- Processing Day: {current_day_dt} -----") # Corrected f-string and escaping
        # Corrected line: Ensure 'date' is treated as a column name string
        # Use full_df to get all historical data up to the day before current_day
        past_df = full_df[full_df['date'] < current_day_dt].copy()
        today_row = df[df['date'] == current_day_dt]

        if today_row.empty:
            print(f"DEBUG [{current_day_dt}]: No data for today. Skipping.")
            current_day_dt += timedelta(days=1)
            days_since_last_trade +=1 # Increment even on no-data days for forced trading
            simulator.record_daily_capital(current_day_dt) # Record capital even if no trade
            continue

        min_hist_days_for_strat_features = MAX_LOOKBACK_PERIOD
        if len(past_df) < max(LSTM_LOOKBACK_DAYS, min_hist_days_for_strat_features):
            print(f"DEBUG [{current_day_dt}]: Insufficient past_df data (len: {len(past_df)}). Needs at least {max(LSTM_LOOKBACK_DAYS, min_hist_days_for_strat_features)}. Skipping.")
            current_day_dt += timedelta(days=1)
            days_since_last_trade +=1
            simulator.record_daily_capital(current_day_dt)
            continue
        
        # retrain LSTM 模型
        # Ensure retrain_day is a date object for comparison
        if isinstance(retrain_day, pd.Timestamp): retrain_day = retrain_day.date()

        if (len(past_df) >= LSTM_TRAIN_WINDOW and
           ((not lstm_trained) or (current_day_dt - retrain_day).days >= LSTM_RETRAIN_INTERVAL)):
            train_data = past_df.tail(LSTM_TRAIN_WINDOW)
            # 確保訓練數據本身也足夠長以產生樣本
            if len(train_data) >= LSTM_LOOKBACK_DAYS + LSTM_PREDICT_DAYS + 1:
                print(f"DEBUG [{current_day_dt}]: Training LSTM with {len(train_data)} data points.")
                try:
                    lstm.train(train_data[['date', 'Close']]) # Ensure correct columns
                    retrain_day = current_day_dt
                    lstm_trained = True
                    print(f"DEBUG [{current_day_dt}]: LSTM 訓練成功。")
                except Exception as e:
                    print(f"錯誤 [{current_day_dt}]: LSTM 訓練失敗: {e}")
            else:
                print(f"DEBUG [{current_day_dt}]: LSTM 訓練數據集長度 ({len(train_data)}) 不足以產生訓練樣本。跳過此次訓練。")

        recent_data_for_lstm = past_df.tail(LSTM_LOOKBACK_DAYS)
        lstm_signal = 0
        if lstm_trained and len(recent_data_for_lstm) >= LSTM_LOOKBACK_DAYS:
            try:
                lstm_signal = lstm.predict(recent_data_for_lstm[['date', 'Close']]) # Ensure correct columns
            except Exception as e:
                print(f"錯誤 [{current_day_dt}]: LSTM 預測失敗: {e}")
        print(f"DEBUG [{current_day_dt}]: LSTM trained: {lstm_trained}, LSTM signal: {lstm_signal}")

        market_regime_window = 30
        if len(past_df) >= market_regime_window:
            regime = calculate_market_regime(past_df.tail(market_regime_window * 2))
        else:
            regime = "unknown"
        print(f"DEBUG [{current_day_dt}]: Market regime: {regime}")
        
        current_date_str_for_llm = current_day_dt.strftime('%Y-%m-%d')
        if len(past_df) >= 5:
            cols_for_llm_price = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] if col in past_df.columns]
            price_data_for_llm_str = past_df.tail(5)[cols_for_llm_price].to_string()
        elif not past_df.empty:
            cols_for_llm_price = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] if col in past_df.columns]
            price_data_for_llm_str = past_df[cols_for_llm_price].to_string()
        else:
            price_data_for_llm_str = "No historical price data available for LLM."
        news_sentiment_summary_for_llm = f"Market Regime: {regime}. News details not processed at this call site."
        available_strategies_for_llm = list(strategy_classes.keys())

        # --- 輔助函數：計算年度 PNL ---
        def calculate_yearly_pnl(trade_log_df, year):
            if trade_log_df.empty or 'Date' not in trade_log_df.columns or 'PNL' not in trade_log_df.columns:
                return 0.0
            
            # Ensure 'Date' is datetime like
            try:
                # Attempt to convert, handling potential mixed formats or errors
                log_df_copy = trade_log_df.copy()
                log_df_copy['Date'] = pd.to_datetime(log_df_copy['Date'], errors='coerce')
                # Drop rows where date conversion failed
                log_df_copy.dropna(subset=['Date'], inplace=True)
                
                yearly_trades = log_df_copy[log_df_copy['Date'].dt.year == year]
                return yearly_trades['PNL'].sum()
            except Exception as e:
                print(f"Error in calculate_yearly_pnl during date conversion or sum: {e}")
                return 0.0

        days_current_strategy_active = 0
        if current_active_strategy_name and current_strategy_start_date:
            # Ensure current_strategy_start_date is a date object
            if isinstance(current_strategy_start_date, pd.Timestamp): current_strategy_start_date = current_strategy_start_date.date()
            days_current_strategy_active = (current_day_dt - current_strategy_start_date).days
        
        print(f"DEBUG [{current_day_dt}]: Current active strategy: {current_active_strategy_name}, active for {days_current_strategy_active} days. Min duration: {MIN_STRATEGY_DURATION_DAYS}")

        current_last_trade_pnl = simulator.get_last_trade_pnl()
        current_cumulative_pnl = simulator.get_cumulative_pnl() # Changed to method call

        todays_recommended_signal_for_llm = None
        # --- LLM Strategy Selection and Signal Generation ---
        # forced_trade_action = None  # Reset forced trade decision for the day # 已在迴圈開頭重置
        current_date_str_for_llm = current_day_dt.strftime('%Y-%m-%d')
        if len(past_df) >= 5:
            cols_for_llm_price = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] if col in past_df.columns]
            price_data_for_llm_str = past_df.tail(5)[cols_for_llm_price].to_string()
        elif not past_df.empty:
            cols_for_llm_price = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] if col in past_df.columns]
            price_data_for_llm_str = past_df[cols_for_llm_price].to_string()
        else:
            price_data_for_llm_str = "No historical price data available for LLM."
        news_sentiment_summary_for_llm = f"Market Regime: {regime}. News details not processed at this call site."
        available_strategies_for_llm = list(strategy_classes.keys())

        # --- 策略目標選擇/持續邏輯 ---
        days_current_strategy_active = 0
        if current_active_strategy_name and current_strategy_start_date:
            days_current_strategy_active = (current_day_dt - current_strategy_start_date).days
        
        print(f"DEBUG [{current_day_dt}]: Current active strategy: {current_active_strategy_name}, active for {days_current_strategy_active} days. Min duration: {MIN_STRATEGY_DURATION_DAYS}")

        # 在呼叫 LLM 之前獲取 PNL 數據
        current_last_trade_pnl = simulator.get_last_trade_pnl() # Changed to method call
        current_cumulative_pnl = simulator.get_cumulative_pnl() # Changed to method call

        # --- LLM 決策 ---
        todays_recommended_signal_for_llm = None # Default to None
        # Try to get current strategy's signal to inform forced trade LLM
        if USE_GEMINI_STRATEGY_SELECTION and current_active_strategy_name and current_active_strategy_name in strategy_classes:
            print(f"DEBUG [{current_day_dt}]: Pre-calculating signal for '{current_active_strategy_name}' to inform forced trade LLM.")
            TempStrategyClass = strategy_classes[current_active_strategy_name]
            # Prepare data for this temporary signal generation
            temp_df_for_signal = full_df[full_df['date'] < current_day_dt].copy()
            
            if 'Close' in temp_df_for_signal.columns:
                temp_df_for_signal["MA"] = temp_df_for_signal["Close"].rolling(window=20).mean()
                temp_df_for_signal["RSI"] = compute_rsi(temp_df_for_signal["Close"]) # compute_rsi is defined in main()
            else:
                print(f"WARNING [{current_day_dt}]: 'Close' column missing in temp_df_for_signal for LLM pre-signal.")

            if current_active_strategy_name == "TrendStrategy":
                if not temp_df_for_signal.empty and 'Prediction' not in temp_df_for_signal.columns: # Avoid overwriting if already there
                    # Ensure the last row exists before trying to write to .loc
                    if not temp_df_for_signal.empty:
                         temp_df_for_signal.loc[temp_df_for_signal.index[-1], 'Prediction'] = lstm_signal
                elif temp_df_for_signal.empty:
                    print(f"WARNING [{current_day_dt}]: temp_df_for_signal is empty, cannot attach LSTM prediction for TrendStrategy (LLM pre-signal).")

            temp_strategy_instance = TempStrategyClass(params=current_active_strategy_params if current_active_strategy_params else {})
            min_lookback_temp = getattr(temp_strategy_instance, 'MIN_LOOKBACK', MAX_LOOKBACK_PERIOD)

            if len(temp_df_for_signal) >= min_lookback_temp:
                temp_pos_status = "Long" if simulator.current_position > 0 else "Short" if simulator.current_position < 0 else None
                try:
                    raw_signal = temp_strategy_instance.generate_signal(
                        data_slice=temp_df_for_signal,
                        current_index=current_day_dt, # Consistent with later signal generation
                        position=temp_pos_status
                    )
                    if raw_signal == "BUY":
                        todays_recommended_signal_for_llm = 1
                    elif raw_signal == "SELL":
                        todays_recommended_signal_for_llm = -1
                    # else: # HOLD or None or any other string
                    #    todays_recommended_signal_for_llm = 0 
                    # Corrected logic: Explicitly set to 0 for HOLD or other non-BUY/SELL signals
                    else: 
                        todays_recommended_signal_for_llm = 0

                    print(f"DEBUG [{current_day_dt}]: Pre-calculated signal for LLM: {raw_signal} -> {todays_recommended_signal_for_llm}")
                except Exception as e_sig:
                    print(f"DEBUG [{current_day_dt}]: Error generating temp signal for LLM: {e_sig}")
                    todays_recommended_signal_for_llm = 0 # Default to neutral on error
            else:
                print(f"DEBUG [{current_day_dt}]: Not enough data for temp signal for LLM (needs {min_lookback_temp}, got {len(temp_df_for_signal)}).")
                todays_recommended_signal_for_llm = 0


        # 優先處理強制交易的判斷
        if USE_GEMINI_STRATEGY_SELECTION and ENABLE_FORCED_TRADING and days_since_last_trade >= MAX_DAYS_NO_TRADE and not halt_all_new_trades_flag:
            print(f"DEBUG [{current_day_dt}]: Potential forced trade scenario. Days since last trade: {days_since_last_trade}. Calling LLM for forced trade decision.")
            
            forced_decision, _, _, llm_context_forced = select_strategy_and_params( # Expecting 4 return values now
                current_date=current_date_str_for_llm, price_df=full_df,
                news_sentiment_summary=news_sentiment_summary_for_llm, available_strategies=available_strategies_for_llm, 
                lstm_signal=lstm_signal, days_since_last_trade=days_since_last_trade,
                max_days_no_trade=MAX_DAYS_NO_TRADE, is_forced_trade_scenario=True,
                recommended_strategy_name=current_active_strategy_name,
                recommended_strategy_signal=todays_recommended_signal_for_llm,
                # Pass PNL context even for forced trades if it influences decision to abstain/proceed
                last_trade_pnl=current_last_trade_pnl, cumulative_pnl=current_cumulative_pnl
            )
            print(f"DEBUG [{current_day_dt}]: LLM forced trade decision: {forced_decision}")

            if forced_decision in ["ForcedBuy", "ForcedShort"]:
                forced_trade_action = forced_decision
                current_active_strategy_name = None 
                current_active_strategy_params = {}
                current_strategy_start_date = None
                # For forced trades, capital allocation is from FORCED_TRADE_CAPITAL_ALLOCATION
                # No specific LLM factor for forced trades in this setup, but could be added.
                llm_suggested_capital_factor_for_trade = FORCED_TRADE_CAPITAL_ALLOCATION / REGULAR_TRADE_CAPITAL_ALLOCATION if REGULAR_TRADE_CAPITAL_ALLOCATION > 0 else 1.0
                print(f"DEBUG [{current_day_dt}]: LLM forced trade: {forced_trade_action}. Regular strategy reset. Factor based on FORCED_TRADE_CAPITAL_ALLOCATION.")
            elif forced_decision == "AbstainForceTrade":
                forced_trade_action = "AbstainForceTrade" 
                print(f"DEBUG [{current_day_dt}]: LLM decided to AbstainForceTrade.")
            else: 
                print(f"WARNING [{current_day_dt}]: LLM unexpected forced decision: {forced_decision}. Abstaining.")
                forced_trade_action = "AbstainForceTrade"
        
        # 如果沒有執行強制交易 (LLM決定不強制 或 未達到強制交易天數)
        # 則進行常規的策略目標選擇/持續判斷
        if USE_GEMINI_STRATEGY_SELECTION and (not forced_trade_action or forced_trade_action == "AbstainForceTrade") and not halt_all_new_trades_flag:
            if forced_trade_action == "AbstainForceTrade":
                print(f"DEBUG [{current_day_dt}]: LLM abstained from forced trade. Regular strategy selection.")

            should_select_new_strategy_target = False
            if not current_active_strategy_name: 
                should_select_new_strategy_target = True
            elif days_current_strategy_active >= MIN_STRATEGY_DURATION_DAYS:
                should_select_new_strategy_target = True

            if should_select_new_strategy_target:
                print(f"DEBUG [{current_day_dt}]: Calling LLM to select/revise strategy target.")
                # 計算已交易次數與勝率
                trade_log_df = simulator.get_trade_log_df()
                total_trades = len(trade_log_df) if not trade_log_df.empty else 0
                win_rate = None
                if total_trades > 0 and 'PNL' in trade_log_df.columns:
                    wins = (trade_log_df['PNL'] > 0).sum()
                    win_rate = wins / total_trades
                # 新版：接收 protection_action, protection_reason
                strategy_name, strategy_params, llm_capital_factor, llm_context_strat, protection_action, protection_reason = select_strategy_and_params(
                    current_date=current_date_str_for_llm, price_df=full_df,
                    news_sentiment_summary=news_sentiment_summary_for_llm, available_strategies=available_strategies_for_llm,
                    lstm_signal=lstm_signal, days_since_last_trade=days_since_last_trade, 
                    max_days_no_trade=MAX_DAYS_NO_TRADE, is_forced_trade_scenario=False, 
                    current_active_strategy_name=current_active_strategy_name,
                    current_strategy_days_active=days_current_strategy_active,
                    min_strategy_duration=MIN_STRATEGY_DURATION_DAYS,
                    last_trade_pnl=current_last_trade_pnl, cumulative_pnl=current_cumulative_pnl,
                    optimizer_best_params=optimizer_best_params,
                    regime=regime, # 傳入 regime
                    total_trades=total_trades, win_rate=win_rate
                )
                # 新增：根據 LLM 建議啟動/解除保護機制
                if protection_action:
                    print(f"[LLM保護建議] {protection_action}，理由：{protection_reason}")
                    if protection_action == "HALT_TRADING":
                        halt_all_new_trades_flag = True
                    elif protection_action == "REDUCE_RISK":
                        effective_capital_factor_limit = PORTFOLIO_REDUCED_RISK_FACTOR_MAX
                    elif protection_action == "RESUME_NORMAL":
                        halt_all_new_trades_flag = False
                        effective_capital_factor_limit = 1.0

                llm_suggested_capital_factor_for_trade = llm_capital_factor # Store LLM's suggestion
                print(f"DEBUG [{current_day_dt}]: LLM (strategy target) returned: name='{strategy_name}', params={strategy_params}, capital_factor={llm_capital_factor}")

                if strategy_name in strategy_classes:
                    # 若 LLM 有回傳 strategy_params，直接採用
                    if strategy_params:
                        current_active_strategy_params = strategy_params
                    # 若 LLM 無回傳，則 fallback 到 config.py 預設參數（不可用 optimizer_best_params 或 param_grids）
                    else:
                        # 取得 config.py 預設參數（假設每個策略類別有 get_default_params 靜態方法，否則可手動指定）
                        if hasattr(strategy_classes[strategy_name], 'get_default_params'):
                            current_active_strategy_params = strategy_classes[strategy_name].get_default_params()
                        else:
                            current_active_strategy_params = {}
                    current_active_strategy_name = strategy_name
                    current_strategy_start_date = current_day_dt
                    strategy_repeat_count = 0
                    last_llm_selected_strategy = strategy_name
                elif strategy_name == "Abstain" or strategy_name is None:
                    print(f"DEBUG [{current_day_dt}]: LLM abstained from changing strategy target.")
                else:
                    print(f"WARNING [{current_day_dt}]: LLM returned invalid strategy: '{strategy_name}'. Resetting.")
                    current_active_strategy_name = None; current_active_strategy_params = {}; current_strategy_start_date = None
            elif current_active_strategy_name:
                print(f"DEBUG [{current_day_dt}]: Continuing with strategy: {current_active_strategy_name}")
        
        elif not USE_GEMINI_STRATEGY_SELECTION and not halt_all_new_trades_flag:
            if not current_active_strategy_name: 
                current_active_strategy_name = DEFAULT_STRATEGY 
                current_active_strategy_params = {}
                current_strategy_start_date = current_day_dt
                print(f"DEBUG [{current_day_dt}]: Default strategy '{current_active_strategy_name}' activated.")
        
        # --- Determine Final Capital Allocation Factor ---
        final_trade_capital_factor = DEFAULT_CAPITAL_ALLOCATION_FACTOR # Fallback
        if ENABLE_LLM_CAPITAL_ALLOCATION and llm_suggested_capital_factor_for_trade is not None:
            if 0.0 <= llm_suggested_capital_factor_for_trade <= 1.0:
                final_trade_capital_factor = llm_suggested_capital_factor_for_trade
            else:
                print(f"Warning: LLM suggested capital factor {llm_suggested_capital_factor_for_trade} out of range. Using default.")
        
        # Apply portfolio protection limit
        final_trade_capital_factor = min(final_trade_capital_factor, effective_capital_factor_limit)
        print(f"DEBUG [{current_day_dt}]: Capital Allocation: LLM_suggested={llm_suggested_capital_factor_for_trade}, Protection_Limit={effective_capital_factor_limit}, Final_Factor_for_REGULAR_ALLOC={final_trade_capital_factor}")


        log_date_str = current_day_dt.strftime('%Y-%m-%d') 
        trade_executed_this_iteration = None # To store the result of buy/sell

        if halt_all_new_trades_flag:
            print(f"LOG [{log_date_str}]: HALT_TRADING active due to portfolio protection. No new trades.")
            signal_action_str = "HOLD" # Override any signal
            forced_trade_action = None # Ensure no forced trade either

        # --- Execute Forced Trade Action ---
        if forced_trade_action in ["ForcedBuy", "ForcedShort"] and not halt_all_new_trades_flag:
            print(f"LOG [{log_date_str}]: Processing LLM Forced Action: {forced_trade_action}")
            # Use FORCED_TRADE_CAPITAL_ALLOCATION directly for quantity calculation
            # The 'final_trade_capital_factor' is more for regular trades scaled by LLM.
            # Forced trades have their own capital allocation setting.
            capital_for_forced_trade = simulator.cash * FORCED_TRADE_CAPITAL_ALLOCATION
            price_for_trade = today_row['Open'].iloc[0] # Use today's open for trade
            quantity_to_trade = int(capital_for_forced_trade / price_for_trade)
            quantity_to_trade = simulator._adjust_quantity(quantity_to_trade) # Use simulator's method

            if quantity_to_trade > 0:
                if forced_trade_action == "ForcedBuy":
                    trade_executed_this_iteration = simulator.buy(price_for_trade, quantity_to_trade, current_day_dt, trade_type="ForcedBuy")
                elif forced_trade_action == "ForcedShort":
                    trade_executed_this_iteration = simulator.sell(price_for_trade, quantity_to_trade, current_day_dt, trade_type="ForcedShort")
            else:
                print(f"LOG [{log_date_str}]: Forced trade quantity is 0. No trade executed.")
            days_since_last_trade = 0 # Reset counter if forced trade attempted/executed
            trade_occurred_today = True if trade_executed_this_iteration else False


        # --- Execute Regular Strategy Signal ---
        elif (not forced_trade_action or forced_trade_action == "AbstainForceTrade") and not halt_all_new_trades_flag:
            if current_active_strategy_name and current_active_strategy_name in strategy_classes:
                # ... (Strategy signal generation - existing code) ...
                if prev_strategy_logging_name != current_active_strategy_name:
                    print(f"DEBUG [{current_day_dt}]: Attempting to use strategy: {current_active_strategy_name} with params: {current_active_strategy_params}")
                    prev_strategy_logging_name = current_active_strategy_name
                
                StrategyClass = strategy_classes[current_active_strategy_name]
                # Use full_df for all historical data up to day before current_day_dt
                df_for_signal_gen = full_df[full_df['date'] < current_day_dt].copy()


                if 'Close' in df_for_signal_gen.columns:
                    df_for_signal_gen["MA"] = df_for_signal_gen["Close"].rolling(window=20).mean()
                    df_for_signal_gen["RSI"] = compute_rsi(df_for_signal_gen["Close"])
                else:
                    print(f"WARNING [{current_day_dt}]: 'Close' column missing in df_for_signal_gen for {current_active_strategy_name}.")

                if current_active_strategy_name == "TrendStrategy":
                    if not df_for_signal_gen.empty:
                        df_for_signal_gen.loc[df_for_signal_gen.index[-1], 'Prediction'] = lstm_signal
                    else:
                        print(f"WARNING [{current_day_dt}]: df_for_signal_gen is empty for TrendStrategy signal.")
            
                current_active_strategy_instance = StrategyClass(**current_active_strategy_params if current_active_strategy_params else {})
                min_lookback_needed_by_strategy = getattr(current_active_strategy_instance, 'MIN_LOOKBACK', MAX_LOOKBACK_PERIOD)

                if len(df_for_signal_gen) >= min_lookback_needed_by_strategy:
                    position_status = "Long" if simulator.current_position > 0 else "Short" if simulator.current_position < 0 else None
                    try:
                        signal_action_str = current_active_strategy_instance.generate_signal(
                            data_slice=df_for_signal_gen, current_index=current_day_dt, position=position_status
                        )
                        print(f"LOG [{log_date_str}] Strategy Signal ({current_active_strategy_name}): '{signal_action_str}'")
                        if signal_action_str is not None:
                            signal_action_str = signal_action_str.upper() # Convert to uppercase
                        else:
                            signal_action_str = "HOLD"
                    except Exception as e:
                        print(f"ERROR [{log_date_str}] Error generating signal from {current_active_strategy_name}: {e}")
                        signal_action_str = "HOLD" 
                else:
                    print(f"WARNING [{current_day_dt}]: Not enough data for {current_active_strategy_name}. Holding.")
                    signal_action_str = "HOLD"

                # --- Execute trade based on signal_action_str ---
                if signal_action_str in ["BUY", "SELL", "SHORT"]:
                    price_for_trade = today_row['Open'].iloc[0]
                    # Calculate quantity using REGULAR_TRADE_CAPITAL_ALLOCATION * final_trade_capital_factor
                    capital_to_use = simulator.cash * REGULAR_TRADE_CAPITAL_ALLOCATION * final_trade_capital_factor
                    quantity_to_trade = int(capital_to_use / price_for_trade)
                    quantity_to_trade = simulator._adjust_quantity(quantity_to_trade)

                    if quantity_to_trade > 0:
                        if signal_action_str == "BUY":
                            if simulator.current_position < 0 : # Covering short
                                trade_executed_this_iteration = simulator.buy(price_for_trade, simulator.current_position_quantity, current_day_dt, trade_type="RegularCover")
                            elif simulator.current_position == 0: # New long
                                trade_executed_this_iteration = simulator.buy(price_for_trade, quantity_to_trade, current_day_dt, trade_type="RegularBuy")
                        elif signal_action_str == "SELL":
                            if simulator.current_position > 0: # Exiting long
                                trade_executed_this_iteration = simulator.sell(price_for_trade, simulator.current_position_quantity, current_day_dt, trade_type="RegularSell")
                        elif signal_action_str == "SHORT":
                            if simulator.current_position == 0 and ALLOW_SHORT_SELLING: # New short
                                trade_executed_this_iteration = simulator.sell(price_for_trade, quantity_to_trade, current_day_dt, trade_type="RegularShort")
                    else:
                        print(f"LOG [{log_date_str}]: Calculated quantity is 0 for {signal_action_str}. No trade.")
                
                trade_occurred_today = True if trade_executed_this_iteration else False
                if trade_occurred_today:
                    days_since_last_trade = 0
                else:
                    days_since_last_trade += 1
            else: # No active strategy
                 print(f"LOG [{log_date_str}]: No valid active strategy. Holding.")
                 days_since_last_trade += 1
        else: # A forced trade was executed, or halt flag is on
            if not halt_all_new_trades_flag : # if not halting, means forced trade happened or abstained
                 if not forced_trade_action or forced_trade_action == "AbstainForceTrade":
                    days_since_last_trade +=1 # only increment if no trade truly happened

        # --- Daily PNL Update & Portfolio Protection Check ---
        if trade_executed_this_iteration: # A trade log entry was returned
            # Update current_year_pnl_tracker with the PNL from the executed trade
            # Ensure PNL is a float, default to 0.0 if not present or not a number
            pnl_from_trade = trade_executed_this_iteration.get('PNL', 0.0)
            if isinstance(pnl_from_trade, (int, float)):
                current_year_pnl_tracker += pnl_from_trade
            else:
                print(f"Warning: PNL value '{pnl_from_trade}' from trade is not a number. Skipping PNL update for this trade.")

        if ENABLE_PORTFOLIO_PROTECTION and \
           not portfolio_protection_triggered_this_year and \
           protected_profit_level > 0 and \
           current_year > PORTFOLIO_START_YEAR:
            
            # current_year_pnl_tracker should reflect PNL from Jan 1 of current_year up to today
            if current_year_pnl_tracker < 0 and abs(current_year_pnl_tracker) > protected_profit_level:
                portfolio_protection_triggered_this_year = True
                print(f"!!! PORTFOLIO PROTECTION TRIGGERED for year {current_year} !!!")
                print(f"Current Year PNL Tracker: {current_year_pnl_tracker:.2f}, Protected Level: {protected_profit_level:.2f}")
                
                if PORTFOLIO_PROTECTION_ACTION == "HALT_TRADING":
                    halt_all_new_trades_flag = True
                    print("Action: HALT_TRADING activated for the rest of the year.")
                elif PORTFOLIO_PROTECTION_ACTION == "REDUCE_RISK":
                    effective_capital_factor_limit = PORTFOLIO_REDUCED_RISK_FACTOR_MAX
                    print(f"Action: REDUCE_RISK activated. Capital factor limit set to {effective_capital_factor_limit:.2f}")
        # --- End Daily PNL Update & Portfolio Protection Check ---

        # --- Stop Loss / Take Profit Checks for open positions ---
        # These should be checked daily regardless of new signals, if a position is open
        # Make sure current_day_dt is passed to these functions
        if simulator.is_forced_trade_active():
            closed_forced_trade_log = simulator.check_forced_trade_closure(today_row['Close'].iloc[0], current_day_dt)
            if closed_forced_trade_log: 
                trade_occurred_today = True; days_since_last_trade = 0
                # Update PNL tracker if a forced trade was closed by SL/TP
                pnl_from_closure = closed_forced_trade_log.get('PNL', 0.0)
                if isinstance(pnl_from_closure, (int, float)): current_year_pnl_tracker += pnl_from_closure
        elif simulator.current_position != 0: # Regular trade is active
            closed_regular_trade_log = simulator.check_stop_loss_take_profit(today_row['Close'].iloc[0], current_day_dt)
            if closed_regular_trade_log: 
                trade_occurred_today = True; days_since_last_trade = 0
                # Update PNL tracker if a regular trade was closed by SL/TP
                pnl_from_closure = closed_regular_trade_log.get('PNL', 0.0)
                if isinstance(pnl_from_closure, (int, float)): current_year_pnl_tracker += pnl_from_closure
        
        simulator.update_portfolio_value(today_row['Close'].iloc[0]) # Update with day's close
        simulator.record_daily_capital(current_day_dt) # Record capital at end of day

        current_day_dt += timedelta(days=1)
    # --- End Main Loop ---

    # 新增：輸出策略切換紀錄到 strategy_switch_log.csv
    if strategy_switch_log:
        pd.DataFrame(strategy_switch_log).to_csv("strategy_switch_log.csv", index=False)
        print("策略切換紀錄已儲存到 strategy_switch_log.csv")

    # Finalize simulation (e.g., close any open positions)
    # 依 config 決定是否強制清倉
    if FORCE_LIQUIDATE_AT_END:
        # 找到最後一個有資料的交易日
        last_trade_day_row = df[~df['Close'].isnull()].iloc[-1] if not df[~df['Close'].isnull()].empty else None
        if simulator.current_position != 0 and last_trade_day_row is not None:
            closing_price = last_trade_day_row['Close']
            closing_date = last_trade_day_row['date']
            print(f"[強制平倉] End of simulation. Closing position at {closing_price} on {closing_date}")
            if simulator.current_position > 0:
                final_log = simulator.sell(closing_price, simulator.current_position_quantity, closing_date, trade_type="EndOfSim")
            elif simulator.current_position < 0:
                final_log = simulator.buy(closing_price, simulator.current_position_quantity, closing_date, trade_type="EndOfSim")
            if final_log:
                pnl_from_final_closure = final_log.get('PNL', 0.0)
                if isinstance(pnl_from_final_closure, (int, float)):
                    current_year_pnl_tracker += pnl_from_final_closure


    final_trade_log_df = simulator.get_trade_log_df()
    final_daily_capital_df = simulator.get_daily_capital_df()

    # === 結果分析與報告 ===
    # 在寫入 trade_log.csv 前，強制型態轉換，避免報表產生失敗
    final_trade_log_df = simulator.get_trade_log_df()
    if not final_trade_log_df.empty:
        # 強制型態轉換
        final_trade_log_df['Date'] = pd.to_datetime(final_trade_log_df['Date'], format='%Y%m%d', errors='coerce')
        final_trade_log_df['PNL'] = pd.to_numeric(final_trade_log_df['PNL'], errors='coerce')
        final_trade_log_df['Cash'] = pd.to_numeric(final_trade_log_df['Cash'], errors='coerce')
        final_trade_log_df['TradeID'] = pd.to_numeric(final_trade_log_df['TradeID'], errors='coerce')
        final_trade_log_df.to_csv("trade_log.csv", index=False)
        logging.info("Trade log saved to trade_log.csv")
        print(final_trade_log_df.tail())
    else:
        logging.info("No trades were made during the simulation.")

    # 獲取每日資產 DataFrame
    daily_capital_df = simulator.get_daily_capital_df() # simulator.daily_capital is a list of dicts
    #daily_capital_df = daily_results_df.copy()
    if not daily_capital_df.empty:
        # Ensure 'date' is datetime for proper resampling, and 'capital' is numeric
        daily_capital_df['date'] = pd.to_datetime(daily_capital_df['date'])
        daily_capital_df['capital'] = pd.to_numeric(daily_capital_df['capital'])
        
        daily_capital_df.to_csv("daily_capital.csv", index=False) # Save daily capital as well
        logging.info("Daily capital saved to daily_capital.csv")
        print(daily_capital_df.tail())

        # 獲取已完成的交易紀錄以產生月報和整體績效
        report_trade_log_df = simulator.get_trade_log_df() # 使用 simulator 的交易日誌

        # 正確設定 strategy_name_for_report
        # current_active_strategy_name 在模擬迴圈中被更新
        if current_active_strategy_name: # This variable is updated during the simulation loop
            strategy_name_for_report = current_active_strategy_name
        elif not USE_GEMINI_STRATEGY_SELECTION and DEFAULT_STRATEGY: # If LLM selection was off, default was used
            strategy_name_for_report = DEFAULT_STRATEGY
        else:
            # Fallback if no specific strategy was identified
            strategy_name_for_report = "StrategyRun" 

        # 產生月績效報告
        # generate_monthly_report 需要 daily_df, trade_log_df, strategy_name, initial_capital
        report_df, summary_df = generate_monthly_report(
            daily_df=daily_capital_df.copy(), # 傳遞副本以避免意外修改
            raw_trade_log_df=report_trade_log_df.copy(), # 修正參數名稱
            strategy_name=strategy_name_for_report,
            initial_capital=INITIAL_CAPITAL
        )

        # if report_df is not None and not report_df.empty:
        #     csv_filename = f"{strategy_name_for_report}_monthly_performance.csv"
        #     report_df.to_csv(csv_filename, index=True)
        #     logging.info(f"Monthly performance report for {strategy_name_for_report} saved to {csv_filename}")
        #     print(f"--- Monthly Performance Report for {strategy_name_for_report} (tail) ---")
        #     print(report_df.tail())
        #     if summary_df is not None and not summary_df.empty:
        #         summary_filename = f"{strategy_name_for_report}_performance_summary.xlsx"
        #         # summary_df.to_csv(summary_filename, index=False)
        #         # logging.info(f"Performance summary for {strategy_name_for_report} saved to {summary_filename}")
        #         # 輸出到 Excel 的同一個檔案的不同工作表
        #         with pd.ExcelWriter(summary_filename, engine='openpyxl', mode='a' if os.path.exists(summary_filename) else 'w') as writer:
        #             summary_df.to_excel(writer, sheet_name=f"{strategy_name_for_report}_Summary", index=False)
        #         logging.info(f"Performance summary for {strategy_name_for_report} appended to {summary_filename}")
        # else:
        #     logging.info(f"Could not generate monthly performance report for {strategy_name_for_report} (generated report_df is empty or None).")

        # 計算並打印整體績效指標 - 使用 utils.metrics 中的函數
        if not daily_capital_df.empty:
            # calculate_performance_metrics 需要 daily_df 和 trade_log_df (包含 'pnl')
            # trade_log_df 應該是包含所有已執行交易的列表，其中有 'pnl' 欄位
            # simulator.get_trade_log_df() 應該提供這個
            
            # 確保 report_trade_log_df 包含 'PNL' 欄位，並且是數值型態
            if 'PNL' in report_trade_log_df.columns: # simulator 的 PNL 欄位名稱是大寫
                 report_trade_log_df_for_calc = report_trade_log_df.rename(columns={'PNL': 'pnl'})
            else:
                 report_trade_log_df_for_calc = report_trade_log_df.copy()
                 if 'pnl' not in report_trade_log_df_for_calc.columns:
                      report_trade_log_df_for_calc['pnl'] = 0 # 如果沒有 PNL 欄位，則添加一個並設為0

            performance_metrics = calculate_performance_metrics(
                daily_df=daily_capital_df.copy(), 
                trade_log_df=report_trade_log_df_for_calc, # 使用轉換過欄位名的 DataFrame
                initial_capital_for_return_calc=INITIAL_CAPITAL
            )
            logging.info(f"--- Overall Performance Metrics for {strategy_name_for_report} ---")
            for metric, value in performance_metrics.items():
                if isinstance(value, float):
                    logging.info(f"{metric}: {value:.2f}")
                else:
                    logging.info(f"{metric}: {value}")
        else:
            logging.warning(f"Cannot calculate overall performance metrics for {strategy_name_for_report}: daily_capital_df is empty.")

    else:
        logging.info("No daily capital data to generate reports.")


if __name__ == "__main__":
    main()
