import pandas as pd
import inspect
from config import *
from data_loader import load_price_data
from lstm_model import LSTMPredictor
from market_regime import calculate_market_regime
from llm_decision_engine import select_strategy_and_params
from trade_simulator import TradeSimulator
from utils.metrics import generate_monthly_report, calculate_performance_metrics
import strategies # Import the strategies module
from strategies import TrendStrategy, RangeStrategy, BreakoutStrategy, VolumePriceStrategy
from optimizer import StrategyOptimizer
from datetime import timedelta
import logging # Added import for logging

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
    if not full_df.empty:
        full_df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close", # MA 和 RSI 將使用此欄位
            "volume": "Volume"
        }, inplace=True)
        # 驗證 'Close' 是否存在
        if 'Close' not in full_df.columns:
            print("[警告] full_df 在重命名後仍然缺少 'Close' 欄位。請檢查 load_price_data 的回傳。")
            # 可以考慮引發錯誤或採取其他處理
    else:
        print("[警告] load_price_data 回傳空的 DataFrame。程式即將終止。")
        return # Exit if no data

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
    if df.empty:
        raise ValueError(f"回測期間 ({START_DATE} 至 {END_DATE}) 無可用數據，請檢查日期設定或資料來源。")
    print(f"[資訊] 回測資料期間：{df['date'].min()} 至 {df['date'].max()}，共 {len(df)} 筆。")

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
    if ENABLE_STRATEGY_OPTIMIZATION:
        print("[資訊] 啟用策略參數優化")
        param_grids = {
            "TrendStrategy": { # TrendStrategy 目前使用 main.py 計算的固定 MA(20)
                               # 如果要讓 ma_period 可優化，TrendStrategy 需修改以接受並使用此參數
                "rsi_low_entry": [25, 30, 35, 40], # Changed from rsi_low to rsi_low_entry
                "rsi_high_entry": [65, 70, 75, 80], # Changed from rsi_high to rsi_high_entry
                # "ma_period": [10, 20] # 暫時註解，因為 TrendStrategy 未使用
            },
            "RangeStrategy": {
                "window": [10, 15, 20],
                "rsi_low_entry": [25, 30, 35], # Changed from rsi_low
                "rsi_high_entry": [65, 70, 75], # Changed from rsi_high
            },
            "BreakoutStrategy": {
                "window": [10, 15, 20],
                "rsi_low": [30, 40, 45], # Keep as rsi_low as per BreakoutStrategy
                "rsi_high": [60, 65, 70] # Keep as rsi_high as per BreakoutStrategy
            },
            "VolumePriceStrategy": { # VolumePriceStrategy 內部使用固定的 ma_short(5)
                                     # 如果要讓 ma_short_period 可優化，VolumePriceStrategy 需修改
                "volume_ratio": [1.2, 1.5, 1.8],
                "rsi_low": [30, 35, 40], # Keep as rsi_low
                "rsi_high": [60, 65, 70], # Keep as rsi_high
                # "ma_short_period": [5, 10] # 暫時註解，因為 VolumePriceStrategy 未使用
            },
        }

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
                # 確保 pre_start_for_opt 有足夠數據且不含過多 NaN，否則優化器可能無法有效運行
                # The check for max window size and > 20 days
                min_days_for_opt = max(max(grid.get("window", [0])) if grid.get("window") else 0, 20)

                if not pre_start_for_opt.empty and len(pre_start_for_opt) >= min_days_for_opt:
                    print(f"[資訊] 開始優化 {name}，使用 {len(pre_start_for_opt)} 筆數據。所需最小數據: {min_days_for_opt}")
                    result = opt.optimize(pre_start_for_opt.copy()) # 使用 .copy() 避免 SettingWithCopyWarning
                    best_params = result.get("best_params") or {}
                    print(f"[優化] {name} 最佳參數: {best_params}")
                    # 注意：此處找到的 best_params 並未實際更新 param_grids 或影響後續 LLM 的參數選擇範圍
                    # 如果需要，可以在此處修改 param_grids[name] = {k: [v] for k, v in best_params.items()}
                    # 或者將 best_params 傳遞給 LLM 作為一個強提示
                else:
                    print(f"[優化警告] {name} 的 pre_start_for_opt 數據不足 ({len(pre_start_for_opt)} 筆, 需要 {min_days_for_opt} 筆) 或 grid 配置不當，跳過優化。")

    print(f"DEBUG: strategy_classes after potential optimization: {strategy_classes}")
    print(f"DEBUG: param_grids used for LLM: {param_grids}")

    # 初始化用於收集結果的列表
    trade_logs = []
    daily_results_df = pd.DataFrame(columns=['date', 'capital']) # Initialize daily_results_df

    # === 主流程 ===
    current_day = pd.to_datetime(START_DATE).date()
    days_since_last_trade = 0 # 初始化無交易日計數器

    # Initialize retrain_day based on pre-training outcome
    if initial_train_successful and initial_train_end_date is not None:
        retrain_day = initial_train_end_date
        print(f"DEBUG: LSTM 預訓練成功，上次訓練日設為 {retrain_day}")
    else:
        # If no successful pre-training, set retrain_day to ensure training happens early in the loop
        retrain_day = current_day - timedelta(days=LSTM_RETRAIN_INTERVAL + 1)
        print(f"DEBUG: LSTM 未進行預訓練或預訓練失敗，下次訓練將盡早於 {current_day} 之後開始。")


    prev_strategy_logging_name = None # 用於日誌記錄的變數，以避免重複打印相同的策略

    while current_day <= pd.to_datetime(END_DATE).date():
        trade_occurred_today = False # Initialize for the current day
        signal_action_str = None # Reset for the current day's strategy signal output
        forced_trade_action = None # Reset forced trade decision for the day

        print(f"\\nDEBUG: ----- Processing Day: {current_day} -----") # Corrected f-string and escaping
        # Corrected line: Ensure 'date' is treated as a column name string
        # Use full_df to get all historical data up to the day before current_day
        past_df = full_df[full_df['date'] < current_day].copy()
        today_row = df[df['date'] == current_day]

        if today_row.empty:
            print(f"DEBUG [{current_day}]: No data for today. Skipping.")
            current_day += timedelta(days=1)
            continue

        min_hist_days_for_strat_features = MAX_LOOKBACK_PERIOD
        if len(past_df) < max(LSTM_LOOKBACK_DAYS, min_hist_days_for_strat_features):
            print(f"DEBUG [{current_day}]: Insufficient past_df data (len: {len(past_df)}). Needs at least {max(LSTM_LOOKBACK_DAYS, min_hist_days_for_strat_features)}. Skipping.")
            current_day += timedelta(days=1)
            continue
        
        # retrain LSTM 模型
        if (len(past_df) >= LSTM_TRAIN_WINDOW and
           ((not lstm_trained) or (current_day - retrain_day).days >= LSTM_RETRAIN_INTERVAL)):
            train_data = past_df.tail(LSTM_TRAIN_WINDOW)
            # 確保訓練數據本身也足夠長以產生樣本
            if len(train_data) >= LSTM_LOOKBACK_DAYS + LSTM_PREDICT_DAYS + 1:
                print(f"DEBUG [{current_day}]: Training LSTM with {len(train_data)} data points.")
                try:
                    lstm.train(train_data)
                    retrain_day = current_day
                    lstm_trained = True
                    print(f"DEBUG [{current_day}]: LSTM 訓練成功。")
                except Exception as e:
                    print(f"錯誤 [{current_day}]: LSTM 訓練失敗: {e}")
            else:
                print(f"DEBUG [{current_day}]: LSTM 訓練數據集長度 ({len(train_data)}) 不足以產生訓練樣本。跳過此次訓練。")


        # 預測未來走勢
        recent_data_for_lstm = past_df.tail(LSTM_LOOKBACK_DAYS)
        lstm_signal = 0 # 預設為0
        if lstm_trained and len(recent_data_for_lstm) >= LSTM_LOOKBACK_DAYS:
            try:
                lstm_signal = lstm.predict(recent_data_for_lstm)
            except Exception as e:
                print(f"錯誤 [{current_day}]: LSTM 預測失敗: {e}")
                lstm_signal = 0 # 預測失敗則訊號為0
        print(f"DEBUG [{current_day}]: LSTM trained: {lstm_trained}, LSTM signal: {lstm_signal}")

        # 判斷市場狀態
        # 確保用於市場狀態判斷的數據足夠長 (calculate_market_regime 預設 window=30)
        market_regime_window = 30
        if len(past_df) >= market_regime_window:
            regime = calculate_market_regime(past_df.tail(market_regime_window * 2)) # 給更長一點的數據以防邊界
        else:
            regime = "unknown" # 或其他預設值
        print(f"DEBUG [{current_day}]: Market regime: {regime}")
        
        # --- LLM Strategy Selection and Signal Generation ---
        # forced_trade_action = None  # Reset forced trade decision for the day # 已在迴圈開頭重置
        current_date_str_for_llm = current_day.strftime('%Y-%m-%d')
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
            days_current_strategy_active = (current_day - current_strategy_start_date).days
        
        print(f"DEBUG [{current_day}]: Current active strategy: {current_active_strategy_name}, active for {days_current_strategy_active} days. Min duration: {MIN_STRATEGY_DURATION_DAYS}")

        # 在呼叫 LLM 之前獲取 PNL 數據
        current_last_trade_pnl = simulator.get_last_trade_pnl() # Changed to method call
        current_cumulative_pnl = simulator.get_cumulative_pnl() # Changed to method call

        # --- LLM 決策 ---
        todays_recommended_signal_for_llm = None # Default to None
        # Try to get current strategy's signal to inform forced trade LLM
        if USE_GEMINI_STRATEGY_SELECTION and current_active_strategy_name and current_active_strategy_name in strategy_classes:
            print(f"DEBUG [{current_day}]: Pre-calculating signal for '{current_active_strategy_name}' to inform forced trade LLM.")
            TempStrategyClass = strategy_classes[current_active_strategy_name]
            # Prepare data for this temporary signal generation
            temp_df_for_signal = full_df[full_df['date'] < current_day].copy()
            
            if 'Close' in temp_df_for_signal.columns:
                temp_df_for_signal["MA"] = temp_df_for_signal["Close"].rolling(window=20).mean()
                temp_df_for_signal["RSI"] = compute_rsi(temp_df_for_signal["Close"]) # compute_rsi is defined in main()
            else:
                print(f"WARNING [{current_day}]: 'Close' column missing in temp_df_for_signal for LLM pre-signal.")

            if current_active_strategy_name == "TrendStrategy":
                if not temp_df_for_signal.empty and 'Prediction' not in temp_df_for_signal.columns: # Avoid overwriting if already there
                    # Ensure the last row exists before trying to write to .loc
                    if not temp_df_for_signal.empty:
                         temp_df_for_signal.loc[temp_df_for_signal.index[-1], 'Prediction'] = lstm_signal
                elif temp_df_for_signal.empty:
                    print(f"WARNING [{current_day}]: temp_df_for_signal is empty, cannot attach LSTM prediction for TrendStrategy (LLM pre-signal).")

            temp_strategy_instance = TempStrategyClass(params=current_active_strategy_params if current_active_strategy_params else {})
            min_lookback_temp = getattr(temp_strategy_instance, 'MIN_LOOKBACK', MAX_LOOKBACK_PERIOD)

            if len(temp_df_for_signal) >= min_lookback_temp:
                temp_pos_status = "Long" if simulator.current_position > 0 else "Short" if simulator.current_position < 0 else None
                try:
                    raw_signal = temp_strategy_instance.generate_signal(
                        data_slice=temp_df_for_signal,
                        current_index=current_day, # Consistent with later signal generation
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

                    print(f"DEBUG [{current_day}]: Pre-calculated signal for LLM: {raw_signal} -> {todays_recommended_signal_for_llm}")
                except Exception as e_sig:
                    print(f"DEBUG [{current_day}]: Error generating temp signal for LLM: {e_sig}")
                    todays_recommended_signal_for_llm = 0 # Default to neutral on error
            else:
                print(f"DEBUG [{current_day}]: Not enough data for temp signal for LLM (needs {min_lookback_temp}, got {len(temp_df_for_signal)}).")
                todays_recommended_signal_for_llm = 0


        # 優先處理強制交易的判斷
        if USE_GEMINI_STRATEGY_SELECTION and ENABLE_FORCED_TRADING and days_since_last_trade >= MAX_DAYS_NO_TRADE:
            print(f"DEBUG [{current_day}]: Potential forced trade scenario. Days since last trade: {days_since_last_trade}. Calling LLM for forced trade decision.")
            
            forced_decision, _, llm_context_forced = select_strategy_and_params(
                current_date=current_date_str_for_llm,
                price_data_str=price_data_for_llm_str,
                news_sentiment_summary=news_sentiment_summary_for_llm,
                available_strategies=available_strategies_for_llm, 
                lstm_signal=lstm_signal,
                days_since_last_trade=days_since_last_trade,
                max_days_no_trade=MAX_DAYS_NO_TRADE,
                is_forced_trade_scenario=True,
                recommended_strategy_name=current_active_strategy_name,
                recommended_strategy_signal=todays_recommended_signal_for_llm # Pass the generated signal
            )
            print(f"DEBUG [{current_day}]: LLM forced trade decision: {forced_decision}")

            if forced_decision in ["ForcedBuy", "ForcedShort"]:
                forced_trade_action = forced_decision
                current_active_strategy_name = None 
                current_active_strategy_params = {}
                current_strategy_start_date = None
                print(f"DEBUG [{current_day}]: LLM decided a forced trade: {forced_trade_action}. Regular strategy target reset.")
            elif forced_decision == "AbstainForceTrade":
                forced_trade_action = "AbstainForceTrade" 
                print(f"DEBUG [{current_day}]: LLM decided to AbstainForceTrade.")
            else: 
                print(f"WARNING [{current_day}]: LLM returned unexpected forced trade decision: {forced_decision}. Defaulting to AbstainForceTrade.")
                forced_trade_action = "AbstainForceTrade"
        
        # 如果沒有執行強制交易 (LLM決定不強制 或 未達到強制交易天數)
        # 則進行常規的策略目標選擇/持續判斷
        if USE_GEMINI_STRATEGY_SELECTION and (not forced_trade_action or forced_trade_action == "AbstainForceTrade"):
            if forced_trade_action == "AbstainForceTrade":
                print(f"DEBUG [{current_day}]: LLM abstained from forced trade. Proceeding with regular strategy target selection/evaluation.")

            should_select_new_strategy_target = False
            if not current_active_strategy_name: 
                print(f"DEBUG [{current_day}]: No active strategy. LLM will select an initial strategy target.")
                should_select_new_strategy_target = True
            elif days_current_strategy_active >= MIN_STRATEGY_DURATION_DAYS:
                print(f"DEBUG [{current_day}]: Current strategy '{current_active_strategy_name}' has met min duration ({days_current_strategy_active} >= {MIN_STRATEGY_DURATION_DAYS}). LLM can select a new target.")
                should_select_new_strategy_target = True
            # else: # No change to should_select_new_strategy_target, it remains False
            #    print(f"DEBUG [{current_day}]: Current strategy '{current_active_strategy_name}' active for {days_current_strategy_active} days, continues as target (min duration {MIN_STRATEGY_DURATION_DAYS} not met).")


            if should_select_new_strategy_target:
                print(f"DEBUG [{current_day}]: Calling LLM to select/revise strategy target (is_forced_trade_scenario=False).")
                strategy_name, strategy_params, llm_context_strat = select_strategy_and_params(
                    current_date=current_date_str_for_llm,
                    price_data_str=price_data_for_llm_str,
                    news_sentiment_summary=news_sentiment_summary_for_llm,
                    available_strategies=available_strategies_for_llm,
                    lstm_signal=lstm_signal,
                    days_since_last_trade=days_since_last_trade, 
                    max_days_no_trade=MAX_DAYS_NO_TRADE, 
                    is_forced_trade_scenario=False, 
                    current_active_strategy_name=current_active_strategy_name,
                    current_strategy_days_active=days_current_strategy_active,
                    min_strategy_duration=MIN_STRATEGY_DURATION_DAYS,
                    last_trade_pnl=current_last_trade_pnl,
                    cumulative_pnl=current_cumulative_pnl
                )
                print(f"DEBUG [{current_day}]: LLM (for strategy target) returned: name='{strategy_name}', params={strategy_params}")

                if strategy_name in strategy_classes:
                    if current_active_strategy_name != strategy_name or current_active_strategy_params != (strategy_params or {}):
                        print(f"DEBUG [{current_day}]: LLM selected new/updated strategy target: {strategy_name} with params {strategy_params}")
                        current_active_strategy_name = strategy_name
                        current_active_strategy_params = strategy_params or {}
                        current_strategy_start_date = current_day 
                        strategy_repeat_count = 0 # Reset repeat count on new strategy
                        last_llm_selected_strategy = strategy_name # Track last selected strategy
                    # else:
                        # print(f"DEBUG [{current_day}]: LLM re-selected same strategy target '{strategy_name}' with same params. Duration counter continues.")
                elif strategy_name == "Abstain" or strategy_name is None:
                    print(f"DEBUG [{current_day}]: LLM (for strategy target) returned '{strategy_name}'. No change to current strategy target '{current_active_strategy_name}'.")
                else: 
                    print(f"WARNING [{current_day}]: LLM (for strategy target) returned invalid strategy: '{strategy_name}'. Resetting active strategy.")
                    current_active_strategy_name = None
                    current_active_strategy_params = {}
                    current_strategy_start_date = None
            elif current_active_strategy_name: 
                 print(f"DEBUG [{current_day}]: Continuing with strategy target: {current_active_strategy_name} with params {current_active_strategy_params} (min duration not met or no need to change).")
        
        elif not USE_GEMINI_STRATEGY_SELECTION:
            print(f"DEBUG [{current_day}]: USE_GEMINI_STRATEGY_SELECTION is False. No LLM calls for strategy selection or forced trading.")
            if not current_active_strategy_name: 
                current_active_strategy_name = "TrendStrategy" 
                current_active_strategy_params = {}
                current_strategy_start_date = current_day
                print(f"DEBUG [{current_day}]: Default strategy '{current_active_strategy_name}' activated.")

        # --- Signal Generation from active strategy (if not a forced trade by LLM and a strategy is active) ---
        log_date_str = current_day.strftime('%Y-%m-%d') 

        # 只有在沒有強制交易動作，且有活躍的常規策略時，才由常規策略產生訊號
        if not forced_trade_action or forced_trade_action == "AbstainForceTrade":
            if current_active_strategy_name and current_active_strategy_name in strategy_classes:
                if prev_strategy_logging_name != current_active_strategy_name:
                    print(f"DEBUG [{current_day}]: Attempting to use strategy: {current_active_strategy_name} with params: {current_active_strategy_params}")
                    prev_strategy_logging_name = current_active_strategy_name
                
                StrategyClass = strategy_classes[current_active_strategy_name]
                df_for_signal_gen = full_df[full_df['date'] < current_day].copy()

                if 'Close' in df_for_signal_gen.columns:
                    df_for_signal_gen["MA"] = df_for_signal_gen["Close"].rolling(window=20).mean() # Corrected: removed backslash
                    df_for_signal_gen["RSI"] = compute_rsi(df_for_signal_gen["Close"])      # Corrected: removed backslash
                else:
                    print(f"WARNING [{current_day}]: 'Close' column missing in df_for_signal_gen. MA/RSI might be incorrect for {current_active_strategy_name}.")

                if current_active_strategy_name == "TrendStrategy":
                    if not df_for_signal_gen.empty:
                        df_for_signal_gen.loc[df_for_signal_gen.index[-1], 'Prediction'] = lstm_signal
                    else:
                        print(f"WARNING [{current_day}]: df_for_signal_gen is empty, cannot attach LSTM prediction for TrendStrategy.")
            
                current_active_strategy = StrategyClass(**current_active_strategy_params if current_active_strategy_params else {})
                min_lookback_needed_by_strategy = getattr(current_active_strategy, 'MIN_LOOKBACK', MAX_LOOKBACK_PERIOD)

                if len(df_for_signal_gen) >= min_lookback_needed_by_strategy:
                    print(f"LOG [{log_date_str}] [{STOCK_SYMBOL}] Strategy: {current_active_strategy_name}, Params: {current_active_strategy_params}, Data len: {len(df_for_signal_gen)}")
                    try:
                        position_status = None
                        if simulator.current_position > 0: position_status = "Long"
                        elif simulator.current_position < 0: position_status = "Short"
                        
                        print(f"DEBUG [{current_day}]: Calling {current_active_strategy_name}.generate_signal with df_for_signal_gen (len {len(df_for_signal_gen)}), current_index={current_day}, position='{position_status}'")
                        signal_action_str = current_active_strategy.generate_signal(
                            data_slice=df_for_signal_gen,
                            current_index=current_day,
                            position=position_status
                        )
                        print(f"LOG [{log_date_str}] [{STOCK_SYMBOL}] Strategy Signal Output ({current_active_strategy_name}): '{signal_action_str}'")
                    except Exception as e:
                        print(f"ERROR [{log_date_str}] [{STOCK_SYMBOL}] Error generating signal from strategy {current_active_strategy_name}: {e}")
                        signal_action_str = "HOLD" # Default to HOLD on error
                else:
                    print(f"WARNING [{current_day}]: Not enough data for strategy {current_active_strategy_name} (needs {min_lookback_needed_by_strategy}, got {len(df_for_signal_gen)}). Holding.")
                    signal_action_str = "HOLD"
        elif forced_trade_action:
             print(f"LOG [{log_date_str}] [{STOCK_SYMBOL}] LLM forced action {forced_trade_action} will be processed. Regular strategy skipped.")
        else: # No active strategy, or strategy not in classes
            print(f"LOG [{log_date_str}] [{STOCK_SYMBOL}] No valid active strategy or forced action. Holding.")


        # --- Determine Final Trade Action and Execute ---
        final_trade_action = 0 
        # quantity_to_trade will be calculated later

        # 1. 優先處理 LLM 強制交易決策
        if forced_trade_action == "ForcedBuy": # This variable is set after LLM call for forced trade
            final_trade_action = 2 # 強制買入
            print(f"DEBUG [{current_day}]: LLM decided ForcedBuy. Quantity to be calculated at execution.")
        elif forced_trade_action == "ForcedShort":
            final_trade_action = -2 # 強制賣出（放空）
            print(f"DEBUG [{current_day}]: LLM decided ForcedShort. Quantity to be calculated at execution.")
        elif forced_trade_action == "AbstainForceTrade":
            final_trade_action = 0 
            print(f"DEBUG [{current_day}]: AbstainForceTrade action by LLM. No trade.")
        
        # 2. 如果沒有 LLM 強制交易 (或LLM決定不強制)，則處理一般策略訊號
        elif signal_action_str: # This elif implies not a ForcedBuy/ForcedShort by LLM
            # 統一格式，首字母大寫
            action = str(signal_action_str).strip().capitalize()
            if action == "Buy":
                final_trade_action = 1
                print(f"DEBUG [{current_day}]: Regular BUY signal. Quantity to be calculated at execution.")
            elif action == "Sell" or action == "Short":
                final_trade_action = -1
                print(f"DEBUG [{current_day}]: Regular SELL/SHORT signal. Quantity to be calculated at execution.")
            elif action == "Cover":
                # Cover 視為平空單，等同於 Buy
                final_trade_action = 1
                print(f"DEBUG [{current_day}]: Regular COVER signal (close short). Quantity to be calculated at execution.")
            elif action == "Hold":
                final_trade_action = 0
                print(f"DEBUG [{current_day}]: Regular HOLD signal.")
            else:
                print(f"WARNING [{current_day}]: Unknown signal_action_str: '{signal_action_str}'. Holding.")
                final_trade_action = 0
        # else: final_trade_action remains 0 if no forced action and no signal_action_str

        # Execute trade if data for today is available
        if not today_row.empty and 'Open' in today_row.columns and 'Close' in today_row.columns:
            current_price_open = today_row['Open'].iloc[0]
            current_price_close = today_row['Close'].iloc[0]

            trade_log_entry = None 
            quantity_to_trade = 0 
            trade_type_label = "Unknown"

            if final_trade_action != 0:
                available_cash = simulator.cash # Use direct attribute access

                if abs(final_trade_action) == 2: # Forced Trade (ForcedBuy=2, ForcedShort=-2)
                    trade_type_label = "Forced"
                    if available_cash > 0 and FORCED_TRADE_CAPITAL_ALLOCATION > 0 and current_price_open > 0:
                        quantity_to_trade = int((available_cash * FORCED_TRADE_CAPITAL_ALLOCATION) / current_price_open)
                        quantity_to_trade = simulator._adjust_quantity(quantity_to_trade)
                    
                    if final_trade_action == -2 and not ALLOW_SHORT_SELLING:
                        print(f"LOG [{log_date_str}] [{STOCK_SYMBOL}] Forced Short signal by LLM ignored: Short selling disabled.")
                        quantity_to_trade = 0 
               
                elif abs(final_trade_action) == 1: # Regular Strategy Trade (Buy=1, Sell=-1)
                    trade_type_label = "Regular"
                    if final_trade_action == 1: # Buy
                        if available_cash > 0 and REGULAR_TRADE_CAPITAL_ALLOCATION > 0 and current_price_open > 0:
                            quantity_to_trade = int((available_cash * REGULAR_TRADE_CAPITAL_ALLOCATION) / current_price_open)
                            quantity_to_trade = simulator._adjust_quantity(quantity_to_trade)
                    elif final_trade_action == -1: # Sell (to close long or open short)
                        if simulator.current_position > 0: # Closing existing long position
                            quantity_to_trade = simulator.current_position_quantity 
                        elif ALLOW_SHORT_SELLING: # Opening new short position
                            if available_cash > 0 and REGULAR_TRADE_CAPITAL_ALLOCATION > 0 and current_price_open > 0:
                                quantity_to_trade = int((available_cash * REGULAR_TRADE_CAPITAL_ALLOCATION) / current_price_open)
                                quantity_to_trade = simulator._adjust_quantity(quantity_to_trade)
                        else: # Not allowed to short and no long position to close
                            print(f"LOG [{log_date_str}] [{STOCK_SYMBOL}] Strategy Sell (to open short) ignored: Short selling disabled and no long position.")
                            quantity_to_trade = 0
               
                if quantity_to_trade <= 0:
                    if not (final_trade_action == -1 and simulator.current_position > 0 and quantity_to_trade > 0):
                        if final_trade_action != 0:
                             print(f"DEBUG [{log_date_str}] Action {final_trade_action} intended, but calculated quantity is {quantity_to_trade}. No trade executed.")
                        final_trade_action = 0 
               
                if final_trade_action != 0 and quantity_to_trade > 0:
                    if final_trade_action == 1 or final_trade_action == 2: # Buy or ForcedBuy
                        print(f"DEBUG [{log_date_str}] Attempting BUY: Price={current_price_open}, Qty={quantity_to_trade}, Type={trade_type_label}") # Corrected f-string
                        trade_log_entry = simulator.buy(price=current_price_open, quantity=quantity_to_trade, date_to_log=current_day, trade_type=trade_type_label)
                    elif final_trade_action == -1 or final_trade_action == -2: # Sell or ForcedShort
                        print(f"DEBUG [{log_date_str}] Attempting SELL: Price={current_price_open}, Qty={quantity_to_trade}, Type={trade_type_label}") # Corrected f-string
                        trade_log_entry = simulator.sell(price=current_price_open, quantity=quantity_to_trade, date_to_log=current_day, trade_type=trade_type_label)

            if trade_log_entry:
                trade_logs.append(trade_log_entry)
                trade_occurred_today = True
                print(f"LOG [{log_date_str}] [{STOCK_SYMBOL}] Trade Executed: {trade_log_entry}")
            
            # Daily EOD updates: portfolio value, stop-loss/take-profit checks
            simulator.update_portfolio_value(current_price_close)
            
            sl_tp_log_entries = [] 
            if not simulator.is_forced_trade_active() and simulator.current_position != 0 : 
                 sl_tp_log_entry = simulator.check_stop_loss_take_profit(current_price_close, date_to_log=current_day)
                 if sl_tp_log_entry:
                    sl_tp_log_entries.append(sl_tp_log_entry)

            forced_closure_log_entries = []
            if ENABLE_FORCED_TRADING and simulator.is_forced_trade_active():
                forced_closure_log_entry = simulator.check_forced_trade_closure(current_price_close, date_to_log=current_day)
                if forced_closure_log_entry:
                    forced_closure_log_entries.append(forced_closure_log_entry)

            for log_entry in sl_tp_log_entries + forced_closure_log_entries:
                trade_logs.append(log_entry)
                trade_occurred_today = True 
                print(f"LOG [{log_date_str}] [{STOCK_SYMBOL}] Position Closed (SL/TP/Forced): {log_entry}")

            # Update days_since_last_trade based on whether any trade activity occurred THIS day
            if trade_occurred_today:
                days_since_last_trade = 0
            else:
                days_since_last_trade += 1
            
            current_total_value = simulator.cash + simulator.portfolio_value
            daily_results_df.loc[len(daily_results_df)] = {'date': current_day, 'capital': current_total_value}
            print(f"DEBUG [{log_date_str}] EOD. Cash: {simulator.cash:.2f}, Portfolio Value: {simulator.portfolio_value:.2f}, Total: {current_total_value:.2f}, Position: {simulator.current_position}, Days no trade: {days_since_last_trade}")

        else: 
            print(f"DEBUG [{current_day}]: Skipping trade execution and daily EOD updates due to missing Open/Close price for the day.")
            days_since_last_trade += 1 # Still a day of no trading activity
            if not daily_results_df.empty:
                last_capital = daily_results_df['capital'].iloc[-1]
                daily_results_df.loc[len(daily_results_df)] = {'date': current_day, 'capital': last_capital}
                print(f"DEBUG [{log_date_str}] Missing price data. Carrying forward last capital: {last_capital:.2f}. Days no trade: {days_since_last_trade}")
            else:
                 daily_results_df.loc[len(daily_results_df)] = {'date': current_day, 'capital': INITIAL_CAPITAL} 
                 print(f"DEBUG [{log_date_str}] Missing price data & no prior capital. Recording initial capital: {INITIAL_CAPITAL:.2f}. Days no trade: {days_since_last_trade}")

        current_day += timedelta(days=1)
        # --- END 日期迴圈 ---

    # === 回測結束後的處理 ===
    # 確保最後的部位被平倉 (如果 trade_simulator.py 中的 simulate 方法沒有處理)
    # simulator.simulate() 方法末尾已經加入了平倉邏輯，此處可能無需重複
    # 但如果直接調用 simulator 的 buy/sell 等方法，則需要確保此處有平倉
    if simulator.current_position != 0 and not df.empty:
        last_day_data = df.iloc[-1] # Use the last day from the backtest period df
        last_close_price = simulator._round_price(last_day_data["Close"])
        last_date = last_day_data["date"] # This should be a date object
        logging.info(f"SIMULATOR: Main loop end. Closing remaining position at {last_close_price} on {last_date}")
        final_log = None
        if simulator.current_position > 0: # Long position
            final_log = simulator.sell(last_close_price, abs(simulator.current_position), date_to_log=last_date, trade_type="EndOfLoop")
        elif simulator.current_position < 0: # Short position
            final_log = simulator.buy(last_close_price, abs(simulator.current_position), date_to_log=last_date, trade_type="EndOfLoop")
        if final_log:
            trade_logs.append(final_log)
            # Update daily capital for the very last day after closing
            simulator.update_portfolio_value(last_close_price) # Should be 0 after closing
            current_total_capital = simulator.cash + simulator.portfolio_value
            # Check if last_date already exists, if so, update, else append
            found_last_date_capital = False
            for record in simulator.daily_capital:
                if record['date'] == last_date:
                    record['capital'] = current_total_capital
                    found_last_date_capital = True
                    break
            if not found_last_date_capital:
                 simulator.daily_capital.append({"date": last_date, "capital": current_total_capital})


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
            trade_log_df=report_trade_log_df.copy(), # 傳遞副本
            strategy_name=strategy_name_for_report,
            initial_capital=INITIAL_CAPITAL
        )

        if report_df is not None and not report_df.empty:
            csv_filename = f"{strategy_name_for_report}_monthly_performance.csv"
            report_df.to_csv(csv_filename, index=True)
            logging.info(f"Monthly performance report for {strategy_name_for_report} saved to {csv_filename}")
            print(f"--- Monthly Performance Report for {strategy_name_for_report} (tail) ---")
            print(report_df.tail())
            if summary_df is not None and not summary_df.empty:
                summary_filename = f"{strategy_name_for_report}_performance_summary.xlsx"
                # summary_df.to_csv(summary_filename, index=False)
                # logging.info(f"Performance summary for {strategy_name_for_report} saved to {summary_filename}")
                # 輸出到 Excel 的同一個檔案的不同工作表
                with pd.ExcelWriter(summary_filename, engine='openpyxl', mode='a' if os.path.exists(summary_filename) else 'w') as writer:
                    summary_df.to_excel(writer, sheet_name=f"{strategy_name_for_report}_Summary", index=False)
                logging.info(f"Performance summary for {strategy_name_for_report} appended to {summary_filename}")

        else:
            logging.info(f"Could not generate monthly performance report for {strategy_name_for_report} (generated report_df is empty or None).")

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
    # 設定日誌級別
    logging.basicConfig(level=logging.INFO, format='%(levelname)s [%(asctime)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # logging.getLogger().setLevel(logging.DEBUG) # Uncomment for more detailed debug messages from all modules
    
    # 匯入 os 以便檢查檔案是否存在 (用於 Excel 寫入)
    import os

    main()
