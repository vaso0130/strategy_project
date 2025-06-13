import pandas as pd
from config import *
from data_loader import load_price_data
from lstm_model import LSTMPredictor
from market_regime import calculate_market_regime
from llm_decision_engine import select_strategy_and_params
from trade_simulator import TradeSimulator
from utils.metrics import generate_monthly_report
from strategies import TrendStrategy, RangeStrategy, BreakoutStrategy, VolumePriceStrategy # 確保匯入類別
from optimizer import StrategyOptimizer

from datetime import timedelta

# === 載入資料 ===
# 下載更久以前的資料以確保 LSTM 能取得足夠的訓練樣本
# trading_days_needed_for_lstm_pretrain = LSTM_TRAIN_WINDOW + LSTM_LOOKBACK_DAYS
# # Estimate calendar days: multiply by ~1.4-1.5 to account for non-trading days (weekends, holidays)
# # Using 1.5 as a slightly safer multiplier. (252 trading days / 365.25 calendar days approx 0.69. Inverse is 1.45)
# calendar_days_for_lstm_pretrain = int(trading_days_needed_for_lstm_pretrain * 1.5) # Increased buffer
# real_start = (
#     pd.to_datetime(START_DATE)
#     - timedelta(days=calendar_days_for_lstm_pretrain + 5) # +5 for a bit more buffer and non-overlapping
# ).strftime("%Y-%m-%d")

# 更保守的估计：假设一年约250个交易日，需要 N 个交易日，则大约需要 N/250 年。
# 一年365天。所以需要 N * (365/250) 个日历天。365/250 = 1.46
required_trading_days = LSTM_TRAIN_WINDOW + LSTM_LOOKBACK_DAYS
estimated_calendar_days = int(required_trading_days * 1.5) + 5 # 1.5 作為乘數，額外5天緩衝

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
    print("[警告] load_price_data 回傳空的 DataFrame。")


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


# === 初始化模擬器 ===
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
    forced_trade_capital_allocation=FORCED_TRADE_CAPITAL_ALLOCATION
)

# 修改：儲存策略類別本身，而不是實例
strategy_classes = {
    "TrendStrategy": TrendStrategy, # 注意：沒有 ()
    "RangeStrategy": RangeStrategy,
    "BreakoutStrategy": BreakoutStrategy,
    "VolumePriceStrategy": VolumePriceStrategy
}


# === 建立回測資料框架 ===
# Yahoo Finance 下載的日期時間包含時區偏移，與我們以日期為單位的迴圈相比
# 可能出現 00:00 與 01:00 的差異，導致找不到當天的資料。
# 因此僅保留日期部分進行比對。
df['date'] = pd.to_datetime(df['date']).dt.date
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
    rs = gain / loss
    return 100 - (100 / (1 + rs))

if 'Close' in df.columns:
    df["RSI"] = compute_rsi(df["Close"])
else:
    print("[警告] df 中缺少 'Close' 欄位，無法計算 RSI。")

print("DEBUG: df after initial processing and feature engineering:")
print(df.head(3))
print(df.tail(3))
print(df.info())
print("DEBUG: Null values in df after feature engineering:")
print(df.isnull().sum())

# Initialize the main loop's lstm_trained flag based on pre-training result
lstm_trained = lstm.is_trained # 正確初始化 lstm_trained

# === 策略參數優化 ===
if ENABLE_STRATEGY_OPTIMIZATION:
    print("[資訊] 啟用策略參數優化")
    param_grids = {
        "TrendStrategy": { # TrendStrategy 目前使用 main.py 計算的固定 MA(20)
                           # 如果要讓 ma_period 可優化，TrendStrategy 需修改以接受並使用此參數
            "rsi_low": [25, 30, 35, 40],
            "rsi_high": [65, 70, 75, 80],
            # "ma_period": [10, 20] # 暫時註解，因為 TrendStrategy 未使用
        },
        "RangeStrategy": {
            "window": [10, 15, 20],
            "rsi_low": [25, 30, 35],
            "rsi_high": [65, 70, 75],
        },
        "BreakoutStrategy": {
            "window": [10, 15, 20],
            "rsi_low": [30, 40, 45],
            "rsi_high": [60, 65, 70]
        },
        "VolumePriceStrategy": { # VolumePriceStrategy 內部使用固定的 ma_short(5)
                                 # 如果要讓 ma_short_period 可優化，VolumePriceStrategy 需修改
            "volume_ratio": [1.2, 1.5, 1.8],
            "rsi_low": [30, 35, 40],
            "rsi_high": [60, 65, 70],
            # "ma_short_period": [5, 10] # 暫時註解，因為 VolumePriceStrategy 未使用
        },
    }

    eval_map = {
        "win_rate": lambda m: m["win_rate"],
        "total_return": lambda m: m["total_return"],
        "sharpe": lambda m: m.get("sharpe", 0),
    }
    evaluator = eval_map.get(STRATEGY_EVALUATOR, lambda m: m.get("sharpe", 0))
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


all_signals_for_simulation = []
prev_strategy = None

while current_day <= pd.to_datetime(END_DATE).date():
    print(f"\\nDEBUG: ----- Processing Day: {current_day} -----")
    past_df = df[df['date'] < current_day].copy()
    today_row = df[df['date'] == current_day]

    if len(today_row) == 0:
        print(f"DEBUG [{current_day}]: No data for today. Skipping.")
        current_day += timedelta(days=1)
        continue

    # 確保 past_df 至少有 LSTM_LOOKBACK_DAYS 天的數據用於 LSTM 預測
    # 並且策略本身也需要足夠的歷史數據 (例如計算 MA, RSI 等)
    # 這裡的 LSTM_LOOKBACK_DAYS 是一個較為通用的最小回溯期檢查
    min_hist_days_for_strat_features = 20 # 例如 MA20 需要至少20天
    if len(past_df) < max(LSTM_LOOKBACK_DAYS, min_hist_days_for_strat_features):
        print(f"DEBUG [{current_day}]: Insufficient past_df data (len: {len(past_df)}). Needs at least {max(LSTM_LOOKBACK_DAYS, min_hist_days_for_strat_features)}. Skipping.")
        current_day += timedelta(days=1)
        continue
    
    # retrain LSTM 模型
    if len(past_df) >= LSTM_TRAIN_WINDOW and \
       ((not lstm_trained) or (current_day - retrain_day).days >= LSTM_RETRAIN_INTERVAL):
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

    dummy_metrics = {name: {"sharpe": 0.5, "win_rate": 0.6, "bias": 1} for name in strategy_classes}
    
    # --- 強制交易決策 ---
    llm_context_for_forced_trade = False
    if ENABLE_FORCED_TRADING and days_since_last_trade >= MAX_DAYS_NO_TRADE:
        print(f"DEBUG [{current_day}]: 連續 {days_since_last_trade} 天無交易，達到 MAX_DAYS_NO_TRADE ({MAX_DAYS_NO_TRADE})。考慮強制交易。")
        llm_context_for_forced_trade = True
        # LLM 決策引擎將處理此情境，並可能返回強制交易訊號 (2 或 -2)
        # select_strategy_and_params 需要能夠處理 is_forced_trade_scenario
        
    # 傳遞 is_forced_trade_scenario 給 LLM
    llm_result = select_strategy_and_params(
        regime, 
        lstm_signal, 
        dummy_metrics, 
        param_grids,
        is_forced_trade_scenario=llm_context_for_forced_trade # 新增參數
    )
    selected_name = llm_result["strategy"] # LLM 可能返回 "ForcedBuy", "ForcedShort", 或常規策略名
    selected_params = llm_result.get("params", {}) # ForcedTrade 可能沒有 params
    print(f"DEBUG [{current_day}]: LLM selected: {selected_name}, params: {selected_params}")

    signal_value_for_current_day_action = 0 # 預設為無行動

    if selected_name == "ForcedBuy":
        signal_value_for_current_day_action = 2
        print(f"DEBUG [{current_day}]: LLM 決定強制買入。")
    elif selected_name == "ForcedShort":
        signal_value_for_current_day_action = -2
        print(f"DEBUG [{current_day}]: LLM 決定強制做空。")
    elif selected_name == "AbstainForceTrade": # LLM 決定在強制情境下不交易
        signal_value_for_current_day_action = 0
        print(f"DEBUG [{current_day}]: LLM 決定在強制交易情境下不進行交易。")
    else: # 常規策略選擇
        strategy_class_constructor = strategy_classes.get(selected_name)
        if not strategy_class_constructor:
            print(f"錯誤 [{current_day}]: LLM 選擇的策略 '{selected_name}' 未在 strategy_classes 中定義。跳過本日。")
            current_day += timedelta(days=1)
            days_since_last_trade +=1 # 即使策略選擇錯誤，也算一天無交易
            continue
            
        strategy = strategy_class_constructor(**selected_params)
        print(f"DEBUG [{current_day}]: Instantiated strategy: {strategy}")
        
        if past_df.tail(5).isnull().any().any():
             print(f"DEBUG [{current_day}]: WARNING - NaNs found in recent past_df data passed to strategy (last 5 rows):")
             print(past_df[['date', 'Close', 'MA', 'RSI']].tail(5))

        signals_df = strategy.generate_signals(past_df)
        
        print(f"DEBUG [{current_day}]: Signals DataFrame from strategy (tail 3):")
        if not signals_df.empty:
            print(signals_df[['date', 'signal']].tail(3))
        else:
            print("Signals DataFrame is empty.")

        if not signals_df.empty and 'signal' in signals_df.columns:
            signal_value_for_current_day_action = signals_df['signal'].iloc[-1]
        else:
            print(f"DEBUG [{current_day}]: Strategy {selected_name} signals_df 格式不正確或為空。")
            
    print(f"DEBUG [{current_day}]: Signal value for current day action: {signal_value_for_current_day_action}")
    
    all_signals_for_simulation.append({
        'date': current_day,
        'signal': signal_value_for_current_day_action
    })

    # 更新無交易日計數器
    if signal_value_for_current_day_action != 0: # 如果有任何交易訊號 (包括強制交易)
        days_since_last_trade = 0
    else:
        days_since_last_trade += 1
    
    print(f"DEBUG [{current_day}]: Days since last trade: {days_since_last_trade}")

    if prev_strategy != selected_name and selected_name not in ["ForcedBuy", "ForcedShort", "AbstainForceTrade"]:
        print(f"[{current_day}] LLM 選擇策略: {selected_name} (參數: {selected_params})")
        prev_strategy = selected_name
    elif selected_name in ["ForcedBuy", "ForcedShort"]:
         print(f"[{current_day}] LLM 執行強制交易: {selected_name}")
         prev_strategy = selected_name # 也更新 prev_strategy 以便追蹤
    elif selected_name == "AbstainForceTrade":
        print(f"[{current_day}] LLM 在強制交易情境下選擇不動作。")
        # prev_strategy 保持不變或設為 None/Abstain，取決於是否想追蹤此狀態

    current_day += timedelta(days=1)

# --- 迴圈結束後，執行一次完整的模擬 ---
if all_signals_for_simulation:
    simulation_input_signals_df = pd.DataFrame(all_signals_for_simulation)
    # 確保 simulation_input_signals_df 的日期是唯一的，以防萬一
    simulation_input_signals_df = simulation_input_signals_df.drop_duplicates(subset=['date'], keep='last')
    
    print("\nDEBUG: --- Preparing for Simulation ---")
    print("DEBUG: simulation_input_signals_df head:")
    print(simulation_input_signals_df.head())
    print("DEBUG: simulation_input_signals_df tail:")
    print(simulation_input_signals_df.tail())
    print("DEBUG: Value counts for 'signal' in simulation_input_signals_df:")
    print(simulation_input_signals_df['signal'].value_counts())
    
    if not simulation_input_signals_df.empty:
        print(f"DEBUG: Dates range in simulation_input_signals_df: {simulation_input_signals_df['date'].min()} to {simulation_input_signals_df['date'].max()}")
    
    print("DEBUG: Main df passed to simulator (head):")
    # 確保傳遞給模擬器的 df 包含回測期間的數據
    sim_df = df[(df['date'] >= pd.to_datetime(START_DATE).date()) & (df['date'] <= pd.to_datetime(END_DATE).date())]
    # 或者，如果 simulator.simulate 可以處理包含 START_DATE 之前數據的 df，則直接傳遞 df
    # 假設 simulator.simulate 會根據 signal_df 的日期範圍來操作 df
    
    # 確保傳遞給模擬器的 df 包含 'Open', 'High', 'Low', 'Close'
    # 並且這些欄位在模擬期間是有效的
    required_cols = ['date', 'Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols):
        print(f"錯誤: 主 DataFrame 缺少模擬器所需欄位: {required_cols}")
        # 可以在此處終止或設定預設值
        trades_df = pd.DataFrame()
        daily_results_df = pd.DataFrame(columns=['date', 'capital']) # Ensure it's an empty DataFrame
        final_capital = INITIAL_CAPITAL
    else:
        trades_df, final_capital, daily_results_df_from_sim = simulator.simulate(df.copy(), simulation_input_signals_df) # MODIFIED: Unpack 3 values
        
        trade_logs.extend(trades_df.to_dict('records'))
        daily_results_df = daily_results_df_from_sim # MODIFIED: Assign directly
        # daily_results_list.extend(simulator.daily_capital) # REMOVED
else:
    print("[警告] 沒有訊號可供模擬。")
    trades_df = pd.DataFrame()
    # daily_results_df is already an empty DataFrame from initialization
    final_capital = INITIAL_CAPITAL

# === 產生報告 ===
# trade_log_df = pd.DataFrame(trade_logs) # trade_logs is already a list of dicts from simulator.trades

# Convert the list of trade dictionaries (from simulator.trades) to a DataFrame
raw_trades_df = pd.DataFrame(trade_logs)

# Process raw_trades_df to create round-trip trades for generate_monthly_report
if not raw_trades_df.empty:
    raw_trades_df['Date'] = pd.to_datetime(raw_trades_df['Date'])
    processed_trades = []
    # Group by TradeID to pair entries and exits
    for trade_id, group in raw_trades_df.groupby('TradeID'):
        if len(group) == 2: # Expecting one entry and one exit
            entry_trade = group[group['Action'].isin(['Buy', 'Short', 'ForcedBuy', 'ForcedShort'])].iloc[0]
            exit_trade = group[~group['Action'].isin(['Buy', 'Short', 'ForcedBuy', 'ForcedShort'])].iloc[0]
            
            side = 'long' if entry_trade['Action'] in ['Buy', 'ForcedBuy'] else 'short'
            
            # PNL is logged with the exit trade by the simulator
            processed_trades.append({
                'entry_date': entry_trade['Date'],
                'exit_date': exit_trade['Date'],
                'entry_price': entry_trade['Price'],
                'exit_price': exit_trade['Price'],
                'quantity': entry_trade['Quantity'], # Assuming quantity is same for entry and exit
                'side': side,
                'pnl': exit_trade['PNL'], # PNL from the exit trade
                'trade_id': trade_id
            })
        elif len(group) == 1: # Potentially an open trade closed at the end, or an error
            # This case might need more specific handling if it's a valid scenario
            # For now, we'll log it if it looks like a closing trade with PNL
            trade = group.iloc[0]
            if trade['PNL'] is not None and not pd.isna(trade['PNL']):
                 print(f"[警告] TradeID {trade_id} 只有一筆交易紀錄，但包含 PNL，可能為期末平倉。將嘗試記錄。")
                 # We need to infer entry details or skip. For now, skipping single legs for monthly report.
                 # To include, one might need to fetch entry from an earlier point or make assumptions.
                 pass # Not adding to processed_trades as entry_date/price is missing for a round trip

    trade_log_df_for_report = pd.DataFrame(processed_trades)
else:
    trade_log_df_for_report = pd.DataFrame(columns=[
        'entry_date', 'exit_date', 'entry_price', 'exit_price', 'quantity', 'side', 'pnl', 'trade_id'
    ])

print(f"\n回測結束 ({START_DATE} to {END_DATE})")
print(f"最終資產: {final_capital:,.0f} TWD")

# Use trade_log_df_for_report for generating the report
if not trade_log_df_for_report.empty and not daily_results_df.empty:
    try:
        # generate_monthly_report expects 'trade_log_df' as the argument name
        generate_monthly_report(daily_results_df, trade_log_df=trade_log_df_for_report, strategy_name=f"{STOCK_SYMBOL}_LLM_Dynamic_Strategy", initial_capital=INITIAL_CAPITAL)
        print(f"\n已產生交易報告: {STOCK_SYMBOL}_LLM_Dynamic_Strategy_monthly_report.xlsx")
        
        # Save the raw_trades_df (from simulator) as the detailed transaction log if needed
        if SAVE_RAW_TRADE_LOG and not raw_trades_df.empty:
            raw_trades_df.to_csv(f"{STOCK_SYMBOL}_LLM_Dynamic_Strategy_raw_trades.csv", index=False)
            print(f"已產生原始交易紀錄: {STOCK_SYMBOL}_LLM_Dynamic_Strategy_raw_trades.csv")

    except KeyError as e:
        print(f"錯誤：產生月報表時發生 KeyError: {e}。檢查傳遞給 generate_monthly_report 的 DataFrame 欄位。")
        print("DEBUG: trade_log_df_for_report columns:", trade_log_df_for_report.columns)
    except Exception as e:
        print(f"錯誤：產生月報表失敗: {e}")
elif trade_log_df_for_report.empty:
    print("\n沒有任何完整交易回合可供報告。")
    if SAVE_RAW_TRADE_LOG and not raw_trades_df.empty:
            raw_trades_df.to_csv(f"{STOCK_SYMBOL}_LLM_Dynamic_Strategy_raw_trades.csv", index=False)
            print(f"已產生原始交易紀錄 (無完整回合): {STOCK_SYMBOL}_LLM_Dynamic_Strategy_raw_trades.csv")
else: 
    print("\n沒有足夠的每日資產數據來產生報告。")
