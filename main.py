import pandas as pd
from config import *
from data_loader import load_price_data
from lstm_model import LSTMPredictor
from market_regime import calculate_market_regime
from llm_decision_engine import select_strategy_and_params
from trade_simulator import TradeSimulator
from utils.metrics import generate_monthly_report
from strategies import *  # 匯入所有策略類別
from optimizer import StrategyOptimizer

from datetime import timedelta

# === 載入資料 ===
# 下載更久以前的資料以確保 LSTM 能取得足夠的訓練樣本
real_start = (
    pd.to_datetime(START_DATE)
    - timedelta(days=LSTM_TRAIN_WINDOW + LSTM_LOOKBACK_DAYS)
).strftime("%Y-%m-%d")
df = load_price_data(symbol=STOCK_SYMBOL, start_date=real_start, end_date=END_DATE)

# === 初始化元件 ===
lstm = LSTMPredictor(lookback_days=LSTM_LOOKBACK_DAYS, predict_days=LSTM_PREDICT_DAYS, epochs=1)

# === 初始化模擬器 ===
simulator = TradeSimulator(
    initial_capital=INITIAL_CAPITAL,
    stop_loss=STOP_LOSS_THRESHOLD,
    allow_short=ALLOW_SHORT_SELLING
)

strategy_classes = {
    "TrendStrategy": TrendStrategy(),
    "RangeStrategy": RangeStrategy(),
    "BreakoutStrategy": BreakoutStrategy(),
    "VolumePriceStrategy": VolumePriceStrategy()
}


# === 建立回測資料框架 ===
# Yahoo Finance 下載的日期時間包含時區偏移，與我們以日期為單位的迴圈相比
# 可能出現 00:00 與 01:00 的差異，導致找不到當天的資料。
# 因此僅保留日期部分進行比對。
df['date'] = pd.to_datetime(df['date']).dt.date
df = df.sort_values("date").reset_index(drop=True)

# 補上策略可能使用的欄位名稱與指標
df["Close"] = df["close"]
df["Volume"] = df["volume"]
df["MA"] = df["Close"].rolling(window=20).mean()

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df["RSI"] = compute_rsi(df["Close"])

# === 策略參數優化 ===
if ENABLE_STRATEGY_OPTIMIZATION:
    print("[資訊] 啟用策略參數優化")
    param_grids = {
        "TrendStrategy": {"rsi_low": [30, 40], "rsi_high": [60, 70]},
        "RangeStrategy": {
            "window": [10, 20],
            "rsi_low": [45, 50],
            "rsi_high": [65, 70],
        },
        "BreakoutStrategy": {"window": [15, 20]},
        "VolumePriceStrategy": {"volume_ratio": [1.5, 2.0]},
    }

    eval_map = {
        "win_rate": lambda m: m["win_rate"],
        "total_return": lambda m: m["total_return"],
        "sharpe": lambda m: m.get("sharpe", 0),
    }
    evaluator = eval_map.get(STRATEGY_EVALUATOR, lambda m: m.get("sharpe", 0))
    pre_start = df[df["date"] < pd.to_datetime(START_DATE).date()]

    optimized = {}
    for name, strat in strategy_classes.items():
        grid = param_grids.get(name)
        if grid:
            opt = StrategyOptimizer(type(strat), grid, evaluator)
            result = opt.optimize(pre_start)
            best = result.get("best_params") or {}
            optimized[name] = type(strat)(**best)
            print(f"[優化] {name} 最佳參數: {best}")
        else:
            optimized[name] = strat
    strategy_classes = optimized

# 於迴圈開始前嘗試以歷史資料訓練 LSTM，若資料不足則待迴圈中再訓練
lstm_trained = False
initial_train = df[df['date'] < pd.to_datetime(START_DATE).date()].tail(LSTM_TRAIN_WINDOW)
if len(initial_train) >= LSTM_TRAIN_WINDOW:
    lstm.train(initial_train)
    lstm_trained = True

daily_results = []
trade_logs = []

# === 主流程 ===
current_day = pd.to_datetime(START_DATE).date()
retrain_day = current_day

while current_day <= pd.to_datetime(END_DATE).date():
    past_df = df[df['date'] < current_day]
    today_row = df[df['date'] == current_day]

    if len(today_row) == 0 or len(past_df) < LSTM_LOOKBACK_DAYS:
        current_day += timedelta(days=1)
        continue

    # retrain LSTM 模型（首次資料足夠時亦在此訓練）
    if len(past_df) >= LSTM_TRAIN_WINDOW and \
       ((not lstm_trained) or (current_day - retrain_day).days >= LSTM_RETRAIN_INTERVAL):
        train_data = past_df.tail(LSTM_TRAIN_WINDOW)
        lstm.train(train_data)
        retrain_day = current_day
        lstm_trained = True

    # 預測未來走勢
    recent_data = past_df.tail(LSTM_LOOKBACK_DAYS)
    lstm_signal = lstm.predict(recent_data) if lstm_trained else 0

    # 判斷市場狀態
    regime = calculate_market_regime(past_df.tail(60))

    # 計算每個策略的績效（這裡可換成真績效，先用空值）
    dummy_metrics = {name: {"sharpe": 0.5, "win_rate": 0.6, "bias": 1} for name in strategy_classes}
    
    # 讓 LLM 選策略＋參數
    llm_result = select_strategy_and_params(regime, lstm_signal, dummy_metrics, param_grids)
    selected_name = llm_result["strategy"]
    selected_params = llm_result["params"]

    # 動態建立策略物件
    strategy_class_constructor = strategy_classes[selected_name] # 直接取得類別
    strategy = strategy_class_constructor(**selected_params)
    
    # 產生訊號時，只給策略當前日期之前的資料，避免用到未來函數
    # 並且訊號應該是針對接下來的交易日
    # 這裡的 signals 應該是 DataFrame，包含 'date' 和 'signal'
    signals_df = strategy.generate_signals(past_df.copy()) # 傳遞過去的數據讓策略判斷

    # 執行模擬交易
    # simulator 需要的是包含當日開盤價等資訊的 df，以及策略產生的 signal_df
    # 假設 simulator.simulate 會處理好對應日期的交易
    # 我們需要傳遞整個歷史 df 和 signal_df 給 simulator，讓它內部按日期迭代
    # 或者，我們在每日迴圈中，只處理當日的訊號和交易

    # 簡化：假設 simulate 內部會處理好對齊
    # 實際上，simulator.simulate 應該在每日迴圈外被呼叫一次，傳入完整的 df 和 signals_df
    # 或者，每日呼叫，但 simulator 內部需要維護狀態

    # --- 每日模擬方式 ---
    # 取得當日訊號
    current_signal_row = signals_df[signals_df['date'] == current_day]
    signal_for_today = current_signal_row['signal'].iloc[0] if not current_signal_row.empty else 0

    # 模擬器處理當日交易 (這種方式 simulator 需要能逐日更新狀態)
    # 為了簡化，我們這裡先採用一次性模擬，然後從結果中提取當日交易
    # 但更常見的做法是 simulator.step(date, price, signal)

    # --- 為了與你現有 simulator 結構匹配，我們在迴圈外模擬 ---
    # 這部分邏輯需要調整，目前 simulator.simulate 預期接收完整的 df 和 signal_df
    # 我們先保持每日選擇策略，但模擬器在迴圈外執行

    # 這裡的 trade_logs 和 daily_results 應該在模擬器執行後才更新
    # 先收集每日訊號
    if 'all_signals_for_simulation' not in locals():
        all_signals_for_simulation = []
    
    # 儲存當日LLM選擇的策略所產生的訊號
    # 注意：signals_df 是基於 past_df 產生的，可能包含多天訊號
    # 我們只取 current_day 對應的訊號（如果策略設計是如此）
    # 或者，策略 generate_signals 只回傳下一日的訊號
    
    # 假設 strategy.generate_signals(past_df) 會回傳一個包含未來日期的訊號 DataFrame
    # 我們需要篩選出 current_day 的訊號
    day_specific_signal = signals_df[signals_df['date'] == current_day]
    if not day_specific_signal.empty:
         all_signals_for_simulation.append({
             'date': current_day,
             'signal': day_specific_signal['signal'].iloc[0],
             'open': today_row['open'].iloc[0], # 加入當日價格資訊給模擬器
             'close': today_row['close'].iloc[0],
             'high': today_row['high'].iloc[0],
             'low': today_row['low'].iloc[0]
         })


    if 'prev_strategy' not in locals() or prev_strategy != selected_name:
        print(f"[{current_day}] LLM 選擇策略: {selected_name} (參數: {selected_params})")
        prev_strategy = selected_name
    
    current_day += timedelta(days=1)
    # --- 每日迴圈結束 ---

# --- 迴圈結束後，執行一次完整的模擬 ---
if 'all_signals_for_simulation' in locals() and all_signals_for_simulation:
    simulation_input_signals_df = pd.DataFrame(all_signals_for_simulation)
    
    # 模擬器需要完整的價格數據 df 和訊號 df
    # 確保 df 包含 'open', 'high', 'low', 'close'
    # 這裡的 df 是最開始載入的完整價格數據
    trades_df, final_capital = simulator.simulate(df, simulation_input_signals_df)
    
    trade_logs.extend(trades_df.to_dict('records'))
    # daily_results 需要從 simulator.daily_capital 獲取
    daily_results_df = pd.DataFrame(simulator.daily_capital)
else:
    print("[警告] 沒有訊號可供模擬。")
    trades_df = pd.DataFrame()
    daily_results_df = pd.DataFrame(columns=['date', 'capital'])
    final_capital = INITIAL_CAPITAL


# === 產生報告 ===
trade_log_df = pd.DataFrame(trade_logs)
# daily_results_df 已在上面從 simulator.daily_capital 產生

# generate_monthly_report(daily_results_df, trade_log_df, "LLM_Dynamic_Strategy", INITIAL_CAPITAL)

# 輸出最終結果
print(f"\n回測結束 ({START_DATE} to {END_DATE})")
print(f"最終資產: {final_capital:,.0f} TWD")
# ... (其他總結指標) ...

if not trades_df.empty:
    generate_monthly_report(daily_results_df, trades_df, "LLM_Dynamic_Strategy", INITIAL_CAPITAL)
    print(f"\n已產生交易報告: LLM_Dynamic_Strategy_monthly_report.xlsx")
    print(f"已產生交易紀錄: LLM_Dynamic_Strategy_trades.csv")
else:
    print("\n沒有任何交易產生。")
