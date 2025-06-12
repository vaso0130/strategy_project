import pandas as pd
from config import *
from data_loader import load_price_data
from lstm_model import LSTMPredictor
from market_regime import calculate_market_regime
from llm_decision_engine import select_strategy
from trade_simulator import TradeSimulator
from utils.metrics import generate_monthly_report
from strategies import *  # 匯入所有策略類別

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
simulator = TradeSimulator(initial_capital=INITIAL_CAPITAL, stop_loss=STOP_LOSS_THRESHOLD, allow_short=ALLOW_SHORT_SELLING)
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
    selected_name = select_strategy(regime, lstm_signal, dummy_metrics)
    strategy = strategy_classes[selected_name]

    # 將 LSTM 預測結果附加到今日資料供策略參考
    today_row = today_row.copy()
    pred_text = {1: "up", -1: "down"}.get(lstm_signal)
    today_row["Prediction"] = pred_text
    signals = strategy.generate_signals(pd.concat([past_df, today_row], ignore_index=True))
    today_signal = signals.iloc[-1] if len(signals) > 0 else 0

    # 建立當日資料與 signal
    signal_df = pd.DataFrame({
        "date": [current_day],
        "close": today_row['close'].values,
        "signal": [today_signal]
    })

    # 模擬交易
    trades, capital = simulator.simulate(signal_df)
    trade_logs.extend(trades.to_dict('records'))
    daily_results.append({"date": current_day, "capital": capital})

    current_day += timedelta(days=1)

# === 匯出績效報告 ===
daily_df = pd.DataFrame(daily_results)
trade_df = pd.DataFrame(trade_logs)

generate_monthly_report(
    daily_df=daily_df,
    trade_log_df=trade_df,
    strategy_name="LSTMStrategy",
    initial_capital=INITIAL_CAPITAL
)
