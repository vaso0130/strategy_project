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
real_start = (pd.to_datetime(START_DATE) - timedelta(days=LSTM_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
df = load_price_data(symbol=STOCK_SYMBOL, start_date=real_start, end_date=END_DATE)

# === 初始化元件 ===
lstm = LSTMPredictor(lookback_days=LSTM_LOOKBACK_DAYS, predict_days=LSTM_PREDICT_DAYS)
simulator = TradeSimulator(initial_capital=INITIAL_CAPITAL, stop_loss=STOP_LOSS_THRESHOLD, allow_short=ALLOW_SHORT_SELLING)
strategy_classes = {
    "TrendStrategy": TrendStrategy(),
    "RangeStrategy": RangeStrategy(),
    "BreakoutStrategy": BreakoutStrategy(),
    "VolumePriceStrategy": VolumePriceStrategy()
}

# === 建立回測資料框架 ===
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values("date").reset_index(drop=True)
daily_results = []
trade_logs = []

# === 主流程 ===
current_day = pd.to_datetime(START_DATE)
retrain_day = current_day

while current_day <= pd.to_datetime(END_DATE):
    past_df = df[df['date'] < current_day]
    today_row = df[df['date'] == current_day]

    if len(today_row) == 0 or len(past_df) < LSTM_LOOKBACK_DAYS:
        current_day += timedelta(days=1)
        continue

    # retrain LSTM 模型
    if (current_day - retrain_day).days >= LSTM_RETRAIN_INTERVAL:
        train_data = past_df.tail(LSTM_TRAIN_WINDOW) 
        lstm.train(train_data)
        retrain_day = current_day

    # 預測未來走勢
    recent_data = past_df.tail(LSTM_LOOKBACK_DAYS)
    lstm_signal = lstm.predict(recent_data)

    # 判斷市場狀態
    regime = calculate_market_regime(past_df.tail(60))

    # 計算每個策略的績效（這裡可換成真績效，先用空值）
    dummy_metrics = {name: {"sharpe": 0.5, "win_rate": 0.6, "bias": 1} for name in strategy_classes}
    selected_name = select_strategy(regime, lstm_signal, dummy_metrics)
    strategy = strategy_classes[selected_name]

    # 套用策略產出 signal
    concat_df = pd.concat([past_df, today_row], ignore_index=True)
    signals = strategy.generate_signals(concat_df)
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
