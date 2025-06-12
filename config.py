# === 股票與回測設定 ===
STOCK_SYMBOL = "0050.TW"
START_DATE = "2022-01-01"
END_DATE = "2025-06-01"

# === 回測參數設定 ===
INITIAL_CAPITAL = 1_000_000       # 初始資金（TWD）
STOP_LOSS_THRESHOLD = 0.08        # 停損線 8%
ALLOW_SHORT_SELLING = True        # 是否允許放空

# === LSTM 模型設定 ===
LSTM_TRAIN_WINDOW = 200           # LSTM 模型訓練所需的最小資料點數
LSTM_LOOKBACK_DAYS = 45           # LSTM 模型觀察的天數
LSTM_PREDICT_DAYS = 5            # 預測未來幾天平均價格
LSTM_RETRAIN_INTERVAL = 45        # 每幾天 retrain 一次
LSTM_EPOCHS = 20                  # 訓練 epoch 數
LSTM_LEARNING_RATE = 0.001        # 學習率

# === 策略優化設定 ===
ENABLE_STRATEGY_OPTIMIZATION = True
STRATEGY_EVALUATOR = "sharpe"     # 可選: "sharpe", "win_rate", "total_return"

# === Gemini LLM 設定 ===
USE_GEMINI_STRATEGY_SELECTION = True
GEMINI_MODEL_NAME = "gemini-1.5-flash"
