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
LSTM_LOOKBACK_DAYS = 30           # LSTM 模型觀察的天數
LSTM_PREDICT_DAYS = 10             # 預測未來幾天平均價格
LSTM_RETRAIN_INTERVAL = 15        # 每幾天 retrain 一次
LSTM_EPOCHS = 20                  # 訓練 epoch 數
LSTM_LEARNING_RATE = 0.001        # 學習率
SHORT_QTY_CAP= 5000               # 放空時的最大數量

# === 策略優化設定 ===
ENABLE_STRATEGY_OPTIMIZATION = True
STRATEGY_EVALUATOR = "total_return"     # 可選: "sharpe", "win_rate", "total_return"

# === Gemini LLM 設定 ===
USE_GEMINI_STRATEGY_SELECTION = True
GEMINI_MODEL_NAME = "gemini-1.5-flash"

# Forced Trading Parameters
MAX_DAYS_NO_TRADE = 10  # 連續未交易N天後考慮強制進場
ENABLE_FORCED_TRADING = True # 是否啟用強制進場功能
FORCED_TRADE_TAKE_PROFIT_PCT = 0.05  # 強制進場交易的止盈百分比 (例如 5%)
FORCED_TRADE_STOP_LOSS_PCT = 0.02    # 強制進場交易的止損百分比 (例如 2%)
FORCED_TRADE_USE_TRAILING_STOP = False  # 強制進場交易是否使用移動止損
FORCED_TRADE_CAPITAL_ALLOCATION = 0.25  # 強制進場時使用的資金比例 (例如 25% 的現金)

# === 報告與紀錄設定 ===
SAVE_RAW_TRADE_LOG = True # 是否儲存原始的逐筆交易紀錄 CSV
