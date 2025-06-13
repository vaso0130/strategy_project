# === 股票與回測設定 ===
STOCK_SYMBOL = "3481.TW"
START_DATE = "2022-01-01"
END_DATE = "2022-10-01"

# === 回測參數設定 ===
INITIAL_CAPITAL = 1_000_000       # 初始資金（TWD）
STOP_LOSS_THRESHOLD = 0.08        # 停損線 8%
ALLOW_SHORT_SELLING = True        # 是否允許放空

# === LSTM 模型設定 ===
LSTM_TRAIN_WINDOW = 200           # LSTM 模型訓練所需的最小資料點數
LSTM_LOOKBACK_DAYS = 20           # LSTM 模型觀察的天數
LSTM_PREDICT_DAYS = 10             # 預測未來幾天平均價格
LSTM_RETRAIN_INTERVAL = 15        # 每幾天 retrain 一次
LSTM_EPOCHS = 25                  # 訓練 epoch 數
LSTM_LEARNING_RATE = 0.002        # 學習率
SHORT_QTY_CAP= 50000          # 放空時的最大數量

# === 策略優化設定 ===
ENABLE_STRATEGY_OPTIMIZATION = True
STRATEGY_EVALUATOR = "total_return"     # 可選: "sharpe", "win_rate", "total_return"

# === Gemini LLM 設定 ===
USE_GEMINI_STRATEGY_SELECTION = True
GEMINI_MODEL_NAME = "gemini-1.5-flash"
DEFAULT_STRATEGY = "trend_strategy" # 新增預設策略

# Forced Trading Parameters
MAX_DAYS_NO_TRADE = 10  # 連續未交易N天後考慮強制進場
ENABLE_FORCED_TRADING = True # 是否啟用強制進場功能
FORCED_TRADE_TAKE_PROFIT_PCT = 0.04  # 強制進場交易的止盈百分比 
FORCED_TRADE_STOP_LOSS_PCT = 0.02    # 強制進場交易的止損百分比 
FORCED_TRADE_USE_TRAILING_STOP = True  # 強制進場交易是否使用移動止損
FORCED_TRADE_CAPITAL_ALLOCATION = 0.4  # 強制進場時使用的資金比例 

# === 新增：一般策略交易資金分配 ===
REGULAR_TRADE_CAPITAL_ALLOCATION = 0.20 # 一般策略進場時使用的資金比例 (例如 20% 的現金)

# === 報告與紀錄設定 ===
SAVE_RAW_TRADE_LOG = True # 是否儲存原始的逐筆交易紀錄 CSV
SAVE_LLM_CONTEXT_LOG = True

# --- 新增參數 ---
MAX_LOOKBACK_PERIOD = 60  # 策略計算所需的最大回溯期 (例如，某些策略可能需要60天的數據)

# --- 策略目標持續性設定 ---
MIN_STRATEGY_DURATION_DAYS = 20 # LLM選擇的策略目標建議至少執行的天數

# === 交易單位與精度設定 ===
TRADE_UNIT = 1000 # 交易單位，設為 1 可交易零股，設為 1000 代表一張股票
PRICE_PRECISION_RULES = { 
    (0, 10): 2,      # 10元以下，小數點後2位
    (10, 50): 2,     # 10元至50元以下，小數點後2位
    (50, 100): 2,    # 50元至100元以下，小數點後2位
    (100, 500): 1,   # 100元至500元以下，小數點後1位
    (500, 1000): 1,  # 500元至1000元以下，小數點後1位
    (1000, float('inf')): 0 # 1000元以上，沒有小數點
}

# === 交易成本設定 ===
COMMISSION_RATE = 0.001425  # 手續費率 (例如 0.1425%)
MIN_COMMISSION = 20         # 最低手續費 (例如 NT$20)


