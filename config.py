# === 股票與回測設定 ===
STOCK_SYMBOL = "2376.TW" 
START_DATE = "2022-01-01"
END_DATE = "2022-10-01"

# === 回測參數設定 ===
INITIAL_CAPITAL = 1_000_000       # 初始資金（TWD）
STOP_LOSS_THRESHOLD = 0.08        # 停損線
ALLOW_SHORT_SELLING = True        # 是否允許放空

# === LSTM 模型設定 ===
LSTM_TRAIN_WINDOW = 200           # LSTM 模型訓練所需的最小資料點數
LSTM_LOOKBACK_DAYS = 30           # LSTM 模型觀察的天數
LSTM_PREDICT_DAYS = 30             # 預測未來幾天平均價格
LSTM_RETRAIN_INTERVAL = 5        # 每幾天 retrain 一次
LSTM_EPOCHS = 20                  # 訓練 epoch 數
LSTM_LEARNING_RATE = 0.0017        # 學習率
SHORT_QTY_CAP= 50000          # 放空時的最大數量

# === 策略優化設定 ===
ENABLE_STRATEGY_OPTIMIZATION = True
STRATEGY_EVALUATOR = "win_rate"     # 可選: "sharpe", "win_rate", "total_return"

# === Gemini LLM 設定 ===
USE_GEMINI_STRATEGY_SELECTION = True
GEMINI_MODEL_NAME = "gemini-1.5-flash"
DEFAULT_STRATEGY = "trend_strategy" # 新增預設策略
ENABLE_LLM_CAPITAL_ALLOCATION = True  # 新增：是否啟用 LLM 決定資金分配
DEFAULT_CAPITAL_ALLOCATION_FACTOR = 0.5 # 新增：LLM 未提供或關閉此功能時的預設資金分配因子 (此因子會乘上 REGULAR_TRADE_CAPITAL_ALLOCATION)

# Forced Trading Parameters
MAX_DAYS_NO_TRADE = 45  # 連續未交易N天後考慮強制進場
ENABLE_FORCED_TRADING = True # 是否啟用強制進場功能
FORCED_TRADE_TAKE_PROFIT_PCT = 0.03  # 強制進場交易的止盈百分比 
FORCED_TRADE_STOP_LOSS_PCT = 0.015    # 強制進場交易的止損百分比 
FORCED_TRADE_USE_TRAILING_STOP = True  # 強制進場交易是否使用移動止損
FORCED_TRADE_CAPITAL_ALLOCATION = 0.15  # 強制進場時使用的資金比例 

# === 新增：一般策略交易資金分配 ===
REGULAR_TRADE_CAPITAL_ALLOCATION = 0.8 # 一般策略進場時使用的資金比例 (例如 0.8 代表總資金的80%中的一部分，具體取決於LLM因子)

# === 新增：整體投資組合保護機制 ===
ENABLE_PORTFOLIO_PROTECTION = True      # 是否啟用整體投資組合保護
PORTFOLIO_START_YEAR = 2022             # 計算保護基準的起始年份 (指第一個完整產生PNL的年份)
PORTFOLIO_PROFIT_PROTECTION_THRESHOLD = 0.5 # 保護獲利的百分比 (例如0.5代表保護前一年累積獲利的50%)
# 當保護機制觸發時的動作："HALT_TRADING" (停止所有新交易), "REDUCE_RISK" (大幅降低後續交易的資金分配因子上限)
PORTFOLIO_PROTECTION_ACTION = "REDUCE_RISK"
PORTFOLIO_REDUCED_RISK_FACTOR_MAX = 0.1 # "REDUCE_RISK" 狀態下的最大LLM資金分配因子 (例如0.1)

# === 報告與紀錄設定 ===
SAVE_RAW_TRADE_LOG = True # 是否儲存原始的逐筆交易紀錄 CSV
SAVE_LLM_CONTEXT_LOG = True

# --- 新增參數 ---
MAX_LOOKBACK_PERIOD = 70  # 策略計算所需的最大回溯期 (例如，某些策略可能需要60天的數據)

# --- 策略目標持續性設定 ---
MIN_STRATEGY_DURATION_DAYS = 25 # LLM選擇的策略目標建議至少執行的天數

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
SHORT_MARGIN_RATE_STOCK=0.9 # 股票放空保證金率 (例如 90%)

