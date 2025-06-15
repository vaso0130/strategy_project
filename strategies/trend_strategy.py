import pandas as pd
from .strategy_base import Strategy

class TrendStrategy(Strategy):
    """
    趨勢策略：
    - 預測為上漲 + 收盤價高於均線 + RSI 低 => Buy
    - 預測為下跌 + 收盤價低於均線 + RSI 高 => Short
    - 有持倉時：如反向訊號出現 => 平倉
    """

    @staticmethod
    def get_default_params():
        from config import TREND_STRATEGY_DEFAULT_PARAMS
        return TREND_STRATEGY_DEFAULT_PARAMS.copy()

    def generate_signal(self, data_slice, current_index, position):
        row = data_slice.iloc[-1]
        # 新增：允許策略讀取 LSTM_PREDICTION 欄位（若有）與 optimizer 最佳參數（若有）
        lstm_pred = row.get("LSTM_PREDICTION", None)

        date_info = row.name # Assuming index is datetime or has date info

        prediction = row.get("Prediction", 0)
        close = row["Close"]
        ma = row.get("MA")
        rsi = row.get("RSI")

        # 檢查關鍵指標是否有效
        if pd.isna(close) or pd.isna(ma) or pd.isna(rsi):
            return None

        # 優化：放寬RSI與預測條件
        rsi_low_entry = self.params.get("rsi_low_entry", 45)
        rsi_high_entry = self.params.get("rsi_high_entry", 55)
        rsi_exit_threshold = self.params.get("rsi_exit_threshold", 10)
        allow_short = self.params.get("allow_short", True)
        
        action = None

        if position is None:
            # 預測上漲，價格在均線之上，RSI未過高
            if prediction > 0 and close > ma * 0.99 and rsi < rsi_high_entry + 5:
                action = "Buy"
            # 預測下跌，價格在均線之下，RSI未過低
            elif allow_short and prediction < 0 and close < ma * 1.01 and rsi > rsi_low_entry - 5:
                action = "Short"
            # 無預測時，但技術指標出現強烈信號 (可選，增加交易機會)
            elif prediction == 0:
                if close > ma * 0.995 and rsi < rsi_low_entry + 10:
                    action = "Buy"
                elif allow_short and close < ma * 1.005 and rsi > rsi_high_entry - 10:
                    action = "Short"

        elif position == "Long":
            # 預測轉為下跌 或 價格跌破均線 或 RSI大幅回落
            if prediction < 0 or close < ma * 1.01 or rsi < (rsi_high_entry - rsi_exit_threshold):
                action = "Sell"


        elif position == "Short":
            # 預測轉為上漲 或 價格突破均線 或 RSI大幅回升
            if prediction > 0 or close > ma * 0.99 or rsi > (rsi_low_entry + rsi_exit_threshold):
                action = "Cover"

        return action
