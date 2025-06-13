import pandas as pd
from .strategy_base import Strategy

class TrendStrategy(Strategy):
    """
    趨勢策略：
    - 預測為上漲 + 收盤價高於均線 + RSI 低 => Buy
    - 預測為下跌 + 收盤價低於均線 + RSI 高 => Short
    - 有持倉時：如反向訊號出現 => 平倉
    """

    def generate_signal(self, data_slice, current_index, position):
        row = data_slice.iloc[-1]

        prediction = row.get("Prediction", 0) # 預設為0 (無明確趨勢)
        close = row["Close"]
        ma = row.get("MA") # 使用 .get() 避免 KeyError
        rsi = row.get("RSI") # 使用 .get() 避免 KeyError

        # 檢查關鍵指標是否有效
        if pd.isna(close) or pd.isna(ma) or pd.isna(rsi):
            # print(f"DEBUG [{row.get('date', 'Unknown Date')} TrendStrategy]: Indicator NaN. Close: {close}, MA: {ma}, RSI: {rsi}")
            return None

        # 放寬 RSI 門檻以增加交易機會
        rsi_low_entry = self.params.get("rsi_low_entry", 40) # 原為 rsi_low, 50
        rsi_high_entry = self.params.get("rsi_high_entry", 60) # 原為 rsi_high, 70
        rsi_exit_threshold = self.params.get("rsi_exit_threshold", 15) # RSI反轉多少點出場
        allow_short = self.params.get("allow_short", True)

        action = None

        if position is None:
            # 預測上漲，價格在均線之上，RSI未過高
            if prediction > 0 and close > ma and rsi < rsi_high_entry: # RSI < 60 (原 < 70)
                action = "Buy"
            # 預測下跌，價格在均線之下，RSI未過低
            elif allow_short and prediction < 0 and close < ma and rsi > rsi_low_entry: # RSI > 40 (原 > 50)
                action = "Short"
            # 無預測時，但技術指標出現強烈信號 (可選，增加交易機會)
            elif prediction == 0:
                if close > ma and rsi < rsi_low_entry + 5: # 例如 RSI < 45
                    action = "Buy"
                elif allow_short and close < ma and rsi > rsi_high_entry - 5: # 例如 RSI > 55
                    action = "Short"

        elif position == "Long":
            # 預測轉為下跌 或 價格跌破均線 或 RSI大幅回落
            if prediction < 0 or close < ma or rsi < (rsi_high_entry - rsi_exit_threshold): # 例如 RSI < 60-15=45
                action = "Sell"

        elif position == "Short":
            # 預測轉為上漲 或 價格突破均線 或 RSI大幅回升
            if prediction > 0 or close > ma or rsi > (rsi_low_entry + rsi_exit_threshold): # 例如 RSI > 40+15=55
                action = "Cover"

        return action
