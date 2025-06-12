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

        prediction = row.get("Prediction", None)
        close = row["Close"]
        ma = row["MA"]
        rsi = row["RSI"]

        rsi_low = self.params.get("rsi_low", 30)
        rsi_high = self.params.get("rsi_high", 70)
        allow_short = self.params.get("allow_short", True)

        if position is None:
            if prediction == "up" and close > ma and rsi < rsi_low:
                return "Buy"
            elif prediction == "down" and close < ma and rsi > rsi_high and allow_short:
                return "Short"

        elif position == "Long":
            if prediction == "down" or close < ma or rsi > rsi_high:
                return "Sell"

        elif position == "Short":
            if prediction == "up" or close > ma or rsi < rsi_low:
                return "Cover"

        return None
