import pandas as pd
from .strategy_base import Strategy

class RangeStrategy(Strategy):
    """
    盤整策略：
    - 收盤價低於區間下緣（支撐） + RSI低 => Buy
    - 收盤價高於區間上緣（壓力） + RSI高 => Short
    - 有持倉時，如價格反向進入另一端或 RSI反轉 => 平倉
    """

    def generate_signal(self, data_slice, current_index, position):
        row = data_slice.iloc[-1]

        close = row["Close"]
        rsi = row["RSI"]

        # 從參數取得支撐/壓力區間
        support = self.params.get("support", row["Close"] * 0.95)
        resistance = self.params.get("resistance", row["Close"] * 1.05)
        rsi_low = self.params.get("rsi_low", 40)
        rsi_high = self.params.get("rsi_high", 60)
        allow_short = self.params.get("allow_short", True)

        if position is None:
            if close < support and rsi < rsi_low:
                return "Buy"
            elif close > resistance and rsi > rsi_high and allow_short:
                return "Short"

        elif position == "Long":
            if close > resistance or rsi > rsi_high:
                return "Sell"

        elif position == "Short":
            if close < support or rsi < rsi_low:
                return "Cover"

        return None
