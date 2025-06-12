import pandas as pd
from .strategy_base import Strategy

class BreakoutStrategy(Strategy):
    """
    突破策略：
    - 價格突破最近 N 天高點 => Buy
    - 價格跌破最近 N 天低點 => Short
    - 持倉後如價格反向或 RSI 過高/過低則平倉
    """

    def generate_signal(self, data_slice, current_index, position):
        row = data_slice.iloc[-1]

        close = row["Close"]
        rsi = row["RSI"]

        window = self.params.get("window", 20)
        rsi_high = self.params.get("rsi_high", 70)
        rsi_low = self.params.get("rsi_low", 30)
        allow_short = self.params.get("allow_short", True)

        recent_high = data_slice["Close"].iloc[-window:].max()
        recent_low = data_slice["Close"].iloc[-window:].min()

        if position is None:
            if close > recent_high and rsi < rsi_high:
                return "Buy"
            elif close < recent_low and rsi > rsi_low and allow_short:
                return "Short"

        elif position == "Long":
            if close < recent_high or rsi > rsi_high:
                return "Sell"

        elif position == "Short":
            if close > recent_low or rsi < rsi_low:
                return "Cover"

        return None
