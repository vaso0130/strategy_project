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

        # 從參數取得支撐/壓力區間，若未指定則使用近 N 日高低點
        window = self.params.get("window", 20)
        support = self.params.get(
            "support",
            data_slice["Close"].rolling(window=window).min().iloc[-1],
        )
        resistance = self.params.get(
            "resistance",
            data_slice["Close"].rolling(window=window).max().iloc[-1],
        )
        rsi_low = self.params.get("rsi_low", 50)
        rsi_high = self.params.get("rsi_high", 70)
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
