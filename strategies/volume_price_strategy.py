import pandas as pd
from .strategy_base import Strategy

class VolumePriceStrategy(Strategy):
    """
    量價策略：
    - 成交量明顯放大且價格突破短期均線 => Buy
    - 成交量放大且價格跌破短期均線 => Short
    - 價格回落均線或 RSI 過熱過冷時出場
    """

    def generate_signal(self, data_slice, current_index, position):
        row = data_slice.iloc[-1]
        prev_row = data_slice.iloc[-2] if len(data_slice) >= 2 else row

        price = row["Close"]
        volume = row["Volume"]

        ma_short = data_slice["Close"].rolling(window=5).mean().iloc[-1]
        volume_ma = data_slice["Volume"].rolling(window=5).mean().iloc[-1]

        rsi = row.get("RSI", 50)

        volume_threshold = self.params.get("volume_ratio", 1.5)
        rsi_high = self.params.get("rsi_high", 60)
        rsi_low = self.params.get("rsi_low", 40)
        allow_short = self.params.get("allow_short", True)

        # Volume spike
        volume_spike = volume > volume_threshold * volume_ma

        if position is None:
            if volume_spike and price > ma_short and rsi < rsi_high:
                return "Buy"
            elif volume_spike and price < ma_short and rsi > rsi_low and allow_short:
                return "Short"

        elif position == "Long":
            if price < ma_short or rsi > rsi_high:
                return "Sell"

        elif position == "Short":
            if price > ma_short or rsi < rsi_low:
                return "Cover"

        return None
