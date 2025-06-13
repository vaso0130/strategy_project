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
        # 至少需要 5 筆數據來計算 rolling(window=5)
        if len(data_slice) < 5:
            # print(f"DEBUG [VolumePriceStrategy]: Not enough data for rolling window. Len: {len(data_slice)}")
            return None

        row = data_slice.iloc[-1]
        # prev_row = data_slice.iloc[-2] if len(data_slice) >= 2 else row # prev_row 未在此邏輯中使用

        price = row["Close"]
        volume = row.get("Volume") # 使用 .get()
        rsi = row.get("RSI") # 使用 .get()

        if pd.isna(price) or pd.isna(volume) or pd.isna(rsi):
            # print(f"DEBUG [{row.get('date', 'Unknown Date')} VolumePriceStrategy]: Price, Volume or RSI NaN.")
            return None

        ma_short = data_slice["Close"].rolling(window=5).mean().iloc[-1]
        volume_ma = data_slice["Volume"].rolling(window=5).mean().iloc[-1]

        if pd.isna(ma_short) or pd.isna(volume_ma):
            # print(f"DEBUG [{row.get('date', 'Unknown Date')} VolumePriceStrategy]: ma_short or volume_ma NaN.")
            return None


        volume_threshold = self.params.get("volume_ratio", 1.5)
        rsi_high = self.params.get("rsi_high", 70)
        rsi_low = self.params.get("rsi_low", 50)
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
