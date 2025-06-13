from .strategy_base import Strategy
import pandas as pd

class BreakoutStrategy(Strategy):
    """
    突破策略：
    - 價格突破 N 日高點 + RSI 未過熱 => Buy
    - 價格跌破 N 日低點 + RSI 未過冷 => Short
    - 持倉時，價格反轉或 RSI 極度反向 => 平倉
    """

    def generate_signal(self, data_slice, current_index, position):
        row = data_slice.iloc[-1]
        close = row["Close"]
        high = row["High"]
        low = row["Low"]
        rsi = row.get("RSI")

        if pd.isna(close) or pd.isna(high) or pd.isna(low) or pd.isna(rsi):
            return None

        window = self.params.get("window", 20)
        if len(data_slice) < window:
            return None

        highest_high = data_slice["High"].rolling(window=window).max().iloc[-1]
        lowest_low = data_slice["Low"].rolling(window=window).min().iloc[-1]
        short_ma_window = max(5, window // 2)
        if len(data_slice) < short_ma_window:
            return None
        current_ma = data_slice["Close"].rolling(window=short_ma_window).mean().iloc[-1]

        if pd.isna(highest_high) or pd.isna(lowest_low) or pd.isna(current_ma):
            return None

        # 優化：放寬RSI與突破條件
        rsi_low_entry = self.params.get("rsi_low", 50)
        rsi_high_entry = self.params.get("rsi_high", 70)
        rsi_low_exit = self.params.get("rsi_low_exit", 30)
        rsi_high_exit = self.params.get("rsi_high_exit", 80)
        allow_short = self.params.get("allow_short", True)
        breakout_tol = self.params.get("breakout_tol", 0.995)  # 允許接近高/低點

        action = None
        if position is None:
            if close > highest_high * breakout_tol and rsi < rsi_high_entry:
                action = "Buy"
            elif allow_short and close < lowest_low / breakout_tol and rsi > rsi_low_entry:
                action = "Short"
        elif position == "Long":
            # 平倉條件放寬
            if close < current_ma * 1.01 or rsi > rsi_high_exit:
                action = "Sell"
        elif position == "Short":
            if close > current_ma * 0.99 or rsi < rsi_low_exit:
                action = "Cover"
        return action
