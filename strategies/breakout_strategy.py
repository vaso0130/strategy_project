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
        # 確保有足夠數據計算 rolling window
        if len(data_slice) < window:
            return None

        # .iloc[-1] 確保取到的是當前最新的滾動值
        highest_high = data_slice["High"].rolling(window=window).max().iloc[-1]
        lowest_low = data_slice["Low"].rolling(window=window).min().iloc[-1]
        # 使用一個較短的均線作為趨勢參考或止損參考
        short_ma_window = max(5, window // 2) # 確保 short_ma_window 至少為5
        if len(data_slice) < short_ma_window: # 增加對 short_ma_window 的數據長度檢查
            return None
        current_ma = data_slice["Close"].rolling(window=short_ma_window).mean().iloc[-1]


        if pd.isna(highest_high) or pd.isna(lowest_low) or pd.isna(current_ma):
            return None

        # 調整預設 RSI 門檻以增加交易機會
        rsi_low_entry = self.params.get("rsi_low_entry", 40) # 用於做空入場時，RSI 不要太低 (原為 rsi_low)
        rsi_high_entry = self.params.get("rsi_high_entry", 60) # 用於做多入場時，RSI 不要太高 (原為 rsi_high)
        
        rsi_low_exit = self.params.get("rsi_low_exit", 25) # 用於做空出場的 RSI 極冷區 (新增)
        rsi_high_exit = self.params.get("rsi_high_exit", 75) # 用於做多出場的 RSI 極熱區 (新增)

        allow_short = self.params.get("allow_short", True)

        action = None
        if position is None:
            if close > highest_high and rsi < rsi_high_entry:  # 突破高點，RSI 未過熱
                action = "Buy"
            elif allow_short and close < lowest_low and rsi > rsi_low_entry:  # 跌破低點，RSI 未過冷
                action = "Short"
        elif position == "Long":
            # 跌破短期均線或 RSI 極度超買則平倉
            if close < current_ma or rsi > rsi_high_exit:
                action = "Sell"
        elif position == "Short":
            # 突破短期均線或 RSI 極度超賣則平倉
            if close > current_ma or rsi < rsi_low_exit:
                action = "Cover"
        
        return action
