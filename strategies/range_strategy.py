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
        rsi = row.get("RSI") # 使用 .get() 避免 KeyError

        if pd.isna(close) or pd.isna(rsi):
            return None # 如果基本數據缺失，則不產生信號

        # 從參數取得支撐/壓力區間，若未指定則使用近 N 日高低點
        window = self.params.get("window", 20)

        # 確保有足夠數據計算 rolling window，否則 support/resistance 可能為 NaN
        if len(data_slice) < window:
            return None # 數據不足，不產生信號

        support = self.params.get(
            "support",
            data_slice["Close"].rolling(window=window).min().iloc[-1],
        )
        resistance = self.params.get(
            "resistance",
            data_slice["Close"].rolling(window=window).max().iloc[-1],
        )

        if pd.isna(support) or pd.isna(resistance):
            return None # 支撐或壓力計算失敗，不產生信號

        # 放寬RSI門檻以增加交易機會
        rsi_low_entry = self.params.get("rsi_low_entry", 35) # 原為 rsi_low, 50
        rsi_high_entry = self.params.get("rsi_high_entry", 65) # 原為 rsi_high, 70
        rsi_mid_exit = self.params.get("rsi_mid_exit", 50) # 新增：RSI回到中間區域平倉
        allow_short = self.params.get("allow_short", True)

        action = None

        if position is None:
            if close < support and rsi < rsi_low_entry:
                action = "Buy"
            elif allow_short and close > resistance and rsi > rsi_high_entry:
                action = "Short"

        elif position == "Long":
            # 價格回到壓力區 或 RSI 過高 或 RSI 回到中間值 => 平倉
            if close > resistance or rsi > rsi_high_entry + 5 or rsi > rsi_mid_exit + 5: # 稍微放寬出場條件
                action = "Sell"
            # 新增：若價格跌破支撐區也平倉 (停損概念)
            elif close < support * 0.99: # 假設支撐下方1%為停損
                action = "Sell"


        elif position == "Short":
            # 價格回到支撐區 或 RSI 過低 或 RSI 回到中間值 => 平倉
            if close < support or rsi < rsi_low_entry - 5 or rsi < rsi_mid_exit - 5: # 稍微放寬出場條件
                action = "Cover"
            # 新增：若價格突破壓力區也平倉 (停損概念)
            elif close > resistance * 1.01: # 假設壓力上方1%為停損
                action = "Cover"

        return action
