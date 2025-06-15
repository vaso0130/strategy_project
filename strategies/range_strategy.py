import pandas as pd
from .strategy_base import Strategy

class RangeStrategy(Strategy):
    """
    盤整策略：
    - 收盤價低於區間下緣（支撐） + RSI低 => Buy
    - 收盤價高於區間上緣（壓力） + RSI高 => Short
    - 有持倉時，如價格反向進入另一端或 RSI反轉 => 平倉
    """

    @staticmethod
    def get_default_params():
        from config import RANGE_STRATEGY_DEFAULT_PARAMS
        return RANGE_STRATEGY_DEFAULT_PARAMS.copy()

    def generate_signal(self, data_slice, current_index, position):
        row = data_slice.iloc[-1]
        # 新增：允許策略讀取 LSTM_PREDICTION 欄位（若有）
        lstm_pred = row.get("LSTM_PREDICTION", None)
        # 你可以根據 lstm_pred 進行進階判斷，例如：
        # if lstm_pred == 1: ...
        # 目前先保留原本邏輯，未強制納入 LSTM 預測
        date_info = row.name # Assuming index is datetime or has date info

        close = row["Close"]
        rsi = row.get("RSI")

        if pd.isna(close) or pd.isna(rsi):
            return None

        window = self.params.get("window", 20)
        if len(data_slice) < window:
            return None

        support = self.params.get("support", data_slice["Close"].rolling(window=window).min().iloc[-1])
        resistance = self.params.get("resistance", data_slice["Close"].rolling(window=window).max().iloc[-1])

        if pd.isna(support) or pd.isna(resistance):
            return None

        # 優化：放寬RSI與區間條件
        rsi_low_entry = self.params.get("rsi_low_entry", 45)
        rsi_high_entry = self.params.get("rsi_high_entry", 55)
        rsi_mid_exit = self.params.get("rsi_mid_exit", 50)
        allow_short = self.params.get("allow_short", True)
        range_tol = self.params.get("range_tol", 1.01)  # 允許接近支撐/壓力

        action = None

        if position is None:
            if close < support * range_tol and rsi < rsi_low_entry:
                action = "Buy"
            elif allow_short and close > resistance / range_tol and rsi > rsi_high_entry:
                action = "Short"

        elif position == "Long":
            if close > resistance * 0.99 or rsi > rsi_high_entry + 5 or rsi > rsi_mid_exit + 5:
                action = "Sell"
            elif close < support * 0.98:
                action = "Sell"

        elif position == "Short":
            if close < support * 1.02 or rsi < rsi_low_entry - 5 or rsi < rsi_mid_exit - 5:
                action = "Cover"
            elif close > resistance * 1.02:
                action = "Cover"

        return action
