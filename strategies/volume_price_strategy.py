import pandas as pd
from .strategy_base import Strategy

class VolumePriceStrategy(Strategy):
    MIN_LOOKBACK = 5 # 新增 MIN_LOOKBACK 類別屬性
    """
    量價策略：
    - 成交量明顯放大且價格突破短期均線 => Buy
    - 成交量放大且價格跌破短期均線 => Short
    - 價格回落均線或 RSI 過熱過冷時出場
    """

    def generate_signal(self, data_slice, current_index, position):
        # 至少需要 5 筆數據來計算 rolling(window=5)
        if len(data_slice) < 5:
            return None
        row = data_slice.iloc[-1]
        # 新增：允許策略讀取 LSTM_PREDICTION 欄位（若有）與 optimizer 最佳參數（若有）
        lstm_pred = row.get("LSTM_PREDICTION", None)
        # 你可以根據 lstm_pred 進行進階判斷，例如：
        # if lstm_pred == 1: ...
        # 目前先保留原本邏輯，未強制納入 LSTM 預測

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
        # 優化：降低volume spike門檻，放寬RSI
        volume_threshold = self.params.get("volume_ratio", 1.2)
        rsi_high = self.params.get("rsi_high", 75)
        rsi_low = self.params.get("rsi_low", 45)
        allow_short = self.params.get("allow_short", True)
        volume_spike = volume > volume_threshold * volume_ma
        action = None
        if position is None:
            if (volume_spike or volume > volume_ma) and price > ma_short * 0.99 and rsi < rsi_high:
                action = "Buy"
            elif (volume_spike or volume > volume_ma) and price < ma_short * 1.01 and rsi > rsi_low and allow_short:
                action = "Short"
        elif position == "Long":
            if price < ma_short * 1.01 or rsi > rsi_high:
                action = "Sell"
        elif position == "Short":
            if price > ma_short * 0.99 or rsi < rsi_low:
                action = "Cover"
        return action

    @staticmethod
    def get_default_params():
        from config import VOLUME_PRICE_STRATEGY_DEFAULT_PARAMS
        return VOLUME_PRICE_STRATEGY_DEFAULT_PARAMS.copy()
