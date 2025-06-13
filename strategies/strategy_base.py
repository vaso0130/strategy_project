from abc import ABC, abstractmethod
import pandas as pd # 移到檔案開頭

class Strategy(ABC):
    """策略基底類別，所有子策略都應繼承並實作 :func:`generate_signal`。"""

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def generate_signal(self, data_slice, current_index, position):
        """
        根據當前資料切片與狀態產生交易信號。

        :param data_slice: 包含技術指標與價格的 DataFrame（rolling window）
        :param current_index: 現在在整體資料中的索引
        :param position: 是否有持倉（None / "Long" / "Short"）
        :return: "Buy" / "Sell" / "Short" / "Cover" / None
        """
        pass

    def generate_signals(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """根據歷史資料逐步產生數值化的交易訊號序列，回傳 DataFrame。"""
        if 'date' not in price_df.columns:
            raise ValueError("price_df 必須包含 'date' 欄位。")

        signal_values = []
        dates = []
        position = None # 'Long', 'Short', or None

        # print(f"DEBUG Base: Generating signals for {type(self).__name__} with params {self.params}")

        for i in range(len(price_df)):
            # 確保 data_slice 包含日期，並且是 DataFrame
            data_slice = price_df.iloc[: i + 1]
            current_date = price_df['date'].iloc[i]
            dates.append(current_date)

            # --- DEBUG PRINTS ---
            print(f"DEBUG Base [{current_date}]: About to call generate_signal for {type(self).__name__}")
            print(f"DEBUG Base [{current_date}]:   data_slice length: {len(data_slice)}, type: {type(data_slice)}")
            print(f"DEBUG Base [{current_date}]:   current_index (i): {i}, type: {type(i)}")
            print(f"DEBUG Base [{current_date}]:   position: '{position}', type: {type(position)}")
            # --- END DEBUG PRINTS ---
            
            action = self.generate_signal(data_slice, i, position)
            
            # print(f"DEBUG Base [{current_date}]: Action from {type(self).__name__}: {action}")

            current_signal_value = 0 # 預設為無訊號/平倉
            if action == "Buy":
                if position != "Long": # 只有在未持有多單或空倉時才真正買入
                    current_signal_value = 1
                position = "Long"
            elif action == "Short":
                if position != "Short": # 只有在未持有空單或多單時才真正放空
                    current_signal_value = -1
                position = "Short"
            elif action in ("Sell", "Cover"):
                if position is not None: # 只有在有持倉時才平倉
                    current_signal_value = 0 # Signal 0 for closing a position
                position = None
            # else: # 無明確動作 (action is None)
                # 維持前一天的訊號 (代表持倉不動)
                # 如果前一天是 Long (1)，今天繼續 Long (1)
                # 如果前一天是 Short (-1)，今天繼續 Short (-1)
                # 如果前一天是 None (0)，今天繼續 None (0)
                # 這個邏輯在 TradeSimulator 中處理持倉更合適
                # 這裡的 signal 代表的是當日的 *動作* 訊號
                # 因此若無 action，則 signal 為 0 (不開新倉，也不平現有倉)

            signal_values.append(current_signal_value)

        # 建立包含 'date' 和 'signal' 的 DataFrame
        signals_df = pd.DataFrame({'date': dates, 'signal': signal_values})
        return signals_df
