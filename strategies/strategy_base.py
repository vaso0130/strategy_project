from abc import ABC, abstractmethod

class Strategy(ABC):
    """
    策略基底類別，所有子策略都應繼承並實作 generate_signal 方法。
    """

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

def generate_signals(self, price_df):
        """根據歷史資料逐步產生數值化的交易訊號序列。"""
        import pandas as pd

        signals = []
        position = None
        for i in range(len(price_df)):
            data_slice = price_df.iloc[: i + 1]
            action = self.generate_signal(data_slice, i, position)

            if action == "Buy":
                position = "Long"
                signals.append(1)
            elif action == "Short":
                position = "Short"
                signals.append(-1)
            elif action in ("Sell", "Cover"):
                position = None
                signals.append(0)
            else:  # 無動作，維持原持倉方向
                if position == "Long":
                    signals.append(1)
                elif position == "Short":
                    signals.append(-1)
                else:
                    signals.append(0)

        return pd.Series(signals, index=price_df.index)
