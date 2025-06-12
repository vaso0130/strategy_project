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

    def generate_signals(self, df):
        """產生整段資料的交易訊號序列。

        逐列傳入 :meth:`generate_signal`，並根據回傳的字串更新持倉狀態，
        最終回傳對應的數值信號（1、-1、0）Series。

        1 -> Buy
        -1 -> Short
        0 -> 觀望或平倉
        """
        import pandas as pd

        signals = []
        position = None

        for idx in range(len(df)):
            slice_df = df.iloc[: idx + 1]
            action = self.generate_signal(slice_df, idx, position)

            if action == "Buy":
                position = "Long"
                signals.append(1)
            elif action == "Short":
                position = "Short"
                signals.append(-1)
            elif action in ("Sell", "Cover"):
                position = None
                signals.append(0)
            else:
                signals.append(0)

        return pd.Series(signals, index=df.index)
