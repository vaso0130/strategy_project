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
