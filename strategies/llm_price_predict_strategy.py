from .strategy_base import Strategy

class LLMPricePredictStrategy(Strategy):
    """
    策略說明：
    由 LLM 預測未來價格，若預測價格高於現價一定百分比（或落在指定區間），則買進；
    若低於現價一定百分比，則賣出或放空。
    參數：
        - predict_price_key: LLM 回傳 dict 內預測價格的 key（如 'predicted_price'）
        - buy_threshold_pct: 預測價高於現價多少百分比才買進（如 0.02 代表+2%）
        - sell_threshold_pct: 預測價低於現價多少百分比才賣出/放空（如 0.02 代表-2%）
    """
    def __init__(self, predict_price_key='predicted_price', buy_threshold_pct=0.02, sell_threshold_pct=0.02, **kwargs):
        super().__init__(**kwargs)
        self.predict_price_key = predict_price_key
        self.buy_threshold_pct = buy_threshold_pct
        self.sell_threshold_pct = sell_threshold_pct

    def generate_signal(self, data_slice, current_index, position=None, llm_predict_result=None):
        # llm_predict_result: dict, 應包含 LLM 預測價格
        if llm_predict_result is None or self.predict_price_key not in llm_predict_result:
            return None
        predicted_price = llm_predict_result[self.predict_price_key]
        try:
            predicted_price = float(predicted_price)
        except Exception:
            return None
        # 取當日現價（可用收盤或開盤）
        today_row = data_slice.iloc[-1]
        current_price = today_row['Close'] if 'Close' in today_row else today_row.get('Open', None)
        if current_price is None:
            return None
        # 判斷買賣
        if predicted_price >= current_price * (1 + self.buy_threshold_pct):
            return 'BUY'
        elif predicted_price <= current_price * (1 - self.sell_threshold_pct):
            return 'SELL'
        else:
            return 'HOLD'
