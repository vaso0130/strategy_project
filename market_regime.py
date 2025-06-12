import pandas as pd
import numpy as np

def calculate_market_regime(df: pd.DataFrame, window: int = 30) -> str:
    """
    根據過去 `window` 天的價格判斷市場型態：'trend' 或 'range'
    df 必須包含欄位 ['date', 'Close']
    """
    if len(df) < window:
        return 'range'  # 資料不足預設為盤整

    closes = df['Close'].tail(window).reset_index(drop=True) # 改為大寫 'Close'
    returns = closes.pct_change().dropna()

    # 1. 計算波動率
    volatility = returns.std()

    # 2. 計算均線斜率（簡單線性回歸）
    x = np.arange(len(closes))
    y = closes.values
    slope = np.polyfit(x, y, 1)[0]  # 取一次項斜率

    # 3. 價格偏離中位數的程度
    price_deviation = np.abs(closes - closes.median()).mean() / closes.median()

    # 閾值設定（可再優化）
    is_trending = (
        volatility > 0.01 and      # 波動率 > 1%
        abs(slope) > 0.05 and      # 斜率明顯
        price_deviation > 0.02     # 偏離中位數 > 2%
    )

    return 'trend' if is_trending else 'range'
