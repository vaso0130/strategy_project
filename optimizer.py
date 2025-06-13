from itertools import product
import pandas as pd
import numpy as np
from strategies import *
# 新增：從 config 匯入必要的參數
from config import INITIAL_CAPITAL, STOP_LOSS_THRESHOLD, ALLOW_SHORT_SELLING, STOCK_SYMBOL # 新增 STOCK_SYMBOL

class StrategyOptimizer:
    def __init__(self, strategy_class, param_grid: dict, evaluator):
        """
        strategy_class: 策略類別（如 TrendStrategy）
        param_grid: 參數搜尋空間（dict of lists）
        evaluator: 評估函式（如：lambda result_df: result_df['sharpe']）
        """
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.evaluator = evaluator

    def generate_param_combinations(self):
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        for combination in product(*values):
            yield dict(zip(keys, combination))

    def optimize(self, price_df: pd.DataFrame) -> dict:
        best_params = None
        best_score = -np.inf

        for params in self.generate_param_combinations():
            strategy = self.strategy_class(**params)
            signals_output = strategy.generate_signals(price_df.copy())

            # 確保 signals_df 是一個 DataFrame 且 'signal' 欄位存在
            if isinstance(signals_output, pd.Series):
                signals_df = signals_output.to_frame(name='signal')
                # 如果 Series 的索引是日期，需要重設索引以得到 'date' 欄
                if isinstance(signals_df.index, pd.DatetimeIndex) or isinstance(signals_df.index, pd.PeriodIndex):
                    signals_df = signals_df.reset_index()
                    # 確保日期欄位名稱是 'date'
                    if 'index' in signals_df.columns and 'date' not in signals_df.columns: # pandas < 2.0
                         signals_df = signals_df.rename(columns={'index': 'date'})
                    elif signals_df.index.name == 'date' and 'date' not in signals_df.columns: # pandas >= 2.0
                         signals_df = signals_df.reset_index(names='date')


            elif isinstance(signals_output, pd.DataFrame):
                signals_df = signals_output
            else:
                # 如果策略回傳的不是 Series 或 DataFrame，需要處理錯誤或轉換
                print(f"[錯誤] 策略 {self.strategy_class.__name__} 未回傳有效的訊號格式。")
                continue # 跳過此參數組合

            # 再次確認 'date' 和 'signal' 欄位存在
            if 'date' not in signals_df.columns or 'signal' not in signals_df.columns:
                print(f"[錯誤] 策略 {self.strategy_class.__name__} 回傳的 signals_df 缺少 'date' 或 'signal' 欄位。")
                print(f"Signals_df columns: {signals_df.columns}")
                # 嘗試從索引恢復日期 (如果適用)
                if 'date' not in signals_df.columns and (isinstance(signals_df.index, pd.DatetimeIndex) or isinstance(signals_df.index, pd.PeriodIndex)):
                    signals_df = signals_df.reset_index()
                    if 'index' in signals_df.columns and 'date' not in signals_df.columns:
                         signals_df = signals_df.rename(columns={'index': 'date'})
                    elif signals_df.index.name == 'date' and 'date' not in signals_df.columns:
                         signals_df = signals_df.reset_index(names='date')

                if 'date' not in signals_df.columns or 'signal' not in signals_df.columns:
                    continue # 如果還是缺少必要欄位，則跳過


            from trade_simulator import TradeSimulator
            sim = TradeSimulator(
                initial_capital=INITIAL_CAPITAL,
                stop_loss=STOP_LOSS_THRESHOLD,
                allow_short=ALLOW_SHORT_SELLING,
                stock_symbol=STOCK_SYMBOL, # 新增 stock_symbol 參數
                # 在優化器中，我們可能不需要啟用強制交易，或者需要從 config 傳遞這些參數
                # 為了簡單起見，這裡暫時禁用或使用預設值
                enable_forced_trading=False # 或者從 config 匯入並傳遞
            )
            # 修改：接收 simulate 回傳的三個值
            trades, final_capital, daily_capital_df = sim.simulate(price_df.copy(), signals_df)
            metrics = sim.calculate_metrics(trades)


            # 計算每日策略報酬以評估 Sharpe Ratio
            # 確保 signals_df['signal'] 和 price_df['close'] 的索引對齊
            # 並且 signals_df 只有 'date' 和 'signal'
            
            # 合併價格和訊號以計算報酬
            # merged_for_returns = pd.merge(price_df[['date', 'close']], signals_df[['date', 'signal']], on='date', how='left')
            # 使用 'Close' (大寫)
            merged_for_returns = pd.merge(price_df[['date', 'Close']], signals_df[['date', 'signal']], on='date', how='left')
            merged_for_returns['signal'] = merged_for_returns['signal'].fillna(0)
            
            # 計算每日回報率 (基於收盤價)
            # merged_for_returns['price_change'] = merged_for_returns['close'].pct_change()
            merged_for_returns['price_change'] = merged_for_returns['Close'].pct_change()
            merged_for_returns['strategy_return'] = merged_for_returns['price_change'] * merged_for_returns['signal'].shift(1) # 訊號在隔天生效
            merged_for_returns = merged_for_returns.dropna()

            if merged_for_returns.empty:
                continue  # 如果合併後的資料為空，則跳過此組合

            # 計算年化報酬率和年化波動率
            annual_return = merged_for_returns['strategy_return'].mean() * 252
            annual_volatility = merged_for_returns['strategy_return'].std() * np.sqrt(252)

            # 計算 Sharpe Ratio
            sharpe = (
                annual_return / annual_volatility
                if annual_volatility != 0 and annual_volatility is not np.nan
                else 0
            )

            metrics['sharpe'] = sharpe

            score = self.evaluator(metrics)
            if score > best_score:
                best_score = score
                best_params = params

        return {
            'best_params': best_params,
            'best_score': best_score
        }
