from itertools import product
import pandas as pd
import numpy as np
from strategies import *
# 新增：從 config 匯入必要的參數
from config import INITIAL_CAPITAL, STOP_LOSS_THRESHOLD, ALLOW_SHORT_SELLING

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
            # 確保傳遞給 generate_signals 的是 DataFrame
            signals_df = strategy.generate_signals(price_df.copy())


            # 假設 price_df 中已有 'date' 和 'close'
            # simulator 需要完整的市場數據 df 和訊號 df
            # signals_df 應該包含 'date' 和 'signal'
            from trade_simulator import TradeSimulator
            # 修改：傳入必要的參數來初始化 TradeSimulator
            sim = TradeSimulator(
                initial_capital=INITIAL_CAPITAL,
                stop_loss=STOP_LOSS_THRESHOLD,
                allow_short=ALLOW_SHORT_SELLING
            )
            # 確保傳遞給 simulate 的是原始價格 df 和訊號 df
            trades, _ = sim.simulate(price_df.copy(), signals_df) # 傳遞 price_df 和 signals_df
            metrics = sim.calculate_metrics(trades)


            # 計算每日策略報酬以評估 Sharpe Ratio
            # 確保 signals_df['signal'] 和 price_df['close'] 的索引對齊
            # 並且 signals_df 只有 'date' 和 'signal'
            
            # 合併價格和訊號以計算報酬
            merged_for_returns = pd.merge(price_df[['date', 'close']], signals_df[['date', 'signal']], on='date', how='left')
            merged_for_returns['signal'] = merged_for_returns['signal'].fillna(0)

            returns = merged_for_returns['close'].pct_change().fillna(0)
            positions = merged_for_returns['signal'].shift().fillna(0) # 訊號通常是隔天生效
            strat_returns = returns * positions
            sharpe = (
                strat_returns.mean() / strat_returns.std() * np.sqrt(252)
                if strat_returns.std() != 0 and strat_returns.std() is not np.nan
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
