from itertools import product
import pandas as pd
import numpy as np
from strategies import *

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
            signals = strategy.generate_signals(price_df)

            # 假設 price_df 中已有 'date' 和 'close'
            temp_df = price_df.copy()
            temp_df['signal'] = signals
            from trade_simulator import TradeSimulator
            sim = TradeSimulator()
            trades, _ = sim.simulate(temp_df)
            metrics = sim.calculate_metrics(trades)

            # 計算每日策略報酬以評估 Sharpe Ratio
            returns = temp_df['close'].pct_change().fillna(0)
            positions = signals.shift().fillna(0)
            strat_returns = returns * positions
            sharpe = (
                strat_returns.mean() / strat_returns.std() * np.sqrt(252)
                if strat_returns.std() != 0
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
