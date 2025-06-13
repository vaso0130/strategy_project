from itertools import product
import pandas as pd
import numpy as np
from strategies import *
# 新增：從 config 匯入必要的參數
from config import (
    INITIAL_CAPITAL, STOP_LOSS_THRESHOLD, ALLOW_SHORT_SELLING, STOCK_SYMBOL, MAX_LOOKBACK_PERIOD,
    SHORT_QTY_CAP, TRADE_UNIT, PRICE_PRECISION_RULES, ENABLE_FORCED_TRADING,
    FORCED_TRADE_TAKE_PROFIT_PCT, FORCED_TRADE_STOP_LOSS_PCT,
    FORCED_TRADE_USE_TRAILING_STOP, FORCED_TRADE_CAPITAL_ALLOCATION # 新增匯入
)
from utils.metrics import calculate_performance_metrics # <--- 新增匯入

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

        # 確定策略所需的最小回溯期
        min_lookback_for_optimizer = MAX_LOOKBACK_PERIOD # 預設值
        # 嘗試從策略類本身獲取 MIN_LOOKBACK
        if hasattr(self.strategy_class, 'MIN_LOOKBACK') and isinstance(self.strategy_class.MIN_LOOKBACK, int):
            min_lookback_for_optimizer = self.strategy_class.MIN_LOOKBACK
        else:
            # 如果 MIN_LOOKBACK 是實例屬性或未定義，嘗試實例化一個臨時對象來獲取它
            try:
                # 使用第一組參數（如果存在）或空字典來實例化
                temp_params = list(self.generate_param_combinations())[0] if self.param_grid and list(self.generate_param_combinations()) else {}
                temp_instance = self.strategy_class(**temp_params)
                if hasattr(temp_instance, 'MIN_LOOKBACK') and isinstance(temp_instance.MIN_LOOKBACK, int):
                    min_lookback_for_optimizer = temp_instance.MIN_LOOKBACK
            except Exception as e:
                # print(f"[優化器警告] 無法動態獲取 {self.strategy_class.__name__} 的 MIN_LOOKBACK: {e}。使用預設值 {MAX_LOOKBACK_PERIOD}")
                pass # 如果實例化失敗或沒有 MIN_LOOKBACK 屬性，則使用預設值

        for params in self.generate_param_combinations():
            # 在此處檢查 price_df 的長度
            if len(price_df) < min_lookback_for_optimizer:
                # print(f"[優化器資訊] 跳過參數 {params} for {self.strategy_class.__name__}，因為數據長度 {len(price_df)} < 所需最小回溯期 {min_lookback_for_optimizer}")
                continue # 數據不足，跳過此參數組合

            # Ensure price_df is not empty and has 'date' and 'Close'
            if price_df.empty or not all(col in price_df.columns for col in ['date', 'Close']):
                # Handle empty or malformed price_df
                print(f"Warning: price_df is empty or missing required columns for strategy {self.strategy_class.__name__} with params {params}. Skipping.")
                continue # Or return if you want to penalize this parameter set

            # Convert date columns to datetime objects if they are not already
            # This is crucial for merging and for time-based operations in strategies
            try:
                price_df['date'] = pd.to_datetime(price_df['date'])
                # Ensure signals_df['date'] is also datetime if it exists and is used before merging
                # The signals_df is generated *after* this block, so its date column will be handled by generate_signal
            except Exception as e:
                print(f"Error converting price_df['date'] to datetime: {e}. Skipping params {params}.")
                continue # Or return

            # Generate signals
            # The strategy's generate_signal method should handle its own date conversions if necessary
            # and return a DataFrame with a 'date' column (ideally already as datetime) and a 'signal' column.
            signals_df = pd.DataFrame() # Initialize to empty DataFrame
            signals_output = None
            try:
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


                # Ensure signals_df['date'] is also datetime before merging
                if not signals_df.empty and 'date' in signals_df.columns:
                    try:
                        signals_df['date'] = pd.to_datetime(signals_df['date'])
                    except Exception as e:
                        print(f"Error converting signals_df['date'] to datetime: {e}. Proceeding with merge if possible, but may fail.")
            
            except Exception as e:
                print(f"[錯誤] 策略 {self.strategy_class.__name__} 執行時發生例外：{e}")
                continue # 跳過此參數組合

            from trade_simulator import TradeSimulator
            sim = TradeSimulator(
                initial_capital=INITIAL_CAPITAL,
                stop_loss=STOP_LOSS_THRESHOLD,
                allow_short=ALLOW_SHORT_SELLING,
                stock_symbol=STOCK_SYMBOL,
                short_qty_cap=SHORT_QTY_CAP,
                trade_unit=TRADE_UNIT,
                price_precision_rules=PRICE_PRECISION_RULES,
                enable_forced_trading=ENABLE_FORCED_TRADING, # 使用 config 中的值
                forced_trade_take_profit_pct=FORCED_TRADE_TAKE_PROFIT_PCT, # 使用 config 中的值
                forced_trade_stop_loss_pct=FORCED_TRADE_STOP_LOSS_PCT, # 使用 config 中的值
                forced_trade_use_trailing_stop=FORCED_TRADE_USE_TRAILING_STOP, # 使用 config 中的值
                forced_trade_capital_allocation=FORCED_TRADE_CAPITAL_ALLOCATION # 使用 config 中的值
            )
            
            # sim.simulate 現在回傳 trade_log_df 和 daily_capital_df
            trade_log_df, daily_capital_df = sim.simulate(price_df.copy(), signals_df)

            # 新增檢查以確保回傳的是 DataFrame
            if not isinstance(trade_log_df, pd.DataFrame):
                print(f"[優化器錯誤] sim.simulate 回傳的 trade_log_df 不是 DataFrame，而是 {type(trade_log_df)}。跳過此參數組合。")
                # print(f"Trade log content if string: {str(trade_log_df)[:500]}") # 如果是字串，印出部分內容以供偵錯
                continue
            if not isinstance(daily_capital_df, pd.DataFrame):
                print(f"[優化器錯誤] sim.simulate 回傳的 daily_capital_df 不是 DataFrame，而是 {type(daily_capital_df)}。跳過此參數組合。")
                continue

            # 準備 trade_log_df 以便傳遞給 calculate_performance_metrics
            # utils.metrics.calculate_performance_metrics 期望 'pnl' (小寫)
            # sim.get_trade_log_df() (由 sim.simulate 間接呼叫) 回傳 'PNL' (大寫)
            if 'PNL' in trade_log_df.columns:
                trade_log_df_for_metrics = trade_log_df.rename(columns={'PNL': 'pnl'})
            else:
                # 如果沒有 'PNL'，則複製一份並確保 'pnl' 欄位存在 (預設為0)
                trade_log_df_for_metrics = trade_log_df.copy()
                if 'pnl' not in trade_log_df_for_metrics.columns:
                    trade_log_df_for_metrics['pnl'] = 0.0
            
            # 使用 utils.metrics 中的函數計算績效
            metrics = calculate_performance_metrics(
                daily_df=daily_capital_df, 
                trade_log_df=trade_log_df_for_metrics,
                initial_capital_for_return_calc=INITIAL_CAPITAL
            )

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

            # 將 sharpe ratio 加入到 metrics 字典中
            if isinstance(metrics, dict): # 確保 metrics 是一個字典
                metrics['sharpe'] = sharpe
            else:
                # 如果 metrics 不是字典 (例如，如果 calculate_performance_metrics 失敗並回傳 None)
                # 則建立一個新的字典
                print(f"[優化器警告] calculate_performance_metrics 未回傳有效的字典。Sharpe Ratio 將是唯一指標。")
                metrics = {'sharpe': sharpe}


            score = self.evaluator(metrics)
            if score > best_score:
                best_score = score
                best_params = params

        return {
            'best_params': best_params,
            'best_score': best_score
        }
