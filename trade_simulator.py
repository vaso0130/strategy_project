import pandas as pd

class TradeSimulator:
    def __init__(self, initial_capital: int, stop_loss: float, allow_short: bool):
        self.initial_capital = initial_capital
        self.stop_loss = stop_loss
        self.allow_short = allow_short
        self.reset()

    def reset(self):
        self.cash = self.initial_capital
        self.position = 0  # 持有股數
        self.entry_price = 0
        self.trades = []
        self.holding = False
        self.direction = None  # 'long' or 'short'
        self.daily_capital = [] # 用於記錄每日資產

    def simulate(self, df: pd.DataFrame, signal_df: pd.DataFrame):
        """
        模擬交易過程

        Args:
            df (pd.DataFrame): 包含 'date', 'open', 'high', 'low', 'close' 的市場數據
            signal_df (pd.DataFrame): 包含 'date', 'signal' 的交易訊號，
                                     signal: 1 (買入), -1 (賣出/放空), 0 (平倉/觀望)
        Returns:
            pd.DataFrame: 交易紀錄
            float: 最終資產
        """
        self.reset()
        if df.empty or signal_df.empty:
            return pd.DataFrame(self.trades), self.initial_capital

        # 合併市場數據與訊號，確保日期對齊
        merged_df = pd.merge(df, signal_df, on='date', how='left').fillna(0)

        for i, row in merged_df.iterrows():
            current_date = row['date']
            current_price = row['Open'] # 假設以開盤價交易
            signal = row['signal']

            # 停損檢查 (簡易版，可再優化)
            if self.holding:
                if self.direction == 'long' and current_price < self.entry_price * (1 - self.stop_loss):
                    signal = 0 # 觸發停損，強制平倉
                    # print(f"{current_date} Long Stop Loss triggered at {current_price}")
                elif self.direction == 'short' and current_price > self.entry_price * (1 + self.stop_loss):
                    signal = 0 # 觸發停損，強制平倉
                    # print(f"{current_date} Short Stop Loss triggered at {current_price}")


            if signal == 1 and not self.holding: # 買入訊號且未持倉
                self.position = self.cash // current_price
                if self.position > 0:
                    self.cash -= self.position * current_price
                    self.entry_price = current_price
                    self.holding = True
                    self.direction = 'long'
                    self.trades.append({
                        "entry_date": current_date, "exit_date": None,
                        "entry_price": current_price, "exit_price": None,
                        "side": "long", "pnl": None, "shares": self.position
                    })
            elif signal == -1 and self.allow_short and not self.holding : # 放空訊號且允許放空且未持倉
                self.position = self.cash // current_price # 假設以等值現金放空
                if self.position > 0:
                    # 放空時，現金增加 (保證金概念簡化)
                    # self.cash += self.position * current_price # 實際券商操作更複雜
                    self.entry_price = current_price
                    self.holding = True
                    self.direction = 'short'
                    self.trades.append({
                        "entry_date": current_date, "exit_date": None,
                        "entry_price": current_price, "exit_price": None,
                        "side": "short", "pnl": None, "shares": self.position
                    })
            elif signal == 0 and self.holding: # 平倉訊號且持倉
                last_trade = self.trades[-1]
                last_trade["exit_date"] = current_date
                last_trade["exit_price"] = current_price
                
                if self.direction == 'long':
                    self.cash += self.position * current_price
                    pnl = (current_price - self.entry_price) * self.position
                elif self.direction == 'short':
                    # 空單回補，現金減少買回成本，獲利為 (賣出價 - 回補價) * 股數
                    self.cash -= self.position * current_price # 回補成本
                    self.cash += self.position * self.entry_price # 初始賣空收入 (簡化模型)
                    pnl = (self.entry_price - current_price) * self.position
                
                last_trade["pnl"] = pnl
                self.position = 0
                self.entry_price = 0
                self.holding = False
                self.direction = None

            # 記錄每日資產
            current_total_value = self.cash
            if self.holding and self.direction == 'long':
                current_total_value += self.position * row['Close'] # 以收盤價計算當日持有價值
            elif self.holding and self.direction == 'short':
                 # 空單的市值計算較複雜，簡化為：初始賣空所得 + (初始賣空價 - 現價) * 股數
                current_total_value += (self.position * self.entry_price) + (self.entry_price - row['Close']) * self.position

            self.daily_capital.append({'date': current_date, 'capital': current_total_value})

        # 處理期末仍持倉的情況 (以最後一天收盤價平倉)
        if self.holding and not merged_df.empty:
            last_row = merged_df.iloc[-1]
            last_trade = self.trades[-1]
            last_trade["exit_date"] = last_row['date']
            last_trade["exit_price"] = last_row['Close'] # 以最後一天收盤價平倉

            if self.direction == 'long':
                self.cash += self.position * last_row['Close']
                pnl = (last_row['Close'] - self.entry_price) * self.position
            elif self.direction == 'short':
                self.cash -= self.position * last_row['Close']
                self.cash += self.position * self.entry_price
                pnl = (self.entry_price - last_row['Close']) * self.position
            
            last_trade["pnl"] = pnl
            self.position = 0
            # 更新最後一天的資產
            if self.daily_capital:
                 self.daily_capital[-1]['capital'] = self.cash


        trades_df = pd.DataFrame(self.trades)
        return trades_df, self.cash # 回傳交易紀錄和最終現金

    def calculate_metrics(self, trades_df):
        # 若 'pnl' 欄位不存在則補上
        if 'pnl' not in trades_df.columns:
            trades_df['pnl'] = None
        total_return = (self.cash - self.initial_capital) / self.initial_capital
        total_trades = len(trades_df[trades_df['pnl'].notnull()])
        win_trades = trades_df[trades_df['pnl'] > 0]
        win_rate = len(win_trades) / total_trades if total_trades > 0 else 0
        avg_pnl = trades_df['pnl'].mean() if 'pnl' in trades_df else 0
        return {
            'final_cash': self.cash,
            'total_return': total_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'average_pnl': avg_pnl
        }
