import pandas as pd

class TradeSimulator:
    def __init__(self, initial_capital=1000000, stop_loss=0.08, allow_short=True):
        self.initial_capital = initial_capital
        self.stop_loss = stop_loss
        self.allow_short = allow_short
        self.reset()

    def reset(self):
        self.cash = self.initial_capital
        self.position = 0
        self.entry_price = 0
        self.trades = []
        self.holding = False
        self.direction = None  # 'long' or 'short'

    def simulate(self, df):
        """
        df 需包含 ['date', 'close', 'signal'] 欄位
        signal: 1=做多, -1=放空, 0=觀望
        """
        for i, row in df.iterrows():
            date = row['date']
            price = row['close']
            signal = row['signal']

            # 平倉條件：停損或反向訊號
            if self.holding:
                change = (price - self.entry_price) / self.entry_price
                if self.direction == 'short':
                    change = -change
                if change <= -self.stop_loss or (signal != 0 and signal != (1 if self.direction == 'long' else -1)):
                    pnl = (price - self.entry_price) * self.position if self.direction == 'long' else (self.entry_price - price) * self.position
                    self.cash += pnl
                    self.trades.append({
                        'entry_date': None,
                        'entry_price': None,
                        'exit_date': date,
                        'exit_price': price,
                        'pnl': pnl,
                        'side': self.direction
                    })
                    self.position = 0
                    self.holding = False
                    self.direction = None

            # 開倉條件
            if not self.holding and signal != 0:
                self.entry_price = price
                self.position = self.cash // price
                self.direction = 'long' if signal == 1 else 'short'
                self.holding = True
                if self.direction == 'long':
                    self.cash -= self.position * price
                self.trades.append({
                    'entry_date': date,
                    'entry_price': price,
                    'exit_date': None,
                    'exit_price': None,
                    'pnl': None,
                    'side': self.direction
                })

        # 強制平倉最後一天
        if self.holding:
            final_price = df.iloc[-1]['close']
            pnl = (final_price - self.entry_price) * self.position if self.direction == 'long' else (self.entry_price - final_price) * self.position
            self.cash += pnl
            self.trades[-1]['exit_date'] = df.iloc[-1]['date']
            self.trades[-1]['exit_price'] = final_price
            self.trades[-1]['pnl'] = pnl

        return pd.DataFrame(self.trades), self.cash

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
