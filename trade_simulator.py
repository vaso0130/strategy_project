import pandas as pd
import numpy as np

class TradeSimulator:
    def __init__(self, initial_capital: int, stop_loss: float, allow_short: bool, stock_symbol: str, short_qty_cap: int = 1000,
                 # New parameters for forced trading
                 enable_forced_trading: bool = False,
                 forced_trade_take_profit_pct: float = 0.05,
                 forced_trade_stop_loss_pct: float = 0.02,
                 forced_trade_use_trailing_stop: bool = False,
                 forced_trade_capital_allocation: float = 0.25): # Added forced_trade_capital_allocation
        self.initial_capital = initial_capital
        self.stop_loss = stop_loss
        self.allow_short = allow_short
        self.stock_symbol = stock_symbol
        self.short_qty_cap = short_qty_cap
        # Store forced trading parameters
        self.enable_forced_trading = enable_forced_trading
        self.forced_trade_take_profit_pct = forced_trade_take_profit_pct
        self.forced_trade_stop_loss_pct = forced_trade_stop_loss_pct
        self.forced_trade_use_trailing_stop = forced_trade_use_trailing_stop
        self.forced_trade_capital_allocation = forced_trade_capital_allocation # Added this assignment
        
        self.reset() # Moved reset call to after all parameters are set

    def reset(self):
        self.cash = self.initial_capital
        self.position = 0
        self.entry_price = 0
        self.trades = []
        self.holding = False # This variable seems unused, consider removing if confirmed
        self.direction = None
        self.daily_capital = []
        self.trade_id_counter = 0
        self.current_trade_id = None

        # Reset forced trade state (for later use)
        self.is_forced_trade = False
        self.forced_trade_entry_price = 0
        self.forced_trade_take_profit_target = 0
        self.forced_trade_stop_loss_target = 0
        self.forced_trade_direction = None
        self.forced_trade_trailing_stop_price = 0 # For trailing stop

    def _calculate_trade_cost(self, price, quantity, action):
        """
        計算交易成本

        Args:
            price (float): 交易價格
            quantity (int): 交易股數
            action (str): 交易行為類型，如 "Buy", "Sell", "ShortEntry", "CoverExit"

        Returns:
            float: 交易成本
        """
        # 假設每股手續費為0.1425%，並設有最低手續費NT$20
        commission_rate = 0.001425
        min_commission = 20

        if action in ["Buy", "Sell"]:
            # 買入或賣出時計算手續費
            cost = max(price * quantity * commission_rate, min_commission)
        elif action in ["ShortEntry", "CoverExit"]:
            # 做空進場或回補時計算手續費
            cost = max(price * quantity * commission_rate, min_commission)
        else:
            cost = 0

        return cost

    def simulate(self, price_df: pd.DataFrame, signal_df: pd.DataFrame):
        self.reset()
        if 'date' not in price_df.columns or 'Close' not in price_df.columns or 'Open' not in price_df.columns or \
           'High' not in price_df.columns or 'Low' not in price_df.columns: # Added High/Low check
            raise ValueError("price_df 必須包含 'date', 'Open', 'High', 'Low', 'Close' 欄位。")
        if 'date' not in signal_df.columns or 'signal' not in signal_df.columns:
            raise ValueError("signal_df 必須包含 'date', 'signal' 欄位。")

        merged_df = pd.merge(price_df, signal_df, on="date", how="left").fillna({'signal':0}) # 用0填充缺失信號

        for i in range(len(merged_df)):
            row = merged_df.iloc[i]
            current_date = row["date"]
            current_price = row["Close"] # Used as default execution price unless overridden by TP/SL
            open_price = row["Open"]
            high_price = row["High"]
            low_price = row["Low"]
            signal = row["signal"]
            original_signal_for_debug = signal # Keep original signal for logging/debugging

            current_total_value = self.cash + (self.position * current_price) # Approximation using close
            self.daily_capital.append({'date': current_date, 'capital': current_total_value})

            # --- Forced Trade TP/SL Check ---
            if self.is_forced_trade:
                executed_forced_sl_tp = False
                if self.forced_trade_direction == "long":
                    # Trailing Stop for forced long
                    if self.forced_trade_use_trailing_stop:
                        self.forced_trade_trailing_stop_price = max(self.forced_trade_trailing_stop_price, high_price * (1 - self.forced_trade_stop_loss_pct))
                        if low_price <= self.forced_trade_trailing_stop_price:
                            signal = -1 # Trigger sell (forced SL)
                            current_price = self.forced_trade_trailing_stop_price # Execute at SL price
                            # print(f"DEBUG [{current_date}] Forced Long TRAILING SL triggered at {current_price}")
                            executed_forced_sl_tp = True
                    # Fixed Stop Loss for forced long
                    elif low_price <= self.forced_trade_stop_loss_target:
                        signal = -1 # Trigger sell (forced SL)
                        current_price = self.forced_trade_stop_loss_target # Execute at SL price
                        # print(f"DEBUG [{current_date}] Forced Long FIXED SL triggered at {current_price}")
                        executed_forced_sl_tp = True
                    # Take Profit for forced long
                    if not executed_forced_sl_tp and high_price >= self.forced_trade_take_profit_target:
                        signal = -1 # Trigger sell (forced TP)
                        current_price = self.forced_trade_take_profit_target # Execute at TP price
                        # print(f"DEBUG [{current_date}] Forced Long TP triggered at {current_price}")
                        executed_forced_sl_tp = True

                elif self.forced_trade_direction == "short":
                    # Trailing Stop for forced short
                    if self.forced_trade_use_trailing_stop:
                        self.forced_trade_trailing_stop_price = min(self.forced_trade_trailing_stop_price, low_price * (1 + self.forced_trade_stop_loss_pct))
                        if high_price >= self.forced_trade_trailing_stop_price:
                            signal = 1 # Trigger cover (forced SL)
                            current_price = self.forced_trade_trailing_stop_price # Execute at SL price
                            # print(f"DEBUG [{current_date}] Forced Short TRAILING SL triggered at {current_price}")
                            executed_forced_sl_tp = True
                    # Fixed Stop Loss for forced short
                    elif high_price >= self.forced_trade_stop_loss_target:
                        signal = 1 # Trigger cover (forced SL)
                        current_price = self.forced_trade_stop_loss_target # Execute at SL price
                        # print(f"DEBUG [{current_date}] Forced Short FIXED SL triggered at {current_price}")
                        executed_forced_sl_tp = True
                    # Take Profit for forced short
                    if not executed_forced_sl_tp and low_price <= self.forced_trade_take_profit_target:
                        signal = 1 # Trigger cover (forced TP)
                        current_price = self.forced_trade_take_profit_target # Execute at TP price
                        # print(f"DEBUG [{current_date}] Forced Short TP triggered at {current_price}")
                        executed_forced_sl_tp = True
                
                if executed_forced_sl_tp:
                    self.is_forced_trade = False # Reset forced trade state after TP/SL

            # --- Regular Stop Loss Check ---
            if not self.is_forced_trade and self.position != 0: # Only apply if not in a forced trade and have a position
                if self.position > 0: # Long position
                    stop_loss_price = self.entry_price * (1 - self.stop_loss)
                    if low_price < stop_loss_price:
                        signal = -1 # Trigger sell
                        current_price = stop_loss_price # Execute at SL price
                        # print(f"DEBUG [{current_date}] Regular Long SL triggered at {current_price}. Original signal: {original_signal_for_debug}")
                elif self.position < 0: # Short position
                    stop_loss_price = self.entry_price * (1 + self.stop_loss)
                    if high_price > stop_loss_price:
                        signal = 1 # Trigger cover
                        current_price = stop_loss_price # Execute at SL price
                        # print(f"DEBUG [{current_date}] Regular Short SL triggered at {current_price}. Original signal: {original_signal_for_debug}")


            # --- Signal Processing ---
            # Forced Buy Signal (2)
            if signal == 2 and self.position == 0 and self.enable_forced_trading:
                self.is_forced_trade = True
                self.forced_trade_direction = "long"
                self.forced_trade_entry_price = open_price # Use day's open for forced trade
                
                # Calculate quantity for forced trade based on capital allocation
                forced_capital_to_use = self.cash * self.forced_trade_capital_allocation # MODIFIED
                
                if self.forced_trade_entry_price > 0:
                    quantity_to_buy = np.floor(forced_capital_to_use / self.forced_trade_entry_price)
                    if quantity_to_buy > 0:
                        self.trade_id_counter += 1
                        self.current_trade_id = self.trade_id_counter
                        trade_cost_effect = self._calculate_trade_cost(self.forced_trade_entry_price, quantity_to_buy, "Buy")
                        self.cash -= (self.forced_trade_entry_price * quantity_to_buy + trade_cost_effect)
                        self.position += quantity_to_buy
                        self.entry_price = self.forced_trade_entry_price # Regular entry price also set
                        
                        self.forced_trade_take_profit_target = self.forced_trade_entry_price * (1 + self.forced_trade_take_profit_pct)
                        self.forced_trade_stop_loss_target = self.forced_trade_entry_price * (1 - self.forced_trade_stop_loss_pct)
                        if self.forced_trade_use_trailing_stop:
                             self.forced_trade_trailing_stop_price = self.forced_trade_entry_price * (1 - self.forced_trade_stop_loss_pct)


                        self.trades.append({
                            "Date": pd.to_datetime(current_date).strftime('%Y%m%d'), "Action": "ForcedBuy",
                            "Symbol": self.stock_symbol, "Price": self.forced_trade_entry_price, "Quantity": quantity_to_buy,
                            "PNL": 0, "Cash": self.cash, "TradeID": self.current_trade_id
                        })
                        # print(f"DEBUG [{current_date}] Forced Buy executed at {self.forced_trade_entry_price}. TP: {self.forced_trade_take_profit_target}, SL: {self.forced_trade_stop_loss_target}")
                    else: # Not enough capital or zero price
                        self.is_forced_trade = False
                        self.forced_trade_direction = None
            # Forced Short Signal (-2)
            elif signal == -2 and self.position == 0 and self.allow_short and self.enable_forced_trading:
                self.is_forced_trade = True
                self.forced_trade_direction = "short"
                self.forced_trade_entry_price = open_price # Use day's open for forced trade

                # Example: forced_capital_to_use = self.cash * self.forced_trade_capital_allocation
                forced_capital_to_use = self.cash * self.forced_trade_capital_allocation # MODIFIED

                if self.forced_trade_entry_price > 0:
                    quantity_based_on_capital = np.floor(forced_capital_to_use / self.forced_trade_entry_price)
                    quantity_to_short = min(quantity_based_on_capital, self.short_qty_cap)
                    
                    if quantity_to_short > 0:
                        self.trade_id_counter += 1
                        self.current_trade_id = self.trade_id_counter
                        trade_cost_effect = self._calculate_trade_cost(self.forced_trade_entry_price, quantity_to_short, "ShortEntry")
                        self.cash -= trade_cost_effect
                        self.position -= quantity_to_short
                        self.entry_price = self.forced_trade_entry_price

                        self.forced_trade_take_profit_target = self.forced_trade_entry_price * (1 - self.forced_trade_take_profit_pct)
                        self.forced_trade_stop_loss_target = self.forced_trade_entry_price * (1 + self.forced_trade_stop_loss_pct)
                        if self.forced_trade_use_trailing_stop:
                            self.forced_trade_trailing_stop_price = self.forced_trade_entry_price * (1 + self.forced_trade_stop_loss_pct)

                        self.trades.append({
                            "Date": pd.to_datetime(current_date).strftime('%Y%m%d'), "Action": "ForcedShort",
                            "Symbol": self.stock_symbol, "Price": self.forced_trade_entry_price, "Quantity": quantity_to_short,
                            "PNL": 0, "Cash": self.cash, "TradeID": self.current_trade_id
                        })
                        # print(f"DEBUG [{current_date}] Forced Short executed at {self.forced_trade_entry_price}. TP: {self.forced_trade_take_profit_target}, SL: {self.forced_trade_stop_loss_target}")
                    else: # Not enough capital or zero price
                        self.is_forced_trade = False
                        self.forced_trade_direction = None

            elif signal == 1: # Buy or Cover
                if self.position == 0: # Open Long
                    affordable_cash = self.cash * 0.99
                    if current_price > 0: # Use current_price (Close) for regular buy
                        quantity_to_buy = np.floor(affordable_cash / current_price)
                        if quantity_to_buy > 0:
                            self.trade_id_counter += 1
                            self.current_trade_id = self.trade_id_counter
                            trade_cost_effect = self._calculate_trade_cost(current_price, quantity_to_buy, "Buy")
                            self.cash -= (current_price * quantity_to_buy + trade_cost_effect)
                            self.position += quantity_to_buy
                            self.entry_price = current_price
                            self.trades.append({
                                "Date": pd.to_datetime(current_date).strftime('%Y%m%d'), "Action": "Buy",
                                "Symbol": self.stock_symbol, "Price": current_price, "Quantity": quantity_to_buy,
                                "PNL": 0, "Cash": self.cash, "TradeID": self.current_trade_id
                            })
                elif self.position < 0: # Cover Short
                    quantity_to_cover = abs(self.position)
                    pnl = (self.entry_price - current_price) * quantity_to_cover # PNL based on current_price (Close)
                    trade_cost_effect = self._calculate_trade_cost(current_price, quantity_to_cover, "CoverExit")
                    self.cash += pnl - trade_cost_effect # Update cash with PNL
                    self.trades.append({
                        "Date": pd.to_datetime(current_date).strftime('%Y%m%d'), "Action": "Cover",
                        "Symbol": self.stock_symbol, "Price": current_price, "Quantity": quantity_to_cover,
                        "PNL": pnl - trade_cost_effect, "Cash": self.cash, "TradeID": self.current_trade_id
                    })
                    self.position = 0
                    self.entry_price = 0
                    self.current_trade_id = None # Reset TradeID after closing round trip

            elif signal == -1: # Sell or Short
                if self.position > 0: # Sell Long
                    quantity_to_sell = self.position
                    pnl = (current_price - self.entry_price) * quantity_to_sell # PNL based on current_price (Close)
                    trade_cost_effect = self._calculate_trade_cost(current_price, quantity_to_sell, "Sell")
                    self.cash += (current_price * quantity_to_sell) - trade_cost_effect # Update cash with PNL
                    self.trades.append({
                        "Date": pd.to_datetime(current_date).strftime('%Y%m%d'), "Action": "Sell",
                        "Symbol": self.stock_symbol, "Price": current_price, "Quantity": quantity_to_sell,
                        "PNL": pnl - trade_cost_effect, "Cash": self.cash, "TradeID": self.current_trade_id
                    })
                    self.position = 0
                    self.entry_price = 0
                    self.current_trade_id = None # Reset TradeID
                elif self.position == 0 and self.allow_short: # Open Short
                    max_short_value_from_cash = self.cash * 0.5
                    quantity_based_on_cash = 0
                    if current_price > 0: # Use current_price (Close) for regular short
                        quantity_based_on_cash = np.floor(max_short_value_from_cash / current_price)
                    quantity_to_short = min(quantity_based_on_cash, self.short_qty_cap)
                    if quantity_to_short > 0:
                        self.trade_id_counter += 1
                        self.current_trade_id = self.trade_id_counter
                        trade_cost_effect = self._calculate_trade_cost(current_price, quantity_to_short, "ShortEntry")
                        self.cash -= trade_cost_effect # Cash changes by commission only for shorting
                        self.position -= quantity_to_short
                        self.entry_price = current_price
                        self.trades.append({
                            "Date": pd.to_datetime(current_date).strftime('%Y%m%d'), "Action": "Short",
                            "Symbol": self.stock_symbol, "Price": current_price, "Quantity": quantity_to_short,
                            "PNL": 0, "Cash": self.cash, "TradeID": self.current_trade_id
                        })
            elif signal == 0: # Hold or Close existing position
                if self.is_forced_trade: # If in a forced trade, signal 0 does not close it. TP/SL must handle it.
                    pass
                elif self.position > 0: # Close Long (regular)
                    quantity_to_sell = self.position
                    pnl = (current_price - self.entry_price) * quantity_to_sell
                    trade_cost_effect = self._calculate_trade_cost(current_price, quantity_to_sell, "Sell")
                    self.cash += (current_price * quantity_to_sell) - trade_cost_effect
                    self.trades.append({
                        "Date": pd.to_datetime(current_date).strftime('%Y%m%d'), "Action": "Sell",
                        "Symbol": self.stock_symbol, "Price": current_price, "Quantity": quantity_to_sell,
                        "PNL": pnl - trade_cost_effect, "Cash": self.cash, "TradeID": self.current_trade_id
                    })
                    self.position = 0
                    self.entry_price = 0
                    self.current_trade_id = None
                elif self.position < 0: # Close Short (regular)
                    quantity_to_cover = abs(self.position)
                    pnl = (self.entry_price - current_price) * quantity_to_cover
                    trade_cost_effect = self._calculate_trade_cost(current_price, quantity_to_cover, "CoverExit")
                    self.cash += pnl - trade_cost_effect
                    self.trades.append({
                        "Date": pd.to_datetime(current_date).strftime('%Y%m%d'), "Action": "Cover",
                        "Symbol": self.stock_symbol, "Price": current_price, "Quantity": quantity_to_cover,
                        "PNL": pnl - trade_cost_effect, "Cash": self.cash, "TradeID": self.current_trade_id
                    })
                    self.position = 0
                    self.entry_price = 0
                    self.current_trade_id = None

        # End of simulation: close any open positions
        if self.position != 0: # Check if any position is open
            current_price_exec = merged_df.iloc[-1]["Close"] # Default execution price
            current_date_exec = merged_df.iloc[-1]["date"]
            action_exec = ""
            pnl_exec = 0
            
            if self.is_forced_trade: # If a forced trade is still open at the end, close it
                # print(f"DEBUG [{current_date_exec}] Closing open FORCED position at end of simulation.")
                if self.forced_trade_direction == "long":
                    action_exec = "Sell"
                    quantity_to_sell = self.position
                    pnl_exec = (current_price_exec - self.forced_trade_entry_price) * quantity_to_sell
                    trade_cost_effect = self._calculate_trade_cost(current_price_exec, quantity_to_sell, "Sell")
                    self.cash += (current_price_exec * quantity_to_sell) - trade_cost_effect
                    self.trades.append({
                        "Date": pd.to_datetime(current_date_exec).strftime('%Y%m%d'), "Action": action_exec, 
                        "Symbol": self.stock_symbol, "Price": current_price_exec, "Quantity": quantity_to_sell, 
                        "PNL": pnl_exec - trade_cost_effect, "Cash": self.cash, "TradeID": self.current_trade_id
                    })
                elif self.forced_trade_direction == "short":
                    action_exec = "Cover"
                    quantity_to_cover = abs(self.position)
                    pnl_exec = (self.forced_trade_entry_price - current_price_exec) * quantity_to_cover
                    trade_cost_effect = self._calculate_trade_cost(current_price_exec, quantity_to_cover, "CoverExit")
                    self.cash += pnl_exec - trade_cost_effect
                    self.trades.append({
                        "Date": pd.to_datetime(current_date_exec).strftime('%Y%m%d'), "Action": action_exec, 
                        "Symbol": self.stock_symbol, "Price": current_price_exec, "Quantity": quantity_to_cover, 
                        "PNL": pnl_exec - trade_cost_effect, "Cash": self.cash, "TradeID": self.current_trade_id
                    })
                self.is_forced_trade = False # Reset
                self.forced_trade_direction = None

            elif self.position > 0: # Regular long position
                # print(f"DEBUG [{current_date_exec}] Closing open REGULAR LONG position at end of simulation.")
                action_exec = "Sell"
                quantity_to_sell = self.position
                pnl_exec = (current_price_exec - self.entry_price) * quantity_to_sell
                trade_cost_effect = self._calculate_trade_cost(current_price_exec, quantity_to_sell, "Sell")
                self.cash += (current_price_exec * quantity_to_sell) - trade_cost_effect
                self.trades.append({
                    "Date": pd.to_datetime(current_date_exec).strftime('%Y%m%d'), "Action": action_exec, "Symbol": self.stock_symbol,
                    "Price": current_price_exec, "Quantity": quantity_to_sell, "PNL": pnl_exec - trade_cost_effect, "Cash": self.cash,
                    "TradeID": self.current_trade_id 
                })
            elif self.position < 0: # Regular short position
                # print(f"DEBUG [{current_date_exec}] Closing open REGULAR SHORT position at end of simulation.")
                action_exec = "Cover"
                quantity_to_cover = abs(self.position)
                pnl_exec = (self.entry_price - current_price_exec) * quantity_to_cover
                trade_cost_effect = self._calculate_trade_cost(current_price_exec, quantity_to_cover, "CoverExit")
                self.cash += pnl_exec - trade_cost_effect
                self.trades.append({
                    "Date": pd.to_datetime(current_date_exec).strftime('%Y%m%d'), "Action": action_exec, "Symbol": self.stock_symbol,
                    "Price": current_price_exec, "Quantity": quantity_to_cover, "PNL": pnl_exec - trade_cost_effect, "Cash": self.cash,
                    "TradeID": self.current_trade_id
                })
            
            self.position = 0
            self.current_trade_id = None # Reset
        
        if self.daily_capital and self.daily_capital[-1]['date'] == merged_df.iloc[-1]["date"]:
             self.daily_capital[-1]['capital'] = self.cash
        elif merged_df.empty: # Handle case where merged_df is empty
            pass
        else: # Append last day's capital if not already updated
            self.daily_capital.append({'date': merged_df.iloc[-1]["date"], 'capital': self.cash })


        return pd.DataFrame(self.trades), self.cash, pd.DataFrame(self.daily_capital)

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
