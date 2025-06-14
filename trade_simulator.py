import pandas as pd
import numpy as np
import config # Import config

class TradeSimulator:
    def __init__(self, initial_capital: int, stop_loss: float, allow_short: bool, stock_symbol: str, short_qty_cap: int = 1000,
                 # New parameters for forced trading
                 enable_forced_trading: bool = False,
                 forced_trade_take_profit_pct: float = 0.05,
                 forced_trade_stop_loss_pct: float = 0.02,
                 forced_trade_use_trailing_stop: bool = False,
                 forced_trade_capital_allocation: float = 0.25, # Added forced_trade_capital_allocation
                 # New parameters for trade unit and precision
                 trade_unit: int = 1, 
                 price_precision_rules: dict = None): 
        self.initial_capital = initial_capital
        self.stop_loss = stop_loss # For regular trades
        self.allow_short = allow_short
        self.stock_symbol = stock_symbol
        self.short_qty_cap = short_qty_cap
        
        self.enable_forced_trading = enable_forced_trading
        self.forced_trade_take_profit_pct = forced_trade_take_profit_pct
        self.forced_trade_stop_loss_pct = forced_trade_stop_loss_pct
        self.forced_trade_use_trailing_stop = forced_trade_use_trailing_stop
        self.forced_trade_capital_allocation = forced_trade_capital_allocation

        self.trade_unit = trade_unit
        self.price_precision_rules = price_precision_rules if price_precision_rules else {}
        self._log_trade_details = True # 可在初始化時設定是否記錄詳細交易日誌

        # Initialize attributes that will be used later
        self.cash = self.initial_capital
        self.current_position = 0  # Shares held, positive for long, negative for short
        self.current_position_quantity = 0 # Absolute quantity of shares
        self.entry_price = 0
        self.portfolio_value = 0
        self.trade_log = []  # Initialize trade_log here
        self.daily_capital = [] # Initialize daily_capital here
        self.trade_id_counter = 0
        self.cumulative_pnl = 0.0  # Initialize cumulative_pnl
        self.last_actual_trade_pnl = 0.0 # Initialize last_actual_trade_pnl

        # For regular trades
        self.current_trade_id = None
        self.current_trade_entry_price = 0
        self.current_trade_quantity = 0
        self.current_trade_take_profit_price = None
        self.current_trade_stop_loss_price = None

        # For forced trades
        self.current_forced_trade_id = None
        self.current_forced_trade_entry_price = 0
        self.current_forced_trade_quantity = 0
        self.current_forced_trade_take_profit_price = None
        self.current_forced_trade_stop_loss_price = None
        self.current_forced_trade_highest_price_since_entry = 0 # For trailing stop
        self.current_forced_trade_lowest_price_since_entry = float('inf') # For trailing stop

        self.is_forced_trade = False # Initialize is_forced_trade
        self.forced_trade_direction = None # Initialize forced_trade_direction
        self.forced_trade_entry_price = 0.0
        self.forced_trade_take_profit_target = 0.0
        self.forced_trade_stop_loss_target = 0.0
        self.forced_trade_trailing_stop_price = 0.0 # Initialized, will be updated if trailing stop is used

        # Load from config
        self.commission_rate = config.COMMISSION_RATE
        self.min_commission = config.MIN_COMMISSION

        # 新增：追蹤融券保證金凍結
        self.margin_held = 0.0  # 追蹤融券保證金凍結

    def get_last_trade_pnl(self):
        """獲取最近一筆已實現損益的交易的 PNL。"""
        # Assuming self.last_actual_trade_pnl is correctly updated in buy/sell upon trade closure.
        return self.last_actual_trade_pnl

    def get_cumulative_pnl(self):
        """獲取當前的累計總損益。"""
        # Assuming self.cumulative_pnl is correctly updated in buy/sell.
        return self.cumulative_pnl

    def reset(self):
        self.cash = self.initial_capital
        self.current_position = 0
        self.current_position_quantity = 0 # Ensure reset
        self.entry_price = 0
        self.trade_log = [] # Changed from self.trades = []
        # self.holding = False # This variable seems unused, consider removing if confirmed
        self.direction = None # "long", "short" for regular trades
        self.daily_capital = []
        self.trade_id_counter = 0
        self.current_trade_id = None
        self.portfolio_value = 0.0 # Reset portfolio_value
        self.cumulative_pnl = 0.0  # Reset cumulative_pnl
        self.last_actual_trade_pnl = 0.0 # Reset last_actual_trade_pnl

        self.is_forced_trade = False # True if a forced trade is currently active
        self.forced_trade_entry_price = 0
        self.forced_trade_take_profit_target = 0
        self.forced_trade_stop_loss_target = 0
        self.forced_trade_direction = None # "long" or "short" for forced trades
        self.forced_trade_trailing_stop_price = 0
        self.current_forced_trade_id = None # Separate ID for forced trades if needed, or use current_trade_id

    def _get_price_precision(self, price: float) -> int:
        """根據價格決定小數點位數"""
        for (lower, upper), precision in self.price_precision_rules.items():
            if lower <= price < upper:
                return precision
        return 0 # 預設情況 (例如價格非常高或規則未涵蓋)

    def _round_price(self, price: float) -> float:
        """根據價格級距規則調整價格"""
        precision = self._get_price_precision(price)
        if precision == 0:
            return round(price)
        return round(price, precision)

    def _adjust_quantity(self, quantity: float) -> int:
        """將股數調整為交易單位的整數倍，並確保為整數"""
        if self.trade_unit == 1: # 允許零股
            return int(round(quantity))
        else: # 只能是 N 張
            return int(round(quantity / self.trade_unit)) * self.trade_unit

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
        # 使用從 config 加載的設定
        if action in ["Buy", "Sell", "ShortEntry", "CoverExit"]:
            cost = max(price * quantity * self.commission_rate, self.min_commission)
        else:
            cost = 0
        return cost

    def update_portfolio_value(self, current_price: float):
        """Updates the current portfolio value based on the current price."""
        if self.current_position != 0:
            self.portfolio_value = self.current_position * current_price
        else:
            self.portfolio_value = 0.0

    def is_forced_trade_active(self) -> bool:
        """Checks if a forced trade is currently active."""
        return self.is_forced_trade

    def buy(self, price: float, quantity: int, date_to_log, trade_type: str = "Regular", capital_allocation_factor: float = 1.0): # Added capital_allocation_factor
        original_price_arg = price
        price = self._round_price(price)

        # Adjust quantity based on capital_allocation_factor if it's an entry trade
        # This is a simplified assumption; actual capital for trade is determined in main.py
        # Here, we assume 'quantity' has already been scaled by the factor if it's a new position.
        # For closing trades, quantity should be based on existing position.

        print(f"SIMULATOR: Buy {quantity} at {price} (Original: {original_price_arg}) (Type: {trade_type}) Date: {date_to_log}")
        
        adjusted_quantity = self._adjust_quantity(quantity)
        if adjusted_quantity == 0:
            print(f"SIMULATOR WARNING: Buy quantity {quantity} adjusted to 0. No trade.")
            return None

        cost_of_shares = adjusted_quantity * price
        trade_commission = self._calculate_trade_cost(price, adjusted_quantity, "Buy")
        
        self.trade_id_counter += 1
        current_trade_id = self.trade_id_counter
        
        pnl = 0.0
        log_action = "Buy" # Default action

        if self.current_position < 0:  # Buying to cover a short
            if trade_type == "Regular": log_action = "CoverExit"
            elif trade_type == "RegularSL": log_action = "CoverSL"
            elif trade_type == "ForcedTSL": log_action = "ForcedCoverTSL"
            elif trade_type == "ForcedSL": log_action = "ForcedCoverSL"
            elif trade_type == "ForcedTP": log_action = "ForcedCoverTP"
            elif trade_type == "EndOfSim": log_action = "CoverEndOfSim"
            else: log_action = "CoverExit" # Default cover action

            # Calculate PNL for short trade
            # entry_price for short is the price we sold at
            pnl = (self.entry_price - price) * adjusted_quantity - trade_commission 
            # For short, PNL = (Sell Price - Buy Price) * Quantity - Commission
            # Here, self.entry_price is the initial sell price. 'price' is the current buy price.

            self.cash -= cost_of_shares # Cash out for buying shares
            self.cash -= trade_commission # Cash out for commission
            
            # Release margin
            margin_to_release = self.entry_price * adjusted_quantity * config.SHORT_MARGIN_RATE_STOCK
            margin_to_release = min(margin_to_release, self.margin_held)
            self.cash += margin_to_release # Margin returns to cash
            self.margin_held -= margin_to_release
            
            self.current_position += adjusted_quantity
            self.current_position_quantity -= adjusted_quantity # Decrease absolute quantity

            self.last_actual_trade_pnl = pnl # Update last trade PNL
            self.cumulative_pnl += pnl    # Update cumulative PNL

            if self.current_position == 0: # Position closed
                self.entry_price = 0
                self.direction = None
                if self.is_forced_trade and trade_type.startswith("Forced"): self.is_forced_trade = False


        elif self.current_position == 0: # Opening a new long position
            log_action = "BuyEntry"
            if self.cash < cost_of_shares + trade_commission:
                print(f"SIMULATOR WARNING: Not enough cash for BuyEntry. Cash: {self.cash}, Needed: {cost_of_shares + trade_commission}")
                return None
            self.cash -= (cost_of_shares + trade_commission)
            self.current_position += adjusted_quantity
            self.current_position_quantity += adjusted_quantity
            self.entry_price = price # Set entry price for the new long position
            self.direction = "long"
            # PNL is 0 for an entry trade
            self.last_actual_trade_pnl = 0.0 
            if trade_type.startswith("Forced"):
                self.is_forced_trade = True
                self.forced_trade_direction = "long"
                self.forced_trade_entry_price = price
                self.forced_trade_take_profit_target = self._round_price(price * (1 + self.forced_trade_take_profit_pct))
                self.forced_trade_stop_loss_target = self._round_price(price * (1 - self.forced_trade_stop_loss_pct))
                if self.forced_trade_use_trailing_stop:
                    self.forced_trade_trailing_stop_price = self.forced_trade_stop_loss_target
                else:
                    self.forced_trade_trailing_stop_price = 0.0 # Or specific non-triggering value for long


        else: # Adding to an existing long position (less common in simple strategies)
            log_action = "BuyAdd"
            if self.cash < cost_of_shares + trade_commission:
                print(f"SIMULATOR WARNING: Not enough cash to add to long position. Cash: {self.cash}, Needed: {cost_of_shares + trade_commission}")
                return None
            # Update average entry price
            self.entry_price = (self.entry_price * self.current_position + price * adjusted_quantity) / (self.current_position + adjusted_quantity)
            self.cash -= (cost_of_shares + trade_commission)
            self.current_position += adjusted_quantity
            self.current_position_quantity += adjusted_quantity
            # PNL is 0 for an entry/add trade
            self.last_actual_trade_pnl = 0.0

        self.update_portfolio_value(price)

        trade_log_entry = {
            "Date": pd.Timestamp(date_to_log).strftime('%Y%m%d'), # Ensure consistent date format
            "Action": log_action, "Symbol": self.stock_symbol, "Price": price, 
            "Quantity": adjusted_quantity, 
            "PNL": round(pnl, 2), # PNL is now calculated for closing trades
            "Cash": round(self.cash, 2), 
            "TradeID": current_trade_id, # Use the generated trade ID
            "PortfolioValue": round(self.portfolio_value + self.cash, 2), # Portfolio value should be total assets
            "TradeType": trade_type,
            "CumulativePNL": round(self.cumulative_pnl, 2) # Log cumulative PNL
        }
        self.trade_log.append(trade_log_entry)
        return trade_log_entry

    def sell(self, price: float, quantity: int, date_to_log, trade_type: str = "Regular", capital_allocation_factor: float = 1.0): # Added capital_allocation_factor
        original_price_arg = price
        price = self._round_price(price)
        
        # Similar to buy, quantity adjustment logic for entry trades would be handled before calling sell.
        # For closing trades, quantity is based on existing position.

        print(f"SIMULATOR: Sell {quantity} at {price} (Original: {original_price_arg}) (Type: {trade_type}) Date: {date_to_log}")

        adjusted_quantity = self._adjust_quantity(quantity)
        if adjusted_quantity == 0:
            print(f"SIMULATOR WARNING: Sell quantity {quantity} adjusted to 0. No trade.")
            return None
        
        if self.current_position > 0 and adjusted_quantity > self.current_position_quantity: # Selling more than held (long)
            print(f"SIMULATOR WARNING: Attempting to sell {adjusted_quantity} but only hold {self.current_position_quantity}. Adjusting to sell all.")
            adjusted_quantity = self.current_position_quantity
        
        if self.current_position == 0 and not self.allow_short and not trade_type.startswith("Forced"): # Cannot open short
            print(f"SIMULATOR WARNING: Short selling not allowed and not a forced trade. Sell order for {adjusted_quantity} ignored.")
            return None
        
        # If trying to short more than cap (only for new short entries)
        if self.current_position == 0 and self.allow_short and adjusted_quantity > self.short_qty_cap:
            print(f"SIMULATOR WARNING: Attempting to short {adjusted_quantity} exceeds cap {self.short_qty_cap}. Adjusting to cap.")
            adjusted_quantity = self.short_qty_cap


        proceeds_from_shares = adjusted_quantity * price
        cost_action_for_commission = "Sell" if self.current_position > 0 else "ShortEntry"
        trade_commission = self._calculate_trade_cost(price, adjusted_quantity, cost_action_for_commission)

        self.trade_id_counter += 1
        current_trade_id = self.trade_id_counter
        
        pnl = 0.0
        log_action = "Sell" # Default action

        if self.current_position > 0:  # Selling to close a long position
            if trade_type == "Regular": log_action = "SellExit"
            elif trade_type == "RegularSL": log_action = "SellSL"
            elif trade_type == "ForcedTSL": log_action = "ForcedSellTSL"
            elif trade_type == "ForcedSL": log_action = "ForcedSellSL"
            elif trade_type == "ForcedTP": log_action = "ForcedSellTP"
            elif trade_type == "EndOfSim": log_action = "SellEndOfSim"
            else: log_action = "SellExit"

            # Calculate PNL for long trade
            # entry_price for long is the price we bought at
            pnl = (price - self.entry_price) * adjusted_quantity - trade_commission
            
            self.cash += proceeds_from_shares # Cash in from selling shares
            self.cash -= trade_commission   # Cash out for commission
            
            self.current_position -= adjusted_quantity
            self.current_position_quantity -= adjusted_quantity

            self.last_actual_trade_pnl = pnl
            self.cumulative_pnl += pnl

            if self.current_position == 0: # Position closed
                self.entry_price = 0
                self.direction = None
                if self.is_forced_trade and trade_type.startswith("Forced"): self.is_forced_trade = False

        elif self.allow_short or trade_type.startswith("Forced"):  # Opening a new short position or adding to it
            if self.current_position == 0 : # New Short Entry
                log_action = "ShortEntry"
                self.entry_price = price # Set entry price for the new short position
                self.direction = "short"
                if trade_type.startswith("Forced"):
                    self.is_forced_trade = True
                    self.forced_trade_direction = "short"
                    self.forced_trade_entry_price = price
                    self.forced_trade_take_profit_target = self._round_price(price * (1 - self.forced_trade_take_profit_pct))
                    self.forced_trade_stop_loss_target = self._round_price(price * (1 + self.forced_trade_stop_loss_pct))
                    if self.forced_trade_use_trailing_stop:
                        self.forced_trade_trailing_stop_price = self.forced_trade_stop_loss_target
                    else:
                        self.forced_trade_trailing_stop_price = float('inf') # Or specific non-triggering value for short
                 # PNL is 0 for an entry trade
                self.last_actual_trade_pnl = 0.0
            else: # Adding to existing short
                log_action = "ShortAdd"
                # Update average entry price for short
                self.entry_price = (self.entry_price * abs(self.current_position) + price * adjusted_quantity) / (abs(self.current_position) + adjusted_quantity)
                 # PNL is 0 for an add trade
                self.last_actual_trade_pnl = 0.0

            self.cash += proceeds_from_shares # Cash in from selling shares (borrowed)
            self.cash -= trade_commission   # Cash out for commission

            # Hold margin
            margin_to_hold = price * adjusted_quantity * config.SHORT_MARGIN_RATE_STOCK
            if self.cash < margin_to_hold: # Not enough cash for margin after commission
                print(f"SIMULATOR WARNING: Not enough cash for short margin. Cash: {self.cash}, Margin Needed: {margin_to_hold}. Short trade failed/reduced.")
                # This part needs careful handling: either reject trade or reduce quantity.
                # For simplicity, let's assume if we can't cover margin, the trade might fail or be reduced.
                # Here we'll just print a warning. A more robust system would adjust/reject.
            self.cash -= margin_to_hold
            self.margin_held += margin_to_hold
            
            self.current_position -= adjusted_quantity # Position becomes more negative
            self.current_position_quantity += adjusted_quantity # Absolute quantity increases
           
        else: # Should not happen if logic is correct (e.g. trying to short when not allowed and not forced)
            print(f"SIMULATOR ERROR: Sell condition not met. Position: {self.current_position}, AllowShort: {self.allow_short}, TradeType: {trade_type}")
            return None

        self.update_portfolio_value(price)

        trade_log_entry = {
            "Date": pd.Timestamp(date_to_log).strftime('%Y%m%d'),
            "Action": log_action, "Symbol": self.stock_symbol, "Price": price, 
            "Quantity": adjusted_quantity, 
            "PNL": round(pnl, 2),
            "Cash": round(self.cash, 2), 
            "TradeID": current_trade_id,
            "PortfolioValue": round(self.portfolio_value + self.cash + self.margin_held, 2), # Portfolio value includes cash and held margin
            "TradeType": trade_type,
            "CumulativePNL": round(self.cumulative_pnl, 2)
        }
        self.trade_log.append(trade_log_entry)
        return trade_log_entry

    def check_stop_loss_take_profit(self, current_price: float, date_to_log): # Added date_to_log
        """Checks and executes stop-loss or take-profit for regular trades."""
        
        if self.is_forced_trade or self.current_position == 0 or not self.direction:
            return None # Not a regular trade or no position

        trade_log = None
        # Regular Stop Loss
        if self.direction == "long":
            stop_loss_price = self._round_price(self.entry_price * (1 - self.stop_loss))
            if current_price <= stop_loss_price:
                print(f"SIMULATOR: Regular Long SL triggered at {stop_loss_price}")
                trade_log = self.sell(stop_loss_price, abs(self.current_position), date_to_log, trade_type="RegularSL") # Pass date
        elif self.direction == "short":
            stop_loss_price = self._round_price(self.entry_price * (1 + self.stop_loss))
            if current_price >= stop_loss_price:
                print(f"SIMULATOR: Regular Short SL triggered at {stop_loss_price}")
                trade_log = self.buy(stop_loss_price, abs(self.current_position), date_to_log, trade_type="RegularSL") # Pass date
        
        if trade_log:
            self.direction = None # Position closed
            # PNL is handled within buy/sell
        return trade_log

    def check_forced_trade_closure(self, current_price: float, date_to_log): # Added date_to_log
        """Checks and executes TP/SL for FORCED trades."""
        if not self.is_forced_trade or self.current_position == 0:
            return None

        trade_log = None
        price_to_trade_at = current_price # Default, might be overridden by SL/TP target price

        if self.forced_trade_direction == "long":
            # Update trailing stop for long
            if self.forced_trade_use_trailing_stop:
                potential_new_trailing_stop = self._round_price(current_price * (1 - self.forced_trade_stop_loss_pct))
                self.forced_trade_trailing_stop_price = max(self.forced_trade_trailing_stop_price, potential_new_trailing_stop)
                
                if current_price <= self.forced_trade_trailing_stop_price:
                    print(f"SIMULATOR: Forced Long TRAILING SL triggered at {self.forced_trade_trailing_stop_price}")
                    price_to_trade_at = self.forced_trade_trailing_stop_price
                    trade_log = self.sell(price_to_trade_at, abs(self.current_position), date_to_log, trade_type="ForcedTSL") # Pass date
            # Fixed SL for long (only if not using trailing or trailing not hit)
            elif current_price <= self.forced_trade_stop_loss_target and not trade_log:
                print(f"SIMULATOR: Forced Long FIXED SL triggered at {self.forced_trade_stop_loss_target}")
                price_to_trade_at = self.forced_trade_stop_loss_target
                trade_log = self.sell(price_to_trade_at, abs(self.current_position), date_to_log, trade_type="ForcedSL") # Pass date
            
            # TP for long (only if no SL hit)
            if not trade_log and current_price >= self.forced_trade_take_profit_target:
                print(f"SIMULATOR: Forced Long TP triggered at {self.forced_trade_take_profit_target}")
                price_to_trade_at = self.forced_trade_take_profit_target
                trade_log = self.sell(price_to_trade_at, abs(self.current_position), date_to_log, trade_type="ForcedTP") # Pass date

        elif self.forced_trade_direction == "short":
            # Update trailing stop for short
            if self.forced_trade_use_trailing_stop:
                potential_new_trailing_stop = self._round_price(current_price * (1 + self.forced_trade_stop_loss_pct))
                # For short, trailing stop moves down, so min
                self.forced_trade_trailing_stop_price = min(self.forced_trade_trailing_stop_price if self.forced_trade_trailing_stop_price > 0 else float('inf'), potential_new_trailing_stop)

                if current_price >= self.forced_trade_trailing_stop_price:
                    print(f"SIMULATOR: Forced Short TRAILING SL triggered at {self.forced_trade_trailing_stop_price}")
                    price_to_trade_at = self.forced_trade_trailing_stop_price
                    trade_log = self.buy(price_to_trade_at, abs(self.current_position), date_to_log, trade_type="ForcedTSL") # Pass date
            # Fixed SL for short (only if not using trailing or trailing not hit)
            elif current_price >= self.forced_trade_stop_loss_target and not trade_log:
                print(f"SIMULATOR: Forced Short FIXED SL triggered at {self.forced_trade_stop_loss_target}")
                price_to_trade_at = self.forced_trade_stop_loss_target
                trade_log = self.buy(price_to_trade_at, abs(self.current_position), date_to_log, trade_type="ForcedSL") # Pass date

            # TP for short (only if no SL hit)
            if not trade_log and current_price <= self.forced_trade_take_profit_target:
                print(f"SIMULATOR: Forced Short TP triggered at {self.forced_trade_take_profit_target}")
                price_to_trade_at = self.forced_trade_take_profit_target
                trade_log = self.buy(price_to_trade_at, abs(self.current_position), date_to_log, trade_type="ForcedTP") # Pass date

        if trade_log: # If a closure happened
            self.is_forced_trade = False # Reset forced trade state
            self.forced_trade_direction = None
            # PNL is handled within buy/sell
        return trade_log
        
    def simulate(self, price_df: pd.DataFrame, signal_df: pd.DataFrame, 
                 # The optimizer_capital_allocation_pct is more like a global setting for the optimizer's context.
                 # The LLM's dynamic capital_allocation_factor will be passed from main.py for each trade decision.
                 optimizer_capital_allocation_pct: float = 0.5): 
        self.reset()

        if not all(col in price_df.columns for col in ['date', 'Close', 'Open', 'High', 'Low']):
            raise ValueError("Price data must contain 'date', 'Close', 'Open', 'High', 'Low' columns.")
        if not all(col in signal_df.columns for col in ['date', 'signal']): # Assuming 'signal' is the primary signal column
            raise ValueError("Signal data must contain 'date' and 'signal' columns.")

        price_df['date'] = pd.to_datetime(price_df['date'])
        signal_df['date'] = pd.to_datetime(signal_df['date'])
        merged_df = pd.merge(price_df, signal_df, on="date", how="left")
        # Forward fill signals for days without new signals, common in some strategies
        merged_df['signal'] = merged_df['signal'].ffill().fillna(0)


        for i in range(len(merged_df)):
            row = merged_df.iloc[i]
            current_date = row['date']
            current_price_close = self._round_price(row['Close']) # Use close for daily checks, open for trades
            current_price_open = self._round_price(row['Open'])   # Assume trades happen at open
            signal = row['signal'] # This is the raw signal from strategy, not the LLM decision yet

            # Record daily capital BEFORE any trades for the day
            self.update_portfolio_value(current_price_close) # Update with close of previous day or current day before trade
            self.record_daily_capital(current_date)

            # 1. Check for forced trade closure (uses current_price_open or a more granular price if available)
            if self.is_forced_trade_active():
                self.check_forced_trade_closure(current_price_open, current_date) # Pass date

            # 2. Check for regular trade stop-loss/take-profit (uses current_price_open)
            if not self.is_forced_trade_active() and self.current_position != 0:
                 self.check_stop_loss_take_profit(current_price_open, current_date) # Pass date

            # 3. Process new signals (this part will be heavily modified by main.py's logic)
            # The simple simulator here just acts on a raw signal.
            # In the integrated version, main.py will call buy/sell directly based on LLM + strategy.
            # This loop is more for a standalone simulator.
            # For now, we'll keep a simplified version.
            # The actual buy/sell calls with quantities determined by LLM factor will come from main.py.
            
            # Example: if signal == 1 and self.current_position == 0: # Buy signal
            #     # Quantity determination would be complex, involving REGULAR_TRADE_CAPITAL_ALLOCATION * llm_factor
            #     # For this standalone simulator, let's assume a fixed quantity or simple logic
            #     qty_to_buy = self._adjust_quantity(self.cash * optimizer_capital_allocation_pct / current_price_open)
            #     if qty_to_buy > 0 : self.buy(current_price_open, qty_to_buy, current_date, trade_type="RegularEntry")
            # elif signal == -1 and self.current_position == 0 and self.allow_short: # Short signal
            #     qty_to_short = self._adjust_quantity(self.cash * optimizer_capital_allocation_pct / current_price_open)
            #     if qty_to_short > 0 : self.sell(current_price_open, qty_to_short, current_date, trade_type="ShortEntry")
            # elif signal == 0 and self.current_position > 0: # Exit long
            #     self.sell(current_price_open, self.current_position_quantity, current_date, trade_type="RegularExit")
            # elif signal == 0 and self.current_position < 0: # Cover short
            #     self.buy(current_price_open, self.current_position_quantity, current_date, trade_type="CoverExit")
            pass # The main trading decisions will be driven by main.py

        if self.current_position != 0 and not merged_df.empty:
            last_day_info = merged_df.iloc[-1]
            last_price = self._round_price(last_day_info['Close'])
            last_date = last_day_info['date']
            print(f"SIMULATOR: End of simulation. Closing position of {self.current_position} at {last_price} on {last_date}")
            if self.current_position > 0:
                self.sell(last_price, self.current_position_quantity, last_date, trade_type="EndOfSim")
            elif self.current_position < 0:
                self.buy(last_price, self.current_position_quantity, last_date, trade_type="EndOfSim")
        
        # Final capital record for the very last day if not already recorded
        if not merged_df.empty:
            self.update_portfolio_value(merged_df.iloc[-1]['Close'])
            self.record_daily_capital(merged_df.iloc[-1]['date'])

        return self.get_trade_log_df(), self.get_daily_capital_df()

    def record_daily_capital(self, date_to_log): # Changed date to date_to_log for clarity
        current_date_str = pd.Timestamp(date_to_log).strftime('%Y-%m-%d')
        
        # Calculate current total capital: cash + value of current holdings (if any) + margin held (for shorts)
        current_holdings_value = 0
        if self.current_position != 0 and self.entry_price != 0: # Need a current market price here ideally
             # This is tricky: portfolio value for daily capital should use the day's closing price.
             # For simplicity, if called after update_portfolio_value, it uses the price from there.
             current_holdings_value = self.portfolio_value # self.portfolio_value is just position * price

        total_capital = self.cash + current_holdings_value + self.margin_held

        if not self.daily_capital or self.daily_capital[-1]['date'] != current_date_str:
            self.daily_capital.append({
                "date": current_date_str,
                "capital": round(total_capital, 2)
            })

    def get_trade_log_df(self):
        """將內部交易日誌轉換為 DataFrame。"""
        # 確保即使 trade_log 為空，也回傳正確結構的 DataFrame
        # 欄位名稱應與 utils/metrics.py 中 _convert_trades_to_paired_format 期望的輸入一致
        # 且與 optimizer.py 中 sim.simulate 回傳的 trade_log_df 欄位一致
        # 主要欄位：Date, Action, Symbol, Price, Quantity, PNL, TradeID, TradeType
        
        # 修正：使用 self.trade_log 而不是 self.trades
        if not self.trade_log:
            # Define columns and their dtypes for an empty DataFrame
            # Ensure 'TradeType' is included here
            expected_columns_with_types = {
                "Date": "datetime64[ns]", "Action": str, "Symbol": str, "Price": float,
                "Quantity": "Int64", "PNL": float, "Cash": float, "TradeID": "Int64",
                "PortfolioValue": float, "TradeType": str
            }
            # Create an empty DataFrame with specified columns and dtypes
            df = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in expected_columns_with_types.items()})
            # For "Int64", explicitly use Int64Dtype if pandas version supports it well for empty series
            if "Quantity" in df.columns: df["Quantity"] = pd.Series(dtype=pd.Int64Dtype())
            if "TradeID" in df.columns: df["TradeID"] = pd.Series(dtype=pd.Int64Dtype())
            return df

        df = pd.DataFrame(self.trade_log)

        expected_columns = {
            "Date": "datetime64[ns]", "Action": str, "Symbol": str, "Price": float,
            "Quantity": "Int64", "PNL": float, "Cash": float, "TradeID": "Int64",
            "PortfolioValue": float, "TradeType": str # Ensure TradeType is expected
        }

        if not df.empty:
            for col, dtype in expected_columns.items():
                if col in df.columns:
                    # Attempt to convert, handling potential all-NaN or mixed type issues gracefully
                    try:
                        if df[col].isnull().all():
                            if dtype == "Int64":
                                df[col] = pd.Series(dtype=pd.Int64Dtype(), index=df.index)
                            elif dtype == float:
                                df[col] = pd.Series(dtype=np.float64, index=df.index)
                            elif dtype == "datetime64[ns]":
                                df[col] = pd.to_datetime(pd.Series(dtype='datetime64[ns]', index=df.index), errors='coerce')
                            elif dtype == str:
                                df[col] = pd.Series(dtype=str, index=df.index)
                            else:
                                df[col] = pd.Series(dtype=object, index=df.index) # Fallback for other types
                        elif dtype == "datetime64[ns]":
                            df[col] = pd.to_datetime(df[col], format='%Y%m%d', errors='coerce')
                        elif dtype == "Int64":
                             # Convert to numeric first to handle non-integer strings like "1.0" before Int64
                            df[col] = pd.to_numeric(df[col], errors='coerce').astype(pd.Int64Dtype())
                        else:
                            df[col] = df[col].astype(dtype)
                    except Exception as e:
                        print(f"Warning: Could not convert column '{col}' to {dtype}. Error: {e}")
                        # Fallback or leave as is, depending on desired robustness
                        if dtype == str and col not in df.columns.get_indexer_for(df.select_dtypes(include=object).columns): # if not already object/string
                             df[col] = df[col].astype(str) # try a final cast to string if it's a critical string col
                else:
                    # Add missing columns with appropriate empty series
                    if dtype == "Int64":
                        df[col] = pd.Series(dtype=pd.Int64Dtype(), index=df.index)
                    elif dtype == float:
                        df[col] = pd.Series(dtype=np.float64, index=df.index)
                    elif dtype == "datetime64[ns]":
                         df[col] = pd.Series(dtype='datetime64[ns]', index=df.index)
                    elif dtype == str: # Ensure missing string columns are added as string type
                        df[col] = pd.Series(dtype=str, index=df.index)
                    else:
                        df[col] = pd.Series(dtype=object, index=df.index)
            
            # Specific re-assurance for critical string columns if they were added as object and are all None/NaN
            for col_name in ["Action", "Symbol", "TradeType"]:
                if col_name in df.columns and df[col_name].isnull().all():
                     df[col_name] = df[col_name].astype(str)
                elif col_name in df.columns and df[col_name].dtype == 'object': # if it's object, ensure it's string
                     df[col_name] = df[col_name].astype(str)


        # Ensure all expected columns are present, even if trade_log was empty or missing some fields
        # This loop is crucial for ensuring DataFrame structure consistency
        for col, dtype in expected_columns.items():
            if col not in df.columns:
                if dtype == "Int64":
                    df[col] = pd.Series(dtype=pd.Int64Dtype())
                elif dtype == float:
                    df[col] = pd.Series(dtype=np.float64)
                elif dtype == "datetime64[ns]":
                    df[col] = pd.Series(dtype='datetime64[ns]')
                elif dtype == str:
                     df[col] = pd.Series(dtype=str) # Ensure string type for new string columns
                else:
                    df[col] = pd.Series(dtype=object)
        return df

    def get_daily_capital_df(self):
        if not self.daily_capital:
            # Return an empty DataFrame with expected columns and dtypes
            df = pd.DataFrame()
            df['date'] = pd.Series(dtype='datetime64[ns]')
            df['capital'] = pd.Series(dtype=float)
            return df
        
        df = pd.DataFrame(self.daily_capital)
        # Ensure correct dtypes
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['capital'] = pd.to_numeric(df['capital'], errors='coerce')
        return df

    def _round_price(self, price):
        """Helper function to round price based on strategy or default to 2 decimal places."""
        # Default rounding logic, can be enhanced or made strategy-specific
        return round(price, 2)
