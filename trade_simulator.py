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

        # Load from config
        self.commission_rate = config.COMMISSION_RATE
        self.min_commission = config.MIN_COMMISSION

        # 新增：追蹤融券保證金凍結
        self.margin_held = 0.0  # 追蹤融券保證金凍結

    def get_last_trade_pnl(self):
        """獲取最近一筆已實現損益的交易的 PNL。"""
        trade_log_df = self.get_trade_log_df()
        if not trade_log_df.empty:
            # PNL 欄位已從小寫 'pnl' 改為 'PNL'，這裡保持一致
            closed_trades = trade_log_df[trade_log_df['PNL'].notna()]
            if not closed_trades.empty:
                return closed_trades['PNL'].iloc[-1]
        return 0.0

    def get_cumulative_pnl(self):
        """獲取當前的累計總損益。"""
        daily_capital_df = self.get_daily_capital_df()
        if not daily_capital_df.empty:
            # 確保 'capital' 欄位存在
            if 'capital' in daily_capital_df.columns:
                return daily_capital_df['capital'].iloc[-1] - self.initial_capital
            elif 'Capital' in daily_capital_df.columns: # 兼容大小寫
                return daily_capital_df['Capital'].iloc[-1] - self.initial_capital
        return 0.0

    def reset(self):
        self.cash = self.initial_capital
        self.current_position = 0
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

    def buy(self, price: float, quantity: int, date_to_log, trade_type: str = "Regular"): # Added date_to_log
        """Executes a buy trade."""
        original_price_arg = price # Keep original for logging if needed, though we'll use rounded
        price = self._round_price(price) # Apply price precision rules at the beginning

        # Returns a trade log entry (dict) or None if trade failed
        print(f"SIMULATOR: Buy {quantity} at {price} (Original: {original_price_arg}) (Type: {trade_type}) Date: {date_to_log}")
        
        adjusted_quantity = self._adjust_quantity(quantity)
        if adjusted_quantity == 0:
            print(f"SIMULATOR WARNING: Buy quantity {quantity} adjusted to 0. No trade.")
            return None

        cost = adjusted_quantity * price # Cost of shares
        trade_cost_effect = self._calculate_trade_cost(price, adjusted_quantity, "Buy") # Commission

        self.trade_id_counter += 1
        current_trade_id = self.trade_id_counter
        
        pnl = 0.0 # Initialize PNL for this trade
        log_action = ""

        if self.current_position < 0: # Buying to cover a short
            if trade_type == "Regular": log_action = "Cover"
            elif trade_type == "RegularSL": log_action = "Cover_SL"
            elif trade_type == "ForcedTSL": log_action = "ForcedCover_TSL"
            elif trade_type == "ForcedSL": log_action = "ForcedCover_SL"
            elif trade_type == "ForcedTP": log_action = "ForcedCover_TP"
            elif trade_type == "EndOfSim": log_action = "Cover_EndOfSim"
            else: log_action = f"Cover_{trade_type}"

            pnl = (self.entry_price - price) * adjusted_quantity - trade_cost_effect
            # 1. 釋放保證金
            amount_to_release_from_margin = self.entry_price * adjusted_quantity
            self.margin_held -= amount_to_release_from_margin
            # 2. 扣手續費
            self.cash -= trade_cost_effect
            # 3. 盈虧進出現金
            self.cash += (self.entry_price - price) * adjusted_quantity

            self.current_position += adjusted_quantity 
            if self.current_position == 0:
                self.entry_price = 0
                self.direction = None 
                if self.is_forced_trade and self.forced_trade_direction == "short": 
                    self.is_forced_trade = False
                    self.forced_trade_direction = None
                    self.current_forced_trade_id = None 
            
            self.last_actual_trade_pnl = pnl 
            self.cumulative_pnl += pnl 
            print(f"SIMULATOR: Covering short. Qty: {adjusted_quantity}, Price: {price}, PNL: {pnl:.2f}")

        else: # Opening or adding to a long position
            if self.cash < cost + trade_cost_effect:
                print(f"SIMULATOR WARNING: Insufficient cash for buy. Have {self.cash}, need {cost + trade_cost_effect}")
                return None
            if trade_type == "Regular": log_action = "Buy"
            elif trade_type == "Forced": log_action = "ForcedBuy" 
            else: log_action = f"Buy_{trade_type}" 
            self.cash -= (cost + trade_cost_effect) # 只在多單開倉/加碼時扣款
            if self.current_position > 0 : 
                new_total_quantity = self.current_position + adjusted_quantity
                self.entry_price = ((self.entry_price * self.current_position) + (price * adjusted_quantity)) / new_total_quantity
                self.current_position = new_total_quantity
            else: 
                self.entry_price = price
                self.current_position += adjusted_quantity
            if log_action == "ForcedBuy":
                self.is_forced_trade = True
                self.forced_trade_entry_price = price 
                self.forced_trade_direction = "long"
                self.current_forced_trade_id = current_trade_id 
                self.current_forced_trade_quantity = adjusted_quantity 
                self.forced_trade_take_profit_target = self._round_price(price * (1 + self.forced_trade_take_profit_pct))
                self.forced_trade_stop_loss_target = self._round_price(price * (1 - self.forced_trade_stop_loss_pct))
                if self.forced_trade_use_trailing_stop:
                    self.forced_trade_trailing_stop_price = self.forced_trade_stop_loss_target 
                    self.current_forced_trade_highest_price_since_entry = price 
            else: 
                self.direction = "long"
        
        self.update_portfolio_value(price) # Update portfolio value based on the trade price for consistency at trade time

        trade_log_entry = {
            "Date": pd.Timestamp(date_to_log).strftime('%Y%m%d'),
            "Action": log_action, "Symbol": self.stock_symbol, "Price": price, 
            "Quantity": adjusted_quantity, 
            # 修正：只要是平空單（self.current_position >= 0 且 log_action != "Buy"），都記錄正確 PNL
            "PNL": round(pnl, 2) if self.current_position >= 0 and log_action != "Buy" else 0.0,
            "Cash": round(self.cash, 2), 
            "TradeID": current_trade_id,
            "PortfolioValue": round(self.portfolio_value, 2),
            "TradeType": trade_type
        }
        self.trade_log.append(trade_log_entry)
        return trade_log_entry

    def sell(self, price: float, quantity: int, date_to_log, trade_type: str = "Regular"): # Added date_to_log
        """Executes a sell trade (either to close a long or open/add to a short)."""
        original_price_arg = price
        price = self._round_price(price) # Apply price precision rules at the beginning

        print(f"SIMULATOR: Sell {quantity} at {price} (Original: {original_price_arg}) (Type: {trade_type}) Date: {date_to_log}")

        adjusted_quantity = self._adjust_quantity(quantity)
        if adjusted_quantity == 0:
            print(f"SIMULATOR WARNING: Sell quantity {quantity} adjusted to 0. No trade.")
            return None

        if self.current_position > 0 and adjusted_quantity > self.current_position: 
            print(f"SIMULATOR WARNING: Attempting to sell {adjusted_quantity} but only hold {self.current_position}. Adjusting to sell all.")
            adjusted_quantity = self.current_position
        
        if self.current_position == 0 and not self.allow_short and (trade_type == "Regular" or trade_type.startswith("OptShort")):
            print("SIMULATOR WARNING: Short selling not allowed for regular/optimizer trades.")
            return None
        
        if self.current_position == 0 and trade_type == "ForcedShort" and not self.enable_forced_trading:
             print("SIMULATOR WARNING: Short selling not allowed for forced trades if not enabled.")
             return None

        self.trade_id_counter += 1
        current_trade_id = self.trade_id_counter

        # 將 proceeds 與 trade_cost_effect 的計算移到所有 adjusted_quantity 調整之後
        proceeds = adjusted_quantity * price
        cost_action = "Sell" if self.current_position > 0 else "ShortEntry"
        trade_cost_effect = self._calculate_trade_cost(price, adjusted_quantity, cost_action)
        
        pnl = 0.0 # Initialize PNL for this trade
        log_action = ""

        if self.current_position > 0: # Selling a long position
            if trade_type == "Regular": log_action = "Sell"
            elif trade_type == "RegularSL": log_action = "Sell_SL"
            elif trade_type == "ForcedTSL": log_action = "ForcedSell_TSL"
            elif trade_type == "ForcedSL": log_action = "ForcedSell_SL"
            elif trade_type == "ForcedTP": log_action = "ForcedSell_TP"
            elif trade_type == "EndOfSim": log_action = "Sell_EndOfSim"
            else: log_action = f"Sell_{trade_type}" 

            pnl = (price - self.entry_price) * adjusted_quantity - trade_cost_effect
            self.cash += (proceeds - trade_cost_effect) # Add net proceeds
            self.current_position -= adjusted_quantity
            
            if self.current_position == 0:
                self.entry_price = 0
                self.direction = None # Reset regular trade direction
                if self.is_forced_trade and self.forced_trade_direction == "long": 
                    self.is_forced_trade = False
                    self.forced_trade_direction = None
                    self.current_forced_trade_id = None
            
            self.last_actual_trade_pnl = pnl 
            self.cumulative_pnl += pnl 
            print(f"SIMULATOR: Closing long. Qty: {adjusted_quantity}, Price: {price}, PNL: {pnl:.2f}")
        
        elif self.allow_short or trade_type == "ForcedShort": # Opening or adding to a short position
            if adjusted_quantity > self.short_qty_cap and self.current_position == 0 : 
                 print(f"SIMULATOR WARNING: Sell quantity {adjusted_quantity} exceeds short cap {self.short_qty_cap}. Adjusting.")
                 adjusted_quantity = self.short_qty_cap
            if adjusted_quantity == 0: return None
            
            if trade_type == "Regular": log_action = "Short"
            elif trade_type == "Forced": log_action = "ForcedShort" 
            # elif trade_type.startswith("OptShort"): log_action = trade_type # Keep OptShort_StrategyName
            else: log_action = f"Short_{trade_type}"

            # MODIFIED CASH AND MARGIN LOGIC FOR SHORTING
            self.cash -= trade_cost_effect  # Cash only pays for commission
            self.margin_held += proceeds     # Proceeds from short sell are held as margin

            # Proper averaging for entry price if adding to existing short position:
            if self.current_position < 0: # Adding to an existing short
                new_total_quantity_abs = abs(self.current_position) + adjusted_quantity
                self.entry_price = ((self.entry_price * abs(self.current_position)) + (price * adjusted_quantity)) / new_total_quantity_abs
                self.current_position -= adjusted_quantity
            else: # Opening new short
                self.entry_price = price 
                self.current_position -= adjusted_quantity


            if log_action == "ForcedShort":
                self.is_forced_trade = True
                self.forced_trade_entry_price = price # Use the (rounded) trade price
                self.forced_trade_direction = "short"
                self.current_forced_trade_id = current_trade_id # Associate with this trade
                self.current_forced_trade_quantity = adjusted_quantity # Store quantity of forced trade
                self.forced_trade_take_profit_target = self._round_price(price * (1 - self.forced_trade_take_profit_pct))
                self.forced_trade_stop_loss_target = self._round_price(price * (1 + self.forced_trade_stop_loss_pct))
                if self.forced_trade_use_trailing_stop:
                    self.forced_trade_trailing_stop_price = self.forced_trade_stop_loss_target # Initialize trailing stop
                    self.current_forced_trade_lowest_price_since_entry = price # Initialize for trailing stop
            else: 
                self.direction = "short"
        else: 
            print(f"SIMULATOR WARNING: Sell condition not met. Allow short: {self.allow_short}, Trade Type: {trade_type}")
            return None

        self.update_portfolio_value(price) # Update portfolio value based on the trade price

        trade_log_entry = {
            "Date": pd.Timestamp(date_to_log).strftime('%Y%m%d'),
            "Action": log_action, "Symbol": self.stock_symbol, "Price": price, 
            "Quantity": adjusted_quantity, 
            # 修正：只要是平空單（self.current_position >= 0 且 log_action != "Buy"），都記錄正確 PNL
            "PNL": round(pnl, 2) if self.current_position >= 0 and log_action != "Buy" else 0.0,
            "Cash": round(self.cash, 2), 
            "TradeID": current_trade_id,
            "PortfolioValue": round(self.portfolio_value, 2),
            "TradeType": trade_type
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
        
    def simulate(self, price_df: pd.DataFrame, signal_df: pd.DataFrame, optimizer_capital_allocation_pct: float = 0.5): # Added optimizer_capital_allocation_pct
        self.reset()

        if not all(col in price_df.columns for col in ['date', 'Close', 'Open', 'High', 'Low']):
            raise ValueError("price_df 必須包含 'date', 'Open', 'High', 'Low', 'Close' 欄位。")
        if not all(col in signal_df.columns for col in ['date', 'signal']):
            raise ValueError("signal_df 必須包含 'date', 'signal' 欄位。")

        # Ensure 'date' columns are of the same type for merging, typically datetime
        price_df['date'] = pd.to_datetime(price_df['date'])
        signal_df['date'] = pd.to_datetime(signal_df['date'])

        merged_df = pd.merge(price_df, signal_df, on="date", how="left")
        merged_df['signal'] = merged_df['signal'].fillna(0)


        for i in range(len(merged_df)):
            row = merged_df.iloc[i]
            current_date = row["date"] 
            current_price_open = self._round_price(row["Open"])
            # current_price_high = self._round_price(row["High"]) # Available if needed for more precise SL/TP
            # current_price_low = self._round_price(row["Low"])   # Available if needed for more precise SL/TP
            current_price_close = self._round_price(row["Close"])
            signal_value = row["signal"] 

            trade_executed_this_step = False 
            log_entry_this_step = None 

            # Priority 1: Forced Trade Management (check for closure at Open price)
            if self.is_forced_trade_active():
                # Using Open price for SL/TP checks in optimizer context for simplicity.
                # More complex H/L checks could be added if check_forced_trade_closure is adapted.
                log_entry_this_step = self.check_forced_trade_closure(current_price_open, date_to_log=current_date)
                if log_entry_this_step:
                    trade_executed_this_step = True

            # Priority 2: Regular Position Stop-Loss/Take-Profit (at Open price)
            if not trade_executed_this_step and self.current_position != 0 and not self.is_forced_trade_active() and self.direction:
                log_entry_this_step = self.check_stop_loss_take_profit(current_price_open, date_to_log=current_date)
                if log_entry_this_step:
                    trade_executed_this_step = True
            
            # Priority 3: Act on New Signal
            if not trade_executed_this_step or self.current_position == 0:
                if not self.is_forced_trade_active(): 
                    
                    if signal_value == 1: # Buy or Cover
                        if self.current_position < 0: 
                            qty_to_cover = self._adjust_quantity(abs(self.current_position))
                            if qty_to_cover > 0:
                                log_entry_this_step = self.buy(current_price_open, qty_to_cover, date_to_log=current_date, trade_type="OptCover")
                                if log_entry_this_step: trade_executed_this_step = True
                        elif self.current_position == 0: 
                            capital_for_trade = self.cash * optimizer_capital_allocation_pct
                            qty_to_buy_float = 0
                            if current_price_open > 0: # Avoid division by zero
                                qty_to_buy_float = capital_for_trade / current_price_open
                            qty_to_buy = self._adjust_quantity(qty_to_buy_float)
                            

                            max_affordable_qty = 0
                            if current_price_open > 0 : 
                                cost_per_unit_approx = current_price_open * (1 + 0.002) 
                                if cost_per_unit_approx > 0:
                                     max_affordable_qty = self._adjust_quantity(self.cash / cost_per_unit_approx)
                            

                            qty_to_buy = min(qty_to_buy, max_affordable_qty)

                            if qty_to_buy > 0:
                                log_entry_this_step = self.buy(current_price_open, qty_to_buy, date_to_log=current_date, trade_type="OptBuy")
                                if log_entry_this_step: trade_executed_this_step = True

                    elif signal_value == -1: # Sell or Short
                        if self.current_position > 0: 
                            qty_to_sell = self._adjust_quantity(abs(self.current_position))
                            if qty_to_sell > 0:
                                log_entry_this_step = self.sell(current_price_open, qty_to_sell, date_to_log=current_date, trade_type="OptSell")
                                if log_entry_this_step: trade_executed_this_step = True
                        elif self.current_position == 0 and self.allow_short: 
                            capital_at_risk_proxy = self.cash * optimizer_capital_allocation_pct 
                            qty_to_short_float = 0
                            if current_price_open > 0: # Avoid division by zero
                                qty_to_short_float = capital_at_risk_proxy / current_price_open
                            qty_to_short = self._adjust_quantity(qty_to_short_float)
                            qty_to_short = min(qty_to_short, self.short_qty_cap) 

                            if qty_to_short > 0:
                                log_entry_this_step = self.sell(current_price_open, qty_to_short, date_to_log=current_date, trade_type="OptShort")
                                if log_entry_this_step: trade_executed_this_step = True
            
            self.update_portfolio_value(current_price_close) 
            self.daily_capital.append({
                "date": current_date, 
                "capital": round(self.cash + self.portfolio_value, 2) # Round daily capital
            })
            # Reset trade_executed_today for the next day
            self.trade_executed_today = False


        # 在模擬結束時，如果仍有未平倉部位，則以最後一天的收盤價平倉
        if self.current_position != 0 and not merged_df.empty:
            last_day_data = merged_df.iloc[-1]
            last_close_price = self._round_price(last_day_data["Close"])
            last_date = last_day_data["date"]
            print(f"SIMULATOR: End of simulation. Closing position at {last_close_price} on {last_date}")
            if self.current_position > 0: # Long position
                self.sell(last_close_price, abs(self.current_position), date_to_log=last_date, trade_type="EndOfSim")
            elif self.current_position < 0: # Short position
                self.buy(last_close_price, abs(self.current_position), date_to_log=last_date, trade_type="EndOfSim")
        
        # Return both trade log and daily capital
        return self.get_trade_log_df(), pd.DataFrame(self.daily_capital)

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
