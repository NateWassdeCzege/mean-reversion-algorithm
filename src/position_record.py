import uuid
import pandas as pd

class PairPositionRecord:
    def __init__(self, entry_time, price, size, trade_type, time_reversion=None, ticker='KO'):
        self.trade_id = f"TRADE_{uuid.uuid4().hex[:6].upper()}"
        self.entry_time = entry_time
        self.ticker = ticker
        self.tick_life = 0  # Starts at 0 when the trade is opened
        
        # Entry Details
        self.entry_price = price
        self.size = size
        self.trade_type = trade_type.upper()  # 'LONG' or 'SHORT'
        self.time_reversion = time_reversion
        
        # Exit Trackers
        self.exit_time = None
        self.exit_price = None
        self.profit_loss = None
        self.exit_reason = None

    def increment_tick(self):
        """Increases the age of the trade by 1 tick (1 hour)."""
        self.tick_life += 1

    def close_trade(self, exit_time, exit_price, exit_reason="unspecified"):
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = exit_reason
        
        # Calculate PnL based on trade direction
        if self.trade_type == 'LONG':
            self.profit_loss = (self.exit_price - self.entry_price) * self.size
        else:  # SHORT
            self.profit_loss = (self.entry_price - self.exit_price) * self.size
            
        return self.profit_loss


class PairsPortfolioManager:
    def __init__(self, initial_cash):
        self.cash = initial_cash
        self.active_pairs = {}   
        self.closed_pairs_log = {} 
        self.master_transaction_list = [] 

    def update_time_step(self):
        """
        Call this exactly once per loop iteration to age all open positions.
        """
        for trade in self.active_pairs.values():
            trade.increment_tick()

    def open_new_pair(self, entry_time, price, size, trade_type, time_reversion=None, ticker='KO'):
        """Opens a new trade and reserves required margin/cash."""
        total_cost = price * size
        
        if total_cost > self.cash:
            print(f"Skipping trade: Insufficient Cash. Required: ${total_cost:.2f}, Cash: ${self.cash:.2f}")
            return None
            
        new_trade = PairPositionRecord(
            entry_time=entry_time,
            price=price,
            size=size,
            trade_type=trade_type,
            time_reversion=time_reversion,
            ticker=ticker
        )

        self.cash -= total_cost 
        self.active_pairs[new_trade.trade_id] = new_trade
        
        return new_trade.trade_id

    def close_trade_by_id(self, trade_id, exit_time, exit_price, exit_reason="unspecified"):
        """Closes a specific trade by its ID and updates cash balance."""
        if trade_id not in self.active_pairs: 
            return None

        trade = self.active_pairs.pop(trade_id)
        
        # Pass exit_reason to PairPositionRecord
        pnl = trade.close_trade(exit_time, exit_price, exit_reason)
        
        capital_returned = (trade.entry_price * trade.size)
        self.cash += (capital_returned + pnl)

        self.closed_pairs_log[trade_id] = trade
        
        log_entry = {
            'trade_id': trade.trade_id,
            'entry_time': trade.entry_time,
            'exit_time': trade.exit_time,
            'ticker': trade.ticker,
            'trade_type': trade.trade_type,
            'size': trade.size,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'exit_reason': trade.exit_reason,
            'profit_loss': trade.profit_loss,
            'tick_life': trade.tick_life,
            'time_reversion': trade.time_reversion,
            'account_cash_balance': self.cash
        }
        self.master_transaction_list.append(log_entry)
        
        return pnl

    def close_all_active_pairs(self, exit_time, exit_price, exit_reason="unspecified"):
        """Closes all currently open positions."""
        for trade_id in list(self.active_pairs.keys()):
            pnl = self.close_trade_by_id(trade_id, exit_time, exit_price, exit_reason)
            if pnl is not None:
                print(f"Closed trade {trade_id} ({exit_reason}) with PnL: ${pnl:.2f}")

    def export_trade_log(self, filename="backtest_results.csv"):
        """Saves the master transaction list to a CSV file."""
        if not self.master_transaction_list:
            print("No trades were made during this run.")
            return
            
        df = pd.DataFrame(self.master_transaction_list)
        df.to_csv(filename, index=False)
        print(f"Success! Saved {len(df)} trades to {filename}")