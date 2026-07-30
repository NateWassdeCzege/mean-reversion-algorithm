import uuid
import pandas as pd

class PairPositionRecord:
    def __init__(self, entry_time, price, size, trade_type, ticker='KO'):
        self.trade_id = f"TRADE_{uuid.uuid4().hex[:6].upper()}"
        self.entry_time = entry_time
        self.ticker = ticker
        
        # Entry Details
        self.entry_price = price
        self.size = size
        self.trade_type = trade_type.upper()  # 'LONG' or 'SHORT'
        
        # Exit Trackers
        self.exit_time = None
        self.exit_price = None
        self.profit_loss = None

    def close_trade(self, exit_time, exit_price):
        self.exit_time = exit_time
        self.exit_price = exit_price
        
        # Calculate PnL based on trade direction
        if self.trade_type == 'LONG':
            self.profit_loss = (self.exit_price - self.entry_price) * self.size
        else: # SHORT
            self.profit_loss = (self.entry_price - self.exit_price) * self.size
            
        return self.profit_loss


class PairsPortfolioManager:
    def __init__(self, initial_cash):
        self.cash = initial_cash
        self.active_pairs = {}   
        self.closed_pairs_log = {} 
        self.master_transaction_list = [] 

    def open_new_pair(self, entry_time, price, size, trade_type):
        """Matches the 4 arguments passed from signal_engine.py"""
        new_trade = PairPositionRecord(entry_time, price, size, trade_type)
        
        # Calculate total capital required
        total_cost = price * size
        
        if total_cost > self.cash:
            print(f"Skipping trade: Insufficient Cash. Cost: ${total_cost:.2f}, Cash: ${self.cash:.2f}")
            return None
            
        # Deduct capital and log active trade
        self.cash -= total_cost 
        self.active_pairs[new_trade.trade_id] = new_trade
        
        return new_trade.trade_id

    def close_trade_by_id(self, trade_id, exit_time, exit_price):
        """Closes a specific trade by its ID"""
        if trade_id not in self.active_pairs: 
            return

        trade = self.active_pairs.pop(trade_id)
        
        # Trigger the exit logic
        pnl = trade.close_trade(exit_time, exit_price)
        
        # Return initial capital allocated PLUS total profit/loss
        capital_returned = (trade.entry_price * trade.size)
        self.cash += (capital_returned + pnl)

        self.closed_pairs_log[trade_id] = trade
        
        # Log the transaction without PEP or Spread variables
        self.master_transaction_list.append({
            'trade_id': trade.trade_id,
            'entry_time': trade.entry_time,
            'exit_time': trade.exit_time,
            'ticker': trade.ticker,
            'trade_type': trade.trade_type,
            'size': trade.size,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'profit_loss': trade.profit_loss,
            'account_cash_balance': self.cash
        })

    def close_all_active_pairs(self, exit_time, exit_price):
        """Closes all currently open positions (matches signal_engine.py arguments)."""
        
        # We use list(keys) because we are removing items from the dictionary as we loop
        for trade_id in list(self.active_pairs.keys()):
            trade = self.active_pairs.pop(trade_id)
            
            # Trigger the built-in exit logic
            pnl = trade.close_trade(exit_time, exit_price)
            
            # Return initial capital plus PnL
            capital_returned = (trade.entry_price * trade.size)
            self.cash += (capital_returned + pnl)

            self.closed_pairs_log[trade_id] = trade
            
            # Log the transaction
            self.master_transaction_list.append({
                'trade_id': trade.trade_id,
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'ticker': trade.ticker,
                'trade_type': trade.trade_type,
                'size': trade.size,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'profit_loss': trade.profit_loss,
                'account_cash_balance': self.cash
            })
            
            print(f"Closed trade {trade_id} with PnL: ${pnl:.2f}")

    def export_trade_log(self, filename="backtest_results.csv"):
        """Saves the master transaction list to a file for analysis."""
        if not self.master_transaction_list:
            print("No trades were made during this run.")
            return
            
        # Convert the list of dictionaries into a clean DataFrame
        df = pd.DataFrame(self.master_transaction_list)
        
        # Save it to your computer
        df.to_csv(filename, index=False)
        print(f"Success! Saved {len(df)} trades to {filename}")