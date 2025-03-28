"""
Portfolio Manager Module

This module handles portfolio management, including tracking positions,
calculating performance metrics, and managing the overall portfolio.
"""

import logging
from datetime import datetime
import json
import os

logger = logging.getLogger("TradingBot.PortfolioManager")

class PortfolioManager:
    """
    Portfolio manager class that handles tracking positions and performance.
    """
    
    def __init__(self, initial_balance=10000.0, portfolio_file="portfolio.json"):
        """
        Initialize the portfolio manager with an initial balance.
        
        Args:
            initial_balance (float): Initial portfolio balance
            portfolio_file (str): File to save portfolio data
        """
        self.portfolio_file = portfolio_file
        
        # Try to load existing portfolio
        if os.path.exists(portfolio_file):
            try:
                with open(portfolio_file, 'r') as f:
                    portfolio_data = json.load(f)
                
                self.balance = portfolio_data.get("balance", initial_balance)
                self.positions = portfolio_data.get("positions", {})
                self.trade_history = portfolio_data.get("trade_history", [])
                
                logger.info(f"Loaded portfolio from {portfolio_file} with balance {self.balance:.2f}")
            except Exception as e:
                logger.error(f"Failed to load portfolio from {portfolio_file}: {e}")
                self._initialize_portfolio(initial_balance)
        else:
            self._initialize_portfolio(initial_balance)
    
    def _initialize_portfolio(self, initial_balance):
        """
        Initialize a new portfolio.
        
        Args:
            initial_balance (float): Initial portfolio balance
        """
        self.balance = initial_balance
        self.positions = {}  # Symbol -> position details
        self.trade_history = []
        
        logger.info(f"Initialized new portfolio with balance {initial_balance:.2f}")
    
    def save_portfolio(self):
        """
        Save the portfolio to a file.
        """
        portfolio_data = {
            "balance": self.balance,
            "positions": self.positions,
            "trade_history": self.trade_history,
            "last_updated": datetime.now().isoformat()
        }
        
        try:
            with open(self.portfolio_file, 'w') as f:
                json.dump(portfolio_data, f, indent=4)
            
            logger.info(f"Saved portfolio to {self.portfolio_file}")
        except Exception as e:
            logger.error(f"Failed to save portfolio to {self.portfolio_file}: {e}")
    
    def get_portfolio_value(self, current_prices):
        """
        Calculate the total portfolio value including balance and positions.
        
        Args:
            current_prices (dict): Current prices for all symbols
            
        Returns:
            float: Total portfolio value
        """
        total_value = self.balance
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position_value = position["size"] * current_prices[symbol]
                total_value += position_value
        
        logger.debug(f"Current portfolio value: {total_value:.2f}")
        return total_value
    
    def open_position(self, symbol, position_type, entry_price, size, stop_loss=None, take_profit=None):
        """
        Open a new position.
        
        Args:
            symbol (str): Trading symbol
            position_type (str): Type of position ('long' or 'short')
            entry_price (float): Entry price of the position
            size (float): Size of the position in base currency
            stop_loss (float, optional): Stop-loss price
            take_profit (float, optional): Take-profit price
            
        Returns:
            bool: Whether the position was successfully opened
        """
        # Check if we already have a position for this symbol
        if symbol in self.positions:
            logger.warning(f"Already have a position for {symbol}. Close it first.")
            return False
        
        # Calculate the cost of the position
        position_cost = entry_price * size
        
        # Check if we have enough balance
        if position_cost > self.balance:
            logger.warning(f"Insufficient balance to open position for {symbol}. "
                          f"Required: {position_cost:.2f}, Available: {self.balance:.2f}")
            return False
        
        # Deduct the cost from the balance
        self.balance -= position_cost
        
        # Create the position
        position = {
            "symbol": symbol,
            "type": position_type.lower(),
            "entry_price": entry_price,
            "size": size,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "open_time": datetime.now().timestamp()
        }
        
        # Add the position to our positions
        self.positions[symbol] = position
        
        logger.info(f"Opened {position_type} position for {symbol} at {entry_price:.2f} "
                   f"with size {size:.6f} (cost: {position_cost:.2f})")
        
        # Save the updated portfolio
        self.save_portfolio()
        
        return True
    
    def close_position(self, symbol, exit_price, reason="manual"):
        """
        Close an existing position.
        
        Args:
            symbol (str): Trading symbol
            exit_price (float): Exit price of the position
            reason (str): Reason for closing the position
            
        Returns:
            tuple: (bool, float) - Whether the position was successfully closed and the profit/loss
        """
        # Check if we have a position for this symbol
        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}")
            return False, 0.0
        
        position = self.positions[symbol]
        entry_price = position["entry_price"]
        size = position["size"]
        position_type = position["type"]
        
        # Calculate the value of the position at exit
        position_value = exit_price * size
        
        # Calculate profit/loss
        if position_type == "long":
            pnl = (exit_price - entry_price) * size
        else:  # short position
            pnl = (entry_price - exit_price) * size
        
        # Add the position value plus profit/loss to the balance
        self.balance += position_value
        
        # Create a trade record
        trade_record = {
            "symbol": symbol,
            "type": position_type,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": size,
            "pnl": pnl,
            "pnl_percent": (pnl / (entry_price * size)) * 100,
            "open_time": position["open_time"],
            "close_time": datetime.now().timestamp(),
            "reason": reason
        }
        
        # Add the trade to history
        self.trade_history.append(trade_record)
        
        # Remove the position
        del self.positions[symbol]
        
        logger.info(f"Closed {position_type} position for {symbol} at {exit_price:.2f} "
                   f"with size {size:.6f} (P&L: {pnl:.2f}, {reason})")
        
        # Save the updated portfolio
        self.save_portfolio()
        
        return True, pnl
    
    def get_position(self, symbol):
        """
        Get the details of a position.
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            dict: Position details or None if no position exists
        """
        return self.positions.get(symbol)
    
    def has_position(self, symbol):
        """
        Check if we have a position for a symbol.
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            bool: Whether we have a position for the symbol
        """
        return symbol in self.positions
    
    def get_performance_metrics(self):
        """
        Calculate performance metrics for the portfolio.
        
        Returns:
            dict: Performance metrics
        """
        if not self.trade_history:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_profit_loss": 0.0,
                "average_profit_loss": 0.0,
                "max_profit": 0.0,
                "max_loss": 0.0
            }
        
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for trade in self.trade_history if trade["pnl"] > 0)
        losing_trades = sum(1 for trade in self.trade_history if trade["pnl"] < 0)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        total_pnl = sum(trade["pnl"] for trade in self.trade_history)
        average_pnl = total_pnl / total_trades if total_trades > 0 else 0.0
        
        max_profit = max((trade["pnl"] for trade in self.trade_history), default=0.0)
        max_loss = min((trade["pnl"] for trade in self.trade_history), default=0.0)
        
        metrics = {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_profit_loss": total_pnl,
            "average_profit_loss": average_pnl,
            "max_profit": max_profit,
            "max_loss": max_loss
        }
        
        logger.info(f"Performance metrics: {metrics}")
        return metrics