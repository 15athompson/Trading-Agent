#!/usr/bin/env python3
"""
Backtesting Module for Trading Bot

This module provides functionality for backtesting trading strategies.
"""

import logging
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from strategies import create_strategy
from risk_manager import RiskManager

logger = logging.getLogger("TradingBot.Backtesting")

class Backtester:
    """
    Backtester for evaluating trading strategies on historical data.
    """
    
    def __init__(self, config_path="config.json", initial_balance=10000.0):
        """
        Initialize the backtester with configuration.
        
        Args:
            config_path (str): Path to the configuration file
            initial_balance (float): Initial balance for backtesting
        """
        self.config = self._load_config(config_path)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {}
        self.trade_history = []
        self.portfolio_history = []
        
        # Initialize risk manager
        self.risk_manager = RiskManager(self.config["trading"]["risk_management"])
        
        # Initialize strategy
        strategy_name = self.config["trading"]["strategy"]
        strategy_params = self.config["trading"].get("strategy_params", {})
        self.strategy = create_strategy(strategy_name, strategy_params)
        
        logger.info(f"Backtester initialized with strategy: {strategy_name}")
    
    def _load_config(self, config_path):
        """
        Load configuration from a JSON file.
        
        Args:
            config_path (str): Path to the configuration file
            
        Returns:
            dict: Configuration parameters
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Configuration file {config_path} not found. Using default configuration.")
            return {
                "api": {
                    "provider": "demo",
                    "api_key": "YOUR_API_KEY",
                    "api_secret": "YOUR_API_SECRET"
                },
                "trading": {
                    "symbols": ["BTC/USD", "ETH/USD"],
                    "interval": "1h",
                    "strategy": "simple_moving_average",
                    "strategy_params": {
                        "short_window": 20,
                        "long_window": 50
                    },
                    "risk_management": {
                        "max_position_size": 0.1,
                        "stop_loss": 0.05,
                        "take_profit": 0.1
                    }
                },
                "bot_settings": {
                    "update_interval": 60,
                    "backtest_mode": True
                }
            }
    
    def _generate_synthetic_data(self, start_date, end_date):
        """
        Generate synthetic historical data for backtesting.
        
        Args:
            start_date (datetime): Start date for the data
            end_date (datetime): End date for the data
            
        Returns:
            dict: Dictionary of historical data for each symbol
        """
        symbols = self.config["trading"]["symbols"]
        data = {}
        
        for symbol in symbols:
            # Generate synthetic price data
            days = (end_date - start_date).days + 1
            
            # Set initial price based on symbol
            if symbol == "BTC/USD":
                initial_price = 40000.0
                volatility = 0.03
            elif symbol == "ETH/USD":
                initial_price = 2000.0
                volatility = 0.04
            else:
                initial_price = 100.0
                volatility = 0.02
            
            # Generate daily returns with some autocorrelation
            np.random.seed(42)  # For reproducibility
            returns = np.random.normal(0.0005, volatility, days)
            
            # Add some autocorrelation
            for i in range(1, days):
                returns[i] = 0.7 * returns[i] + 0.3 * returns[i-1]
            
            # Generate prices
            prices = [initial_price]
            for ret in returns:
                prices.append(prices[-1] * (1 + ret))
            
            # Generate dates
            dates = [start_date + timedelta(days=i) for i in range(days)]
            
            # Create DataFrame
            df = pd.DataFrame({
                "date": dates,
                "price": prices[:-1],  # Remove the extra price
                "volume": np.random.uniform(1000, 10000, days)
            })
            
            # Store data
            data[symbol] = df
        
        logger.info(f"Generated synthetic data for {len(symbols)} symbols from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        return data
    
    def _prepare_data_for_strategy(self, symbol, df, date):
        """
        Prepare data for strategy analysis.
        
        Args:
            symbol (str): Trading symbol
            df (DataFrame): Historical data for the symbol
            date (datetime): Current date
            
        Returns:
            dict: Data in the format expected by the strategy
        """
        # Filter data up to the current date
        historical_df = df[df["date"] <= date].copy()
        
        if len(historical_df) < 50:  # Require at least 50 data points
            return None
        
        # Get current price
        current_price = historical_df.iloc[-1]["price"]
        
        # Get historical prices
        historical_prices = historical_df["price"].tolist()
        
        # Prepare data
        data = {
            "symbol": symbol,
            "current_price": current_price,
            "historical_prices": historical_prices,
            "timestamp": date.timestamp()
        }
        
        return data
    
    def _open_position(self, symbol, position_type, price, size, stop_loss_price, take_profit_price, date):
        """
        Open a new position.
        
        Args:
            symbol (str): Trading symbol
            position_type (str): Type of position ("long" or "short")
            price (float): Entry price
            size (float): Position size
            stop_loss_price (float): Stop-loss price
            take_profit_price (float): Take-profit price
            date (datetime): Date of the trade
            
        Returns:
            bool: Whether the position was opened successfully
        """
        # Check if we already have a position for this symbol
        if symbol in self.positions:
            logger.warning(f"Already have a position for {symbol}, cannot open another")
            return False
        
        # Calculate cost
        cost = price * size
        
        # Check if we have enough balance
        if cost > self.balance:
            logger.warning(f"Insufficient balance to open position for {symbol}")
            return False
        
        # Deduct cost from balance
        self.balance -= cost
        
        # Create position
        self.positions[symbol] = {
            "type": position_type,
            "entry_price": price,
            "size": size,
            "stop_loss": stop_loss_price,
            "take_profit": take_profit_price,
            "entry_date": date
        }
        
        # Record trade
        self.trade_history.append({
            "date": date.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "type": "BUY" if position_type == "long" else "SELL",
            "price": price,
            "size": size,
            "pnl": 0.0
        })
        
        logger.info(f"Opened {position_type.upper()} position for {symbol} at {price:.2f} with size {size:.6f}")
        
        return True
    
    def _close_position(self, symbol, price, reason, date):
        """
        Close an existing position.
        
        Args:
            symbol (str): Trading symbol
            price (float): Exit price
            reason (str): Reason for closing the position
            date (datetime): Date of the trade
            
        Returns:
            tuple: (success, pnl) - Whether the position was closed successfully and the P&L
        """
        # Check if we have a position for this symbol
        if symbol not in self.positions:
            logger.warning(f"No position for {symbol} to close")
            return False, 0.0
        
        # Get position details
        position = self.positions[symbol]
        entry_price = position["entry_price"]
        size = position["size"]
        position_type = position["type"]
        
        # Calculate P&L
        if position_type == "long":
            pnl = (price - entry_price) * size
        else:  # short position
            pnl = (entry_price - price) * size
        
        # Add proceeds to balance
        self.balance += (price * size + pnl)
        
        # Record trade
        self.trade_history.append({
            "date": date.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "type": "SELL" if position_type == "long" else "BUY",
            "price": price,
            "size": size,
            "pnl": pnl
        })
        
        # Remove position
        del self.positions[symbol]
        
        logger.info(f"Closed {position_type.upper()} position for {symbol} at {price:.2f} with P&L {pnl:.2f} due to {reason}")
        
        return True, pnl
    
    def _calculate_portfolio_value(self, prices, date):
        """
        Calculate the total portfolio value.
        
        Args:
            prices (dict): Current prices for each symbol
            date (datetime): Current date
            
        Returns:
            float: Total portfolio value
        """
        value = self.balance
        
        for symbol, position in self.positions.items():
            if symbol in prices:
                current_price = prices[symbol]
                size = position["size"]
                value += current_price * size
        
        # Record portfolio value
        self.portfolio_history.append({
            "date": date,
            "value": value
        })
        
        return value
    
    def run_backtest(self, start_date, end_date):
        """
        Run a backtest over a specified date range.
        
        Args:
            start_date (datetime): Start date for the backtest
            end_date (datetime): End date for the backtest
            
        Returns:
            dict: Backtest results
        """
        # Generate synthetic data
        data = self._generate_synthetic_data(start_date, end_date)
        
        # Initialize results
        results = {
            "initial_balance": self.initial_balance,
            "final_balance": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_profit": 0.0,
            "total_loss": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_return": 0.0,
            "total_return_pct": 0.0,
            "portfolio_history": []
        }
        
        # Reset backtester state
        self.balance = self.initial_balance
        self.positions = {}
        self.trade_history = []
        self.portfolio_history = []
        
        # Get symbols
        symbols = self.config["trading"]["symbols"]
        
        # Run backtest day by day
        current_date = start_date
        max_portfolio_value = self.initial_balance
        
        while current_date <= end_date:
            # Skip weekends (optional)
            if current_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                current_date += timedelta(days=1)
                continue
            
            # Get current prices
            current_prices = {}
            for symbol in symbols:
                df = data[symbol]
                prices = df[df["date"] == current_date]["price"]
                if not prices.empty:
                    current_prices[symbol] = prices.iloc[0]
            
            # Skip if no prices available
            if not current_prices:
                current_date += timedelta(days=1)
                continue
            
            # Calculate portfolio value
            portfolio_value = self._calculate_portfolio_value(current_prices, current_date)
            
            # Update max portfolio value
            if portfolio_value > max_portfolio_value:
                max_portfolio_value = portfolio_value
            
            # Calculate drawdown
            drawdown = max_portfolio_value - portfolio_value
            drawdown_pct = drawdown / max_portfolio_value * 100
            
            if drawdown_pct > results["max_drawdown_pct"]:
                results["max_drawdown"] = drawdown
                results["max_drawdown_pct"] = drawdown_pct
            
            # Check existing positions for stop-loss and take-profit
            for symbol in list(self.positions.keys()):
                if symbol in current_prices:
                    position = self.positions[symbol]
                    current_price = current_prices[symbol]
                    
                    # Check if we should close the position
                    should_close, reason = self.risk_manager.should_close_position(position, current_price)
                    
                    if should_close:
                        success, pnl = self._close_position(symbol, current_price, reason, current_date)
                        
                        if success:
                            # Update results
                            results["total_trades"] += 1
                            
                            if pnl > 0:
                                results["winning_trades"] += 1
                                results["total_profit"] += pnl
                            else:
                                results["losing_trades"] += 1
                                results["total_loss"] += abs(pnl)
            
            # Generate signals for each symbol
            for symbol in symbols:
                # Prepare data for strategy
                symbol_data = self._prepare_data_for_strategy(symbol, data[symbol], current_date)
                
                if symbol_data is None:
                    continue
                
                # Generate signal
                signal = self.strategy.generate_signal(symbol, symbol_data)
                
                # Execute signal
                action = signal["action"]
                confidence = signal["confidence"]
                current_price = current_prices.get(symbol)
                
                if current_price is None:
                    continue
                
                # Check if we already have a position for this symbol
                has_position = symbol in self.positions
                
                if action == "buy" and not has_position and confidence > 0.2:
                    # Calculate position size
                    position_size = self.risk_manager.calculate_position_size(
                        portfolio_value, current_price, confidence)
                    
                    # Calculate stop-loss and take-profit prices
                    stop_loss_price = self.risk_manager.calculate_stop_loss_price(current_price, "long")
                    take_profit_price = self.risk_manager.calculate_take_profit_price(current_price, "long")
                    
                    # Open a long position
                    self._open_position(
                        symbol, "long", current_price, position_size,
                        stop_loss_price, take_profit_price, current_date)
                
                elif action == "sell" and has_position:
                    # Close the position
                    position = self.positions[symbol]
                    if position["type"] == "long":
                        success, pnl = self._close_position(symbol, current_price, "signal", current_date)
                        
                        if success:
                            # Update results
                            results["total_trades"] += 1
                            
                            if pnl > 0:
                                results["winning_trades"] += 1
                                results["total_profit"] += pnl
                            else:
                                results["losing_trades"] += 1
                                results["total_loss"] += abs(pnl)
            
            # Move to next day
            current_date += timedelta(days=1)
        
        # Close any remaining positions at the end of the backtest
        for symbol in list(self.positions.keys()):
            if symbol in current_prices:
                current_price = current_prices[symbol]
                success, pnl = self._close_position(symbol, current_price, "end_of_backtest", end_date)
                
                if success:
                    # Update results
                    results["total_trades"] += 1
                    
                    if pnl > 0:
                        results["winning_trades"] += 1
                        results["total_profit"] += pnl
                    else:
                        results["losing_trades"] += 1
                        results["total_loss"] += abs(pnl)
        
        # Calculate final results
        results["final_balance"] = self.balance
        results["total_return"] = results["final_balance"] - results["initial_balance"]
        results["total_return_pct"] = results["total_return"] / results["initial_balance"] * 100
        
        if results["total_trades"] > 0:
            results["win_rate"] = results["winning_trades"] / results["total_trades"]
        
        if results["total_loss"] > 0:
            results["profit_factor"] = results["total_profit"] / results["total_loss"]
        
        # Calculate Sharpe ratio
        if len(self.portfolio_history) > 1:
            portfolio_values = [entry["value"] for entry in self.portfolio_history]
            daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            if len(daily_returns) > 0 and np.std(daily_returns) > 0:
                results["sharpe_ratio"] = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        
        # Store portfolio history
        results["portfolio_history"] = self.portfolio_history
        
        # Save results to file
        self._save_results(results, start_date, end_date)
        
        return results
    
    def _save_results(self, results, start_date, end_date):
        """
        Save backtest results to a file.
        
        Args:
            results (dict): Backtest results
            start_date (datetime): Start date of the backtest
            end_date (datetime): End date of the backtest
        """
        # Create directory if it doesn't exist
        os.makedirs("backtest_results", exist_ok=True)
        
        # Create filename
        strategy_name = self.config["trading"]["strategy"]
        filename = f"backtest_{strategy_name}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
        filepath = os.path.join("backtest_results", filename)
        
        # Convert portfolio history dates to strings
        for entry in results["portfolio_history"]:
            entry["date"] = entry["date"].strftime("%Y-%m-%d")
        
        # Save results
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Backtest results saved to {filepath}")
        
        # Generate and save chart
        self._generate_chart(results, start_date, end_date)
    
    def _generate_chart(self, results, start_date, end_date):
        """
        Generate and save a chart of the backtest results.
        
        Args:
            results (dict): Backtest results
            start_date (datetime): Start date of the backtest
            end_date (datetime): End date of the backtest
        """
        try:
            # Convert dates from strings back to datetime
            for entry in results["portfolio_history"]:
                entry["date"] = datetime.strptime(entry["date"], "%Y-%m-%d")
            
            # Create DataFrame
            df = pd.DataFrame(results["portfolio_history"])
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Plot portfolio value
            plt.subplot(2, 1, 1)
            plt.plot(df["date"], df["value"])
            plt.title("Portfolio Value")
            plt.xlabel("Date")
            plt.ylabel("Value ($)")
            plt.grid(True)
            
            # Plot drawdown
            plt.subplot(2, 1, 2)
            max_value = df["value"].cummax()
            drawdown = (df["value"] - max_value) / max_value * 100
            plt.fill_between(df["date"], drawdown, 0, color="red", alpha=0.3)
            plt.title("Drawdown")
            plt.xlabel("Date")
            plt.ylabel("Drawdown (%)")
            plt.grid(True)
            
            # Add text with results
            plt.figtext(0.01, 0.01, f"Strategy: {self.config['trading']['strategy']}\n"
                                   f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n"
                                   f"Initial Balance: ${results['initial_balance']:.2f}\n"
                                   f"Final Balance: ${results['final_balance']:.2f}\n"
                                   f"Total Return: ${results['total_return']:.2f} ({results['total_return_pct']:.2f}%)\n"
                                   f"Total Trades: {results['total_trades']}\n"
                                   f"Win Rate: {results['win_rate']*100:.2f}%\n"
                                   f"Profit Factor: {results['profit_factor']:.2f}\n"
                                   f"Max Drawdown: {results['max_drawdown_pct']:.2f}%\n"
                                   f"Sharpe Ratio: {results['sharpe_ratio']:.2f}",
                       fontsize=10, verticalalignment="bottom")
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0.1, 1, 0.95])
            
            # Create filename
            strategy_name = self.config["trading"]["strategy"]
            filename = f"backtest_{strategy_name}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
            filepath = os.path.join("backtest_results", filename)
            
            # Save chart
            plt.savefig(filepath)
            plt.close()
            
            logger.info(f"Backtest chart saved to {filepath}")
        
        except Exception as e:
            logger.error(f"Failed to generate chart: {e}")


def run_backtest(config_path="config.json", start_date=None, end_date=None, initial_balance=10000.0):
    """
    Run a backtest with the specified parameters.
    
    Args:
        config_path (str): Path to the configuration file
        start_date (str): Start date for the backtest (format: YYYY-MM-DD)
        end_date (str): End date for the backtest (format: YYYY-MM-DD)
        initial_balance (float): Initial balance for backtesting
        
    Returns:
        dict: Backtest results
    """
    # Parse dates
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Create backtester
    backtester = Backtester(config_path, initial_balance)
    
    # Run backtest
    results = backtester.run_backtest(start_date, end_date)
    
    return results


if __name__ == "__main__":
    import argparse
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Trading Bot Backtester")
    parser.add_argument("--config", "-c", type=str, default="config.json",
                        help="Path to configuration file")
    parser.add_argument("--start-date", "-s", type=str, default=None,
                        help="Start date for backtesting (format: YYYY-MM-DD)")
    parser.add_argument("--end-date", "-e", type=str, default=None,
                        help="End date for backtesting (format: YYYY-MM-DD)")
    parser.add_argument("--initial-balance", "-b", type=float, default=10000.0,
                        help="Initial balance for backtesting")
    parser.add_argument("--strategy", "-t", type=str, default=None,
                        help="Strategy to backtest (overrides config)")
    
    args = parser.parse_args()
    
    # Update strategy if specified
    if args.strategy:
        # Load config
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # Update strategy
        config["trading"]["strategy"] = args.strategy
        
        # Save config
        with open(args.config, 'w') as f:
            json.dump(config, f, indent=4)
        
        logger.info(f"Updated strategy to: {args.strategy}")
    
    # Run backtest
    results = run_backtest(
        config_path=args.config,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_balance=args.initial_balance
    )
    
    # Print results
    print("\n=== Backtest Results ===")
    print(f"Initial Balance: ${results['initial_balance']:.2f}")
    print(f"Final Balance: ${results['final_balance']:.2f}")
    print(f"Total Return: ${results['total_return']:.2f} ({results['total_return_pct']:.2f}%)")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']*100:.2f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Results saved to backtest_results directory")