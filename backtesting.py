"""
Backtesting Module

This module provides functionality for backtesting trading strategies
on historical data to evaluate their performance.
"""

import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from copy import deepcopy

from strategies import create_strategy
from portfolio_manager import PortfolioManager
from risk_manager import RiskManager

logger = logging.getLogger("TradingBot.Backtesting")

class Backtester:
    """
    Backtester class for evaluating trading strategies on historical data.
    """
    
    def __init__(self, config_path="config.json", output_dir="backtest_results"):
        """
        Initialize the backtester with configuration.
        
        Args:
            config_path (str): Path to the configuration file
            output_dir (str): Directory to save backtest results
        """
        self.config = self._load_config(config_path)
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Initialize components
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
            logger.error(f"Configuration file {config_path} not found.")
            raise
    
    def load_historical_data(self, data_path=None, start_date=None, end_date=None):
        """
        Load historical data for backtesting.
        
        Args:
            data_path (str): Path to the historical data file
            start_date (str): Start date for backtesting (format: YYYY-MM-DD)
            end_date (str): End date for backtesting (format: YYYY-MM-DD)
            
        Returns:
            dict: Historical data for each symbol
        """
        if data_path:
            # Load data from file
            try:
                data = pd.read_csv(data_path, parse_dates=['timestamp'])
                logger.info(f"Loaded historical data from {data_path}")
            except Exception as e:
                logger.error(f"Failed to load historical data from {data_path}: {e}")
                raise
        else:
            # Generate synthetic data for demo purposes
            logger.info("Generating synthetic historical data for backtesting")
            data = self._generate_synthetic_data(start_date, end_date)
        
        # Filter by date range if specified
        if start_date:
            start_date = pd.to_datetime(start_date)
            data = data[data['timestamp'] >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            data = data[data['timestamp'] <= end_date]
        
        # Organize data by symbol
        symbols = self.config["trading"]["symbols"]
        historical_data = {}
        
        for symbol in symbols:
            symbol_data = data[data['symbol'] == symbol].copy()
            if len(symbol_data) == 0:
                logger.warning(f"No data found for symbol {symbol}")
                continue
            
            # Sort by timestamp
            symbol_data.sort_values('timestamp', inplace=True)
            
            historical_data[symbol] = {
                "symbol": symbol,
                "timestamps": symbol_data['timestamp'].tolist(),
                "historical_prices": symbol_data['close'].tolist(),
                "opens": symbol_data['open'].tolist(),
                "highs": symbol_data['high'].tolist(),
                "lows": symbol_data['low'].tolist(),
                "closes": symbol_data['close'].tolist(),
                "volumes": symbol_data['volume'].tolist()
            }
            
            logger.info(f"Prepared historical data for {symbol}: {len(symbol_data)} data points")
        
        return historical_data
    
    def _generate_synthetic_data(self, start_date=None, end_date=None):
        """
        Generate synthetic data for backtesting.
        
        Args:
            start_date (str): Start date for backtesting (format: YYYY-MM-DD)
            end_date (str): End date for backtesting (format: YYYY-MM-DD)
            
        Returns:
            pandas.DataFrame: Synthetic historical data
        """
        # Set default date range if not specified
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='1D')
        
        # Generate data for each symbol
        symbols = self.config["trading"]["symbols"]
        data_frames = []
        
        for symbol in symbols:
            # Set initial price based on symbol
            if symbol == "BTC/USD":
                initial_price = 50000.0
                volatility = 0.03
            elif symbol == "ETH/USD":
                initial_price = 3000.0
                volatility = 0.04
            else:
                initial_price = 100.0
                volatility = 0.02
            
            # Generate price series with random walk and some trend
            n = len(date_range)
            prices = np.zeros(n)
            prices[0] = initial_price
            
            # Add some trend and seasonality
            trend = np.linspace(0, 0.5, n)  # Upward trend
            seasonality = 0.1 * np.sin(np.linspace(0, 10 * np.pi, n))  # Seasonal pattern
            
            # Generate random walk with drift
            for i in range(1, n):
                daily_return = np.random.normal(0.0005, volatility)  # Mean daily return and volatility
                prices[i] = prices[i-1] * (1 + daily_return + trend[i] / n + seasonality[i])
            
            # Create DataFrame
            df = pd.DataFrame({
                'timestamp': date_range,
                'symbol': symbol,
                'open': prices * (1 + np.random.normal(0, 0.005, n)),
                'high': prices * (1 + np.random.normal(0.01, 0.005, n)),
                'low': prices * (1 - np.random.normal(0.01, 0.005, n)),
                'close': prices,
                'volume': initial_price * 10 * (1 + np.random.normal(0, 0.2, n))
            })
            
            data_frames.append(df)
        
        # Combine all data
        combined_data = pd.concat(data_frames, ignore_index=True)
        
        logger.info(f"Generated synthetic data for {len(symbols)} symbols from {start_date.date()} to {end_date.date()}")
        return combined_data
    
    def run_backtest(self, historical_data, initial_balance=10000.0):
        """
        Run a backtest using the specified strategy and historical data.
        
        Args:
            historical_data (dict): Historical data for each symbol
            initial_balance (float): Initial portfolio balance
            
        Returns:
            dict: Backtest results
        """
        logger.info(f"Starting backtest with initial balance: {initial_balance:.2f}")
        
        # Initialize portfolio for backtesting
        portfolio = PortfolioManager(initial_balance=initial_balance, portfolio_file=None)
        
        # Track portfolio value over time
        portfolio_values = []
        trade_history = []
        signals_history = []
        
        # Get the list of timestamps (assuming all symbols have the same timestamps)
        symbol = list(historical_data.keys())[0]
        timestamps = historical_data[symbol]["timestamps"]
        
        # Run the backtest for each timestamp
        for i, timestamp in enumerate(timestamps):
            current_prices = {}
            
            # Get current prices for all symbols
            for symbol, data in historical_data.items():
                if i < len(data["historical_prices"]):
                    current_prices[symbol] = data["closes"][i]
            
            # Generate signals for each symbol
            signals = []
            for symbol, data in historical_data.items():
                if i < len(data["historical_prices"]):
                    # Prepare data for strategy
                    strategy_data = {
                        "symbol": symbol,
                        "current_price": current_prices[symbol],
                        "historical_prices": data["historical_prices"][:i+1],
                        "timestamps": data["timestamps"][:i+1],
                        "opens": data["opens"][:i+1],
                        "highs": data["highs"][:i+1],
                        "lows": data["lows"][:i+1],
                        "closes": data["closes"][:i+1],
                        "volumes": data["volumes"][:i+1]
                    }
                    
                    # Generate signal
                    signal = self.strategy.generate_signal(symbol, strategy_data)
                    signals.append(signal)
                    
                    # Record signal
                    signal_record = deepcopy(signal)
                    signal_record["timestamp"] = timestamp
                    signals_history.append(signal_record)
            
            # Execute trades based on signals
            for signal in signals:
                symbol = signal["symbol"]
                action = signal["action"]
                confidence = signal["confidence"]
                current_price = current_prices[symbol]
                
                # Check if we already have a position for this symbol
                has_position = portfolio.has_position(symbol)
                
                if action == "buy" and not has_position and confidence > 0.2:
                    # Calculate position size
                    portfolio_value = portfolio.get_portfolio_value(current_prices)
                    position_size = self.risk_manager.calculate_position_size(
                        portfolio_value, current_price, confidence)
                    
                    # Calculate stop-loss and take-profit prices
                    stop_loss_price = self.risk_manager.calculate_stop_loss_price(current_price, "long")
                    take_profit_price = self.risk_manager.calculate_take_profit_price(current_price, "long")
                    
                    # Open a long position
                    success = portfolio.open_position(
                        symbol, "long", current_price, position_size, stop_loss_price, take_profit_price)
                    
                    if success:
                        logger.debug(f"[{timestamp}] Opened LONG position for {symbol} at {current_price:.2f}")
                
                elif action == "sell" and has_position:
                    # Close the position
                    position = portfolio.get_position(symbol)
                    if position["type"] == "long":
                        success, pnl = portfolio.close_position(symbol, current_price, "signal")
                        if success:
                            logger.debug(f"[{timestamp}] Closed LONG position for {symbol} at {current_price:.2f}")
                
                # Check existing positions for stop-loss and take-profit
                if has_position:
                    position = portfolio.get_position(symbol)
                    should_close, reason = self.risk_manager.should_close_position(position, current_price)
                    
                    if should_close:
                        success, pnl = portfolio.close_position(symbol, current_price, reason)
                        if success:
                            logger.debug(f"[{timestamp}] Closed position for {symbol} at {current_price:.2f} due to {reason}")
                    else:
                        # Update trailing stop if applicable
                        updated_position = self.risk_manager.update_trailing_stop(position, current_price)
                        if updated_position != position:
                            portfolio.positions[symbol] = updated_position
            
            # Record portfolio value
            portfolio_value = portfolio.get_portfolio_value(current_prices)
            portfolio_values.append({
                "timestamp": timestamp,
                "value": portfolio_value
            })
        
        # Get final trade history
        trade_history = portfolio.trade_history
        
        # Calculate performance metrics
        metrics = portfolio.get_performance_metrics()
        
        # Calculate additional metrics
        initial_value = initial_balance
        final_value = portfolio_values[-1]["value"] if portfolio_values else initial_balance
        
        total_return = (final_value / initial_value - 1) * 100
        
        # Calculate drawdown
        max_drawdown, max_drawdown_duration = self._calculate_drawdown(portfolio_values)
        
        # Calculate Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_values)
        
        # Prepare results
        results = {
            "initial_balance": initial_balance,
            "final_balance": portfolio.balance,
            "final_portfolio_value": final_value,
            "total_return_pct": total_return,
            "max_drawdown_pct": max_drawdown,
            "max_drawdown_duration_days": max_drawdown_duration,
            "sharpe_ratio": sharpe_ratio,
            "total_trades": metrics["total_trades"],
            "winning_trades": metrics["winning_trades"],
            "losing_trades": metrics["losing_trades"],
            "win_rate": metrics["win_rate"],
            "total_profit_loss": metrics["total_profit_loss"],
            "average_profit_loss": metrics["average_profit_loss"],
            "portfolio_values": portfolio_values,
            "trade_history": trade_history,
            "signals_history": signals_history
        }
        
        logger.info(f"Backtest completed: {results['total_return_pct']:.2f}% return, "
                   f"{results['total_trades']} trades, {results['win_rate']*100:.2f}% win rate")
        
        return results
    
    def _calculate_drawdown(self, portfolio_values):
        """
        Calculate the maximum drawdown and its duration.
        
        Args:
            portfolio_values (list): List of portfolio values over time
            
        Returns:
            tuple: (max_drawdown_pct, max_drawdown_duration_days)
        """
        if not portfolio_values:
            return 0.0, 0
        
        # Extract values
        values = [entry["value"] for entry in portfolio_values]
        timestamps = [entry["timestamp"] for entry in portfolio_values]
        
        # Calculate drawdown
        peak = values[0]
        max_drawdown = 0.0
        max_drawdown_duration = 0
        current_drawdown_start = 0
        
        for i, value in enumerate(values):
            if value > peak:
                peak = value
                current_drawdown_start = i
            
            drawdown = (peak - value) / peak * 100
            
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                # Calculate duration in days
                duration = (timestamps[i] - timestamps[current_drawdown_start]) / (24 * 3600)
                max_drawdown_duration = duration
        
        return max_drawdown, max_drawdown_duration
    
    def _calculate_sharpe_ratio(self, portfolio_values, risk_free_rate=0.02):
        """
        Calculate the Sharpe ratio.
        
        Args:
            portfolio_values (list): List of portfolio values over time
            risk_free_rate (float): Annual risk-free rate
            
        Returns:
            float: Sharpe ratio
        """
        if len(portfolio_values) < 2:
            return 0.0
        
        # Extract values
        values = np.array([entry["value"] for entry in portfolio_values])
        
        # Calculate daily returns
        daily_returns = np.diff(values) / values[:-1]
        
        # Calculate annualized Sharpe ratio
        mean_daily_return = np.mean(daily_returns)
        std_daily_return = np.std(daily_returns)
        
        if std_daily_return == 0:
            return 0.0
        
        # Assuming 252 trading days in a year
        sharpe_ratio = (mean_daily_return * 252 - risk_free_rate) / (std_daily_return * np.sqrt(252))
        
        return sharpe_ratio
    
    def save_results(self, results, filename=None):
        """
        Save backtest results to a file.
        
        Args:
            results (dict): Backtest results
            filename (str): Filename to save results
            
        Returns:
            str: Path to the saved results file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            strategy_name = self.strategy.name
            filename = f"backtest_{strategy_name}_{timestamp}.json"
        
        file_path = os.path.join(self.output_dir, filename)
        
        # Create a copy of results that can be serialized to JSON
        serializable_results = deepcopy(results)
        
        # Convert timestamps to strings
        for entry in serializable_results["portfolio_values"]:
            if isinstance(entry["timestamp"], (datetime, pd.Timestamp)):
                entry["timestamp"] = entry["timestamp"].isoformat()
        
        for entry in serializable_results["signals_history"]:
            if isinstance(entry["timestamp"], (datetime, pd.Timestamp)):
                entry["timestamp"] = entry["timestamp"].isoformat()
        
        for trade in serializable_results["trade_history"]:
            trade["open_time"] = datetime.fromtimestamp(trade["open_time"]).isoformat()
            trade["close_time"] = datetime.fromtimestamp(trade["close_time"]).isoformat()
        
        try:
            with open(file_path, 'w') as f:
                json.dump(serializable_results, f, indent=4)
            
            logger.info(f"Saved backtest results to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to save backtest results: {e}")
            return None
    
    def plot_results(self, results, filename=None):
        """
        Plot backtest results.
        
        Args:
            results (dict): Backtest results
            filename (str): Filename to save the plot
            
        Returns:
            str: Path to the saved plot file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            strategy_name = self.strategy.name
            filename = f"backtest_{strategy_name}_{timestamp}.png"
        
        file_path = os.path.join(self.output_dir, filename)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot portfolio value
        timestamps = [entry["timestamp"] for entry in results["portfolio_values"]]
        values = [entry["value"] for entry in results["portfolio_values"]]
        
        ax1.plot(timestamps, values, label="Portfolio Value")
        ax1.set_title(f"Backtest Results - {self.strategy.name}")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Portfolio Value")
        ax1.grid(True)
        ax1.legend()
        
        # Plot trades
        buy_timestamps = []
        buy_values = []
        sell_timestamps = []
        sell_values = []
        
        for trade in results["trade_history"]:
            open_time = datetime.fromtimestamp(trade["open_time"]) if isinstance(trade["open_time"], (int, float)) else trade["open_time"]
            close_time = datetime.fromtimestamp(trade["close_time"]) if isinstance(trade["close_time"], (int, float)) else trade["close_time"]
            
            # Find portfolio value at open and close times
            open_value = None
            close_value = None
            
            for i, entry in enumerate(results["portfolio_values"]):
                entry_time = entry["timestamp"]
                if isinstance(entry_time, str):
                    entry_time = pd.to_datetime(entry_time)
                
                if open_time <= entry_time and open_value is None:
                    open_value = values[i]
                
                if close_time <= entry_time and close_value is None:
                    close_value = values[i]
                    break
            
            if open_value is not None:
                buy_timestamps.append(open_time)
                buy_values.append(open_value)
            
            if close_value is not None:
                sell_timestamps.append(close_time)
                sell_values.append(close_value)
        
        ax1.scatter(buy_timestamps, buy_values, color='green', marker='^', label="Buy")
        ax1.scatter(sell_timestamps, sell_values, color='red', marker='v', label="Sell")
        ax1.legend()
        
        # Plot drawdown
        drawdowns = []
        peak = values[0]
        
        for value in values:
            if value > peak:
                peak = value
                drawdowns.append(0.0)
            else:
                drawdown = (peak - value) / peak * 100
                drawdowns.append(drawdown)
        
        ax2.fill_between(timestamps, drawdowns, color='red', alpha=0.3)
        ax2.set_title("Drawdown")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Drawdown (%)")
        ax2.grid(True)
        ax2.invert_yaxis()  # Invert y-axis to show drawdowns as negative
        
        # Add summary statistics as text
        summary = (
            f"Total Return: {results['total_return_pct']:.2f}%\n"
            f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {results['max_drawdown_pct']:.2f}%\n"
            f"Win Rate: {results['win_rate']*100:.2f}%\n"
            f"Total Trades: {results['total_trades']}"
        )
        
        plt.figtext(0.01, 0.01, summary, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        # Adjust layout and save
        plt.tight_layout()
        
        try:
            plt.savefig(file_path)
            logger.info(f"Saved backtest plot to {file_path}")
            plt.close(fig)
            return file_path
        except Exception as e:
            logger.error(f"Failed to save backtest plot: {e}")
            plt.close(fig)
            return None


def run_backtest(config_path="config.json", data_path=None, start_date=None, end_date=None, 
                initial_balance=10000.0, output_dir="backtest_results"):
    """
    Run a backtest with the specified parameters.
    
    Args:
        config_path (str): Path to the configuration file
        data_path (str): Path to the historical data file
        start_date (str): Start date for backtesting (format: YYYY-MM-DD)
        end_date (str): End date for backtesting (format: YYYY-MM-DD)
        initial_balance (float): Initial portfolio balance
        output_dir (str): Directory to save backtest results
        
    Returns:
        dict: Backtest results
    """
    # Initialize backtester
    backtester = Backtester(config_path, output_dir)
    
    # Load historical data
    historical_data = backtester.load_historical_data(data_path, start_date, end_date)
    
    # Run backtest
    results = backtester.run_backtest(historical_data, initial_balance)
    
    # Save results
    backtester.save_results(results)
    
    # Plot results
    backtester.plot_results(results)
    
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
    parser = argparse.ArgumentParser(description="Backtest trading strategies")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to configuration file")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to historical data file")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date for backtesting (format: YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None,
                        help="End date for backtesting (format: YYYY-MM-DD)")
    parser.add_argument("--balance", type=float, default=10000.0,
                        help="Initial portfolio balance")
    parser.add_argument("--output", type=str, default="backtest_results",
                        help="Directory to save backtest results")
    
    args = parser.parse_args()
    
    # Run backtest
    run_backtest(
        config_path=args.config,
        data_path=args.data,
        start_date=args.start,
        end_date=args.end,
        initial_balance=args.balance,
        output_dir=args.output
    )