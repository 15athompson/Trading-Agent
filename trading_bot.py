#!/usr/bin/env python3
"""
Trading AI Agent Bot

This script implements a trading bot that can analyze market data,
make trading decisions based on configured strategies, and execute trades.
"""

import time
import logging
from datetime import datetime
import json
import os
import argparse
import importlib
import sys

# Import custom modules
from data_fetcher import create_data_fetcher
from strategies import create_strategy
from risk_manager import RiskManager
from portfolio_manager import PortfolioManager
from notification_system import NotificationSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TradingBot")

class TradingBot:
    """
    Main trading bot class that handles data fetching, analysis, and trade execution.
    """

    def __init__(self, config_path="config.json", notification_config_path="notification_config.json"):
        """
        Initialize the trading bot with configuration.

        Args:
            config_path (str): Path to the configuration file
            notification_config_path (str): Path to the notification configuration file
        """
        self.config = self._load_config(config_path)
        self.market_data = {}
        self.current_prices = {}
        self.running = False

        # Initialize components
        self.data_fetcher = create_data_fetcher(self.config["api"])
        self.risk_manager = RiskManager(self.config["trading"]["risk_management"])
        self.portfolio_manager = PortfolioManager()
        self.notification_system = NotificationSystem(notification_config_path)

        # Initialize strategies
        self.strategies = []
        strategy_name = self.config["trading"]["strategy"]
        strategy_params = self.config["trading"].get("strategy_params", {})

        # Check if we need to import ML strategy
        if strategy_name in ["ml_strategy", "ml_ensemble"]:
            try:
                # Try to import the ML strategy module
                if "ml_strategy" not in sys.modules:
                    importlib.import_module("ml_strategy")
                logger.info("ML strategy module imported successfully")
            except ImportError:
                logger.warning("ML strategy module not found. Falling back to SMA strategy.")
                strategy_name = "simple_moving_average"

        self.strategies.append(create_strategy(strategy_name, strategy_params))

        logger.info("Trading bot initialized with strategy: " + strategy_name)

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
            return self._create_default_config(config_path)

    def _create_default_config(self, config_path):
        """
        Create a default configuration file.

        Args:
            config_path (str): Path to save the default configuration

        Returns:
            dict: Default configuration parameters
        """
        default_config = {
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
                    "max_position_size": 0.1,  # 10% of portfolio
                    "stop_loss": 0.05,  # 5% loss
                    "take_profit": 0.1  # 10% profit
                }
            },
            "bot_settings": {
                "update_interval": 60,  # seconds
                "backtest_mode": False,
                "use_ml": False,
                "use_notifications": True
            }
        }

        try:
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            logger.info(f"Default configuration created at {config_path}")
        except Exception as e:
            logger.error(f"Failed to create default configuration: {e}")

        return default_config

    def fetch_market_data(self):
        """
        Fetch market data from the configured API.
        """
        logger.info("Fetching market data...")

        symbols = self.config["trading"]["symbols"]
        interval = self.config["trading"]["interval"]

        for symbol in symbols:
            # Fetch current price
            current_price = self.data_fetcher.fetch_current_price(symbol)
            if current_price is not None:
                self.current_prices[symbol] = current_price

            # Fetch historical data
            historical_data = self.data_fetcher.fetch_historical_data(symbol, interval)
            if historical_data is not None:
                self.market_data[symbol] = historical_data

        logger.info(f"Market data updated for {len(self.market_data)} symbols")

    def analyze_data(self):
        """
        Analyze market data using the configured strategies.

        Returns:
            list: Trading signals generated by the strategies
        """
        logger.info("Analyzing market data...")
        signals = []

        for symbol, data in self.market_data.items():
            # Add current price to the data
            if symbol in self.current_prices:
                data["current_price"] = self.current_prices[symbol]

            # Apply each strategy to generate signals
            for strategy in self.strategies:
                signal = strategy.generate_signal(symbol, data)
                signals.append(signal)

        logger.info(f"Generated {len(signals)} trading signals")
        return signals

    def execute_trades(self, signals):
        """
        Execute trades based on the generated signals.

        Args:
            signals (list): Trading signals to act upon
        """
        logger.info("Executing trades...")

        # Get portfolio value for position sizing
        portfolio_value = self.portfolio_manager.get_portfolio_value(self.current_prices)

        for signal in signals:
            symbol = signal["symbol"]
            action = signal["action"]
            confidence = signal["confidence"]

            # Skip if no current price available
            if symbol not in self.current_prices:
                logger.warning(f"No current price available for {symbol}, skipping trade execution")
                continue

            current_price = self.current_prices[symbol]

            # Check if we already have a position for this symbol
            has_position = self.portfolio_manager.has_position(symbol)

            if action == "buy" and not has_position and confidence > 0.2:
                # Calculate position size based on portfolio value and confidence
                position_size = self.risk_manager.calculate_position_size(
                    portfolio_value, current_price, confidence)

                # Calculate stop-loss and take-profit prices
                stop_loss_price = self.risk_manager.calculate_stop_loss_price(current_price, "long")
                take_profit_price = self.risk_manager.calculate_take_profit_price(current_price, "long")

                # Open a long position
                success = self.portfolio_manager.open_position(
                    symbol, "long", current_price, position_size, stop_loss_price, take_profit_price)

                if success:
                    logger.info(f"Opened LONG position for {symbol} at {current_price:.2f} "
                               f"with size {position_size:.6f} and confidence {confidence:.2f}")

                    # Send notification
                    trade_details = {
                        "symbol": symbol,
                        "action": "buy",
                        "price": current_price,
                        "size": position_size
                    }
                    self.notification_system.notify_trade_executed(trade_details)

            elif action == "sell" and has_position:
                # Close the position
                position = self.portfolio_manager.get_position(symbol)
                if position["type"] == "long":
                    success, pnl = self.portfolio_manager.close_position(symbol, current_price, "signal")
                    if success:
                        logger.info(f"Closed LONG position for {symbol} at {current_price:.2f} "
                                   f"with P&L {pnl:.2f}")

                        # Send notification
                        position_details = {
                            "symbol": symbol,
                            "type": "long",
                            "entry_price": position["entry_price"],
                            "exit_price": current_price,
                            "size": position["size"],
                            "pnl": pnl,
                            "pnl_percent": (pnl / (position["entry_price"] * position["size"])) * 100,
                            "reason": "signal"
                        }
                        self.notification_system.notify_position_closed(position_details)

            elif action == "sell" and not has_position and confidence > 0.2:
                # For simplicity, we're not implementing short selling in this example
                # In a real bot, you would implement short selling logic here
                pass

            # Check existing positions for stop-loss and take-profit
            if has_position:
                position = self.portfolio_manager.get_position(symbol)
                should_close, reason = self.risk_manager.should_close_position(position, current_price)

                if should_close:
                    success, pnl = self.portfolio_manager.close_position(symbol, current_price, reason)
                    if success:
                        logger.info(f"Closed position for {symbol} at {current_price:.2f} "
                                   f"with P&L {pnl:.2f} due to {reason}")

                        # Send notification
                        position_details = {
                            "symbol": symbol,
                            "type": position["type"],
                            "entry_price": position["entry_price"],
                            "exit_price": current_price,
                            "size": position["size"],
                            "pnl": pnl,
                            "pnl_percent": (pnl / (position["entry_price"] * position["size"])) * 100,
                            "reason": reason
                        }

                        if reason == "stop_loss":
                            self.notification_system.notify_stop_loss_triggered(position)
                        elif reason == "take_profit":
                            self.notification_system.notify_take_profit_triggered(position)

                        self.notification_system.notify_position_closed(position_details)
                else:
                    # Update trailing stop if applicable
                    updated_position = self.risk_manager.update_trailing_stop(position, current_price)
                    if updated_position != position:
                        self.portfolio_manager.positions[symbol] = updated_position
                        self.portfolio_manager.save_portfolio()

    def print_status(self):
        """
        Print the current status of the trading bot.
        """
        logger.info("=== Trading Bot Status ===")

        # Print current prices
        logger.info("Current Prices:")
        for symbol, price in self.current_prices.items():
            logger.info(f"  {symbol}: {price:.2f}")

        # Print portfolio information
        balance = self.portfolio_manager.balance
        positions = self.portfolio_manager.positions
        portfolio_value = self.portfolio_manager.get_portfolio_value(self.current_prices)

        logger.info(f"Portfolio Balance: {balance:.2f}")
        logger.info(f"Portfolio Value: {portfolio_value:.2f}")
        logger.info(f"Open Positions: {len(positions)}")

        for symbol, position in positions.items():
            entry_price = position["entry_price"]
            current_price = self.current_prices.get(symbol, entry_price)
            size = position["size"]
            position_type = position["type"]

            if position_type == "long":
                pnl = (current_price - entry_price) * size
                pnl_percent = (current_price / entry_price - 1) * 100
            else:  # short position
                pnl = (entry_price - current_price) * size
                pnl_percent = (entry_price / current_price - 1) * 100

            logger.info(f"  {symbol} ({position_type.upper()}): Size={size:.6f}, "
                       f"Entry={entry_price:.2f}, Current={current_price:.2f}, "
                       f"P&L={pnl:.2f} ({pnl_percent:.2f}%)")

        # Print performance metrics
        metrics = self.portfolio_manager.get_performance_metrics()
        logger.info("Performance Metrics:")
        logger.info(f"  Total Trades: {metrics['total_trades']}")
        logger.info(f"  Win Rate: {metrics['win_rate']*100:.2f}%")
        logger.info(f"  Total P&L: {metrics['total_profit_loss']:.2f}")
        logger.info(f"  Average P&L: {metrics['average_profit_loss']:.2f}")

    def run(self):
        """
        Run the trading bot in a loop.
        """
        self.running = True
        logger.info("Trading bot started")

        # Send notification
        self.notification_system.notify_bot_started()

        try:
            while self.running:
                self.fetch_market_data()
                signals = self.analyze_data()
                self.execute_trades(signals)
                self.print_status()

                # Sleep for the configured interval
                sleep_time = self.config["bot_settings"]["update_interval"]
                logger.info(f"Sleeping for {sleep_time} seconds")
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
        except Exception as e:
            logger.error(f"Error in trading bot: {e}", exc_info=True)
            # Send error notification
            self.notification_system.notify_error(str(e))
        finally:
            self.running = False
            logger.info("Trading bot stopped")
            # Send notification
            self.notification_system.notify_bot_stopped()

    def stop(self):
        """
        Stop the trading bot.
        """
        self.running = False
        logger.info("Trading bot stopping...")


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Trading AI Agent Bot")
    parser.add_argument("--config", "-c", type=str, default="config.json",
                        help="Path to configuration file")
    parser.add_argument("--notification-config", "-n", type=str, default="notification_config.json",
                        help="Path to notification configuration file")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--backtest", "-b", action="store_true",
                        help="Run in backtest mode")
    parser.add_argument("--backtest-start", type=str, default=None,
                        help="Start date for backtesting (format: YYYY-MM-DD)")
    parser.add_argument("--backtest-end", type=str, default=None,
                        help="End date for backtesting (format: YYYY-MM-DD)")
    parser.add_argument("--dashboard", "-d", action="store_true",
                        help="Start the web dashboard")
    parser.add_argument("--dashboard-port", type=int, default=8050,
                        help="Port for the web dashboard")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger("TradingBot").setLevel(logging.DEBUG)

    # Check if we should run in backtest mode
    if args.backtest:
        try:
            # Import backtesting module
            from backtesting import run_backtest

            # Run backtest
            logger.info("Running in backtest mode")
            results = run_backtest(
                config_path=args.config,
                start_date=args.backtest_start,
                end_date=args.backtest_end
            )

            logger.info(f"Backtest completed with {results['total_return_pct']:.2f}% return")

        except ImportError:
            logger.error("Backtesting module not found. Please make sure backtesting.py is available.")
            sys.exit(1)

    # Check if we should start the dashboard
    elif args.dashboard:
        try:
            # Import dashboard module
            from dashboard import TradingBotDashboard

            # Start dashboard
            logger.info(f"Starting dashboard on port {args.dashboard_port}")
            dashboard = TradingBotDashboard(
                config_path=args.config,
                port=args.dashboard_port,
                debug=args.verbose
            )

            dashboard.run()

        except ImportError:
            logger.error("Dashboard module not found. Please make sure dashboard.py is available.")
            sys.exit(1)

    # Otherwise, run the trading bot normally
    else:
        # Create and run the trading bot
        bot = TradingBot(args.config, args.notification_config)
        bot.run()