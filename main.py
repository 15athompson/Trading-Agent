#!/usr/bin/env python3
"""
Main script for the Trading AI Agent Bot.

This script provides a command-line interface for running the trading bot,
backtesting strategies, and starting the web dashboard.
"""

import argparse
import logging
import sys
import os
import json
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TradingBot.Main")

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Trading AI Agent Bot")
    
    # General options
    parser.add_argument("--config", "-c", type=str, default="config.json",
                        help="Path to configuration file")
    parser.add_argument("--notification-config", "-n", type=str, default="notification_config.json",
                        help="Path to notification configuration file")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--run", "-r", action="store_true",
                           help="Run the trading bot (default)")
    mode_group.add_argument("--backtest", "-b", action="store_true",
                           help="Run in backtest mode")
    mode_group.add_argument("--dashboard", "-d", action="store_true",
                           help="Start the web dashboard")
    mode_group.add_argument("--example", "-e", action="store_true",
                           help="Run the example script")
    mode_group.add_argument("--setup", "-s", action="store_true",
                           help="Set up configuration files")
    
    # Backtest options
    parser.add_argument("--backtest-start", type=str, default=None,
                        help="Start date for backtesting (format: YYYY-MM-DD)")
    parser.add_argument("--backtest-end", type=str, default=None,
                        help="End date for backtesting (format: YYYY-MM-DD)")
    parser.add_argument("--initial-balance", type=float, default=10000.0,
                        help="Initial balance for backtesting")
    
    # Dashboard options
    parser.add_argument("--dashboard-port", type=int, default=8050,
                        help="Port for the web dashboard")
    
    # Strategy options
    parser.add_argument("--strategy", type=str, default=None,
                        help="Trading strategy to use")
    
    return parser.parse_args()

def setup_configuration(args):
    """
    Set up configuration files.
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
    logger.info("Setting up configuration files")
    
    # Create config.json
    config = {
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
            "backtest_mode": False,
            "use_ml": False,
            "use_notifications": True
        }
    }
    
    # Write config.json
    with open(args.config, 'w') as f:
        json.dump(config, f, indent=4)
    
    logger.info(f"Created configuration file: {args.config}")
    
    # Create notification_config.json
    from notification_system import create_notification_config
    notification_config = create_notification_config(args.notification_config)
    
    logger.info(f"Created notification configuration file: {args.notification_config}")
    
    # Create directories
    directories = ["models", "backtest_results"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
    
    logger.info("Configuration setup complete")

def run_trading_bot(args):
    """
    Run the trading bot.
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
    from trading_bot import TradingBot
    
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
    
    # Create and run the trading bot
    bot = TradingBot(args.config, args.notification_config)
    bot.run()

def run_backtest(args):
    """
    Run a backtest.
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
    try:
        # Import backtesting module
        from backtesting import run_backtest
        
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
        
        # Set default date range if not specified
        if not args.backtest_start:
            args.backtest_start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        if not args.backtest_end:
            args.backtest_end = datetime.now().strftime('%Y-%m-%d')
        
        # Run backtest
        logger.info(f"Running backtest from {args.backtest_start} to {args.backtest_end}")
        results = run_backtest(
            config_path=args.config,
            start_date=args.backtest_start,
            end_date=args.backtest_end,
            initial_balance=args.initial_balance
        )
        
        logger.info(f"Backtest completed with {results['total_return_pct']:.2f}% return")
        logger.info(f"Win rate: {results['win_rate']*100:.2f}%")
        logger.info(f"Total trades: {results['total_trades']}")
        logger.info(f"Max drawdown: {results['max_drawdown_pct']:.2f}%")
        logger.info(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
    
    except ImportError:
        logger.error("Backtesting module not found. Please make sure backtesting.py is available.")
        sys.exit(1)

def start_dashboard(args):
    """
    Start the web dashboard.
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
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

def run_example(args):
    """
    Run the example script.
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
    try:
        # Import example module
        from example import run_demo
        
        # Run example
        logger.info("Running example script")
        run_demo()
    
    except ImportError:
        logger.error("Example module not found. Please make sure example.py is available.")
        sys.exit(1)

def main():
    """
    Main function.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger("TradingBot").setLevel(logging.DEBUG)
    
    # Run the selected mode
    if args.setup:
        setup_configuration(args)
    elif args.backtest:
        run_backtest(args)
    elif args.dashboard:
        start_dashboard(args)
    elif args.example:
        run_example(args)
    else:
        # Default to running the trading bot
        run_trading_bot(args)

if __name__ == "__main__":
    main()