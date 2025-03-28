#!/usr/bin/env python3
"""
Example script demonstrating how to use the trading bot.
"""

import time
import logging
import os
import json
import argparse
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TradingBotExample")

def run_demo():
    """
    Run a demo of the trading bot.
    """
    logger.info("Starting trading bot demo")

    # Import trading bot
    from trading_bot import TradingBot

    # Create a trading bot with the default configuration
    bot = TradingBot()

    # Run the bot for a few iterations
    try:
        # Fetch market data
        logger.info("Fetching initial market data...")
        bot.fetch_market_data()

        # Run for 5 iterations
        for i in range(5):
            logger.info(f"Iteration {i+1}/5")

            # Analyze data and generate signals
            signals = bot.analyze_data()

            # Print signals
            logger.info("Generated signals:")
            for signal in signals:
                logger.info(f"  {signal['symbol']}: {signal['action'].upper()} "
                           f"with confidence {signal['confidence']:.2f}")

            # Execute trades based on signals
            bot.execute_trades(signals)

            # Print portfolio status
            bot.print_status()

            # Wait before next iteration
            if i < 4:  # Don't sleep after the last iteration
                logger.info("Waiting for 5 seconds...")
                time.sleep(5)

        logger.info("Demo completed successfully")

    except KeyboardInterrupt:
        logger.info("Demo stopped by user")
    except Exception as e:
        logger.error(f"Error in demo: {e}", exc_info=True)

def run_strategy_comparison():
    """
    Run a comparison of different trading strategies.
    """
    logger.info("Starting strategy comparison")

    # Import backtesting module
    try:
        from backtesting import run_backtest
    except ImportError:
        logger.error("Backtesting module not found. Please make sure backtesting.py is available.")
        return

    # Define strategies to compare
    strategies = [
        "simple_moving_average",
        "rsi_strategy",
        "macd_strategy",
        "bollinger_bands_strategy"
    ]

    # Try to add ML strategies if available
    try:
        import ml_strategy
        strategies.append("ml_strategy")
        strategies.append("ml_ensemble")
        logger.info("ML strategies added to comparison")
    except ImportError:
        logger.info("ML strategies not available for comparison")

    # Define backtest parameters
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    initial_balance = 10000.0

    # Create temporary config file
    temp_config_path = "temp_config.json"

    # Load default config
    with open("config.json", 'r') as f:
        config = json.load(f)

    # Results storage
    results = {}

    # Run backtest for each strategy
    for strategy in strategies:
        logger.info(f"Testing strategy: {strategy}")

        # Update config with current strategy
        config["trading"]["strategy"] = strategy

        # Save config
        with open(temp_config_path, 'w') as f:
            json.dump(config, f, indent=4)

        # Run backtest
        try:
            backtest_results = run_backtest(
                config_path=temp_config_path,
                start_date=start_date,
                end_date=end_date,
                initial_balance=initial_balance
            )

            # Store results
            results[strategy] = {
                "total_return_pct": backtest_results["total_return_pct"],
                "win_rate": backtest_results["win_rate"],
                "total_trades": backtest_results["total_trades"],
                "max_drawdown_pct": backtest_results["max_drawdown_pct"],
                "sharpe_ratio": backtest_results["sharpe_ratio"]
            }

            logger.info(f"Strategy {strategy}: {backtest_results['total_return_pct']:.2f}% return, "
                       f"{backtest_results['win_rate']*100:.2f}% win rate")

        except Exception as e:
            logger.error(f"Error testing strategy {strategy}: {e}")

    # Clean up
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)

    # Print comparison results
    logger.info("\n=== Strategy Comparison Results ===")
    logger.info(f"{'Strategy':<25} {'Return':<10} {'Win Rate':<10} {'Trades':<10} {'Max DD':<10} {'Sharpe':<10}")
    logger.info("-" * 75)

    for strategy, result in results.items():
        logger.info(f"{strategy:<25} {result['total_return_pct']:>8.2f}% {result['win_rate']*100:>8.2f}% "
                   f"{result['total_trades']:>8} {result['max_drawdown_pct']:>8.2f}% {result['sharpe_ratio']:>8.2f}")

    # Find best strategy
    if results:
        best_strategy = max(results.items(), key=lambda x: x[1]["total_return_pct"])
        logger.info(f"\nBest strategy by return: {best_strategy[0]} with {best_strategy[1]['total_return_pct']:.2f}% return")

        best_sharpe = max(results.items(), key=lambda x: x[1]["sharpe_ratio"])
        logger.info(f"Best strategy by Sharpe ratio: {best_sharpe[0]} with Sharpe ratio of {best_sharpe[1]['sharpe_ratio']:.2f}")

    logger.info("Strategy comparison completed")

def run_ml_training_demo():
    """
    Run a demo of ML strategy training.
    """
    logger.info("Starting ML strategy training demo")

    # Check if ML strategy is available
    try:
        from ml_strategy import MLStrategy
    except ImportError:
        logger.error("ML strategy module not found. Please make sure ml_strategy.py is available.")
        return

    # Import trading bot
    from trading_bot import TradingBot
    from data_fetcher import DemoDataFetcher

    # Create a demo data fetcher
    data_fetcher = DemoDataFetcher({
        "provider": "demo"
    })

    # Define symbols and interval
    symbols = ["BTC/USD", "ETH/USD"]
    interval = "1h"

    # Fetch historical data
    logger.info("Fetching historical data for ML training...")
    market_data = {}

    for symbol in symbols:
        # Fetch historical data
        historical_data = data_fetcher.fetch_historical_data(symbol, interval, limit=500)
        if historical_data is not None:
            market_data[symbol] = historical_data

    logger.info(f"Fetched historical data for {len(market_data)} symbols")

    # Create ML strategy
    ml_params = {
        "model_type": "random_forest",
        "prediction_horizon": 5,
        "feature_window": 20,
        "confidence_threshold": 0.6,
        "retrain_interval": 30,
        "model_path": "models"
    }

    ml_strategy = MLStrategy(ml_params)

    # Train models
    for symbol, data in market_data.items():
        logger.info(f"Training ML model for {symbol}...")
        model = ml_strategy.train_model(symbol, data)

        if model is not None:
            logger.info(f"Successfully trained model for {symbol}")
        else:
            logger.warning(f"Failed to train model for {symbol}")

    # Generate signals
    logger.info("Generating signals with trained models...")

    for symbol, data in market_data.items():
        signal = ml_strategy.generate_signal(symbol, data)

        logger.info(f"ML Signal for {symbol}: {signal['action'].upper()} with confidence {signal['confidence']:.2f}")

        if "metrics" in signal:
            for metric, value in signal["metrics"].items():
                logger.info(f"  {metric}: {value}")

    logger.info("ML strategy training demo completed")

def run_notification_demo():
    """
    Run a demo of the notification system.
    """
    logger.info("Starting notification system demo")

    # Import notification system
    try:
        from notification_system import NotificationSystem
    except ImportError:
        logger.error("Notification system module not found. Please make sure notification_system.py is available.")
        return

    # Create notification system
    notification_system = NotificationSystem()

    # Test trade executed notification
    trade_details = {
        "symbol": "BTC/USD",
        "action": "buy",
        "price": 50000.0,
        "size": 0.1
    }
    notification_system.notify_trade_executed(trade_details)

    # Test position closed notification
    position_details = {
        "symbol": "BTC/USD",
        "type": "long",
        "entry_price": 50000.0,
        "exit_price": 55000.0,
        "size": 0.1,
        "pnl": 500.0,
        "pnl_percent": 10.0,
        "reason": "take_profit"
    }
    notification_system.notify_position_closed(position_details)

    # Test stop-loss triggered notification
    stop_loss_details = {
        "symbol": "ETH/USD",
        "type": "long",
        "entry_price": 3000.0,
        "stop_loss": 2850.0,
        "size": 1.0
    }
    notification_system.notify_stop_loss_triggered(stop_loss_details)

    # Test take-profit triggered notification
    take_profit_details = {
        "symbol": "ETH/USD",
        "type": "long",
        "entry_price": 3000.0,
        "take_profit": 3300.0,
        "size": 1.0
    }
    notification_system.notify_take_profit_triggered(take_profit_details)

    # Test bot started notification
    notification_system.notify_bot_started()

    # Test bot stopped notification
    notification_system.notify_bot_stopped()

    # Test error notification
    notification_system.notify_error("Connection to exchange API failed: Timeout")

    logger.info("Notification system demo completed")

def run_dashboard_demo():
    """
    Run a demo of the web dashboard.
    """
    logger.info("Starting web dashboard demo")

    # Import dashboard
    try:
        from dashboard import TradingBotDashboard
    except ImportError:
        logger.error("Dashboard module not found. Please make sure dashboard.py is available.")
        return

    # Create and run dashboard
    dashboard = TradingBotDashboard(port=8050, debug=False)

    logger.info("Starting dashboard on port 8050. Press Ctrl+C to stop.")

    try:
        dashboard.run()
    except KeyboardInterrupt:
        logger.info("Dashboard demo stopped by user")
    except Exception as e:
        logger.error(f"Error in dashboard demo: {e}")

def run_comprehensive_demo():
    """
    Run a comprehensive demo of all features.
    """
    logger.info("Starting comprehensive demo")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Trading Bot Comprehensive Demo")
    parser.add_argument("--feature", type=str, choices=["basic", "strategies", "ml", "notifications", "dashboard", "all"],
                       default="basic", help="Feature to demonstrate")

    args = parser.parse_args()

    # Run selected demo
    if args.feature == "basic" or args.feature == "all":
        logger.info("\n=== Basic Trading Bot Demo ===")
        run_demo()

    if args.feature == "strategies" or args.feature == "all":
        logger.info("\n=== Strategy Comparison Demo ===")
        run_strategy_comparison()

    if args.feature == "ml" or args.feature == "all":
        logger.info("\n=== ML Strategy Training Demo ===")
        run_ml_training_demo()

    if args.feature == "notifications" or args.feature == "all":
        logger.info("\n=== Notification System Demo ===")
        run_notification_demo()

    if args.feature == "dashboard" or args.feature == "all":
        logger.info("\n=== Web Dashboard Demo ===")
        run_dashboard_demo()

    logger.info("Comprehensive demo completed")

if __name__ == "__main__":
    run_comprehensive_demo()