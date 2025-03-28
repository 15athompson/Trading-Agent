#!/usr/bin/env python3
"""
Example script demonstrating how to use the trading bot.
"""

import time
import logging
from trading_bot import TradingBot

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

if __name__ == "__main__":
    run_demo()