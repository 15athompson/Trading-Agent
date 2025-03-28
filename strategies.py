"""
Trading Strategies Module

This module contains various trading strategies that can be used by the trading bot.
"""

import numpy as np
import logging

logger = logging.getLogger("TradingBot.Strategies")

class Strategy:
    """Base strategy class that all strategies should inherit from."""
    
    def __init__(self, params=None):
        """
        Initialize the strategy with parameters.
        
        Args:
            params (dict): Strategy parameters
        """
        self.params = params or {}
        self.name = "base_strategy"
    
    def generate_signal(self, symbol, data):
        """
        Generate a trading signal based on the market data.
        
        Args:
            symbol (str): Trading symbol
            data (dict): Market data for the symbol
            
        Returns:
            dict: Trading signal
        """
        # Base strategy doesn't generate any signals
        return {
            "symbol": symbol,
            "action": "hold",
            "confidence": 0.0,
            "timestamp": data.get("timestamp", 0)
        }


class SimpleMovingAverageStrategy(Strategy):
    """
    Simple Moving Average (SMA) crossover strategy.
    
    Generates buy signals when the short-term SMA crosses above the long-term SMA,
    and sell signals when the short-term SMA crosses below the long-term SMA.
    """
    
    def __init__(self, params=None):
        """
        Initialize the SMA strategy with parameters.
        
        Args:
            params (dict): Strategy parameters including short_window and long_window
        """
        default_params = {
            "short_window": 20,
            "long_window": 50
        }
        super().__init__(params or default_params)
        self.name = "simple_moving_average"
        self.short_window = self.params.get("short_window", 20)
        self.long_window = self.params.get("long_window", 50)
        
        logger.info(f"SMA Strategy initialized with short_window={self.short_window}, long_window={self.long_window}")
    
    def calculate_sma(self, prices, window):
        """
        Calculate the Simple Moving Average.
        
        Args:
            prices (list): List of prices
            window (int): Window size for the moving average
            
        Returns:
            float: The Simple Moving Average value
        """
        if len(prices) < window:
            return None
        return sum(prices[-window:]) / window
    
    def generate_signal(self, symbol, data):
        """
        Generate a trading signal based on SMA crossover.
        
        Args:
            symbol (str): Trading symbol
            data (dict): Market data for the symbol, including historical prices
            
        Returns:
            dict: Trading signal
        """
        if "historical_prices" not in data or len(data["historical_prices"]) < self.long_window:
            logger.warning(f"Not enough historical data for {symbol} to generate SMA signal")
            return super().generate_signal(symbol, data)
        
        prices = data["historical_prices"]
        
        # Calculate SMAs
        short_sma = self.calculate_sma(prices, self.short_window)
        long_sma = self.calculate_sma(prices, self.long_window)
        
        if short_sma is None or long_sma is None:
            return super().generate_signal(symbol, data)
        
        # Calculate previous SMAs (one period ago)
        prev_short_sma = self.calculate_sma(prices[:-1], self.short_window)
        prev_long_sma = self.calculate_sma(prices[:-1], self.long_window)
        
        if prev_short_sma is None or prev_long_sma is None:
            return super().generate_signal(symbol, data)
        
        # Generate signal based on crossover
        action = "hold"
        confidence = 0.0
        
        # Bullish crossover (short SMA crosses above long SMA)
        if prev_short_sma <= prev_long_sma and short_sma > long_sma:
            action = "buy"
            # Calculate confidence based on the strength of the crossover
            confidence = min(1.0, (short_sma - long_sma) / long_sma)
        
        # Bearish crossover (short SMA crosses below long SMA)
        elif prev_short_sma >= prev_long_sma and short_sma < long_sma:
            action = "sell"
            # Calculate confidence based on the strength of the crossover
            confidence = min(1.0, (long_sma - short_sma) / long_sma)
        
        logger.info(f"SMA Signal for {symbol}: {action.upper()} with confidence {confidence:.2f}")
        
        return {
            "symbol": symbol,
            "action": action,
            "confidence": confidence,
            "timestamp": data.get("timestamp", 0),
            "metrics": {
                "short_sma": short_sma,
                "long_sma": long_sma
            }
        }


class RSIStrategy(Strategy):
    """
    Relative Strength Index (RSI) strategy.
    
    Generates buy signals when RSI is below the oversold threshold,
    and sell signals when RSI is above the overbought threshold.
    """
    
    def __init__(self, params=None):
        """
        Initialize the RSI strategy with parameters.
        
        Args:
            params (dict): Strategy parameters including period, oversold, and overbought thresholds
        """
        default_params = {
            "period": 14,
            "oversold": 30,
            "overbought": 70
        }
        super().__init__(params or default_params)
        self.name = "rsi_strategy"
        self.period = self.params.get("period", 14)
        self.oversold = self.params.get("oversold", 30)
        self.overbought = self.params.get("overbought", 70)
        
        logger.info(f"RSI Strategy initialized with period={self.period}, oversold={self.oversold}, overbought={self.overbought}")
    
    def calculate_rsi(self, prices):
        """
        Calculate the Relative Strength Index.
        
        Args:
            prices (list): List of prices
            
        Returns:
            float: The RSI value
        """
        if len(prices) < self.period + 1:
            return None
        
        # Calculate price changes
        deltas = np.diff(prices)
        
        # Calculate gains and losses
        gains = np.clip(deltas, 0, None)
        losses = -np.clip(deltas, None, 0)
        
        # Calculate average gains and losses
        avg_gain = np.mean(gains[:self.period])
        avg_loss = np.mean(losses[:self.period])
        
        # Calculate RS and RSI
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signal(self, symbol, data):
        """
        Generate a trading signal based on RSI values.
        
        Args:
            symbol (str): Trading symbol
            data (dict): Market data for the symbol, including historical prices
            
        Returns:
            dict: Trading signal
        """
        if "historical_prices" not in data or len(data["historical_prices"]) < self.period + 1:
            logger.warning(f"Not enough historical data for {symbol} to generate RSI signal")
            return super().generate_signal(symbol, data)
        
        prices = data["historical_prices"]
        rsi = self.calculate_rsi(prices)
        
        if rsi is None:
            return super().generate_signal(symbol, data)
        
        action = "hold"
        confidence = 0.0
        
        # Oversold condition (buy signal)
        if rsi < self.oversold:
            action = "buy"
            # Calculate confidence based on how oversold the asset is
            confidence = min(1.0, (self.oversold - rsi) / self.oversold)
        
        # Overbought condition (sell signal)
        elif rsi > self.overbought:
            action = "sell"
            # Calculate confidence based on how overbought the asset is
            confidence = min(1.0, (rsi - self.overbought) / (100 - self.overbought))
        
        logger.info(f"RSI Signal for {symbol}: {action.upper()} with confidence {confidence:.2f} (RSI: {rsi:.2f})")
        
        return {
            "symbol": symbol,
            "action": action,
            "confidence": confidence,
            "timestamp": data.get("timestamp", 0),
            "metrics": {
                "rsi": rsi
            }
        }


# Strategy factory to create strategy instances based on name
def create_strategy(strategy_name, params=None):
    """
    Create a strategy instance based on the strategy name.
    
    Args:
        strategy_name (str): Name of the strategy
        params (dict): Strategy parameters
        
    Returns:
        Strategy: An instance of the requested strategy
    """
    strategies = {
        "simple_moving_average": SimpleMovingAverageStrategy,
        "rsi_strategy": RSIStrategy
    }
    
    if strategy_name not in strategies:
        logger.warning(f"Strategy '{strategy_name}' not found. Using base strategy.")
        return Strategy(params)
    
    return strategies[strategy_name](params)