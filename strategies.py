"""
Trading Strategies Module

This module contains various trading strategies that can be used by the trading bot.
"""

import numpy as np
import logging
import importlib
import sys

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


class MACDStrategy(Strategy):
    """
    Moving Average Convergence Divergence (MACD) strategy.

    Generates buy signals when the MACD line crosses above the signal line,
    and sell signals when the MACD line crosses below the signal line.
    """

    def __init__(self, params=None):
        """
        Initialize the MACD strategy with parameters.

        Args:
            params (dict): Strategy parameters including fast_period, slow_period, and signal_period
        """
        default_params = {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9
        }
        super().__init__(params or default_params)
        self.name = "macd_strategy"
        self.fast_period = self.params.get("fast_period", 12)
        self.slow_period = self.params.get("slow_period", 26)
        self.signal_period = self.params.get("signal_period", 9)

        logger.info(f"MACD Strategy initialized with fast_period={self.fast_period}, "
                   f"slow_period={self.slow_period}, signal_period={self.signal_period}")

    def calculate_ema(self, prices, period):
        """
        Calculate the Exponential Moving Average.

        Args:
            prices (list): List of prices
            period (int): Period for EMA calculation

        Returns:
            float: The EMA value
        """
        if len(prices) < period:
            return None

        # Calculate multiplier
        multiplier = 2 / (period + 1)

        # Calculate initial SMA
        sma = sum(prices[:period]) / period

        # Calculate EMA
        ema = sma
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema

        return ema

    def calculate_macd(self, prices):
        """
        Calculate the MACD line, signal line, and histogram.

        Args:
            prices (list): List of prices

        Returns:
            tuple: (macd_line, signal_line, histogram)
        """
        if len(prices) < self.slow_period + self.signal_period:
            return None, None, None

        # Calculate fast and slow EMAs
        fast_ema = self.calculate_ema(prices, self.fast_period)
        slow_ema = self.calculate_ema(prices, self.slow_period)

        if fast_ema is None or slow_ema is None:
            return None, None, None

        # Calculate MACD line
        macd_line = fast_ema - slow_ema

        # Calculate signal line (EMA of MACD line)
        # For simplicity, we'll use a simple average here
        signal_line = self.calculate_ema(prices[-self.signal_period:], self.signal_period)

        if signal_line is None:
            return macd_line, None, None

        # Calculate histogram
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def generate_signal(self, symbol, data):
        """
        Generate a trading signal based on MACD crossover.

        Args:
            symbol (str): Trading symbol
            data (dict): Market data for the symbol, including historical prices

        Returns:
            dict: Trading signal
        """
        if "historical_prices" not in data or len(data["historical_prices"]) < self.slow_period + self.signal_period:
            logger.warning(f"Not enough historical data for {symbol} to generate MACD signal")
            return super().generate_signal(symbol, data)

        prices = data["historical_prices"]

        # Calculate MACD for current period
        macd_line, signal_line, histogram = self.calculate_macd(prices)

        if macd_line is None or signal_line is None or histogram is None:
            return super().generate_signal(symbol, data)

        # Calculate MACD for previous period
        prev_prices = prices[:-1]
        prev_macd_line, prev_signal_line, prev_histogram = self.calculate_macd(prev_prices)

        if prev_macd_line is None or prev_signal_line is None or prev_histogram is None:
            return super().generate_signal(symbol, data)

        # Generate signal based on crossover
        action = "hold"
        confidence = 0.0

        # Bullish crossover (MACD line crosses above signal line)
        if prev_macd_line <= prev_signal_line and macd_line > signal_line:
            action = "buy"
            # Calculate confidence based on the strength of the crossover
            confidence = min(1.0, abs(histogram) / abs(signal_line))

        # Bearish crossover (MACD line crosses below signal line)
        elif prev_macd_line >= prev_signal_line and macd_line < signal_line:
            action = "sell"
            # Calculate confidence based on the strength of the crossover
            confidence = min(1.0, abs(histogram) / abs(signal_line))

        logger.info(f"MACD Signal for {symbol}: {action.upper()} with confidence {confidence:.2f}")

        return {
            "symbol": symbol,
            "action": action,
            "confidence": confidence,
            "timestamp": data.get("timestamp", 0),
            "metrics": {
                "macd_line": macd_line,
                "signal_line": signal_line,
                "histogram": histogram
            }
        }


class BollingerBandsStrategy(Strategy):
    """
    Bollinger Bands strategy.

    Generates buy signals when price touches the lower band,
    and sell signals when price touches the upper band.
    """

    def __init__(self, params=None):
        """
        Initialize the Bollinger Bands strategy with parameters.

        Args:
            params (dict): Strategy parameters including period and std_dev
        """
        default_params = {
            "period": 20,
            "std_dev": 2.0
        }
        super().__init__(params or default_params)
        self.name = "bollinger_bands_strategy"
        self.period = self.params.get("period", 20)
        self.std_dev = self.params.get("std_dev", 2.0)

        logger.info(f"Bollinger Bands Strategy initialized with period={self.period}, std_dev={self.std_dev}")

    def calculate_bollinger_bands(self, prices):
        """
        Calculate the Bollinger Bands.

        Args:
            prices (list): List of prices

        Returns:
            tuple: (middle_band, upper_band, lower_band)
        """
        if len(prices) < self.period:
            return None, None, None

        # Calculate middle band (SMA)
        middle_band = sum(prices[-self.period:]) / self.period

        # Calculate standard deviation
        std_dev = np.std(prices[-self.period:])

        # Calculate upper and lower bands
        upper_band = middle_band + (self.std_dev * std_dev)
        lower_band = middle_band - (self.std_dev * std_dev)

        return middle_band, upper_band, lower_band

    def generate_signal(self, symbol, data):
        """
        Generate a trading signal based on Bollinger Bands.

        Args:
            symbol (str): Trading symbol
            data (dict): Market data for the symbol, including historical prices

        Returns:
            dict: Trading signal
        """
        if "historical_prices" not in data or len(data["historical_prices"]) < self.period:
            logger.warning(f"Not enough historical data for {symbol} to generate Bollinger Bands signal")
            return super().generate_signal(symbol, data)

        prices = data["historical_prices"]
        current_price = prices[-1]

        # Calculate Bollinger Bands
        middle_band, upper_band, lower_band = self.calculate_bollinger_bands(prices)

        if middle_band is None or upper_band is None or lower_band is None:
            return super().generate_signal(symbol, data)

        # Generate signal based on price position relative to bands
        action = "hold"
        confidence = 0.0

        # Price touches or crosses below lower band (buy signal)
        if current_price <= lower_band:
            action = "buy"
            # Calculate confidence based on how far price is below lower band
            confidence = min(1.0, (lower_band - current_price) / lower_band)

        # Price touches or crosses above upper band (sell signal)
        elif current_price >= upper_band:
            action = "sell"
            # Calculate confidence based on how far price is above upper band
            confidence = min(1.0, (current_price - upper_band) / upper_band)

        logger.info(f"Bollinger Bands Signal for {symbol}: {action.upper()} with confidence {confidence:.2f}")

        return {
            "symbol": symbol,
            "action": action,
            "confidence": confidence,
            "timestamp": data.get("timestamp", 0),
            "metrics": {
                "middle_band": middle_band,
                "upper_band": upper_band,
                "lower_band": lower_band
            }
        }


# Try to import ML strategies
try:
    from ml_strategy import MLStrategy, MLEnsembleStrategy
    ML_STRATEGIES_AVAILABLE = True
    logger.info("ML strategies imported successfully")
except ImportError:
    ML_STRATEGIES_AVAILABLE = False
    logger.warning("ML strategies not available. Make sure ml_strategy.py is in the path.")


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
        "rsi_strategy": RSIStrategy,
        "macd_strategy": MACDStrategy,
        "bollinger_bands_strategy": BollingerBandsStrategy
    }

    # Add ML strategies if available
    if ML_STRATEGIES_AVAILABLE:
        strategies["ml_strategy"] = MLStrategy
        strategies["ml_ensemble"] = MLEnsembleStrategy

    if strategy_name not in strategies:
        logger.warning(f"Strategy '{strategy_name}' not found. Using simple moving average strategy.")
        return SimpleMovingAverageStrategy(params)

    return strategies[strategy_name](params)