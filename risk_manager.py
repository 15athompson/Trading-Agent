"""
Risk Management Module

This module handles risk management for the trading bot, including position sizing,
stop-loss, and take-profit calculations.
"""

import logging

logger = logging.getLogger("TradingBot.RiskManager")

class RiskManager:
    """
    Risk manager class that handles position sizing and risk control.
    """
    
    def __init__(self, config):
        """
        Initialize the risk manager with configuration.
        
        Args:
            config (dict): Risk management configuration
        """
        self.config = config
        self.max_position_size = config.get("max_position_size", 0.1)  # Default to 10% of portfolio
        self.stop_loss = config.get("stop_loss", 0.05)  # Default to 5% stop loss
        self.take_profit = config.get("take_profit", 0.1)  # Default to 10% take profit
        
        logger.info(f"Risk manager initialized with max_position_size={self.max_position_size}, "
                   f"stop_loss={self.stop_loss}, take_profit={self.take_profit}")
    
    def calculate_position_size(self, portfolio_value, current_price, signal_confidence):
        """
        Calculate the position size for a trade based on portfolio value and signal confidence.
        
        Args:
            portfolio_value (float): Total portfolio value
            current_price (float): Current price of the asset
            signal_confidence (float): Confidence level of the trading signal (0.0 to 1.0)
            
        Returns:
            float: Position size in base currency
        """
        # Calculate the maximum amount to risk based on portfolio value
        max_amount = portfolio_value * self.max_position_size
        
        # Adjust the position size based on signal confidence
        adjusted_amount = max_amount * signal_confidence
        
        # Calculate the position size in units of the asset
        position_size = adjusted_amount / current_price
        
        logger.info(f"Calculated position size: {position_size:.6f} units "
                   f"(value: {adjusted_amount:.2f}, confidence: {signal_confidence:.2f})")
        
        return position_size
    
    def calculate_stop_loss_price(self, entry_price, position_type):
        """
        Calculate the stop-loss price for a position.
        
        Args:
            entry_price (float): Entry price of the position
            position_type (str): Type of position ('long' or 'short')
            
        Returns:
            float: Stop-loss price
        """
        if position_type.lower() == 'long':
            stop_loss_price = entry_price * (1 - self.stop_loss)
        else:  # short position
            stop_loss_price = entry_price * (1 + self.stop_loss)
        
        logger.info(f"Calculated stop-loss price: {stop_loss_price:.2f} for {position_type} position "
                   f"(entry: {entry_price:.2f}, stop-loss: {self.stop_loss:.2%})")
        
        return stop_loss_price
    
    def calculate_take_profit_price(self, entry_price, position_type):
        """
        Calculate the take-profit price for a position.
        
        Args:
            entry_price (float): Entry price of the position
            position_type (str): Type of position ('long' or 'short')
            
        Returns:
            float: Take-profit price
        """
        if position_type.lower() == 'long':
            take_profit_price = entry_price * (1 + self.take_profit)
        else:  # short position
            take_profit_price = entry_price * (1 - self.take_profit)
        
        logger.info(f"Calculated take-profit price: {take_profit_price:.2f} for {position_type} position "
                   f"(entry: {entry_price:.2f}, take-profit: {self.take_profit:.2%})")
        
        return take_profit_price
    
    def should_close_position(self, position, current_price):
        """
        Determine if a position should be closed based on stop-loss and take-profit levels.
        
        Args:
            position (dict): Position information including entry_price, type, stop_loss, and take_profit
            current_price (float): Current price of the asset
            
        Returns:
            tuple: (bool, str) - Whether to close the position and the reason
        """
        if not position:
            return False, "No position"
        
        entry_price = position.get("entry_price", 0)
        position_type = position.get("type", "long")
        stop_loss_price = position.get("stop_loss", self.calculate_stop_loss_price(entry_price, position_type))
        take_profit_price = position.get("take_profit", self.calculate_take_profit_price(entry_price, position_type))
        
        if position_type.lower() == 'long':
            # Check stop-loss for long position
            if current_price <= stop_loss_price:
                logger.info(f"Stop-loss triggered for long position at {current_price:.2f} "
                           f"(stop-loss: {stop_loss_price:.2f})")
                return True, "stop_loss"
            
            # Check take-profit for long position
            if current_price >= take_profit_price:
                logger.info(f"Take-profit triggered for long position at {current_price:.2f} "
                           f"(take-profit: {take_profit_price:.2f})")
                return True, "take_profit"
        else:  # short position
            # Check stop-loss for short position
            if current_price >= stop_loss_price:
                logger.info(f"Stop-loss triggered for short position at {current_price:.2f} "
                           f"(stop-loss: {stop_loss_price:.2f})")
                return True, "stop_loss"
            
            # Check take-profit for short position
            if current_price <= take_profit_price:
                logger.info(f"Take-profit triggered for short position at {current_price:.2f} "
                           f"(take-profit: {take_profit_price:.2f})")
                return True, "take_profit"
        
        return False, "No trigger"
    
    def update_trailing_stop(self, position, current_price):
        """
        Update the trailing stop-loss for a position.
        
        Args:
            position (dict): Position information
            current_price (float): Current price of the asset
            
        Returns:
            dict: Updated position with new stop-loss
        """
        if not position:
            return position
        
        entry_price = position.get("entry_price", 0)
        position_type = position.get("type", "long")
        current_stop_loss = position.get("stop_loss", self.calculate_stop_loss_price(entry_price, position_type))
        
        # Update trailing stop for long position
        if position_type.lower() == 'long':
            # Calculate new potential stop-loss based on current price
            new_stop_loss = current_price * (1 - self.stop_loss)
            
            # Only update if the new stop-loss is higher than the current one
            if new_stop_loss > current_stop_loss:
                position["stop_loss"] = new_stop_loss
                logger.info(f"Updated trailing stop-loss for long position to {new_stop_loss:.2f} "
                           f"(current price: {current_price:.2f})")
        
        # Update trailing stop for short position
        else:
            # Calculate new potential stop-loss based on current price
            new_stop_loss = current_price * (1 + self.stop_loss)
            
            # Only update if the new stop-loss is lower than the current one
            if new_stop_loss < current_stop_loss:
                position["stop_loss"] = new_stop_loss
                logger.info(f"Updated trailing stop-loss for short position to {new_stop_loss:.2f} "
                           f"(current price: {current_price:.2f})")
        
        return position