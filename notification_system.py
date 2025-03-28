"""
Notification System for Trading Bot

This module provides functionality for sending notifications about trading bot events.
"""

import logging
import json
import os
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

logger = logging.getLogger("TradingBot.Notifications")

class NotificationSystem:
    """
    Notification system for sending alerts about trading bot events.
    """
    
    def __init__(self, config_path="notification_config.json"):
        """
        Initialize the notification system with configuration.
        
        Args:
            config_path (str): Path to the notification configuration file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Initialize notification channels
        self.channels = {
            "email": self._send_email_notification,
            "telegram": self._send_telegram_notification,
            "webhook": self._send_webhook_notification,
            "console": self._send_console_notification
        }
        
        logger.info("Notification system initialized")
    
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
            logger.info(f"Notification configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Notification configuration file {config_path} not found. Creating default configuration.")
            return self._create_default_config(config_path)
        except Exception as e:
            logger.error(f"Failed to load notification configuration: {e}")
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
            "enabled": True,
            "channels": {
                "email": {
                    "enabled": False,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "your_email@gmail.com",
                    "password": "your_app_password",
                    "from_email": "your_email@gmail.com",
                    "to_email": "recipient@example.com",
                    "use_tls": True
                },
                "telegram": {
                    "enabled": False,
                    "bot_token": "YOUR_BOT_TOKEN",
                    "chat_id": "YOUR_CHAT_ID"
                },
                "webhook": {
                    "enabled": False,
                    "url": "https://example.com/webhook",
                    "headers": {
                        "Content-Type": "application/json"
                    }
                },
                "console": {
                    "enabled": True
                }
            },
            "notification_types": {
                "trade_executed": {
                    "enabled": True,
                    "channels": ["console"]
                },
                "position_closed": {
                    "enabled": True,
                    "channels": ["console"]
                },
                "stop_loss_triggered": {
                    "enabled": True,
                    "channels": ["console"]
                },
                "take_profit_triggered": {
                    "enabled": True,
                    "channels": ["console"]
                },
                "bot_started": {
                    "enabled": True,
                    "channels": ["console"]
                },
                "bot_stopped": {
                    "enabled": True,
                    "channels": ["console"]
                },
                "error": {
                    "enabled": True,
                    "channels": ["console"]
                }
            }
        }
        
        try:
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            logger.info(f"Default notification configuration created at {config_path}")
        except Exception as e:
            logger.error(f"Failed to create default notification configuration: {e}")
        
        return default_config
    
    def _send_email_notification(self, subject, message):
        """
        Send an email notification.
        
        Args:
            subject (str): Email subject
            message (str): Email message
            
        Returns:
            bool: Whether the email was sent successfully
        """
        if not self.config["channels"]["email"]["enabled"]:
            logger.debug("Email notifications are disabled")
            return False
        
        email_config = self.config["channels"]["email"]
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = email_config["from_email"]
            msg["To"] = email_config["to_email"]
            msg["Subject"] = subject
            
            # Add message body
            msg.attach(MIMEText(message, "plain"))
            
            # Connect to SMTP server
            server = smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"])
            
            if email_config["use_tls"]:
                server.starttls()
            
            # Login
            server.login(email_config["username"], email_config["password"])
            
            # Send email
            server.send_message(msg)
            
            # Close connection
            server.quit()
            
            logger.info(f"Email notification sent to {email_config['to_email']}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False
    
    def _send_telegram_notification(self, subject, message):
        """
        Send a Telegram notification.
        
        Args:
            subject (str): Notification subject
            message (str): Notification message
            
        Returns:
            bool: Whether the notification was sent successfully
        """
        if not self.config["channels"]["telegram"]["enabled"]:
            logger.debug("Telegram notifications are disabled")
            return False
        
        telegram_config = self.config["channels"]["telegram"]
        
        try:
            # Format message
            formatted_message = f"*{subject}*\n\n{message}"
            
            # Send message
            url = f"https://api.telegram.org/bot{telegram_config['bot_token']}/sendMessage"
            data = {
                "chat_id": telegram_config["chat_id"],
                "text": formatted_message,
                "parse_mode": "Markdown"
            }
            
            response = requests.post(url, json=data)
            
            if response.status_code == 200:
                logger.info("Telegram notification sent")
                return True
            else:
                logger.error(f"Failed to send Telegram notification: {response.text}")
                return False
        
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
            return False
    
    def _send_webhook_notification(self, subject, message):
        """
        Send a webhook notification.
        
        Args:
            subject (str): Notification subject
            message (str): Notification message
            
        Returns:
            bool: Whether the notification was sent successfully
        """
        if not self.config["channels"]["webhook"]["enabled"]:
            logger.debug("Webhook notifications are disabled")
            return False
        
        webhook_config = self.config["channels"]["webhook"]
        
        try:
            # Prepare data
            data = {
                "subject": subject,
                "message": message,
                "timestamp": datetime.now().isoformat()
            }
            
            # Send webhook
            response = requests.post(
                webhook_config["url"],
                json=data,
                headers=webhook_config["headers"]
            )
            
            if response.status_code >= 200 and response.status_code < 300:
                logger.info(f"Webhook notification sent to {webhook_config['url']}")
                return True
            else:
                logger.error(f"Failed to send webhook notification: {response.text}")
                return False
        
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False
    
    def _send_console_notification(self, subject, message):
        """
        Send a console notification.
        
        Args:
            subject (str): Notification subject
            message (str): Notification message
            
        Returns:
            bool: Whether the notification was sent successfully
        """
        if not self.config["channels"]["console"]["enabled"]:
            logger.debug("Console notifications are disabled")
            return False
        
        try:
            # Print notification to console
            print(f"\n=== {subject} ===")
            print(message)
            print("=" * (len(subject) + 8))
            
            logger.info(f"Console notification: {subject}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to send console notification: {e}")
            return False
    
    def send_notification(self, notification_type, subject, message):
        """
        Send a notification of a specific type.
        
        Args:
            notification_type (str): Type of notification
            subject (str): Notification subject
            message (str): Notification message
            
        Returns:
            bool: Whether the notification was sent successfully
        """
        if not self.config["enabled"]:
            logger.debug("Notifications are disabled")
            return False
        
        if notification_type not in self.config["notification_types"]:
            logger.warning(f"Unknown notification type: {notification_type}")
            return False
        
        notification_config = self.config["notification_types"][notification_type]
        
        if not notification_config["enabled"]:
            logger.debug(f"{notification_type} notifications are disabled")
            return False
        
        # Send notifications through enabled channels
        success = False
        
        for channel in notification_config["channels"]:
            if channel in self.channels:
                channel_success = self.channels[channel](subject, message)
                success = success or channel_success
            else:
                logger.warning(f"Unknown notification channel: {channel}")
        
        return success
    
    def notify_trade_executed(self, trade_details):
        """
        Send a notification about a trade being executed.
        
        Args:
            trade_details (dict): Details of the trade
            
        Returns:
            bool: Whether the notification was sent successfully
        """
        symbol = trade_details["symbol"]
        action = trade_details["action"].upper()
        price = trade_details["price"]
        size = trade_details["size"]
        
        subject = f"Trade Executed: {action} {symbol}"
        message = f"A {action} trade for {symbol} has been executed.\n\n" \
                 f"Details:\n" \
                 f"- Symbol: {symbol}\n" \
                 f"- Action: {action}\n" \
                 f"- Price: ${price:.2f}\n" \
                 f"- Size: {size:.6f}\n" \
                 f"- Value: ${price * size:.2f}\n" \
                 f"- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return self.send_notification("trade_executed", subject, message)
    
    def notify_position_closed(self, position_details):
        """
        Send a notification about a position being closed.
        
        Args:
            position_details (dict): Details of the closed position
            
        Returns:
            bool: Whether the notification was sent successfully
        """
        symbol = position_details["symbol"]
        position_type = position_details["type"].upper()
        entry_price = position_details["entry_price"]
        exit_price = position_details["exit_price"]
        size = position_details["size"]
        pnl = position_details["pnl"]
        pnl_percent = position_details["pnl_percent"]
        reason = position_details["reason"].capitalize()
        
        subject = f"Position Closed: {symbol} ({reason})"
        message = f"A {position_type} position for {symbol} has been closed.\n\n" \
                 f"Details:\n" \
                 f"- Symbol: {symbol}\n" \
                 f"- Type: {position_type}\n" \
                 f"- Entry Price: ${entry_price:.2f}\n" \
                 f"- Exit Price: ${exit_price:.2f}\n" \
                 f"- Size: {size:.6f}\n" \
                 f"- P&L: ${pnl:.2f} ({pnl_percent:.2f}%)\n" \
                 f"- Reason: {reason}\n" \
                 f"- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return self.send_notification("position_closed", subject, message)
    
    def notify_stop_loss_triggered(self, position_details):
        """
        Send a notification about a stop-loss being triggered.
        
        Args:
            position_details (dict): Details of the position
            
        Returns:
            bool: Whether the notification was sent successfully
        """
        symbol = position_details["symbol"]
        position_type = position_details["type"].upper()
        entry_price = position_details["entry_price"]
        stop_price = position_details["stop_loss"]
        size = position_details["size"]
        
        # Calculate P&L
        if position_type == "LONG":
            pnl = (stop_price - entry_price) * size
            pnl_percent = (stop_price / entry_price - 1) * 100
        else:  # short position
            pnl = (entry_price - stop_price) * size
            pnl_percent = (entry_price / stop_price - 1) * 100
        
        subject = f"Stop-Loss Triggered: {symbol}"
        message = f"A stop-loss has been triggered for {symbol}.\n\n" \
                 f"Details:\n" \
                 f"- Symbol: {symbol}\n" \
                 f"- Type: {position_type}\n" \
                 f"- Entry Price: ${entry_price:.2f}\n" \
                 f"- Stop Price: ${stop_price:.2f}\n" \
                 f"- Size: {size:.6f}\n" \
                 f"- P&L: ${pnl:.2f} ({pnl_percent:.2f}%)\n" \
                 f"- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return self.send_notification("stop_loss_triggered", subject, message)
    
    def notify_take_profit_triggered(self, position_details):
        """
        Send a notification about a take-profit being triggered.
        
        Args:
            position_details (dict): Details of the position
            
        Returns:
            bool: Whether the notification was sent successfully
        """
        symbol = position_details["symbol"]
        position_type = position_details["type"].upper()
        entry_price = position_details["entry_price"]
        take_profit_price = position_details["take_profit"]
        size = position_details["size"]
        
        # Calculate P&L
        if position_type == "LONG":
            pnl = (take_profit_price - entry_price) * size
            pnl_percent = (take_profit_price / entry_price - 1) * 100
        else:  # short position
            pnl = (entry_price - take_profit_price) * size
            pnl_percent = (entry_price / take_profit_price - 1) * 100
        
        subject = f"Take-Profit Triggered: {symbol}"
        message = f"A take-profit has been triggered for {symbol}.\n\n" \
                 f"Details:\n" \
                 f"- Symbol: {symbol}\n" \
                 f"- Type: {position_type}\n" \
                 f"- Entry Price: ${entry_price:.2f}\n" \
                 f"- Take-Profit Price: ${take_profit_price:.2f}\n" \
                 f"- Size: {size:.6f}\n" \
                 f"- P&L: ${pnl:.2f} ({pnl_percent:.2f}%)\n" \
                 f"- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return self.send_notification("take_profit_triggered", subject, message)
    
    def notify_bot_started(self):
        """
        Send a notification about the bot being started.
        
        Returns:
            bool: Whether the notification was sent successfully
        """
        subject = "Trading Bot Started"
        message = f"The trading bot has been started.\n\n" \
                 f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return self.send_notification("bot_started", subject, message)
    
    def notify_bot_stopped(self):
        """
        Send a notification about the bot being stopped.
        
        Returns:
            bool: Whether the notification was sent successfully
        """
        subject = "Trading Bot Stopped"
        message = f"The trading bot has been stopped.\n\n" \
                 f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return self.send_notification("bot_stopped", subject, message)
    
    def notify_error(self, error_message):
        """
        Send a notification about an error.
        
        Args:
            error_message (str): Error message
            
        Returns:
            bool: Whether the notification was sent successfully
        """
        subject = "Trading Bot Error"
        message = f"An error occurred in the trading bot:\n\n" \
                 f"{error_message}\n\n" \
                 f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return self.send_notification("error", subject, message)


# Create a default notification configuration file
def create_notification_config(config_path="notification_config.json"):
    """
    Create a default notification configuration file.
    
    Args:
        config_path (str): Path to save the configuration file
        
    Returns:
        dict: Default configuration
    """
    notification_system = NotificationSystem(config_path)
    return notification_system.config


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
    parser = argparse.ArgumentParser(description="Trading Bot Notification System")
    parser.add_argument("--config", type=str, default="notification_config.json",
                        help="Path to notification configuration file")
    parser.add_argument("--create-config", action="store_true",
                        help="Create a default notification configuration file")
    parser.add_argument("--test", action="store_true",
                        help="Send test notifications")
    
    args = parser.parse_args()
    
    if args.create_config:
        config = create_notification_config(args.config)
        print(f"Created default notification configuration at {args.config}")
    
    if args.test:
        notification_system = NotificationSystem(args.config)
        
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
        
        print("Test notifications sent")