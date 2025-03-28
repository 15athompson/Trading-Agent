# Trading AI Agent Bot

A customizable trading bot that can analyze market data, make trading decisions based on configured strategies, and execute trades.

## Features

- **Modular Design**: Easily extendable with new data sources, strategies, and risk management techniques
- **Multiple Trading Strategies**: Includes Simple Moving Average (SMA), Relative Strength Index (RSI), MACD, and Bollinger Bands strategies
- **Machine Learning Strategies**: Includes ML-based strategies that can learn from market patterns
- **Backtesting**: Test strategies on historical data to evaluate performance
- **Web Dashboard**: Monitor the bot's performance through a web interface
- **Notifications**: Receive alerts for trades, positions, and errors
- **Risk Management**: Position sizing, stop-loss, and take-profit mechanisms
- **Portfolio Tracking**: Keeps track of positions, trades, and performance metrics
- **Demo Mode**: Test the bot with simulated market data before using real money
- **Configurable**: Customize the bot's behavior through simple JSON configuration files

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/trading-ai-bot.git
cd trading-ai-bot
```

2. Install dependencies:
```bash
pip install numpy pandas matplotlib plotly dash scikit-learn requests
```

## Configuration

The bot is configured through a `config.json` file. Here's an example configuration:

```json
{
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
        "backtest_mode": false,
        "use_ml": false,
        "use_notifications": true
    }
}
```

### Configuration Options

- **api**: API configuration
  - `provider`: Data provider (currently supports "demo")
  - `api_key`: API key for the provider
  - `api_secret`: API secret for the provider

- **trading**: Trading configuration
  - `symbols`: List of trading symbols to monitor
  - `interval`: Time interval for historical data (e.g., "1h", "1d")
  - `strategy`: Trading strategy to use
  - `strategy_params`: Parameters for the selected strategy
  - `risk_management`: Risk management parameters
    - `max_position_size`: Maximum position size as a fraction of portfolio value
    - `stop_loss`: Stop-loss percentage
    - `take_profit`: Take-profit percentage

- **bot_settings**: General bot settings
  - `update_interval`: Time between updates in seconds
  - `backtest_mode`: Whether to run in backtest mode
  - `use_ml`: Whether to use machine learning strategies
  - `use_notifications`: Whether to send notifications

## Notification Configuration

Notifications are configured through a `notification_config.json` file. Here's an example configuration:

```json
{
    "enabled": true,
    "channels": {
        "email": {
            "enabled": false,
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "username": "your_email@gmail.com",
            "password": "your_app_password",
            "from_email": "your_email@gmail.com",
            "to_email": "recipient@example.com",
            "use_tls": true
        },
        "telegram": {
            "enabled": false,
            "bot_token": "YOUR_BOT_TOKEN",
            "chat_id": "YOUR_CHAT_ID"
        },
        "webhook": {
            "enabled": false,
            "url": "https://example.com/webhook",
            "headers": {
                "Content-Type": "application/json"
            }
        },
        "console": {
            "enabled": true
        }
    },
    "notification_types": {
        "trade_executed": {
            "enabled": true,
            "channels": ["console"]
        },
        "position_closed": {
            "enabled": true,
            "channels": ["console"]
        },
        "stop_loss_triggered": {
            "enabled": true,
            "channels": ["console"]
        },
        "take_profit_triggered": {
            "enabled": true,
            "channels": ["console"]
        },
        "bot_started": {
            "enabled": true,
            "channels": ["console"]
        },
        "bot_stopped": {
            "enabled": true,
            "channels": ["console"]
        },
        "error": {
            "enabled": true,
            "channels": ["console"]
        }
    }
}
```

## Usage

### Running the Trading Bot

Run the bot with the default configuration:

```bash
python trading_bot.py
```

Specify custom configuration files:

```bash
python trading_bot.py --config my_config.json --notification-config my_notification_config.json
```

Enable verbose logging:

```bash
python trading_bot.py --verbose
```

### Running Backtests

Run a backtest with the default configuration:

```bash
python trading_bot.py --backtest
```

Specify a date range for backtesting:

```bash
python trading_bot.py --backtest --backtest-start 2023-01-01 --backtest-end 2023-12-31
```

### Starting the Web Dashboard

Start the web dashboard:

```bash
python trading_bot.py --dashboard
```

Specify a custom port for the dashboard:

```bash
python trading_bot.py --dashboard --dashboard-port 8080
```

### Testing Notifications

Test the notification system:

```bash
python notification_system.py --test
```

Create a default notification configuration:

```bash
python notification_system.py --create-config
```

## Available Strategies

### Simple Moving Average (SMA)

Generates buy signals when the short-term SMA crosses above the long-term SMA, and sell signals when the short-term SMA crosses below the long-term SMA.

Parameters:
- `short_window`: Window size for the short-term SMA (default: 20)
- `long_window`: Window size for the long-term SMA (default: 50)

### Relative Strength Index (RSI)

Generates buy signals when RSI is below the oversold threshold, and sell signals when RSI is above the overbought threshold.

Parameters:
- `period`: Period for RSI calculation (default: 14)
- `oversold`: Oversold threshold (default: 30)
- `overbought`: Overbought threshold (default: 70)

### Moving Average Convergence Divergence (MACD)

Generates buy signals when the MACD line crosses above the signal line, and sell signals when the MACD line crosses below the signal line.

Parameters:
- `fast_period`: Period for the fast EMA (default: 12)
- `slow_period`: Period for the slow EMA (default: 26)
- `signal_period`: Period for the signal line (default: 9)

### Bollinger Bands

Generates buy signals when the price touches the lower band, and sell signals when the price touches the upper band.

Parameters:
- `period`: Period for the middle band calculation (default: 20)
- `std_dev`: Number of standard deviations for the bands (default: 2.0)

### Machine Learning Strategy

Uses machine learning to predict price movements and generate trading signals.

Parameters:
- `model_type`: Type of machine learning model to use (default: "random_forest")
- `prediction_horizon`: Days to predict ahead (default: 5)
- `feature_window`: Days of history to use for features (default: 20)
- `confidence_threshold`: Minimum confidence for signals (default: 0.6)
- `retrain_interval`: Days between model retraining (default: 30)

### Machine Learning Ensemble Strategy

Combines multiple machine learning models to generate trading signals.

Parameters:
- `models`: List of machine learning models to use (default: ["random_forest", "gradient_boosting"])
- `prediction_horizon`: Days to predict ahead (default: 5)
- `feature_window`: Days of history to use for features (default: 20)
- `confidence_threshold`: Minimum confidence for signals (default: 0.6)
- `retrain_interval`: Days between model retraining (default: 30)

## Adding New Strategies

To add a new strategy:

1. Create a new strategy class in `strategies.py` that inherits from the `Strategy` base class
2. Implement the `generate_signal` method
3. Add the strategy to the `create_strategy` function in `strategies.py`

## Adding New Data Sources

To add a new data source:

1. Create a new data fetcher class in `data_fetcher.py` that inherits from the `DataFetcher` base class
2. Implement the `fetch_current_price` and `fetch_historical_data` methods
3. Add the data fetcher to the `create_data_fetcher` function in `data_fetcher.py`

## Adding New Notification Channels

To add a new notification channel:

1. Add the channel configuration to the `channels` section in `notification_config.json`
2. Implement a method in the `NotificationSystem` class in `notification_system.py` to send notifications through the channel
3. Add the method to the `channels` dictionary in the `__init__` method

## Disclaimer

This trading bot is for educational purposes only. Use it at your own risk. The authors are not responsible for any financial losses incurred from using this software.

## License

This project is licensed under the MIT License - see the LICENSE file for details.#   T r a d i n g - A g e n t 
 
 