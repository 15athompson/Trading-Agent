# Trading AI Agent Bot

A customizable trading bot that can analyze market data, make trading decisions based on configured strategies, and execute trades.

## Features

- **Modular Design**: Easily extendable with new data sources, strategies, and risk management techniques
- **Multiple Trading Strategies**: Includes Simple Moving Average (SMA) and Relative Strength Index (RSI) strategies
- **Risk Management**: Position sizing, stop-loss, and take-profit mechanisms
- **Portfolio Tracking**: Keeps track of positions, trades, and performance metrics
- **Demo Mode**: Test the bot with simulated market data before using real money
- **Configurable**: Customize the bot's behavior through a simple JSON configuration file

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/trading-ai-bot.git
cd trading-ai-bot
```

2. Install dependencies:
```bash
pip install numpy
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
        "backtest_mode": false
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

## Usage

Run the bot with the default configuration:

```bash
python trading_bot.py
```

Specify a custom configuration file:

```bash
python trading_bot.py --config my_config.json
```

Enable verbose logging:

```bash
python trading_bot.py --verbose
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

## Disclaimer

This trading bot is for educational purposes only. Use it at your own risk. The authors are not responsible for any financial losses incurred from using this software.

## License

This project is licensed under the MIT License - see the LICENSE file for details.