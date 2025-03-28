#!/usr/bin/env python3
"""
Web Dashboard for Trading Bot

This module provides a web-based dashboard for monitoring and controlling the trading bot.
"""

import logging
import json
import os
import time
import threading
from datetime import datetime
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
import urllib.parse

logger = logging.getLogger("TradingBot.Dashboard")

# HTML template for the dashboard
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Bot Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; margin-bottom: 20px; }}
        .card {{ background-color: white; border-radius: 5px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); padding: 20px; margin-bottom: 20px; }}
        .card-title {{ font-size: 18px; font-weight: bold; margin-bottom: 15px; color: #2c3e50; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); grid-gap: 20px; }}
        .status {{ display: inline-block; padding: 5px 10px; border-radius: 3px; font-size: 14px; font-weight: bold; }}
        .status-running {{ background-color: #2ecc71; color: white; }}
        .status-stopped {{ background-color: #e74c3c; color: white; }}
        .btn {{ display: inline-block; padding: 8px 16px; background-color: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; text-decoration: none; }}
        .btn:hover {{ background-color: #2980b9; }}
        .btn-danger {{ background-color: #e74c3c; }}
        .btn-danger:hover {{ background-color: #c0392b; }}
        .btn-success {{ background-color: #2ecc71; }}
        .btn-success:hover {{ background-color: #27ae60; }}
        table {{ width: 100%; border-collapse: collapse; }}
        table, th, td {{ border: 1px solid #ddd; }}
        th, td {{ padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .positive {{ color: #2ecc71; }}
        .negative {{ color: #e74c3c; }}
        .refresh-btn {{ float: right; margin-bottom: 10px; }}
        .footer {{ text-align: center; margin-top: 30px; padding: 20px; color: #7f8c8d; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Trading Bot Dashboard</h1>
        <p>Monitor and control your trading bot</p>
    </div>

    <div class="container">
        <div class="card">
            <div class="card-title">Bot Status</div>
            <div>
                Status: <span class="status {status_class}">{status}</span>
                <div style="margin-top: 15px;">
                    <a href="/start" class="btn btn-success">Start Bot</a>
                    <a href="/stop" class="btn btn-danger">Stop Bot</a>
                    <a href="/refresh" class="btn">Refresh</a>
                </div>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <div class="card-title">Portfolio Summary</div>
                <table>
                    <tr>
                        <td>Balance:</td>
                        <td>${portfolio_balance}</td>
                    </tr>
                    <tr>
                        <td>Portfolio Value:</td>
                        <td>${portfolio_value}</td>
                    </tr>
                    <tr>
                        <td>Open Positions:</td>
                        <td>{open_positions}</td>
                    </tr>
                    <tr>
                        <td>Total P&L:</td>
                        <td class="{pnl_class}">${total_pnl} ({total_pnl_percent}%)</td>
                    </tr>
                </table>
            </div>

            <div class="card">
                <div class="card-title">Performance Metrics</div>
                <table>
                    <tr>
                        <td>Total Trades:</td>
                        <td>{total_trades}</td>
                    </tr>
                    <tr>
                        <td>Win Rate:</td>
                        <td>{win_rate}%</td>
                    </tr>
                    <tr>
                        <td>Average P&L:</td>
                        <td class="{avg_pnl_class}">${avg_pnl}</td>
                    </tr>
                    <tr>
                        <td>Last Updated:</td>
                        <td>{last_updated}</td>
                    </tr>
                </table>
            </div>
        </div>

        <div class="card">
            <div class="card-title">Market Data</div>
            <a href="/refresh" class="btn refresh-btn">Refresh</a>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Price</th>
                        <th>24h Change</th>
                        <th>Signal</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
                    {market_data_rows}
                </tbody>
            </table>
        </div>

        <div class="card">
            <div class="card-title">Open Positions</div>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Type</th>
                        <th>Entry Price</th>
                        <th>Current Price</th>
                        <th>Size</th>
                        <th>P&L</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {positions_rows}
                </tbody>
            </table>
        </div>

        <div class="card">
            <div class="card-title">Recent Trades</div>
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Symbol</th>
                        <th>Type</th>
                        <th>Price</th>
                        <th>Size</th>
                        <th>P&L</th>
                    </tr>
                </thead>
                <tbody>
                    {trades_rows}
                </tbody>
            </table>
        </div>

        <div class="card">
            <div class="card-title">Bot Configuration</div>
            <pre>{config_json}</pre>
            <a href="/config" class="btn">Edit Configuration</a>
        </div>
    </div>

    <div class="footer">
        <p>Trading Bot Dashboard v1.0 | &copy; 2025</p>
    </div>

    <script>
        // Auto-refresh the page every 60 seconds
        setTimeout(function() {{
            window.location.href = '/refresh';
        }}, 60000);
    </script>
</body>
</html>"""

class TradingBotDashboard:
    """
    Web dashboard for the trading bot.
    """
    
    def __init__(self, config_path="config.json", port=8050, debug=False):
        """
        Initialize the dashboard.
        
        Args:
            config_path (str): Path to the configuration file
            port (int): Port to run the dashboard on
            debug (bool): Whether to enable debug mode
        """
        self.config_path = config_path
        self.port = port
        self.debug = debug
        self.bot_running = False
        self.bot_thread = None
        self.bot = None
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize data
        self.portfolio_data = {
            "balance": 10000.00,
            "value": 10000.00,
            "open_positions": 0,
            "total_pnl": 0.00,
            "total_pnl_percent": 0.00
        }
        
        self.performance_metrics = {
            "total_trades": 0,
            "win_rate": 0.00,
            "avg_pnl": 0.00,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.market_data = {
            "BTC/USD": {
                "price": 50000.00,
                "change_24h": 2.5,
                "signal": "HOLD",
                "confidence": 0.00
            },
            "ETH/USD": {
                "price": 3000.00,
                "change_24h": 1.8,
                "signal": "HOLD",
                "confidence": 0.00
            }
        }
        
        self.positions = {}
        
        self.trades = [
            {
                "date": "2025-03-28 10:15:22",
                "symbol": "BTC/USD",
                "type": "BUY",
                "price": 49500.00,
                "size": 0.1,
                "pnl": 0.00
            },
            {
                "date": "2025-03-28 12:30:45",
                "symbol": "BTC/USD",
                "type": "SELL",
                "price": 50200.00,
                "size": 0.1,
                "pnl": 70.00
            }
        ]
        
        logger.info(f"Dashboard initialized on port {port}")
    
    def _load_config(self):
        """
        Load configuration from a JSON file.
        
        Returns:
            dict: Configuration parameters
        """
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Configuration file {self.config_path} not found. Using default configuration.")
            return {
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
    
    def _save_config(self, config):
        """
        Save configuration to a JSON file.
        
        Args:
            config (dict): Configuration parameters
        """
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)
            logger.info(f"Configuration saved to {self.config_path}")
            self.config = config
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def _start_bot(self):
        """
        Start the trading bot.
        """
        if self.bot_running:
            logger.warning("Trading bot is already running")
            return
        
        try:
            # Import trading bot
            from trading_bot import TradingBot
            
            # Create and start the bot
            self.bot = TradingBot(self.config_path)
            
            # Start the bot in a separate thread
            self.bot_thread = threading.Thread(target=self._run_bot)
            self.bot_thread.daemon = True
            self.bot_thread.start()
            
            self.bot_running = True
            logger.info("Trading bot started")
        
        except Exception as e:
            logger.error(f"Failed to start trading bot: {e}")
    
    def _run_bot(self):
        """
        Run the trading bot.
        """
        try:
            # Run the bot
            self.bot.run()
        except Exception as e:
            logger.error(f"Error in trading bot: {e}")
        finally:
            self.bot_running = False
    
    def _stop_bot(self):
        """
        Stop the trading bot.
        """
        if not self.bot_running:
            logger.warning("Trading bot is not running")
            return
        
        try:
            # Stop the bot
            if self.bot:
                self.bot.stop()
            
            # Wait for the thread to finish
            if self.bot_thread:
                self.bot_thread.join(timeout=5)
            
            self.bot_running = False
            logger.info("Trading bot stopped")
        
        except Exception as e:
            logger.error(f"Failed to stop trading bot: {e}")
    
    def _update_data(self):
        """
        Update dashboard data from the trading bot.
        """
        if not self.bot_running or not self.bot:
            # Generate demo data
            self._update_demo_data()
            return
        
        try:
            # Update portfolio data
            self.portfolio_data["balance"] = self.bot.portfolio_manager.balance
            self.portfolio_data["value"] = self.bot.portfolio_manager.get_portfolio_value(self.bot.current_prices)
            self.portfolio_data["open_positions"] = len(self.bot.portfolio_manager.positions)
            
            # Calculate total P&L
            initial_balance = 10000.00  # Assuming initial balance
            self.portfolio_data["total_pnl"] = self.portfolio_data["value"] - initial_balance
            self.portfolio_data["total_pnl_percent"] = (self.portfolio_data["value"] / initial_balance - 1) * 100
            
            # Update performance metrics
            metrics = self.bot.portfolio_manager.get_performance_metrics()
            self.performance_metrics["total_trades"] = metrics["total_trades"]
            self.performance_metrics["win_rate"] = metrics["win_rate"] * 100
            self.performance_metrics["avg_pnl"] = metrics["average_profit_loss"]
            self.performance_metrics["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Update market data
            for symbol, price in self.bot.current_prices.items():
                if symbol not in self.market_data:
                    self.market_data[symbol] = {
                        "price": price,
                        "change_24h": 0.0,
                        "signal": "HOLD",
                        "confidence": 0.00
                    }
                else:
                    # Calculate 24h change (demo)
                    old_price = self.market_data[symbol]["price"]
                    change = (price / old_price - 1) * 100 if old_price > 0 else 0
                    
                    self.market_data[symbol]["price"] = price
                    self.market_data[symbol]["change_24h"] = change
            
            # Update positions
            self.positions = self.bot.portfolio_manager.positions
            
            # Update trades
            trades = self.bot.portfolio_manager.trade_history
            if trades:
                self.trades = trades
        
        except Exception as e:
            logger.error(f"Failed to update dashboard data: {e}")
    
    def _update_demo_data(self):
        """
        Update dashboard with demo data.
        """
        # Update market data with random changes
        import random
        
        for symbol in self.market_data:
            # Random price change (-2% to +2%)
            change_pct = (random.random() * 4 - 2) / 100
            old_price = self.market_data[symbol]["price"]
            new_price = old_price * (1 + change_pct)
            
            self.market_data[symbol]["price"] = new_price
            self.market_data[symbol]["change_24h"] = change_pct * 100
            
            # Random signal
            signals = ["BUY", "SELL", "HOLD"]
            weights = [0.2, 0.2, 0.6]
            signal = random.choices(signals, weights=weights)[0]
            
            self.market_data[symbol]["signal"] = signal
            self.market_data[symbol]["confidence"] = random.random() * 0.5 if signal != "HOLD" else 0.0
        
        # Update last updated time
        self.performance_metrics["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _generate_html(self):
        """
        Generate HTML for the dashboard.
        
        Returns:
            str: HTML content
        """
        # Update data
        self._update_data()
        
        # Generate market data rows
        market_data_rows = ""
        for symbol, data in self.market_data.items():
            change_class = "positive" if data["change_24h"] >= 0 else "negative"
            change_sign = "+" if data["change_24h"] >= 0 else ""
            
            market_data_rows += f"""
            <tr>
                <td>{symbol}</td>
                <td>${data["price"]:.2f}</td>
                <td class="{change_class}">{change_sign}{data["change_24h"]:.2f}%</td>
                <td>{data["signal"]}</td>
                <td>{data["confidence"]:.2f}</td>
            </tr>
            """
        
        # Generate positions rows
        positions_rows = ""
        for symbol, position in self.positions.items():
            entry_price = position["entry_price"]
            current_price = self.market_data.get(symbol, {}).get("price", entry_price)
            size = position["size"]
            position_type = position["type"]
            
            if position_type == "long":
                pnl = (current_price - entry_price) * size
                pnl_percent = (current_price / entry_price - 1) * 100
            else:  # short position
                pnl = (entry_price - current_price) * size
                pnl_percent = (entry_price / current_price - 1) * 100
            
            pnl_class = "positive" if pnl >= 0 else "negative"
            pnl_sign = "+" if pnl >= 0 else ""
            
            positions_rows += f"""
            <tr>
                <td>{symbol}</td>
                <td>{position_type.upper()}</td>
                <td>${entry_price:.2f}</td>
                <td>${current_price:.2f}</td>
                <td>{size:.6f}</td>
                <td class="{pnl_class}">{pnl_sign}${pnl:.2f} ({pnl_sign}{pnl_percent:.2f}%)</td>
                <td><a href="/close_position?symbol={symbol}" class="btn btn-danger">Close</a></td>
            </tr>
            """
        
        if not self.positions:
            positions_rows = """
            <tr>
                <td colspan="7" style="text-align: center;">No open positions</td>
            </tr>
            """
        
        # Generate trades rows
        trades_rows = ""
        for trade in self.trades[:10]:  # Show only the 10 most recent trades
            pnl_class = "positive" if trade["pnl"] > 0 else "negative" if trade["pnl"] < 0 else ""
            pnl_sign = "+" if trade["pnl"] > 0 else ""
            
            trades_rows += f"""
            <tr>
                <td>{trade["date"]}</td>
                <td>{trade["symbol"]}</td>
                <td>{trade["type"]}</td>
                <td>${trade["price"]:.2f}</td>
                <td>{trade["size"]:.6f}</td>
                <td class="{pnl_class}">{pnl_sign}${trade["pnl"]:.2f}</td>
            </tr>
            """
        
        if not self.trades:
            trades_rows = """
            <tr>
                <td colspan="6" style="text-align: center;">No trades yet</td>
            </tr>
            """
        
        # Format portfolio data
        pnl_class = "positive" if self.portfolio_data["total_pnl"] >= 0 else "negative"
        pnl_sign = "+" if self.portfolio_data["total_pnl"] >= 0 else ""
        total_pnl = f"{pnl_sign}{self.portfolio_data['total_pnl']:.2f}"
        total_pnl_percent = f"{pnl_sign}{self.portfolio_data['total_pnl_percent']:.2f}"
        
        # Format performance metrics
        avg_pnl_class = "positive" if self.performance_metrics["avg_pnl"] >= 0 else "negative"
        avg_pnl_sign = "+" if self.performance_metrics["avg_pnl"] >= 0 else ""
        avg_pnl = f"{avg_pnl_sign}{self.performance_metrics['avg_pnl']:.2f}"
        
        # Format config JSON
        config_json = json.dumps(self.config, indent=4)
        
        # Generate HTML
        html = HTML_TEMPLATE.format(
            status="Running" if self.bot_running else "Stopped",
            status_class="status-running" if self.bot_running else "status-stopped",
            portfolio_balance=f"{self.portfolio_data['balance']:.2f}",
            portfolio_value=f"{self.portfolio_data['value']:.2f}",
            open_positions=self.portfolio_data["open_positions"],
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            pnl_class=pnl_class,
            total_trades=self.performance_metrics["total_trades"],
            win_rate=f"{self.performance_metrics['win_rate']:.2f}",
            avg_pnl=avg_pnl,
            avg_pnl_class=avg_pnl_class,
            last_updated=self.performance_metrics["last_updated"],
            market_data_rows=market_data_rows,
            positions_rows=positions_rows,
            trades_rows=trades_rows,
            config_json=config_json
        )
        
        return html
    
    def run(self):
        """
        Run the dashboard server.
        """
        # Define request handler
        dashboard = self
        
        class DashboardHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                """Handle GET requests."""
                parsed_path = urllib.parse.urlparse(self.path)
                path = parsed_path.path
                
                if path == "/":
                    # Main dashboard
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(dashboard._generate_html().encode())
                
                elif path == "/start":
                    # Start the bot
                    dashboard._start_bot()
                    self.send_response(302)
                    self.send_header("Location", "/")
                    self.end_headers()
                
                elif path == "/stop":
                    # Stop the bot
                    dashboard._stop_bot()
                    self.send_response(302)
                    self.send_header("Location", "/")
                    self.end_headers()
                
                elif path == "/refresh":
                    # Refresh the dashboard
                    self.send_response(302)
                    self.send_header("Location", "/")
                    self.end_headers()
                
                elif path == "/close_position":
                    # Close a position
                    query = urllib.parse.parse_qs(parsed_path.query)
                    symbol = query.get("symbol", [""])[0]
                    
                    if symbol and dashboard.bot_running and dashboard.bot:
                        try:
                            # Get current price
                            current_price = dashboard.bot.current_prices.get(symbol)
                            
                            if current_price and dashboard.bot.portfolio_manager.has_position(symbol):
                                # Close the position
                                success, pnl = dashboard.bot.portfolio_manager.close_position(symbol, current_price, "manual")
                                
                                if success:
                                    logger.info(f"Closed position for {symbol} with P&L {pnl:.2f}")
                        
                        except Exception as e:
                            logger.error(f"Failed to close position: {e}")
                    
                    self.send_response(302)
                    self.send_header("Location", "/")
                    self.end_headers()
                
                elif path == "/config":
                    # Show config editor (simplified)
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    
                    html = f"""
                    <!DOCTYPE html>
                    <html lang="en">
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>Trading Bot Configuration</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 20px; }}
                            textarea {{ width: 100%; height: 400px; font-family: monospace; }}
                            .btn {{ display: inline-block; padding: 8px 16px; background-color: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; text-decoration: none; }}
                        </style>
                    </head>
                    <body>
                        <h1>Trading Bot Configuration</h1>
                        <p>Edit the configuration below:</p>
                        <form action="/save_config" method="post">
                            <textarea name="config">{json.dumps(dashboard.config, indent=4)}</textarea>
                            <p>
                                <button type="submit" class="btn">Save Configuration</button>
                                <a href="/" class="btn">Cancel</a>
                            </p>
                        </form>
                    </body>
                    </html>
                    """
                    
                    self.wfile.write(html.encode())
                
                else:
                    # 404 Not Found
                    self.send_response(404)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(b"404 Not Found")
            
            def do_POST(self):
                """Handle POST requests."""
                if self.path == "/save_config":
                    # Get content length
                    content_length = int(self.headers["Content-Length"])
                    
                    # Read and parse form data
                    post_data = self.rfile.read(content_length).decode()
                    form_data = urllib.parse.parse_qs(post_data)
                    
                    # Get config JSON
                    config_json = form_data.get("config", [""])[0]
                    
                    try:
                        # Parse and save config
                        config = json.loads(config_json)
                        dashboard._save_config(config)
                        
                        # Redirect to dashboard
                        self.send_response(302)
                        self.send_header("Location", "/")
                        self.end_headers()
                    
                    except json.JSONDecodeError:
                        # Invalid JSON
                        self.send_response(400)
                        self.send_header("Content-type", "text/html")
                        self.end_headers()
                        self.wfile.write(b"Invalid JSON configuration")
                
                else:
                    # 404 Not Found
                    self.send_response(404)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(b"404 Not Found")
        
        # Create and start the server
        server = HTTPServer(("localhost", self.port), DashboardHandler)
        
        # Open browser
        webbrowser.open(f"http://localhost:{self.port}")
        
        logger.info(f"Dashboard server started at http://localhost:{self.port}")
        
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            # Stop the bot if it's running
            if self.bot_running:
                self._stop_bot()
            
            # Close the server
            server.server_close()
            logger.info("Dashboard server stopped")


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
    parser = argparse.ArgumentParser(description="Trading Bot Dashboard")
    parser.add_argument("--config", "-c", type=str, default="config.json",
                        help="Path to configuration file")
    parser.add_argument("--port", "-p", type=int, default=8050,
                        help="Port to run the dashboard on")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create and run dashboard
    dashboard = TradingBotDashboard(
        config_path=args.config,
        port=args.port,
        debug=args.debug
    )
    
    dashboard.run()