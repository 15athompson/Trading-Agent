"""
Web Dashboard for Trading Bot

This module provides a web dashboard for monitoring the trading bot's performance.
"""

import os
import json
import logging
import threading
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import flask

# Import trading bot modules
from trading_bot import TradingBot
from portfolio_manager import PortfolioManager

logger = logging.getLogger("TradingBot.Dashboard")

class TradingBotDashboard:
    """
    Web dashboard for monitoring the trading bot.
    """
    
    def __init__(self, config_path="config.json", port=8050, debug=False):
        """
        Initialize the dashboard.
        
        Args:
            config_path (str): Path to the configuration file
            port (int): Port to run the dashboard on
            debug (bool): Whether to run in debug mode
        """
        self.config_path = config_path
        self.port = port
        self.debug = debug
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Dashboard loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self.config = {}
        
        # Initialize trading bot
        self.bot = None
        self.bot_thread = None
        self.bot_running = False
        
        # Initialize portfolio manager
        self.portfolio_manager = PortfolioManager()
        
        # Initialize data storage
        self.market_data = {}
        self.signals = []
        self.portfolio_values = []
        self.last_update = datetime.now()
        
        # Initialize Flask server
        self.server = flask.Flask(__name__)
        
        # Initialize Dash app
        self.app = dash.Dash(__name__, server=self.server)
        self.app.title = "Trading Bot Dashboard"
        
        # Set up the dashboard layout
        self._setup_layout()
        
        # Set up callbacks
        self._setup_callbacks()
        
        logger.info("Dashboard initialized")
    
    def _setup_layout(self):
        """
        Set up the dashboard layout.
        """
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Trading Bot Dashboard", style={"margin-bottom": "0px"}),
                html.P("Real-time monitoring and control of your trading bot", 
                       style={"margin-top": "0px", "color": "#666"})
            ], style={"text-align": "center", "margin-bottom": "20px"}),
            
            # Control panel
            html.Div([
                html.Div([
                    html.H3("Bot Control", style={"margin-top": "0px"}),
                    html.Div([
                        html.Button("Start Bot", id="start-bot-button", 
                                   className="control-button start-button"),
                        html.Button("Stop Bot", id="stop-bot-button", 
                                   className="control-button stop-button"),
                        html.Div(id="bot-status", className="status-indicator")
                    ], style={"display": "flex", "align-items": "center", "gap": "10px"}),
                    html.Div(id="control-message", style={"margin-top": "10px", "color": "#666"})
                ], className="control-panel"),
                
                html.Div([
                    html.H3("Portfolio Summary", style={"margin-top": "0px"}),
                    html.Div(id="portfolio-summary", className="summary-panel")
                ], className="control-panel"),
                
                html.Div([
                    html.H3("Market Overview", style={"margin-top": "0px"}),
                    html.Div(id="market-overview", className="summary-panel")
                ], className="control-panel")
            ], style={"display": "flex", "justify-content": "space-between", "margin-bottom": "20px"}),
            
            # Main content
            html.Div([
                # Tabs
                dcc.Tabs([
                    # Portfolio tab
                    dcc.Tab(label="Portfolio", children=[
                        html.Div([
                            html.H3("Portfolio Performance"),
                            dcc.Graph(id="portfolio-chart"),
                            html.H3("Open Positions"),
                            html.Div(id="positions-table"),
                            html.H3("Trade History"),
                            html.Div(id="trade-history-table")
                        ], className="tab-content")
                    ]),
                    
                    # Market Data tab
                    dcc.Tab(label="Market Data", children=[
                        html.Div([
                            html.Div([
                                html.H3("Symbol"),
                                dcc.Dropdown(
                                    id="symbol-dropdown",
                                    options=[],
                                    value=None
                                )
                            ], style={"width": "200px", "margin-bottom": "20px"}),
                            dcc.Graph(id="market-chart"),
                            html.H3("Trading Signals"),
                            html.Div(id="signals-table")
                        ], className="tab-content")
                    ]),
                    
                    # Settings tab
                    dcc.Tab(label="Settings", children=[
                        html.Div([
                            html.H3("Bot Configuration"),
                            html.Div([
                                html.Div([
                                    html.Label("Trading Strategy"),
                                    dcc.Dropdown(
                                        id="strategy-dropdown",
                                        options=[
                                            {"label": "Simple Moving Average", "value": "simple_moving_average"},
                                            {"label": "RSI Strategy", "value": "rsi_strategy"},
                                            {"label": "ML Strategy", "value": "ml_strategy"},
                                            {"label": "ML Ensemble", "value": "ml_ensemble"}
                                        ],
                                        value=self.config.get("trading", {}).get("strategy", "simple_moving_average")
                                    )
                                ], className="settings-item"),
                                
                                html.Div([
                                    html.Label("Update Interval (seconds)"),
                                    dcc.Input(
                                        id="update-interval-input",
                                        type="number",
                                        min=10,
                                        max=3600,
                                        value=self.config.get("bot_settings", {}).get("update_interval", 60)
                                    )
                                ], className="settings-item"),
                                
                                html.Div([
                                    html.Label("Risk Management"),
                                    html.Div([
                                        html.Div([
                                            html.Label("Max Position Size (%)"),
                                            dcc.Input(
                                                id="max-position-size-input",
                                                type="number",
                                                min=1,
                                                max=100,
                                                value=self.config.get("trading", {}).get("risk_management", {}).get("max_position_size", 10) * 100
                                            )
                                        ], className="settings-subitem"),
                                        
                                        html.Div([
                                            html.Label("Stop Loss (%)"),
                                            dcc.Input(
                                                id="stop-loss-input",
                                                type="number",
                                                min=1,
                                                max=50,
                                                value=self.config.get("trading", {}).get("risk_management", {}).get("stop_loss", 5) * 100
                                            )
                                        ], className="settings-subitem"),
                                        
                                        html.Div([
                                            html.Label("Take Profit (%)"),
                                            dcc.Input(
                                                id="take-profit-input",
                                                type="number",
                                                min=1,
                                                max=100,
                                                value=self.config.get("trading", {}).get("risk_management", {}).get("take_profit", 10) * 100
                                            )
                                        ], className="settings-subitem")
                                    ], style={"display": "flex", "gap": "20px"})
                                ], className="settings-item"),
                                
                                html.Button("Save Settings", id="save-settings-button", 
                                           className="control-button save-button"),
                                html.Div(id="settings-message", style={"margin-top": "10px", "color": "#666"})
                            ], className="settings-container")
                        ], className="tab-content")
                    ])
                ])
            ], className="main-content"),
            
            # Footer
            html.Div([
                html.P(f"Last updated: ", style={"margin": "0px"}),
                html.P(id="last-update-time", style={"margin": "0px", "font-weight": "bold"})
            ], style={"display": "flex", "gap": "5px", "justify-content": "center", "margin-top": "20px"}),
            
            # Refresh interval
            dcc.Interval(
                id="refresh-interval",
                interval=5 * 1000,  # in milliseconds
                n_intervals=0
            )
        ], className="dashboard-container")
        
        # Add CSS
        self._add_css()
    
    def _add_css(self):
        """
        Add CSS styles to the dashboard.
        """
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 0;
                        background-color: #f5f5f5;
                    }
                    
                    .dashboard-container {
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                    }
                    
                    .control-panel {
                        background-color: white;
                        border-radius: 5px;
                        padding: 15px;
                        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                        width: 30%;
                    }
                    
                    .control-button {
                        padding: 8px 15px;
                        border: none;
                        border-radius: 4px;
                        cursor: pointer;
                        font-weight: bold;
                    }
                    
                    .start-button {
                        background-color: #4CAF50;
                        color: white;
                    }
                    
                    .stop-button {
                        background-color: #f44336;
                        color: white;
                    }
                    
                    .save-button {
                        background-color: #2196F3;
                        color: white;
                    }
                    
                    .status-indicator {
                        padding: 5px 10px;
                        border-radius: 4px;
                        font-weight: bold;
                    }
                    
                    .status-running {
                        background-color: #4CAF50;
                        color: white;
                    }
                    
                    .status-stopped {
                        background-color: #f44336;
                        color: white;
                    }
                    
                    .main-content {
                        background-color: white;
                        border-radius: 5px;
                        padding: 20px;
                        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                    }
                    
                    .tab-content {
                        padding: 20px 0;
                    }
                    
                    .summary-panel {
                        margin-top: 10px;
                    }
                    
                    .settings-container {
                        display: flex;
                        flex-direction: column;
                        gap: 20px;
                    }
                    
                    .settings-item {
                        margin-bottom: 15px;
                    }
                    
                    .settings-subitem {
                        margin-top: 10px;
                    }
                    
                    table {
                        width: 100%;
                        border-collapse: collapse;
                    }
                    
                    th, td {
                        padding: 8px;
                        text-align: left;
                        border-bottom: 1px solid #ddd;
                    }
                    
                    th {
                        background-color: #f2f2f2;
                    }
                </style>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''
    
    def _setup_callbacks(self):
        """
        Set up the dashboard callbacks.
        """
        # Callback to update bot status
        @self.app.callback(
            [Output("bot-status", "children"),
             Output("bot-status", "className"),
             Output("control-message", "children")],
            [Input("start-bot-button", "n_clicks"),
             Input("stop-bot-button", "n_clicks"),
             Input("refresh-interval", "n_intervals")]
        )
        def update_bot_status(start_clicks, stop_clicks, n_intervals):
            ctx = dash.callback_context
            
            if not ctx.triggered:
                # Initial load
                if self.bot_running:
                    return "Running", "status-indicator status-running", ""
                else:
                    return "Stopped", "status-indicator status-stopped", ""
            
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            if button_id == "start-bot-button" and start_clicks:
                if not self.bot_running:
                    self._start_bot()
                    return "Running", "status-indicator status-running", "Bot started successfully"
                else:
                    return "Running", "status-indicator status-running", "Bot is already running"
            
            elif button_id == "stop-bot-button" and stop_clicks:
                if self.bot_running:
                    self._stop_bot()
                    return "Stopped", "status-indicator status-stopped", "Bot stopped successfully"
                else:
                    return "Stopped", "status-indicator status-stopped", "Bot is already stopped"
            
            # Refresh interval
            if self.bot_running:
                return "Running", "status-indicator status-running", ""
            else:
                return "Stopped", "status-indicator status-stopped", ""
        
        # Callback to update portfolio summary
        @self.app.callback(
            Output("portfolio-summary", "children"),
            [Input("refresh-interval", "n_intervals")]
        )
        def update_portfolio_summary(n_intervals):
            balance = self.portfolio_manager.balance
            positions = self.portfolio_manager.positions
            
            # Calculate portfolio value
            portfolio_value = balance
            for symbol, position in positions.items():
                if symbol in self.market_data and "current_price" in self.market_data[symbol]:
                    current_price = self.market_data[symbol]["current_price"]
                    portfolio_value += position["size"] * current_price
            
            # Calculate performance metrics
            metrics = self.portfolio_manager.get_performance_metrics()
            
            return html.Div([
                html.Div([
                    html.Div("Balance:", style={"font-weight": "bold"}),
                    html.Div(f"${balance:.2f}")
                ], style={"display": "flex", "justify-content": "space-between"}),
                
                html.Div([
                    html.Div("Portfolio Value:", style={"font-weight": "bold"}),
                    html.Div(f"${portfolio_value:.2f}")
                ], style={"display": "flex", "justify-content": "space-between"}),
                
                html.Div([
                    html.Div("Open Positions:", style={"font-weight": "bold"}),
                    html.Div(f"{len(positions)}")
                ], style={"display": "flex", "justify-content": "space-between"}),
                
                html.Div([
                    html.Div("Total Trades:", style={"font-weight": "bold"}),
                    html.Div(f"{metrics['total_trades']}")
                ], style={"display": "flex", "justify-content": "space-between"}),
                
                html.Div([
                    html.Div("Win Rate:", style={"font-weight": "bold"}),
                    html.Div(f"{metrics['win_rate']*100:.2f}%")
                ], style={"display": "flex", "justify-content": "space-between"}),
                
                html.Div([
                    html.Div("Total P&L:", style={"font-weight": "bold"}),
                    html.Div(f"${metrics['total_profit_loss']:.2f}")
                ], style={"display": "flex", "justify-content": "space-between"})
            ])
        
        # Callback to update market overview
        @self.app.callback(
            Output("market-overview", "children"),
            [Input("refresh-interval", "n_intervals")]
        )
        def update_market_overview(n_intervals):
            if not self.market_data:
                return html.Div("No market data available")
            
            market_overview = []
            
            for symbol, data in self.market_data.items():
                if "current_price" in data:
                    current_price = data["current_price"]
                    
                    # Calculate 24h change if available
                    change_24h = 0.0
                    change_24h_pct = 0.0
                    
                    if "historical_prices" in data and len(data["historical_prices"]) > 24:
                        price_24h_ago = data["historical_prices"][-24]
                        change_24h = current_price - price_24h_ago
                        change_24h_pct = (change_24h / price_24h_ago) * 100
                    
                    # Determine color based on change
                    color = "green" if change_24h >= 0 else "red"
                    
                    market_overview.append(html.Div([
                        html.Div(symbol, style={"font-weight": "bold"}),
                        html.Div([
                            html.Span(f"${current_price:.2f}", style={"margin-right": "10px"}),
                            html.Span(f"{change_24h_pct:+.2f}%", style={"color": color})
                        ], style={"display": "flex"})
                    ], style={"display": "flex", "justify-content": "space-between", "margin-bottom": "5px"}))
            
            return html.Div(market_overview)
        
        # Callback to update portfolio chart
        @self.app.callback(
            Output("portfolio-chart", "figure"),
            [Input("refresh-interval", "n_intervals")]
        )
        def update_portfolio_chart(n_intervals):
            # Create a figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add portfolio value trace
            if self.portfolio_values:
                timestamps = [entry["timestamp"] for entry in self.portfolio_values]
                values = [entry["value"] for entry in self.portfolio_values]
                
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=values,
                        name="Portfolio Value",
                        line=dict(color="#2196F3", width=2)
                    ),
                    secondary_y=False
                )
            
            # Add trade markers
            buy_timestamps = []
            buy_values = []
            sell_timestamps = []
            sell_values = []
            
            for trade in self.portfolio_manager.trade_history:
                # Convert timestamps to datetime
                open_time = datetime.fromtimestamp(trade["open_time"])
                close_time = datetime.fromtimestamp(trade["close_time"])
                
                # Find portfolio value at open and close times
                open_value = None
                close_value = None
                
                for entry in self.portfolio_values:
                    entry_time = entry["timestamp"]
                    if isinstance(entry_time, str):
                        entry_time = datetime.fromisoformat(entry_time)
                    
                    if abs((entry_time - open_time).total_seconds()) < 3600 and open_value is None:
                        open_value = entry["value"]
                    
                    if abs((entry_time - close_time).total_seconds()) < 3600 and close_value is None:
                        close_value = entry["value"]
                
                if open_value is not None:
                    buy_timestamps.append(open_time)
                    buy_values.append(open_value)
                
                if close_value is not None:
                    sell_timestamps.append(close_time)
                    sell_values.append(close_value)
            
            # Add buy markers
            if buy_timestamps:
                fig.add_trace(
                    go.Scatter(
                        x=buy_timestamps,
                        y=buy_values,
                        mode="markers",
                        name="Buy",
                        marker=dict(
                            color="green",
                            size=10,
                            symbol="triangle-up"
                        )
                    ),
                    secondary_y=False
                )
            
            # Add sell markers
            if sell_timestamps:
                fig.add_trace(
                    go.Scatter(
                        x=sell_timestamps,
                        y=sell_values,
                        mode="markers",
                        name="Sell",
                        marker=dict(
                            color="red",
                            size=10,
                            symbol="triangle-down"
                        )
                    ),
                    secondary_y=False
                )
            
            # Update layout
            fig.update_layout(
                title="Portfolio Performance",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            return fig
        
        # Callback to update positions table
        @self.app.callback(
            Output("positions-table", "children"),
            [Input("refresh-interval", "n_intervals")]
        )
        def update_positions_table(n_intervals):
            positions = self.portfolio_manager.positions
            
            if not positions:
                return html.Div("No open positions")
            
            # Create table data
            table_data = []
            
            for symbol, position in positions.items():
                entry_price = position["entry_price"]
                size = position["size"]
                position_type = position["type"]
                open_time = datetime.fromtimestamp(position["open_time"]).strftime("%Y-%m-%d %H:%M:%S")
                
                # Get current price
                current_price = entry_price
                if symbol in self.market_data and "current_price" in self.market_data[symbol]:
                    current_price = self.market_data[symbol]["current_price"]
                
                # Calculate P&L
                if position_type == "long":
                    pnl = (current_price - entry_price) * size
                    pnl_percent = (current_price / entry_price - 1) * 100
                else:  # short position
                    pnl = (entry_price - current_price) * size
                    pnl_percent = (entry_price / current_price - 1) * 100
                
                table_data.append({
                    "Symbol": symbol,
                    "Type": position_type.upper(),
                    "Size": f"{size:.6f}",
                    "Entry Price": f"${entry_price:.2f}",
                    "Current Price": f"${current_price:.2f}",
                    "P&L": f"${pnl:.2f} ({pnl_percent:.2f}%)",
                    "Open Time": open_time
                })
            
            # Create table
            table = dash_table.DataTable(
                id="positions-table-data",
                columns=[{"name": col, "id": col} for col in table_data[0].keys()],
                data=table_data,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left"},
                style_data_conditional=[
                    {
                        "if": {"filter_query": "{P&L} contains '('"},
                        "color": "green"
                    },
                    {
                        "if": {"filter_query": "{P&L} contains '-'"},
                        "color": "red"
                    }
                ]
            )
            
            return table
        
        # Callback to update trade history table
        @self.app.callback(
            Output("trade-history-table", "children"),
            [Input("refresh-interval", "n_intervals")]
        )
        def update_trade_history_table(n_intervals):
            trade_history = self.portfolio_manager.trade_history
            
            if not trade_history:
                return html.Div("No trade history")
            
            # Create table data
            table_data = []
            
            for trade in trade_history:
                symbol = trade["symbol"]
                trade_type = trade["type"]
                entry_price = trade["entry_price"]
                exit_price = trade["exit_price"]
                size = trade["size"]
                pnl = trade["pnl"]
                pnl_percent = trade["pnl_percent"]
                open_time = datetime.fromtimestamp(trade["open_time"]).strftime("%Y-%m-%d %H:%M:%S")
                close_time = datetime.fromtimestamp(trade["close_time"]).strftime("%Y-%m-%d %H:%M:%S")
                reason = trade["reason"]
                
                table_data.append({
                    "Symbol": symbol,
                    "Type": trade_type.upper(),
                    "Size": f"{size:.6f}",
                    "Entry Price": f"${entry_price:.2f}",
                    "Exit Price": f"${exit_price:.2f}",
                    "P&L": f"${pnl:.2f} ({pnl_percent:.2f}%)",
                    "Open Time": open_time,
                    "Close Time": close_time,
                    "Reason": reason.capitalize()
                })
            
            # Sort by close time (most recent first)
            table_data.sort(key=lambda x: x["Close Time"], reverse=True)
            
            # Create table
            table = dash_table.DataTable(
                id="trade-history-table-data",
                columns=[{"name": col, "id": col} for col in table_data[0].keys()],
                data=table_data,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left"},
                style_data_conditional=[
                    {
                        "if": {"filter_query": "{P&L} contains '('"},
                        "color": "green"
                    },
                    {
                        "if": {"filter_query": "{P&L} contains '-'"},
                        "color": "red"
                    }
                ],
                page_size=10
            )
            
            return table
        
        # Callback to update symbol dropdown
        @self.app.callback(
            Output("symbol-dropdown", "options"),
            [Input("refresh-interval", "n_intervals")]
        )
        def update_symbol_dropdown(n_intervals):
            symbols = list(self.market_data.keys())
            
            if not symbols:
                return []
            
            return [{"label": symbol, "value": symbol} for symbol in symbols]
        
        # Callback to update symbol dropdown value
        @self.app.callback(
            Output("symbol-dropdown", "value"),
            [Input("symbol-dropdown", "options")]
        )
        def update_symbol_dropdown_value(options):
            if not options:
                return None
            
            return options[0]["value"]
        
        # Callback to update market chart
        @self.app.callback(
            Output("market-chart", "figure"),
            [Input("symbol-dropdown", "value"),
             Input("refresh-interval", "n_intervals")]
        )
        def update_market_chart(symbol, n_intervals):
            if not symbol or symbol not in self.market_data:
                return go.Figure()
            
            data = self.market_data[symbol]
            
            if "historical_prices" not in data or not data["historical_prices"]:
                return go.Figure()
            
            # Create a figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add price trace
            timestamps = data["timestamps"]
            prices = data["historical_prices"]
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=prices,
                    name="Price",
                    line=dict(color="#2196F3", width=2)
                ),
                secondary_y=False
            )
            
            # Add volume trace if available
            if "volumes" in data and data["volumes"]:
                volumes = data["volumes"]
                
                fig.add_trace(
                    go.Bar(
                        x=timestamps,
                        y=volumes,
                        name="Volume",
                        marker=dict(color="#9E9E9E", opacity=0.3)
                    ),
                    secondary_y=True
                )
            
            # Add signals
            buy_signals = []
            sell_signals = []
            
            for signal in self.signals:
                if signal["symbol"] == symbol:
                    if signal["action"] == "buy":
                        buy_signals.append({
                            "timestamp": signal["timestamp"],
                            "price": prices[timestamps.index(signal["timestamp"])] if signal["timestamp"] in timestamps else None,
                            "confidence": signal["confidence"]
                        })
                    elif signal["action"] == "sell":
                        sell_signals.append({
                            "timestamp": signal["timestamp"],
                            "price": prices[timestamps.index(signal["timestamp"])] if signal["timestamp"] in timestamps else None,
                            "confidence": signal["confidence"]
                        })
            
            # Add buy signals
            if buy_signals:
                buy_timestamps = [signal["timestamp"] for signal in buy_signals if signal["price"] is not None]
                buy_prices = [signal["price"] for signal in buy_signals if signal["price"] is not None]
                
                fig.add_trace(
                    go.Scatter(
                        x=buy_timestamps,
                        y=buy_prices,
                        mode="markers",
                        name="Buy Signal",
                        marker=dict(
                            color="green",
                            size=10,
                            symbol="triangle-up"
                        )
                    ),
                    secondary_y=False
                )
            
            # Add sell signals
            if sell_signals:
                sell_timestamps = [signal["timestamp"] for signal in sell_signals if signal["price"] is not None]
                sell_prices = [signal["price"] for signal in sell_signals if signal["price"] is not None]
                
                fig.add_trace(
                    go.Scatter(
                        x=sell_timestamps,
                        y=sell_prices,
                        mode="markers",
                        name="Sell Signal",
                        marker=dict(
                            color="red",
                            size=10,
                            symbol="triangle-down"
                        )
                    ),
                    secondary_y=False
                )
            
            # Update layout
            fig.update_layout(
                title=f"{symbol} Price Chart",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                yaxis2_title="Volume",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            return fig
        
        # Callback to update signals table
        @self.app.callback(
            Output("signals-table", "children"),
            [Input("symbol-dropdown", "value"),
             Input("refresh-interval", "n_intervals")]
        )
        def update_signals_table(symbol, n_intervals):
            if not symbol:
                return html.Div("No symbol selected")
            
            # Filter signals for the selected symbol
            symbol_signals = [signal for signal in self.signals if signal["symbol"] == symbol]
            
            if not symbol_signals:
                return html.Div("No signals for this symbol")
            
            # Create table data
            table_data = []
            
            for signal in symbol_signals:
                timestamp = datetime.fromtimestamp(signal["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                action = signal["action"].upper()
                confidence = signal["confidence"]
                
                # Get additional metrics if available
                metrics = signal.get("metrics", {})
                metrics_str = ", ".join([f"{k}: {v}" for k, v in metrics.items()])
                
                table_data.append({
                    "Timestamp": timestamp,
                    "Action": action,
                    "Confidence": f"{confidence:.2f}",
                    "Metrics": metrics_str
                })
            
            # Sort by timestamp (most recent first)
            table_data.sort(key=lambda x: x["Timestamp"], reverse=True)
            
            # Create table
            table = dash_table.DataTable(
                id="signals-table-data",
                columns=[{"name": col, "id": col} for col in table_data[0].keys()],
                data=table_data,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left"},
                style_data_conditional=[
                    {
                        "if": {"filter_query": "{Action} contains 'BUY'"},
                        "color": "green"
                    },
                    {
                        "if": {"filter_query": "{Action} contains 'SELL'"},
                        "color": "red"
                    }
                ],
                page_size=10
            )
            
            return table
        
        # Callback to update last update time
        @self.app.callback(
            Output("last-update-time", "children"),
            [Input("refresh-interval", "n_intervals")]
        )
        def update_last_update_time(n_intervals):
            return self.last_update.strftime("%Y-%m-%d %H:%M:%S")
        
        # Callback to save settings
        @self.app.callback(
            Output("settings-message", "children"),
            [Input("save-settings-button", "n_clicks")],
            [State("strategy-dropdown", "value"),
             State("update-interval-input", "value"),
             State("max-position-size-input", "value"),
             State("stop-loss-input", "value"),
             State("take-profit-input", "value")]
        )
        def save_settings(n_clicks, strategy, update_interval, max_position_size, stop_loss, take_profit):
            if not n_clicks:
                return ""
            
            # Update configuration
            if "trading" not in self.config:
                self.config["trading"] = {}
            
            if "bot_settings" not in self.config:
                self.config["bot_settings"] = {}
            
            if "risk_management" not in self.config["trading"]:
                self.config["trading"]["risk_management"] = {}
            
            self.config["trading"]["strategy"] = strategy
            self.config["bot_settings"]["update_interval"] = update_interval
            self.config["trading"]["risk_management"]["max_position_size"] = max_position_size / 100
            self.config["trading"]["risk_management"]["stop_loss"] = stop_loss / 100
            self.config["trading"]["risk_management"]["take_profit"] = take_profit / 100
            
            # Save configuration
            try:
                with open(self.config_path, 'w') as f:
                    json.dump(self.config, f, indent=4)
                
                logger.info(f"Saved configuration to {self.config_path}")
                return "Settings saved successfully. Restart the bot to apply changes."
            except Exception as e:
                logger.error(f"Failed to save configuration: {e}")
                return f"Failed to save settings: {e}"
    
    def _start_bot(self):
        """
        Start the trading bot.
        """
        if self.bot_running:
            logger.warning("Bot is already running")
            return
        
        logger.info("Starting trading bot")
        
        # Initialize trading bot
        self.bot = TradingBot(self.config_path)
        
        # Start bot in a separate thread
        self.bot_thread = threading.Thread(target=self._run_bot)
        self.bot_thread.daemon = True
        self.bot_thread.start()
        
        self.bot_running = True
    
    def _stop_bot(self):
        """
        Stop the trading bot.
        """
        if not self.bot_running:
            logger.warning("Bot is already stopped")
            return
        
        logger.info("Stopping trading bot")
        
        # Stop the bot
        if self.bot:
            self.bot.stop()
        
        # Wait for the thread to finish
        if self.bot_thread:
            self.bot_thread.join(timeout=5)
        
        self.bot_running = False
    
    def _run_bot(self):
        """
        Run the trading bot and update dashboard data.
        """
        try:
            # Initialize data
            self.bot.fetch_market_data()
            
            # Main loop
            while self.bot_running:
                # Update market data
                self.bot.fetch_market_data()
                self.market_data = self.bot.market_data
                
                # Generate signals
                signals = self.bot.analyze_data()
                for signal in signals:
                    self.signals.append(signal)
                
                # Execute trades
                self.bot.execute_trades(signals)
                
                # Update portfolio value
                portfolio_value = self.bot.portfolio_manager.get_portfolio_value(self.bot.current_prices)
                self.portfolio_values.append({
                    "timestamp": datetime.now(),
                    "value": portfolio_value
                })
                
                # Update last update time
                self.last_update = datetime.now()
                
                # Sleep for the configured interval
                sleep_time = self.bot.config["bot_settings"]["update_interval"]
                time.sleep(sleep_time)
        
        except Exception as e:
            logger.error(f"Error in bot thread: {e}", exc_info=True)
            self.bot_running = False
    
    def run(self):
        """
        Run the dashboard.
        """
        logger.info(f"Starting dashboard on port {self.port}")
        self.app.run_server(debug=self.debug, port=self.port)


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
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to configuration file")
    parser.add_argument("--port", type=int, default=8050,
                        help="Port to run the dashboard on")
    parser.add_argument("--debug", action="store_true",
                        help="Run in debug mode")
    
    args = parser.parse_args()
    
    # Create and run dashboard
    dashboard = TradingBotDashboard(
        config_path=args.config,
        port=args.port,
        debug=args.debug
    )
    
    dashboard.run()