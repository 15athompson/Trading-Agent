"""
Machine Learning Strategy Module

This module implements trading strategies based on machine learning models.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
import os
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from strategies import Strategy

logger = logging.getLogger("TradingBot.MLStrategy")

class MLStrategy(Strategy):
    """
    Machine Learning strategy base class.
    """
    
    def __init__(self, params=None):
        """
        Initialize the ML strategy with parameters.
        
        Args:
            params (dict): Strategy parameters
        """
        default_params = {
            "model_type": "random_forest",
            "prediction_horizon": 5,  # Days to predict ahead
            "feature_window": 20,  # Days of history to use for features
            "confidence_threshold": 0.6,  # Minimum confidence for signals
            "retrain_interval": 30,  # Days between model retraining
            "model_path": "models"  # Directory to save trained models
        }
        
        super().__init__(params or default_params)
        self.name = "ml_strategy"
        
        self.model_type = self.params.get("model_type", "random_forest")
        self.prediction_horizon = self.params.get("prediction_horizon", 5)
        self.feature_window = self.params.get("feature_window", 20)
        self.confidence_threshold = self.params.get("confidence_threshold", 0.6)
        self.retrain_interval = self.params.get("retrain_interval", 30)
        self.model_path = self.params.get("model_path", "models")
        
        # Create model directory if it doesn't exist
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        
        # Initialize models dictionary
        self.models = {}
        self.last_trained = {}
        
        logger.info(f"ML Strategy initialized with model_type={self.model_type}, "
                   f"prediction_horizon={self.prediction_horizon}, "
                   f"feature_window={self.feature_window}")
    
    def _get_model_path(self, symbol):
        """
        Get the path to save/load the model for a symbol.
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            str: Path to the model file
        """
        # Replace / with _ in symbol name for filename
        safe_symbol = symbol.replace("/", "_")
        return os.path.join(self.model_path, f"{safe_symbol}_{self.model_type}_model.pkl")
    
    def _create_model(self):
        """
        Create a new machine learning model.
        
        Returns:
            object: Machine learning model
        """
        if self.model_type == "random_forest":
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                ))
            ])
        elif self.model_type == "gradient_boosting":
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                ))
            ])
        else:
            logger.warning(f"Unknown model type: {self.model_type}, using random forest")
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                ))
            ])
        
        return model
    
    def _load_model(self, symbol):
        """
        Load a trained model for a symbol.
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            object: Trained machine learning model or None if not found
        """
        model_path = self._get_model_path(symbol)
        
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                logger.info(f"Loaded trained model for {symbol} from {model_path}")
                return model
            except Exception as e:
                logger.error(f"Failed to load model for {symbol}: {e}")
        
        logger.info(f"No trained model found for {symbol}, creating new model")
        return self._create_model()
    
    def _save_model(self, symbol, model):
        """
        Save a trained model for a symbol.
        
        Args:
            symbol (str): Trading symbol
            model (object): Trained machine learning model
            
        Returns:
            bool: Whether the model was successfully saved
        """
        model_path = self._get_model_path(symbol)
        
        try:
            joblib.dump(model, model_path)
            logger.info(f"Saved trained model for {symbol} to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model for {symbol}: {e}")
            return False
    
    def _extract_features(self, data):
        """
        Extract features from market data for machine learning.
        
        Args:
            data (dict): Market data
            
        Returns:
            pandas.DataFrame: Features for machine learning
        """
        if "historical_prices" not in data or len(data["historical_prices"]) < self.feature_window:
            logger.warning(f"Not enough historical data to extract features")
            return None
        
        # Get price data
        prices = np.array(data["historical_prices"])
        timestamps = np.array(data["timestamps"])
        
        if "opens" in data and "highs" in data and "lows" in data and "closes" in data and "volumes" in data:
            opens = np.array(data["opens"])
            highs = np.array(data["highs"])
            lows = np.array(data["lows"])
            closes = np.array(data["closes"])
            volumes = np.array(data["volumes"])
        else:
            # If OHLCV data is not available, use prices for all
            opens = prices
            highs = prices
            lows = prices
            closes = prices
            volumes = np.ones_like(prices) * 1000  # Default volume
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        
        # Calculate technical indicators
        df = self._calculate_indicators(df)
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        return df
    
    def _calculate_indicators(self, df):
        """
        Calculate technical indicators for feature engineering.
        
        Args:
            df (pandas.DataFrame): Price data
            
        Returns:
            pandas.DataFrame: Data with technical indicators
        """
        # Calculate returns
        df['return_1d'] = df['close'].pct_change(1)
        df['return_5d'] = df['close'].pct_change(5)
        df['return_10d'] = df['close'].pct_change(10)
        
        # Calculate moving averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        
        # Calculate exponential moving averages
        df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        
        # Calculate moving average convergence divergence (MACD)
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Calculate relative strength index (RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Calculate Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Calculate Average True Range (ATR)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr_14'] = df['true_range'].rolling(window=14).mean()
        
        # Calculate price relative to moving averages
        df['price_sma_5_ratio'] = df['close'] / df['sma_5']
        df['price_sma_10_ratio'] = df['close'] / df['sma_10']
        df['price_sma_20_ratio'] = df['close'] / df['sma_20']
        
        # Calculate moving average crossovers
        df['sma_5_10_cross'] = (df['sma_5'] > df['sma_10']).astype(int)
        df['sma_10_20_cross'] = (df['sma_10'] > df['sma_20']).astype(int)
        
        # Calculate volume indicators
        df['volume_sma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_5']
        
        # Drop temporary columns
        df.drop(['tr1', 'tr2', 'tr3'], axis=1, inplace=True)
        
        return df
    
    def _prepare_training_data(self, df):
        """
        Prepare training data for machine learning.
        
        Args:
            df (pandas.DataFrame): Data with features
            
        Returns:
            tuple: (X, y) - Features and target variables
        """
        # Create target variable: 1 if price goes up by 2% or more in the next N days, 0 otherwise
        future_return = df['close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
        df['target'] = (future_return > 0.02).astype(int)
        
        # Drop rows with NaN target values
        df.dropna(inplace=True)
        
        # Select features
        feature_columns = [
            'return_1d', 'return_5d', 'return_10d',
            'sma_5', 'sma_10', 'sma_20',
            'ema_5', 'ema_10', 'ema_20',
            'macd', 'macd_signal', 'macd_hist',
            'rsi_14',
            'bb_width',
            'atr_14',
            'price_sma_5_ratio', 'price_sma_10_ratio', 'price_sma_20_ratio',
            'sma_5_10_cross', 'sma_10_20_cross',
            'volume_ratio'
        ]
        
        # Normalize features by close price to make them scale-invariant
        price_scale_features = [
            'sma_5', 'sma_10', 'sma_20',
            'ema_5', 'ema_10', 'ema_20',
            'bb_middle', 'bb_upper', 'bb_lower',
            'atr_14'
        ]
        
        for feature in price_scale_features:
            if feature in feature_columns:
                df[feature] = df[feature] / df['close']
        
        X = df[feature_columns].values
        y = df['target'].values
        
        return X, y
    
    def train_model(self, symbol, data):
        """
        Train a machine learning model for a symbol.
        
        Args:
            symbol (str): Trading symbol
            data (dict): Market data
            
        Returns:
            object: Trained machine learning model or None if training failed
        """
        logger.info(f"Training model for {symbol}")
        
        # Extract features
        df = self._extract_features(data)
        if df is None or len(df) < self.feature_window + self.prediction_horizon:
            logger.warning(f"Not enough data to train model for {symbol}")
            return None
        
        # Prepare training data
        X, y = self._prepare_training_data(df)
        if len(X) == 0 or len(y) == 0:
            logger.warning(f"Failed to prepare training data for {symbol}")
            return None
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train model
        model = self._create_model()
        
        try:
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            logger.info(f"Model evaluation for {symbol}: "
                       f"Accuracy={accuracy:.4f}, Precision={precision:.4f}, "
                       f"Recall={recall:.4f}, F1={f1:.4f}")
            
            # Save model
            self._save_model(symbol, model)
            
            # Update last trained timestamp
            self.last_trained[symbol] = datetime.now().timestamp()
            
            return model
        
        except Exception as e:
            logger.error(f"Failed to train model for {symbol}: {e}")
            return None
    
    def should_retrain(self, symbol, current_timestamp):
        """
        Determine if the model should be retrained.
        
        Args:
            symbol (str): Trading symbol
            current_timestamp (float): Current timestamp
            
        Returns:
            bool: Whether the model should be retrained
        """
        if symbol not in self.last_trained:
            return True
        
        last_trained = self.last_trained[symbol]
        days_since_trained = (current_timestamp - last_trained) / (24 * 3600)
        
        return days_since_trained >= self.retrain_interval
    
    def generate_signal(self, symbol, data):
        """
        Generate a trading signal based on machine learning predictions.
        
        Args:
            symbol (str): Trading symbol
            data (dict): Market data
            
        Returns:
            dict: Trading signal
        """
        # Check if we have enough data
        if "historical_prices" not in data or len(data["historical_prices"]) < self.feature_window:
            logger.warning(f"Not enough historical data for {symbol} to generate ML signal")
            return super().generate_signal(symbol, data)
        
        current_timestamp = data.get("timestamps", [datetime.now().timestamp()])[-1]
        
        # Load or train model
        if symbol not in self.models or self.should_retrain(symbol, current_timestamp):
            if self.should_retrain(symbol, current_timestamp):
                logger.info(f"Retraining model for {symbol}")
            
            model = self.train_model(symbol, data)
            if model is not None:
                self.models[symbol] = model
            elif symbol not in self.models:
                self.models[symbol] = self._load_model(symbol)
        
        model = self.models[symbol]
        
        # Extract features for prediction
        df = self._extract_features(data)
        if df is None or len(df) == 0:
            logger.warning(f"Failed to extract features for {symbol}")
            return super().generate_signal(symbol, data)
        
        # Select the most recent data point for prediction
        feature_columns = [
            'return_1d', 'return_5d', 'return_10d',
            'sma_5', 'sma_10', 'sma_20',
            'ema_5', 'ema_10', 'ema_20',
            'macd', 'macd_signal', 'macd_hist',
            'rsi_14',
            'bb_width',
            'atr_14',
            'price_sma_5_ratio', 'price_sma_10_ratio', 'price_sma_20_ratio',
            'sma_5_10_cross', 'sma_10_20_cross',
            'volume_ratio'
        ]
        
        # Normalize features by close price
        price_scale_features = [
            'sma_5', 'sma_10', 'sma_20',
            'ema_5', 'ema_10', 'ema_20',
            'bb_middle', 'bb_upper', 'bb_lower',
            'atr_14'
        ]
        
        for feature in price_scale_features:
            if feature in feature_columns:
                df[feature] = df[feature] / df['close']
        
        # Get the latest data point
        latest_features = df[feature_columns].iloc[-1].values.reshape(1, -1)
        
        # Make prediction
        try:
            prediction = model.predict(latest_features)[0]
            confidence = model.predict_proba(latest_features)[0][prediction]
            
            # Generate signal based on prediction
            if prediction == 1 and confidence >= self.confidence_threshold:
                action = "buy"
            elif prediction == 0 and confidence >= self.confidence_threshold:
                action = "sell"
            else:
                action = "hold"
            
            logger.info(f"ML Signal for {symbol}: {action.upper()} with confidence {confidence:.2f}")
            
            return {
                "symbol": symbol,
                "action": action,
                "confidence": confidence,
                "timestamp": current_timestamp,
                "metrics": {
                    "prediction": int(prediction),
                    "confidence": float(confidence)
                }
            }
        
        except Exception as e:
            logger.error(f"Failed to generate prediction for {symbol}: {e}")
            return super().generate_signal(symbol, data)


class MLEnsembleStrategy(MLStrategy):
    """
    Machine Learning Ensemble strategy that combines multiple models.
    """
    
    def __init__(self, params=None):
        """
        Initialize the ML Ensemble strategy with parameters.
        
        Args:
            params (dict): Strategy parameters
        """
        default_params = {
            "models": ["random_forest", "gradient_boosting"],
            "prediction_horizon": 5,
            "feature_window": 20,
            "confidence_threshold": 0.6,
            "retrain_interval": 30,
            "model_path": "models"
        }
        
        super().__init__(params or default_params)
        self.name = "ml_ensemble"
        
        self.model_types = self.params.get("models", ["random_forest", "gradient_boosting"])
        
        # Initialize models dictionary for each model type
        self.models = {model_type: {} for model_type in self.model_types}
        self.last_trained = {model_type: {} for model_type in self.model_types}
        
        logger.info(f"ML Ensemble Strategy initialized with models={self.model_types}, "
                   f"prediction_horizon={self.prediction_horizon}, "
                   f"feature_window={self.feature_window}")
    
    def _get_model_path(self, symbol, model_type):
        """
        Get the path to save/load the model for a symbol and model type.
        
        Args:
            symbol (str): Trading symbol
            model_type (str): Type of model
            
        Returns:
            str: Path to the model file
        """
        # Replace / with _ in symbol name for filename
        safe_symbol = symbol.replace("/", "_")
        return os.path.join(self.model_path, f"{safe_symbol}_{model_type}_model.pkl")
    
    def _load_model(self, symbol, model_type):
        """
        Load a trained model for a symbol and model type.
        
        Args:
            symbol (str): Trading symbol
            model_type (str): Type of model
            
        Returns:
            object: Trained machine learning model or None if not found
        """
        model_path = self._get_model_path(symbol, model_type)
        
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                logger.info(f"Loaded trained {model_type} model for {symbol} from {model_path}")
                return model
            except Exception as e:
                logger.error(f"Failed to load {model_type} model for {symbol}: {e}")
        
        logger.info(f"No trained {model_type} model found for {symbol}, creating new model")
        
        # Set the model type temporarily for creating the model
        original_model_type = self.model_type
        self.model_type = model_type
        model = self._create_model()
        self.model_type = original_model_type
        
        return model
    
    def _save_model(self, symbol, model, model_type):
        """
        Save a trained model for a symbol and model type.
        
        Args:
            symbol (str): Trading symbol
            model (object): Trained machine learning model
            model_type (str): Type of model
            
        Returns:
            bool: Whether the model was successfully saved
        """
        model_path = self._get_model_path(symbol, model_type)
        
        try:
            joblib.dump(model, model_path)
            logger.info(f"Saved trained {model_type} model for {symbol} to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save {model_type} model for {symbol}: {e}")
            return False
    
    def train_model(self, symbol, data, model_type):
        """
        Train a machine learning model for a symbol and model type.
        
        Args:
            symbol (str): Trading symbol
            data (dict): Market data
            model_type (str): Type of model
            
        Returns:
            object: Trained machine learning model or None if training failed
        """
        logger.info(f"Training {model_type} model for {symbol}")
        
        # Extract features
        df = self._extract_features(data)
        if df is None or len(df) < self.feature_window + self.prediction_horizon:
            logger.warning(f"Not enough data to train {model_type} model for {symbol}")
            return None
        
        # Prepare training data
        X, y = self._prepare_training_data(df)
        if len(X) == 0 or len(y) == 0:
            logger.warning(f"Failed to prepare training data for {model_type} model for {symbol}")
            return None
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Set the model type temporarily for creating the model
        original_model_type = self.model_type
        self.model_type = model_type
        model = self._create_model()
        self.model_type = original_model_type
        
        try:
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            logger.info(f"{model_type} model evaluation for {symbol}: "
                       f"Accuracy={accuracy:.4f}, Precision={precision:.4f}, "
                       f"Recall={recall:.4f}, F1={f1:.4f}")
            
            # Save model
            self._save_model(symbol, model, model_type)
            
            # Update last trained timestamp
            self.last_trained[model_type][symbol] = datetime.now().timestamp()
            
            return model
        
        except Exception as e:
            logger.error(f"Failed to train {model_type} model for {symbol}: {e}")
            return None
    
    def should_retrain(self, symbol, current_timestamp, model_type):
        """
        Determine if the model should be retrained.
        
        Args:
            symbol (str): Trading symbol
            current_timestamp (float): Current timestamp
            model_type (str): Type of model
            
        Returns:
            bool: Whether the model should be retrained
        """
        if symbol not in self.last_trained[model_type]:
            return True
        
        last_trained = self.last_trained[model_type][symbol]
        days_since_trained = (current_timestamp - last_trained) / (24 * 3600)
        
        return days_since_trained >= self.retrain_interval
    
    def generate_signal(self, symbol, data):
        """
        Generate a trading signal based on ensemble of machine learning predictions.
        
        Args:
            symbol (str): Trading symbol
            data (dict): Market data
            
        Returns:
            dict: Trading signal
        """
        # Check if we have enough data
        if "historical_prices" not in data or len(data["historical_prices"]) < self.feature_window:
            logger.warning(f"Not enough historical data for {symbol} to generate ML Ensemble signal")
            return super(MLStrategy, self).generate_signal(symbol, data)
        
        current_timestamp = data.get("timestamps", [datetime.now().timestamp()])[-1]
        
        # Load or train models
        predictions = []
        confidences = []
        
        for model_type in self.model_types:
            # Check if model needs to be loaded or retrained
            if (symbol not in self.models[model_type] or 
                self.should_retrain(symbol, current_timestamp, model_type)):
                
                if self.should_retrain(symbol, current_timestamp, model_type):
                    logger.info(f"Retraining {model_type} model for {symbol}")
                
                model = self.train_model(symbol, data, model_type)
                if model is not None:
                    self.models[model_type][symbol] = model
                elif symbol not in self.models[model_type]:
                    self.models[model_type][symbol] = self._load_model(symbol, model_type)
            
            model = self.models[model_type][symbol]
            
            # Extract features for prediction
            df = self._extract_features(data)
            if df is None or len(df) == 0:
                logger.warning(f"Failed to extract features for {symbol}")
                continue
            
            # Select the most recent data point for prediction
            feature_columns = [
                'return_1d', 'return_5d', 'return_10d',
                'sma_5', 'sma_10', 'sma_20',
                'ema_5', 'ema_10', 'ema_20',
                'macd', 'macd_signal', 'macd_hist',
                'rsi_14',
                'bb_width',
                'atr_14',
                'price_sma_5_ratio', 'price_sma_10_ratio', 'price_sma_20_ratio',
                'sma_5_10_cross', 'sma_10_20_cross',
                'volume_ratio'
            ]
            
            # Normalize features by close price
            price_scale_features = [
                'sma_5', 'sma_10', 'sma_20',
                'ema_5', 'ema_10', 'ema_20',
                'bb_middle', 'bb_upper', 'bb_lower',
                'atr_14'
            ]
            
            for feature in price_scale_features:
                if feature in feature_columns:
                    df[feature] = df[feature] / df['close']
            
            # Get the latest data point
            latest_features = df[feature_columns].iloc[-1].values.reshape(1, -1)
            
            # Make prediction
            try:
                prediction = model.predict(latest_features)[0]
                confidence = model.predict_proba(latest_features)[0][prediction]
                
                predictions.append(prediction)
                confidences.append(confidence)
                
                logger.debug(f"{model_type} prediction for {symbol}: {prediction} with confidence {confidence:.2f}")
            
            except Exception as e:
                logger.error(f"Failed to generate {model_type} prediction for {symbol}: {e}")
        
        # If no predictions were made, return default signal
        if not predictions:
            return super(MLStrategy, self).generate_signal(symbol, data)
        
        # Calculate ensemble prediction
        # Use weighted voting based on confidence
        buy_votes = sum(conf for pred, conf in zip(predictions, confidences) if pred == 1)
        sell_votes = sum(conf for pred, conf in zip(predictions, confidences) if pred == 0)
        
        total_votes = buy_votes + sell_votes
        if total_votes == 0:
            return super(MLStrategy, self).generate_signal(symbol, data)
        
        buy_weight = buy_votes / total_votes
        sell_weight = sell_votes / total_votes
        
        # Generate signal based on ensemble prediction
        if buy_weight > sell_weight and buy_weight >= self.confidence_threshold:
            action = "buy"
            confidence = buy_weight
        elif sell_weight > buy_weight and sell_weight >= self.confidence_threshold:
            action = "sell"
            confidence = sell_weight
        else:
            action = "hold"
            confidence = max(buy_weight, sell_weight)
        
        logger.info(f"ML Ensemble Signal for {symbol}: {action.upper()} with confidence {confidence:.2f}")
        
        return {
            "symbol": symbol,
            "action": action,
            "confidence": confidence,
            "timestamp": current_timestamp,
            "metrics": {
                "buy_weight": float(buy_weight),
                "sell_weight": float(sell_weight),
                "model_predictions": [int(p) for p in predictions],
                "model_confidences": [float(c) for c in confidences]
            }
        }