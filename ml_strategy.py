#!/usr/bin/env python3
"""
Machine Learning Strategies for Trading Bot

This module provides machine learning-based trading strategies.
"""

import os
import logging
import json
import numpy as np
import pickle
from datetime import datetime
from strategies import Strategy

logger = logging.getLogger("TradingBot.Strategies")

class MLStrategy(Strategy):
    """
    Machine Learning strategy that uses trained models to predict price movements.
    """
    
    def __init__(self, params=None):
        """
        Initialize the ML strategy with parameters.
        
        Args:
            params (dict): Strategy parameters
        """
        default_params = {
            "model_type": "random_forest",
            "prediction_horizon": 5,
            "feature_window": 20,
            "confidence_threshold": 0.6,
            "retrain_interval": 30,
            "model_path": "models"
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
        os.makedirs(self.model_path, exist_ok=True)
        
        # Dictionary to store models for each symbol
        self.models = {}
        
        # Dictionary to store last training date for each symbol
        self.last_training = {}
        
        logger.info(f"ML Strategy initialized with model_type={self.model_type}, "
                   f"prediction_horizon={self.prediction_horizon}, "
                   f"feature_window={self.feature_window}")
    
    def _get_model_path(self, symbol):
        """
        Get the path to the model file for a symbol.
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            str: Path to the model file
        """
        # Replace / with _ in symbol name for file path
        safe_symbol = symbol.replace("/", "_")
        return os.path.join(self.model_path, f"{safe_symbol}_{self.model_type}_model.pkl")
    
    def _load_model(self, symbol):
        """
        Load a trained model for a symbol.
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            object: Trained model or None if not found
        """
        model_path = self._get_model_path(symbol)
        
        try:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                logger.info(f"Loaded model for {symbol} from {model_path}")
                return model
            else:
                logger.warning(f"No trained model found for {symbol}")
                return None
        except Exception as e:
            logger.error(f"Failed to load model for {symbol}: {e}")
            return None
    
    def _save_model(self, symbol, model):
        """
        Save a trained model for a symbol.
        
        Args:
            symbol (str): Trading symbol
            model (object): Trained model
            
        Returns:
            bool: Whether the model was saved successfully
        """
        model_path = self._get_model_path(symbol)
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved model for {symbol} to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model for {symbol}: {e}")
            return False
    
    def _extract_features(self, data):
        """
        Extract features from market data.
        
        Args:
            data (dict): Market data for a symbol
            
        Returns:
            tuple: (features, labels) or (features, None) if labels cannot be generated
        """
        if "historical_prices" not in data or len(data["historical_prices"]) < self.feature_window + self.prediction_horizon:
            logger.warning("Not enough historical data to extract features")
            return None, None
        
        prices = data["historical_prices"]
        features = []
        labels = []
        
        # Generate features and labels from historical prices
        for i in range(len(prices) - self.feature_window - self.prediction_horizon + 1):
            # Extract window of prices for features
            window = prices[i:i+self.feature_window]
            
            # Calculate features
            feature_vector = self._calculate_features(window)
            features.append(feature_vector)
            
            # Calculate label (price direction after prediction_horizon days)
            current_price = prices[i+self.feature_window-1]
            future_price = prices[i+self.feature_window+self.prediction_horizon-1]
            
            # 1 if price goes up, 0 if it goes down
            label = 1 if future_price > current_price else 0
            labels.append(label)
        
        return np.array(features), np.array(labels)
    
    def _calculate_features(self, prices):
        """
        Calculate features from a window of prices.
        
        Args:
            prices (list): Window of historical prices
            
        Returns:
            list: Feature vector
        """
        features = []
        
        # Price changes
        price_changes = np.diff(prices) / prices[:-1]
        features.extend([
            np.mean(price_changes),
            np.std(price_changes),
            np.min(price_changes),
            np.max(price_changes)
        ])
        
        # Moving averages
        ma_5 = np.mean(prices[-5:]) if len(prices) >= 5 else np.mean(prices)
        ma_10 = np.mean(prices[-10:]) if len(prices) >= 10 else np.mean(prices)
        ma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else np.mean(prices)
        
        features.extend([
            prices[-1] / ma_5 - 1,  # Price relative to 5-day MA
            prices[-1] / ma_10 - 1,  # Price relative to 10-day MA
            prices[-1] / ma_20 - 1,  # Price relative to 20-day MA
            ma_5 / ma_10 - 1,  # 5-day MA relative to 10-day MA
            ma_5 / ma_20 - 1,  # 5-day MA relative to 20-day MA
            ma_10 / ma_20 - 1  # 10-day MA relative to 20-day MA
        ])
        
        # Volatility
        volatility_5 = np.std(price_changes[-5:]) if len(price_changes) >= 5 else np.std(price_changes)
        volatility_10 = np.std(price_changes[-10:]) if len(price_changes) >= 10 else np.std(price_changes)
        volatility_20 = np.std(price_changes[-20:]) if len(price_changes) >= 20 else np.std(price_changes)
        
        features.extend([
            volatility_5,
            volatility_10,
            volatility_20,
            volatility_5 / volatility_20 if volatility_20 > 0 else 0
        ])
        
        return features
    
    def _create_model(self):
        """
        Create a new machine learning model.
        
        Returns:
            object: Machine learning model
        """
        try:
            if self.model_type == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(n_estimators=100, random_state=42)
            
            elif self.model_type == "gradient_boosting":
                from sklearn.ensemble import GradientBoostingClassifier
                return GradientBoostingClassifier(n_estimators=100, random_state=42)
            
            elif self.model_type == "svm":
                from sklearn.svm import SVC
                return SVC(probability=True, random_state=42)
            
            elif self.model_type == "logistic_regression":
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(random_state=42)
            
            else:
                logger.warning(f"Unknown model type: {self.model_type}. Using random forest.")
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(n_estimators=100, random_state=42)
        
        except ImportError:
            logger.error("scikit-learn is not installed. Please install it to use ML strategies.")
            return None
    
    def train_model(self, symbol, data):
        """
        Train a machine learning model for a symbol.
        
        Args:
            symbol (str): Trading symbol
            data (dict): Market data for the symbol
            
        Returns:
            object: Trained model or None if training failed
        """
        # Extract features and labels
        features, labels = self._extract_features(data)
        
        if features is None or labels is None or len(features) == 0 or len(labels) == 0:
            logger.warning(f"Not enough data to train model for {symbol}")
            return None
        
        try:
            # Create model
            model = self._create_model()
            
            if model is None:
                return None
            
            # Train model
            model.fit(features, labels)
            
            # Save model
            self._save_model(symbol, model)
            
            # Update last training date
            self.last_training[symbol] = datetime.now()
            
            # Store model in memory
            self.models[symbol] = model
            
            logger.info(f"Trained model for {symbol} with {len(features)} samples")
            
            return model
        
        except Exception as e:
            logger.error(f"Failed to train model for {symbol}: {e}")
            return None
    
    def _should_retrain(self, symbol):
        """
        Check if a model should be retrained.
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            bool: Whether the model should be retrained
        """
        # If no model exists, retrain
        if symbol not in self.models:
            return True
        
        # If no last training date, retrain
        if symbol not in self.last_training:
            return True
        
        # Check if retrain_interval days have passed since last training
        days_since_training = (datetime.now() - self.last_training[symbol]).days
        return days_since_training >= self.retrain_interval
    
    def generate_signal(self, symbol, data):
        """
        Generate a trading signal based on ML predictions.
        
        Args:
            symbol (str): Trading symbol
            data (dict): Market data for the symbol
            
        Returns:
            dict: Trading signal
        """
        # Check if we need to load or train a model
        if symbol not in self.models:
            # Try to load model
            model = self._load_model(symbol)
            
            if model is None:
                # Train new model
                model = self.train_model(symbol, data)
            
            if model is None:
                # If still no model, use base strategy
                logger.warning(f"No ML model available for {symbol}. Using base strategy.")
                return super().generate_signal(symbol, data)
            
            self.models[symbol] = model
        
        # Check if we should retrain the model
        if self._should_retrain(symbol):
            logger.info(f"Retraining model for {symbol}")
            model = self.train_model(symbol, data)
            
            if model is None:
                # If training fails, use existing model
                model = self.models[symbol]
        else:
            model = self.models[symbol]
        
        # Extract features for prediction
        if "historical_prices" not in data or len(data["historical_prices"]) < self.feature_window:
            logger.warning(f"Not enough historical data for {symbol} to generate ML signal")
            return super().generate_signal(symbol, data)
        
        prices = data["historical_prices"]
        window = prices[-self.feature_window:]
        features = [self._calculate_features(window)]
        
        try:
            # Make prediction
            prediction = model.predict(features)[0]
            
            # Get prediction probability
            prediction_proba = model.predict_proba(features)[0]
            confidence = prediction_proba[prediction]
            
            # Generate signal
            action = "buy" if prediction == 1 else "sell"
            
            # Only generate a signal if confidence is above threshold
            if confidence < self.confidence_threshold:
                action = "hold"
                confidence = 0.0
            
            logger.info(f"ML Signal for {symbol}: {action.upper()} with confidence {confidence:.2f}")
            
            return {
                "symbol": symbol,
                "action": action,
                "confidence": confidence,
                "timestamp": data.get("timestamp", 0),
                "metrics": {
                    "prediction": prediction,
                    "prediction_proba": prediction_proba.tolist(),
                    "model_type": self.model_type
                }
            }
        
        except Exception as e:
            logger.error(f"Failed to generate ML signal for {symbol}: {e}")
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
        
        # Call parent constructor with default params
        super().__init__(params or default_params)
        
        # Override name and model_type
        self.name = "ml_ensemble"
        self.models_list = self.params.get("models", ["random_forest", "gradient_boosting"])
        
        # Dictionary to store ensemble models for each symbol
        self.ensemble_models = {}
        
        logger.info(f"ML Ensemble Strategy initialized with models={self.models_list}, "
                   f"prediction_horizon={self.prediction_horizon}, "
                   f"feature_window={self.feature_window}")
    
    def _get_model_path(self, symbol, model_type):
        """
        Get the path to the model file for a symbol and model type.
        
        Args:
            symbol (str): Trading symbol
            model_type (str): Type of model
            
        Returns:
            str: Path to the model file
        """
        # Replace / with _ in symbol name for file path
        safe_symbol = symbol.replace("/", "_")
        return os.path.join(self.model_path, f"{safe_symbol}_{model_type}_model.pkl")
    
    def _load_ensemble_models(self, symbol):
        """
        Load trained ensemble models for a symbol.
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            dict: Dictionary of trained models or empty dict if none found
        """
        ensemble = {}
        
        for model_type in self.models_list:
            model_path = self._get_model_path(symbol, model_type)
            
            try:
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    logger.info(f"Loaded {model_type} model for {symbol} from {model_path}")
                    ensemble[model_type] = model
                else:
                    logger.warning(f"No trained {model_type} model found for {symbol}")
            except Exception as e:
                logger.error(f"Failed to load {model_type} model for {symbol}: {e}")
        
        return ensemble
    
    def _save_ensemble_models(self, symbol, ensemble):
        """
        Save trained ensemble models for a symbol.
        
        Args:
            symbol (str): Trading symbol
            ensemble (dict): Dictionary of trained models
            
        Returns:
            bool: Whether all models were saved successfully
        """
        success = True
        
        for model_type, model in ensemble.items():
            model_path = self._get_model_path(symbol, model_type)
            
            try:
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Saved {model_type} model for {symbol} to {model_path}")
            except Exception as e:
                logger.error(f"Failed to save {model_type} model for {symbol}: {e}")
                success = False
        
        return success
    
    def _create_model(self, model_type):
        """
        Create a new machine learning model of the specified type.
        
        Args:
            model_type (str): Type of model to create
            
        Returns:
            object: Machine learning model
        """
        try:
            if model_type == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(n_estimators=100, random_state=42)
            
            elif model_type == "gradient_boosting":
                from sklearn.ensemble import GradientBoostingClassifier
                return GradientBoostingClassifier(n_estimators=100, random_state=42)
            
            elif model_type == "svm":
                from sklearn.svm import SVC
                return SVC(probability=True, random_state=42)
            
            elif model_type == "logistic_regression":
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(random_state=42)
            
            else:
                logger.warning(f"Unknown model type: {model_type}. Using random forest.")
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(n_estimators=100, random_state=42)
        
        except ImportError:
            logger.error("scikit-learn is not installed. Please install it to use ML strategies.")
            return None
    
    def train_model(self, symbol, data):
        """
        Train ensemble models for a symbol.
        
        Args:
            symbol (str): Trading symbol
            data (dict): Market data for the symbol
            
        Returns:
            dict: Dictionary of trained models or None if training failed
        """
        # Extract features and labels
        features, labels = self._extract_features(data)
        
        if features is None or labels is None or len(features) == 0 or len(labels) == 0:
            logger.warning(f"Not enough data to train ensemble models for {symbol}")
            return None
        
        try:
            ensemble = {}
            
            # Train each model in the ensemble
            for model_type in self.models_list:
                # Create model
                model = self._create_model(model_type)
                
                if model is None:
                    continue
                
                # Train model
                model.fit(features, labels)
                
                # Add to ensemble
                ensemble[model_type] = model
            
            if not ensemble:
                logger.warning(f"Failed to train any models for {symbol}")
                return None
            
            # Save ensemble
            self._save_ensemble_models(symbol, ensemble)
            
            # Update last training date
            self.last_training[symbol] = datetime.now()
            
            # Store ensemble in memory
            self.ensemble_models[symbol] = ensemble
            
            logger.info(f"Trained ensemble of {len(ensemble)} models for {symbol} with {len(features)} samples")
            
            return ensemble
        
        except Exception as e:
            logger.error(f"Failed to train ensemble models for {symbol}: {e}")
            return None
    
    def generate_signal(self, symbol, data):
        """
        Generate a trading signal based on ensemble predictions.
        
        Args:
            symbol (str): Trading symbol
            data (dict): Market data for the symbol
            
        Returns:
            dict: Trading signal
        """
        # Check if we need to load or train ensemble models
        if symbol not in self.ensemble_models:
            # Try to load ensemble
            ensemble = self._load_ensemble_models(symbol)
            
            if not ensemble:
                # Train new ensemble
                ensemble = self.train_model(symbol, data)
            
            if not ensemble:
                # If still no ensemble, use base strategy
                logger.warning(f"No ML ensemble available for {symbol}. Using base strategy.")
                return super(MLStrategy, self).generate_signal(symbol, data)
            
            self.ensemble_models[symbol] = ensemble
        
        # Check if we should retrain the ensemble
        if self._should_retrain(symbol):
            logger.info(f"Retraining ensemble for {symbol}")
            ensemble = self.train_model(symbol, data)
            
            if not ensemble:
                # If training fails, use existing ensemble
                ensemble = self.ensemble_models[symbol]
        else:
            ensemble = self.ensemble_models[symbol]
        
        # Extract features for prediction
        if "historical_prices" not in data or len(data["historical_prices"]) < self.feature_window:
            logger.warning(f"Not enough historical data for {symbol} to generate ML ensemble signal")
            return super(MLStrategy, self).generate_signal(symbol, data)
        
        prices = data["historical_prices"]
        window = prices[-self.feature_window:]
        features = [self._calculate_features(window)]
        
        try:
            # Make predictions with each model
            predictions = []
            confidences = []
            
            for model_type, model in ensemble.items():
                prediction = model.predict(features)[0]
                prediction_proba = model.predict_proba(features)[0]
                confidence = prediction_proba[prediction]
                
                predictions.append(prediction)
                confidences.append(confidence)
            
            # Calculate ensemble prediction (majority vote)
            buy_votes = predictions.count(1)
            sell_votes = predictions.count(0)
            
            if buy_votes > sell_votes:
                action = "buy"
                confidence = sum([confidences[i] for i in range(len(predictions)) if predictions[i] == 1]) / buy_votes
            elif sell_votes > buy_votes:
                action = "sell"
                confidence = sum([confidences[i] for i in range(len(predictions)) if predictions[i] == 0]) / sell_votes
            else:
                # Tie - use average confidence
                buy_confidence = sum([confidences[i] for i in range(len(predictions)) if predictions[i] == 1]) / buy_votes if buy_votes > 0 else 0
                sell_confidence = sum([confidences[i] for i in range(len(predictions)) if predictions[i] == 0]) / sell_votes if sell_votes > 0 else 0
                
                if buy_confidence > sell_confidence:
                    action = "buy"
                    confidence = buy_confidence
                else:
                    action = "sell"
                    confidence = sell_confidence
            
            # Only generate a signal if confidence is above threshold
            if confidence < self.confidence_threshold:
                action = "hold"
                confidence = 0.0
            
            logger.info(f"ML Ensemble Signal for {symbol}: {action.upper()} with confidence {confidence:.2f}")
            
            return {
                "symbol": symbol,
                "action": action,
                "confidence": confidence,
                "timestamp": data.get("timestamp", 0),
                "metrics": {
                    "buy_votes": buy_votes,
                    "sell_votes": sell_votes,
                    "models": list(ensemble.keys())
                }
            }
        
        except Exception as e:
            logger.error(f"Failed to generate ML ensemble signal for {symbol}: {e}")
            return super(MLStrategy, self).generate_signal(symbol, data)


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
    parser = argparse.ArgumentParser(description="ML Trading Strategies")
    parser.add_argument("--train", action="store_true",
                        help="Train ML models")
    parser.add_argument("--symbol", type=str, default="BTC/USD",
                        help="Symbol to train model for")
    parser.add_argument("--model-type", type=str, default="random_forest",
                        choices=["random_forest", "gradient_boosting", "svm", "logistic_regression", "ensemble"],
                        help="Type of model to train")
    
    args = parser.parse_args()
    
    if args.train:
        logger.info(f"Training {args.model_type} model for {args.symbol}")
        
        # Import data fetcher
        from data_fetcher import DemoDataFetcher
        
        # Create data fetcher
        data_fetcher = DemoDataFetcher({
            "provider": "demo"
        })
        
        # Fetch historical data
        data = data_fetcher.fetch_historical_data(args.symbol, "1d", limit=500)
        
        if data is None:
            logger.error(f"Failed to fetch historical data for {args.symbol}")
            exit(1)
        
        # Train model
        if args.model_type == "ensemble":
            strategy = MLEnsembleStrategy()
        else:
            strategy = MLStrategy({
                "model_type": args.model_type
            })
        
        model = strategy.train_model(args.symbol, data)
        
        if model is not None:
            logger.info(f"Successfully trained {args.model_type} model for {args.symbol}")
        else:
            logger.error(f"Failed to train {args.model_type} model for {args.symbol}")
    else:
        logger.info("No action specified. Use --train to train ML models.")