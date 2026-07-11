# models/lstm_model.py
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

class LSTMForecaster(nn.Module):
    """Standard LSTM model for inflow forecasting"""
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=7):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out


class InflowForecaster:
    """Wrapper for LSTM inflow forecasting"""
    
    def __init__(self, lookback=30, forecast_horizon=7, model_path=None):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        self.training_losses = []
        
        if model_path and os.path.exists(f"{model_path}_lstm.pth"):
            self.load(model_path)
    
    def prepare_data(self, data):
        """Prepare sequences for training"""
        X, y = [], []
        for i in range(len(data) - self.lookback - self.forecast_horizon + 1):
            X.append(data[i:i+self.lookback])
            y.append(data[i+self.lookback:i+self.lookback+self.forecast_horizon])
        return np.array(X), np.array(y)
    
    def train(self, historical_inflows, epochs=100, lr=0.001):
        """Train LSTM model"""
        # Normalize data
        scaled_data = self.scaler.fit_transform(historical_inflows.reshape(-1, 1)).flatten()
        
        # Prepare sequences
        X, y = self.prepare_data(scaled_data)
        if len(X) == 0:
            raise ValueError("Not enough data for training")
        
        X = X.reshape(-1, self.lookback, 1)
        
        # Train/validation split
        split_idx = int(len(X) * 0.9)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Build model
        self.model = LSTMForecaster(input_size=1, hidden_size=64, num_layers=2, output_size=self.forecast_horizon)
        
        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        criterion = nn.MSELoss()
        
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        
        self.training_losses = {'train': [], 'val': []}
        best_val_loss = float('inf')
        
        print(f"\n📈 Training LSTM...")
        print(f"   Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            optimizer.zero_grad()
            output = self.model(X_train_tensor)
            loss = criterion(output, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            train_loss = loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_output = self.model(X_val_tensor)
                val_loss = criterion(val_output, y_val_tensor).item()
            
            self.training_losses['train'].append(train_loss)
            self.training_losses['val'].append(val_loss)
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
            
            if (epoch + 1) % 20 == 0:
                print(f"   Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        self.is_trained = True
        print(f"\n✅ LSTM model trained! Best val loss: {best_val_loss:.6f}")
        
        return self.model
    
    def predict(self, recent_inflows, days=7):
        """Predict next days inflow"""
        # Handle insufficient data (fallback to simple method)
        if len(recent_inflows) < self.lookback:
            # Fallback: use simple moving average or default value
            if len(recent_inflows) >= 7:
                # Use recent trend
                trend = np.mean(recent_inflows[-7:]) - np.mean(recent_inflows[-14:-7]) if len(recent_inflows) >= 14 else 0
                predictions = []
                last = recent_inflows[-1] if recent_inflows else 42000
                for i in range(days):
                    pred = last + trend * (i + 1)
                    predictions.append(max(1000, min(60000, pred)))
                    last = pred
                return predictions
            # Default: normal inflow
            return [42000 + (i * 100) for i in range(days)]
        
        # Check if model is trained
        if not self.is_trained or self.model is None:
            # Fallback to trend-based prediction
            trend = np.mean(recent_inflows[-7:]) - np.mean(recent_inflows[-14:-7]) if len(recent_inflows) >= 14 else 0
            predictions = []
            last = recent_inflows[-1]
            for i in range(days):
                pred = last + trend * (i + 1)
                predictions.append(max(1000, min(60000, pred)))
                last = pred
            return predictions
        
        try:
            # Scale input
            recent_scaled = self.scaler.transform(np.array(recent_inflows[-self.lookback:]).reshape(-1, 1)).flatten()
            
            # Verify size
            if len(recent_scaled) != self.lookback:
                # Fallback
                return [42000] * days
            
            X_pred = torch.FloatTensor(recent_scaled).reshape(1, self.lookback, 1)
            
            self.model.eval()
            with torch.no_grad():
                predictions_scaled = self.model(X_pred).numpy().flatten()
            
            # Inverse transform
            predictions = self.scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
            predictions = [max(1000, min(60000, p)) for p in predictions]
            
            return predictions
            
        except Exception as e:
            print(f"Prediction error: {e}, using fallback")
            # Fallback
            return [42000] * days
    
    def save(self, path_prefix):
        """Save model and scaler"""
        if self.model:
            torch.save(self.model.state_dict(), f"{path_prefix}_lstm.pth")
            joblib.dump(self.scaler, f"{path_prefix}_scaler.pkl")
            joblib.dump(self.training_losses, f"{path_prefix}_losses.pkl")
            print(f"✅ LSTM model saved to {path_prefix}_lstm.pth")
    
    def load(self, path_prefix):
        """Load model and scaler"""
        self.model = LSTMForecaster(output_size=self.forecast_horizon)
        self.model.load_state_dict(torch.load(f"{path_prefix}_lstm.pth"))
        self.scaler = joblib.load(f"{path_prefix}_scaler.pkl")
        if os.path.exists(f"{path_prefix}_losses.pkl"):
            self.training_losses = joblib.load(f"{path_prefix}_losses.pkl")
        self.is_trained = True
        print(f"✅ LSTM model loaded from {path_prefix}_lstm.pth")
