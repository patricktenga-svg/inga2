# models/anomaly_detector.py
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

class AnomalyDetector:
    """Detect anomalies in sensor readings"""
    
    def __init__(self, model_path=None):
        self.model = IsolationForest(contamination=0.05, random_state=42)
        self.is_trained = False
        
        if model_path and os.path.exists(f"{model_path}_anomaly.pkl"):
            self.load(model_path)
    
    def generate_training_data(self, n_samples=1000):
        """Generate normal sensor data for training"""
        np.random.seed(42)
        # Normal operating conditions
        normal_data = np.random.multivariate_normal(
            mean=[1200, 500e6, 58, 42000],
            cov=[[100, 10, 5, 200], [10, 50, 2, 100], [5, 2, 2, 50], [200, 100, 50, 5000]],
            size=n_samples
        )
        return normal_data
    
    def train(self, X=None):
        """Train anomaly detector"""
        if X is None:
            X = self.generate_training_data()
        
        self.model.fit(X)
        self.is_trained = True
        print(f"✅ Anomaly detector trained on {len(X)} samples")
        return self.model
    
    def detect(self, turbine, storage, head, inflow):
        """Detect if current reading is anomalous"""
        if not self.is_trained:
            self.train()
        
        reading = np.array([[turbine, storage, head, inflow]])
        prediction = self.model.predict(reading)
        
        is_anomaly = (prediction[0] == -1)
        
        # Additional rule-based checks
        reason = None
        if turbine > 2200 or turbine < 100:
            reason = f"Abnormal turbine flow: {turbine} m³/s"
        elif storage < 250e6:
            reason = f"Critically low storage: {storage/1e6:.0f} Mm³"
        elif head < 35 or head > 65:
            reason = f"Abnormal head: {head} m"
        elif inflow < 2000:
            reason = f"Extreme drought: {inflow} m³/s"
        
        return {
            'is_anomaly': is_anomaly or (reason is not None),
            'reason': reason,
            'ml_score': float(prediction[0])
        }
    
    def save(self, path_prefix):
        """Save model"""
        if self.is_trained:
            joblib.dump(self.model, f"{path_prefix}_anomaly.pkl")
            print(f"✅ Anomaly detector saved to {path_prefix}_anomaly.pkl")
    
    def load(self, path_prefix):
        """Load model"""
        self.model = joblib.load(f"{path_prefix}_anomaly.pkl")
        self.is_trained = True
        print(f"✅ Anomaly detector loaded from {path_prefix}_anomaly.pkl")
