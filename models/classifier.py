# models/classifier.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

class ScenarioClassifier:
    """Classify hydrological scenarios: NORMAL / DRY / VERY DRY"""
    
    def __init__(self, model_path=None):
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.scenarios = ['NORMAL', 'DRY', 'VERY_DRY']
        
        if model_path and os.path.exists(f"{model_path}_classifier.pkl"):
            self.load(model_path)
    
    def generate_training_data(self, n_samples=5000):
        """Generate synthetic training data"""
        np.random.seed(42)
        X = []
        y = []
        
        for _ in range(n_samples):
            inflow = np.random.uniform(1000, 55000)
            storage = np.random.uniform(200e6, 800e6)
            prev_inflows = inflow * np.random.uniform(0.5, 1.5, 3)
            rainfall = np.random.uniform(0, 150)
            
            X.append([inflow, storage, prev_inflows[0], prev_inflows[1], prev_inflows[2], rainfall])
            
            if inflow > 25000:
                y.append(0)  # NORMAL
            elif inflow > 4000:
                y.append(1)  # DRY
            else:
                y.append(2)  # VERY_DRY
        
        return np.array(X), np.array(y)
    
    def train(self, X=None, y=None):
        """Train classifier"""
        if X is None or y is None:
            X, y = self.generate_training_data()
        
        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, y)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.classifier.predict(X_scaled)
        accuracy = np.mean(y_pred == y)
        print(f"✅ Classifier trained (accuracy: {accuracy:.3f})")
        
        return self.classifier
    
    def predict(self, inflow, storage, prev_inflows=None, rainfall=50):
        """Predict scenario for current conditions"""
        if not self.is_trained:
            self.train()
        
        if prev_inflows is None:
            prev_inflows = [inflow * 0.9, inflow * 0.8, inflow * 0.7]
        
        features = np.array([[inflow, storage, prev_inflows[0], prev_inflows[1], prev_inflows[2], rainfall]])
        features_scaled = self.scaler.transform(features)
        
        prediction = self.classifier.predict(features_scaled)[0]
        probabilities = self.classifier.predict_proba(features_scaled)[0]
        
        return {
            'scenario': self.scenarios[prediction],
            'confidence': float(np.max(probabilities)),
            'probabilities': {s: float(p) for s, p in zip(self.scenarios, probabilities)}
        }
    
    def get_recommendation(self, scenario):
        """Get operational recommendation based on scenario"""
        recommendations = {
            'NORMAL': {
                'turbine': 1800,
                'irrigation': 350,
                'action': 'PEAK',
                'message': 'Conditions idéales – Production maximale recommandée'
            },
            'DRY': {
                'turbine': 1200,
                'irrigation': 250,
                'action': 'NOMINAL',
                'message': 'Conditions sèches – Adopter une stratégie équilibrée'
            },
            'VERY DRY': {
                'turbine': 500,
                'irrigation': 150,
                'action': 'LOW POWER',
                'message': 'Sécheresse sévère – Réduire impérativement les lâchers'
            }
        }
        return recommendations.get(scenario, recommendations['NORMAL'])
    
    def save(self, path_prefix):
        """Save model"""
        if self.is_trained:
            joblib.dump(self.classifier, f"{path_prefix}_classifier.pkl")
            joblib.dump(self.scaler, f"{path_prefix}_scaler.pkl")
            print(f"✅ Classifier saved to {path_prefix}_classifier.pkl")
    
    def load(self, path_prefix):
        """Load model"""
        self.classifier = joblib.load(f"{path_prefix}_classifier.pkl")
        self.scaler = joblib.load(f"{path_prefix}_scaler.pkl")
        self.is_trained = True
        print(f"✅ Classifier loaded from {path_prefix}_classifier.pkl")
