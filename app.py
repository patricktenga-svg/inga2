# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os

# Import models
from models.lstm_model import InflowForecaster
from models.classifier import ScenarioClassifier
from models.anomaly_detector import AnomalyDetector
from models.rl_agent import RLAgent
from utils.data_exporter import export_to_csv, export_to_excel
from utils.visualizations import (
    create_gauge_chart, create_forecast_chart,
    create_reservoir_dashboard, create_scenario_radar
)

st.set_page_config(page_title="INGA II - AI Monitoring", page_icon="🏭", layout="wide")


# ============================================================================
# Loading historical data
# ============================================================================
@st.cache_data
def load_historical_data(filepath="data/historical_inflows.csv"):
    """Charge les données historiques depuis le fichier CSV."""
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        st.success(f"✅ Données historiques chargées: {len(df)} jours ({df['date'].min().date()} au {df['date'].max().date()})")
        return df['inflow'].values
    else:
        st.warning(f"⚠️ Fichier {filepath} non trouvé. Utilisation de données synthétiques.")
        # Générer des données synthétiques de secours
        synthetic_inflows = np.random.normal(42000, 5000, 2000)
        return synthetic_inflows


# ============================================================================
# Loading models
# ============================================================================
@st.cache_resource
def load_models(historical_inflows=None):
    """Load or train all ML models"""
    forecaster = InflowForecaster(model_path="models/inga_ii")
    classifier = ScenarioClassifier(model_path="models/inga_ii")
    anomaly_detector = AnomalyDetector(model_path="models/inga_ii")
    rl_agent = RLAgent(model_path="models/inga_ii")
    
    # Train forecaster if needed
    if not forecaster.is_trained:
        if historical_inflows is not None and len(historical_inflows) > 100:
            with st.spinner("📈 Entraînement du modèle LSTM avec des données historiques..."):
                forecaster.train(historical_inflows, epochs=50)
                forecaster.save("models/inga_ii")
            st.success("✅ Modèle LSTM entraîné avec succès !")
        else:
            st.warning("⚠️ Pas assez de données historiques. Le modèle LSTM utilisera un mécanisme de prédiction par défaut.")
    
    # Train classifier if needed
    if not classifier.is_trained:
        with st.spinner("🎯 Entraînement du classifieur de scénarios..."):
            classifier.train()
            classifier.save("models/inga_ii")
    
    # Train anomaly detector if needed
    if not anomaly_detector.is_trained:
        with st.spinner("🚨 Entraînement du détecteur d'anomalies..."):
            anomaly_detector.train()
            anomaly_detector.save("models/inga_ii")
    
    return forecaster, classifier, anomaly_detector, rl_agent


# ============================================================================
# Data simulator in real time
# ============================================================================
class RealTimeDataSimulator:
    def __init__(self):
        self.base_inflow = 42000
        self.base_storage = 500e6
        self.base_turbine = 1200
        self.base_irrigation = 250
        self.seasonal_amplitude = 8000
        self.step = 0
    
    def get_current_data(self):
        now = datetime.now()
        day_of_year = now.timetuple().tm_yday
        hour = now.hour
        
        seasonal = np.sin(2 * np.pi * day_of_year / 365) * self.seasonal_amplitude
        daily = np.sin(2 * np.pi * hour / 24) * 2000
        
        inflow = self.base_inflow + seasonal + daily + np.random.normal(0, 300)
        inflow = max(1000, min(60000, inflow))
        
        storage = self.base_storage + seasonal * 1000 + np.random.normal(0, 5e6)
        storage = max(200e6, min(800e6, storage))
        
        head = 58 * (storage / 800e6)
        turbine = self.base_turbine + np.random.normal(0, 50)
        turbine = max(0, min(2200, turbine))
        irrigation = 250 + np.random.normal(0, 20)
        irrigation = max(0, min(500, irrigation))
        hydropower = 0.9 * 1000 * 9.81 * head * turbine / 1e6
        
        return {
            'timestamp': now,
            'inflow': round(inflow),
            'storage': round(storage),
            'head': round(head, 1),
            'turbine': round(turbine),
            'irrigation': round(irrigation),
            'hydropower': round(hydropower)
        }


# ============================================================================
# Main Function
# ============================================================================
def main():
    st.title("🏭 INGA II HYDROELECTRIC PLANT - DRC")
    st.markdown("### 🤖 AI-Powered Real-Time Monitoring & Decision Support")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Controls")
        update_freq = st.slider("Update frequency (seconds)", 1, 10, 3)
        
        st.markdown("---")
        st.markdown("### 📊 Technical Specifications")
        st.markdown("""
        - **Installed Capacity:** 1,280 MW
        - **Turbines:** 8 × 160 MW
        - **Gross Head:** 58 m
        - **Intake Capacity:** 2,200 m³/s
        """)
        
        st.markdown("---")
        st.markdown("### 💾 Data Export")
        if st.button("📥 Export to CSV"):
            csv_data = export_to_csv(st.session_state.data_history)
            if csv_data:
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"inga_ii_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        if st.button("📊 Export to Excel"):
            excel_data = export_to_excel(st.session_state.data_history)
            if excel_data:
                st.download_button(
                    label="Download Excel",
                    data=excel_data,
                    file_name=f"inga_ii_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        st.markdown("---")
        st.markdown("### 📈 Data Status")
        if 'data_history' in st.session_state and len(st.session_state.data_history) > 0:
            st.metric("Records collected", len(st.session_state.data_history))
    
    # Initialize session state
    if 'data_history' not in st.session_state:
        st.session_state.data_history = []
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = True
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    
    # Load historical data
    with st.spinner("📂 Chargement des données historiques..."):
        historical_inflows = load_historical_data("data/historical_inflows.csv")
    
    # Load models with historical data
    with st.spinner("🤖 Chargement et entraînement des modèles IA..."):
        forecaster, classifier, anomaly_detector, rl_agent = load_models(historical_inflows)
    
    simulator = RealTimeDataSimulator()
    placeholder = st.empty()
    
    # Display info about training
    if forecaster.is_trained:
        st.success("✅ Tous les modèles IA sont prêts et entraînés!")
    else:
        st.info("ℹ️ Le système fonctionne avec des prédictions par défaut en attendant plus de données historiques.")
    
    st.markdown("---")
    
    while st.session_state.simulation_running:
        st.session_state.frame_count += 1
        frame = st.session_state.frame_count
        
        current_data = simulator.get_current_data()
        st.session_state.data_history.append(current_data)
        if len(st.session_state.data_history) > 500:
            st.session_state.data_history = st.session_state.data_history[-500:]
        
        # Get forecasts and predictions
        inflows = [d['inflow'] for d in st.session_state.data_history]
        forecast = forecaster.predict(inflows)
        scenario_result = classifier.predict(current_data['inflow'], current_data['storage'])
        anomaly = anomaly_detector.detect(
            current_data['turbine'], current_data['storage'], 
            current_data['head'], current_data['inflow']
        )
        
        # State for RL
        state = [
            current_data['storage'] / 800e6,
            current_data['inflow'] / 60000,
            current_data['head'] / 70,
            current_data['turbine'] / 2200
        ]
        recommendation = rl_agent.get_recommendation(scenario_result['scenario'], state)
        
        with placeholder.container():
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("⚡ Power", f"{current_data['hydropower']:,} MW", 
                       delta=f"{current_data['hydropower'] - 1280:.0f} vs nominal")
            col2.metric("💧 Inflow", f"{current_data['inflow']:,} m³/s")
            col3.metric("📦 Storage", f"{current_data['storage']/1e6:.0f} Mm³")
            col4.metric("🚰 Irrigation", f"{current_data['irrigation']:,} m³/s")
            
            # Scenario and Recommendation
            col_status, col_rec = st.columns(2)
            with col_status:
                if scenario_result['scenario'] == "VERY DRY":
                    st.error(f"🚨 CRITICAL: {scenario_result['scenario']} (confidence: {scenario_result['confidence']:.0%})")
                elif scenario_result['scenario'] == "DRY":
                    st.warning(f"⚠️ WARNING: {scenario_result['scenario']} (confidence: {scenario_result['confidence']:.0%})")
                else:
                    st.success(f"✅ NORMAL: {scenario_result['scenario']} (confidence: {scenario_result['confidence']:.0%})")
                
                st.plotly_chart(
                    create_scenario_radar(scenario_result['probabilities']), 
                    use_container_width=True,
                    key=f"radar_{frame}"
                )
            
            with col_rec:
                st.info(f"🎯 **AI Recommendation:** {recommendation['action']}")
                st.metric("Recommended Turbine", f"{recommendation['turbine']} m³/s")
                st.metric("Recommended Irrigation", f"{recommendation['irrigation']} m³/s")
                st.caption(recommendation['message'])
            
            # Gauges with unique keys
            col_g1, col_g2, col_g3 = st.columns(3)
            with col_g1:
                st.plotly_chart(
                    create_gauge_chart(current_data['inflow'], 0, 60000, "Inflow", "m³/s"), 
                    use_container_width=True,
                    key=f"gauge_inflow_{frame}"
                )
            with col_g2:
                st.plotly_chart(
                    create_gauge_chart(current_data['storage']/1e6, 200, 800, "Storage", "Mm³"), 
                    use_container_width=True,
                    key=f"gauge_storage_{frame}"
                )
            with col_g3:
                st.plotly_chart(
                    create_gauge_chart(current_data['head'], 30, 70, "Head", "m"), 
                    use_container_width=True,
                    key=f"gauge_head_{frame}"
                )
            
            # Forecast chart
            st.plotly_chart(
                create_forecast_chart(inflows, forecast), 
                use_container_width=True,
                key=f"forecast_{frame}"
            )
            
            # Reservoir Dashboard
            if len(st.session_state.data_history) > 50:
                with st.expander("🏭 View Full Reservoir Dashboard"):
                    st.plotly_chart(
                        create_reservoir_dashboard(st.session_state.data_history), 
                        use_container_width=True,
                        key=f"dashboard_{frame}"
                    )
            
            # Training history (if available)
            if forecaster.training_losses and len(forecaster.training_losses['train']) > 0:
                with st.expander("📊 View LSTM Training History"):
                    import plotly.graph_objects as go
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(y=forecaster.training_losses['train'], name='Train Loss'))
                    fig_loss.add_trace(go.Scatter(y=forecaster.training_losses['val'], name='Val Loss'))
                    fig_loss.update_layout(title="Training History", xaxis_title="Epoch", yaxis_title="Loss")
                    st.plotly_chart(fig_loss, use_container_width=True, key=f"loss_{frame}")
            
            # Anomaly
            if anomaly['is_anomaly']:
                st.warning(f"🚨 Anomaly detected: {anomaly['reason'] or 'ML model detected unusual pattern'}")
            else:
                st.success("✅ All systems normal")
            
            st.caption(f"🕐 Last update: {datetime.now().strftime('%H:%M:%S')} | Frame: {frame}")
        
        time.sleep(update_freq)


if __name__ == "__main__":
    main()
