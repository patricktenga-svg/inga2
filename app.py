# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

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
# RECOMMENDATION MESSAGES IN ENGLISH
# ============================================================================
RECOMMENDATION_MESSAGES = {
    'NORMAL': {
        'message': '✅ Optimal conditions – Maximum production recommended',
        'action': 'PEAK'
    },
    'DRY': {
        'message': '⚠️ Dry conditions – Balanced strategy recommended',
        'action': 'NOMINAL'
    },
    'VERY DRY': {
        'message': '🔴 Severe drought – Reduce releases immediately',
        'action': 'LOW POWER'
    }
}


# ============================================================================
# REPORT GENERATOR
# ============================================================================
def generate_html_report(data_history, latest_data, scenario_result, recommendation, forecast):
    """Generate an HTML report with all monitoring data"""
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>INGA II Monitoring Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f6fa; }}
            .header {{ background: linear-gradient(135deg, #1a5276, #2e86c1); color: white; padding: 30px; border-radius: 10px; text-align: center; }}
            .metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
            .metric-card {{ background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-align: center; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #1a5276; }}
            .section {{ background: white; padding: 20px; border-radius: 10px; margin: 20px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .scenario-normal {{ color: #2ecc71; font-weight: bold; }}
            .scenario-dry {{ color: #f39c12; font-weight: bold; }}
            .scenario-critical {{ color: #e74c3c; font-weight: bold; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #1a5276; color: white; }}
            .footer {{ text-align: center; color: #7f8c8d; margin-top: 30px; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🏭 INGA II HYDROELECTRIC PLANT</h1>
            <p>Real-Time Monitoring Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="metric-grid">
            <div class="metric-card">
                <div>⚡ Power</div>
                <div class="metric-value">{latest_data['hydropower']:,} MW</div>
            </div>
            <div class="metric-card">
                <div>💧 Inflow</div>
                <div class="metric-value">{latest_data['inflow']:,} m³/s</div>
            </div>
            <div class="metric-card">
                <div>📦 Storage</div>
                <div class="metric-value">{latest_data['storage']/1e6:.0f} Mm³</div>
            </div>
            <div class="metric-card">
                <div>🚰 Irrigation</div>
                <div class="metric-value">{latest_data['irrigation']:,} m³/s</div>
            </div>
        </div>
        
        <div class="section">
            <h2>🎯 Scenario & Recommendation</h2>
            <p><strong>Scenario:</strong> <span class="scenario-{scenario_result['scenario'].lower().replace(' ', '')}">{scenario_result['scenario']}</span></p>
            <p><strong>Confidence:</strong> {scenario_result['confidence']:.1%}</p>
            <p><strong>AI Recommendation:</strong> {recommendation['action']}</p>
            <p><strong>Recommended Turbine:</strong> {recommendation['turbine']} m³/s</p>
            <p><strong>Recommended Irrigation:</strong> {recommendation['irrigation']} m³/s</p>
            <p><strong>Message:</strong> {recommendation['message']}</p>
        </div>
        
        <div class="section">
            <h2>📊 Forecast (7 days ahead)</h2>
            <table>
                <tr><th>Day</th><th>Predicted Inflow (m³/s)</th></tr>
                {''.join([f'<tr><td>Day {i+1}</td><td>{forecast[i]:.0f}</td></tr>' for i in range(len(forecast))])}
            </table>
        </div>
        
        <div class="section">
            <h2>📋 System Status</h2>
            <p><strong>Records Collected:</strong> {len(data_history)}</p>
            <p><strong>Frame:</strong> {len(data_history)}</p>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="footer">
            <p>INGA II AI Monitoring System | Powered by Streamlit</p>
            <p>© {datetime.now().year} - All Rights Reserved</p>
        </div>
    </body>
    </html>
    """
    return html

def generate_pdf_report(data_history, latest_data, scenario_result, recommendation, forecast):
    """Generate a PDF report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24, textColor=colors.HexColor('#1a5276'), alignment=TA_CENTER)
    story.append(Paragraph("🏭 INGA II HYDROELECTRIC PLANT", title_style))
    story.append(Paragraph(f"Real-Time Monitoring Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Metrics table
    metrics_data = [
        ['Metric', 'Value'],
        ['⚡ Power', f"{latest_data['hydropower']:,} MW"],
        ['💧 Inflow', f"{latest_data['inflow']:,} m³/s"],
        ['📦 Storage', f"{latest_data['storage']/1e6:.0f} Mm³"],
        ['🚰 Irrigation', f"{latest_data['irrigation']:,} m³/s"],
        ['Head', f"{latest_data['head']} m"]
    ]
    
    table = Table(metrics_data, colWidths=[200, 200])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5276')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    story.append(Spacer(1, 20))
    
    # Scenario & Recommendation
    story.append(Paragraph("🎯 Scenario & AI Recommendation", styles['Heading2']))
    story.append(Paragraph(f"<b>Scenario:</b> {scenario_result['scenario']}", styles['Normal']))
    story.append(Paragraph(f"<b>Confidence:</b> {scenario_result['confidence']:.1%}", styles['Normal']))
    story.append(Paragraph(f"<b>AI Recommendation:</b> {recommendation['action']}", styles['Normal']))
    story.append(Paragraph(f"<b>Recommended Turbine:</b> {recommendation['turbine']} m³/s", styles['Normal']))
    story.append(Paragraph(f"<b>Recommended Irrigation:</b> {recommendation['irrigation']} m³/s", styles['Normal']))
    story.append(Paragraph(f"<b>Message:</b> {recommendation['message']}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Forecast
    story.append(Paragraph("📊 7-Day Forecast", styles['Heading2']))
    forecast_data = [['Day', 'Predicted Inflow (m³/s)']]
    for i, val in enumerate(forecast):
        forecast_data.append([f'Day {i+1}', f'{val:.0f}'])
    
    forecast_table = Table(forecast_data, colWidths=[150, 150])
    forecast_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5276')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(forecast_table)
    story.append(Spacer(1, 20))
    
    # Footer
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Paragraph("INGA II AI Monitoring System | Powered by Streamlit", styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer


# ============================================================================
# DATA EXPORT WITH FILTERS
# ============================================================================
def filter_data_by_period(data_history, start_date=None, end_date=None):
    """Filter data by date range"""
    if not data_history:
        return []
    
    # Convert to DataFrame
    df = pd.DataFrame(data_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    if start_date:
        df = df[df['timestamp'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['timestamp'] <= pd.to_datetime(end_date)]
    
    return df.to_dict('records')


def export_filtered_data(data_history, format='csv', metrics=None, start_date=None, end_date=None):
    """Export filtered data with selected metrics"""
    filtered_data = filter_data_by_period(data_history, start_date, end_date)
    
    if not filtered_data:
        return None
    
    df = pd.DataFrame(filtered_data)
    
    # Select only requested metrics
    if metrics:
        available_metrics = [m for m in metrics if m in df.columns]
        if available_metrics:
            df = df[available_metrics]
    
    if format == 'csv':
        return df.to_csv(index=False).encode('utf-8')
    elif format == 'excel':
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='INGA II Data')
        return output.getvalue()
    return None


# ============================================================================
# Loading historical data
# ============================================================================
@st.cache_data
def load_historical_data(filepath="data/historical_inflows.csv"):
    """Loading historical data from CSV file..."""
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        st.success(f"✅ Historical data loaded: {len(df)} days ({df['date'].min().date()} to {df['date'].max().date()})")
        return df['inflow'].values
    else:
        st.warning(f"⚠️ File {filepath} not found. Using synthetic data instead.")
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
    
    if not forecaster.is_trained:
        if historical_inflows is not None and len(historical_inflows) > 100:
            with st.spinner("📈 Training LSTM model with historical data..."):
                forecaster.train(historical_inflows, epochs=50)
                forecaster.save("models/inga_ii")
            st.success("✅ LSTM model trained successfully!")
        else:
            st.warning("⚠️ Insufficient historical data. The LSTM model will use a default prediction mechanism.")
    
    if not classifier.is_trained:
        with st.spinner("🎯 Training scenario classifier..."):
            classifier.train()
            classifier.save("models/inga_ii")
    
    if not anomaly_detector.is_trained:
        with st.spinner("🚨 Training anomaly detector..."):
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
        st.markdown("### 📅 Data Export with Filters")
        
        # Date range filter
        if 'data_history' in st.session_state and len(st.session_state.data_history) > 0:
            # Get min and max dates from data
            df_history = pd.DataFrame(st.session_state.data_history)
            if 'timestamp' in df_history.columns:
                df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
                min_date = df_history['timestamp'].min().date()
                max_date = df_history['timestamp'].max().date()
                
                col_date1, col_date2 = st.columns(2)
                with col_date1:
                    start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
                with col_date2:
                    end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
                
                # Metric selection
                available_metrics = ['timestamp', 'inflow', 'storage', 'head', 'turbine', 'irrigation', 'hydropower']
                selected_metrics = st.multiselect(
                    "Select metrics to export",
                    available_metrics,
                    default=['timestamp', 'inflow', 'hydropower', 'storage']
                )
                
                # Export format
                export_format = st.radio("Export format", ["CSV", "Excel"], horizontal=True)
                
                if st.button("📥 Export Filtered Data", use_container_width=True):
                    data = export_filtered_data(
                        st.session_state.data_history,
                        format=export_format.lower(),
                        metrics=selected_metrics,
                        start_date=start_date,
                        end_date=end_date
                    )
                    if data:
                        file_ext = "csv" if export_format.lower() == "csv" else "xlsx"
                        mime = "text/csv" if export_format.lower() == "csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        st.download_button(
                            label=f"📥 Download {export_format}",
                            data=data,
                            file_name=f"inga_ii_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_ext}",
                            mime=mime,
                            use_container_width=True
                        )
            else:
                st.info("No data available. Wait for data collection.")
        else:
            st.info("No data available. Wait for data collection.")
        
        st.markdown("---")
        st.markdown("### 📄 Report Download")
        
        if 'data_history' in st.session_state and len(st.session_state.data_history) > 0:
            if st.button("📄 Generate Report", use_container_width=True):
                with st.spinner("Generating report..."):
                    latest_data = st.session_state.data_history[-1] if st.session_state.data_history else None
                    if latest_data:
                        inflows = [d['inflow'] for d in st.session_state.data_history]
                        forecast = forecaster.predict(inflows) if forecaster.is_trained else [42000] * 7
                        scenario_result = classifier.predict(latest_data['inflow'], latest_data['storage'])
                        state = [
                            latest_data['storage'] / 800e6,
                            latest_data['inflow'] / 60000,
                            latest_data['head'] / 70,
                            latest_data['turbine'] / 2200
                        ]
                        recommendation = rl_agent.get_recommendation(scenario_result['scenario'], state)
                        
                        html_report = generate_html_report(
                            st.session_state.data_history,
                            latest_data,
                            scenario_result,
                            recommendation,
                            forecast
                        )
                        
                        pdf_buffer = generate_pdf_report(
                            st.session_state.data_history,
                            latest_data,
                            scenario_result,
                            recommendation,
                            forecast
                        )
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                label="📊 Download PDF Report",
                                data=pdf_buffer,
                                file_name=f"inga_ii_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                        with col2:
                            st.download_button(
                                label="🌐 Download HTML Report",
                                data=html_report,
                                file_name=f"inga_ii_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                mime="text/html",
                                use_container_width=True
                            )
        else:
            st.info("No data available. Wait for data collection.")
        
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
    with st.spinner("📂 Loading historical data..."):
        historical_inflows = load_historical_data("data/historical_inflows.csv")
    
    # Load models with historical data
    with st.spinner("🤖 Loading and training AI models..."):
        forecaster, classifier, anomaly_detector, rl_agent = load_models(historical_inflows)
    
    simulator = RealTimeDataSimulator()
    placeholder = st.empty()
    
    # Display info about training
    if forecaster.is_trained:
        st.success("""
        ✅ **All AI models are ready and trained!**  
        - LSTM Forecaster: ✅  
        - Scenario Classifier: ✅  
        - Anomaly Detector: ✅  
        - RL Agent: ✅
        """)
    else:
        st.warning("⚠️ The system is currently using default predictions until more historical data becomes available.")
    
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
        
        # Get recommendation with English messages
        scenario_name = scenario_result['scenario']
        rec_data = RECOMMENDATION_MESSAGES.get(scenario_name, RECOMMENDATION_MESSAGES['NORMAL'])
        
        recommendation = {
            'action': rec_data['action'],
            'turbine': 1800 if scenario_name == 'NORMAL' else (1200 if scenario_name == 'DRY' else 500),
            'irrigation': 350 if scenario_name == 'NORMAL' else (250 if scenario_name == 'DRY' else 150),
            'message': rec_data['message']
        }
        
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
