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

st.set_page_config(
    page_title="INGA II - AI Monitoring",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# ADAPTIVE THEME CSS (Light & Dark Mode Compatible)
# ============================================================================
st.markdown("""
<style>
    /* ===== VARIABLES ADAPTIVES ===== */
    :root {
        --bg-primary: #f0f4f8;
        --bg-secondary: #ffffff;
        --text-primary: #1a2a3a;
        --text-secondary: #2c3e50;
        --border-color: #e8ecf0;
        --shadow-color: rgba(0,0,0,0.08);
        --card-bg: #ffffff;
        --badge-bg: rgba(0,0,0,0.05);
        --badge-text: #555;
        --success-bg: #d4edda;
        --warning-bg: #fff3cd;
        --danger-bg: #f8d7da;
        --info-bg: #d1ecf1;
    }
    
    /* ===== DARK MODE ===== */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --text-primary: #e8e8e8;
            --text-secondary: #b0b0b0;
            --border-color: #2a2a4a;
            --shadow-color: rgba(0,0,0,0.4);
            --card-bg: #1e2a4a;
            --badge-bg: rgba(255,255,255,0.08);
            --badge-text: #ccc;
            --success-bg: #1a3a2a;
            --warning-bg: #3a2a1a;
            --danger-bg: #3a1a1a;
            --info-bg: #1a2a3a;
        }
    }
    
    /* ===== DARK MODE OVERRIDE (Streamlit theme) ===== */
    [data-testid="stAppViewContainer"] {
        background-color: var(--bg-primary);
    }
    
    /* ===== HEADER ===== */
    .header-container {
        background: linear-gradient(135deg, rgba(26, 82, 118, 0.92), rgba(46, 134, 193, 0.85)), 
                    url('https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Inga_Dam_DRC_2020.jpg/1280px-Inga_Dam_DRC_2020.jpg');
        background-size: cover;
        background-position: center;
        padding: 40px 30px;
        border-radius: 20px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .header-title {
        color: white !important;
        font-size: 42px;
        font-weight: 700;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.5);
        margin: 0;
        letter-spacing: 1px;
    }
    
    .header-subtitle {
        color: rgba(255,255,255,0.95) !important;
        font-size: 18px;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.4);
        margin-top: 5px;
    }
    
    .header-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        padding: 5px 20px;
        border-radius: 20px;
        color: white !important;
        font-size: 12px;
        margin-top: 10px;
        border: 1px solid rgba(255,255,255,0.25);
    }
    
    /* ===== METRIC CARDS ===== */
    .metric-card {
        background: var(--card-bg);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 15px var(--shadow-color);
        transition: transform 0.2s, box-shadow 0.2s;
        border-left: 4px solid #2e86c1;
        color: var(--text-primary);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px var(--shadow-color);
    }
    
    .metric-card .metric-label {
        color: var(--text-secondary);
        font-size: 14px;
    }
    
    .metric-card .metric-value {
        color: var(--text-primary);
        font-size: 24px;
        font-weight: bold;
    }
    
    /* ===== STATUS BOXES ===== */
    .scenario-normal {
        background: var(--success-bg);
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 10px;
        color: var(--text-primary);
    }
    
    .scenario-dry {
        background: var(--warning-bg);
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 10px;
        color: var(--text-primary);
    }
    
    .scenario-critical {
        background: var(--danger-bg);
        border-left: 5px solid #dc3545;
        padding: 15px;
        border-radius: 10px;
        color: var(--text-primary);
    }
    
    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-color);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text-primary);
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s;
        background: var(--card-bg);
        color: var(--text-primary);
        border: 1px solid var(--border-color);
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px var(--shadow-color);
    }
    
    /* ===== METRIC (Streamlit) ===== */
    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
    }
    
    /* ===== INFO/WARNING/SUCCESS BOXES ===== */
    .stAlert {
        background-color: var(--card-bg) !important;
        color: var(--text-primary) !important;
    }
    
    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        background-color: var(--card-bg) !important;
        color: var(--text-primary) !important;
        border-color: var(--border-color) !important;
    }
    
    /* ===== INFO BOX ===== */
    .info-box {
        background: var(--info-bg);
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #2196F3;
        color: var(--text-primary);
    }
    
    .success-box {
        background: var(--success-bg);
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #28a745;
        color: var(--text-primary);
    }
    
    .warning-box {
        background: var(--warning-bg);
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #ffc107;
        color: var(--text-primary);
    }
    
    /* ===== DATAFRAME ===== */
    .stDataFrame {
        background: var(--card-bg) !important;
        color: var(--text-primary) !important;
    }
    
    /* ===== PLOTLY ===== */
    .js-plotly-plot .plotly {
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HEADER WITH BACKGROUND IMAGE
# ============================================================================
# Check if local image exists
header_image = "data/INGA.jpg"
if os.path.exists(header_image):
    with open(header_image, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    header_bg = f"url('data:image/jpeg;base64,{image_data}')"
else:
    header_bg = "url('https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Inga_Dam_DRC_2020.jpg/1280px-Inga_Dam_DRC_2020.jpg')"

st.markdown(f"""
<style>
    .header-container {{
        background: linear-gradient(135deg, rgba(26, 82, 118, 0.92), rgba(46, 134, 193, 0.85)), 
                    {header_bg};
        background-size: cover;
        background-position: center;
        padding: 40px 30px;
        border-radius: 20px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="header-container">
    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
        <div>
            <h1 class="header-title">🏭 INGA II HYDROELECTRIC PLANT</h1>
            <p class="header-subtitle">🤖 AI-Powered Real-Time Monitoring & Decision Support</p>
            <div style="margin-top: 15px;">
                <span class="header-badge">⚡ 1,280 MW Capacity</span>
                <span class="header-badge" style="margin-left: 10px;">🌊 Congo River, DRC</span>
                <span class="header-badge" style="margin-left: 10px;">🔄 Live: {datetime.now().strftime('%H:%M:%S')}</span>
            </div>
        </div>
        <div style="text-align: right; color: rgba(255,255,255,0.85); font-size: 12px;">
            <div>8 × 160 MW Turbines</div>
            <div>58 m Gross Head</div>
            <div>2,200 m³/s Intake</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# RECOMMENDATION MESSAGES IN ENGLISH
# ============================================================================
RECOMMENDATION_MESSAGES = {
    'NORMAL': {
        'message': 'Optimal conditions – Maximum production recommended',
        'action': 'PEAK'
    },
    'DRY': {
        'message': 'Dry conditions – Balanced strategy recommended',
        'action': 'NOMINAL'
    },
    'VERY DRY': {
        'message': 'Severe drought – Reduce releases immediately',
        'action': 'LOW POWER'
    }
}


# ============================================================================
# REPORT GENERATOR
# ============================================================================
def get_date_range(data_history):
    """Get dynamic date range from data"""
    if not data_history:
        return "No data"
    
    df = pd.DataFrame(data_history)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        min_date = df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')
        max_date = df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
        return f"{min_date} to {max_date}"
    return "No date range"

def generate_html_report(data_history, latest_data, scenario_result, recommendation, forecast):
    """Generate an HTML report with all monitoring data"""
    
    if not data_history or not latest_data:
        return "<html><body><h1>No data available</h1></body></html>"
    
    date_range = get_date_range(data_history)
    
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
            <p style="font-size: 12px; opacity: 0.8;">Data Period: {date_range}</p>
        </div>
        
        <div class="metric-grid">
            <div class="metric-card">
                <div>⚡ Power</div>
                <div class="metric-value">{latest_data.get('hydropower', 0):,} MW</div>
            </div>
            <div class="metric-card">
                <div>💧 Inflow</div>
                <div class="metric-value">{latest_data.get('inflow', 0):,} m³/s</div>
            </div>
            <div class="metric-card">
                <div>📦 Storage</div>
                <div class="metric-value">{latest_data.get('storage', 0)/1e6:.0f} Mm³</div>
            </div>
            <div class="metric-card">
                <div>🚰 Irrigation</div>
                <div class="metric-value">{latest_data.get('irrigation', 0):,} m³/s</div>
            </div>
        </div>
        
        <div class="section">
            <h2>🎯 Scenario & Recommendation</h2>
            <p><strong>Scenario:</strong> <span class="scenario-{scenario_result.get('scenario', 'NORMAL').lower()}">{scenario_result.get('scenario', 'NORMAL')}</span></p>
            <p><strong>Confidence:</strong> {scenario_result.get('confidence', 0):.1%}</p>
            <p><strong>AI Recommendation:</strong> {recommendation.get('action', 'N/A')}</p>
            <p><strong>Recommended Turbine:</strong> {recommendation.get('turbine', 0)} m³/s</p>
            <p><strong>Recommended Irrigation:</strong> {recommendation.get('irrigation', 0)} m³/s</p>
            <p><strong>Message:</strong> {recommendation.get('message', 'N/A')}</p>
        </div>
        
        <div class="section">
            <h2>📊 Forecast (7 days ahead)</h2>
            <table>
                <tr><th>Day</th><th>Predicted Inflow (m³/s)</th></tr>
                {''.join([f'<tr><td>Day {i+1}</td><td>{forecast[i]:.0f}</td></tr>' for i in range(len(forecast))]) if forecast else '<tr><td colspan="2">No forecast available</td></tr>'}
            </table>
        </div>
        
        <div class="section">
            <h2>📋 System Status</h2>
            <p><strong>Records Collected:</strong> {len(data_history)}</p>
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
    
    if not data_history or not latest_data:
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        story.append(Paragraph("No data available for report", styles['Heading1']))
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    date_range = get_date_range(data_history)
    
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24, textColor=colors.HexColor('#1a5276'), alignment=TA_CENTER)
    story.append(Paragraph("🏭 INGA II HYDROELECTRIC PLANT", title_style))
    story.append(Paragraph(f"Real-Time Monitoring Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Paragraph(f"Data Period: {date_range}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    metrics_data = [
        ['Metric', 'Value'],
        ['⚡ Power', f"{latest_data.get('hydropower', 0):,} MW"],
        ['💧 Inflow', f"{latest_data.get('inflow', 0):,} m³/s"],
        ['📦 Storage', f"{latest_data.get('storage', 0)/1e6:.0f} Mm³"],
        ['🚰 Irrigation', f"{latest_data.get('irrigation', 0):,} m³/s"],
        ['Head', f"{latest_data.get('head', 0)} m"]
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
    
    story.append(Paragraph("🎯 Scenario & AI Recommendation", styles['Heading2']))
    story.append(Paragraph(f"<b>Scenario:</b> {scenario_result.get('scenario', 'NORMAL')}", styles['Normal']))
    story.append(Paragraph(f"<b>Confidence:</b> {scenario_result.get('confidence', 0):.1%}", styles['Normal']))
    story.append(Paragraph(f"<b>AI Recommendation:</b> {recommendation.get('action', 'N/A')}", styles['Normal']))
    story.append(Paragraph(f"<b>Recommended Turbine:</b> {recommendation.get('turbine', 0)} m³/s", styles['Normal']))
    story.append(Paragraph(f"<b>Recommended Irrigation:</b> {recommendation.get('irrigation', 0)} m³/s", styles['Normal']))
    story.append(Paragraph(f"<b>Message:</b> {recommendation.get('message', 'N/A')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    if forecast:
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
    
    df = pd.DataFrame(data_history)
    if 'timestamp' in df.columns:
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
        return df['inflow'].values
    else:
        synthetic_inflows = np.random.normal(42000, 5000, 2000)
        return synthetic_inflows


# ============================================================================
# Loading models
# ============================================================================
@st.cache_resource
def load_models():
    """Load or train all ML models"""
    historical_inflows = load_historical_data("data/historical_inflows.csv")
    
    forecaster = InflowForecaster(model_path="models/inga_ii")
    classifier = ScenarioClassifier(model_path="models/inga_ii")
    anomaly_detector = AnomalyDetector(model_path="models/inga_ii")
    rl_agent = RLAgent(model_path="models/inga_ii")
    
    if not forecaster.is_trained and historical_inflows is not None and len(historical_inflows) > 100:
        forecaster.train(historical_inflows, epochs=50)
        forecaster.save("models/inga_ii")
    
    if not classifier.is_trained:
        classifier.train()
        classifier.save("models/inga_ii")
    
    if not anomaly_detector.is_trained:
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
    # Initialize session state FIRST
    if 'data_history' not in st.session_state:
        st.session_state.data_history = []
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = True
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    
    # Load models ONCE and store in session state
    if 'forecaster' not in st.session_state:
        with st.spinner("🤖 Loading AI models..."):
            forecaster, classifier, anomaly_detector, rl_agent = load_models()
            st.session_state.forecaster = forecaster
            st.session_state.classifier = classifier
            st.session_state.anomaly_detector = anomaly_detector
            st.session_state.rl_agent = rl_agent
            st.session_state.models_loaded = True
    
    # Get models from session state
    forecaster = st.session_state.forecaster
    classifier = st.session_state.classifier
    anomaly_detector = st.session_state.anomaly_detector
    rl_agent = st.session_state.rl_agent
    
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
        st.markdown("### 📅 Data Export")
        
        data_count = len(st.session_state.data_history)
        st.metric("📊 Records", data_count)
        
        if data_count > 0:
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
                
                available_metrics = ['timestamp', 'inflow', 'storage', 'head', 'turbine', 'irrigation', 'hydropower']
                selected_metrics = st.multiselect(
                    "Select metrics",
                    available_metrics,
                    default=['timestamp', 'inflow', 'hydropower', 'storage']
                )
                
                export_format = st.radio("Format", ["CSV", "Excel"], horizontal=True)
                
                if st.button("📥 Export Data", use_container_width=True):
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
            st.info("⏳ Collecting data...")
        
        st.markdown("---")
        st.markdown("### 📄 Report")
        
        if data_count > 5:
            if st.button("📄 Generate Report", use_container_width=True):
                with st.spinner("Generating report..."):
                    latest_data = st.session_state.data_history[-1]
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
                            label="📊 Download PDF",
                            data=pdf_buffer,
                            file_name=f"inga_ii_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    with col2:
                        st.download_button(
                            label="🌐 Download HTML",
                            data=html_report,
                            file_name=f"inga_ii_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                            mime="text/html",
                            use_container_width=True
                        )
        else:
            st.info("⏳ Need at least 5 records for a report.")
    
    # Display AI status
    if forecaster.is_trained:
        st.success("✅ **All AI models ready** | LSTM ✓ Classifier ✓ Anomaly ✓ RL ✓")
    else:
        st.warning("⚠️ Using default predictions until more data is available.")
    
    st.markdown("---")
    
    # Main loop
    simulator = RealTimeDataSimulator()
    placeholder = st.empty()
    
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
                       delta=f"{current_data['hydropower'] - 1280:.0f}")
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
            
            # Gauges
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
                with st.expander("🏭 View Full Dashboard"):
                    st.plotly_chart(
                        create_reservoir_dashboard(st.session_state.data_history), 
                        use_container_width=True,
                        key=f"dashboard_{frame}"
                    )
            
            # Anomaly
            if anomaly['is_anomaly']:
                st.warning(f"🚨 Anomaly: {anomaly['reason'] or 'Unusual pattern detected'}")
            else:
                st.success("✅ All systems normal")
            
            st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')} | Records: {len(st.session_state.data_history)}")
        
        time.sleep(update_freq)


if __name__ == "__main__":
    main()
