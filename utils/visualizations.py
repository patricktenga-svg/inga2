# utils/visualizations.py
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

def create_gauge_chart(value, min_val, max_val, title, unit, threshold_good=None, threshold_bad=None):
    """Create a gauge chart with color-coded zones"""
    if threshold_good is None:
        threshold_good = min_val + 0.66 * (max_val - min_val)
    if threshold_bad is None:
        threshold_bad = min_val + 0.33 * (max_val - min_val)
    
    if value >= threshold_good:
        color = '#2ecc71'
    elif value >= threshold_bad:
        color = '#f39c12'
    else:
        color = '#e74c3c'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        delta={'reference': (min_val + max_val) / 2},
        title={'text': title, 'font': {'size': 14}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [min_val, threshold_bad], 'color': '#ffcccc'},
                {'range': [threshold_bad, threshold_good], 'color': '#fff3cc'},
                {'range': [threshold_good, max_val], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        },
        number={'suffix': f" {unit}", 'font': {'size': 24, 'color': color}}
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def create_attention_heatmap(attention_weights, lookback_days=30):
    """Create attention mechanism heatmap visualization"""
    fig = go.Figure(data=go.Heatmap(
        z=[attention_weights],
        colorscale='Viridis',
        zmin=0,
        zmax=1,
        colorbar=dict(title="Attention Weight"),
        y=['Attention'],
        x=list(range(1, lookback_days + 1)),
        hovertemplate='Day %{x}<br>Attention: %{z:.3f}<extra></extra>'
    ))
    fig.update_layout(
        title="🎯 LSTM Attention Weights (Last 30 days)",
        xaxis_title="Days before prediction",
        yaxis_title="",
        height=200,
        margin=dict(l=40, r=20, t=40, b=20)
    )
    return fig


def create_forecast_chart(historical_inflows, forecast_values, confidence_interval=0.2):
    """Create enhanced forecast chart with confidence bands"""
    hist_vals = historical_inflows[-30:] if len(historical_inflows) >= 30 else historical_inflows
    hist_days = list(range(-len(hist_vals), 0))
    forecast_days = list(range(1, len(forecast_values) + 1))
    
    upper_bound = [v * (1 + confidence_interval) for v in forecast_values]
    lower_bound = [v * (1 - confidence_interval) for v in forecast_values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hist_days, y=hist_vals,
        mode='lines+markers',
        name='Historical Inflow',
        line=dict(color='#3498db', width=2),
        marker=dict(size=6, color='#3498db')
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_days, y=forecast_values,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#e74c3c', width=3, dash='dot'),
        marker=dict(size=8, color='#e74c3c', symbol='diamond')
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_days + forecast_days[::-1],
        y=upper_bound + lower_bound[::-1],
        fill='toself',
        fillcolor='rgba(231, 76, 60, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name=f'{confidence_interval*100}% Confidence Interval'
    ))
    
    fig.add_hline(y=42000, line_dash="dash", line_color="green", 
                  annotation_text="Normal Inflow (42,000 m³/s)", 
                  annotation_position="bottom right")
    
    fig.update_layout(
        title="📈 7-Day Inflow Forecast with Confidence Bands",
        xaxis_title="Days (0 = today)",
        yaxis_title="Inflow (m³/s)",
        height=400,
        hovermode='x unified',
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig


def create_training_history_plot(training_losses):
    """Create training history visualization"""
    # Correction: utiliser des arguments nommés correctement
    fig = make_subplots(
        rows=1, 
        cols=2, 
        subplot_titles=('Loss Evolution', 'Loss (Log Scale)')
    )
    
    epochs = list(range(1, len(training_losses['train']) + 1))
    
    # Linear scale
    fig.add_trace(
        go.Scatter(x=epochs, y=training_losses['train'], mode='lines', name='Train Loss', line=dict(color='#3498db', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=training_losses['val'], mode='lines', name='Validation Loss', line=dict(color='#e74c3c', width=2)),
        row=1, col=1
    )
    
    # Log scale
    fig.add_trace(
        go.Scatter(x=epochs, y=training_losses['train'], mode='lines', name='Train Loss', line=dict(color='#3498db', width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=training_losses['val'], mode='lines', name='Validation Loss', line=dict(color='#e74c3c', width=2)),
        row=1, col=2
    )
    
    fig.update_yaxes(type="log", title="Loss (log scale)", row=1, col=2)
    fig.update_xaxes(title="Epoch", row=1, col=1)
    fig.update_xaxes(title="Epoch", row=1, col=2)
    fig.update_yaxes(title="Loss", row=1, col=1)
    
    fig.update_layout(height=400, title_text="📊 LSTM Training History", showlegend=True)
    
    return fig


def create_reservoir_dashboard(data_history):
    """Create comprehensive reservoir dashboard with multiple charts"""
    df = pd.DataFrame(data_history[-100:])
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Inflow & Hydropower', 'Reservoir Storage', 
                       'Turbine vs Irrigation', 'Head Evolution',
                       'Hydropower Efficiency', 'Storage-Flow Correlation'),
        specs=[[{"secondary_y": True}, {}], [{}, {}], [{}, {}]]
    )
    
    # 1. Inflow & Hydropower (dual axis)
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['inflow'], name='Inflow (m³/s)', line=dict(color='#3498db', width=2)),
        row=1, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['hydropower'], name='Hydropower (MW)', line=dict(color='#e74c3c', width=2, dash='dot')),
        row=1, col=1, secondary_y=True
    )
    
    # 2. Reservoir Storage
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['storage']/1e6, name='Storage (Mm³)', fill='tozeroy', line=dict(color='#2ecc71', width=2)),
        row=1, col=2
    )
    
    # 3. Turbine vs Irrigation
    fig.add_trace(
        go.Bar(x=df['timestamp'], y=df['turbine'], name='Turbine Release', marker_color='#f39c12'),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=df['timestamp'], y=df['irrigation'], name='Irrigation Release', marker_color='#1abc9c'),
        row=2, col=1
    )
    
    # 4. Head Evolution
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['head'], name='Head (m)', fill='tozeroy', line=dict(color='#9b59b6', width=2)),
        row=2, col=2
    )
    
    # 5. Hydropower Efficiency
    efficiency = df['hydropower'] / (df['turbine'] + 1) * 10
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=efficiency, name='Efficiency (MW per 100 m³/s)', line=dict(color='#e67e22', width=2)),
        row=3, col=1
    )
    
    # 6. Storage-Flow Correlation
    fig.add_trace(
        go.Scatter(x=df['storage']/1e6, y=df['inflow'], mode='markers', name='Storage vs Inflow',
                  marker=dict(size=8, color=df['hydropower'], colorscale='Viridis', showscale=True)),
        row=3, col=2
    )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="🏭 INGA II Reservoir Dashboard",
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Flow (m³/s)", secondary_y=False, row=1, col=1)
    fig.update_yaxes(title_text="Power (MW)", secondary_y=True, row=1, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_xaxes(title_text="Storage (Mm³)", row=3, col=2)
    
    return fig


def create_scenario_radar(scenario_probs):
    """Create radar chart for scenario probabilities"""
    categories = list(scenario_probs.keys())
    values = list(scenario_probs.values())
    
    categories.append(categories[0])
    values.append(values[0])
    
    colors = []
    for i, val in enumerate(values[:-1]):
        if val == max(values[:-1]):
            colors.append('#2ecc71')
        else:
            colors.append('#95a5a6')
    colors.append(colors[0])
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        marker=dict(color=colors),
        line=dict(color='#2c3e50', width=2)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickformat='.0%'),
            angularaxis=dict(tickfont=dict(size=12, weight='bold'))
        ),
        title="🎯 Scenario Classification Probabilities",
        height=350,
        showlegend=False
    )
    
    return fig


def create_performance_metrics_card(metrics):
    """Create a metrics card with multiple KPIs"""
    fig = go.Figure()
    
    n_metrics = len(metrics)
    width = 1.0 / n_metrics
    
    for i, (name, value) in enumerate(metrics.items()):
        fig.add_trace(go.Indicator(
            mode="number",
            value=value,
            title={"text": name},
            domain={"x": [width * i, width * (i + 1)], "y": [0, 1]}
        ))
    
    fig.update_layout(height=150, margin=dict(l=0, r=0, t=0, b=0))
    return fig


def create_comparison_bar_chart(categories, values_normal, values_dry, values_very_dry, title, ylabel):
    """Create comparison bar chart for different scenarios"""
    fig = go.Figure(data=[
        go.Bar(name='Normal', x=categories, y=values_normal, marker_color='#2ecc71'),
        go.Bar(name='Dry', x=categories, y=values_dry, marker_color='#f39c12'),
        go.Bar(name='Very Dry', x=categories, y=values_very_dry, marker_color='#e74c3c')
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title=ylabel,
        barmode='group',
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig


def create_correlation_matrix(df, columns):
    """Create correlation matrix heatmap"""
    corr_matrix = df[columns].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title="📊 Correlation Matrix",
        height=500,
        width=600
    )
    
    return fig
