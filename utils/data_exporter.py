# utils/data_exporter.py
import pandas as pd
import streamlit as st
from io import BytesIO
import base64

def export_to_csv(data_history):
    """Export data history to CSV"""
    if not data_history:
        return None
    
    df = pd.DataFrame(data_history)
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="inga_ii_data_export.csv">📥 Download CSV</a>'
    return href

def export_to_excel(data_history):
    """Export data history to Excel"""
    if not data_history:
        return None
    
    df = pd.DataFrame(data_history)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='INGA_II_Data', index=False)
    
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="inga_ii_data_export.xlsx">📊 Download Excel</a>'
    return href

def get_graph_data(data_history, metric):
    """Extract data from a specific metric for download"""
    if not data_history:
        return None
    
    df = pd.DataFrame(data_history)
    if metric in df.columns:
        return df[['timestamp', metric]]
    return None
