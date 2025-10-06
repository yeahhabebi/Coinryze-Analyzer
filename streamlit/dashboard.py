# streamlit/dashboard.py
import os, io
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from urllib.parse import urljoin

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")  # in Render set to your backend URL
CSV_LOCAL = os.getenv("CSV_PATH", "/app/frontend/coinryze_history.csv")
S3_BASE_URL = os.getenv("S3_BASE_URL", "")

st.set_page_config(page_title="CoinryzeAnalyzer", layout="wide")
st.title("🎯 CoinryzeAnalyzer — Live Dashboard")

st.sidebar.header("Data Source")
data_mode = st.sidebar.radio("Choose data source", ("Sample Data", "Uploaded CSV", "Live CSV"))
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
refresh = st.sidebar.button("Refresh")

def sample_df(n=120):
    import random
    rows = []
    now = datetime.utcnow()
    for i in range(n):
        num = random.randint(0,49)
        color = "Green" if num%2==0 else ("Purple" if num%5==0 else "Red")
        rows.append({
            "issue_id": f"S{i+1}",
            "timestamp": (now - pd.Timedelta(minutes=(n-i))).strftime("%Y-%m-%d %H:%M:%S"),
            "number": num,
            "color": color,
            "size": "Big" if num>=25 else "Small",
            "odd_even": "Even" if num%2==0 else "Odd"
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=20)
def load_df():
    if data_mode == "Sample Data":
        return sample_df()
    if data_mode == "Uploaded CSV":
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file)
        return pd.DataFrame()
    # Live CSV: try S3, then local
    if S3_BASE_URL:
        try:
            url = urljoin(S3_BASE_URL, os.path.basename(CSV_LOCAL))
            r = requests.get(url, timeout=10); r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text))
        except Exception:
            pass
    if os.path.exists(CSV_LOCAL):
        try:
            return pd.read_csv(CSV_LOCAL)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

if refresh:
    st.experimental_rerun()

df = load_df()

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Recent Draws")
    if df.empty:
        st.info("No data to display")
    else:
        df_plot = df.copy()
        df_plot['ts'] = pd.to_datetime(df_plot['timestamp'])
        df_plot = df_plot.sort_values('ts')
        fig = px.line(df_plot, x='ts', y='number', title='Number over Time', markers=True)
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.pie(df_plot, names='color', title='Color Distribution', color_discrete_map={'Red':'#ff4d4d','Green':'#66ff66','Purple':'#cc99ff'})
        st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.subheader("Prediction")
    last_n = st.number_input("Use last N numbers for prediction", min_value=3, max_value=100, value=30)
    if st.button("Predict Next"):
        if df.empty:
            st.warning("No data available")
        else:
            seq = df['number'].astype(int).tolist()[-last_n:]
            try:
                resp = requests.post(urljoin(BACKEND_URL, "/predict"), json={"last_numbers": seq}, timeout=10)
                resp.raise_for_status()
                pred = resp.json()
                st.metric("Predicted Number", pred['predicted_number'])
                st.write("Color:", pred['predicted_color'])
                st.write("Size:", pred['predicted_size'])
                st.write("Odd/Even:", pred['odd_even'])
                st.write("Confidence (est):", pred.get('confidence', 0.0))
            except Exception as e:
                st.error("Prediction call failed: " + str(e))

st.markdown("---")
st.subheader("Live results table")
if not df.empty:
    cell_colors = []
    for _, r in df.iterrows():
        row_colors = []
        color_val = str(r.get('color','')).lower()
        for c in df.columns:
            if c == 'color':
                if 'red' in color_val:
                    row_colors.append('#ffcccc')
                elif 'green' in color_val:
                    row_colors.append('#ccffcc')
                elif 'purple' in color_val:
                    row_colors.append('#e6ccff')
                else:
                    row_colors.append('white')
            else:
                row_colors.append('white')
        cell_colors.append(row_colors)
    table = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns), fill_color='#4b0082', font=dict(color='white', size=12)),
        cells=dict(values=[df[c].astype(str).tolist() for c in df.columns],
                   fill_color=list(map(list, zip(*cell_colors))),
                   align='left'))
    ])
    st.plotly_chart(table, use_container_width=True)
else:
    st.info("No table data")
