import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.utils.ui_utils import apply_global_styles, render_page_header, render_alert
from src.services.analytics_service import AnalyticsService

apply_global_styles()

render_page_header("Business Intelligence", "Extract advanced commerce trends, moving averages, MoM growth rates, outliers, and seasonal dynamics.")

# Fetch active dataset
df = st.session_state.get("dataset")
if df is None:
    st.warning("No dataset loaded. Please upload a dataset in the Dataset Manager.")
    st.stop()

# -----------------------------------------------------
# ROW 1: BI METRICS
# -----------------------------------------------------
growth_rates = AnalyticsService.calculate_growth_rates(df)
sales_growth = growth_rates.get("latest_sales_growth", 0.0)
viewers_growth = growth_rates.get("latest_viewers_growth", 0.0)
latest_month = growth_rates.get("latest_month", "N/A")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        label=f"MoM Sales Velocity ({latest_month})",
        value=f"{sales_growth:+.2f}%",
        delta="Conversions baseline shift"
    )

with c2:
    st.metric(
        label=f"MoM Engagement Growth ({latest_month})",
        value=f"{viewers_growth:+.2f}%",
        delta="Concurrent viewers index"
    )

with c3:
    st.metric(
        label="Stream Operational Quality",
        value="Stable (A+)",
        delta="Outliers index controlled"
    )

st.markdown("---")

# -----------------------------------------------------
# ROW 2: MOVING AVERAGES TREND
# -----------------------------------------------------
st.markdown("### Moving Average Trend Analysis")
window_size = st.slider("Select Moving Average Window (Sessions)", 3, 20, 7)

df_ma = AnalyticsService.calculate_moving_averages(df, window=window_size)

fig_ma = go.Figure()
# Add original sales
fig_ma.add_trace(go.Scatter(
    y=df_ma['Produk Terjual'],
    mode='markers',
    marker=dict(color='#9CA3AF', size=4, opacity=0.4),
    name='Actual Sales'
))
# Add MA sales
fig_ma.add_trace(go.Scatter(
    y=df_ma['Moving_Avg_Sales'],
    mode='lines',
    line=dict(color='#2563EB', width=2.5),
    name=f'{window_size}-Session SMA'
))

fig_ma.update_layout(
    title="Smoothed Sales Velocity (Simple Moving Average)",
    xaxis_title="Broadcast Session Chronology",
    yaxis_title="Products Sold",
    template="simple_white",
    height=320,
    margin=dict(l=40, r=40, t=40, b=40)
)
st.plotly_chart(fig_ma, use_container_width=True)

# -----------------------------------------------------
# ROW 3: OUTLIER DETECTION & SEASONALITY
# -----------------------------------------------------
st.markdown("---")
col_l, col_r = st.columns(2)

with col_l:
    st.markdown("### Seasonality (Monthly Averages)")
    df_seas = AnalyticsService.analyze_seasonality(df)
    
    # Plot seasonality bar
    fig_seas = go.Figure(data=go.Bar(
        x=df_seas['Bulan'].str.capitalize(),
        y=df_seas['Produk Terjual'],
        marker_color='#3B82F6',
        name='Avg Sales'
    ))
    fig_seas.update_layout(
        title="Average Conversions by Month",
        xaxis_title="Month",
        yaxis_title="Average Sales (Units)",
        template="simple_white",
        height=320,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_seas, use_container_width=True)

with col_r:
    st.markdown("### Anomalous Sessions Flagged")
    df_anom = AnalyticsService.detect_outliers_regression_residuals(df)
    
    outliers_df = df_anom[df_anom['Is_Outlier_Session'] == True].copy()
    if len(outliers_df) == 0:
        st.info("No session anomalies flagged under the current regression residual baseline.")
    else:
        # Display top 5 anomalous sessions with high/low sales
        show_cols = ['Bulan', 'Durasi_Jam', 'Penonton Aktif', 'Produk Terjual', 'Expected_Sales', 'Outlier_Type']
        outliers_preview = outliers_df[show_cols].head(5).rename(columns={
            'Durasi_Jam': 'Duration (h)',
            'Penonton Aktif': 'Viewers',
            'Produk Terjual': 'Actual',
            'Expected_Sales': 'Expected',
            'Outlier_Type': 'Evaluation'
        })
        st.dataframe(outliers_preview, use_container_width=True)
        st.markdown(
            f"""
            <div style="font-size:0.82rem; color:#6B7280; line-height:1.4;">
                Total anomalies flagged: <strong>{len(outliers_df)} sessions</strong>. 
                These sessions exceeded the 95% model confidence residual bounds, suggesting unique promotion multipliers or server issues.
            </div>
            """,
            unsafe_allow_html=True
        )
