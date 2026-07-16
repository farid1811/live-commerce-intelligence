import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src.utils.ui_utils import apply_global_styles, render_page_header, render_alert
from src.services.data_service import DataService
from src.services.prediction_service import PredictionService

apply_global_styles()

render_page_header("Prediction Analytics", "Interactive Plotly explorations of historical sessions, predictions, 3D regression surfaces, and residual errors.")

# Active Model & Data
model_repo = st.session_state.get("model_repo")
active_model = st.session_state.get("active_model")
df = st.session_state.get("dataset")

if df is None:
    st.warning("No dataset loaded. Please upload a dataset in the Dataset Manager.")
    st.stop()

if active_model is None:
    # Use fallback
    pred_service = PredictionService(model_repo)
    active_model = pred_service.get_fallback_model()

model_type = active_model.get("model_type", "Linear")
data_type = active_model.get("data_type", "All Data")
metrics = active_model.get("metrics", {})

st.markdown(f"Running on model: **{active_model.get('name')}** | R²: `{metrics.get('r2', 0.0):.4f}` | MAE: `{metrics.get('mae', 0.0):.2f}`")

# Get processed data for prediction analysis
df_proc = DataService.get_preprocessed_dataset(df, data_type)
X = df_proc[['Durasi_Jam', 'Penonton Aktif']].values
y = df_proc['Produk Terjual'].values

# Run predictions on all samples using active model
pred_service = PredictionService(model_repo)
y_preds = []
for i in range(len(df_proc)):
    res = pred_service.predict(X[i, 0], int(X[i, 1]), active_model)
    y_preds.append(res["raw_correct"])
y_preds = np.array(y_preds)
residuals = y - y_preds

# -----------------------------------------------------
# CHART 1: 3D REGRESSION SURFACE
# -----------------------------------------------------
st.markdown("### 3D Regression Surface Analysis")
st.markdown("Rotate and zoom to see the interaction between streaming duration, active viewers, and sales.")

# Generate grid for 3D surface
dur_grid = np.linspace(X[:, 0].min(), X[:, 0].max(), 25)
view_grid = np.linspace(X[:, 1].min(), X[:, 1].max(), 25)
dur_mesh, view_mesh = np.meshgrid(dur_grid, view_grid)

sales_mesh = np.zeros_like(dur_mesh)
for r in range(dur_mesh.shape[0]):
    for c in range(dur_mesh.shape[1]):
        res = pred_service.predict(dur_mesh[r, c], int(view_mesh[r, c]), active_model)
        sales_mesh[r, c] = res["raw_correct"]

# Plot 3D surface
fig_3d = go.Figure()

# Add actual scatter points
fig_3d.add_trace(go.Scatter3d(
    x=X[:, 0],
    y=X[:, 1],
    z=y,
    mode='markers',
    marker=dict(size=4, color='#1F2937', opacity=0.8),
    name='Actual Sessions'
))

# Add prediction surface
fig_3d.add_trace(go.Surface(
    x=dur_grid,
    y=view_grid,
    z=sales_mesh,
    colorscale='Blues',
    opacity=0.85,
    name='Regression Fit',
    showscale=False
))

fig_3d.update_layout(
    scene=dict(
        xaxis_title='Duration (Hours)',
        yaxis_title='Active Viewers',
        zaxis_title='Products Sold',
        xaxis=dict(gridcolor='#E5E7EB', backgroundcolor='#F9FAFB'),
        yaxis=dict(gridcolor='#E5E7EB', backgroundcolor='#F9FAFB'),
        zaxis=dict(gridcolor='#E5E7EB', backgroundcolor='#F9FAFB')
    ),
    template="simple_white",
    height=550,
    margin=dict(l=0, r=0, t=30, b=0)
)
st.plotly_chart(fig_3d, use_container_width=True)

# -----------------------------------------------------
# CHARTS 2 & 3: 2D ACTUAL VS PREDICTED & RESIDUALS
# -----------------------------------------------------
st.markdown("---")
st.markdown("### Accuracy & Residual Distribution")

c1, c2 = st.columns(2)

with c1:
    # Actual vs Predicted
    fig_avp = go.Figure()
    fig_avp.add_trace(go.Scatter(
        x=y, y=y_preds, mode='markers',
        marker=dict(color='#2563EB', size=6, opacity=0.7, line=dict(color='white', width=0.5)),
        name='Predicted'
    ))
    
    # Perfect fit line
    fig_avp.add_trace(go.Scatter(
        x=[y.min(), y.max()], y=[y.min(), y.max()],
        mode='lines', line=dict(color='#EF4444', width=1.5, dash='dash'),
        name='Perfect Target'
    ))
    
    fig_avp.update_layout(
        title="Actual vs. Predicted",
        xaxis_title="Actual Sales",
        yaxis_title="Predicted Sales",
        template="simple_white",
        height=350,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_avp, use_container_width=True)

with c2:
    # Residual Plot
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(
        x=y_preds, y=residuals, mode='markers',
        marker=dict(color='#0D9488', size=6, opacity=0.7, line=dict(color='white', width=0.5)),
        name='Residual'
    ))
    
    # Zero line
    fig_res.add_trace(go.Scatter(
        x=[y_preds.min(), y_preds.max()], y=[0, 0],
        mode='lines', line=dict(color='#EF4444', width=1.5, dash='dash'),
        name='Zero Error'
    ))
    
    fig_res.update_layout(
        title="Residual Plot Analysis",
        xaxis_title="Predicted Sales",
        yaxis_title="Residual (Actual - Pred)",
        template="simple_white",
        height=350,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_res, use_container_width=True)
