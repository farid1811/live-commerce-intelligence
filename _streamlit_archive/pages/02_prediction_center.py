import streamlit as st
import numpy as np
from src.utils.ui_utils import apply_global_styles, render_page_header, render_alert
from src.services.prediction_service import PredictionService

apply_global_styles()

render_page_header("Prediction Center", "Simulate live commerce broadcasts and project sales results based on duration and engagement.")

# Setup repositories & service
model_repo = st.session_state.get("model_repo")
pred_service = PredictionService(model_repo)

# List available models
available_models = model_repo.list_models()
if not available_models:
    # Use fallback
    st.info("No registered models found. System is currently running on default thesis model parameters.")
    model_choices = ["Default Thesis Model"]
    models_dict = {"Default Thesis Model": pred_service.get_fallback_model()}
else:
    model_choices = [m["name"] for m in available_models]
    models_dict = {m["name"]: model_repo.load_model(m["filename"]) for m in available_models}

# Find active model name to set as default select index
active_model_meta = model_repo.get_active_model()
active_name = active_model_meta.get("name") if active_model_meta else "Default Thesis Model"
default_idx = model_choices.index(active_name) if active_name in model_choices else 0

# Sidebar inputs
st.sidebar.header("🕹️ Simulation Inputs")

selected_model_name = st.sidebar.selectbox("Predictive Model", model_choices, index=default_idx)
selected_model = models_dict.get(selected_model_name)

# Slider limits based on actual data describe
durasi_input = st.sidebar.slider("Live Duration (Hours)", 0.5, 24.0, 8.0, step=0.5)
penonton_input = st.sidebar.slider("Active Viewers (Concurrent Peaks)", 1, 500, 100, step=5)

# Calculate Prediction
pred_res = pred_service.predict(durasi_input, penonton_input, selected_model)

# -----------------------------------------------------
# UI PANELS & METRICS DISPLAY
# -----------------------------------------------------
c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        label="Predicted Sales",
        value=f"{pred_res['prediction_correct']} units",
        delta=f"Unscaled thesis: {pred_res['prediction_thesis']}"
    )

with c2:
    st.metric(
        label="Prediction Interval",
        value=f"{pred_res['lower_bound']} - {pred_res['upper_bound']} units",
        delta="95% Confidence range"
    )

with c3:
    st.metric(
        label="Simulated Configuration",
        value=f"{durasi_input} hrs / {penonton_input} viewers",
        delta=f"Risk level: {pred_res['risk_level']}"
    )

# Business Insights & Recommendation Card
st.markdown("### AI Strategic Interpretation")

# Styling warning/success based on risk
if pred_res['risk_level'] == "Low":
    render_alert(f"**Interpretation:** {pred_res['interpretation']}", type="success")
elif pred_res['risk_level'] == "Medium":
    render_alert(f"**Interpretation:** {pred_res['interpretation']}", type="info")
else:
    render_alert(f"**Interpretation:** {pred_res['interpretation']}", type="warning")

st.markdown(
    f"""
    <div style="background-color: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 1.25rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
        <h4 style="margin-top:0; color:#111827; font-weight:700;">📋 Recommendation Summary</h4>
        <p style="font-size:0.95rem; color:#4B5563; line-height:1.5; margin-bottom:0;">
            {pred_res['recommendation']}
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Active Model Details Card
st.markdown("### Model Formula & Parameter Weights")

metrics = pred_res["metrics"]
r2_val = metrics.get("r2", 0.5)
mae_val = metrics.get("mae", 10.0)
rmse_val = metrics.get("rmse", 14.0)
mape_val = metrics.get("mape", 66.0)

# Unpack weights based on model type
theta_thesis = selected_model.get("theta_thesis")
bias_thesis = selected_model.get("bias_thesis")
model_type = selected_model.get("model_type", "Linear")

st.markdown(
    f"""
    <div style="background-color:#F9FAFB; border: 1px solid #E5E7EB; border-radius:12px; padding:1.25rem;">
        <div style="font-size:0.75rem; color:#6B7280; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:4px;">Mathematical Formula</div>
        <code style="font-size:1.05rem; color:#2563EB; font-weight:700; background-color:#EFF6FF; padding:4px 10px; border-radius:6px; display:block; margin-bottom:15px; border:1px solid #BFDBFE;">
            {pred_res['equation']}
        </code>
        <div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:16px;">
            <div style="min-width:120px;">
                <span style="font-size:0.8rem; color:#6B7280;">Thesis Intercept (bias)</span>
                <div style="font-size:1.1rem; font-weight:600; color:#1F2937;">{bias_thesis:.6f}</div>
            </div>
            <div style="min-width:120px;">
                <span style="font-size:0.8rem; color:#6B7280;">B1 (Duration coeff)</span>
                <div style="font-size:1.1rem; font-weight:600; color:#1F2937;">{theta_thesis[0]:.6f}</div>
            </div>
            <div style="min-width:120px;">
                <span style="font-size:0.8rem; color:#6B7280;">B2 (Viewer coeff)</span>
                <div style="font-size:1.1rem; font-weight:600; color:#1F2937;">
                    {theta_thesis[1]:.6f}
                </div>
            </div>
            <div style="min-width:120px;">
                <span style="font-size:0.8rem; color:#6B7280;">R² Score</span>
                <div style="font-size:1.1rem; font-weight:600; color:#10B981;">{r2_val:.4f}</div>
            </div>
            <div style="min-width:120px;">
                <span style="font-size:0.8rem; color:#6B7280;">MAE</span>
                <div style="font-size:1.1rem; font-weight:600; color:#1F2937;">{mae_val:.2f}</div>
            </div>
            <div style="min-width:120px;">
                <span style="font-size:0.8rem; color:#6B7280;">RMSE</span>
                <div style="font-size:1.1rem; font-weight:600; color:#1F2937;">{rmse_val:.2f}</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
