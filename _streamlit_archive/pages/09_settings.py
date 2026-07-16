import os
import streamlit as st
from src.utils.ui_utils import apply_global_styles, render_page_header, render_alert
from src.utils.config import MODEL_DIR, DEFAULT_DATASET

apply_global_styles()

render_page_header("System Settings", "Configure application preferences, active models, paths, and clear registry metadata.")

model_repo = st.session_state.get("model_repo")

# -----------------------------------------------------
# ROW 1: REGISTRY FILE SYSTEM INFORMATION
# -----------------------------------------------------
st.markdown("### System Paths & Infrastructure")

st.markdown(
    f"""
    <div style="background-color: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 1.25rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 2rem;">
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #F3F4F6;">
                <td style="padding: 10px 0; font-weight: 600; color: #4B5563; width: 30%;">Registry Storage Location</td>
                <td style="padding: 10px 0; font-family: monospace; color: #111827;">{MODEL_DIR}</td>
            </tr>
            <tr style="border-bottom: 1px solid #F3F4F6;">
                <td style="padding: 10px 0; font-weight: 600; color: #4B5563;">Default Dataset File</td>
                <td style="padding: 10px 0; font-family: monospace; color: #111827;">{DEFAULT_DATASET}</td>
            </tr>
            <tr style="border-bottom: 1px solid #F3F4F6;">
                <td style="padding: 10px 0; font-weight: 600; color: #4B5563;">Registered Models Count</td>
                <td style="padding: 10px 0; color: #111827; font-weight: 700;">{len(model_repo.list_models())} models</td>
            </tr>
            <tr>
                <td style="padding: 10px 0; font-weight: 600; color: #4B5563;">System Status</td>
                <td style="padding: 10px 0; color: #10B981; font-weight: 700;">🟢 Online / Healthy</td>
            </tr>
        </table>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------
# ROW 2: SYSTEM MANAGEMENT & CLEANSING
# -----------------------------------------------------
st.markdown("### System Administration")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Model Registry Actions")
    st.write("Clean the models_registry folder to delete all trained weights and reset to system fallback parameters.")
    
    confirm_clear = st.checkbox("I confirm that I want to delete all custom models.")
    clear_btn = st.button("🔴 Purge Custom Model Registry", disabled=not confirm_clear)
    
    if clear_btn:
        purged = 0
        for file in os.listdir(MODEL_DIR):
            if file.endswith(".pkl"):
                try:
                    os.remove(os.path.join(MODEL_DIR, file))
                    purged += 1
                except Exception:
                    pass
        
        # Reset state
        st.session_state["active_model"] = None
        st.session_state["active_model"] = model_repo.get_active_model()
        
        render_alert(f"Purged {purged} custom models from registry. System fell back to default parameters.", type="warning")
        time_sleep_wait = 1.5
        import time
        time.sleep(time_sleep_wait)
        st.rerun()

with col2:
    st.markdown("#### Thesis Methodology Settings")
    st.write("Ensure positive weights constraints are strictly active across all training loops.")
    st.info(
        "✓ SGD Positive Constraints option is **Locked** to **Active** to prevent methodology violations "
        "relative to independent variables (Live Duration, Active Viewers) in the thesis."
    )
