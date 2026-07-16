import streamlit as st
from src.utils.ui_utils import apply_global_styles, render_page_header, render_alert
from src.services.reporting_service import ReportingService
from src.services.prediction_service import PredictionService

apply_global_styles()

render_page_header("Reports Center", "Generate, preview, and download professional executive intelligence reports in multiple formats.")

# Fetch active dataset & model
df = st.session_state.get("dataset")
active_model = st.session_state.get("active_model")
model_repo = st.session_state.get("model_repo")

if df is None:
    st.warning("No dataset loaded. Please upload a dataset in the Dataset Manager.")
    st.stop()

if active_model is None:
    # Use fallback
    pred_service = PredictionService(model_repo)
    active_model = pred_service.get_fallback_model()

# -----------------------------------------------------
# ROW 1: GENERATE & DOWNLOAD BUTTONS
# -----------------------------------------------------
st.markdown("### Export Executive Deliverables")

# Generate files
excel_bytes = ReportingService.generate_excel_report(df, active_model)
md_report = ReportingService.generate_markdown_report(df, active_model)
png_chart_bytes = ReportingService.generate_chart_png(df, active_model)
pdf_bytes = ReportingService.generate_pdf_report(df, active_model)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_bytes,
        file_name="convora_executive_dashboard.pdf",
        mime="application/pdf",
        use_container_width=True
    )

with c2:
    st.download_button(
        label="📊 Download Excel Sheets",
        data=excel_bytes,
        file_name="convora_bi_dataset.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

with c3:
    st.download_button(
        label="📈 Download PNG Charts",
        data=png_chart_bytes,
        file_name="convora_model_performance.png",
        mime="image/png",
        use_container_width=True
    )

with c4:
    st.download_button(
        label="📝 Download Markdown",
        data=md_report,
        file_name="convora_summary_report.md",
        mime="text/markdown",
        use_container_width=True
    )

st.markdown("---")

# -----------------------------------------------------
# ROW 2: REPORT PREVIEW
# -----------------------------------------------------
st.markdown("### Executive Report Live Preview")

st.markdown(
    """
    <div style="background-color: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05); max-height: 600px; overflow-y: auto;">
    """,
    unsafe_allow_html=True
)

st.markdown(md_report)

st.markdown(
    """
    </div>
    """,
    unsafe_allow_html=True
)
