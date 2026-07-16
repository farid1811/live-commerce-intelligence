import streamlit as st
from src.utils.ui_utils import apply_global_styles
from src.repositories.dataset_repository import DatasetRepository
from src.repositories.model_repository import ModelRepository

# Initialize global page configs and style injections
apply_global_styles()

# -----------------------------------------------------
# INITIALIZE GLOBAL SESSION STATE
# -----------------------------------------------------
if "dataset" not in st.session_state:
    try:
        repo = DatasetRepository()
        st.session_state["dataset"] = repo.load_data()
    except Exception as e:
        st.error(f"Error loading default dataset: {e}")
        st.session_state["dataset"] = None

if "model_repo" not in st.session_state:
    st.session_state["model_repo"] = ModelRepository()

if "active_model" not in st.session_state:
    st.session_state["active_model"] = st.session_state["model_repo"].get_active_model()

# -----------------------------------------------------
# STREAMLIT PAGE CONFIG & NAVIGATION
# -----------------------------------------------------
st.sidebar.markdown(
    """
    <div style="padding: 10px 0px; text-align: center; border-bottom: 1px solid #F3F4F6; margin-bottom: 20px;">
        <h2 style="font-size: 1.5rem; font-weight: 800; color: #2563EB; margin: 0; letter-spacing: -0.04em;">CONVORA</h2>
        <p style="font-size: 0.75rem; font-weight: 600; color: #6B7280; margin: 2px 0 0 0; text-transform: uppercase; letter-spacing: 0.05em;">Live Commerce BI</p>
    </div>
    """,
    unsafe_allow_html=True
)

dashboard = st.Page("pages/01_dashboard.py", title="Executive Dashboard", icon="📊")
prediction = st.Page("pages/02_prediction_center.py", title="Prediction Center", icon="🔮")
training = st.Page("pages/03_training_center.py", title="Training Center", icon="⚙️")
dataset = st.Page("pages/04_dataset_manager.py", title="Dataset Manager", icon="📁")
analytics = st.Page("pages/05_analytics.py", title="Analytics", icon="📈")
bi = st.Page("pages/06_business_intelligence.py", title="Business Intelligence", icon="💡")
reports = st.Page("pages/07_reports.py", title="Reports Center", icon="📄")
experiments = st.Page("pages/08_experiments.py", title="Experiments Center", icon="🧪")
settings = st.Page("pages/09_settings.py", title="System Settings", icon="🛠️")

# Build navigation menu structure
pg = st.navigation({
    "Monitor": [dashboard],
    "Analytics & BI": [analytics, bi],
    "ML Engine": [prediction, training, experiments],
    "Data & Admin": [dataset, reports, settings]
})

# Run the active page
pg.run()
