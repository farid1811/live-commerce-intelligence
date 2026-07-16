import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.utils.ui_utils import apply_global_styles, render_page_header, render_alert
from src.repositories.dataset_repository import DatasetRepository
from src.services.data_service import DataService

apply_global_styles()

render_page_header("Dataset Manager", "Upload, inspect, and analyze the quality of live streaming commerce data.")

# Setup repository
dataset_repo = DatasetRepository()

# -----------------------------------------------------
# DATASET UPLOAD ZONE
# -----------------------------------------------------
st.markdown("### Upload Custom Session Dataset")
uploaded_file = st.file_uploader("Upload CSV or Excel spreadsheet (.xlsx, .xls, .csv)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Save uploaded file temporarily to read it
        import os
        from src.utils.config import DATA_DIR
        temp_path = os.path.join(DATA_DIR, f"temp_{uploaded_file.name}")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load and validate
        temp_repo = DatasetRepository(temp_path)
        temp_df = temp_repo.load_data()
        
        is_valid, msg = temp_repo.validate_schema(temp_df)
        if is_valid:
            # Overwrite session dataset
            st.session_state["dataset"] = temp_df
            # Save as target dataset
            target_path = os.path.join(DATA_DIR, "Data Skripsi uploaded.xlsx")
            temp_repo.save_data(temp_df, target_path)
            # Update default file in repository
            st.session_state["dataset_path"] = target_path
            
            # Clean up temp
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            render_alert(f"Successfully loaded and verified dataset: **{uploaded_file.name}** ({len(temp_df)} sessions)", type="success")
        else:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            st.error(f"Schema Validation Error: {msg}")
    except Exception as e:
        st.error(f"Error parsing uploaded file: {e}")

# Fetch active dataset from session state
df = st.session_state.get("dataset")
if df is None:
    st.info("No active dataset loaded. Please upload a dataset file.")
    st.stop()

# -----------------------------------------------------
# DATASET PREVIEW & STATS SECTION
# -----------------------------------------------------
tab_preview, tab_stats, tab_correlations = st.tabs(["📁 Dataset Preview", "📊 Quality Statistics", "📐 Correlation Matrix"])

with tab_preview:
    st.markdown("#### Live Session Data Preview (10 first records)")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown(f"**Total Record Count:** `{len(df)} sessions` | **Columns:** `{', '.join(df.columns)}`")

with tab_stats:
    st.markdown("#### Baseline Summary Statistics")
    stats_df = DataService.get_summary_statistics(df)
    st.table(stats_df)
    
    st.markdown(
        """
        > [!NOTE]
        > **Missing Values** and **Outliers** are auto-flagged using the standard interquartile range (IQR) threshold. 
        > Standard preprocessing is executed inside the Training Center using these statistics.
        """
    )

with tab_correlations:
    st.markdown("#### Pearson Correlation Coefficient")
    corr_df = DataService.get_correlation_matrix(df)
    
    # Render Plotly Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.index,
        colorscale=[[0, '#EFF6FF'], [0.5, '#93C5FD'], [1, '#2563EB']],
        zmin=-1.0, zmax=1.0,
        text=np.round(corr_df.values, 3),
        texttemplate="%{text}",
        showscale=True
    ))
    
    fig.update_layout(
        title="Feature Interaction Heatmap",
        template="simple_white",
        height=350,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(
        """
        **Correlation Insights:**
        - Positive values near 1 indicate strong proportional relationships.
        - The target **Produk Terjual** correlation coefficient shows the direct proportional impact of streaming hours vs concurrent engagement audience.
        """
    )
