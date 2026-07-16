import os
import streamlit as st

def apply_global_styles():
    """Applies the page layout config and injects custom CSS from styles.css."""
    # Run set_page_config if not already set (Streamlit will warn if set multiple times, but this is safe if catch error)
    try:
        st.set_page_config(
            page_title="Convora | AI Live Commerce Intelligence",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except st.errors.StreamlitAPIException:
        # Page config already set, which is fine
        pass

    # Read and inject CSS
    css_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "styles.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def render_page_header(title: str, subtitle: str):
    """Renders a standard premium header for each SaaS page module."""
    st.markdown(
        f"""
        <div style="margin-bottom: 2rem;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 0.85rem; font-weight: 700; background-color: #2563EB; color: #FFFFFF; padding: 2px 8px; border-radius: 4px; letter-spacing: 0.05em; text-transform: uppercase;">SaaS Suite</span>
                <span style="font-size: 0.85rem; font-weight: 500; color: #6B7280;">Convora Platform</span>
            </div>
            <h1 style="font-size: 2.25rem; font-weight: 800; color: #111827; margin-top: 4px; margin-bottom: 4px; letter-spacing: -0.03em;">{title}</h1>
            <p style="font-size: 1rem; color: #4B5563; margin-top: 0; font-weight: 400;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_alert(text: str, type="info"):
    """Renders custom colored banner panels."""
    if type == "success":
        st.markdown(f'<div class="success-panel">✓ {text}</div>', unsafe_allow_html=True)
    elif type == "warning":
        st.markdown(f'<div class="warning-panel">⚠ {text}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-panel">ℹ {text}</div>', unsafe_allow_html=True)
