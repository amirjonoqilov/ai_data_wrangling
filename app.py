import streamlit as st

st.set_page_config(
    page_title="Data Wrangler",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #131722; }
.block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
h1 { font-size: 1.75rem !important; margin-bottom: 0.3rem !important; }
h2 { font-size: 1.25rem !important; }
h3 { font-size: 1.05rem !important; }
div[data-testid="metric-container"] {
    background: #1a1e2e;
    border: 1px solid #2a3050;
    border-radius: 8px;
    padding: 10px 14px;
}
div[data-testid="stDataFrame"] { border: 1px solid #2a3050; border-radius: 6px; }
.stTabs [data-baseweb="tab"] { background-color: #1a1e2e; border-radius: 6px 6px 0 0; padding: 6px 14px; }
.stTabs [aria-selected="true"] { background-color: #2a3050 !important; }
div[data-testid="stExpander"] { border: 1px solid #2a3050; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

from utils.state import init_state, has_data, reset_session
from pages import upload, cleaning, visualization, export

init_state()

PAGES = {
    "📂 Upload & Overview": upload,
    "🧹 Cleaning & Prep Studio": cleaning,
    "📊 Visualization Builder": visualization,
    "📤 Export & Report": export,
}

with st.sidebar:
    st.markdown("## 🔬 Data Wrangler")
    st.markdown("---")
    page_label = st.radio("nav", list(PAGES.keys()), label_visibility="collapsed")
    st.markdown("---")
    if has_data():
        df = st.session_state.working_df
        st.success(f"**{st.session_state.filename}**")
        st.caption(f"{df.shape[0]:,} rows · {df.shape[1]} cols · {len(st.session_state.transform_log)} steps")
        if st.button("🔄 Reset Session", use_container_width=True, type="secondary"):
            reset_session()
            st.rerun()
    else:
        st.info("No data loaded yet.")
    st.markdown("---")
    st.caption("v1.1 · Built with Streamlit")

PAGES[page_label].render()