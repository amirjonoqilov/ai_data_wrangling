import streamlit as st
import pandas as pd
import numpy as np
from utils.state import has_data
from utils.io import load_file


@st.cache_data(show_spinner=False)
def _overview_stats(df_hash: int, df: pd.DataFrame) -> dict:
    missing = df.isnull().sum()
    return {
        "n_missing_cells": int(missing.sum()),
        "n_duplicates": int(df.duplicated().sum()),
        "memory_kb": df.memory_usage(deep=True).sum() / 1024,
        "missing_df": pd.DataFrame({
            "Column": missing.index,
            "Missing Count": missing.values,
            "Missing %": (missing / len(df) * 100).round(2).values,
        }).query("`Missing Count` > 0").reset_index(drop=True),
        "dtype_df": pd.DataFrame({
            "Column": df.columns,
            "Type": [str(dt) for dt in df.dtypes],
            "Non-Null": df.notnull().sum().values,
            "Unique": df.nunique().values,
        }),
    }


def _load_into_session(df: pd.DataFrame, name: str):
    st.session_state.raw_df = df
    st.session_state.working_df = df.copy()
    st.session_state.filename = name
    st.session_state.transform_log = []
    st.session_state.history = []


def render():
    st.title("📂 Upload & Overview")

    col_upload, col_sample = st.columns([3, 2])
    with col_upload:
        uploaded = st.file_uploader(
            "Upload CSV, Excel, or JSON",
            type=["csv", "xlsx", "xls", "json"],
            label_visibility="collapsed",
        )
    with col_sample:
        sample_choice = st.selectbox(
            "Or load a sample dataset",
            ["— select a sample —", "retail_sales.csv", "employee_analytics.csv"],
            key="sample_select",
        )

    current_name = st.session_state.get("filename")

    if uploaded is not None and uploaded.name != current_name:
        with st.spinner("Loading…"):
            df, err = load_file(uploaded)
        if err:
            st.error(f"Load error: {err}")
        else:
            _load_into_session(df, uploaded.name)
            st.rerun()

    elif sample_choice != "— select a sample —" and sample_choice != current_name:
        try:
            df = pd.read_csv(f"sample_data/{sample_choice}")
            _load_into_session(df, sample_choice)
            st.rerun()
        except Exception as e:
            st.error(f"Could not load sample: {e}")

    if not has_data():
        st.info("👆 Upload a file or select a sample dataset to get started.")
        _render_sample_cards()
        return

    df = st.session_state.working_df
    stats = _overview_stats(id(df), df)

    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]}")
    c3.metric("Missing Cells", f"{stats['n_missing_cells']:,}")
    c4.metric("Duplicates", f"{stats['n_duplicates']:,}")
    c5.metric("Memory", f"{stats['memory_kb']:.1f} KB")

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Preview", "🔢 Column Info", "📊 Summary Stats", "❓ Missing Values"])

    with tab1:
        max_rows = min(500, len(df))
        n = st.slider("Rows to show", 5, max_rows, min(20, max_rows), key="prev_n")
        st.dataframe(df.head(n), use_container_width=True)

    with tab2:
        st.dataframe(stats["dtype_df"], use_container_width=True, hide_index=True)

    with tab3:
        try:
            desc = df.describe(include="all").T.reset_index().rename(columns={"index": "column"})
            st.dataframe(desc, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Could not compute summary statistics: {e}")

    with tab4:
        miss = stats["missing_df"]
        if miss.empty:
            st.success("🎉 No missing values in this dataset.")
        else:
            total_pct = stats["n_missing_cells"] / df.size * 100
            st.caption(f"Total missing: **{stats['n_missing_cells']:,}** cells ({total_pct:.2f}% of all data)")
            st.dataframe(miss, use_container_width=True, hide_index=True)


def _render_sample_cards():
    st.markdown("### 📦 Sample Datasets")
    c1, c2 = st.columns(2)
    with c1:
        st.info(
            "**retail_sales.csv** — 1,230 rows × 11 cols\n\n"
            "Retail orders with categories, regions, pricing, and discounts.\n\n"
            "**Dirty data:** inconsistent casing in `returned`, `$`-prefixed revenue strings, "
            "missing values across 5 columns, ~30 duplicate rows."
        )
    with c2:
        st.info(
            "**employee_analytics.csv** — 1,540 rows × 12 cols\n\n"
            "HR analytics: department, salary, performance, remote days.\n\n"
            "**Dirty data:** extreme salary outliers, mixed city casing, "
            "missing `salary`/`age`/`performance_score`, ~40 duplicate rows."
        )