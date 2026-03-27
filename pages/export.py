import streamlit as st
import pandas as pd
from datetime import datetime
from utils.state import has_data
from utils.io import df_to_csv_bytes, df_to_excel_bytes, build_report, build_json_recipe, build_python_script


def render():
    st.title("📤 Export & Report")

    if not has_data():
        st.warning("No data loaded. Go to **📂 Upload & Overview** first.")
        return

    df = st.session_state.working_df
    raw_df = st.session_state.raw_df
    log = st.session_state.transform_log
    filename = st.session_state.filename or "dataset"
    stem = filename.rsplit(".", 1)[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original Rows", f"{raw_df.shape[0]:,}")
    c2.metric("Current Rows", f"{df.shape[0]:,}")
    c3.metric("Current Cols", f"{df.shape[1]}")
    c4.metric("Steps Applied", f"{len(log)}")
    st.markdown("---")

    st.subheader("📁 Export Cleaned Data")
    ec1, ec2 = st.columns(2)
    with ec1:
        st.download_button(
            "⬇️ Download CSV",
            data=df_to_csv_bytes(df),
            file_name=f"{stem}_cleaned_{ts}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with ec2:
        try:
            st.download_button(
                "⬇️ Download Excel (.xlsx)",
                data=df_to_excel_bytes(df),
                file_name=f"{stem}_cleaned_{ts}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"Excel export unavailable: {e}")

    st.markdown("---")
    st.subheader("📊 Step-by-Step Log")
    if log:
        tbl = pd.DataFrame([{
            "#": i + 1,
            "Operation": s["operation"],
            "Columns": ", ".join(str(c) for c in s.get("columns", [])) or "—",
            "Key Params": ", ".join(f"{k}={v}" for k, v in s.get("params", {}).items() if k != "mapping") or "—",
            "Shape Before": f"{s['shape_before'][0]:,} × {s['shape_before'][1]}" if s.get("shape_before") else "—",
            "Timestamp": s["timestamp"],
        } for i, s in enumerate(log)])
        st.dataframe(tbl, use_container_width=True, hide_index=True)
    else:
        st.info("No transformations recorded in this session.")

    st.markdown("---")
    st.subheader("📋 Transformation Report")
    report_text = build_report(log, raw_df, df)
    st.text_area("Preview", report_text, height=250, key="rpt_preview")
    st.download_button(
        "⬇️ Download Report (.txt)",
        data=report_text.encode("utf-8"),
        file_name=f"{stem}_report_{ts}.txt",
        mime="text/plain",
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("🔁 JSON Recipe")
    st.caption("Replay the exact same transformation sequence on a new dataset.")
    json_str = build_json_recipe(log, filename)
    preview = json_str[:2000] + ("\n… (truncated)" if len(json_str) > 2000 else "")
    st.code(preview, language="json")
    st.download_button(
        "⬇️ Download JSON Recipe",
        data=json_str.encode("utf-8"),
        file_name=f"{stem}_recipe_{ts}.json",
        mime="application/json",
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("🐍 Python Replay Script")
    st.caption("Auto-generated script that reproduces all transformations.")
    py_str = build_python_script(log, filename)
    st.code(py_str, language="python")
    st.download_button(
        "⬇️ Download Python Script (.py)",
        data=py_str.encode("utf-8"),
        file_name=f"{stem}_replay_{ts}.py",
        mime="text/plain",
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("🔍 Data Preview")
    n = st.slider("Rows", 5, min(500, len(df)), 20, key="exp_n")
    st.dataframe(df.head(n), use_container_width=True)