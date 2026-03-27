import streamlit as st
from datetime import datetime

_MAX_HISTORY = 20


def init_state():
    defaults = {
        "raw_df": None,
        "working_df": None,
        "filename": None,
        "transform_log": [],
        "history": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_session():
    st.session_state.raw_df = None
    st.session_state.working_df = None
    st.session_state.filename = None
    st.session_state.transform_log = []
    st.session_state.history = []


def log_transform(operation: str, params: dict, columns=None):
    df = st.session_state.working_df
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "operation": operation,
        "params": params,
        "columns": columns or [],
        "shape_before": list(df.shape) if df is not None else [],
    }
    history = st.session_state.history
    history.append(df.copy())
    if len(history) > _MAX_HISTORY:
        history.pop(0)
    st.session_state.transform_log.append(entry)


def undo_last() -> bool:
    if not st.session_state.history:
        return False
    st.session_state.working_df = st.session_state.history.pop()
    if st.session_state.transform_log:
        st.session_state.transform_log.pop()
    return True


def reset_to_raw() -> bool:
    if st.session_state.raw_df is None:
        return False
    st.session_state.working_df = st.session_state.raw_df.copy()
    st.session_state.transform_log = []
    st.session_state.history = []
    return True


def has_data() -> bool:
    return st.session_state.get("working_df") is not None