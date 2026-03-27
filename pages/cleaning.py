import streamlit as st
import pandas as pd
import numpy as np
from utils.state import has_data, log_transform, undo_last, reset_to_raw
from utils import transforms as T


def render():
    st.title("🧹 Cleaning & Preparation Studio")

    if not has_data():
        st.warning("No data loaded. Go to **📂 Upload & Overview** first.")
        return

    _workflow_bar()

    df = st.session_state.working_df
    st.caption(f"Shape: **{df.shape[0]:,} × {df.shape[1]}** · Steps applied: **{len(st.session_state.transform_log)}**")
    st.markdown("---")

    tabs = st.tabs(["🔧 Missing Values", "🔧 Duplicates", "🔧 Type Conversion",
                    "🔧 Categorical", "🔧 Numeric & Outliers", "🔧 Column Ops", "🔧 Validation"])
    with tabs[0]: _missing_section()
    with tabs[1]: _duplicates_section()
    with tabs[2]: _type_conversion_section()
    with tabs[3]: _categorical_section()
    with tabs[4]: _numeric_section()
    with tabs[5]: _column_ops_section()
    with tabs[6]: _validation_section()


def _workflow_bar():
    with st.expander("📜 Transformation Log & Undo", expanded=False):
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            if st.button("↩️ Undo Last", use_container_width=True):
                if undo_last():
                    st.rerun()
                else:
                    st.warning("Nothing to undo.")
        with c2:
            if st.button("🔁 Reset to Raw", use_container_width=True, type="secondary"):
                if reset_to_raw():
                    st.rerun()
        with c3:
            log = st.session_state.transform_log
            st.caption(f"History depth: {len(st.session_state.history)} / {len(log)} steps logged")

        log = st.session_state.transform_log
        if log:
            rows = []
            for i, e in enumerate(reversed(log), 1):
                rows.append({
                    "#": len(log) + 1 - i,
                    "Operation": e["operation"],
                    "Columns": ", ".join(str(c) for c in e.get("columns", [])) or "—",
                    "Time": e["timestamp"],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No transformations applied yet.")


def _commit(new_df: pd.DataFrame, operation: str, params: dict, columns=None):
    log_transform(operation, params, columns)
    st.session_state.working_df = new_df
    st.toast(f"✅ {operation}", icon="✅")
    st.rerun()


def _num_cols(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _cat_cols(df):
    return [c for c in df.columns if df[c].dtype == object or str(df[c].dtype) == "category"]


# ── Missing Values ────────────────────────────────────────────────────────────

def _missing_section():
    df = st.session_state.working_df
    miss = df.isnull().sum()
    null_cols = miss[miss > 0].index.tolist()

    if not null_cols:
        st.success("🎉 No missing values in the current dataset.")
        return

    miss_df = pd.DataFrame({
        "Column": null_cols,
        "Missing": miss[null_cols].values,
        "% Missing": (miss[null_cols] / len(df) * 100).round(2).values,
    })
    st.dataframe(miss_df, use_container_width=True, hide_index=True)

    st.markdown("#### Drop Rows")
    drop_row_cols = st.multiselect(
        "Drop rows where any selected column is null (empty = drop any null row)",
        null_cols, key="mv_drop_row_cols",
    )
    if st.button("Drop Rows", key="mv_drop_rows"):
        new_df = T.drop_rows_with_nulls(df, drop_row_cols or None)
        removed = len(df) - len(new_df)
        _commit(new_df, "Drop Rows with Nulls",
                {"columns": drop_row_cols or "any", "rows_removed": removed},
                drop_row_cols or list(df.columns))

    st.markdown("#### Drop Columns by Missing Threshold")
    thresh = st.slider("Drop columns with missing % above:", 10, 100, 50, 5, key="mv_col_thresh")
    candidate_cols = [c for c in null_cols if miss[c] / len(df) * 100 >= thresh]
    st.caption(f"Columns that would be dropped ({len(candidate_cols)}): {', '.join(candidate_cols) or 'none'}")
    if st.button("Drop Columns", key="mv_drop_cols"):
        new_df = T.drop_cols_with_nulls(df, thresh / 100)
        dropped = [c for c in df.columns if c not in new_df.columns]
        if not dropped:
            st.info("No columns meet this threshold.")
        else:
            _commit(new_df, "Drop Columns by Null Threshold", {"threshold": f"{thresh}%"}, dropped)

    st.markdown("#### Fill Missing Values")
    fill_cols = st.multiselect("Columns to fill", null_cols, key="mv_fill_cols")
    strategy = st.selectbox("Strategy", ["mean", "median", "mode", "constant", "ffill", "bfill"], key="mv_strategy")
    fill_val = None
    if strategy == "constant":
        fill_val = st.text_input("Constant fill value", key="mv_const")
    if strategy in ("mean", "median"):
        non_numeric = [c for c in fill_cols if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            st.warning(f"Non-numeric columns will be skipped for {strategy}: {non_numeric}")
    if st.button("Fill Missing", key="mv_fill"):
        if not fill_cols:
            st.warning("Select at least one column.")
        else:
            new_df, errors = T.fill_missing(df, fill_cols, strategy, fill_val)
            for err in errors:
                st.warning(err)
            valid_cols = [c for c in fill_cols if c not in [e.split("`")[1] for e in errors if "`" in e]]
            if valid_cols or not errors:
                _commit(new_df, "Fill Missing Values", {"strategy": strategy, "fill_value": fill_val}, fill_cols)


# ── Duplicates ────────────────────────────────────────────────────────────────

def _duplicates_section():
    df = st.session_state.working_df
    n_full = int(df.duplicated().sum())
    st.metric("Duplicate Rows (full match)", n_full)

    subset_cols = st.multiselect(
        "Check subset of columns (empty = all columns)",
        df.columns.tolist(), key="dup_subset",
    )
    keep = st.radio("Which copy to keep?", ["first", "last", "none (drop all)"], horizontal=True, key="dup_keep")

    subset = subset_cols or None
    keep_arg = False if "none" in keep else keep.split()[0]
    n_subset = int(df.duplicated(subset=subset).sum())
    st.info(f"Rows that would be removed: **{n_subset}**")

    if n_subset > 0 and st.checkbox("Preview duplicate groups", key="dup_preview"):
        sort_by = subset or df.columns.tolist()[:3]
        dup_df = df[df.duplicated(subset=subset, keep=False)].sort_values(by=sort_by)
        st.dataframe(dup_df.head(300), use_container_width=True)

    if st.button("Remove Duplicates", key="dup_remove"):
        new_df = T.remove_duplicates(df, subset=subset, keep=keep_arg)
        _commit(new_df, "Drop Duplicate Rows",
                {"subset": subset, "keep": keep_arg, "rows_removed": len(df) - len(new_df)},
                subset or [])


# ── Type Conversion ───────────────────────────────────────────────────────────

def _type_conversion_section():
    df = st.session_state.working_df
    col = st.selectbox("Column", df.columns.tolist(), key="tc_col")
    st.caption(f"Current type: `{df[col].dtype}` &nbsp;|&nbsp; Sample values: `{df[col].dropna().head(4).tolist()}`")

    target = st.selectbox(
        "Convert to",
        ["numeric", "dirty_numeric", "datetime", "categorical", "string", "boolean"],
        key="tc_target",
        help="dirty_numeric strips symbols like $, commas, % before converting"
    )
    dayfirst = False
    if target == "datetime":
        dayfirst = st.checkbox("Day-first format? (e.g. 31/01/2024)", key="tc_dayfirst")

    if st.button("Convert Type", key="tc_apply"):
        new_df, err = T.convert_type(df, col, target, dayfirst=dayfirst)
        if err:
            st.error(f"Conversion failed: {err}")
        else:
            coerced = new_df[col].isna().sum() - df[col].isna().sum()
            note = f" ({coerced} values coerced to NaN)" if coerced > 0 else ""
            _commit(new_df, "Convert Type", {"target_type": target, "dayfirst": dayfirst}, [col])


# ── Categorical Tools ─────────────────────────────────────────────────────────

def _categorical_section():
    df = st.session_state.working_df
    cats = _cat_cols(df)
    if not cats:
        st.info("No text/category columns found. Use **Type Conversion** to cast columns first.")
        return

    st.markdown("#### Trim Whitespace")
    trim_cols = st.multiselect("Columns to trim", cats, key="cat_trim")
    if st.button("Trim", key="cat_trim_btn"):
        if trim_cols:
            _commit(T.trim_strings(df, trim_cols), "Trim Strings", {}, trim_cols)
        else:
            st.warning("Select at least one column.")

    st.markdown("#### Normalize Case")
    case_cols = st.multiselect("Columns", cats, key="cat_case_cols")
    case = st.radio("Apply", ["lower", "upper", "title"], horizontal=True, key="cat_case")
    if st.button("Apply Case", key="cat_case_btn"):
        if case_cols:
            _commit(T.normalize_case(df, case_cols, case), "Normalize Case", {"case": case}, case_cols)
        else:
            st.warning("Select at least one column.")

    st.markdown("#### Value Mapping")
    map_col = st.selectbox("Column", cats, key="cat_map_col")
    uniq = df[map_col].dropna().unique().tolist()
    st.caption(f"{len(uniq)} unique values — showing first 25: `{uniq[:25]}`")
    mapping_text = st.text_area(
        "Mapping (one `old value: new value` per line)",
        key="cat_mapping",
        placeholder="new york: New York\n  new york: New York\nNEW YORK: New York",
        height=100,
    )
    if st.button("Apply Mapping", key="cat_map_btn"):
        mapping = {}
        for line in mapping_text.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                mapping[k.strip()] = v.strip()
        if not mapping:
            st.warning("No valid `old: new` lines found.")
        else:
            _commit(T.apply_mapping(df, map_col, mapping), "Value Mapping", {"mapping": mapping}, [map_col])

    st.markdown("#### Group Rare Categories")
    rare_col = st.selectbox("Column", cats, key="cat_rare_col")
    freq_thresh = st.slider("Minimum frequency % to keep", 1, 25, 5, key="cat_rare_thresh")
    rare_label = st.text_input("Replace rare with", "Other", key="cat_rare_label")
    freq_preview = df[rare_col].value_counts(normalize=True).mul(100).round(1)
    rare_count = (freq_preview < freq_thresh).sum()
    st.caption(f"{rare_count} categories below {freq_thresh}% threshold would be grouped")
    if st.button("Group Rare", key="cat_rare_btn"):
        _commit(
            T.group_rare_categories(df, rare_col, freq_thresh / 100, rare_label),
            "Group Rare Categories",
            {"threshold": freq_thresh / 100, "label": rare_label},
            [rare_col],
        )

    st.markdown("#### One-Hot Encoding")
    ohe_cols = st.multiselect("Columns to encode", cats, key="cat_ohe_cols")
    drop_first = st.checkbox("Drop first dummy column (avoid multicollinearity)", key="cat_ohe_drop")
    if ohe_cols:
        expected_new = sum(df[c].nunique() - (1 if drop_first else 0) for c in ohe_cols)
        st.caption(f"Will add ~{expected_new} new columns, remove {len(ohe_cols)} original columns")
    if st.button("One-Hot Encode", key="cat_ohe_btn"):
        if ohe_cols:
            _commit(T.one_hot_encode(df, ohe_cols, drop_first), "One-Hot Encoding", {"drop_first": drop_first}, ohe_cols)
        else:
            st.warning("Select at least one column.")


# ── Numeric & Outliers ────────────────────────────────────────────────────────

def _numeric_section():
    df = st.session_state.working_df
    nums = _num_cols(df)
    if not nums:
        st.info("No numeric columns found. Use **Type Conversion** to cast columns first.")
        return

    st.markdown("#### Outlier Detection & Handling")
    out_col = st.selectbox("Column", nums, key="out_col")
    col_data = df[out_col].dropna()

    method = st.radio("Detection method", ["IQR (1.5×)", "Z-Score"], horizontal=True, key="out_method")
    z_thresh = 3.0
    if "Z-Score" in method:
        z_thresh = st.slider("Z-Score threshold", 1.5, 5.0, 3.0, 0.1, key="out_z")

    if "IQR" in method:
        n_out = T.outlier_count_iqr(col_data)
        lower, upper = T.outlier_bounds_iqr(col_data)
        st.info(f"**{n_out}** outliers detected — bounds: [{lower:,.2f}, {upper:,.2f}]")
    else:
        n_out = T.outlier_count_zscore(col_data, z_thresh)
        st.info(f"**{n_out}** outliers detected (|z| > {z_thresh})")

    action = st.radio("Action", ["cap (clip to bounds)", "remove rows", "show only"], horizontal=True, key="out_action")
    if st.button("Handle Outliers", key="out_apply"):
        if "show" in action:
            st.info("No changes made — 'show only' selected.")
        elif n_out == 0:
            st.info("No outliers detected with current settings.")
        else:
            act = "cap" if "cap" in action else "remove"
            if "IQR" in method:
                new_df = T.handle_outliers_iqr(df, out_col, act)
            else:
                new_df = T.handle_outliers_zscore(df, out_col, act, z_thresh)
            _commit(new_df, "Handle Outliers",
                    {"method": method, "action": act, "z_threshold": z_thresh, "n_outliers": n_out},
                    [out_col])

    st.markdown("#### Feature Scaling")
    scale_cols = st.multiselect("Columns to scale", nums, key="scale_cols")
    scale_method = st.radio("Method", ["Min-Max (0–1)", "Z-Score (standardize)"], horizontal=True, key="scale_method")
    if st.button("Apply Scaling", key="scale_apply"):
        if not scale_cols:
            st.warning("Select at least one column.")
        else:
            if "Min-Max" in scale_method:
                _commit(T.scale_minmax(df, scale_cols), "Scale (Min-Max)", {}, scale_cols)
            else:
                _commit(T.scale_zscore(df, scale_cols), "Scale (Z-Score)", {}, scale_cols)


# ── Column Operations ─────────────────────────────────────────────────────────

def _column_ops_section():
    df = st.session_state.working_df
    all_cols = df.columns.tolist()
    nums = _num_cols(df)

    st.markdown("#### Rename Column")
    ren_col = st.selectbox("Column to rename", all_cols, key="cop_ren_col")
    new_name = st.text_input("New name", key="cop_new_name", placeholder="new_column_name")
    if st.button("Rename", key="cop_ren_apply"):
        new_name = new_name.strip()
        if not new_name:
            st.warning("Enter a new name.")
        elif new_name in df.columns and new_name != ren_col:
            st.error(f"Column `{new_name}` already exists.")
        else:
            _commit(T.rename_column(df, ren_col, new_name), "Rename Column",
                    {"old": ren_col, "new": new_name}, [ren_col])

    st.markdown("#### Drop Columns")
    drop_cols = st.multiselect("Columns to drop", all_cols, key="cop_drop_cols")
    if st.button("Drop Selected", key="cop_drop_apply"):
        if not drop_cols:
            st.warning("Select at least one column.")
        else:
            _commit(T.drop_columns(df, drop_cols), "Drop Columns", {}, drop_cols)

    st.markdown("#### Add Computed Column")
    new_col_name = st.text_input("New column name", key="cop_formula_name", placeholder="profit_margin")
    formula = st.text_input(
        "Formula (column names as variables, `np` available)",
        key="cop_formula",
        placeholder="(revenue - cost) / revenue",
    )
    if all_cols:
        st.caption(f"Available: `{'`, `'.join(all_cols[:12])}{'…' if len(all_cols) > 12 else ''}`")
    if st.button("Add Column", key="cop_formula_apply"):
        name = new_col_name.strip()
        expr = formula.strip()
        if not name or not expr:
            st.warning("Provide both a column name and a formula.")
        elif name in df.columns:
            st.error(f"Column `{name}` already exists. Choose a different name.")
        else:
            new_df, err = T.add_formula_column(df, name, expr)
            if err:
                st.error(f"Formula error: {err}")
            else:
                _commit(new_df, "Add Formula Column", {"name": name, "formula": expr}, [name])

    st.markdown("#### Bin Numeric Column")
    if not nums:
        st.info("No numeric columns available for binning.")
        return
    bin_col = st.selectbox("Column to bin", nums, key="cop_bin_col")
    n_bins = st.slider("Number of bins", 2, 20, 5, key="cop_n_bins")
    bin_strat = st.radio("Strategy", ["equal_width", "equal_frequency"], horizontal=True, key="cop_bin_strat",
                         help="equal_width: same range per bin · equal_frequency: same count per bin")
    if st.button("Bin Column", key="cop_bin_apply"):
        new_df, err = T.bin_column(df, bin_col, n_bins, strategy=bin_strat)
        if err:
            st.error(f"Binning error: {err}")
        else:
            _commit(new_df, "Bin Column", {"n_bins": n_bins, "strategy": bin_strat}, [bin_col])


# ── Validation ────────────────────────────────────────────────────────────────

def _validation_section():
    df = st.session_state.working_df
    all_cols = df.columns.tolist()
    nums = _num_cols(df)
    cats = [c for c in all_cols if df[c].dtype == object]

    st.markdown("#### Non-Null Check")
    null_check_cols = st.multiselect("Columns that must not be null", all_cols, key="val_null_cols")
    if st.button("Run Non-Null Check", key="val_null_btn"):
        all_violations = []
        for col in null_check_cols:
            v = T.validate_non_null(df, col)
            if not v.empty:
                all_violations.append(v)
        if not null_check_cols:
            st.warning("Select at least one column.")
        elif all_violations:
            combined = pd.concat(all_violations, ignore_index=True)
            st.warning(f"**{len(combined)} null violations** found across {len(all_violations)} column(s).")
            st.dataframe(combined.head(300), use_container_width=True)
        else:
            st.success("✅ No null violations found.")

    st.markdown("#### Numeric Range Check")
    if not nums:
        st.info("No numeric columns available.")
    else:
        range_col = st.selectbox("Column", nums, key="val_range_col")
        col_min = float(df[range_col].min()) if pd.notna(df[range_col].min()) else 0.0
        col_max = float(df[range_col].max()) if pd.notna(df[range_col].max()) else 100.0
        rc1, rc2 = st.columns(2)
        with rc1:
            min_val = st.number_input("Expected minimum", value=col_min, key="val_range_min")
        with rc2:
            max_val = st.number_input("Expected maximum", value=col_max, key="val_range_max")
        if st.button("Check Range", key="val_range_btn"):
            if min_val >= max_val:
                st.error("Minimum must be less than maximum.")
            else:
                v = T.validate_range(df, range_col, min_val, max_val)
                if v.empty:
                    st.success(f"✅ No range violations in `{range_col}`.")
                else:
                    st.warning(f"**{len(v)} violations** in `{range_col}`.")
                    st.dataframe(v.head(300), use_container_width=True)

    st.markdown("#### Allowed Categories Check")
    if not cats:
        st.info("No text columns available.")
    else:
        cat_col = st.selectbox("Column", cats, key="val_cat_col")
        uniq = sorted(df[cat_col].dropna().unique().tolist())
        allowed = st.multiselect("Allowed values", uniq, default=uniq, key="val_cat_allowed")
        if st.button("Check Categories", key="val_cat_btn"):
            if not allowed:
                st.warning("Select at least one allowed value.")
            else:
                v = T.validate_categories(df, cat_col, allowed)
                if v.empty:
                    st.success(f"✅ No category violations in `{cat_col}`.")
                else:
                    st.warning(f"**{len(v)} violations** in `{cat_col}`.")
                    st.dataframe(v.head(300), use_container_width=True)