import streamlit as st
import pandas as pd
import io
import json
from datetime import datetime


@st.cache_data(show_spinner=False)
def load_bytes(data: bytes, name: str) -> tuple:
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(data))
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(data))
        elif name.endswith(".json"):
            df = pd.read_json(io.BytesIO(data))
        else:
            return None, f"Unsupported file type: {name}"
        return df, None
    except Exception as e:
        return None, str(e)


def load_file(uploaded_file) -> tuple:
    raw = uploaded_file.read()
    return load_bytes(raw, uploaded_file.name.lower())


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False)
    return buf.getvalue()


def build_report(transform_log: list, raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> str:
    lines = [
        "=" * 60,
        "DATA WRANGLING TRANSFORMATION REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        f"\nOriginal shape : {raw_df.shape[0]:,} rows x {raw_df.shape[1]} cols",
        f"Final shape    : {clean_df.shape[0]:,} rows x {clean_df.shape[1]} cols",
        f"Rows removed   : {raw_df.shape[0] - clean_df.shape[0]:,}",
        f"Total steps    : {len(transform_log)}",
        "\n" + "-" * 60,
        "TRANSFORMATION STEPS",
        "-" * 60,
    ]
    for i, step in enumerate(transform_log, 1):
        lines.append(f"\nStep {i}: {step['operation']}")
        lines.append(f"  Timestamp    : {step['timestamp']}")
        if step.get("columns"):
            lines.append(f"  Columns      : {', '.join(str(c) for c in step['columns'])}")
        if step.get("params"):
            for k, v in step["params"].items():
                lines.append(f"  {k:<14}: {v}")
        if step.get("shape_before"):
            sb = step["shape_before"]
            lines.append(f"  Shape before : {sb[0]:,} x {sb[1]}")
    lines += ["\n" + "=" * 60, "END OF REPORT", "=" * 60]
    return "\n".join(lines)


def build_json_recipe(transform_log: list, filename: str) -> str:
    recipe = {
        "source_file": filename,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "steps": transform_log,
    }
    return json.dumps(recipe, indent=2, default=str)


def build_python_script(transform_log: list, filename: str) -> str:
    lines = [
        "import pandas as pd",
        "import numpy as np",
        "",
        f'df = pd.read_csv("{filename}")',
        "",
    ]
    for step in transform_log:
        op = step["operation"]
        p = step.get("params", {})
        cols = step.get("columns", [])
        lines.append(f"# Step: {op}")

        if op == "Fill Missing Values":
            strategy = p.get("strategy", "")
            val = p.get("fill_value", "")
            for col in cols:
                if strategy == "mean":
                    lines.append(f'df["{col}"] = df["{col}"].fillna(df["{col}"].mean())')
                elif strategy == "median":
                    lines.append(f'df["{col}"] = df["{col}"].fillna(df["{col}"].median())')
                elif strategy == "mode":
                    lines.append(f'df["{col}"] = df["{col}"].fillna(df["{col}"].mode()[0])')
                elif strategy == "constant":
                    lines.append(f'df["{col}"] = df["{col}"].fillna({repr(val)})')
                elif strategy in ("ffill", "bfill"):
                    lines.append(f'df["{col}"] = df["{col}"].{strategy}()')

        elif op == "Drop Rows with Nulls":
            subset = p.get("columns")
            if subset and subset != "any":
                lines.append(f"df = df.dropna(subset={repr(subset)})")
            else:
                lines.append("df = df.dropna()")

        elif op == "Drop Columns by Null Threshold":
            thresh = p.get("threshold", "50%").replace("%", "")
            lines.append(f"min_count = int((1 - {thresh}/100) * len(df))")
            lines.append("df = df.dropna(axis=1, thresh=min_count)")

        elif op == "Drop Duplicate Rows":
            subset = p.get("subset")
            keep = p.get("keep", "first")
            sub_str = f"subset={repr(subset)}, " if subset else ""
            keep_str = f'"{keep}"' if keep else "False"
            lines.append(f"df = df.drop_duplicates({sub_str}keep={keep_str})")

        elif op == "Convert Type":
            col = cols[0] if cols else "?"
            tt = p.get("target_type", "")
            if tt == "numeric":
                lines.append(f'df["{col}"] = pd.to_numeric(df["{col}"], errors="coerce")')
            elif tt == "dirty_numeric":
                lines.append(f'df["{col}"] = pd.to_numeric(df["{col}"].astype(str).str.replace(r"[^\\d.\\-]", "", regex=True), errors="coerce")')
            elif tt == "datetime":
                dayfirst = p.get("dayfirst", False)
                lines.append(f'df["{col}"] = pd.to_datetime(df["{col}"], dayfirst={dayfirst}, errors="coerce")')
            elif tt == "categorical":
                lines.append(f'df["{col}"] = df["{col}"].astype("category")')
            else:
                lines.append(f'df["{col}"] = df["{col}"].astype("{tt}")')

        elif op == "Trim Strings":
            for col in cols:
                lines.append(f'df["{col}"] = df["{col}"].astype(str).str.strip()')

        elif op == "Normalize Case":
            case = p.get("case", "lower")
            for col in cols:
                lines.append(f'df["{col}"] = df["{col}"].astype(str).str.{case}()')

        elif op == "Value Mapping":
            mapping = p.get("mapping", {})
            for col in cols:
                lines.append(f'df["{col}"] = df["{col}"].replace({repr(mapping)})')

        elif op == "Group Rare Categories":
            col = cols[0] if cols else "?"
            thresh = p.get("threshold", 0.05)
            label = p.get("label", "Other")
            lines.append(f'freq = df["{col}"].value_counts(normalize=True)')
            lines.append(f'rare = freq[freq < {thresh}].index')
            lines.append(f'df["{col}"] = df["{col}"].where(~df["{col}"].isin(rare), other={repr(label)})')

        elif op == "One-Hot Encoding":
            drop_first = p.get("drop_first", False)
            lines.append(f"df = pd.get_dummies(df, columns={repr(cols)}, drop_first={drop_first}, dtype=int)")

        elif op == "Handle Outliers":
            col = cols[0] if cols else "?"
            method = p.get("method", "IQR")
            action = p.get("action", "cap")
            if "IQR" in method:
                lines.append(f'q1, q3 = df["{col}"].quantile(0.25), df["{col}"].quantile(0.75)')
                lines.append(f'iqr = q3 - q1')
                lines.append(f'lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr')
                if action == "cap":
                    lines.append(f'df["{col}"] = df["{col}"].clip(lower=lower, upper=upper)')
                else:
                    lines.append(f'df = df[(df["{col}"] >= lower) & (df["{col}"] <= upper)]')
            else:
                thresh = p.get("z_threshold", 3.0)
                lines.append(f'from scipy import stats as sc')
                lines.append(f'z = abs(sc.zscore(df["{col}"].dropna()))')
                if action == "cap":
                    lines.append(f'mean, std = df["{col}"].mean(), df["{col}"].std()')
                    lines.append(f'df["{col}"] = df["{col}"].clip(mean - {thresh}*std, mean + {thresh}*std)')
                else:
                    lines.append(f'df = df[(z <= {thresh}) | df["{col}"].isna()]')

        elif op == "Scale (Min-Max)":
            for col in cols:
                lines.append(f'df["{col}"] = (df["{col}"] - df["{col}"].min()) / (df["{col}"].max() - df["{col}"].min())')

        elif op == "Scale (Z-Score)":
            for col in cols:
                lines.append(f'df["{col}"] = (df["{col}"] - df["{col}"].mean()) / df["{col}"].std()')

        elif op == "Drop Columns":
            lines.append(f"df = df.drop(columns={repr(cols)}, errors='ignore')")

        elif op == "Rename Column":
            lines.append(f'df = df.rename(columns={{{repr(p.get("old"))}: {repr(p.get("new"))}}})')

        elif op == "Add Formula Column":
            lines.append(f'df["{p.get("name")}"] = {p.get("formula")}')

        elif op == "Bin Column":
            col = cols[0] if cols else "?"
            n = p.get("n_bins", 5)
            strategy = p.get("strategy", "equal_width")
            if strategy == "equal_width":
                lines.append(f'df["{col}_bin"] = pd.cut(df["{col}"], bins={n})')
            else:
                lines.append(f'df["{col}_bin"] = pd.qcut(df["{col}"], q={n}, duplicates="drop")')

        else:
            lines.append(f"# (no codegen for: {op}  params={p})")

        lines.append("")

    lines += ['df.to_csv("output_cleaned.csv", index=False)', 'print("Done. Shape:", df.shape)']
    return "\n".join(lines)