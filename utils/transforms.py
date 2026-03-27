import pandas as pd
import numpy as np
from scipy import stats as scipy_stats


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _is_string_like(series: pd.Series) -> bool:
    return (
        series.dtype == object
        or pd.api.types.is_string_dtype(series)
        or str(series.dtype) == "category"
    )


def drop_rows_with_nulls(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    return df.dropna(subset=cols if cols else None)


def drop_cols_with_nulls(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    min_count = int((1 - threshold) * len(df))
    return df.dropna(axis=1, thresh=min_count)


def fill_missing(df: pd.DataFrame, cols: list, strategy: str, fill_value=None) -> tuple:
    df = df.copy()
    errors = []
    for col in cols:
        try:
            if strategy == "constant":
                typed_val = fill_value
                if _is_numeric(df[col]):
                    try:
                        typed_val = float(fill_value)
                    except (TypeError, ValueError):
                        pass
                df[col] = df[col].fillna(typed_val)
            elif strategy == "mean":
                if not _is_numeric(df[col]):
                    errors.append(f"`{col}` is not numeric — cannot fill with mean")
                    continue
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == "median":
                if not _is_numeric(df[col]):
                    errors.append(f"`{col}` is not numeric — cannot fill with median")
                    continue
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "mode":
                mode_vals = df[col].mode()
                if len(mode_vals) > 0:
                    df[col] = df[col].fillna(mode_vals[0])
            elif strategy == "ffill":
                df[col] = df[col].ffill()
            elif strategy == "bfill":
                df[col] = df[col].bfill()
        except Exception as e:
            errors.append(f"`{col}`: {e}")
    return df, errors


def remove_duplicates(df: pd.DataFrame, subset=None, keep="first") -> pd.DataFrame:
    return df.drop_duplicates(subset=subset or None, keep=keep)


def convert_type(df: pd.DataFrame, col: str, target_type: str, dayfirst=False) -> tuple:
    df = df.copy()
    try:
        if target_type == "numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif target_type == "dirty_numeric":
            cleaned = df[col].astype(str).str.replace(r"[^\d.\-]", "", regex=True)
            df[col] = pd.to_numeric(cleaned, errors="coerce")
        elif target_type == "categorical":
            df[col] = df[col].astype("category")
        elif target_type == "string":
            df[col] = df[col].astype(str)
        elif target_type == "datetime":
            df[col] = pd.to_datetime(df[col], dayfirst=dayfirst, errors="coerce")
        elif target_type == "boolean":
            mapping = {"true": True, "false": False, "yes": True, "no": False, "1": True, "0": False}
            df[col] = df[col].astype(str).str.lower().map(mapping)
        return df, None
    except Exception as e:
        return df, str(e)


def _safe_str_op(series: pd.Series, op: str) -> pd.Series:
    """Apply a str accessor op while preserving NaN (avoids converting NaN->'nan')."""
    null_mask = series.isna()
    result = getattr(series.astype(str).str, op)()
    return result.where(~null_mask, other=pd.NA)


def trim_strings(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if _is_string_like(df[col]):
            df[col] = _safe_str_op(df[col], "strip")
    return df


def normalize_case(df: pd.DataFrame, cols: list, case: str) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if _is_string_like(df[col]) and case in ("lower", "upper", "title", "strip"):
            df[col] = _safe_str_op(df[col], case)
    return df


def apply_mapping(df: pd.DataFrame, col: str, mapping: dict) -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].map(lambda x: mapping.get(x, x) if pd.notna(x) else x)
    return df


def group_rare_categories(df: pd.DataFrame, col: str, threshold: float, label: str = "Other") -> pd.DataFrame:
    df = df.copy()
    freq = df[col].value_counts(normalize=True)
    rare = freq[freq < threshold].index
    df[col] = df[col].where(~df[col].isin(rare), other=label)
    return df


def one_hot_encode(df: pd.DataFrame, cols: list, drop_first: bool = False) -> pd.DataFrame:
    return pd.get_dummies(df, columns=cols, drop_first=drop_first, dtype=int)


def outlier_bounds_iqr(series: pd.Series) -> tuple:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def outlier_count_iqr(series: pd.Series) -> int:
    lower, upper = outlier_bounds_iqr(series)
    return int(((series < lower) | (series > upper)).sum())


def outlier_count_zscore(series: pd.Series, threshold: float) -> int:
    clean = series.dropna()
    if len(clean) < 3:
        return 0
    return int((np.abs(scipy_stats.zscore(clean)) > threshold).sum())


def handle_outliers_iqr(df: pd.DataFrame, col: str, action: str) -> pd.DataFrame:
    df = df.copy()
    lower, upper = outlier_bounds_iqr(df[col].dropna())
    if action == "remove":
        df = df[~((df[col] < lower) | (df[col] > upper))]
    elif action == "cap":
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def handle_outliers_zscore(df: pd.DataFrame, col: str, action: str, threshold: float = 3.0) -> pd.DataFrame:
    df = df.copy()
    clean = df[col].dropna()
    if len(clean) < 3:
        return df
    z = np.abs(scipy_stats.zscore(clean))
    outlier_idx = clean.index[z > threshold]
    if action == "remove":
        df = df.drop(index=outlier_idx)
    elif action == "cap":
        mean, std = df[col].mean(), df[col].std()
        df[col] = df[col].clip(lower=mean - threshold * std, upper=mean + threshold * std)
    return df


def scale_minmax(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        mn, mx = df[col].min(), df[col].max()
        if mx != mn:
            df[col] = (df[col] - mn) / (mx - mn)
    return df


def scale_zscore(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        mean, std = df[col].mean(), df[col].std()
        if std != 0:
            df[col] = (df[col] - mean) / std
    return df


def rename_column(df: pd.DataFrame, old: str, new: str) -> pd.DataFrame:
    return df.rename(columns={old: new})


def drop_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    return df.drop(columns=cols, errors="ignore")


def add_formula_column(df: pd.DataFrame, col_name: str, formula: str) -> tuple:
    df = df.copy()
    try:
        local_vars = {c: df[c] for c in df.columns}
        local_vars["np"] = np
        result = eval(formula, {"__builtins__": {}}, local_vars)
        df[col_name] = result
        return df, None
    except Exception as e:
        return df, str(e)


def bin_column(df: pd.DataFrame, col: str, n_bins: int, strategy="equal_width") -> tuple:
    df = df.copy()
    new_col = f"{col}_bin"
    try:
        if strategy == "equal_width":
            df[new_col] = pd.cut(df[col], bins=n_bins)
        else:
            df[new_col] = pd.qcut(df[col], q=n_bins, duplicates="drop")
        df[new_col] = df[new_col].astype(str)
        return df, None
    except Exception as e:
        return df, str(e)


def validate_range(df: pd.DataFrame, col: str, min_val, max_val) -> pd.DataFrame:
    mask = df[col].notna() & ((df[col] < min_val) | (df[col] > max_val))
    out = df[mask][[col]].copy()
    out["violation"] = f"{col} out of range [{min_val}, {max_val}]"
    return out.reset_index()


def validate_categories(df: pd.DataFrame, col: str, allowed: list) -> pd.DataFrame:
    mask = df[col].notna() & ~df[col].isin(allowed)
    out = df[mask][[col]].copy()
    out["violation"] = f"{col} not in allowed set"
    return out.reset_index()


def validate_non_null(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df[df[col].isna()][[]].copy()
    out["column"] = col
    out["violation"] = "null value"
    return out.reset_index()