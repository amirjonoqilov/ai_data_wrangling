import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utils.state import has_data

try:
    import plotly.express as px
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

CHART_TYPES = ["Histogram", "Box Plot", "Scatter", "Line", "Bar", "Heatmap (Correlation)"]


def render():
    st.title("📊 Visualization Builder")

    if not has_data():
        st.warning("No data loaded. Go to **📂 Upload & Overview** first.")
        return

    df = st.session_state.working_df
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype) == "category"]

    if not num_cols and not cat_cols:
        st.warning("No plottable columns found in the current dataset.")
        return

    use_plotly = PLOTLY_OK and st.sidebar.checkbox("Use Plotly (interactive)", value=False)
    st.sidebar.markdown("### Filters")
    filtered_df = _apply_filters(df, cat_cols, num_cols)

    n_filtered = len(filtered_df)
    if n_filtered == 0:
        st.warning("All rows filtered out. Adjust sidebar filters.")
        return

    st.caption(f"Visualizing **{n_filtered:,} / {len(df):,}** rows")

    chart_type = st.selectbox("Chart Type", CHART_TYPES, key="chart_type")
    st.markdown("---")

    cfg_col, chart_col = st.columns([1, 2])
    with cfg_col:
        config = _chart_config(chart_type, filtered_df, num_cols, cat_cols)
    with chart_col:
        if config is not None:
            _render_chart(filtered_df, chart_type, config, use_plotly)


def _apply_filters(df: pd.DataFrame, cat_cols: list, num_cols: list) -> pd.DataFrame:
    filtered = df
    shown_cat = [c for c in cat_cols[:5] if df[c].nunique() <= 40]
    for col in shown_cat:
        uniq = sorted(df[col].dropna().unique().tolist(), key=str)
        sel = st.sidebar.multiselect(col, uniq, default=uniq, key=f"sf_c_{col}")
        if sel:
            filtered = filtered[filtered[col].isin(sel)]
        else:
            filtered = filtered[filtered[col].isna()]

    for col in num_cols[:3]:
        mn = df[col].min()
        mx = df[col].max()
        if pd.notna(mn) and pd.notna(mx) and mn < mx:
            mn, mx = float(mn), float(mx)
            lo, hi = st.sidebar.slider(col, mn, mx, (mn, mx), key=f"sf_n_{col}")
            filtered = filtered[(filtered[col] >= lo) & (filtered[col] <= hi)]
    return filtered


def _chart_config(chart_type: str, df: pd.DataFrame, num_cols: list, cat_cols: list):
    cfg = {}
    if chart_type == "Histogram":
        if not num_cols:
            st.warning("No numeric columns for histogram.")
            return None
        cfg["col"] = st.selectbox("Column", num_cols, key="h_col")
        cfg["bins"] = st.slider("Bins", 5, 100, 20, key="h_bins")
        cfg["hue"] = st.selectbox("Color by", ["None"] + cat_cols, key="h_hue")

    elif chart_type == "Box Plot":
        if not num_cols:
            st.warning("No numeric columns for box plot.")
            return None
        cfg["y"] = st.selectbox("Y (numeric)", num_cols, key="bp_y")
        cfg["x"] = st.selectbox("Group by (optional)", ["None"] + cat_cols, key="bp_x")

    elif chart_type == "Scatter":
        if len(num_cols) < 2:
            st.warning("Need at least 2 numeric columns for scatter plot.")
            return None
        cfg["x"] = st.selectbox("X axis", num_cols, key="sc_x")
        y_opts = [c for c in num_cols if c != cfg["x"]]
        cfg["y"] = st.selectbox("Y axis", y_opts, key="sc_y")
        cfg["hue"] = st.selectbox("Color by", ["None"] + cat_cols, key="sc_hue")
        cfg["size"] = st.selectbox("Size by", ["None"] + num_cols, key="sc_size")

    elif chart_type == "Line":
        if not num_cols:
            st.warning("No numeric columns for line chart.")
            return None
        cfg["x"] = st.selectbox("X axis", df.columns.tolist(), key="ln_x")
        cfg["y"] = st.selectbox("Y axis", num_cols, key="ln_y")
        cfg["group"] = st.selectbox("Group by", ["None"] + cat_cols, key="ln_group")
        cfg["agg"] = st.selectbox("Aggregate Y by", ["mean", "sum", "median", "count"], key="ln_agg")

    elif chart_type == "Bar":
        x_opts = cat_cols or df.columns.tolist()
        if not x_opts:
            st.warning("No categorical columns for bar chart.")
            return None
        cfg["x"] = st.selectbox("Category (X)", x_opts, key="br_x")
        cfg["y"] = st.selectbox("Value (Y)", ["count"] + num_cols, key="br_y")
        cfg["agg"] = st.selectbox("Aggregation", ["count", "mean", "sum", "median"], key="br_agg") if cfg["y"] != "count" else "count"
        cfg["top_n"] = st.slider("Top N bars", 5, 50, 15, key="br_topn")
        cfg["sort"] = st.checkbox("Sort descending", True, key="br_sort")

    elif chart_type == "Heatmap (Correlation)":
        if len(num_cols) < 2:
            st.warning("Need at least 2 numeric columns for correlation heatmap.")
            return None
        cfg["cols"] = st.multiselect("Columns (empty = all numeric)", num_cols, key="hm_cols")

    return cfg


def _render_chart(df, chart_type, config, use_plotly):
    try:
        dispatch = {
            "Histogram": _hist,
            "Box Plot": _box,
            "Scatter": _scatter,
            "Line": _line,
            "Bar": _bar,
            "Heatmap (Correlation)": _heatmap,
        }
        dispatch[chart_type](df, config, use_plotly)
    except Exception as e:
        st.error(f"Chart error: {e}")
        st.exception(e)


def _mpl_fig(figsize=(9, 5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0e1117")
    ax.set_facecolor("#1a1e2e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3050")
    ax.tick_params(colors="#9aabcc", labelsize=8)
    ax.xaxis.label.set_color("#9aabcc")
    ax.yaxis.label.set_color("#9aabcc")
    ax.title.set_color("#dde8f5")
    return fig, ax


def _show_mpl(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


ACCENT = "#4e7fff"
PALETTE = "#4e7fff #e05c5c #50c87c #f0a050 #a070d0 #40c0d0 #f0d050 #c07050".split()


def _hist(df, cfg, plotly):
    col, bins = cfg["col"], cfg["bins"]
    hue = cfg["hue"] if cfg["hue"] != "None" else None
    data = df[col].dropna()
    if len(data) == 0:
        st.warning("No data after dropping nulls.")
        return
    if plotly and PLOTLY_OK:
        fig = px.histogram(df.dropna(subset=[col]), x=col, nbins=bins, color=hue,
                           template="plotly_dark", title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = _mpl_fig()
        if hue:
            for i, (val, grp) in enumerate(df.groupby(hue)):
                grp[col].dropna().hist(bins=bins, ax=ax, alpha=0.7, label=str(val), color=PALETTE[i % len(PALETTE)])
            ax.legend(fontsize=8)
        else:
            ax.hist(data, bins=bins, color=ACCENT, edgecolor="#0e1117", alpha=0.9)
        ax.set_xlabel(col); ax.set_ylabel("Frequency"); ax.set_title(f"Distribution of {col}")
        _show_mpl(fig)


def _box(df, cfg, plotly):
    y = cfg["y"]
    x = cfg["x"] if cfg["x"] != "None" else None
    if df[y].dropna().empty:
        st.warning("No data to plot.")
        return
    if plotly and PLOTLY_OK:
        fig = px.box(df, x=x, y=y, template="plotly_dark", title=f"Box Plot: {y}" + (f" by {x}" if x else ""))
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = _mpl_fig()
        if x:
            groups = df.groupby(x)
            data = [g[y].dropna().values for _, g in groups]
            labels = [str(k) for k, _ in groups]
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(bp["boxes"], PALETTE):
                patch.set_facecolor(color); patch.set_alpha(0.75)
            for el in ["whiskers", "caps", "medians"]:
                for ln in bp[el]: ln.set_color("#9aabcc")
            ax.set_xticklabels(labels, rotation=30, ha="right")
        else:
            ax.boxplot(df[y].dropna(), patch_artist=True, boxprops=dict(facecolor=ACCENT, alpha=0.8))
        ax.set_ylabel(y); ax.set_title(f"Box Plot: {y}" + (f" by {x}" if x else ""))
        _show_mpl(fig)


def _scatter(df, cfg, plotly):
    x, y = cfg["x"], cfg["y"]
    hue = cfg["hue"] if cfg["hue"] != "None" else None
    size_col = cfg["size"] if cfg["size"] != "None" else None
    use_cols = [c for c in [x, y, hue, size_col] if c]
    plot_df = df[use_cols].dropna()
    if plot_df.empty:
        st.warning("No data after dropping nulls.")
        return
    if plotly and PLOTLY_OK:
        fig = px.scatter(plot_df, x=x, y=y, color=hue, size=size_col,
                         template="plotly_dark", opacity=0.65, title=f"{x} vs {y}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = _mpl_fig()
        sample = plot_df if len(plot_df) <= 3000 else plot_df.sample(3000, random_state=0)
        if hue:
            for i, (val, grp) in enumerate(sample.groupby(hue)):
                ax.scatter(grp[x], grp[y], label=str(val), alpha=0.55, s=15, color=PALETTE[i % len(PALETTE)])
            ax.legend(fontsize=7)
        else:
            ax.scatter(sample[x], sample[y], color=ACCENT, alpha=0.45, s=12)
        ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(f"{x} vs {y}")
        if len(plot_df) > 3000:
            ax.set_title(f"{x} vs {y} (sampled 3,000)")
        _show_mpl(fig)


def _line(df, cfg, plotly):
    x, y, group, agg = cfg["x"], cfg["y"], cfg["group"], cfg["agg"]
    group = group if group != "None" else None
    try:
        by = [x, group] if group else [x]
        plot_df = df.groupby(by)[y].agg(agg).reset_index().sort_values(x)
    except Exception:
        cols = [c for c in [x, y, group] if c]
        plot_df = df[cols].dropna().sort_values(x)
    if plot_df.empty:
        st.warning("No data to plot.")
        return
    if plotly and PLOTLY_OK:
        fig = px.line(plot_df, x=x, y=y, color=group, template="plotly_dark",
                      title=f"{agg}({y}) over {x}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = _mpl_fig()
        xs = plot_df[x].astype(str)
        if group and group in plot_df.columns:
            for i, (val, grp) in enumerate(plot_df.groupby(group)):
                ax.plot(grp[x].astype(str), grp[y], marker="o", ms=3, lw=1.5,
                        label=str(val), color=PALETTE[i % len(PALETTE)])
            ax.legend(fontsize=8)
        else:
            ax.plot(xs, plot_df[y], color=ACCENT, marker="o", ms=3, lw=1.5)
        ticks = list(range(0, len(plot_df), max(1, len(plot_df) // 10)))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(plot_df[x].iloc[i]) for i in ticks], rotation=40, ha="right", fontsize=7)
        ax.set_xlabel(x); ax.set_ylabel(f"{agg}({y})"); ax.set_title(f"{agg}({y}) over {x}")
        _show_mpl(fig)


def _bar(df, cfg, plotly):
    x, y, agg, top_n, sort = cfg["x"], cfg["y"], cfg["agg"], cfg["top_n"], cfg["sort"]
    if y == "count" or agg == "count":
        plot_df = df[x].value_counts().reset_index(name="count").rename(columns={"index": x})
        y_col = "count"
    else:
        plot_df = df.groupby(x)[y].agg(agg).reset_index()
        y_col = y
    if sort:
        plot_df = plot_df.sort_values(y_col, ascending=False)
    plot_df = plot_df.head(top_n)
    if plot_df.empty:
        st.warning("No data to plot.")
        return
    if plotly and PLOTLY_OK:
        fig = px.bar(plot_df, x=x, y=y_col, template="plotly_dark",
                     title=f"{y_col} by {x} (top {top_n})")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = _mpl_fig()
        n = len(plot_df)
        colors = [PALETTE[i % len(PALETTE)] for i in range(n)]
        bars = ax.bar(range(n), plot_df[y_col].values, color=colors, edgecolor="#0e1117", lw=0.4)
        ax.set_xticks(range(n))
        ax.set_xticklabels(plot_df[x].astype(str), rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(y_col); ax.set_title(f"{y_col} by {x} (top {top_n})")
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h * 1.005,
                        f"{h:,.0f}", ha="center", va="bottom", fontsize=6.5, color="#9aabcc")
        _show_mpl(fig)


def _heatmap(df, cfg, _plotly=None):
    cols = cfg["cols"] if cfg["cols"] else df.select_dtypes(include=[np.number]).columns.tolist()
    if len(cols) < 2:
        st.warning("Select at least 2 columns.")
        return
    corr = df[cols].corr()
    n = len(cols)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.75), max(5, n * 0.7)), facecolor="#0e1117")
    ax.set_facecolor("#1a1e2e")
    im = ax.imshow(corr.fillna(0), cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8, color="#9aabcc")
    ax.set_yticklabels(cols, fontsize=8, color="#9aabcc")
    for i in range(n):
        for j in range(n):
            val = corr.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(val) > 0.5 else "#9aabcc")
    ax.set_title("Correlation Matrix", color="#dde8f5", fontsize=12)
    _show_mpl(fig)