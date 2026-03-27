# 🔬 AI-Assisted Data Wrangler & Visualizer

A production-ready mini data preparation studio built with Python and Streamlit. Upload messy data, clean and transform it interactively, visualize insights, and export a fully reproducible workflow — all in a browser.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Regenerate sample datasets
python generate_samples.py

# 3. Launch
streamlit run app.py


## Pages

### 📂 Upload & Overview
- Supports **CSV, Excel (.xlsx/.xls), JSON**
- Guards against re-loading on every render (file identity check)
- Cached overview stats: shape, missing values, duplicates, memory
- Tabs: data preview, column types, summary statistics, missing value table
- Reset session clears all data and transformation history

### 🧹 Cleaning & Preparation Studio

| Tab | Operations |
|-----|-----------|
| **Missing Values** | Drop rows (by column subset or any), drop columns by threshold, fill with mean/median/mode/constant/ffill/bfill — with numeric-type validation |
| **Duplicates** | Full or subset duplicate detection, keep first/last/none, live preview of duplicate groups |
| **Type Conversion** | numeric, dirty_numeric (strips `$`,`,`,`%`), datetime (day-first option), categorical, string, boolean |
| **Categorical** | Trim whitespace (NaN-safe), normalize case (NaN-safe), value mapping via text editor, rare category grouping with live preview, one-hot encoding |
| **Numeric & Outliers** | IQR and Z-score outlier detection with live count, cap or remove action, min-max and z-score scaling |
| **Column Ops** | Rename (duplicate name check), drop, formula columns (`np` available, error caught), equal-width/frequency binning |
| **Validation** | Non-null check, numeric range check, allowed categories check — all return violations table |

**Workflow controls** (top of page):
- ↩️ Undo last step (history stack, up to 20 levels)
- 🔁 Reset to raw data
- Step log table showing operation, columns, timestamp

### 📊 Visualization Builder
- **Chart types:** Histogram, Box Plot, Scatter, Line, Bar, Heatmap (Correlation)
- Sidebar filters: categorical (multi-select, up to 40 unique values) + numeric (range sliders)
- Scatter auto-samples to 3,000 points for performance
- Heatmap handles NaN correlations gracefully
- Optional **Plotly** interactive mode (if installed)
- Stable sidebar filter keys (prefixed `sf_c_` / `sf_n_`) prevent key collisions after transforms

### 📤 Export & Report
- **CSV** export (always available)
- **Excel** export (requires openpyxl)
- **Transformation report** — human-readable `.txt` with all steps, params, shapes, timestamps
- **JSON recipe** — machine-readable, replayable on new datasets
- **Python replay script** — valid, importable Python covering all 14 operation types; AST-verified syntax
- Step-by-step log table with key params and shape-before

---

## Project Structure

```
data_wrangler/
├── app.py                    # Entry point: page config, CSS, sidebar, routing
├── requirements.txt
├── generate_samples.py       # Generates sample datasets with intentional dirty data
├── pages/
│   ├── __init__.py
│   ├── upload.py             # Page A — cached stats, file identity guard
│   ├── cleaning.py           # Page B — all 7 cleaning tabs
│   ├── visualization.py      # Page C — 6 chart types + filters
│   └── export.py             # Page D — all 4 export formats
├── utils/
│   ├── __init__.py
│   ├── state.py              # Session state init, log, undo (20-level cap), reset
│   ├── transforms.py         # Pure transformation functions, pandas 2.x compatible
│   └── io.py                 # File I/O, cached loader, all 4 export builders
└── sample_data/
    ├── retail_sales.csv      # 1,230 rows × 11 cols
    └── employee_analytics.csv # 1,540 rows × 12 cols
```

---

## Sample Datasets

### `retail_sales.csv` (1,230 rows × 11 columns)
Retail order transactions with category, region, pricing, and ratings.

**Intentional data quality issues:**
- `returned`: inconsistent casing (`Yes`, `no`, `YES`) + 19% nulls
- `revenue`: some values prefixed `$` (dirty numeric)
- `discount_pct`, `customer_rating`, `unit_price`: 5–7% missing
- ~30 exact duplicate rows

**Suggested workflow:** dirty_numeric → `revenue` · trim + lower → `returned` · mean fill → `discount_pct` · dedup · OHE → `category`

---

### `employee_analytics.csv` (1,540 rows × 12 columns)
HR analytics: department, seniority, salary, performance, remote days.

**Intentional data quality issues:**
- `salary`: 20 extreme outliers (negative, 500 000+), 6% missing
- `city`: mixed casing + leading whitespace (`  new york`)
- `age`, `performance_score`, `attrition`: 4–5% missing
- ~40 duplicate rows

**Suggested workflow:** trim → `city` · lower → `city` · IQR cap → `salary` · mean fill → `performance_score` · dedup · bin → `years_experience`

---

## Design Notes

- **`st.session_state`** holds raw df, working df, log, and history. `init_state()` is called once in `app.py`, not per-page.
- **`st.cache_data`** is used on file loading (`load_bytes`) and overview statistics (`_overview_stats`). Cache keys use file bytes + name, and `id(df)` for stats.
- **History stack** caps at 20 entries to bound memory use on large datasets.
- **File identity guard** in upload page checks `uploaded.name != current_name` so switching pages does not re-load and clear transforms.
- **`_commit()`** in cleaning logs the transform *then* updates `working_df` and calls `st.rerun()` — ensuring `shape_before` is captured correctly.
- **`_safe_str_op()`** in transforms applies string operations while preserving NaN (avoids the `NaN → "nan"` issue in pandas 2.x StringDtype).
- **`_is_string_like()`** handles `object`, `StringDtype` (pandas 2.x), and `category` dtypes.
- **Python replay script** covers all 14 operation types and passes `ast.parse()` validation.
- Matplotlib uses `Agg` backend for headless/server compatibility.

---

## Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | ≥ 1.35 | App framework |
| pandas | ≥ 2.1 | Data manipulation |
| numpy | ≥ 1.26 | Numeric operations |
| matplotlib | ≥ 3.8 | Core charting |
| scipy | ≥ 1.12 | Z-score outlier detection |
| openpyxl | ≥ 3.1 | Excel read/write |
| plotly | ≥ 5.20 | Optional interactive charts |