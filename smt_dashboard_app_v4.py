
import io, os, re, json
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Oradea SMT — Multi-Day Dashboard v4", layout="wide")

DATA_STORE = "smt_saved_data.parquet"

st.title("📊 Oradea SMT — Multi-Day Line Reject Dashboard (v4)")
st.caption("Adds a **Tower filter toggle**: include or exclude components where `Tower == 1` (tray). Metrics & charts update accordingly.")

# ---------------------- Utils ----------------------
def infer_date_from_filename(name: str):
    if not name:
        return None
    s = name
    m = re.search(r"(\d{1,2})\s*([A-Za-z]{3})[A-Za-z]*\s*(\d{4})", s)
    if m:
        d, mon, y = m.group(1), m.group(2), m.group(3)
        try:
            dt = datetime.strptime(f"{d}{mon}{y}", "%d%b%Y").date()
            return dt
        except: 
            pass
    m = re.search(r"(\d{4})[-_](\d{1,2})[-_](\d{1,2})", s)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return datetime(y, mo, d).date()
        except: 
            pass
    m = re.search(r"(\d{1,2})[.\-](\d{1,2})[.\-](\d{4})", s)
    if m:
        d, mo, y = map(int, m.groups())
        try:
            return datetime(y, mo, d).date()
        except: 
            pass
    return None

def clean_columns(cols: pd.Index) -> list:
    import re
    return [re.sub(r"\s+", " ", str(c)).strip() for c in cols]

def contextual_rename(df: pd.DataFrame) -> pd.DataFrame:
    import re
    cols = list(df.columns)
    new_cols = []
    prev = None
    graph_count = 0
    for c in cols:
        cn = re.sub(r"\s+", " ", str(c)).strip()
        if cn in {"%", "% ", "%"}:
            cn = f"{prev} %" if prev else "Percent"
        if cn.lower() == "graph":
            graph_count += 1
            cn = f"Graph {graph_count}"
        new_cols.append(cn)
        if cn.lower() not in {"graph 1","graph 2","graph 3","percent","graph"}:
            prev = cn.replace("%","").strip()
    seen = {}
    final_cols = []
    for c in new_cols:
        if c not in seen:
            seen[c] = 0
            final_cols.append(c)
        else:
            seen[c] += 1
            final_cols.append(f"{c}.{seen[c]}")
    df.columns = final_cols
    return df

def find_header_row(df_full: pd.DataFrame):
    for i, row in df_full.iterrows():
        row_join = " ".join([str(x) for x in row.values if pd.notna(x)]).lower()
        if ("machine" in row_join and "name" in row_join and "component" in row_join):
            return i
    for i, row in df_full.iterrows():
        cell_value = str(row.iloc[0]).strip().lower()
        if "machine" in cell_value and "name" in cell_value:
            return i
    return None

def parse_excel(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    sheets = xls.sheet_names
    all_tables = []
    for sheet in sheets:
        if not sheet.startswith("Oradea-"):
            continue
        df_full = pd.read_excel(xls, sheet_name=sheet, header=None)
        header_row = find_header_row(df_full)
        if header_row is None:
            continue
        df = pd.read_excel(xls, sheet_name=sheet, header=header_row)
        df = df.loc[:, ~df.columns.isna()]
        df.columns = clean_columns(df.columns)
        df = contextual_rename(df)

        rename_map = {}
        for c in df.columns:
            cl = c.lower()
            if cl.startswith("machine name"): rename_map[c] = "Machine Name"
            elif cl == "component": rename_map[c] = "Component"
            elif cl == "package": rename_map[c] = "Package"
            elif cl == "tower": rename_map[c] = "Tower"
        df = df.rename(columns=rename_map)

        if "Machine Name" not in df.columns or "Component" not in df.columns:
            continue

        df = df[df["Machine Name"].notna()].copy()
        df["Line"] = sheet

        for col in ["Sum", "Wasted", "Rejected %", "Pick Err %", "Ident %", "Tower"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "Rejected %" not in df.columns and all(c in df.columns for c in ["Sum", "Wasted"]):
            df["Rejected %"] = (df["Wasted"].astype(float) / df["Sum"].astype(float)) * 100

        rdate = infer_date_from_filename(file_name) or datetime.today().date()
        df["Report Date"] = pd.to_datetime(rdate)

        all_tables.append(df)

    if not all_tables: 
        return pd.DataFrame()
    return pd.concat(all_tables, ignore_index=True)

def load_store():
    if os.path.exists(DATA_STORE):
        return pd.read_parquet(DATA_STORE)
    return pd.DataFrame()

def save_store(df: pd.DataFrame, mode="append"):
    if df.empty: 
        return
    if os.path.exists(DATA_STORE) and mode == "append":
        cur = pd.read_parquet(DATA_STORE)
        combined = pd.concat([cur, df], ignore_index=True)
        subset_cols = [c for c in ["Line","Component","Machine Name","Report Date","Sum","Wasted"] if c in combined.columns]
        combined = combined.drop_duplicates(subset=subset_cols, keep="last")
        combined.to_parquet(DATA_STORE, index=False)
    else:
        df.to_parquet(DATA_STORE, index=False)

# ---------------------- Sidebar Ingest ----------------------
with st.sidebar:
    st.header("➕ Ingest data")
    daily_files = st.file_uploader("Upload daily Excel report(s)", type=["xlsx"], accept_multiple_files=True)

    st.write("Optional: Upload **Price.xlsx** (columns: `Component`, `Price (USD)`)")
    price_file = st.file_uploader("Upload price list", type=["xlsx"], accept_multiple_files=False, key="price")

    colA, colB = st.columns(2)
    with colA:
        use_saved = st.checkbox("Use saved store", value=True)
    with colB:
        if st.button("🗑️ Clear saved store"):
            if os.path.exists(DATA_STORE):
                os.remove(DATA_STORE)
                st.success("Saved store cleared.")

# Parse uploaded daily files
uploaded_df = pd.DataFrame()
if daily_files:
    parts = []
    for f in daily_files:
        dfp = parse_excel(f.read(), f.name)
        if dfp.empty:
            st.warning(f"No line tables found in {f.name}.")
        else:
            parts.append(dfp)
    if parts:
        uploaded_df = pd.concat(parts, ignore_index=True)
        st.success(f"Ingested {len(daily_files)} file(s), rows: {len(uploaded_df):,}")

        if st.button("💾 Save to store (append)"):
            save_store(uploaded_df, mode="append")
            st.success("Data appended to store.")

# Load store
store_df = load_store() if use_saved else pd.DataFrame()

# Merge current uploaded with store for this session's analysis
data_sources = []
if not store_df.empty: data_sources.append(store_df)
if not uploaded_df.empty: data_sources.append(uploaded_df)
data = pd.concat(data_sources, ignore_index=True) if data_sources else pd.DataFrame()

# Price join
price_df = pd.DataFrame()
if price_file is not None:
    try:
        price_df = pd.read_excel(price_file)
        price_df.columns = [str(c).strip() for c in price_df.columns]
        if "Component" in price_df.columns:
            price_df["Component"] = price_df["Component"].astype(str).str.strip()
    except Exception as e:
        st.error(f"Price file error: {e}")

if not data.empty:
    data["Component"] = data["Component"].astype(str).str.strip()
    if not price_df.empty and "Price (USD)" in price_df.columns:
        data = data.merge(price_df[["Component","Price (USD)"]], on="Component", how="left")
        if "Sum" in data.columns and "Wasted" in data.columns:
            data["Cost Consumed (USD)"] = data["Sum"].astype(float) * data["Price (USD)"].astype(float)
            data["Cost Wasted (USD)"]  = data["Wasted"].astype(float) * data["Price (USD)"].astype(float)

# ---------------------- Filters ----------------------
if data.empty:
    st.info("Upload daily report(s) and/or enable 'Use saved store' to begin.")
    st.stop()

with st.sidebar:
    st.header("🔎 Filters")
    min_d = pd.to_datetime(data["Report Date"]).min().date()
    max_d = pd.to_datetime(data["Report Date"]).max().date()
    start, end = st.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    granularity = st.selectbox("Group granularity", ["Day","Week","Month"], index=0)
    lines = sorted(data["Line"].dropna().unique().tolist())
    sel_lines = st.multiselect("Lines", lines, default=lines[:1])
    include_tower = st.checkbox("Include Tower==1 (tray) components in metrics", value=True,
                                help="Uncheck to exclude rows where Tower==1 from all metrics/charts.")

# Apply filters
mask = (pd.to_datetime(data["Report Date"]).dt.date >= start) & (pd.to_datetime(data["Report Date"]).dt.date <= end)
if sel_lines:
    mask &= data["Line"].isin(sel_lines)
subset = data.loc[mask].copy()

# Apply Tower filter
if "Tower" in subset.columns:
    # Ensure numeric
    subset["Tower"] = pd.to_numeric(subset["Tower"], errors="coerce")
    if not include_tower:
        subset = subset[(subset["Tower"] != 1) & (~subset["Tower"].isna()) | (subset["Tower"].isna())]

if subset.empty:
    st.warning("No data in the selected range/lines (after Tower filter).")
    st.stop()

# Period grouping
subset["Report Date"] = pd.to_datetime(subset["Report Date"])
if granularity == "Day":
    subset["Period"] = subset["Report Date"].dt.date
elif granularity == "Week":
    subset["Period"] = subset["Report Date"].dt.to_period("W-SUN").astype(str)
else:
    subset["Period"] = subset["Report Date"].dt.to_period("M").astype(str)

# ---------------------- KPIs ----------------------
k1, k2, k3, k4, k5 = st.columns(5)
total_placed = pd.to_numeric(subset.get("Sum"), errors="coerce").sum()
total_waste  = pd.to_numeric(subset.get("Wasted"), errors="coerce").sum()
avg_reject   = pd.to_numeric(subset.get("Rejected %"), errors="coerce").mean()
overall_reject = (total_waste / total_placed * 100) if total_placed and not pd.isna(total_placed) and total_placed != 0 else np.nan

k1.metric("Total Components Placed", f"{int(total_placed):,}" if pd.notna(total_placed) else "—")
k2.metric("Total Waste (pcs)", f"{int(total_waste):,}" if pd.notna(total_waste) else "—")
k3.metric("Average Rejected % (mean of rows)", f"{avg_reject:.2f}%" if pd.notna(avg_reject) else "—")
k4.metric("Overall Reject % (Waste/Sum)", f"{overall_reject:.2f}%" if pd.notna(overall_reject) else "—")
if "Cost Wasted (USD)" in subset.columns:
    wasted_cost = pd.to_numeric(subset["Cost Wasted (USD)"], errors="coerce").sum()
    k5.metric("Cost of Waste (USD)", f"${wasted_cost:,.2f}")
else:
    k5.metric("Cost of Waste (USD)", "—")

st.divider()

# ---------------------- Trend ----------------------
st.subheader("Trend — Reject % over time")
metric_choice = st.radio(
    "Metric for trend", 
    ["Average Rejected % (rows mean)", "Overall Reject % (sum waste / sum placed)"],
    index=1,
    horizontal=True
)

trend_base = subset.groupby(["Period","Line"], dropna=False).agg({"Sum":"sum","Wasted":"sum", "Rejected %": "mean"}).reset_index()

if metric_choice.startswith("Overall"):
    trend_base["Trend %"] = (trend_base["Wasted"] / trend_base["Sum"]) * 100
else:
    trend_base["Trend %"] = trend_base["Rejected %"]

fig_trend = px.line(trend_base.sort_values("Period"), x="Period", y="Trend %", color="Line", markers=True)
fig_trend.update_layout(yaxis_title="Reject %", xaxis_title="Period")
st.plotly_chart(fig_trend, use_container_width=True)

# ---------------------- Ranking charts ----------------------
group_by = st.selectbox("Group by (for ranking charts)", ["Component", "Package", "Machine Name"], index=0)
top_n = st.slider("Top N", min_value=5, max_value=50, value=15, step=5)

agg_map = {"Sum":"sum", "Wasted":"sum", "Rejected %":"mean"}
by_group = subset.groupby([group_by], dropna=False).agg(agg_map).reset_index()
by_group["Overall Reject %"] = (by_group["Wasted"] / by_group["Sum"]) * 100
if "Price (USD)" in subset.columns:
    by_group = by_group.merge(subset.groupby(group_by, dropna=False)[["Cost Wasted (USD)"]].sum().reset_index(),
                              on=group_by, how="left")
by_group = by_group.sort_values("Overall Reject %", ascending=False).head(top_n)

left, right = st.columns(2)
with left:
    st.subheader(f"Top {len(by_group)} by Overall Reject % — {group_by}")
    fig1 = px.bar(by_group, x=group_by, y="Overall Reject %", hover_data=["Sum","Wasted"] + (["Cost Wasted (USD)"] if "Cost Wasted (USD)" in by_group.columns else []))
    fig1.update_layout(xaxis_title=group_by, yaxis_title="Overall Reject %")
    st.plotly_chart(fig1, use_container_width=True)
with right:
    st.subheader(f"Top {top_n} by Total Placed (Sum) — {group_by}")
    placed_df = subset.groupby(group_by, dropna=False)["Sum"].sum().reset_index().sort_values("Sum", ascending=False).head(top_n)
    fig2 = px.bar(placed_df, x=group_by, y="Sum")
    fig2.update_layout(xaxis_title=group_by, yaxis_title="Total Placed (Sum)")
    st.plotly_chart(fig2, use_container_width=True)

# If price is available: show Top wasted cost
if "Cost Wasted (USD)" in subset.columns:
    st.subheader(f"Top {top_n} by **Cost of Waste (USD)** — {group_by}")
    cost_df = subset.groupby(group_by, dropna=False)["Cost Wasted (USD)"].sum().reset_index().sort_values("Cost Wasted (USD)", ascending=False).head(top_n)
    fig3 = px.bar(cost_df, x=group_by, y="Cost Wasted (USD)")
    fig3.update_layout(xaxis_title=group_by, yaxis_title="Cost Wasted (USD)")
    st.plotly_chart(fig3, use_container_width=True)

# Data preview & export
with st.expander("Preview filtered data"):
    st.dataframe(subset)

csv = subset.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download filtered data (CSV)", csv, file_name="oradea_filtered.csv", mime="text/csv")

# Info on store
with st.sidebar:
    st.header("💾 Store status")
    if os.path.exists(DATA_STORE):
        sz = os.path.getsize(DATA_STORE)
        st.write(f"Store file: `{DATA_STORE}`")
        st.write(f"Size: {sz/1024:.1f} KB")
        with open(DATA_STORE, "rb") as f:
            b = f.read()
        st.download_button("Download smt_saved_data.parquet", data=b, file_name="smt_saved_data.parquet", mime="application/octet-stream")
    else:
        st.write("No saved store yet.")
