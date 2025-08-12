
import io, os, re, json
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Oradea SMT — Multi-Day Dashboard v5.1", layout="wide")

DATA_STORE = "smt_saved_data.parquet"
PRICE_STORE = "price_store.parquet"
CPK_COMPONENTS = {"CP1", "CC02-05_CPP", "CC02-05"}

st.title("📊 Oradea SMT — Multi-Day Line Reject Dashboard (v5.1)")
st.caption("""
Enhancements over v5: **Restore/Replace production store** by uploading an existing `smt_saved_data.parquet` (or CSV).
""")

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
    return [re.sub(r"\\s+", " ", str(c)).strip() for c in cols]

def contextual_rename(df: pd.DataFrame) -> pd.DataFrame:
    import re
    cols = list(df.columns)
    new_cols = []
    prev = None
    graph_count = 0
    for c in cols:
        cn = re.sub(r"\\s+", " ", str(c)).strip()
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

def load_store(path):
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()

def save_store(df: pd.DataFrame, path: str, mode="append", keys=None):
    if df.empty: 
        return
    if os.path.exists(path) and mode == "append":
        cur = pd.read_parquet(path)
        combined = pd.concat([cur, df], ignore_index=True)
        if keys:
            combined = combined.drop_duplicates(subset=keys, keep="last")
        else:
            combined = combined.drop_duplicates(keep="last")
        combined.to_parquet(path, index=False)
    else:
        df.to_parquet(path, index=False)

# ---------------------- Sidebar Ingest ----------------------
with st.sidebar:
    st.header("➕ Ingest data")
    daily_files = st.file_uploader("Upload daily Excel report(s)", type=["xlsx"], accept_multiple_files=True)

    st.write("Optional: Upload **Price.xlsx** (columns: `Component`, `Price (USD)`)")
    price_file = st.file_uploader("Upload price list", type=["xlsx"], accept_multiple_files=False, key="price")

    colA, colB = st.columns(2)
    with colA:
        use_saved = st.checkbox("Use saved production store", value=True)
    with colB:
        if st.button("🗑️ Clear saved production store"):
            if os.path.exists(DATA_STORE):
                os.remove(DATA_STORE)
                st.success("Saved production store cleared.")

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

        if st.button("💾 Save to production store (append)"):
            save_store(uploaded_df, DATA_STORE, mode="append",
                       keys=[c for c in ["Line","Component","Machine Name","Report Date","Sum","Wasted"] if c in uploaded_df.columns])
            st.success("Data appended to production store.")

# Load production store
store_df = load_store(DATA_STORE) if use_saved else pd.DataFrame()

# Merge current uploaded with store for this session's analysis
data_sources = []
if not store_df.empty: data_sources.append(store_df)
if not uploaded_df.empty: data_sources.append(uploaded_df)
data = pd.concat(data_sources, ignore_index=True) if data_sources else pd.DataFrame()

# ---------------------- Price list persistence ----------------------
with st.sidebar:
    st.header("💵 Price list (persistent)")
    price_store = load_store(PRICE_STORE)
    if not price_store.empty:
        st.write(f"Loaded {len(price_store):,} price rows from store.")
    else:
        st.write("No saved prices yet.")

    if price_file is not None:
        try:
            new_prices = pd.read_excel(price_file)
            new_prices.columns = [str(c).strip() for c in new_prices.columns]
            if "Component" in new_prices.columns and "Price (USD)" in new_prices.columns:
                new_prices["Component"] = new_prices["Component"].astype(str).str.strip()
                save_store(new_prices, PRICE_STORE, mode="append", keys=["Component"])
                st.success(f"Saved/updated {len(new_prices):,} prices to store.")
                price_store = load_store(PRICE_STORE)  # reload
            else:
                st.error("Price file must have columns: Component, Price (USD)")
        except Exception as e:
            st.error(f"Price file error: {e}")
    if not price_store.empty:
        st.download_button("Download price store", data=price_store.to_csv(index=False).encode("utf-8"),
                           file_name="price_store.csv", mime="text/csv")

# Join prices to data
if not data.empty and not price_store.empty:
    data["Component"] = data["Component"].astype(str).str.strip()
    price_store["Component"] = price_store["Component"].astype(str).str.strip()
    data = data.merge(price_store[["Component","Price (USD)"]], on="Component", how="left")
    if "Sum" in data.columns and "Wasted" in data.columns:
        data["Cost Consumed (USD)"] = data["Sum"].astype(float) * data["Price (USD)"].astype(float)
        data["Cost Wasted (USD)"]  = data["Wasted"].astype(float) * data["Price (USD)"].astype(float)

# ---------------------- Filters ----------------------
if data.empty:
    st.info("Upload daily report(s) and/or enable 'Use saved production store' to begin.")
    st.stop()

with st.sidebar:
    st.header("🔎 Filters")
    min_d = pd.to_datetime(data["Report Date"]).min().date()
    max_d = pd.to_datetime(data["Report Date"]).max().date()
    start, end = st.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    granularity = st.selectbox("Group granularity", ["Day","Week","Month"], index=0)
    lines = sorted(data["Line"].dropna().unique().tolist())
    sel_lines = st.multiselect("Lines", lines, default=lines[:1])
    include_tower = st.checkbox("Include Tower==1 (tray) components", value=True)
    exclude_cpk = st.checkbox("Exclude CPK components (CP1, CC02-05_CPP, CC02-05)", value=True)

# Apply base filters
mask = (pd.to_datetime(data["Report Date"]).dt.date >= start) & (pd.to_datetime(data["Report Date"]).dt.date <= end)
if sel_lines:
    mask &= data["Line"].isin(sel_lines)
subset = data.loc[mask].copy()

# Apply Tower filter
if "Tower" in subset.columns:
    subset["Tower"] = pd.to_numeric(subset["Tower"], errors="coerce")
    if not include_tower:
        subset = subset[(subset["Tower"] != 1) | (subset["Tower"].isna())]

# Exclude CPK components
subset["Component"] = subset["Component"].astype(str)
if exclude_cpk:
    subset = subset[~subset["Component"].str.strip().str.upper().isin({"CP1","CC02-05_CPP","CC02-05"})]

if subset.empty:
    st.warning("No data in the selected range/lines (after filters).")
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
k3.metric("Average Rejected % (rows mean)", f"{avg_reject:.2f}%" if pd.notna(avg_reject) else "—")
k4.metric("Overall Reject % (Waste/Sum)", f"{overall_reject:.2f}%" if pd.notna(overall_reject) else "—")
if "Cost Wasted (USD)" in subset.columns:
    wasted_cost = pd.to_numeric(subset["Cost Wasted (USD)"], errors="coerce").sum()
    k5.metric("Cost of Waste (USD)", f"${wasted_cost:,.2f}")
else:
    k5.metric("Cost of Waste (USD)", "—")

st.divider()

# ---------------------- Trend with factory line ----------------------
st.subheader("Trend — Reject % over time (with Factory Overall)")
metric_choice = st.radio(
    "Metric for trend", 
    ["Average Rejected % (rows mean)", "Overall Reject % (sum waste / sum placed)"],
    index=1,
    horizontal=True
)

trend_line = subset.groupby(["Period","Line"], dropna=False).agg({"Sum":"sum","Wasted":"sum", "Rejected %": "mean"}).reset_index()
trend_factory = subset.groupby(["Period"], dropna=False).agg({"Sum":"sum","Wasted":"sum", "Rejected %":"mean"}).reset_index()
trend_factory["Line"] = "Factory Overall"
trend_base = pd.concat([trend_line, trend_factory], ignore_index=True)

if metric_choice.startswith("Overall"):
    trend_base["Trend %"] = (trend_base["Wasted"] / trend_base["Sum"]) * 100
else:
    trend_base["Trend %"] = trend_base["Rejected %"]

fig_trend = px.line(trend_base.sort_values("Period"), x="Period", y="Trend %", color="Line", markers=True)
fig_trend.update_layout(yaxis_title="Reject %", xaxis_title="Period")
st.plotly_chart(fig_trend, use_container_width=True)

# ---------------------- Waste over time (pcs) ----------------------
st.subheader("Factory — Total Waste (pcs) over time")
waste_time = subset.groupby("Period", dropna=False)["Wasted"].sum().reset_index()
fig_waste_t = px.bar(waste_time, x="Period", y="Wasted")
fig_waste_t.update_layout(yaxis_title="Wasted (pcs)", xaxis_title="Period")
st.plotly_chart(fig_waste_t, use_container_width=True)

# ---------------------- Totals by line (placed & waste) ----------------------
st.subheader("Totals by Line — Placed vs Waste (current filters)")
by_line = subset.groupby("Line", dropna=False)[["Sum","Wasted"]].sum().reset_index()
by_line_melt = by_line.melt(id_vars="Line", value_vars=["Sum","Wasted"], var_name="Metric", value_name="Value")
fig_byline = px.bar(by_line_melt, x="Line", y="Value", color="Metric", barmode="group")
fig_byline.update_layout(yaxis_title="Count (pcs)", xaxis_title="Line")
st.plotly_chart(fig_byline, use_container_width=True)

# ---------------------- Ranking charts ----------------------
group_by = st.selectbox("Group by (for ranking charts)", ["Component", "Package", "Machine Name"], index=0)
top_n = st.slider("Top N", min_value=5, max_value=50, value=15, step=5)

agg_map = {"Sum":"sum", "Wasted":"sum", "Rejected %":"mean"}
by_group = subset.groupby([group_by], dropna=False).agg(agg_map).reset_index()
by_group["Overall Reject %"] = (by_group["Wasted"] / by_group["Sum"]) * 100
if "Cost Wasted (USD)" in subset.columns:
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

# Info on stores
with st.sidebar:
    st.header("💾 Store status & restore")
    if os.path.exists(DATA_STORE):
        sz = os.path.getsize(DATA_STORE)
        st.write(f"Production store: `{DATA_STORE}` — {sz/1024:.1f} KB")
        with open(DATA_STORE, "rb") as f:
            b = f.read()
        st.download_button("Download production store", data=b, file_name="smt_saved_data.parquet", mime="application/octet-stream", key="dl_prod")
    else:
        st.write("No production store yet.")

    st.write("---")
    st.write("Restore/Replace store from a local file:")
    restore_file = st.file_uploader("Upload store (.parquet or .csv)", type=["parquet","csv"], key="restore")
    if restore_file is not None:
        try:
            if restore_file.name.lower().endswith(".parquet"):
                df_restore = pd.read_parquet(restore_file)
            else:
                df_restore = pd.read_csv(restore_file)
            if not df_restore.empty:
                # Replace the entire store
                df_restore.to_parquet(DATA_STORE, index=False)
                st.success(f"Restored store with {len(df_restore):,} rows.")
            else:
                st.warning("Uploaded store file is empty.")
        except Exception as e:
            st.error(f"Restore failed: {e}")

    if os.path.exists(PRICE_STORE):
        szp = os.path.getsize(PRICE_STORE)
        st.write(f"Price store: `{PRICE_STORE}` — {szp/1024:.1f} KB")
    else:
        st.write("No price store yet.")
