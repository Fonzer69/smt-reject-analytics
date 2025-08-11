# Oradea SMT — Multi‑Day Dashboard

Streamlit app for analyzing SMT line reject data from daily Excel reports, with optional component pricing and tray (Tower) filtering.

## Files

- `smt_dashboard_app_v4.py` — main app (upload daily reports, save to a Parquet store, filter by date granularity, join prices, include/exclude Tower==1).
- `requirements.txt` — Python dependencies for Streamlit Community Cloud.
- (Optional) `smt_saved_data.parquet` — your local persisted store (not tracked in git).

## Running locally

```bash
pip install -r requirements.txt
streamlit run smt_dashboard_app_v4.py
```

## Deploying to Streamlit Community Cloud

1. Create a **public GitHub repo** and add:
   - `smt_dashboard_app_v4.py`
   - `requirements.txt`
   - `README.md` (this file)
   - (Do **not** commit data files. The app expects uploads.)
2. Go to **https://streamlit.io/cloud** → **Sign in** → **New app**.
3. Select your repo, branch, and main file `smt_dashboard_app_v4.py` → **Deploy**.
4. In the running app:
   - Upload one or more daily Excel reports (e.g., `Oradea (08Aug2025) day.xlsx`).
   - (Optional) Upload `Price.xlsx` with columns: `Component`, `Price (USD)`.
   - Click **“Save to store (append)”** to persist to `smt_saved_data.parquet` (note: Cloud file storage is ephemeral—see below).
   - Use the **Tower toggle** to include/exclude tray components (Tower==1) in metrics.

### Notes on persistence (Streamlit Cloud)
- The app writes to `smt_saved_data.parquet` on the server’s filesystem. On Streamlit Community Cloud this storage can be **cleared on redeploy, update, or inactivity**.
- Recommended options for long-term persistence:
  - Download and re-upload the store file via the sidebar.
  - Or integrate a cloud backend (S3, GCS, Supabase/Postgres).

### Python version
Streamlit Cloud uses a recent Python (3.11+). No extra config is required, but you can pin via a `runtime.txt` if needed (e.g., `python-3.11.9`).

### Troubleshooting
- **Module not found** → ensure `requirements.txt` includes: `streamlit`, `pandas`, `plotly`, `openpyxl`, `pyarrow`.
- **Large uploads** → keep daily Excel files reasonably sized (tens of MB). For very large datasets, consider pre-aggregating.
- **Weird columns** → the parser searches for the “Machine Name / Component” header block in each `Oradea-xxxx` sheet.

## Price file format
A simple Excel with two columns:
```
Component | Price (USD)
```

## License
MIT (or your choice).
