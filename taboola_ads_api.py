# ============================================================
# TABOOLA ADS -> GOOGLE SHEETS PIPELINE
# Pulls campaign-level daily data for all 21 advertiser accounts
# using Taboola Backstage API (OAuth2 client credentials).
# Runs Prophet CPL forecast, writes to Google Sheet.
# ============================================================

import os
import requests
import pandas as pd
import numpy as np
import gspread
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from datetime import datetime, timedelta
import pytz
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

# ============================================================
# CREDENTIALS
# ============================================================

TABOOLA_CLIENT_ID     = os.environ["TABOOLA_CLIENT_ID"]
TABOOLA_CLIENT_SECRET = os.environ["TABOOLA_CLIENT_SECRET"]
TABOOLA_ACCOUNTS = []   

SHEETS_CLIENT_ID     = os.environ["SHEETS_CLIENT_ID"]
SHEETS_CLIENT_SECRET = os.environ["SHEETS_CLIENT_SECRET"]
SHEETS_REFRESH_TOKEN = os.environ["SHEETS_REFRESH_TOKEN"]
TABOOLA_SHEET_ID     = os.environ["TABOOLA_SHEET_ID"]

# ============================================================
# CONFIG
# ============================================================
end_date   = datetime.today()
start_date = end_date - timedelta(days=60)
START_STR  = start_date.strftime("%Y-%m-%d")
END_STR    = end_date.strftime("%Y-%m-%d")

PRIMARY_CONV_ACTION = "leads"
BASE_URL            = "https://backstage.taboola.com/backstage/api/1.0"
TOKEN_URL           = "https://backstage.taboola.com/backstage/oauth/token"

MAX_THREADS  = 10
MAX_RETRIES  = 3
RETRY_WAIT   = 30

# ============================================================
# LOGGING
# ============================================================

pipeline_logs = []

def log_message(level, message):
    """Add a log entry with GMT timestamp"""
    gmt_time = datetime.now(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S")
    pipeline_logs.append({
        "Timestamp": gmt_time,
        "Level": level,
        "Message": message
    })
    print(f"[{gmt_time}] {level}: {message}")

# ============================================================
# OAUTH TOKEN
# ============================================================

_token_cache = {"token": None, "expires_at": 0}

def get_access_token():
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expires_at"] - 60:
        return _token_cache["token"]

    log_message("INFO", "Fetching new Taboola access token")
    resp = requests.post(TOKEN_URL, data={
        "client_id":     TABOOLA_CLIENT_ID,
        "client_secret": TABOOLA_CLIENT_SECRET,
        "grant_type":    "client_credentials",
    })
    resp.raise_for_status()
    data = resp.json()
    _token_cache["token"]      = data["access_token"]
    _token_cache["expires_at"] = now + data.get("expires_in", 3600)
    log_message("INFO", "Taboola access token acquired successfully")
    return _token_cache["token"]

# ============================================================
# BACKSTAGE API HELPERS
# ============================================================

def backstage_get(endpoint, params=None, raise_for_status=True):
    headers = {"Authorization": f"Bearer {get_access_token()}"}
    resp = requests.get(f"{BASE_URL}/{endpoint}", headers=headers, params=params or {})
    if raise_for_status:
        resp.raise_for_status()
    return resp

def fetch_taboola_accounts():
    log_message("INFO", "Fetching account list from Taboola API...")
    try:
        resp = backstage_get("users/current/allowed-accounts").json()
    except Exception as e:
        log_message("ERROR", f"Failed to fetch allowed accounts: {e}")
        raise RuntimeError(f"Failed to fetch allowed accounts: {e}")

    raw_results = resp.get("results", resp.get("items", resp.get("accounts", [])))
    accounts = []
    seen_aids = set()

    for acct in raw_results:
        aid           = str(acct.get("id", acct.get("partner_id", "")))
        account_id    = acct.get("account_id", "")
        name          = acct.get("name", account_id)
        if not acct.get("is_active", True) or str(acct.get("type", "")).lower() == "network":
            continue
        if aid in seen_aids:
            continue
        seen_aids.add(aid)
        accounts.append({"aid": aid, "name": name, "account_id": account_id})
    
    log_message("INFO", f"Retrieved {len(accounts)} active accounts from API")
    return accounts

# ============================================================
# ADS DATA PULL
# ============================================================

def fetch_account_insights(acct):
    aid, acct_name, account_id = acct["aid"], acct["name"], acct["account_id"]
    rows = []
    resolved_id = account_id
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = backstage_get(f"{account_id}/reports/campaign-summary/dimensions/campaign_day_breakdown",
                                 {"start_date": START_STR, "end_date": END_STR, "include_multi_conversions": True},
                                 raise_for_status=False)
            if resp.status_code == 400 and account_id != aid:
                account_id = aid
                resolved_id = aid
                resp = backstage_get(f"{aid}/reports/campaign-summary/dimensions/campaign_day_breakdown",
                                     {"start_date": START_STR, "end_date": END_STR, "include_multi_conversions": True})
            resp.raise_for_status()
            for row in resp.json().get("results", []):
                spent = float(row.get("spent", 0) or 0)
                conversions = float(row.get("cpa_actions_num", 0) or 0)
                rows.append({
                    "Account_ID": aid, "Account_Name": acct_name, "Account_Slug": account_id,
                    "Campaign_ID": str(row.get("campaign", "")), "Campaign_Name": row.get("campaign_name", ""),
                    "Date": row.get("date", ""), "Impressions": int(row.get("impressions", 0) or 0),
                    "Clicks": int(row.get("clicks", 0) or 0), "Spend": round(spent, 2),
                    "Conversions": round(conversions, 2), "Conv_Value": round(float(row.get("conversions_value", 0) or 0), 2),
                    "CTR_Pct": round(float(row.get("ctr", 0) or 0), 4), "CPC": round(float(row.get("cpc", 0) or 0), 4),
                    "CPM": round(float(row.get("cpm", 0) or 0), 4),
                    "ROAS": round(float(row.get("conversions_value", 0) or 0) / spent, 4) if spent > 0 else 0,
                    "CPL": round(spent / conversions, 4) if conversions > 0 else 0
                })
            log_message("INFO", f"Account '{acct_name}' ({aid}): Retrieved {len(rows)} rows")
            return rows, {"Account_ID": aid, "Account_Name": acct_name, "Account_Slug": resolved_id, "Rows": len(rows), "Status": "OK"}
        except Exception as e:
            if attempt == MAX_RETRIES:
                log_message("ERROR", f"Account '{acct_name}' ({aid}): Failed after {MAX_RETRIES} attempts - {str(e)[:100]}")
                return [], {"Account_ID": aid, "Account_Name": acct_name, "Account_Slug": resolved_id, "Rows": 0, "Status": f"FAILED: {str(e)[:50]}"}
            log_message("WARNING", f"Account '{acct_name}' ({aid}): Attempt {attempt} failed, retrying...")
            time.sleep(1)
    return [], {"Account_ID": aid, "Account_Name": acct_name, "Account_Slug": resolved_id, "Rows": 0, "Status": "FAILED"}

def pull_taboola_ads_data():
    log_message("INFO", f"Starting data pull for {len(TABOOLA_ACCOUNTS)} accounts (Date range: {START_STR} to {END_STR})")
    get_access_token()
    all_rows, summary = [], []
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {executor.submit(fetch_account_insights, acct): acct for acct in TABOOLA_ACCOUNTS}
        for future in as_completed(futures):
            rows, s = future.result()
            all_rows.extend(rows)
            summary.append(s)
    
    log_message("INFO", f"Data pull complete: {len(all_rows)} total rows retrieved")
    return pd.DataFrame(all_rows), pd.DataFrame(summary)

# ============================================================
# PROPHET FORECASTING (Restored all CI and KPI columns)
# ============================================================

def run_prophet_forecast_taboola(df):
    log_message("INFO", "Starting Prophet forecast generation")
    try:
        from prophet import Prophet
    except ImportError:
        log_message("ERROR", "Prophet library not available, skipping forecast")
        return pd.DataFrame()

    if df.empty:
        log_message("WARNING", "Empty dataframe provided, skipping forecast")
        return pd.DataFrame()

    df_agg = df.copy()
    df_agg["data_date"] = pd.to_datetime(df_agg["Date"])
    df_agg["cost"] = df_agg["Spend"]
    df_agg["conversions"] = df_agg["Conversions"]
    df_agg["account_name"] = df_agg["Account_Name"]
    df_agg["account_id"] = df_agg["Account_ID"]

    # AGGREGATE TO ACCOUNT LEVEL ONLY (Removed Campaign_ID)
    df_agg = df_agg.groupby(["data_date", "account_name", "account_id"]).agg(
        cost=("cost", "sum"), conversions=("conversions", "sum")).reset_index()

    today = pd.Timestamp.now().normalize()
    df_agg = df_agg[df_agg["data_date"] < today]
    reference_date = df_agg["data_date"].max()
    cutoff = reference_date - pd.Timedelta(days=59)
    df_agg = df_agg[df_agg["data_date"] >= cutoff]

    df_raw = df_agg.copy()
    df_filtered = df_agg[df_agg["conversions"] > 0].copy()
    df_filtered["cpl"] = df_filtered["cost"] / df_filtered["conversions"]

    last7_start = reference_date - pd.Timedelta(days=6)
    prev7_start = reference_date - pd.Timedelta(days=13)
    prev7_end   = reference_date - pd.Timedelta(days=7)
    next7_end   = reference_date + pd.Timedelta(days=7)

    def fit_prophet_series(daily_df, value_col, log_transform=True):
        pdf = daily_df[["data_date", value_col]].rename(columns={"data_date": "ds", value_col: "y"})
        pdf = pdf[pdf["y"] > 0].copy()
        if log_transform: pdf["y"] = np.log(pdf["y"])
        model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False, interval_width=0.8)
        model.fit(pdf)
        forecast = model.predict(model.make_future_dataframe(periods=7))
        if log_transform:
            for c in ["yhat", "yhat_lower", "yhat_upper"]: forecast[c] = np.exp(forecast[c])
        return forecast

    def run_prophet(daily_df, kpi_df):
        daily_df = daily_df.sort_values("data_date").copy()
        daily_df["cpl"] = daily_df["cost"] / daily_df["conversions"]
        Q1, Q3 = daily_df["cpl"].quantile(0.25), daily_df["cpl"].quantile(0.75)
        IQR = Q3 - Q1
        daily_df["cpl_capped"] = daily_df["cpl"].clip(max(Q1 - 1.5 * IQR, 0.01), Q3 + 1.5 * IQR)

        prophet_df = daily_df[["data_date", "cpl_capped"]].rename(columns={"data_date": "ds", "cpl_capped": "y"})
        prophet_df["y"] = np.log(prophet_df["y"])
        model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False, interval_width=0.8)
        model.fit(prophet_df)

        forecast = model.predict(model.make_future_dataframe(periods=7))
        for c in ["yhat", "yhat_lower", "yhat_upper"]: forecast[c] = np.exp(forecast[c])

        cost_f = fit_prophet_series(daily_df, "cost")
        conv_f = fit_prophet_series(daily_df, "conversions")

        # KPI Logic Restored
        actual_window = daily_df[daily_df["data_date"] >= last7_start]
        merged = actual_window[["data_date", "cpl"]].merge(forecast[["ds", "yhat"]].rename(columns={"ds": "data_date"}), on="data_date")
        mape = (np.abs(merged["cpl"] - merged["yhat"]) / merged["cpl"]).mean() * 100 if not merged.empty else np.nan
        reliability = "High" if mape < 15 else ("Medium" if mape < 35 else "Low")

        last7 = kpi_df[kpi_df["data_date"] >= last7_start]
        prev7 = kpi_df[(kpi_df["data_date"] >= prev7_start) & (kpi_df["data_date"] <= prev7_end)]
        l7c, l7v = last7["cost"].sum(), last7["conversions"].sum()
        p7c, p7v = prev7["cost"].sum(), prev7["conversions"].sum()
        l7cpl, p7cpl = (l7c/l7v if l7v > 0 else None), (p7c/p7v if p7v > 0 else None)

        forecast_future = forecast[forecast["ds"] > reference_date].copy()
        avg_f_cpl = forecast_future["yhat"].mean()
        fcast_pct = ((avg_f_cpl - l7cpl) / l7cpl * 100) if l7cpl else None

        return forecast_future, {
            "Last7_CPL": round(l7cpl, 4) if l7cpl else None, "Prev7_CPL": round(p7cpl, 4) if p7cpl else None,
            "Last7_Cost": round(l7c, 2), "Last7_Conversions": int(l7v), "Prev7_Cost": round(p7c, 2), "Prev7_Conversions": int(p7v),
            "Trend": "Improving" if (l7cpl and p7cpl and l7cpl < p7cpl) else "Declining",
            "Current_Pct": round(((l7cpl - p7cpl) / p7cpl * 100), 2) if (l7cpl and p7cpl) else None,
            "Forecast_Avg_CPL": round(avg_f_cpl, 4), "Forecast_Pct": round(fcast_pct, 2) if fcast_pct else None,
            "Forecast_Direction": "Improving" if (fcast_pct and fcast_pct < -8) else "Stable",
            "MAPE": round(mape, 2) if not np.isnan(mape) else None, "Reliability": reliability
        }, cost_f[cost_f["ds"] > reference_date], conv_f[conv_f["ds"] > reference_date]

    results = []
    accounts = df_filtered[["account_name", "account_id"]].drop_duplicates()
    
    log_message("INFO", f"Generating forecasts for {len(accounts)} accounts")

    for _, acct_row in accounts.iterrows():
        acct, aid = acct_row["account_name"], acct_row["account_id"]
        camp_df = df_filtered[df_filtered["account_id"] == aid].copy()
        if len(camp_df) < 14:
            log_message("WARNING", f"Account '{acct}' ({aid}): Insufficient data ({len(camp_df)} rows), skipping forecast")
            continue
        try:
            f_fut, kpis, c_f, v_f = run_prophet(camp_df, df_raw[df_raw["account_id"] == aid])
            cost_map, conv_map = c_f.set_index("ds")["yhat"].to_dict(), v_f.set_index("ds")["yhat"].to_dict()
            last_ds = f_fut["ds"].max()
            for _, row in f_fut.iterrows():
                is_last = (row["ds"] == last_ds)
                results.append({
                    "Date": str(row["ds"].date()), "Account": acct, "Account_ID": aid,
                    "Actual_CPL": None, "Actual_Spend": None, "Actual_Conversions": None,
                    "Forecast_CPL": round(row["yhat"], 4), "Lower_CI": round(row["yhat_lower"], 4), "Upper_CI": round(row["yhat_upper"], 4),
                    "Forecast_Cost": round(cost_map.get(row["ds"], 0), 2), "Forecast_Conv": round(conv_map.get(row["ds"], 0), 2),
                    "Last7_CPL": kpis["Last7_CPL"] if is_last else None, "Prev7_CPL": kpis["Prev7_CPL"] if is_last else None,
                    "Last7_Cost": kpis["Last7_Cost"] if is_last else None, "Last7_Conversions": kpis["Last7_Conversions"] if is_last else None,
                    "Prev7_Cost": kpis["Prev7_Cost"] if is_last else None, "Prev7_Conversions": kpis["Prev7_Conversions"] if is_last else None,
                    "Trend": kpis["Trend"] if is_last else None, "Current_Pct": kpis["Current_Pct"] if is_last else None,
                    "Forecast_Avg_CPL": kpis["Forecast_Avg_CPL"] if is_last else None, "Forecast_Pct": kpis["Forecast_Pct"] if is_last else None,
                    "Forecast_Direction": kpis["Forecast_Direction"] if is_last else None, "MAPE": kpis["MAPE"] if is_last else None, "Reliability": kpis["Reliability"] if is_last else None
                })
            log_message("INFO", f"Account '{acct}' ({aid}): Forecast generated successfully")
        except Exception as e:
            log_message("ERROR", f"Account '{acct}' ({aid}): Forecast failed - {str(e)[:100]}")
            continue

    # Build final table (Restored all CI and KPI column references)
    actual_table = df_filtered.copy().rename(columns={
        "data_date": "Date", "account_name": "Account", "account_id": "Account_ID",
        "cost": "Actual_Spend", "conversions": "Actual_Conversions", "cpl": "Actual_CPL"})
    actual_table["Date"] = actual_table["Date"].astype(str)
    
    forecast_df = pd.DataFrame(results)
    final_table = pd.concat([actual_table, forecast_df], ignore_index=True).fillna("")
    
    col_order = [
        "Date", "Account", "Account_ID", "Actual_CPL", "Actual_Spend", "Actual_Conversions",
        "Forecast_CPL", "Lower_CI", "Upper_CI", "Forecast_Cost", "Forecast_Conv",
        "Last7_CPL", "Prev7_CPL", "Last7_Cost", "Last7_Conversions", "Prev7_Cost", "Prev7_Conversions",
        "Trend", "Current_Pct", "Forecast_Avg_CPL", "Forecast_Pct", "Forecast_Direction", "MAPE", "Reliability"
    ]
    
    log_message("INFO", f"Forecast table generated: {len(final_table)} rows")
    return final_table[col_order]

# ============================================================
# SHEETS WRITE
# ============================================================

def get_sheets_client():
    log_message("INFO", "Authenticating with Google Sheets")
    creds = Credentials(token=None, refresh_token=SHEETS_REFRESH_TOKEN, token_uri="https://oauth2.googleapis.com/token",
                        client_id=SHEETS_CLIENT_ID, client_secret=SHEETS_CLIENT_SECRET,
                        scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
    creds.refresh(Request())
    log_message("INFO", "Google Sheets authentication successful")
    return gspread.authorize(creds)

def write_to_sheet(sh, tab_name, df):
    try:
        ws = sh.worksheet(tab_name)
        ws.clear()
        log_message("INFO", f"Sheet '{tab_name}': Cleared existing data")
    except:
        ws = sh.add_worksheet(title=tab_name, rows=50000, cols=25)
        log_message("INFO", f"Sheet '{tab_name}': Created new worksheet")
    
    if not df.empty:
        ws.update([list(df.columns)])
        data = df.fillna("").values.tolist()
        for i in range(0, len(data), 5000):
            ws.append_rows(data[i:i + 5000], value_input_option="USER_ENTERED")
        log_message("INFO", f"Sheet '{tab_name}': Wrote {len(df)} rows")
    else:
        log_message("WARNING", f"Sheet '{tab_name}': No data to write")

def write_logs_to_sheet(sh):
    """Write logs to sheet, keeping only last 100 rows"""
    try:
        try:
            ws = sh.worksheet("Logs")
            # Read existing logs
            existing_data = ws.get_all_records()
            existing_logs = pd.DataFrame(existing_data) if existing_data else pd.DataFrame()
        except:
            ws = sh.add_worksheet(title="Logs", rows=150, cols=3)
            existing_logs = pd.DataFrame()
            log_message("INFO", "Created new 'Logs' worksheet")
        
        # Combine existing and new logs
        new_logs = pd.DataFrame(pipeline_logs)
        
        if not existing_logs.empty and not new_logs.empty:
            # Ensure columns match
            if set(new_logs.columns) == set(existing_logs.columns):
                combined_logs = pd.concat([existing_logs, new_logs], ignore_index=True)
            else:
                combined_logs = new_logs
        elif not new_logs.empty:
            combined_logs = new_logs
        else:
            combined_logs = existing_logs
        
        # Keep only last 100 rows
        if len(combined_logs) > 500:
            combined_logs = combined_logs.tail(500).reset_index(drop=True)
        
        # Write to sheet
        ws.clear()
        ws.update([list(combined_logs.columns)])
        data = combined_logs.fillna("").values.tolist()
        ws.append_rows(data, value_input_option="USER_ENTERED")
        
        print(f"[{datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S')}] INFO: Logs sheet updated with {len(combined_logs)} total rows (last 500 kept)")
        
    except Exception as e:
        print(f"[{datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Failed to write logs to sheet - {str(e)}")

def main():
    global TABOOLA_ACCOUNTS
    
    log_message("INFO", "Pipeline execution started")
    
    try:
        TABOOLA_ACCOUNTS = fetch_taboola_accounts()
        taboola_df, df_summary = pull_taboola_ads_data()
        forecast_df = run_prophet_forecast_taboola(taboola_df)
        
        log_message("INFO", "Connecting to Google Sheets")
        gc = get_sheets_client()
        sh = gc.open_by_key(TABOOLA_SHEET_ID)
        
        write_to_sheet(sh, "TaboolaAdsData", taboola_df)
        write_to_sheet(sh, "TaboolaSummary", df_summary)
        
        if not forecast_df.empty:
            write_to_sheet(sh, "TaboolaForecast", forecast_df)
        else:
            log_message("WARNING", "No forecast data generated")
        
        # Write logs last
        write_logs_to_sheet(sh)
        
        log_message("INFO", "Pipeline execution completed successfully")
        print("\nAll done! ✓")
        
    except Exception as e:
        log_message("ERROR", f"Pipeline execution failed: {str(e)}")
        # Still try to write logs even if pipeline fails
        try:
            gc = get_sheets_client()
            sh = gc.open_by_key(TABOOLA_SHEET_ID)
            write_logs_to_sheet(sh)
        except:
            pass
        raise

if __name__ == "__main__":
    main()
