import pandas as pd
from prophet import Prophet
import firebase_admin
from firebase_admin import credentials, firestore
import json
import os


def load_supply_data():
    """Load and concatenate supply CSV files for 2024 and 2025."""
    supply_2024 = pd.read_csv("datasets/ByMonthDataset2024.csv")
    supply_2025 = pd.read_csv("datasets/ByMonthDataset2025.csv")
    supply = pd.concat([supply_2024, supply_2025], ignore_index=True)
    print(f"[Supply] Raw concatenated shape: {supply.shape}")
    return supply


def process_supply_data(supply_df):
    """Convert supply data to Prophet format (ds, y) by grouping daily donations."""
    supply_df["Date"] = pd.to_datetime(supply_df["Date"], format="mixed", dayfirst=False)
    daily_counts = supply_df.groupby("Date").size().reset_index(name="y")
    daily_counts.rename(columns={"Date": "ds"}, inplace=True)
    print(f"[Supply] Processed shape: {daily_counts.shape}")
    return daily_counts


def load_demand_data():
    """Load the demand CSV file."""
    demand = pd.read_csv(
        "datasets/Blood-Request-Demand-2024-2025-Blood-Demand-Data.csv"
    )
    print(f"[Demand] Raw shape: {demand.shape}")
    return demand


def process_demand_data(demand_df):
    """Convert demand data to Prophet format (ds, y)."""
    demand_df["ds"] = pd.to_datetime(
        demand_df["Year"].astype(str) + "-" + demand_df["Month"], format="%Y-%B"
    )
    demand_df.rename(columns={"Total": "y"}, inplace=True)
    demand_prophet = demand_df[["ds", "y"]].copy()
    print(f"[Demand] Processed shape: {demand_prophet.shape}")
    return demand_prophet


def fetch_firebase_data():
    """Initialize Firebase and fetch valid donor records from the 'donors' collection."""
    firebase_creds = os.environ.get("FIREBASE_CREDENTIALS")
    if not firebase_creds:
        print("[Firebase] FIREBASE_CREDENTIALS not set. Skipping Firebase data.")
        return pd.DataFrame(columns=["ds", "y"])

    try:
        cred_dict = json.loads(firebase_creds)
        cred = credentials.Certificate(cred_dict)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)

        db = firestore.client()
        docs = db.collection("donors").stream()

        records = []
        for doc in docs:
            data = doc.to_dict()
            if "joinedAt" in data and data.get("isValid", False):
                records.append({"joinedAt": data["joinedAt"]})

        if not records:
            print("[Firebase] No valid records found in 'donors' collection.")
            return pd.DataFrame(columns=["ds", "y"])

        fb_df = pd.DataFrame(records)
        fb_df["joinedAt"] = pd.to_datetime(fb_df["joinedAt"], utc=True)
        fb_df["joinedAt"] = fb_df["joinedAt"].dt.tz_localize(None)
        fb_df["joinedAt"] = fb_df["joinedAt"].dt.normalize()
        fb_df = fb_df[fb_df["joinedAt"] <= pd.Timestamp.today()]
        daily_counts = fb_df.groupby("joinedAt").size().reset_index(name="y")
        daily_counts.rename(columns={"joinedAt": "ds"}, inplace=True)
        print(f"[Firebase] Fetched shape: {daily_counts.shape}")
        return daily_counts

    except Exception as e:
        print(f"[Firebase] Error fetching data: {e}")
        return pd.DataFrame(columns=["ds", "y"])


def fetch_firebase_demand_data():
    """Fetch blood request records from Firestore, aggregate to monthly demand DataFrame."""
    firebase_creds = os.environ.get("FIREBASE_CREDENTIALS")
    if not firebase_creds:
        print("[Firebase/Demand] FIREBASE_CREDENTIALS not set. Skipping Firestore demand data.")
        return pd.DataFrame(columns=["ds", "y"])

    try:
        if not firebase_admin._apps:
            cred_dict = json.loads(firebase_creds)
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)

        db = firestore.client()
        docs = db.collection("blood_requests").stream()

        records = []
        for doc in docs:
            data = doc.to_dict()
            request_date = data.get("requestDate")
            units = data.get("units")
            if request_date is not None and units is not None:
                records.append({"requestDate": request_date, "units": float(units)})

        if not records:
            print("[Firebase/Demand] No records found in 'blood_requests' collection.")
            return pd.DataFrame(columns=["ds", "y"])

        fb_df = pd.DataFrame(records)
        fb_df["requestDate"] = pd.to_datetime(fb_df["requestDate"], utc=True, errors="coerce")
        fb_df = fb_df.dropna(subset=["requestDate"])
        fb_df["requestDate"] = fb_df["requestDate"].dt.tz_localize(None)

        # Bucket each request to the first day of its month for monthly aggregation
        fb_df["ds"] = fb_df["requestDate"].dt.to_period("M").dt.to_timestamp()
        monthly = fb_df.groupby("ds")["units"].sum().reset_index(name="y")
        print(f"[Firebase/Demand] Fetched {len(records)} docs → {len(monthly)} monthly buckets.")
        return monthly

    except Exception as e:
        print(f"[Firebase/Demand] Error fetching data: {e}")
        return pd.DataFrame(columns=["ds", "y"])


def merge_demand_data(historical_df, firestore_df):
    """Merge historical CSV demand with Firestore data.

    Firestore is the source of truth: when a month exists in both datasets,
    the Firestore value replaces the historical CSV value.
    """
    if firestore_df.empty:
        print("[Demand] No Firestore data; using historical CSV only.")
        return historical_df

    # Concatenate with historical first so Firestore rows come last;
    # drop_duplicates(keep='last') then retains the Firestore value for any overlap.
    combined = pd.concat([historical_df, firestore_df], ignore_index=True)
    combined = combined.sort_values("ds")
    combined = combined.drop_duplicates(subset="ds", keep="last").reset_index(drop=True)
    print(
        f"[Demand] Merged shape: {combined.shape} "
        f"(historical: {len(historical_df)}, firestore: {len(firestore_df)})"
    )
    return combined


def ensure_continuous_monthly(df):
    """Resample a monthly (ds, y) dataframe to fill any missing months with 0."""
    df = df.set_index("ds").resample("MS").sum().reset_index()
    df["y"] = df["y"].fillna(0)
    print(f"[Resample] Continuous monthly shape: {df.shape}")
    return df


def train_and_predict(df, label="Model"):
    """Train a Prophet model and predict the next 12 months."""
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=12, freq="MS")
    forecast = model.predict(future)
    print(f"[{label}] Forecast shape: {forecast.shape}")
    return forecast


ACTUALS_THRESHOLD = 5  # Months with actual value below this are treated as missing data


def export_forecast(forecast, actuals_df, filename):
    """Export forecast results to a JSON file.

    For dates present in actuals_df whose value meets the threshold, the actual
    y value is used.  If the actual value is missing (NaN) or below the threshold
    it is treated as missing data and the Prophet yhat prediction is used instead,
    ensuring the dashboard never shows a blank or dropped line.
    """
    result = forecast[["ds", "yhat"]].copy()

    # Merge actual values onto the forecast timeline
    actuals = actuals_df[["ds", "y"]].copy()
    result = result.merge(actuals, on="ds", how="left")

    # Smart fallback: use yhat when actual is absent or below the threshold
    use_actual = result["y"].notna() & (result["y"] >= ACTUALS_THRESHOLD)
    result["value"] = result["yhat"].copy()
    result.loc[use_actual, "value"] = result.loc[use_actual, "y"]

    result["value"] = result["value"].clip(lower=0).round(0).astype(int)
    result["date"] = result["ds"].dt.strftime("%Y-%m-%d")
    output = result[["date", "value"]].to_dict(orient="records")

    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[Export] Saved {filename} with {len(output)} records.")


def export_blood_type_distribution(supply_forecast, demand_forecast, supply_actuals, demand_actuals):
    """Allocate total forecasted supply and demand across blood types
    for every month in the forecast timeline, with perfect rounding that
    guarantees sum(blood types) == total for both supply and demand.

    Actuals are used for historical dates; Prophet yhat is used for future dates,
    matching the behaviour of export_forecast exactly.
    """
    distribution = {
        "O+":  0.450,
        "A+":  0.220,
        "B+":  0.240,
        "AB+": 0.050,
        "O-":  0.015,
        "A-":  0.010,
        "B-":  0.010,
        "AB-": 0.005,
    }

    # Align supply and demand forecasts by date, merging in actuals
    supply_cols = supply_forecast[["ds", "yhat"]].rename(columns={"yhat": "supply_yhat"})
    supply_cols = supply_cols.merge(
        supply_actuals[["ds", "y"]].rename(columns={"y": "y_supply"}),
        on="ds", how="left"
    )
    demand_cols = demand_forecast[["ds", "yhat"]].rename(columns={"yhat": "demand_yhat"})
    demand_cols = demand_cols.merge(
        demand_actuals[["ds", "y"]].rename(columns={"y": "y_demand"}),
        on="ds", how="left"
    )
    merged = supply_cols.merge(demand_cols, on="ds", how="inner")
    merged = merged.sort_values("ds").reset_index(drop=True)

    if merged.empty:
        print("[BloodType] No aligned forecast rows found; skipping export.")
        return

    output = []
    for _, row in merged.iterrows():
        # Smart fallback: use yhat when actual is absent or below threshold (mirrors export_forecast)
        supply_actual = row["y_supply"]
        supply_val = (
            supply_actual
            if (not pd.isna(supply_actual) and supply_actual >= ACTUALS_THRESHOLD)
            else row["supply_yhat"]
        )
        demand_actual = row["y_demand"]
        demand_val = (
            demand_actual
            if (not pd.isna(demand_actual) and demand_actual >= ACTUALS_THRESHOLD)
            else row["demand_yhat"]
        )
        total_supply = max(0, round(supply_val))
        total_demand = max(0, round(demand_val))

        # Round each blood type allocation
        supply_alloc = {bt: max(0, round(total_supply * pct)) for bt, pct in distribution.items()}
        demand_alloc = {bt: max(0, round(total_demand * pct)) for bt, pct in distribution.items()}

        # Apply remainder correction to O+ so sum == total exactly
        supply_alloc["O+"] += total_supply - sum(supply_alloc.values())
        demand_alloc["O+"] += total_demand - sum(demand_alloc.values())

        output.append({
            "date": row["ds"].strftime("%Y-%m-%d"),
            "totals": {
                "supply": total_supply,
                "demand": total_demand,
            },
            "distribution": {
                bt: {
                    "supply": supply_alloc[bt],
                    "demand": demand_alloc[bt],
                }
                for bt in distribution
            },
        })

    with open("forecast_blood_types.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"[BloodType] Exported forecast_blood_types.json with {len(output)} monthly records.")


def main():
    # --- Supply Pipeline ---
    supply_raw = load_supply_data()
    supply_df = process_supply_data(supply_raw)

    # Append Firebase live data
    firebase_df = fetch_firebase_data()
    if not firebase_df.empty:
        supply_df = pd.concat([supply_df, firebase_df], ignore_index=True)
        supply_df = supply_df.groupby("ds")["y"].sum().reset_index()
        print(f"[Supply] After Firebase merge shape: {supply_df.shape}")

    # Filter out records before 2024
    supply_df = supply_df[supply_df["ds"] >= "2024-01-01"].reset_index(drop=True)
    print(f"[Supply] After 2024+ filter shape: {supply_df.shape}")

    # Aggregate to monthly totals so Prophet sees realistic 600+ unit peaks
    supply_df = supply_df.set_index("ds").resample("MS").sum().reset_index()
    print(f"[Supply] Monthly aggregated shape: {supply_df.shape}")

    supply_forecast = train_and_predict(supply_df, label="Supply")
    export_forecast(supply_forecast, supply_df, "forecast_supply.json")

    # --- Demand Pipeline ---
    demand_raw = load_demand_data()
    demand_df = process_demand_data(demand_raw)

    # Fetch real-time demand from Firestore and merge (Firestore wins on overlapping months)
    firebase_demand_df = fetch_firebase_demand_data()
    demand_df = merge_demand_data(demand_df, firebase_demand_df)

    # Filter out records before 2024
    demand_df = demand_df[demand_df["ds"] >= "2024-01-01"].reset_index(drop=True)
    print(f"[Demand] After 2024+ filter shape: {demand_df.shape}")

    # Ensure no month gaps before training
    demand_df = ensure_continuous_monthly(demand_df)

    demand_forecast = train_and_predict(demand_df, label="Demand")
    export_forecast(demand_forecast, demand_df, "forecast_demand.json")

    export_blood_type_distribution(supply_forecast, demand_forecast, supply_df, demand_df)

    print("\n✅ Forecasting complete.")


if __name__ == "__main__":
    main()


