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


def train_and_predict(df, label="Model"):
    """Train a Prophet model and predict the next 12 months."""
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=12, freq="ME")
    forecast = model.predict(future)
    print(f"[{label}] Forecast shape: {forecast.shape}")
    return forecast


def export_forecast(forecast, filename):
    """Export forecast results to a JSON file."""
    result = forecast[["ds", "yhat"]].copy()
    result["date"] = result["ds"].dt.strftime("%Y-%m-%d")
    result["value"] = result["yhat"].clip(lower=0).round(0).astype(int)
    output = result[["date", "value"]].to_dict(orient="records")

    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[Export] Saved {filename} with {len(output)} records.")


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

    supply_forecast = train_and_predict(supply_df, label="Supply")
    export_forecast(supply_forecast, "forecast_supply.json")

    # --- Demand Pipeline ---
    demand_raw = load_demand_data()
    demand_df = process_demand_data(demand_raw)

    demand_forecast = train_and_predict(demand_df, label="Demand")
    export_forecast(demand_forecast, "forecast_demand.json")

    print("\n✅ Forecasting complete.")


if __name__ == "__main__":
    main()
