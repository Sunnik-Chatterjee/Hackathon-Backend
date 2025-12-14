import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# =====================================================
# APP
# =====================================================
app = FastAPI(title="Flight Delay Prediction API")

# =====================================================
# PATHS
# =====================================================
DATA_PATH = "flights_full.csv"
MODEL_PATH = "best_rf_model.pkl"

# =====================================================
# LOAD MODEL FIRST (IMPORTANT)
# =====================================================
model = joblib.load(MODEL_PATH)

# =====================================================
# LOAD & NORMALIZE DATASET
# =====================================================
def normalize_delay_column(df):
    candidates = [
        "delayminutes", "delay_minutes", "delay_min",
        "dep_delay", "arr_delay", "arrival_delay"
    ]
    for col in candidates:
        if col in df.columns:
            df["delayminutes"] = df[col]
            return df
    raise ValueError(f"No delay column found. Columns: {list(df.columns)}")


df = pd.read_csv(DATA_PATH)
df = normalize_delay_column(df)

# ---------------- SANITY CHECK ----------------
required_cols = {
    "airline", "origin", "destination",
    "scheduled_dep", "distance_km",
    "scheduled_duration_min", "delayminutes"
}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Dataset missing columns: {missing}")

df["scheduled_dep"] = pd.to_datetime(df["scheduled_dep"])
df["hour"] = df["scheduled_dep"].dt.hour
df["weekday"] = df["scheduled_dep"].dt.weekday
df["is_sunday"] = (df["weekday"] == 6).astype(int)

# =====================================================
# STATISTICS (FOR CALIBRATION / FALLBACK)
# =====================================================
route_airline_stats = (
    df.groupby(["origin", "destination", "airline"])["delayminutes"]
    .mean()
    .to_dict()
)
airline_stats = df.groupby("airline")["delayminutes"].mean().to_dict()
route_stats = df.groupby(["origin", "destination"])["delayminutes"].mean().to_dict()
GLOBAL_MEAN_DELAY = df["delayminutes"].mean()

# =====================================================
# HELPERS
# =====================================================
def resolve_airline_delay(origin, dest, airline):
    if (origin, dest, airline) in route_airline_stats:
        return route_airline_stats[(origin, dest, airline)]
    if airline in airline_stats:
        return airline_stats[airline]
    if (origin, dest) in route_stats:
        return route_stats[(origin, dest)]
    return GLOBAL_MEAN_DELAY


def route_baseline(origin, dest):
    return route_stats.get((origin, dest), GLOBAL_MEAN_DELAY)


def smooth_delay(raw, baseline):
    return 0.65 * raw + 0.35 * baseline


def estimate_duration(distance):
    return int(distance / 10)


# =====================================================
# FEATURE VECTOR BUILDER (CRITICAL FIX)
# =====================================================
def build_feature_vector(origin, dest, airline, dep_time, distance):
    X = pd.DataFrame(
        0,
        index=[0],
        columns=model.feature_names_in_
    )

    # numeric features
    for col, val in {
        "distance_km": distance,
        "hour": dep_time.hour,
        "is_sunday": int(dep_time.weekday() == 6)
    }.items():
        if col in X.columns:
            X.at[0, col] = val

    # one-hot encoded categorical features
    for col in [
        f"airline_{airline}",
        f"origin_{origin}",
        f"destination_{dest}"
    ]:
        if col in X.columns:
            X.at[0, col] = 1

    return X


# =====================================================
# INPUT SCHEMA
# =====================================================
class FlightInput(BaseModel):
    origin: str
    destination: str
    scheduled_dep: str


# =====================================================
# PREDICTION LOGIC
# =====================================================
def predict_delay_api(input_json):
    origin = input_json["origin"]
    dest = input_json["destination"]
    dep_time = pd.to_datetime(input_json["scheduled_dep"])

    route_rows = df[(df["origin"] == origin) & (df["destination"] == dest)]
    distance = (
        route_rows["distance_km"].mean()
        if not route_rows.empty
        else df["distance_km"].mean()
    )

    baseline = route_baseline(origin, dest)
    predictions = []

    airlines = sorted(df["airline"].unique())

    for airline in airlines:
        X_input = build_feature_vector(
            origin=origin,
            dest=dest,
            airline=airline,
            dep_time=dep_time,
            distance=distance
        )

        # handle fully unseen combinations safely
        if X_input.sum(axis=1).iloc[0] == 0:
            raw_delay = GLOBAL_MEAN_DELAY
        else:
            raw_delay = model.predict(X_input)[0]

        final_delay = smooth_delay(raw_delay, baseline)

        # time-of-day correction
        if dep_time.hour < 7:
            final_delay -= 3
        elif dep_time.hour > 18:
            final_delay += 4

        final_delay = max(0, final_delay)
        delay_prob = min(95, max(5, final_delay * 1.7))

        predictions.append({
            "airline": airline,
            "prediction": "Delayed" if final_delay > 15 else "On Time",
            "predicted_delay_minutes": round(final_delay, 1),
            "delay_probability_percent": round(delay_prob, 1),
            "autofilled_data": {
                "airline": airline,
                "distance_km": int(distance),
                "scheduled_duration_min": estimate_duration(distance),
                "weather": "Clear",
                "origin_traffic": "Moderate",
                "dest_traffic": "Moderate"
            }
        })

    return {
    "origin": origin,
    "destination": dest,
    "predictions": predictions
}



# =====================================================
# ENDPOINTS
# =====================================================
@app.post("/predict")
def predict(input_data: FlightInput):
    return predict_delay_api(input_data.dict())


@app.get("/")
def root():
    return {
        "message": "Flight Delay Prediction API (FIXED)",
        "total_records": len(df),
        "routes": df[["origin", "destination"]].drop_duplicates().shape[0],
        "airlines": sorted(df["airline"].unique())
    }
