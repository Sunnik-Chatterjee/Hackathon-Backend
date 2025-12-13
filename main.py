from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
import numpy as np

app = FastAPI()

# Load model & dataset
BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "best_rf_model.pkl")
df = pd.read_csv(BASE_DIR / "flights_full.csv", parse_dates=["scheduled_dep"])

class UserInput(BaseModel):
    origin: str
    destination: str
    scheduled_dep: str

def autofill_fields(origin, destination):
    route_df = df[(df["origin"] == origin) & (df["destination"] == destination)]
    if len(route_df) == 0:
        raise HTTPException(status_code=404, detail="Route not found in dataset.")
    
    return {
        "airline": route_df["airline"].mode()[0],
        "distance_km": int(route_df["distance_km"].mean()),
        "scheduled_duration_min": int(route_df["scheduled_duration_min"].mean()),
        "weather": route_df["weather"].mode()[0],
        "origin_traffic": route_df["origin_traffic"].mode()[0],
        "dest_traffic": route_df["dest_traffic"].mode()[0]
    }

@app.post("/predict", operation_id="flight_delay_predict")  # ✅ Unique ID
@app.post("/predict", operation_id="flight_delay_predict")
def predict(data: UserInput):
    origin = data.origin.upper()
    destination = data.destination.upper()
    scheduled_dep = data.scheduled_dep
    
    try:
        dt = datetime.strptime(scheduled_dep, "%Y-%m-%d %H:%M")
    except:
        raise HTTPException(status_code=400, detail="scheduled_dep must be 'YYYY-MM-DD HH:MM'")
    
    route_df = df[(df["origin"] == origin) & (df["destination"] == destination)]
    if len(route_df) == 0:
        raise HTTPException(status_code=404, detail="Route not found in dataset.")
    
    # ✅ TOP 5 Airlines for this route
    top_airlines = route_df["airline"].value_counts().head(5).index.tolist()
    
    predictions = []
    for airline in top_airlines:
        # Autofill for this specific airline
        auto = {
            "airline": airline,
            "distance_km": int(route_df[route_df["airline"] == airline]["distance_km"].mean()),
            "scheduled_duration_min": int(route_df[route_df["airline"] == airline]["scheduled_duration_min"].mean()),
            "weather": route_df["weather"].mode()[0],
            "origin_traffic": route_df["origin_traffic"].mode()[0],
            "dest_traffic": route_df["dest_traffic"].mode()[0]
        }
        
        # One-hot encode (same as before)
        input_data = {
            'airline': airline, 'origin': origin, 'destination': destination,
            'scheduled_dep': scheduled_dep, 'scheduled_duration_min': auto["scheduled_duration_min"],
            'distance_km': auto["distance_km"], 'weather': auto["weather"],
            'origin_traffic': auto["origin_traffic"], 'dest_traffic': auto["dest_traffic"],
            'dayofweek': dt.weekday(), 'hourofday': dt.hour
        }
        
        df_input = pd.DataFrame([input_data])
        categoricals = ['airline', 'origin', 'destination', 'weather', 'origin_traffic', 'dest_traffic']
        for col in categoricals:
            dummies = pd.get_dummies(df_input[col], prefix=col).astype(int)
            df_input = pd.concat([df_input.drop(col, axis=1), dummies], axis=1)
        
        df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)
        pred = model.predict(df_input)[0]
        is_delayed = "Delayed" if pred > 15 else "On Time"
        
        predictions.append({
            "airline": airline,
            "prediction": is_delayed,
            "predicted_delay_minutes": round(float(pred), 1),
            "autofilled_data": auto
        })
    
    return {
        "route": f"{origin} → {destination}",
        "top_airlines": top_airlines,
        "predictions": predictions
    }
