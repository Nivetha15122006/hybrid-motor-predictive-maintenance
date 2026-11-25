from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib

app = FastAPI(title="Hybrid Equipment Telemetry Diagnostics API")

bundle = joblib.load("motor_health_model.pkl")
rf_model = bundle["rf_model"]
autoencoder = bundle["autoencoder"]
scaler = bundle["scaler"]
anomaly_threshold = bundle["anomaly_threshold"]
feature_names = bundle["features"]

class SensorInput(BaseModel):
    machine_type: str
    air_temp: float
    proc_temp: float
    rpm: int
    torque: float
    tool_wear: int

@app.get("/")
def serve_ui():
    return FileResponse("index.html")

@app.post("/predict")
def predict_health(data: SensorInput):
    type_code = {"L": 0, "M": 1, "H": 2}.get(data.machine_type, 1)
    temp_diff = round(data.proc_temp - data.air_temp, 2)
    power_w = round(data.torque * (data.rpm * 2 * np.pi / 60), 1)

    max_wear = {0: 200, 1: 220, 2: 240}[type_code]
    rem_life = max(0.0, round((1 - data.tool_wear / max_wear) * 100, 1))

    input_df = pd.DataFrame([[
        type_code, data.air_temp, data.proc_temp, data.rpm,
        data.torque, data.tool_wear, temp_diff, power_w
    ]], columns=feature_names)

    # Engine 1: Supervised Random Forest Risk
    prob = float(rf_model.predict_proba(input_df)[0][1])

    # Engine 2: Unsupervised Neural Autoencoder Anomaly Score
    scaled_input = scaler.transform(input_df)
    reconstructed = autoencoder.predict(scaled_input)
    mse = float(np.mean((scaled_input - reconstructed) ** 2))
    anomaly_index = round(mse / anomaly_threshold, 2)
    is_anomaly = anomaly_index > 1.25

    alerts = []
    if prob >= 0.50:
        if temp_diff < 8.6 and data.rpm < 1380:
            alerts.append("Heat Dissipation Failure: Low thermal gradient & insufficient cooling RPM.")
        if power_w < 3500 or power_w > 9000:
            alerts.append("Power Failure: Mechanical power outside safe 3.5 kW – 9.0 kW operating envelope.")
        if data.tool_wear > 200:
            alerts.append("Tool Wear Failure: Cumulative tool wear exceeded rated limit.")
        if not alerts:
            alerts.append("Overstrain / Combined Telemetry Anomaly detected by Random Forest.")

    if is_anomaly and prob < 0.50:
        alerts.append(f"Neural Autoencoder Drift Warning: Sensor combination deviates from healthy baseline (Anomaly Index: {anomaly_index}x).")

    importances = {
        name: round(float(val) * 100, 1)
        for name, val in zip(feature_names, rf_model.feature_importances_)
    }

    is_danger = (prob >= 0.50) or is_anomaly

    return {
        "failure_prob": round(prob * 100, 1),
        "anomaly_index": anomaly_index,
        "status": "HIGH FAILURE RISK" if prob >= 0.50 else ("ANOMALY DRIFT DETECTED" if is_anomaly else "NORMAL OPERATION"),
        "is_danger": is_danger,
        "remaining_life": rem_life,
        "power_w": power_w,
        "temp_diff": temp_diff,
        "alerts": alerts,
        "importances": importances
    }