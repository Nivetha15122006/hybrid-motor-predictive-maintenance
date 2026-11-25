import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

print("1. Fetching UCI AI4I 2020 Predictive Maintenance dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
df = pd.read_csv(url)
df.to_csv("ai4i2020.csv", index=False)

# Physics-informed Feature Engineering
df["Temp_Diff_K"] = df["Process temperature [K]"] - df["Air temperature [K]"]
df["Power_W"] = df["Torque [Nm]"] * (df["Rotational speed [rpm]"] * 2 * np.pi / 60)
df["Type_Code"] = df["Type"].map({"L": 0, "M": 1, "H": 2})

features = [
    "Type_Code",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Temp_Diff_K",
    "Power_W"
]

X = df[features]
y = df["Machine failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- ENGINE 1: Supervised Random Forest (Known Failure Classification) ---
print("2. Training Engine 1: Balanced Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight="balanced",
    random_state=42
)
rf_model.fit(X_train, y_train)

rf_probs = rf_model.predict_proba(X_test)[:, 1]
rf_preds = rf_model.predict(X_test)
print(f"\n[Random Forest] ROC-AUC Score: {roc_auc_score(y_test, rf_probs):.4f}")
print(classification_report(y_test, rf_preds))

# --- ENGINE 2: Unsupervised Deep Neural Autoencoder (Zero-Day Anomaly Detection) ---
print("3. Training Engine 2: Deep Neural Autoencoder (Healthy Baseline Only)...")
scaler = StandardScaler()
X_train_healthy = X_train[y_train == 0]
X_train_healthy_scaled = scaler.fit_transform(X_train_healthy)

# Bottleneck neural architecture: 8 -> 16 -> 3 -> 16 -> 8
autoencoder = MLPRegressor(
    hidden_layer_sizes=(16, 3, 16),
    activation="relu",
    solver="adam",
    max_iter=300,
    random_state=42
)
autoencoder.fit(X_train_healthy_scaled, X_train_healthy_scaled)

# 95th percentile reconstruction MSE on healthy data = baseline anomaly boundary
healthy_recon = autoencoder.predict(X_train_healthy_scaled)
healthy_mse = np.mean((X_train_healthy_scaled - healthy_recon) ** 2, axis=1)
anomaly_threshold = float(np.percentile(healthy_mse, 95))

# Evaluate Autoencoder separation on test set
X_test_scaled = scaler.transform(X_test)
test_recon = autoencoder.predict(X_test_scaled)
test_mse = np.mean((X_test_scaled - test_recon) ** 2, axis=1)
print(f"[Neural Autoencoder] Zero-Day Anomaly ROC-AUC: {roc_auc_score(y_test, test_mse):.4f}")
print(f"[Neural Autoencoder] Healthy 95th-Percentile MSE Threshold: {anomaly_threshold:.4f}")

# Save dual-engine bundle
joblib.dump({
    "rf_model": rf_model,
    "autoencoder": autoencoder,
    "scaler": scaler,
    "anomaly_threshold": anomaly_threshold,
    "features": features
}, "motor_health_model.pkl")

print("\n4. Saved 'motor_health_model.pkl' and 'ai4i2020.csv' successfully!")