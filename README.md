# Hybrid Equipment Telemetry & Predictive Maintenance System ⚙️📉

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Deep Learning](https://img.shields.io/badge/DL-Neural%20Autoencoder-blue.svg)](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)

An AI-powered, real-time industrial equipment diagnostics and telemetry monitoring dashboard. The system implements a **Hybrid Machine Learning & Deep Learning pipeline**—combining a **Supervised Random Forest Classifier** with an **Unsupervised Deep Neural Autoencoder**—to predict machine failure probability, detect zero-day sensor drift, estimate remaining tool life, and diagnose root-cause mechanical faults.

---

## 📖 Table of Contents
1. [Key Features](#-key-features)
2. [Tech Stack](#-tech-stack)
3. [System Architecture](#-system-architecture)
4. [Hybrid ML & Deep Learning Pipeline](#-hybrid-ml--deep-learning-pipeline)
5. [Dataset & Physics Feature Engineering](#-dataset--physics-feature-engineering)
6. [Project Structure](#-project-structure)
7. [Getting Started Locally](#-getting-started-locally)
8. [License](#-license)

---

## 🌟 Key Features

- **Hybrid Dual-Engine Diagnostics:** Combines **Supervised Classification** (Random Forest) for known historical breakdown patterns with **Unsupervised Anomaly Detection** (Neural Autoencoder) to catch unseen, zero-day sensor drift.
- **Physics-Informed Telemetry Derivation:** Automatically computes real-time **Mechanical Operating Power** ($P = \tau \cdot \omega$) and **Thermal Gradient** ($\Delta T = T_{\text{process}} - T_{\text{air}}$) from raw sensor streams.
- **Root-Cause Fault Localization & Operator Guidance:** Translates raw probabilities and reconstruction errors into plain-English maintenance advisories, pinpointing **Heat Dissipation Failures**, **Power Envelope Violations**, or **Tool Wear Overstrain**.
- **Interactive Real-Time Visualizations:** Features a live **Telemetry Risk & Anomaly Trend Line Chart** (tracking the last 12 sensor readings) alongside a **Sensor Profile vs. Healthy Baseline Radar Chart** powered by Chart.js.
- **One-Click Fault Simulation Presets:** Includes built-in test scenarios (*Simulate Normal*, *Simulate Heat Fault*, and *Simulate Zero-Day Drift*) alongside live sensor sliders and a Remaining Useful Life (RUL) wear-budget tracker.

---

## 🛠️ Tech Stack

- **Machine Learning & Deep Learning:** Python, Scikit-Learn (`RandomForestClassifier`, `MLPRegressor`), Pandas, NumPy, Joblib
- **Backend & REST API:** FastAPI, Uvicorn, Pydantic
- **Frontend & Data Visualization:** HTML5, CSS3 (Custom Industrial Dark Theme), Vanilla JavaScript (ES6), Chart.js (CDN)

---

## ⚙️ System Architecture

```text
  [Live Sensor Sliders / Fault Presets]
                   |
                   v
      (POST /predict — FastAPI Backend)
                   |
                   +---> [Physics Engine] ---------> Mechanical Power (W) & Thermal ΔT (K)
                   |
                   +---> [Engine 1: Random Forest] -> Known Failure Probability (%) + Feature Drivers
                   |
                   +---> [Engine 2: Autoencoder] ---> Zero-Day Anomaly Drift Score (MSE Ratio)
                   |
                   v
  [Diagnostics Dashboard: Status Banner, Operator Summary, Trend & Radar Charts]
```

---

## 🧠 Hybrid ML & Deep Learning Pipeline

The system runs two complementary models trained on the UCI AI4I 2020 industrial telemetry dataset:

### 1. Engine 1 — Supervised Failure Classifier
- **Algorithm:** Cost-Sensitive Random Forest Classifier (`n_estimators=200`, `max_depth=12`, `class_weight="balanced"`)
- **Role:** Learns non-linear decision boundaries across historical healthy and failed operating states while handling severe real-world class imbalance (~3.39% failure rate).
- **Outputs:** Failure probability score ($0\%$–$100\%$), root-cause alert triggers, and global Gini feature importance rankings.

### 2. Engine 2 — Unsupervised Deep Neural Autoencoder
- **Architecture:** Bottleneck Multi-Layer Perceptron (`8 → 16 → 3 → 16 → 8`, ReLU activation, Adam optimizer)
- **Role:** Trained **exclusively on healthy baseline telemetry** (`Machine failure == 0`) with `StandardScaler` normalization. When abnormal sensor combinations occur, reconstruction Mean Squared Error (MSE) spikes relative to the 95th-percentile healthy boundary.
- **Outputs:** Normalized **Anomaly Drift Index** (flagging early-stage drift whenever score exceeds `1.25x` baseline).

---

## 📊 Dataset & Physics Feature Engineering

- **Dataset (`ai4i2020.csv`):** Official **UCI AI4I 2020 Predictive Maintenance Dataset** containing **10,000 industrial equipment records** across Low (`L`), Medium (`M`), and High (`H`) machine quality grades.
- **Raw Telemetry Signals:** Air Temperature (`K`), Process Temperature (`K`), Rotational Speed (`RPM`), Torque (`Nm`), and Cumulative Tool Wear (`min`).
- **Engineered Physics Features:**
  - **Mechanical Power (`Power_W`):** $\text{Torque [Nm]} \times \left(\text{RPM} \times \frac{2\pi}{60}\right)$
  - **Thermal Gradient (`Temp_Diff_K`):** $\text{Process Temperature [K]} - \text{Air Temperature [K]}$

---

## 📁 Project Structure

```text
hybrid-motor-predictive-maintenance/
├── ai4i2020.csv              # UCI AI4I 2020 Predictive Maintenance dataset (10,000 rows)
├── train_model.py            # Feature engineering, dual-engine training & model serialization
├── motor_health_model.pkl    # Serialized Random Forest, Autoencoder, Scaler & thresholds
├── api.py                    # FastAPI REST inference server & root-cause diagnostic logic
├── index.html                # Interactive dark-mode UI with Chart.js trend & radar graphs
├── requirements.txt          # Python package dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # Project documentation
```

---

## 🚀 Getting Started Locally

### Prerequisites
- Python 3.10 or higher
- Git

### Step-by-Step Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Nivetha15122006/hybrid-motor-predictive-maintenance.git](https://github.com/Nivetha15122006/hybrid-motor-predictive-maintenance.git)
   cd hybrid-motor-predictive-maintenance
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   python -m pip install -r requirements.txt
   ```

4. **Train the Hybrid ML & Deep Learning Models:**
   Generates `motor_health_model.pkl` and logs ROC-AUC evaluation metrics:
   ```bash
   python train_model.py
   ```

5. **Run the FastAPI Diagnostics Server:**
   ```bash
   python -m uvicorn api:app --reload
   ```
   Open your browser and navigate to **`http://127.0.0.1:8000`**.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 💻 Built By

**Nivetha G** — M.Sc. AI & ML, Coimbatore Institute of Technology
- **GitHub:** [@Nivetha15122006](https://github.com/Nivetha15122006)