# 🚦 Sustainable NYC Taxi Trip Prediction using Machine Learning

## 📌 Overview

Urban transportation contributes significantly to fuel consumption and carbon emissions. Traffic congestion, inefficient routing, and long idle times increase the environmental impact of taxi services.

This project uses **Machine Learning (XGBoost)** to predict NYC Green Taxi trip durations and analyze operational efficiency. By forecasting trip duration and average speed, the system helps identify **efficient, low-emission routes** that can support sustainable urban transportation.

The project also includes an interactive **Streamlit web application** for real-time trip prediction and model diagnostics.

---

# 🎯 Problem Statement

Taxi inefficiencies such as:
- Traffic congestion
- Longer trip durations
- Low-speed routes
- Idle waiting times

lead to:
- Higher fuel consumption
- Increased CO₂ emissions
- Reduced transportation efficiency

The objective of this project is to:
- Predict trip duration accurately
- Estimate operational efficiency
- Support eco-friendly transportation decisions

---

# 💡 Solution

This system:
- Predicts taxi trip duration
- Estimates average trip speed
- Detects efficient vs inefficient routes
- Provides a real-time prediction dashboard
- Visualizes traffic and congestion trends

---

# 🧠 Machine Learning Pipeline

## 🔹 Data Source
NYC Green Taxi Trip Dataset (Parquet format)

---

## 🔹 Data Preprocessing
- Removed invalid and unrealistic trips
- Filtered trips:
  - Duration < 1 minute
  - Duration > 60 minutes
  - Speed > 100 mph
- Calculated:
  - Trip duration
  - Average speed

---

## 🔹 Feature Engineering
Features used:
- Pickup Location ID
- Dropoff Location ID
- Pickup Hour
- Pickup Day of Week
- Holiday Indicator
- Trip Distance

Additional engineering:
- Haversine distance calculation
- Holiday detection using `holidays`
- Time-based feature extraction

---

## 🔹 Model Used
### XGBoost Regressor
Hyperparameter tuning performed using:
- `RandomizedSearchCV`
- 5-fold cross-validation

---

# 📊 Evaluation Metrics

| Metric | Value |
|--------|--------|
| Mean Absolute Error (MAE) | ~2.8 minutes |

The model demonstrates:
- Good prediction accuracy
- Low bias
- Strong generalization

---

# 📈 Visualizations

The project includes:
- Actual vs Predicted Duration
- Residual Distribution
- Trip Duration Distribution
- Hourly Speed Profile

These visualizations help analyze:
- Prediction quality
- Traffic congestion
- Urban transportation efficiency

---

# 🌱 Sustainability Impact

Efficient routes reduce:
- Fuel consumption
- Travel time
- Traffic congestion
- CO₂ emissions

The project promotes:
- Sustainable transportation
- Smart urban mobility
- Efficient fleet management

---

# 💻 Streamlit Web Application

## Features
- Select pickup and dropoff locations
- Choose pickup day and hour
- Automatic distance calculation
- Predict trip duration
- Estimate average speed
- Identify route efficiency

---

# 📁 Project Structure

```bash
Sustainable_NYC_Taxi_ML/
│
├── app.py
├── emissions_prediction.ipynb
├── models/
│   ├── xgb_model.pkl
│   └── dict_vectorizer.pkl
│
├── green-2024-12.parquet
├── taxi_zone_lookup.csv
├── zone_distance_matrix.csv
├── zones_final.shp
├── requirements.txt
├── README.md
│
├── actual_vs_predicted.png
├── duration_distribution.png
├── hourly_speed_profile.png
└── model_residuals.png