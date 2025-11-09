import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack
import holidays # Needed for feature engineering
import os 
from sklearn.feature_extraction import DictVectorizer
from xgboost import XGBRegressor 


# --- 1. Data/Model Loading and Preparation ---

@st.cache_resource
def load_all_components():
    """Loads all model components, lookups, and the distance matrix."""
    try:
        # Load Model and Vectorizer (paths verified)
        model = joblib.load('models/xgb_model.pkl')
        vectorizer = joblib.load('models/dict_vectorizer.pkl')
        
        # Load Lookup Table
        zone_lookup_df = pd.read_csv('taxi_zone_lookup (2).csv')
        zone_lookup_df['Display_Name'] = zone_lookup_df['Zone'] + ' (' + zone_lookup_df['Borough'] + ')'
        name_to_id = zone_lookup_df.set_index('Display_Name')['LocationID'].to_dict()
        
        # Load Distance Matrix
        distance_matrix = pd.read_csv('zone_distance_matrix.csv')
        
        # Load Raw Data (Needed for graph generation)
        df_raw = pd.read_parquet('green-2024-12.parquet')
        
        return model, vectorizer, name_to_id, distance_matrix, df_raw
    
    except FileNotFoundError as e:
        st.error(f"File not found: {e}. Ensure the model files are in 'models/' and lookup/distance files are in the root.")
        return None, None, None, None, None

# Load all components
model, dv, name_to_id, distance_matrix, df_raw = load_all_components()

# --- 2. Shared Variables & Constants ---

DAYS = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
HOURS = {i: f'{i:02d}:00' for i in range(24)}
DISPLAY_NAMES = sorted([k for k in name_to_id.keys() if isinstance(k, str)])


# --- 3. Graph Generation Function (FIXED Syntax) ---

@st.cache_data
def generate_diagnostics_data(df_input, _model, _dv): 
    """Processes raw data, applies features, and predicts to generate diagnostic metrics."""
    
    # 3a. Re-apply cleaning and feature calculation 
    df_diag = df_input.copy()
    df_diag['duration'] = (df_diag['lpep_dropoff_datetime'] - df_diag['lpep_pickup_datetime']).dt.total_seconds() / 60
    df_diag['avg_speed_mph'] = np.where(df_diag['duration'] > 0, (df_diag['trip_distance'] / df_diag['duration']) * 60, 0)
    df_diag = df_diag[(df_diag['duration'] >= 1) & (df_diag['duration'] <= 60) & 
                     (df_diag['avg_speed_mph'] <= 100) & (df_diag['trip_distance'] > 0)].copy()

    # 3b. Re-create features for the model
    df_diag['pickup_hour'] = df_diag['lpep_pickup_datetime'].dt.hour
    df_diag['pickup_day_of_week'] = df_diag['lpep_pickup_datetime'].dt.dayofweek
    
    if not df_diag.empty:
        us_holidays = holidays.US(years=df_diag['lpep_pickup_datetime'].dt.year.unique())
        df_diag['is_holiday'] = df_diag['lpep_pickup_datetime'].apply(lambda date: 1 if pd.Timestamp(date).date() in us_holidays else 0)
    else:
        df_diag['is_holiday'] = 0
    
    feature_cols = ['PULocationID', 'DOLocationID', 'pickup_hour', 'pickup_day_of_week', 'is_holiday']
    df_diag[feature_cols] = df_diag[feature_cols].astype(str)
    
    # 3c. Vectorize and Predict
    X_diag_dict = df_diag[feature_cols].to_dict(orient='records')
    
    X_diag_sparse = _dv.transform(X_diag_dict) 
    X_diag_num = df_diag[['trip_distance']].values
    X_diag = hstack([X_diag_sparse, X_diag_num])
    
    df_diag['predicted_duration'] = _model.predict(X_diag) 
    df_diag['residual'] = df_diag['duration'] - df_diag['predicted_duration']
    
    return df_diag

# Generate the diagnostic data once
if df_raw is not None and model is not None:
    df_diagnostics = generate_diagnostics_data(df_raw, model, dv) 
else:
    df_diagnostics = None


# --- 4. Frontend Structure and Routing ---

st.set_page_config(page_title="NYC Sustainable Trip Prediction", layout="wide")
st.title("🚦 Least-Emission Trip Duration Predictor (Green Taxi)")

# Create Tabs
tab1, tab2 = st.tabs(["🌎 Predict Trip", "📈 Model Diagnostics"])


# --- TAB 1: PREDICT TRIP (Your Main Functionality) ---
with tab1:
    st.markdown("Use this tool to predict trip duration and estimate operational efficiency *before* the trip starts. Accuracy: **MAE 2.87 min**.")
    
    if model and dv:
        # --- Input Fields ---
        
        default_pickup_name = "JFK Airport (Queens)" 
        default_dropoff_name = "Long Island City/Hunters Point Avenue (Queens)"
        default_pickup_index = DISPLAY_NAMES.index(default_pickup_name) if default_pickup_name in DISPLAY_NAMES else 0
        default_dropoff_index = DISPLAY_NAMES.index(default_dropoff_name) if default_dropoff_name in DISPLAY_NAMES else 0

        # ********** FIX FOR LAYOUT **********
        
        # ROW 1: Pick-up Location and Drop-off Location (in one line)
        col_loc_p, col_loc_d = st.columns(2)

        with col_loc_p:
            pickup_name = st.selectbox("Pick-up Location", options=DISPLAY_NAMES, index=default_pickup_index, key="pickup_loc_key")
        
        with col_loc_d:
            dropoff_name = st.selectbox("Drop-off Location", options=DISPLAY_NAMES, index=default_dropoff_index, key="dropoff_loc_key")
            
        # ROW 2: Day of Week and Pickup Hour (in one line, below Row 1)
        col_day, col_hour = st.columns(2)

        with col_day:
            pickup_day = st.selectbox("Day of Week", options=list(DAYS.keys()), format_func=lambda x: DAYS[x], key="pickup_day_key")
        
        with col_hour:
            pickup_time = st.selectbox("Pickup Hour", options=list(HOURS.keys()), format_func=lambda x: HOURS[x], key="pickup_hour_key")
        
        # ***********************************
        
        st.markdown("---") # Separator line
        
        # --- Prediction Button and Logic ---
        if st.button("Predict Duration & Efficiency", type="primary", key="predict_button"):
            # 4a. MAP
            pickup_loc_id = name_to_id[pickup_name]
            dropoff_loc_id = name_to_id[dropoff_name]

            # 4b. AUTOMATIC DISTANCE CALCULATION
            try:
                trip_distance = distance_matrix.loc[
                    (distance_matrix['PULocationID'] == pickup_loc_id) & 
                    (distance_matrix['DOLocationID'] == dropoff_loc_id), 
                    'Min_Distance_Miles'
                ].iloc[0]
                st.info(f"Calculated Trip Distance (Haversine): {trip_distance:.2f} miles")
            except IndexError:
                st.error("Distance data not found for this zone pair.")
                st.stop()
                
            # 4c. Prepare Input Data for Model
            input_data = pd.DataFrame([{
                'PULocationID': str(pickup_loc_id), 'DOLocationID': str(dropoff_loc_id),
                'pickup_hour': str(pickup_time), 'pickup_day_of_week': str(pickup_day), 'is_holiday': '0'
            }])

            # 4d. Vectorize and Predict
            input_dict = input_data.to_dict(orient='records')
            X_categorical = dv.transform(input_dict)
            X_numerical = np.array([[trip_distance]]) 
            X_final = hstack([X_categorical, X_numerical])
            predicted_duration_min = model.predict(X_final)[0]
            predicted_avg_speed = (trip_distance / predicted_duration_min) * 60
            
            # 5. Display Results
            st.subheader("📊 Prediction Results")
            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1: st.metric("Predicted Duration", f"{predicted_duration_min:.2f} min")
            with col_res2: st.metric("Predicted Avg. Speed", f"{predicted_avg_speed:.1f} mph")
            with col_res3:
                if predicted_avg_speed >= 11.2: st.success("✅ HIGH EFFICIENCY ROUTE")
                else: st.warning("⚠️ AVERAGE/LOW EFFICIENCY")
                st.markdown("**Low-Emission Potential:** Based on optimized model prediction.")


# --- TAB 2: MODEL DIAGNOSTICS (The visualization area) ---
with tab2:
    if df_diagnostics is not None:
        st.header("Model Performance & Congestion Analysis")
        st.markdown("These charts demonstrate the model's predictive accuracy and the key drivers of urban congestion.")
        
        # Use two columns for the graphs
        chart_col1, chart_col2 = st.columns(2)
        chart_col3, chart_col4 = st.columns(2) # New row for charts 3 and 4

        # --- GRAPH 1: Actual vs. Predicted Duration (Fidelity Check) ---
        with chart_col1:
            st.subheader("1. Prediction Fidelity")
            fig, ax = plt.subplots(figsize=(6, 5))
            sample = df_diagnostics.sample(n=5000, random_state=42)
            
            ax.scatter(sample['duration'], sample['predicted_duration'], alpha=0.3, s=10, color='blue')
            max_val = df_diagnostics['duration'].max()
            ax.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='Perfect Prediction (y=x)')
            
            ax.set_title('Actual vs. Predicted Trip Duration')
            ax.set_xlabel('Actual Duration (Minutes)')
            ax.set_ylabel('Predicted Duration (Minutes)')
            ax.set_xlim(0, 60)
            ax.set_ylim(0, 60)
            st.pyplot(fig) 
            plt.close(fig) 

        # --- GRAPH 2: Average Speed by Time of Day (Efficiency Profile) ---
        with chart_col2:
            st.subheader("2. Fleet Efficiency by Hour")
            hourly_speed = df_diagnostics.groupby('pickup_hour')['avg_speed_mph'].median().reset_index()

            fig, ax = plt.subplots(figsize=(6, 5))
            sns.lineplot(x='pickup_hour', y='avg_speed_mph', data=hourly_speed, marker='o', ax=ax)
            ax.axhline(11.2, color='gray', linestyle=':', label='Fleet Average')
            
            ax.set_title('Median Speed by Hour of Day')
            ax.set_xlabel('Hour of Day (0-23)')
            ax.set_ylabel('Median Average Speed (mph)')
            ax.set_xticks(range(0, 24, 3))
            st.pyplot(fig) 
            plt.close(fig)

        # --- GRAPH 3: Residual Distribution (Bias Check) ---
        with chart_col3:
            st.subheader("3. Model Residuals (Bias Check)")
            fig, ax = plt.subplots(figsize=(6, 5))
            
            sns.histplot(df_diagnostics['residual'], bins=50, kde=True, color='purple', ax=ax)
            mean_error = df_diagnostics['residual'].mean()
            ax.axvline(mean_error, color='red', linestyle='--', label=f'Mean Error (Bias): {mean_error:.2f} min')
            
            ax.set_title('Distribution of Prediction Errors')
            ax.set_xlabel('Prediction Error (Minutes)')
            ax.legend()
            st.pyplot(fig) 
            plt.close(fig)

        # --- GRAPH 4: Trip Duration Distribution (Target Variable) ---
        with chart_col4:
            st.subheader("4. Log-Transformed Trip Duration")
            fig, ax = plt.subplots(figsize=(6, 5))
            
            sns.histplot(np.log1p(df_diagnostics['duration']), bins=50, kde=True, ax=ax) 
            
            ax.set_title('Log(Duration) Distribution')
            ax.set_xlabel('Log(1 + Trip Duration in Minutes)')
            ax.set_ylabel('Count')
            st.pyplot(fig) 
            plt.close(fig)


    else:
        st.warning("Cannot generate diagnostic graphs. Please ensure the raw Parquet file ('green-2024-12.parquet') is available and accessible.")

# Final check for model loading message if it failed
if model is None or dv is None:
    st.error("Application failed to load necessary ML components. Check the terminal for FileNotFoundError details.")