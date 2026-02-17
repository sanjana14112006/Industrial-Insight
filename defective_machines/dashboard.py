import streamlit as st
import pandas as pd
import paho.mqtt.client as mqtt
import json
import joblib
import numpy as np
from collections import deque

# 1. Page Config & Branding
st.set_page_config(page_title="Industrial Insight Dashboard", layout="wide")
st.title("🌱 Industrial Insight: Real-Time AI Monitoring")
st.markdown("### Predictive Maintenance for a Sustainable Future")

# 2. INITIALIZE SESSION STATE FIRST (To prevent KeyError)
if 'sensor_history' not in st.session_state:
    st.session_state.sensor_history = deque(maxlen=20)
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = 0.0

# 3. Load the AI Brain
# IMPORTANT: Re-run main.py once before this to ensure version compatibility
model = joblib.load('pdm_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define active sensors (matching main.py cleaning logic)
constant_sensors = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']
feature_cols = [f's_{i}' for i in range(1, 22) if f's_{i}' not in constant_sensors]

# 4. Processing & Prediction Logic
def calculate_fft_energy(signal):
    return np.mean(np.abs(np.fft.fft(signal)))

def predict_rul(history_df):
    if len(history_df) < 20:
        return 0.0
    
    # Extract the most recent data point
    latest_row = history_df.iloc[-1].copy()
    
    # Feature Engineering: Rolling Mean (last 10)
    for s in feature_cols:
        latest_row[f'{s}_rolling_mean'] = history_df[s].tail(10).mean()
        
    # Feature Engineering: FFT Energy (last 20)
    latest_row['s_7_fft_energy'] = calculate_fft_energy(history_df['s_7'])
    latest_row['s_12_fft_energy'] = calculate_fft_energy(history_df['s_12'])
    
    # Prepare input for model (Drop ID columns)
    input_df = pd.DataFrame([latest_row]).drop(columns=['unit_nr', 'time_cycles'])
    
    # Scale and Predict
    scaled_input = scaler.transform(input_df)
    return model.predict(scaled_input)[0]

# 5. MQTT Setup
def on_message(client, userdata, message):
    try:
        payload = json.loads(message.payload.decode("utf-8"))
        # Add to history
        st.session_state.sensor_history.append(payload)
        
        # If we have enough data, update the prediction
        if len(st.session_state.sensor_history) == 20:
            df = pd.DataFrame(list(st.session_state.sensor_history))
            st.session_state.last_prediction = predict_rul(df)
    except Exception as e:
        pass # Handle parsing errors silently in the background

# Setup MQTT Client (Note: Callback API version 1 is used per your current environment)
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, "Dashboard_UI")
client.on_message = on_message
client.connect("broker.hivemq.com", 1883)
client.subscribe("factory/sensor/data")
client.loop_start() 

# 6. UI Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Machine Health")
    st.metric(label="Predicted Remaining Life", value=f"{st.session_state.last_prediction:.1f} Cycles")
    
    if len(st.session_state.sensor_history) < 20:
        st.info(f"Gathering data stream... ({len(st.session_state.sensor_history)}/20)")
    elif st.session_state.last_prediction < 25:
        st.error("🚨 CRITICAL: Maintenance Required")
    else:
        st.success("✅ System Operating Normally")

with col2:
    st.subheader("Live Sensor Analytics")
    if len(st.session_state.sensor_history) > 0:
        chart_df = pd.DataFrame(list(st.session_state.sensor_history))
        # Visualizing the vibration-sensitive sensors
        st.line_chart(chart_df[['s_7', 's_12']])

st.button("Refresh View")

# 7. EXPORT MAINTENANCE REPORT (Sidebar)
st.sidebar.header("📋 Export Insights")
if len(st.session_state.sensor_history) > 0:
    unit_id = st.session_state.sensor_history[-1]['unit_nr']
    current_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_content = f"""INDUSTRIAL INSIGHT: MAINTENANCE REPORT
--------------------------------------
Timestamp: {current_time}
Engine Monitored: Unit #{unit_id}
Predicted RUL: {st.session_state.last_prediction:.2f} cycles
Health Status: {'CRITICAL' if st.session_state.last_prediction < 25 else 'OPTIMAL'}

Action Recommended: {'REPLACE PART IMMEDIATELY' if st.session_state.last_prediction < 25 else 'CONTINUE MONITORING'}
Sustainability Note: Optimized maintenance prevents unnecessary e-waste.
--------------------------------------"""

    st.sidebar.download_button(
        label="Download Health Report",
        data=report_content,
        file_name=f"Unit_{unit_id}_Report.txt",
        mime="text/plain"
    )
else:
    st.sidebar.write("Start stream to generate reports.")