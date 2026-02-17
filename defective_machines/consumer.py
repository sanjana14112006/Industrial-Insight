import pandas as pd
import numpy as np
import joblib
import paho.mqtt.client as mqtt
import json
from collections import deque

# 1. LOAD THE BRAIN (Your .pkl files)
# These files must be in the same folder as this script
model = joblib.load('pdm_model.pkl')
scaler = joblib.load('scaler.pkl')

# Buffer to store the last 20 readings for feature engineering
# Needed to calculate the 'Rolling Mean' and 'FFT' on the fly
WINDOW_SIZE = 20
data_buffer = {}

# Exact list of features your model was trained on
features_list = [
    'setting_1', 'setting_2', 'setting_3', 's_2', 's_3', 's_4', 's_6', 's_7', 
    's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21',
    's_2_rolling_mean', 's_3_rolling_mean', 's_4_rolling_mean', 's_6_rolling_mean',
    's_7_rolling_mean', 's_8_rolling_mean', 's_9_rolling_mean', 's_11_rolling_mean',
    's_12_rolling_mean', 's_13_rolling_mean', 's_14_rolling_mean', 's_15_rolling_mean',
    's_17_rolling_mean', 's_20_rolling_mean', 's_21_rolling_mean', 
    's_7_fft_energy', 's_12_fft_energy'
]

def calculate_fft_energy(signal):
    return np.mean(np.abs(np.fft.fft(signal)))

# 2. DEFINE WHAT TO DO WHEN DATA ARRIVES
def on_message(client, userdata, message):
    try:
        payload = json.loads(message.payload.decode("utf-8"))
        unit_id = int(payload['unit_nr'])
        
        # Initialize buffer for a new engine unit
        if unit_id not in data_buffer:
            data_buffer[unit_id] = deque(maxlen=WINDOW_SIZE)
        
        data_buffer[unit_id].append(payload)
        
        # Once we have 20 cycles of data, we can start predicting
        if len(data_buffer[unit_id]) == WINDOW_SIZE:
            temp_df = pd.DataFrame(list(data_buffer[unit_id]))
            input_row = temp_df.iloc[-1].copy()
            
            # Feature Engineering: Calculate Rolling Means
            active_sensors = [col for col in temp_df.columns if col.startswith('s_')]
            for s in active_sensors:
                input_row[f'{s}_rolling_mean'] = temp_df[s].mean()
            
            # Feature Engineering: Calculate FFT Energy
            input_row['s_7_fft_energy'] = calculate_fft_energy(temp_df['s_7'])
            input_row['s_12_fft_energy'] = calculate_fft_energy(temp_df['s_12'])
            
            # Prepare data for prediction
            final_features = input_row[features_list].values.reshape(1, -1)
            scaled_features = scaler.transform(final_features)
            prediction = model.predict(scaled_features)[0]
            
            # Output the result
            print(f"Engine #{unit_id} | Predicted RUL: {prediction:.2f} cycles")
            
            # Alert Logic for Maintenance
            if prediction < 25:
                print(f"!!! ALERT: Engine #{unit_id} is in CRITICAL zone !!!")
                
    except Exception as e:
        print(f"Error processing message: {e}")

# 3. SET UP THE CONNECTION
# Add CallbackAPIVersion to satisfy the new paho-mqtt 2.0+ requirement
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, "Predictive_Maintenance_Brain")
client.on_message = on_message
client.connect("broker.hivemq.com", 1883)
client.subscribe("factory/sensor/data")

print("Consumer is live... awaiting sensor streams...")
client.loop_forever() # This keeps the script running
