import pandas as pd
import paho.mqtt.client as mqtt
import json
import time
import zipfile

# 1. SETUP THE MQTT CLIENT
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, "Industrial_Sensor_Simulator")
client.connect("broker.hivemq.com", 1883)

# 2. DATA SOURCE SETUP
# Make sure this path matches your actual ZIP file location
zip_path = r'C:\Users\sanja\Downloads\CMAPSSData.zip'
index_names = ['unit_nr', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
col_names = index_names + setting_names + sensor_names

# 3. STREAMING LOGIC
print("Starting Real-Time Sensor Stream...")

try:
    with zipfile.ZipFile(zip_path) as z:
        with z.open('test_FD001.txt') as f:
            # We read the file row-by-row to simulate a live stream
            for chunk in pd.read_csv(f, sep=r'\s+', header=None, names=col_names, chunksize=1):
                data_point = chunk.iloc[0].to_dict()
                
                # Convert to JSON for transmission
                payload = json.dumps(data_point)
                
                # Publish to the topic the Consumer is listening to
                client.publish("factory/sensor/data", payload)
                
                print(f"Sent: Engine #{data_point['unit_nr']} | Cycle {data_point['time_cycles']}")
                
                # Send one cycle per second for the demo
                time.sleep(1) 
except Exception as e:
    print(f"Producer Error: {e}")

client.disconnect()



