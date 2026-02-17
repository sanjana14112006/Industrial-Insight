import pandas as pd
import numpy as np
import zipfile

#----------------------------------------------------------------------------Loading and Merging All Training Data------------------------------------------------------------------------------------
# Define the standard 26 column names
index_names = ['unit_nr', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['s_{}'.format(i) for i in range(1, 22)] 
col_names = index_names + setting_names + sensor_names

zip_path = r'C:\Users\sanja\Downloads\CMAPSSData.zip'
all_training_data = []

# Loop through all 4 training files
with zipfile.ZipFile(zip_path) as z:
    for i in range(1, 5):
        filename = f'train_FD00{i}.txt'
        with z.open(filename) as f:
            # Load the space-separated file
            df_temp = pd.read_csv(f, sep=r'\s+', header=None, names=col_names)
            # Add a column to track which sub-dataset this is
            df_temp['dataset_id'] = i
            all_training_data.append(df_temp)

# Combine into one master DataFrame
train_df = pd.concat(all_training_data, ignore_index=True)
print(f"Total Rows Loaded: {len(train_df)}")



# --------------------------------------------Calculating the Target Variable (RUL)----------------------------------------------------------------------------------------

# 1. Group by dataset and unit to find the maximum life cycle of each engine
max_cycles = train_df.groupby(['dataset_id', 'unit_nr'])['time_cycles'].max().reset_index()
max_cycles.columns = ['dataset_id', 'unit_nr', 'max_cycle']

# 2. Merge this information back into the master DataFrame
train_df = train_df.merge(max_cycles, on=['dataset_id', 'unit_nr'])

# 3. Calculate RUL: (Total Life - Current Age)
train_df['RUL'] = train_df['max_cycle'] - train_df['time_cycles']

# 4. Clean up the helper column
train_df.drop('max_cycle', axis=1, inplace=True)

print("RUL Calculation Complete!")
print(train_df[['dataset_id', 'unit_nr', 'time_cycles', 'RUL']].head())


#------------------------------------------------------- intial data cleaning -------------------------------------------------------------------------------------------------

# Drop constant sensors that provide no predictive value
# Sensors s_1, s_5, s_10, s_16, s_18, and s_19 are typically constant in CMAPSS
constant_sensors = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']
train_df.drop(columns=constant_sensors, inplace=True)

print(f"Remaining Columns after Cleaning: {train_df.columns.tolist()}")



# ------------------------------------------------------------Calculating Rolling Means---------------------------------------------------------------------------------------------

# List of sensor columns that remain after cleaning
# (Recall we already dropped the constant sensors s_1, s_5, etc.)
active_sensors = [col for col in train_df.columns if col.startswith('s_')]
window_size = 10

# Calculate rolling mean for each sensor per engine
# We use 'closed=right' to ensure we only use past data (essential for real-time)
df_rolling = train_df.groupby(['dataset_id', 'unit_nr'])[active_sensors].rolling(window=window_size).mean().reset_index(level=[0,1], drop=True)

# Rename columns to avoid confusion
df_rolling.columns = [col + '_rolling_mean' for col in active_sensors]

# Join the new features back to the main dataframe
train_df = train_df.join(df_rolling)

# Note: The first 9 rows of each engine will be NaN because the window isn't full yet
# We fill these with the first available value to keep the data complete
train_df.fillna(method='bfill', inplace=True)



#-----------------------------------------------------------------Calculating FFT Features--------------------------------------------------------------------------------------------


def calculate_fft_energy(signal):
    # Perform FFT and take the magnitude of the frequencies
    fft_values = np.fft.fft(signal)
    fft_mag = np.abs(fft_values)
    # Return the average energy (mean of magnitudes)
    return np.mean(fft_mag)

# We will apply this to a specific vibration-sensitive sensor like s_7 or s_12
# For a hackathon, doing this for 2-3 key sensors is enough to show the 'spectrum' part
target_sensors = ['s_7', 's_12']

for sensor in target_sensors:
    feature_name = f'{sensor}_fft_energy'
    train_df[feature_name] = train_df.groupby(['dataset_id', 'unit_nr'])[sensor].transform(
        lambda x: x.rolling(window=20).apply(calculate_fft_energy)
    )

# Fill the initial NaN values created by the rolling window
train_df.fillna(method='bfill', inplace=True)


# ------------------------------------------------------------- Model Training & Evaluation---------------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib  # To save the model for real-time use

# 1. Select Features and Target
# We exclude ID columns and the target 'RUL' from our features
features = [col for col in train_df.columns if col not in ['unit_nr', 'time_cycles', 'dataset_id', 'RUL']]
X = train_df[features]
y = train_df['RUL']

# 2. Split into Training and Validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale the data (Min-Max Scaling)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# 4. Train the Model
print("Training Random Forest Model... this may take a minute...")
model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

# 5. Evaluate Accuracy
y_pred = model.predict(X_val_scaled)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Model Training Complete! Validation RMSE: {rmse:.2f} cycles")

# 6. SAVE the model and scaler for the real-time pipeline
joblib.dump(model, 'pdm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and Scaler saved as 'pdm_model.pkl' and 'scaler.pkl'")

