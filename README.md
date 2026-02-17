# 🌱 Industrial Insight: Real-Time AI Monitoring

**Industrial Insight** is an end-to-end Predictive Maintenance solution designed to maximize the lifespan of industrial assets. By moving from scheduled to **condition-based maintenance**, this project directly supports the "Green Bharat" mission by reducing premature e-waste from machinery components.

## 🚀 Key Features
- **Real-Time IoT Pipeline**: Simulates live sensor data from industrial engines via MQTT protocols.
- **Edge Analytics**: Implements Fast Fourier Transform (FFT) and Rolling Mean calculations to extract degradation signals.
- **AI Brain**: Utilizes a Random Forest Regressor trained on the NASA CMAPSS dataset to predict Remaining Useful Life (RUL).
- **Dynamic Dashboard**: A Streamlit interface providing live health metrics and automated maintenance reporting.

## 🛠️ Technical Stack
- **Language**: Python 3.13
- **Machine Learning**: Scikit-Learn, Pandas, NumPy, Joblib
- **IoT & UI**: Paho-MQTT, Streamlit

## 🏁 How to Run
1. **Clone the repository**:
   `git clone https://github.com/sanjana14112006/Industrial-Insight.git`
2. **Install Dependencies**:
   `pip install -r requirements.txt`
3. **Start the Pipeline**:
   - Run `python consumer.py` (AI Inference Engine)
   - Run `streamlit run dashboard.py` (Visual Dashboard)
   - Run `python producer.py` (Sensor Simulator)

## 📊 Model Performance
- **Metric**: Validation RMSE: 47.78 cycles. we can improve it !
- **Impact**: This precision allows for optimized maintenance windows, ensuring 95%+ of a component's safe life is utilized before replacement.

## 🌍 Sustainability Impact
By predicting failure before it occurs, this system prevents the disposal of functional parts, reducing the carbon footprint of industrial manufacturing and supporting a circular economy.
