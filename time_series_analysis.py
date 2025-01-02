import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class TimeSeriesAnalyzer:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.lstm_model = None
        self.arima_model = None
    
    def prepare_sequence_data(self, data, lookback=60):
        """Prepare data for LSTM model."""
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        X, y = [], []
        
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
            
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, lookback):
        """Build LSTM model for time series prediction."""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        self.lstm_model = model
        return model
    
    def fit_arima(self, data):
        """Fit ARIMA model."""
        self.arima_model = ARIMA(data, order=(5,1,0))
        self.arima_model = self.arima_model.fit()
        return self.arima_model
    
    def predict_next_period(self, data, model_type='lstm'):
        """Make predictions for the next period."""
        if model_type == 'lstm':
            # Prepare data and make LSTM prediction
            scaled_data = self.scaler.transform(data.reshape(-1, 1))
            prediction = self.lstm_model.predict(scaled_data.reshape(1, -1, 1))
            return self.scaler.inverse_transform(prediction)[0][0]
        else:
            # Make ARIMA prediction
            return self.arima_model.forecast(steps=1)[0]