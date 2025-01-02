from data_loader import load_sentiment_data, prepare_sentiment_features, split_data
from text_processor import TextProcessor
from sentiment_model import SentimentAnalyzer
from time_series_analysis import TimeSeriesAnalyzer
import pandas as pd
import numpy as np

def main():
    # Load and prepare data
    print("Loading data...")
    df = load_sentiment_data()
    
    # Sentiment Analysis
    print("\nPerforming Sentiment Analysis...")
    text_processor = TextProcessor()
    
    # Clean and vectorize text
    df['cleaned_text'] = df['Text'].apply(text_processor.clean_text)
    X = text_processor.vectorize_text(df['cleaned_text'])
    y = df['Sentiment']
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train and evaluate sentiment model
    sentiment_model = SentimentAnalyzer()
    sentiment_model.train(X_train, y_train)
    
    print("\nSentiment Analysis Results:")
    print(sentiment_model.evaluate(X_test, y_test))
    
    # Time Series Analysis
    print("\nPerforming Time Series Analysis...")
    # Group by timestamp and calculate daily sentiment scores
    daily_sentiment = df.groupby('Timestamp')['Sentiment'].mean().reset_index()
    daily_sentiment = daily_sentiment.set_index('Timestamp')
    
    # Initialize time series analyzer
    ts_analyzer = TimeSeriesAnalyzer()
    
    # Prepare data for LSTM
    lookback = 60
    X_ts, y_ts = ts_analyzer.prepare_sequence_data(
        daily_sentiment['Sentiment'].values, 
        lookback=lookback
    )
    
    # Build and train LSTM model
    lstm_model = ts_analyzer.build_lstm_model(lookback)
    lstm_model.fit(
        X_ts.reshape(-1, lookback, 1),
        y_ts,
        epochs=50,
        batch_size=32,
        verbose=0
    )
    
    # Fit ARIMA model
    arima_model = ts_analyzer.fit_arima(daily_sentiment['Sentiment'])
    
    # Make predictions
    last_sequence = daily_sentiment['Sentiment'].values[-lookback:]
    lstm_prediction = ts_analyzer.predict_next_period(last_sequence, model_type='lstm')
    arima_prediction = ts_analyzer.predict_next_period(daily_sentiment['Sentiment'], model_type='arima')
    
    print("\nPredictions for next period:")
    print(f"LSTM Prediction: {lstm_prediction:.4f}")
    print(f"ARIMA Prediction: {arima_prediction:.4f}")

if __name__ == "__main__":
    main()