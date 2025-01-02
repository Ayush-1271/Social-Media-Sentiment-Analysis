import pandas as pd
from sklearn.model_selection import train_test_split

def load_sentiment_data(file_path='sentimentdataset.csv'):
    """Load and preprocess sentiment data from CSV file."""
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    return df

def prepare_sentiment_features(df):
    """Prepare features for sentiment analysis."""
    # Create features from text data
    return df[['Text', 'Sentiment', 'Platform', 'Retweets', 'Likes']]

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)