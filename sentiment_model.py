from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

class SentimentAnalyzer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def train(self, X_train, y_train):
        """Train the sentiment analysis model."""
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """Make predictions on new data."""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        predictions = self.predict(X_test)
        return classification_report(y_test, predictions)
    
    def save_model(self, filepath):
        """Save the trained model."""
        joblib.dump(self.model, filepath)
    
    @staticmethod
    def load_model(filepath):
        """Load a trained model."""
        return joblib.load(filepath)