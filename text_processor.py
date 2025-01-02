from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')

class TextProcessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and preprocess text data."""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenization and stop words removal
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words]
        
        return ' '.join(tokens)
    
    def vectorize_text(self, texts):
        """Convert text to TF-IDF vectors."""
        return self.vectorizer.fit_transform(texts)
    
    def transform_text(self, texts):
        """Transform new texts using fitted vectorizer."""
        return self.vectorizer.transform(texts)