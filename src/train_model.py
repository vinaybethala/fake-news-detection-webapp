import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from clean_text import clean_text
import os

# Define paths
DATA_PATH = "data/fake_news.csv"
MODEL_PATH = "model/fake_news_model.pkl"

def train():
    print("Loading data...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)
    
    print("Cleaning text...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    print("Vectorizing...")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['label']
    
    print("Training model...")
    model = MultinomialNB()
    model.fit(X, y)
    
    print("Saving model...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump((model, vectorizer), f)
    
    print("Done! Model saved to", MODEL_PATH)

if __name__ == "__main__":
    train()
