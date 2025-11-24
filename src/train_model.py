import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
import os

from .preprocess import clean_text



def train_model():
    # --- Veri Yükleme ---
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")

    df = pd.read_csv(data_path)

    # Temizlik
    df["cleaned"] = df["text"].apply(clean_text)

    X = df["cleaned"]
    y = df["label"]

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf.fit_transform(X)

    # Train-Test
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42
    )

    # Model
    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds, zero_division=0))

    # Modeli kaydet
    model_dir = os.path.join("..", "results")
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, "sentiment_model.pkl"), "wb") as f:
        pickle.dump((model, tfidf), f)

    print("Model kaydedildi: results/sentiment_model.pkl")

if __name__ == "__main__":
    train_model()
