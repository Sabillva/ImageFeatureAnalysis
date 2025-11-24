import pandas as pd
import pickle
import os
from .preprocess import clean_text


from sklearn.metrics import classification_report

def evaluate_model():
    # Test verisini yükle
    test_path = os.path.join(os.path.dirname(__file__), "..", "data", "test.csv")

    df = pd.read_csv(test_path)

    df["cleaned"] = df["text"].apply(clean_text)

    # Model yükle
    with open(os.path.join("..", "results", "sentiment_model.pkl"), "rb") as f:
        model, tfidf = pickle.load(f)

    X = tfidf.transform(df["cleaned"])
    preds = model.predict(X)

    print("Model Değerlendirme Raporu:")
    print(classification_report(df["label"], preds, zero_division=0))

if __name__ == "__main__":
    evaluate_model()
