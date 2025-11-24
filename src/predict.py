import pickle
import os
from .preprocess import clean_text

def predict_sentiment(text):
    with open(os.path.join("..", "results", "sentiment_model.pkl"), "rb") as f:
        model, tfidf = pickle.load(f)

    cleaned = clean_text(text)
    X = tfidf.transform([cleaned])
    pred = model.predict(X)[0]
    return pred

if __name__ == "__main__":
    print(predict_sentiment("ürün çok kötü"))
