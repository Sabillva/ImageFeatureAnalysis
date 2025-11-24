import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Gerekli NLTK paketlerini indir
nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("turkish"))

def clean_text(text):
    # Küçük harf
    text = text.lower()

    # URL sil
    text = re.sub(r"http\S+|www.\S+", "", text)

    # Mention ve hashtag sil
    text = re.sub(r"[@#]\S+", "", text)

    # Emojiler ve noktalama
    text = re.sub(r"[^\w\s]", " ", text)

    # Sayılar
    text = re.sub(r"\d+", "", text)

    # Tokenization
    tokens = word_tokenize(text)

    # Stopwords temizle
    tokens = [t for t in tokens if t not in stop_words]

    # Tek harfli kelimeleri çıkar
    tokens = [t for t in tokens if len(t) > 1]

    return " ".join(tokens)
