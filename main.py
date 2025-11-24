from src.predict import predict_sentiment

text = input("Bir metin girin: ")
print("Duygu Tahmini:", predict_sentiment(text))
