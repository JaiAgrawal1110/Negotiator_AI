# test_nlp.py

from nlp.sentiment import analyze_sentiment
from nlp.classifier import classify_client

msg = "We really like your work, but our budget is only ₹40,000."

print("Sentiment:")
print(analyze_sentiment(msg))

print("\nArchetype:")
print(classify_client(msg))