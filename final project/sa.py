import pandas as pd
from textblob import TextBlob
df = pd.readcsv("twitter_training.csv",'r')

sentiments = []
texts = df['tweet']

for text in texts:
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    sentiments.append(sentiment)

print(sentiment)
