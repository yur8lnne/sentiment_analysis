from textblob import TextBlob
with open("positive.txt",'r') as f:
    text = f.read()
blob = TextBlob(text)
sentiment = blob.sentiment.polarity
print(sentiment)