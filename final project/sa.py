from textblob import TextBlob
with open("abc.txt",'r') as f:
    text = f.read()
blob = TextBlob(text)
sentiment = blob.sentiment.polarity
print(sentiment)