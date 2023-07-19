from textblob import TextBlob
with open(r"C:\Users\조유라\Desktop\Test\data set\negative.txt") as f:
    text = f.read()
blob = TextBlob(text)
sentiment = blob.sentiment.polarity
print(sentiment)
