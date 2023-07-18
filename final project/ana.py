from textblob import TextBlob
from newspaper import Article
url = "https://www.cnn.com/europe/live-news/russia-ukraine-war-news-07-18-23/index.html"

article = Article(url)
article.download()
article.parse()
article.nlp()
text = article.summary
print(text)
blob = TextBlob(text)
sentiment=blob.sentiment.polarity
print(sentiment)