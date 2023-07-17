import pandas as pd
from tqdm.notebook import tqdm
import snscrape.modules.twitter as sntwitter

sntwitter.TwitterSearchScraper("#coding")
tweets =[]
n_tweets =1000

for i,tweet in tqdm(enumerate(sntwitter.get_items()), total=n_tweets):
    data =[
        tweet.date,
        tweet.id,
        tweet.content,
        tweet.user.username,
        tweet.likeCount,
        tweet.retweetCount,
    ]
    tweets.append(data)
    if i>n_tweets:
        break

tweet_df =pd.DataFrame(
    tweets, columns=["date","id","content","username","like_count","retweet_count"]
)

tweet_df.to_csv("tweets.csv", index=False)
