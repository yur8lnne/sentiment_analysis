# 데이터 처리 모듈
import pandas as pd

# 웹크롤링 관련모듈
import snscrape.modules.twitter as sntwitter
import itertools

# 검색하고 싶은 단어
search_word = "elemental"

# 검색하는 기간
start_day = "2023-06-16"
end_day = "2023-07-16"

search_query = search_word + ' since:' + start_day + ' until:' + end_day 

# 지정한 기간에서 검색하고 싶은 단어를 포함한 tweet를 취득
scraped_tweets = sntwitter.TwitterSearchScraper(search_query).get_items()
# 처음부터 1000개의 tweets를 취득
sliced_scraped_tweets = itertools.islice(scraped_tweets, 1000)

# 취득한 트위터 데이터를 리스트로 저장
tweets_list = []
for tweet in sliced_scraped_tweets:
    tweets_list.append([tweet.date, tweet.content])

# 트위터 데이터를 DataFrame으로 변환
df = pd.DataFrame(tweets_list, columns=['Date', 'Content'])

# DataFrame을 CSV 파일로 저장
df.to_csv('twitter_data.csv', index=False)
