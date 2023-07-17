# 데이터 처리 모듈
import pandas as pd

# 웹크롤링 관련모듈
import snscrape.modules.twitter as sntwitter
import itertools
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

#검색하고 싶은 단어
search_word = "elemental"

#검색하는 기간
start_day = "2023-06-16"
end_day = "2023-07-16"

search_query = search_word + ' since:' + start_day + ' until:' + end_day 

#지정한 기간에서 검색하고 싶은 단어를 포함한 tweet를 취득
scraped_tweets = sntwitter.TwitterSearchScraper(search_query).get_items()
#처음부터 1000개의 tweets를 취득
sliced_scraped_tweets = itertools.islice(scraped_tweets, 1000)
