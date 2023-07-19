from textblob import TextBlob
with open("twitter_training.csv",'r') as f:
    text = f.read()
blob = TextBlob(text)
sentiment=blob.sentiment.polarity
print(sentiment)

import pandas as pd
from textblob import TextBlob

csv_file = "your_csv_file.csv"  # 파일 경로를 적절하게 변경해주세요

# CSV 파일 읽기
df = pd.read_csv(csv_file)

# 감성 분석을 위한 빈 리스트 생성
sentiments = []

# 텍스트 열 선택 (예시로 'tweet' 열 선택)
texts = df['tweet']

# 텍스트 열별로 감성 분석 수행
for text in texts:
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    sentiments.append(sentiment)

# 결과를 DataFrame에 추가
df['sentiment'] = sentiments

# 결과 출력
print(df)
