import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from textblob import TextBlob

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df=pd.read_csv(r"C:\Users\조유라\Desktop\Test\data set\twitter_training.csv",names=["id","company","kind","tweet"])
df.head()

df.drop_duplicates(subset=['tweet'], inplace=True)# 중복값 제거
df = df.dropna(how='any') # Null 값 제거
sentiments = []

for text in df["tweet"]:
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    sentiments.append(sentiment)

def calculate_mean(numbers):
    if len(numbers) == 0:
        return 0  # 빈 리스트일 경우 평균은 0으로 정의
    else:
        return sum(numbers) / len(numbers)

avg = calculate_mean(sentiments)
print(avg)
