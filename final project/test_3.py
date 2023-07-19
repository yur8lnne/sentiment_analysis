import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt
from tqdm import tqdm

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_data=pd.read_csv(r"C:\Users\조유라\Desktop\sentiment_analysis\final project\data set\twitter_training.csv",names=["id","company","kind","tweet"])
train_data.head()
test_data=pd.read_csv(r"C:\Users\조유라\Desktop\sentiment_analysis\final project\data set\twitter_validation.csv",names=["id","company","kind","tweet"])
test_data.head()

train_data.drop_duplicates(subset=['tweet'], inplace=True)

print('훈련용 리뷰 개수 :',len(train_data)) # 훈련용 리뷰 개수 출력


import pandas as pd

csv_file = r"path_to_csv_file.csv"
data = pd.read_csv(csv_file, quoting=pd.QUOTE_NONNUMERIC)

# 나머지 코드 실행
