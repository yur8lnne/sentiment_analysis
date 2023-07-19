import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
#from tensorflow.keras.preprocessing.text import Tokenizer
#from tensorflow.keras.preprocessing.sequence import pad_sequences
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_data=pd.read_csv(r"C:\Users\조유라\Desktop\sentiment_analysis\final project\data set\twitter_training.csv",names=["id","company","kind","tweet"])
train_data.head()
test_data=pd.read_csv(r"C:\Users\조유라\Desktop\sentiment_analysis\final project\data set\twitter_validation.csv",names=["id","company","kind","tweet"])
test_data.head()

train_data.drop_duplicates(subset=['tweet'], inplace=True)


#알파벳과 공백을 제외하고 모두 제거
train_data['tweet'] = train_data['tweet'].str.replace("[^a-zA-Z ]","")
#print(train_data[:5])
#print(train_data.loc[train_data.tweet.isnull()][:5])
#train_data = train_data.dropna(how = 'any')
#print(len(train_data))

train_data.drop_duplicates(subset = ['tweet'], inplace=True) # tweet 열에서 중복인 내용이 있다면 중복 제거
train_data['tweet'] = train_data['tweet'].str.replace("[^a-zA-Z ]","") # 정규 표현식 수행
train_data['tweet'] = train_data['tweet'].str.replace('^ +', "") # 공백은 empty 값으로 변경
train_data['tweet'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경
train_data = train_data.dropna(how='any') # Null 값 제거


test_data.drop_duplicates(subset = ['tweet'], inplace=True) # tweet 열에서 중복인 내용이 있다면 중복 제거
test_data['tweet'] = test_data['tweet'].str.replace("[^a-zA-Z ]","") # 정규 표현식 수행
test_data['tweet'] = test_data['tweet'].str.replace('^ +', "") # 공백은 empty 값으로 변경
test_data['tweet'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경
test_data = test_data.dropna(how='any') # Null 값 제거
#print('전처리 후 테스트용 샘플의 개수 :',len(test_data))


import nltk

X_train = []
for sentence in tqdm(train_data['tweet']):
    tokenized_sentence = nltk.word_tokenize(sentence) # 토큰화
    X_train.append(tokenized_sentence)

X_test = []
for sentence in tqdm(train_data['tweet']):
    tokenized_sentence = nltk.word_tokenize(sentence) # 토큰화
    X_test.append(tokenized_sentence)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

threshold = 3
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

# 전체 단어 개수 중 빈도수 2이하인 단어는 제거.
# 0번 패딩 토큰을 고려하여 + 1
vocab_size = total_cnt - rare_cnt + 1
#print('단어 집합의 크기 :',vocab_size)

tokenizer = Tokenizer(vocab_size) 
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

y_train = np.array(train_data['kind'])
y_test = np.array(test_data['kind'])

drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]

print('리뷰의 최대 길이 :',max(len(review) for review in X_train))
print('리뷰의 평균 길이 :',sum(map(len, X_train))/len(X_train))
'''
plt.hist([len(review) for review in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()
'''
def below_threshold_len(max_len, nested_list):
  count = 0
  for sentence in nested_list:
    if(len(sentence) <= max_len):
        count = count + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (count / len(nested_list))*100))

max_len = 75
below_threshold_len(max_len, X_train)
'''
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

embedding_dim = 100
hidden_units = 128

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(LSTM(hidden_units))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=64, validation_split=0.2)

loaded_model = load_model('best_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

with open('tokenizer.pickle', 'wb') as handle:
     pickle.dump(tokenizer, handle)

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
'''