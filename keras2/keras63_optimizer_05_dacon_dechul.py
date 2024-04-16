import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, LSTM, Conv1D
from keras.callbacks import EarlyStopping , ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score ,accuracy_score
from keras.optimizers import Adam

path = 'C:/_data/dacon/dechul/'
#데이터 가져오기
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

# print(train_csv.head(25))


#데이터 전처리

#1 문자 -> 수치 대상: 주택소유상태, 근로기간, 대출등급, 대출목적
unique, count = np.unique(train_csv['근로기간'], return_counts=True)
print(unique, count)
unique, count = np.unique(test_csv['근로기간'], return_counts=True)
print(unique, count)
train_le = LabelEncoder()
test_le = LabelEncoder()
train_csv['주택소유상태'] = train_le.fit_transform(train_csv['주택소유상태'])
train_csv['대출목적'] = train_le.fit_transform(train_csv['대출목적'])
train_csv['근로기간'] = train_le.fit_transform(train_csv['근로기간'])
train_csv['대출등급'] = train_le.fit_transform(train_csv['대출등급'])


test_csv['주택소유상태'] = test_le.fit_transform(test_csv['주택소유상태'])
test_csv['대출목적'] = test_le.fit_transform(test_csv['대출목적'])
test_csv['근로기간'] = test_le.fit_transform(test_csv['근로기간'])

#2. 근로기간 따로 처리
# train_csv['근로기간'] = train_csv['근로기간'].replace({'1 year' : 1, '1 years': 1, '10+years': 10, '10+ years' : 10,
#                                               '2 years' : 2, '3': 3, '3 years' : 3, '4 years' : 4, '5 years' : 5,
#                                               '6 years' : 6, '7 years' : 7, '8 years' : 8, '9 years' : 9, '< 1 year' : 0.5, 
#                                               '<1 year' : 0.5, 'Unknown' : 0}).astype(float)
# test_csv['근로기간'] = test_csv['근로기간'].replace({'1 year' : 1, '1 years': 1, '10+years': 10, '10+ years' : 10,
#                                               '2 years' : 2, '3': 3, '3 years' : 3, '4 years' : 4, '5 years' : 5,
#                                               '6 years' : 6, '7 years' : 7, '8 years' : 8, '9 years' : 9, '< 1 year' : 0.5, 
#                                               '<1 year' : 0.5, 'Unknown' : 0}).astype(float)
# train_csv = train_csv[train_csv['근로기간'] != 'Unknown']
print(test_csv.head(20))
unique, count = np.unique(train_csv['근로기간'], return_counts=True)
print(unique, count)
unique, count = np.unique(test_csv['근로기간'], return_counts=True)
print(unique, count)

#3. split 수치화 대상 int로 변경: 대출기간
print(train_csv['대출기간'].str.split().str[0])
train_csv['대출기간'] = train_csv['대출기간'].str.split().str[0].astype(float)
test_csv['대출기간'] = test_csv['대출기간'].str.split().str[0].astype(float)

x = train_csv.drop('대출등급', axis=1)
y = train_csv['대출등급']

#결측치 확인
print(x.isna().sum())
print(y.isna().sum())

print(x.head())

#클래스 확인
unique, count =  np.unique(y, return_counts=True)
print(unique , count)
print(x.shape)#(90623, 14)
print(y.shape)#(90623, 7)

from sklearn.preprocessing import OneHotEncoder
y = y.values.reshape(-1,1) 
one_hot_y = OneHotEncoder(sparse=False).fit_transform(y)

unique, count = np.unique(one_hot_y, return_counts=True)
print(unique, count) #[0. 1.] [543738  90623]

#데이터 분류
x_train, x_test, y_train, y_test = train_test_split(x, one_hot_y, train_size=0.85, random_state=1234567, stratify=one_hot_y)
print(np.unique(y_test, return_counts=True))

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

print(x_train.shape)#(77029, 13)
print(y_train.shape)#(77029, 7)

#모델 생성
model = Sequential()
model.add(Dense(16, input_shape = (13,)))
model.add(Dense(32,activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(7, activation='softmax'))

es = EarlyStopping(monitor='val_loss', mode = 'min', patience= 100, restore_best_weights=True)
import datetime
date = datetime.datetime.now()
print(date) #2024-01-17 10:52:41.770061
date = date.strftime("%m%d_%H%M")
print(date)


#컴파일 , 훈련
from keras.optimizers import Adam
learning_rates = [1.0, 0.1, 0.01, 0.001, 0.0001]
for learning_rate in learning_rates : 
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])
    history = model.fit(x_train, y_train, epochs=100, batch_size=1000, verbose= 0, validation_split=0.2, callbacks=[es])

    #평가, 예측
    loss = model.evaluate(x_test, y_test)
    print("로스값 : ", loss)
    y_predict = model.predict(x_test)
    arg_y_test = np.argmax(y_test,axis=1)
    arg_y_predict = np.argmax(y_predict, axis=1)

    acc = accuracy_score(arg_y_test, arg_y_predict)
    print(f"accuracy : {acc}")
    print(loss)


    print("lr : {0}, 로스 : {1}".format(learning_rate, loss))
    print("lr : {0}, ACC : {1}".format(learning_rate, acc))

'''
기존 : 
loss : [1.271729826927185, 0.5023883581161499]
============================
lr : 1.0, 로스 : [1.6115056276321411, 0.2992731034755707]
lr : 1.0, ACC : 0.2992731048805815
lr : 0.1, 로스 : [1.599689245223999, 0.2992731034755707]
lr : 0.1, ACC : 0.2992731048805815
lr : 0.01, 로스 : [1.5970239639282227, 0.2992731034755707]
lr : 0.01, ACC : 0.2992731048805815
lr : 0.001, 로스 : [1.596825361251831, 0.2992731034755707]
lr : 0.001, ACC : 0.2992731048805815
lr : 0.0001, 로스 : [1.59682297706604, 0.2992731034755707]
lr : 0.0001, ACC : 0.2992731048805815
'''