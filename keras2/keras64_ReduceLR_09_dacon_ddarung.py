

# # https://dacon.io/competitions/open/235576/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error #mse
import time as tm

#1. 데이터 - 경로데이터를 메모리에 땡겨옴
path = "c:/_data/dacon/ddarung/"
train_csv = pd.read_csv(path + "train.csv", index_col=0) #c:/_data/dacon/ddarung/train.csv
#문자열에서 경로를 찾아줄때 예약어가 있으면 \를 한번 더 붙여줌.
#경로 찾는것 \ ,\\, /, // 다 됨.
#가독성 면에서 슬래시 모양은 하나로 통일.
#default 맨 윗줄을 컬럼명으로 인식.(대부분 컬럼명이 들어감으로)

test_csv = pd.read_csv(path + "test.csv", index_col=0) 
submission_csv = pd.read_csv(path + "submission.csv")
train_csv = train_csv.fillna(test_csv.mean()) # 715 non-null
test_csv = test_csv.fillna(test_csv.mean()) # 715 non-null

######### x와 y를 분리 ###########
x = train_csv.drop(['count'], axis=1) #axis 0이 행 1이 열
# print(x)

y = train_csv['count'] 
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, train_size=0.7, random_state=12345)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#Early Stopping
from keras.callbacks import EarlyStopping , ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', mode='min', patience=15, verbose=1, restore_best_weights=True)
import datetime
date = datetime.datetime.now()
print(date) #2024-01-17 10:52:41.770061
date = date.strftime("%m%d_%H%M")
print(date)

rlr = ReduceLROnPlateau(monitor='val_loss',
                        patience=10, #early stopping 의 절반
                        mode = 'auto',
                        verbose= 1,
                        factor=0.5 #learning rate 를 반으로 줄임.
                        )

#2. 모델
model = Sequential()
model.add(Dense(512, input_dim = len(x.columns)))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
#3. 컴파일, 훈련

from keras.optimizers import Adam

model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mse', 'mae', 'acc'])
start_time = tm.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=70, validation_split=0.3, verbose=0, callbacks=[rlr]) #98
end_time = tm.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("loss : ",loss)
print("r2 : ", r2)
#걸린시간 측정 CPU GPU 비교
print("걸린시간 : ", round(end_time - start_time, 2), "초")

y_submit = model.predict(test_csv) # count 값이 예측됨.
submission_csv['count'] = y_submit
print("로스 : {0}".format(loss))
print("r2 : {0}".format(r2))


'''
기존 : 
loss :  [2606.534423828125, 2606.534423828125, 35.9389533996582, 0.0022831049282103777]
===========
lr : 1.0, 로스 : [19392.50390625, 19392.50390625, 111.47413635253906, 0.0]
lr : 1.0, r2 : -1.783869782252674
lr : 0.01, 로스 : [7363.44580078125, 7363.44580078125, 65.66143798828125, 0.004566209856420755]
lr : 0.01, r2 : -0.05705126876285638
lr : 0.001, 로스 : [7326.0947265625, 7326.0947265625, 65.58233642578125, 0.004566209856420755]
lr : 0.001, r2 : -0.05168952943307725
lr : 0.0001, 로스 : [7322.43896484375, 7322.43896484375, 65.57466125488281, 0.004566209856420755]
lr : 0.0001, r2 : -0.05116461313057363
============ReduceLR10
로스 : [2294.64501953125, 2294.64501953125, 33.50263214111328, 0.004566209856420755]
r2 : 0.6705947474706698
'''