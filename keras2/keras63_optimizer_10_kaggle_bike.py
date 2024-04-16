import pandas as pd
import numpy as np
path = 'C:/_data/kaggle/bike/'
train_csv =pd.read_csv(path + 'train.csv', index_col=0)
test_csv =pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

# train_csv.drop(75)
# test_csv.drop(75)


#데이터 구조 확인
print(train_csv.shape)#(10886, 11)
print(test_csv.shape)#(6493, 8)

#결측치 확인
print(train_csv.isna().sum())
'''
datetime      0
season        0
holiday       0
workingday    0
weather       0
temp          0
atemp         0
humidity      0
windspeed     0
casual        0
registered    0
count         0
'''
print(test_csv.isna().sum())

'''
datetime      0
season        0
holiday       0
workingday    0
weather       0
temp          0
atemp         0
humidity      0
windspeed     0
'''

#데이터 전처리
x = train_csv.drop('count', axis=1).drop('casual', axis=1).drop('registered', axis=1)
# x = train_csv.drop(columns='casual').drop(columns='registered').drop(columns= 'count')
print(x)

print(x.columns)
y = train_csv['count']


print(x.shape) #(10886, 8)
print(y.shape) #(10886, )

#####################*****B O A R D *****######################
train_size = 0.7
random_state = 12345
epochs = 1100
batch_size=10000
###############################################################

    
#데이터
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size= train_size, random_state= random_state)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(512, input_dim = len(x.columns)))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu' ))
model.add(Dense(1))

#Early Stopping
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_accuracy', mode='max', patience= 1100, verbose=1, restore_best_weights=True)

from keras.optimizers import Adam
learning_rates = [1.0, 0.1, 0.01, 0.001, 0.0001]
for learning_rate in learning_rates : 
    # 컴파일, 훈련
    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
    hist = model.fit(x_train, y_train, epochs= 100, batch_size=batch_size, verbose=0, validation_split=0.3, callbacks=[es])

    # 평가, 예측
    loss = model.evaluate(x_test, y_test)


    from sklearn.metrics import r2_score , mean_squared_error, mean_squared_log_error
    y_predict = model.predict(x_test)
    r2 = r2_score(y_test, y_predict)
    submit = model.predict(test_csv)
    print("lr : {0}, 로스 : {1}".format(learning_rate, loss))
    print("lr : {0}, r2 : {1}".format(learning_rate, r2))

'''
기존 : 
loss :   [68176.6484375, 68176.6484375, 189.3523406982422, 0.010716472752392292]
============================
lr : 1.0, 로스 : [24026.90234375, 24026.90234375, 115.27495574951172]
lr : 1.0, r2 : 0.2566649382413213
lr : 0.1, 로스 : [23025.45703125, 23025.45703125, 113.74532318115234]
lr : 0.1, r2 : 0.28764746007152087
lr : 0.01, 로스 : [23010.73046875, 23010.73046875, 113.73739624023438]
lr : 0.01, r2 : 0.288103174255521
lr : 0.001, 로스 : [22912.0859375, 22912.0859375, 113.2538833618164]
lr : 0.001, r2 : 0.29115505421512033
lr : 0.0001, 로스 : [22773.80859375, 22773.80859375, 112.98052215576172]
lr : 0.0001, r2 : 0.2954330351145732
'''