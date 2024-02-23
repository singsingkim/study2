import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical #
from imblearn.over_sampling import SMOTE
import time
start_time = time.time()

#1. 데이터
path = "C:/_data/dacon/dechul//"
train_csv = pd.read_csv(path + "train.csv", index_col=0 )
print(train_csv.shape)  # (96294, 14)
test_csv = pd.read_csv(path + "test.csv", index_col=0 )
print(test_csv.shape)  # (64197, 13)
submission_csv = pd.read_csv(path + "sample_submission.csv")
print(submission_csv.shape)  # (64197, 2)


# 라벨 인코더
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
le = LabelEncoder() # 대출기간, 대출목적, 근로기간, 주택소유상태 // 라벨 인코더 : 카테고리형 피처를 숫자형으로 변환
train_csv['주택소유상태'] = le.fit_transform(train_csv['주택소유상태'])
    # 데이터프레임의 '주택소유상태' 열에 있는 범주형 데이터를 변환하기 위해 
    # LabelEncoder를 사용하여 수치형으로 변환한 후, 다시 '주택소유상태' 열에 할당하는 작업
train_csv['대출목적'] = le.fit_transform(train_csv['대출목적'])
train_csv['대출기간'] = train_csv['대출기간'].str.slice(start=0,stop=3).astype(int)
                                # 문자열을 시작 부분부터 0번째 위치부터 끝 부분까지(3번째 위치 전까지) 잘라내는 작업
    # 데이터프레임(train_csv)의 '대출기간' 열에 있는 문자열을 자르고,
    # 그것을 정수형으로 변환하여 다시 '대출기간' 열에 할당하는 작업
train_csv['근로기간'] = le.fit_transform(train_csv['근로기간'])

test_csv['주택소유상태'] = le.fit_transform(test_csv['주택소유상태'])
test_csv['대출목적'] = le.fit_transform(test_csv['대출목적'])
test_csv['대출기간'] = test_csv['대출기간'].str.slice(start=0,stop=3).astype(int)
test_csv['근로기간'] = le.fit_transform(test_csv['근로기간'])

train_csv['대출등급'] = le.fit_transform(train_csv['대출등급']) # 마지막에 와야함

# print(train_csv.describe)
# print(test_csv.describe)

# print(train_csv.shape)
# print(test_csv.shape)
# print(train_csv.dtypes)
# print(test_csv.dtypes)

# x와 y를 분리
x = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']
print(x.shape, y.shape) # (96294, 13) (96294,)
print(pd.value_counts(y))
# 대출등급
# 1    28817
# 2    27623
# 0    16772
# 3    13354
# 4     7354
# 5     1954
# 6      420

# mms = MinMaxScaler()
# mms.fit(x)
# x = mms.transform(x)
# test_csv=mms.transform(test_csv)

y = np.reshape(y, (-1,1)) 
# y = np.array()


ohe = OneHotEncoder(sparse = False)
ohe.fit(y)
y_ohe = ohe.transform(y)
print(y.shape)  # (96294, 1)



# y_ohe = pd.get_dummies(y, dtype='int')
# print(y_ohe)   
# print(x.shape, y.shape)   # (96294, 13) (96294, 1) // (96294, ) 벡터 형태 -> reshape를 이용해 행렬로 바꿔줘야함


x_train, x_test, y_train, y_test = train_test_split(
            x,
            y_ohe,             
            train_size=0.8,
            random_state=12,
            stratify=y_ohe,
            shuffle=True,
            )
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler  # StandardScaler 표준편차 (편차 쏠릴때 사용) // 
# scaler = MinMaxScaler() # 클래스 정의
scaler = StandardScaler() # 클래스 정의
# scaler = MaxAbsScaler() # 클래스 정의
# scaler = RobustScaler() # 클래스 정의


scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

print(x_train.shape, y_train.shape) # (82812, 13) (82812, 7)

# print('================= smote =====================')
# smote = SMOTE(random_state=10, sampling_strategy='auto')
# x_train, y_train = smote.fit_resample(x_train, y_train)

# print(x_train.shape, y_train.shape) # (173474, 13) (173474, 7)
# print(y_train)  
# 아래 밸류카운트 대출에서 안먹힌다. : 이유 모름
# print(pd.value_counts(y_train)) 



#2. 모델 구성 

model = Sequential()
model.add(Dense(256, input_dim=13, activation='swish'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='swish'))
model.add(Dense(256, activation='swish'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='swish'))
model.add(Dense(64, activation='swish'))
model.add(Dense(16, activation='swish'))
model.add(Dense(8, activation='swish'))
model.add(Dense(7, activation='softmax'))


#3.컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss',
                mode='min',
                patience=1000,
                verbose=1,
                restore_best_weights=True
                )

model.fit(x_train, y_train, epochs=20000, 
                batch_size = 1024,
                validation_split=0.2,
                callbacks=[es],
                verbose=1
                )
end_time = time.time()   #끝나는 시간

#4.평가, 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
arg_pre = np.argmax(y_predict, axis=1)    #  argmax : NumPy 배열에서 가장 높은 값을 가진 값의 인덱스를 반환
arg_test = np.argmax(y_test, axis=1)
y_submit = model.predict(test_csv)
submit = np.argmax(y_submit, axis=1)
submitssion = le.inverse_transform(submit)
      
submission_csv['대출등급'] = submitssion
y_predict = ohe.inverse_transform(y_predict)
y_test = ohe.inverse_transform(y_test)
f1 = f1_score(y_test, y_predict, average='macro')
acc = accuracy_score(y_test, y_predict)
print("로스 : ", results[0])  
print("acc : ", results[1])  
print("f1 : ", f1)  
submission_csv.to_csv(path + "dechul_0201_3.csv", index=False)
print("걸린시간 : ", round(end_time - start_time, 2),"초")


# dechul_0201_3
# 랜덤 : 12
# 로스 :  0.22894105315208435
# acc :  0.9215431809425354
# f1 :  0.8948446149624962
# 걸린시간 :  2581.56 초