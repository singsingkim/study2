# https://dacon.io/competitions/open/236070/data

import numpy as np      # 수치화 연산
import pandas as pd     # 각종 연산 ( 판다스 안의 파일들은 넘파이 형식)
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error    # rmse 사용자정의 하기 위해 불러오는것
import time                

#1. 데이터

path = "c:/_data/dacon/iris//"

# print(path + "aaa.csv") # c:/_data/dacon/ddarung/aaa.csv

train_csv = pd.read_csv(path + "train.csv", index_col = 0)  # 인덱스를 컬럼으로 판단하는걸 방지
# \ \\ / // 다 가능
print(train_csv)
test_csv = pd.read_csv(path + "test.csv", index_col = 0)
print(test_csv)
submission_csv = pd.read_csv(path + "sample_submission.csv")   # 여기 있는 id 는 인덱스 취급하지 않는다.
print(submission_csv)

print(train_csv.shape)          # (120, 5)
print(test_csv.shape)           #  (30, 4) //아래 서브미션과의 열의 합이 11 인것은 id 열 이 중복되어서이다
print(submission_csv.shape)     # (30, 2)

print(train_csv.columns)        # Index(['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
                                #    'petal width (cm)', 'species'],
                                #   dtype='object')

print(train_csv.info())
print(test_csv.info())
print(train_csv.describe())

######### 결측치 처리 1. 제거 #########
train_csv = train_csv.dropna()      # 결측치가 한 행에 하나라도 있으면 그 행을 삭제한다
######### 결측치 처리 2. 0으로 #########
# train_csv = train_csv.fillna(0)   # 결측치 행에 0을 집어 넣는다

# print(train_csv.isnull().sum())
print(train_csv.isna().sum())       # 위 와 같다. isnull() = isna()
print(train_csv.info())
print(train_csv.shape)
print(train_csv)

test_csv = test_csv.fillna(test_csv.mean())     # 널값에 평균을 넣은거
print(test_csv.info())

######### x 와 y 를 분리 #########
x = train_csv.drop(['species'], axis = 1)     # species 를 삭제하는데 count가 열이면 액시스 1, 행이면 0
y = train_csv['species']

print(np.unique(y, return_counts=True))
# (array([0, 1, 2], dtype=int64), array([40, 41, 39], dtype=int64))

print(pd.value_counts(y))
# 1    41
# 0    40
# 2    39

# ========== 원 핫 인코딩 전처리 ==============
# 1) 케라스
from keras.utils import to_categorical
y_ohe = to_categorical(y)   # [1. 0. 0. ] 으로 표현
print(y_ohe)
print(y_ohe.shape)  # (120, 3)

# 2) 판다스
y_ohe2 = pd.get_dummies(y)  # [True  False  False] 으로 표현
print(y_ohe2)
print(y_ohe2.shape) # (120, 3)

'''
# 3) 사이킷런
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()   # (sparse=False)
y = y.reshape(-1, 1)    # (행, 열) 형태로 재정의 // -1 은 열의 정수값에 따라 알아서 행을 맞추어 재정의하라 
y_ohe3 = ohe.fit_transform(y).toarray() # // 투어레이 사용하면 위에 스파라스 안씀. 스파라스 사용하면 투어레이 안씀
print(y_ohe3)
print(y_ohe3.shape) # (178, 3)
'''

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe2, 
            shuffle=True, train_size= 0.7,
            random_state= 7777,
            stratify=y,)    # 스트레티파이 와이(예스)는 분류에서만 쓴다, 트레인 사이즈에 따라 줄여주는것

print(np.unique(y_test, return_counts=True))
# (array([False,  True]), array([72, 36], dtype=int64))

print(x_train.shape, x_test.shape)  # (84, 4) (36, 4)
print(y_train.shape, y_test.shape)  # (84, 3) (36, 3)

print(x.shape, y.shape)
print(x_train.shape, y_train.shape)

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim = 4))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(3, activation = 'softmax'))    

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', 
              optimizer = 'adam', # 이진분류는 아웃풋레이어에 액티베이션은 시그모이드 = 0 ~ 1 확정짓기위해. 히든레이어에 사용해도 가능
              metrics=['acc'])  # # accuracy = acc # 매트릭스 acc 정확도 체크. 가중치에 들어가진 않음 # 애큐러시는 시그모이드를 통해 받은 값을 0.5 를 기준으로 위 아래를 0 또는 1 로 인식한다. 이걸로 이큐러시 몇퍼센트라고 결과를 낸다.

from keras.callbacks import EarlyStopping       # 클래스는 정의가 필요
es = EarlyStopping(monitor = 'val_loss',    # 상당히 중요한 함수
                     mode = 'min',        # max 를 사용하는 경우도 있다 min, max, auto
                     patience=100,      # 최소값 찾은 후 설정값 만큼 훈련 진행  , 발로스 최소값 갱신 한도
                     verbose=1,
                     restore_best_weights=True   # 디폴트는 False # 페이션스 진행 후 최소값을 최종값으로 리턴 
                     )

hist = model.fit(x_train, y_train, epochs = 1000,
                 batch_size = 1, validation_split=0.2,
                 verbose=1, callbacks=[es])

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print("로스 : ", results[0])
print("ACC : ", results[1])

print(y_test)
print(y_predict.shape, y_test.shape)    # (36, 3) (36, 3)

y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)

y_submit = model.predict(test_csv)  # 테스트 파일을 모델에 넣어서 예측값을 뽑아내서. 와이서브밋에 저장한거를 아래 서브미션 씨에스브이에서 저장
print(y_test)
print(y_predict)


######## submission.csv 만들기(species 컬럼에 값만 넣어주면 됌) ########
submission_csv['species'] = np.argmax(y_submit, axis = 1)

print(submission_csv)

submission_csv.to_csv(path + "iris_submission_0112_1.csv", index = False)


from sklearn.metrics import accuracy_score
# acc = accuracy_score(y_predict, y_test)
# print("accuracy_score : ", acc)

def ACC(a,b):
    return accuracy_score(a,b)
acc = ACC(y_test, y_predict)
print("acc : ", acc)

# acc :  0.9722222222222222



