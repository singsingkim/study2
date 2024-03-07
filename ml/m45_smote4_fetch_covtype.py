# 45_4. fetch_covtype
# 45_5. dacon_wine
# 45_6. dacon_dechul
# 45_7. kaggle_obesity

from sklearn.datasets import fetch_covtype
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, f1_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

#1. 데이터
load = fetch_covtype()
x = load.data
y = load.target

lbe = LabelEncoder()
y = lbe.fit_transform(y)

print(np.unique(y, return_counts=True)) #0, 1, 2, 3, 4, 5, 6

y_copy = y.copy()
for i, _ in enumerate(y):
    if y_copy[i] == 0 or y[i] == 1: 
        y[i] = 0
    elif y_copy[i] == 2 or y_copy[i] == 3 : 
        y[i] = 2
    elif y_copy[i] == 4 or y_copy[i] == 5: 
        y[i] = 4

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size=0.8,stratify=y)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#데이터 분류
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=42, stratify=y)

#2. 모델 구성
model = RandomForestClassifier(random_state=42)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print("최종점수 :" ,result)
x_predict = model.predict(x_test)
acc = accuracy_score(y_test, x_predict)
print("acc_score :", acc)
f1 = f1_score(y_test, x_predict, average='macro')
print("f1_score : ", f1)

print(np.unique(y_train, return_counts=True)) 
print("SMOTE 적용 후")
from imblearn.over_sampling import SMOTE
smote = SMOTE( random_state=42, k_neighbors=1)
x_train, y_train = smote.fit_resample(x_train, y_train)
print(np.unique(y_train, return_counts=True)) 

#2. 모델 구성
model = RandomForestClassifier(random_state=42)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print("최종점수 :" ,result)
x_predict = model.predict(x_test)
acc = accuracy_score(y_test, x_predict)
print("acc_score :", acc)
f1 = f1_score(y_test, x_predict, average='macro')
print("f1_score : ", f1)

'''
최종점수 : 0.9552850192766661
acc_score : 0.9552850192766661
(array([0, 1, 2, 3, 4, 5, 6], dtype=int64), array([180064, 240806,  30391,   2335,   8069,  14762,  17433],
      dtype=int64))
SMOTE 적용 후
(array([0, 1, 2, 3, 4, 5, 6], dtype=int64), array([240806, 240806, 240806, 240806, 240806, 240806, 240806],
      dtype=int64))
최종점수 : 0.958899394161924
acc_score : 0.958899394161924

====3개 라벨 축소후
SMOTE 적용 후
(array([0, 2, 4, 6], dtype=int64), array([420870, 420870, 420870, 420870], dtype=int64))
최종점수 : 0.9875619607123187
acc_score : 0.9875619607123187
'''

