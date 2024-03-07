import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

path = "C:/_data/dacon/wine/"

#1.데이터 가져오기
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

train_csv['type'] = train_csv['type'].replace({"white":1, "red":0})
test_csv['type'] = test_csv['type'].replace({"white":1, "red":0})

x = train_csv.drop(columns='quality')
y = train_csv['quality']
y= y-3
print(np.unique(y, return_counts=True)) #3, 4, 5, 6, 7, 8, 9

y_copy = y.copy()
for i, _ in enumerate(y):
    if y_copy[i] == 0 or y[i] == 1 or y_copy[i] == 2: 
        y[i] = 0
    elif y_copy[i] == 3 or y_copy[i] == 4 or y_copy[i] == 5: 
        y[i] = 3
    elif y_copy[i] == 6 or y_copy[i] == 7 or y_copy[i] == 8: 
        y[i] = 6

lbe = LabelEncoder()
y = lbe.fit_transform(y)

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
from imblearn.over_sampling import SMOTE
smote = SMOTE( random_state=42, k_neighbors=1)
x_train, y_train = smote.fit_resample(x_train, y_train)
print("SMOTE 적용 후")
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
최종점수 : 0.6787878787878788
acc_score : 0.6787878787878788
(array([0, 1, 2, 3, 4, 5, 6], dtype=int64), array([  22,  158, 1520, 2054,  785,  129,    4], dtype=int64))
SMOTE 적용 후
(array([0, 1, 2, 3, 4, 5, 6], dtype=int64), array([2054, 2054, 2054, 2054, 2054, 2054, 2054], dtype=int64))
최종점수 : 0.6618181818181819
acc_score : 0.6618181818181819

====3개 라벨 축소후
최종점수 : 0.8290909090909091
acc_score : 0.8290909090909091
f1_score :  0.5383484162895928
(array([0, 1, 2], dtype=int64), array([1700, 2968,    4], dtype=int64))
SMOTE 적용 후
(array([0, 1, 2], dtype=int64), array([2968, 2968, 2968], dtype=int64))
최종점수 : 0.833939393939394
acc_score : 0.833939393939394
f1_score :  0.5458459959209584
'''