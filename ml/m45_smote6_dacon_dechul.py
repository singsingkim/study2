import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, f1_score
import pandas as pd

path = 'C:/_data/dacon/dechul/'
#데이터 가져오기
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

unique, count = np.unique(train_csv['근로기간'], return_counts=True)
unique, count = np.unique(test_csv['근로기간'], return_counts=True)
train_le = LabelEncoder()
test_le = LabelEncoder()
train_csv['주택소유상태'] = train_le.fit_transform(train_csv['주택소유상태'])
train_csv['대출목적'] = train_le.fit_transform(train_csv['대출목적'])
train_csv['근로기간'] = train_le.fit_transform(train_csv['근로기간'])
train_csv['대출등급'] = train_le.fit_transform(train_csv['대출등급'])


test_csv['주택소유상태'] = test_le.fit_transform(test_csv['주택소유상태'])
test_csv['대출목적'] = test_le.fit_transform(test_csv['대출목적'])
test_csv['근로기간'] = test_le.fit_transform(test_csv['근로기간'])

#3. split 수치화 대상 int로 변경: 대출기간
train_csv['대출기간'] = train_csv['대출기간'].str.split().str[0].astype(float)
test_csv['대출기간'] = test_csv['대출기간'].str.split().str[0].astype(float)

x = train_csv.drop(["대출등급"], axis=1)
y = train_csv["대출등급"]

print(np.unique(y, return_counts=True)) #0, 1, 2, 3, 4, 5, 6

y_copy = y.copy()
for i, _ in enumerate(y):
    if y_copy.iloc[i] == 0 or y_copy.iloc[i] == 1 or y_copy.iloc[i] == 2: 
        y_copy.iloc[i] = 0
    elif y_copy.iloc[i] == 3 or y_copy.iloc[i] == 4 or y_copy.iloc[i] == 5: 
        y_copy.iloc[i] = 3
print(np.unique(y_copy, return_counts=True)) 
y = y_copy

lbe = LabelEncoder()
y = lbe.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size=0.8,stratify=y)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

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
최종점수 : 0.8031050417986396
acc_score : 0.8031050417986396
(array([0, 1, 2, 3, 4, 5, 6], dtype=int64), array([13418, 23054, 22098, 10683,  5883,  1563,   336], dtype=int64))
SMOTE 적용 후
(array([0, 1, 2, 3, 4, 5, 6], dtype=int64), array([23054, 23054, 23054, 23054, 23054, 23054, 23054], dtype=int64))
최종점수 : 0.8117763123734358
acc_score : 0.8117763123734358
=====라벨축소 적용 후
최종점수 : 0.947556986344047
acc_score : 0.947556986344047
f1_score :  0.6476373244323294
(array([0, 1, 2], dtype=int64), array([58569, 18130,   336], dtype=int64))
SMOTE 적용 후
(array([0, 1, 2], dtype=int64), array([58569, 58569, 58569], dtype=int64))
최종점수 : 0.9513993457604237
acc_score : 0.9513993457604237
f1_score :  0.7161535396380896
'''