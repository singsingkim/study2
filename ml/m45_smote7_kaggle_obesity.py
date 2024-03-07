import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, f1_score
import pandas as pd

# get data
path = "C:/_data/kaggle/obesity/"
train_csv = pd.read_csv(path + "train.csv")
test_csv = pd.read_csv(path + "test.csv")

# train_csv = colume_preprocessing(train_csv)

train_csv['BMI'] =  train_csv['Weight'] / (train_csv['Height'] ** 2)
test_csv['BMI'] =  test_csv['Weight'] / (test_csv['Height'] ** 2)

lbe = LabelEncoder()
cat_features = train_csv.select_dtypes(include='object').columns.values
for feature in cat_features :
    train_csv[feature] = lbe.fit_transform(train_csv[feature])
    if feature == "CALC" and "Always" not in lbe.classes_ :
        lbe.classes_ = np.append(lbe.classes_, "Always")
    if feature == "NObeyesdad":
        continue
    test_csv[feature] = lbe.transform(test_csv[feature]) 
                
x, y = train_csv.drop(["NObeyesdad"], axis=1), train_csv.NObeyesdad

print(np.unique(y, return_counts=True)) #0, 1, 2, 3, 4, 5, 6

y_copy = y.copy()
for i, _ in enumerate(y):
    if y_copy.iloc[i] == 0 or y_copy.iloc[i] == 1 or y_copy.iloc[i] == 2: 
        y_copy.iloc[i] = 0
    elif y_copy.iloc[i] == 3 or y_copy.iloc[i] == 4 or y_copy.iloc[i] == 5: 
        y_copy.iloc[i] = 3
y = y_copy
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
최종점수 : 0.8933044315992292
acc_score : 0.8933044315992292
(array([0, 1, 2, 3, 4, 5, 6]), array([2018, 2465, 2328, 2598, 3237, 1942, 2018], dtype=int64))
SMOTE 적용 후
(array([0, 1, 2, 3, 4, 5, 6]), array([3237, 3237, 3237, 3237, 3237, 3237, 3237], dtype=int64))
최종점수 : 0.890655105973025
acc_score : 0.890655105973025
=========라벨축소 적용 후
최종점수 : 0.9108863198458574
acc_score : 0.9108863198458574
f1_score :  0.8825685885779402
(array([0, 3, 6]), array([6812, 7777, 2017], dtype=int64))
SMOTE 적용 후
(array([0, 3, 6]), array([7777, 7777, 7777], dtype=int64))
최종점수 : 0.905587668593449
acc_score : 0.905587668593449
f1_score :  0.87703009596836
'''