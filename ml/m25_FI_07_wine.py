import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
# 1. Data
path = "c://_data//dacon//wine//"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv",index_col=0)
# print(test_csv)
submission_csv=pd.read_csv(path + "sample_submission.csv")

train_csv['type']=train_csv['type'].map({'white':1,'red':0}).astype(int)
test_csv['type']=test_csv['type'].map({'white':1,'red':0}).astype(int)

x=train_csv.drop(['quality'],axis=1)
y=train_csv['quality']

label=LabelEncoder()
label.fit(y)
y=label.transform(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=28,stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
from sklearn.preprocessing import OneHotEncoder
   
# 2. Model

rs = 1212
models =[DecisionTreeClassifier(random_state=rs),
    RandomForestClassifier(random_state=rs),
    GradientBoostingClassifier(random_state=rs),
    XGBClassifier(random_state=rs)]

for model in models:
    model.fit(x_train,y_train)
    result= model.score(x_test,y_test)
    print("model.score:",result)
    # y_predict=model.predict(test_csv)
    # acc=accuracy_score(y_test,y_predict)
    # print(model, "acc", acc)
    # print(model.feature_importances_) 
    print(type(model).__name__, ":",model.feature_importances_)   
        




# print(model, "acc", acc)

# print(model.feature_importances_)   
#[0.04519231 0.         0.54879265 0.40601504] - 각각 feature 점수
# print(model, ":",model.feature_importances_)   
#DecisionTreeClassifier(random_state=1212) : [0.04519231 0.         0.54879265 0.40601504]
#RandomForestClassifier(random_state=1212) : [0.09680396 0.02696853 0.39722367 0.47900384]
#GradientBoostingClassifier(random_state=1212) : [0.01522103 0.0134919  0.27358378 0.6977033 ]
# XGBClassifier(random_state=1212) : [0.01192079 0.02112738 0.51003134 0.45692044]