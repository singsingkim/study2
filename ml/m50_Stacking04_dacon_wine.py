from sklearn.datasets import fetch_covtype
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import pandas as pd

#1. 데이터
path = "C:/_data/dacon/wine/"

#1.데이터 가져오기
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

train_csv['type'] = train_csv['type'].replace({"white":1, "red":0})
test_csv['type'] = test_csv['type'].replace({"white":1, "red":0})

x = train_csv.drop(columns='quality')
y = train_csv['quality']

lbe = LabelEncoder()
y = lbe.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size=0.8,stratify=y)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, BaggingRegressor, VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore') #워닝 무시
parameters = {
    'n_estimators': 1000,  # 디폴트 100
    'learning_rate': 0.01,  # 디폴트 0.3 / 0~1 / eta *
    'max_depth': 3,  # 디폴트 0 / 0~inf
    'gamma': 0,
    'min_child_weight' : 0,
    'subsample' : 0.4,
    'colsample_bytree' :0.8,
    'colsample_bylevel' : 0.7,
    'colsample_bynode': 1,
    'reg_alpha': 0,
    'reg_lambda' : 1,
    'random_state' : 42,
}

# 2. 모델 구성
xgb = XGBClassifier()
xgb.set_params(**parameters, eval_metric = 'logloss')
rf = RandomForestClassifier()
lr = LogisticRegression()
from sklearn.ensemble import StackingClassifier
from catboost import CatBoostClassifier

models = [xgb,rf, lr]
for model in models :
    model.fit(x_train, y_train)
    class_name = model.__class__.__name__ 
    score = model.score(x_test,y_test)
    print("{0} ACC : {1:.4f}".format(class_name, score))

#torch, keras 도 연결가능. ==> 생코딩으로 만드는게 더 많음.
model = StackingClassifier(
    estimators=[('XGB', xgb), ('KNN', rf), ('LR', lr)],
    final_estimator=CatBoostClassifier(verbose=0),
    n_jobs=1,
    cv=5
) 
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
pred = model.predict(x_test)
acc = accuracy_score(pred, y_test)
print("스태킹 결과 : {0:.4f}".format(score) )

'''
XGBClassifier ACC : 0.6036
RandomForestClassifier ACC : 0.6827
LogisticRegression ACC : 0.5636
스태킹 결과 : 0.6773
'''