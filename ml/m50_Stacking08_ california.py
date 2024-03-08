from sklearn.datasets import fetch_california_housing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import pandas as pd

#1. 데이터
load = fetch_california_housing()
x = load.data
y = load.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size=0.8)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, BaggingRegressor, VotingClassifier, VotingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
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
xgb = XGBRegressor()
xgb.set_params(**parameters, eval_metric = 'rmse')
rf = RandomForestRegressor()
lr = LinearRegression()
from sklearn.ensemble import StackingRegressor
from catboost import CatBoostRegressor

models = [xgb,rf, lr]
for model in models :
    model.fit(x_train, y_train)
    class_name = model.__class__.__name__ 
    score = model.score(x_test,y_test)
    print("{0} ACC : {1:.4f}".format(class_name, score))

#torch, keras 도 연결가능. ==> 생코딩으로 만드는게 더 많음.
model = StackingRegressor(
    estimators=[('XGB', xgb), ('KNN', rf), ('LR', lr)],
    final_estimator=CatBoostRegressor(verbose=0),
    n_jobs=1,
) 
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
pred = model.predict(x_test)
acc = r2_score(pred, y_test)
print("스태킹 결과 : {0:.4f}".format(score) )
'''
XGBRegressor ACC : 0.7808
RandomForestRegressor ACC : 0.8065
LinearRegression ACC : 0.5828
스태킹 결과 : 0.8040
'''