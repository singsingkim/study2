import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

# 1. Data
# x,y = fetch_california_housing(return_X_y=True)
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#1. 데이터


x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, train_size=0.8,
    # stratify=y
)

scaler = MinMaxScaler()
# scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

n_splits = 5
kFold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
# kFold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

# 'n_estimators' : [100, 200, 300, 400, 500, 1000] / 디폴트 100/ 1~inf / 정수
# 'learning_rate' : [0.1, 0.2 , 0.3, 0.5, 1, 0.01, 0.001] / 디폴트 0.3 / 0~1 eta
# 'max_depth' : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10] / 디폴트 6 / 0~inf / 정수
# 'gamma' : [0,1,2,3,4,5,7, 10, 100] / 디폴트 0 /0~inf
# 'min_child_weight' : [0, 0.2, 0.3, 0.5, 0.7, 1] / 디폴트 1 / 0~1
# 'colsample_bytree' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] / 디폴트 1 / 0~1
# 'colsample_bylevel' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] / 디폴트 1 / 0~1
# 'colsample_bynode' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] / 디폴트 1 / 0~1
# 'reg_alpha' : [0, 0.1, 0.01, 0.001, 1, 2, 10] / 디폴트 0 / 0~inf / L1 절대값 가중치 규제 / alpha
# 'reg_lambda' : [0, 0.1, 0.01, 0.001, 1, 2, 10] / 디폴트 1 / 0~inf / L2 제곱 가중치 규제 / lambda

parameters = {
    'n_estimators' : [100],
    'learning_rate' : [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001],
    'max_depth' : [3],
}

# 2 모델
xgb = XGBRegressor(random_state = 123)
model = RandomizedSearchCV(xgb, parameters, cv=kFold,
                           n_jobs=22)

# 3 훈련
model.fit(x_train, y_train)

# 4 평가, 예측
print('최상의 매개변수 : ', model.best_estimator_)
print('최상의 파라미터 : ', model.best_params_)
print('최상의 점수  : ', model.best_score_)

results = model.score(x_test, y_test)
print('최종점수 : ', results)


# 최상의 매개변수 :  XGBRegressor(base_score=None, booster=None, callbacks=None,
#              colsample_bylevel=None, colsample_bynode=None,
#              colsample_bytree=None, device=None, early_stopping_rounds=None,
#              enable_categorical=False, eval_metric=None, feature_types=None,
#              gamma=None, grow_policy=None, importance_type=None,
#              interaction_constraints=None, learning_rate=0.5, max_bin=None,
#              max_cat_threshold=None, max_cat_to_onehot=None,
#              max_delta_step=None, max_depth=3, max_leaves=None,
#              min_child_weight=None, missing=nan, monotone_constraints=None,
#              multi_strategy=None, n_estimators=100, n_jobs=None,
#              num_parallel_tree=None, random_state=123, ...)
# 최상의 파라미터 :  {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.5}
# 최상의 점수  :  0.8164822959147277
# 최종점수 :  0.5803524768334112