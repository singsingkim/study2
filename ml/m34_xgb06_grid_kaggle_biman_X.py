import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense,Dropout,BatchNormalization, AveragePooling1D, Flatten, Conv2D, LSTM, Bidirectional,Conv1D,MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.metrics import accuracy_score, f1_score
from lightgbm import LGBMClassifier

path= "c:/_data/kaggle/비만/"
train=pd.read_csv(path+"train.csv",index_col=0)
test=pd.read_csv(path+"test.csv",index_col=0)
sample=pd.read_csv(path+"sample_submission.csv")
x= train.drop(['NObeyesdad'],axis=1)
y= train['NObeyesdad']
# print(train.shape,test.shape)   #(20758, 17) (13840, 16)    NObeyesdad
# print(x.shape,y.shape)  #(20758, 16) (20758,)

TRAINSIZE = 0.8
RS = 21
NUM = 51

SAVENAME = f'biman{NUM}'

lb = LabelEncoder()

# 라벨 인코딩할 열 목록
columns_to_encode = ['Gender','family_history_with_overweight','FAVC','CAEC','SMOKE','SCC','CALC','MTRANS']

# 데이터프레임 x의 열에 대해 라벨 인코딩 수행
for column in columns_to_encode:
    lb.fit(x[column])
    x[column] = lb.transform(x[column])

# 데이터프레임 test_csv의 열에 대해 라벨 인코딩 수행
for column in columns_to_encode:
    lb.fit(test[column])
    test[column] = lb.transform(test[column])
    
# print(x['Gender'])
# print(test['CALC'])



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

