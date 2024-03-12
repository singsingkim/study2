import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense,Dropout,BatchNormalization, AveragePooling1D, Flatten, Conv2D, LSTM, Bidirectional,Conv1D,MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.metrics import accuracy_score, f1_score
from lightgbm import LGBMClassifier
####
path= "c:/_data/kaggle/비만/"
train=pd.read_csv(path+"train.csv",index_col=0)
test=pd.read_csv(path+"test.csv",index_col=0)
sample=pd.read_csv(path+"sample_submission.csv")
x= train.drop(['NObeyesdad'],axis=1)
y= train['NObeyesdad']
# print(train.shape,test.shape)   #(20758, 17) (13840, 16)    NObeyesdad
# print(x.shape,y.shape)  #(20758, 16) (20758,)

TRAINSIZE = 0.85
# RS = 62
NUM = 93
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
x_train,x_test,y_train,y_test=train_test_split(
    x,y,train_size=TRAINSIZE,random_state=123,stratify=y)

# print(x_train.shape,y_train.shape)  #(18682, 16) (18682,)
# print(x_test.shape,y_test.shape)    #(2076, 16) (2076,)
import random

# r = random.randint(1, 100)
r = 123

learning_rate = 0.0001     # 1.0 /0.1 / 0.01 / 0.001 / 0.0001

lgbm_params = {"objective": "multiclass",
               "metric": "multi_logloss",
               "verbosity": -1,
               "boosting_type": "gbdt",
               "random_state": r,
               "num_class": 7,
               "learning_rate" :  learning_rate,
               'n_estimators': 800,         #에포
               'feature_pre_filter': False,
               'lambda_l1': 0.011200712112690567,
               'lambda_l2': 0.00015787257542812432,
               'num_leaves': 32,
               'feature_fraction': 0.572803647018765,
               'bagging_fraction': 0.6531630732572037,
               'bagging_freq': 3,
               'min_child_samples': 7}

model = LGBMClassifier(**lgbm_params)

# 모델 학습, 저장
model.fit(x_train, y_train)
# model.booster_.save_model(F"c:\_data\_save\\{SAVENAME}.h5")

# 테스트 데이터에 대한 예측
y_pred = model.predict(x_test)
# y_submit = model.predict(test)
# sample['NObeyesdad']=y_submit
# sample.to_csv(path + F"{SAVENAME}.csv", index=False)
    
# 정확도 평가
accuracy = accuracy_score(y_pred, y_test)

print(f'lr : {learning_rate}, acc : {accuracy:.4f}')


# lr : 1.0, acc : 0.4123

# lr : 0.1, acc : 0.9033

# lr : 0.01, acc : 0.9159

# lr : 0.001, acc : 0.9011

# lr : 0.0001, acc : 0.5295