import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# 1 데이터
path = 'c:/_data/dacon/wine/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(train_csv)

submission_csv=pd.read_csv(path + "sample_submission.csv")

train_csv['type']=train_csv['type'].map({'white':1,'red':0}).astype(int)
test_csv['type']=test_csv['type'].map({'white':1,'red':0}).astype(int)

x=train_csv.drop(['quality'],axis=1)
y=train_csv['quality']

label=LabelEncoder()
label.fit(y)
y=label.transform(y)


#1. 데이터
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=456, train_size=0.8,
    # stratify=y
)




# 2 모델
model = RandomForestClassifier()
# model.set_params(early_stopping_rounds=200, **xgb_params)


# 3 훈련

model.fit(x_train, y_train,
        #   eval_set=[(x_train,y_train), (x_test,y_test)],
        #   verbose=1,
        #   eval_metric='auc'
)

# 4 평가, 예측
results = model.score(x_test,y_test)
print('최종점수', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('add_score', acc)



# random_state = 456
# 최종점수 0.6945454545454546
# add_score 0.6945454545454546











