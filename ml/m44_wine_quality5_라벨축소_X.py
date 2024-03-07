import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

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
y=y-3

#####################################################################
# [실습] y의 클래스를 7개에서 5~3개로 줄여서 성능 비교
#####################################################################
# y = y.copy()    # 알아서 참고 하삼
## 힌트 : for 문 돌리믄 되겠지?

for i, v in enumerate(y):
    if v <= 4:
        y[i] = 0
    elif v == 5:
        y[i] = 1
    elif v == 6:
        y[i] = 2
    elif v == 7:
        y[i] = 3
    elif v == 8:
        y[i] = 4
    else:
        y[i] = 5
        
print(y.value_counts().sort_index())
#####################################################################
#####################################################################

label=LabelEncoder()
label.fit(y)
y=label.transform(y)


#1. 데이터
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, train_size=0.8,
    # stratify=y
)


xgb_params = {'learning_rate': 0.2218036245351803,
            'n_estimators': 10000,
            'max_depth': 3,
            'min_child_weight': 0.07709868781803283,
            'subsample': 0.80309973945344,
            'colsample_bytree': 0.9254025887963853,
            'gamma': 6.628562492458777e-08,
            'reg_alpha': 0.012998871754325427,
            'reg_lambda': 0.10637051171111844}


# 2 모델
model = XGBClassifier()
model.set_params(early_stopping_rounds=200, **xgb_params)


# 3 훈련

model.fit(x_train, y_train,
          eval_set=[(x_train,y_train), (x_test,y_test)],
          verbose=1,
          eval_metric='mlogloss'
)

# 4 평가, 예측
results = model.score(x_test,y_test)
print('최종점수', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict, average='micro')
print('add_score', acc)
print('F1 : ',f1)

# 최종점수 0.9681818181818181
# add_score 0.9681818181818181
# F1 :  0.968181818181818











