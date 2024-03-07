import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

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
    x, y, random_state=123, train_size=0.8,
    # stratify=y
)

####################### smote ##########################
# 데이터 증폭하는 것 중 최고 짱짱
# 증폭량이 늘어날수록 속도가 느리다
print('================= smote =====================')
from imblearn.over_sampling import SMOTE
import sklearn as sk
print('사이킷런 버전 :', sk.__version__)    # 사이킷런 버전 : 1.1.3

smote = SMOTE(random_state=123, k_neighbors=4)
x_train, y_train = smote.fit_resample(x_train, y_train)
print(x_train.shape, y_train.shape) # (159, 13) (159,)


print(pd.value_counts(y_train)) 
# 0    53   
# 1    53
# 2    53
# y 값이  원래 71 -> 트레인스플릿 0.75 먹혀서 53 -> 나머지 53에 맞추어서 증폭
####################### smote ##########################


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
          eval_metric='auc'
)

# 4 평가, 예측
results = model.score(x_test,y_test)
print('최종점수', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('add_score', acc)













