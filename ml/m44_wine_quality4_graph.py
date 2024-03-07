# ######################################################
# 그래프 그린다
# 1. value_counts -> 쓰지말것
# 2. np.unique 의 return_counts 쓰지말것


# ################# 3. groupby 사용할것, count() 사용할것 ##############

# plt.bar 로 그린다. (quality 컬럼)

# 힌트
# 데이터개수(y축) = 데이터갯수. 등 등 등

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


import matplotlib.pyplot as plt

# train_csv에서 quality에 대한 빈도수를 계산합니다.
quality_counts = train_csv.groupby('quality').count()

# quality_counts의 index는 quality 값이므로, 그대로 x축으로 사용합니다.
# y값은 데이터의 개수로 정합니다.
x_values = quality_counts.index
y_values = quality_counts['type']  # 'type'은 임의의 열을 사용하여 빈도수를 계산합니다.

# 시각화
plt.bar(x_values, y_values)
plt.xlabel('Quality')
plt.ylabel('Count')
plt.title('Distribution of Quality')
plt.show()