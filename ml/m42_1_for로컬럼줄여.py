# 1. 켄서
# 2. digits
# 3. fetch_covtype
# 4. dacon_wine
# 5. dacon_대출
# 6. kaggle 비만도
# 7. load_diabetes
# 8. california
# 9. dacon 따릉이
# 10. kaggle bike

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

# 1  데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = {
'n_estimators' : 1000,# / 디폴트 100/ 1~inf / 정수
'learning_rate' : 0.01,# / 디폴트 0.3 / 0~1 eta
'max_depth' : 3,# / 디폴트 6 / 0~inf / 정수 # 트리 깊이
'gamma' : 0,# / 디폴트 0 /0~inf
'min_child_weight' : 0.2,# / 디폴트 1 / 0~1
'subsam;le' : 0.4,      # 드랍아웃 개념

'colsample_bytree' : 0.1,# / 디폴트 1 / 0~1
'colsample_bylevel' : 0.1,# / 디폴트 1 / 0~1
'colsample_bynode' : 0.1,# / 디폴트 1 / 0~1
'reg_alpha' : 0.1,# / 디폴트 0 / 0~inf / L1 절대값 가중치 규제 / alpha # 알파, 람다, L1, L2 규제
'reg_lambda' : 0.1,# / 디폴트 1 / 0~inf / L2 제곱 가중치 규제 / lambda
'verbose' : 0,
}

# 2 모델
model = XGBClassifier()
model.set_params(early_stopping_rounds=10, **parameters)


# 3 훈련

model.fit(x_train, y_train,
          eval_set=[(x_train,y_train), (x_test,y_test)],
          verbose=1,
          eval_metric='logloss'
)

# 4 평가, 예측
results = model.score(x_test,y_test)
print('최종점수', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('add_score', acc)

######################################################################
print(model.feature_importances_)

# for문을 사용해서 피처가 약한놈분터 하나씩 제거해서
# 30, 29, 28, 27... 1 까지  


