# 1. 켄서
# 2. digits
# 3. fetch_covtype
# 4. dacon_wine
# 5. dacon_dechul
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
import warnings

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


thresholds = np.sort(model.feature_importances_)    # 오름차순
print(thresholds)
# [0.00666665 0.00782548 0.01010969 0.01129495 0.0117033  0.01215088
#  0.01375018 0.01423548 0.01443594 0.01670443 0.01775328 0.02069571
#  0.0214064  0.02521631 0.02653417 0.02812296 0.03377594 0.03770857
#  0.03924541 0.04651086 0.04680559 0.04963256 0.05300157 0.05310598
#  0.0556553  0.05593169 0.05786498 0.06428867 0.07019304 0.077674  ]
print('====================================================================')

from sklearn.feature_selection import SelectFromModel

for i in thresholds:     # i 에 thresholds 첫번재 두번째 순서대로 들어간다
    selection = SelectFromModel(model, threshold=i, prefit=False)   # 클래스를 인스턴스화 한다 # 크거나 같은놈을 살린다
    
    select_x_train = selection.transform(x_train)   # x_train 을 select_x_train 으로 변환
    select_x_test = selection.transform(x_test)   # x_test 을 select_x_test 으로 변환
    print(i,'\t변형된 x_train',select_x_train.shape, '\t변형된 x_test',select_x_test.shape)

    select_model = XGBClassifier()
    select_model.set_params(
        early_stopping_rounds=10,
        **parameters,
        eval_metric='logloss',
    )

    select_model.fit(select_x_train, y_train,
                     eval_set=[(select_x_train,y_train),(select_x_test,y_test)],
                    #  verbose=0
                     )

    select_y_predict = select_model.predict(select_x_test)
    score = accuracy_score(y_test,select_y_predict)
    
    print('Trech=%.3f, n=%d, ACC: %2f%%'%(i, select_x_train.shape[1], score*100))









