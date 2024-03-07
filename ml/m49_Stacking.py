import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')


# 1  데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123,
    stratify=y
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# parameters = {
# 'n_estimators' : 1000,# / 디폴트 100/ 1~inf / 정수
# 'learning_rate' : 0.01,# / 디폴트 0.3 / 0~1 eta
# 'max_depth' : 3,# / 디폴트 6 / 0~inf / 정수 # 트리 깊이
# 'gamma' : 0,# / 디폴트 0 /0~inf
# 'min_child_weight' : 0.2,# / 디폴트 1 / 0~1
# 'subsam;le' : 0.4,      # 드랍아웃 개념

# 'colsample_bytree' : 0.1,# / 디폴트 1 / 0~1
# 'colsample_bylevel' : 0.1,# / 디폴트 1 / 0~1
# 'colsample_bynode' : 0.1,# / 디폴트 1 / 0~1
# 'reg_alpha' : 0.1,# / 디폴트 0 / 0~inf / L1 절대값 가중치 규제 / alpha # 알파, 람다, L1, L2 규제
# 'reg_lambda' : 0.1,# / 디폴트 1 / 0~inf / L2 제곱 가중치 규제 / lambda
# 'verbose' : 0,
# }

# 2 모델
# xgb = BaggingClassifier(XGBClassifier())
xgb = XGBClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression()

model_class = [xgb, rf, lr]
for model2 in model_class:
    model2.fit(x_train, y_train)
    y_pred = model2.predict(x_test)
    score2 = accuracy_score(y_test, y_pred)
    class_name = model2.__class__.__name__
    print('{0} 정확도 : {1:4f}'.format(class_name, score2)) # .format =  앞에 내용에 매치된다
    


# model2 = ()
# 맹그러
# XGBClassifier ACC : 0.000
# RandomForestClassifier ACC : 0.000
# LogisticRegression ACC : 0.000
# 스테킹 결과 : 0.000


# 스태킹 분류기 설정
stacking_model = StackingClassifier(
    estimators=[('LR',lr),('RF',rf),('xgb',xgb)],
    final_estimator=LogisticRegression(),  # 메타 모델
    # "메타 모델(Meta Model)"은 기본 모델들의 예측을 조합하여 최종 예측을 
    # 수행하는 모델을 말합니다. 스태킹(Stacking)이나 앙상블(Ensemble) 기법에서 사용되는 개념입니다.
    cv=5  # 기본 모델들을 훈련하기 위한 교차 검증
)

stacking_model.fit(x_train, y_train)
y_pred = stacking_model.predict(x_test)
stacking_score = accuracy_score(y_test, y_pred)
print('스태킹 분류기 정확도: {:.4f}'.format(stacking_score))

# XGBClassifier 정확도 : 0.991228
# RandomForestClassifier 정확도 : 0.982456
# LogisticRegression 정확도 : 0.991228
# 스태킹 분류기 정확도: 0.9912