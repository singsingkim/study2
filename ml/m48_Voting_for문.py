import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
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

model = VotingClassifier(
                        estimators=[('LR',lr),('RF',rf),('xgb',xgb)],
                        voting='soft',
                        # voting='hard',  # 디폴트
                          )

# 3 훈련
model.fit(x_train, y_train,
        #   eval_set=[(x_train,y_train), (x_test,y_test)],
        #   verbose=1,
        #   eval_metric='logloss'
)
# "eval_set"을 사용하면 테스트 세트를 더 작은 데이터 세트로 나누어 
# 훈련한 모델을 평가할 수 있습니다.

# 4 평가, 예측
results = model.score(x_test,y_test)
print('최종점수', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('add_score', acc)

  
model_class = [xgb, rf, lr]
for model2 in model_class:
    model2.fit(x_train, y_train)
    y_pred = model2.predict(x_test)
    score2 = accuracy_score(y_test, y_pred)
    class_name = model2.__class__.__name__
    print('{0} 정확도 : {1:4f}'.format(class_name, score2)) # .format =  앞에 내용에 매치된다
    

   

