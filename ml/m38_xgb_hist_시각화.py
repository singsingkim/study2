from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#1. 데이터

# x, y = load_breast_cancer(return_X_y=True)
x, y = load_diabetes(return_X_y=True)
# x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, train_size=0.8,
    # stratify=y
)

scaler = MinMaxScaler()
# scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

n_splits = 5
kFold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
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
    'n_estimators' : 4000,
    'learning_rate' : 0.005,
    'min_child_weight' : 10,
}

# 2 모델
model = XGBRegressor()
# model = XGBRegressor(random_state = 123, **parameters)

model.set_params(
    **parameters,
    random_state = 123,
    early_stopping_rounds=10,
    )

# 3 훈련
model.fit(x_train, y_train, 
          eval_set=[(x_train, y_train), (x_test, y_test)],
          verbose=100,
        #   eval_metric = 'rmse',       # 회귀 rmse
          eval_metric = 'mae',        # 
        #   eval_metric = 'logloss',    # 이진분류 디폴트 / accuracy 와 유사
        #   eval_metric = 'error',      # 이진분류
        #   eval_metric = 'mlogloss',   # 다중분류 디폴트 / accuracy 와 유사
        #   eval_metric = 'merror',     # 다중분류 
        #   eval_metric = 'auc',        # 이진 다중 전부 (하지만 이진 좋다)
          )


# 4 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 : ', results)
# 최종점수 :  0.956140350877193

from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import mean_absolute_error
y_pred = model. predict(x_test)
# print('r2 : ', r2_score(y_test, y_pred))
print('mae : ', mean_absolute_error(y_test, y_pred))
# print('acc : ', accuracy_score(y_test, y_pred))
# print('f1 : ', f1_score(y_test, y_pred))
# print('auc : ', roc_auc_score(y_test, y_pred))


# f1 :  0.9650349650349651

print('===========================================================')
hist = model.evals_result()
print(hist)

# import matplotlib.pyplot as plt
# plt.plot(hist)
# plt.show()


import matplotlib.pyplot as plt

# 훈련 및 검증 손실 시각화
train_error = hist['validation_0']['mae']
val_error = hist['validation_1']['mae']

epoch = range(len(train_error))

plt.figure(figsize=(10, 5), )
plt.plot(epoch, train_error, label='Train', color='blue')
plt.plot(epoch, val_error, label='Validation', color='red')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.title('Training and Validation Error')
plt.legend()
plt.grid(True)
plt.show()


