# 여태까지는 한 개의 y값만 찾아내고 있었다.
import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier

# 1 데이터
x, y = load_linnerud(return_X_y=True)
print(x, x.shape)   # (20, 3)
# [[  5. 162.  60.]
#  [  2. 110.  60.]
#  [ 12. 101. 101.]
#  [ 12. 105.  37.]
#  [ 13. 155.  58.]
#  [  4. 101.  42.]
#  [  8. 101.  38.]
#  [  6. 125.  40.]
#  [ 15. 200.  40.]
#  [ 17. 251. 250.]
#  [ 17. 120.  38.]
#  [ 13. 210. 115.]
#  [ 14. 215. 105.]
#  [  1.  50.  50.]
#  [  6.  70.  31.]
#  [ 12. 210. 120.]
#  [  4.  60.  25.]
#  [ 11. 230.  80.]
#  [ 15. 225.  73.]
#  [  2. 110.  43.]]
print(y, y.shape)   # (20, 3
# [[191.  36.  50.]
#  [189.  37.  52.]
#  [193.  38.  58.]
#  [162.  35.  62.]
#  [189.  35.  46.]
#  [182.  36.  56.]
#  [211.  38.  56.]
#  [167.  34.  60.]
#  [176.  31.  74.]
#  [154.  33.  56.]
#  [169.  34.  50.]
#  [166.  33.  52.]
#  [154.  34.  64.]
#  [247.  46.  50.]
#  [193.  36.  46.]
#  [202.  37.  62.]
#  [176.  37.  54.]
#  [157.  32.  52.]
#  [156.  33.  54.]
#  [138.  33.  68.]]

# 최종값 -> : x = [  2. 110.  43.] / y = [138.  33.  68.]

#234 모델, 훈련, 평가
model = RandomForestRegressor()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[2, 110, 43]]))    # (n, 1) 행렬형태로 만들어줌

# RandomForestRegressor 스코어 :  3.4277
# [[156.58  34.51  63.46]]

##################################################################################
model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[2, 110, 43]]))    # (n, 1) 행렬형태로 만들어줌

# LinearRegression 스코어 :  7.4567
# [[187.33745435  37.08997099  55.40216714]]

##################################################################################
model = Ridge()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[2, 110, 43]]))    # (n, 1) 행렬형태로 만들어줌

# Ridge 스코어 :  7.4569
# [[187.32842123  37.0873515   55.40215097]]


##################################################################################
model = XGBRegressor()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[2, 110, 43]]))    # (n, 1) 행렬형태로 만들어줌

# XGBRegressor 스코어 :  0.0008
# [[138.0005    33.002136  67.99897 ]]


##################################################################################
'''
model = LGBMRegressor()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[2, 110, 43]]))    # (n, 1) 행렬형태로 만들어줌

# 에러 뜬다
# 컬럼이 여러개 짜리인 멀티아웃풋에서는 안먹힌다.
'''
##################################################################################
model = MultiOutputRegressor(LGBMRegressor())
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[2, 110, 43]]))    # (n, 1) 행렬형태로 만들어줌

# MultiOutputRegressor 스코어 :  8.91
# [[178.6  35.4  56.1]]

##################################################################################
"""
model = CatBoostRegressor()
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[2, 110, 43]]))    # (n, 1) 행렬형태로 만들어줌
# Currently only multi-regression, multilabel and survival objectives work with multidimensional target
"""

# mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
model = CatBoostRegressor(loss_function='MultiRMSE')
model.fit(x, y)
y_pred = model.predict(x)
print(model.__class__.__name__, '스코어 : ',
      round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[2, 110, 43]]))    # (n, 1) 행렬형태로 만들어줌

# CatBoostRegressor 스코어 :  0.0638
# [[138.21649371  32.99740595  67.8741709 ]]
# mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm