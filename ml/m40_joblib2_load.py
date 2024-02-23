from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#1. 데이터

x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, train_size=0.8,
    # stratify=y
)

scaler = MinMaxScaler()
# scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


# 2 모델
model = XGBClassifier()
# model = XGBRegressor(random_state = 123, **parameters)
import pickle 
import joblib
# path = 'C:\_data\_save\_pickle_test\\'
# model = pickle.load(open(path + 'm39_pickle1-save.dat', 'rb'))
path = 'C:\_data\_save\_joblib_test\\'
model = joblib.load( path + 'm40_joblib1_save.dat')

# 3 훈련
model.fit(x_train, y_train, 
          eval_set=[(x_train, y_train), (x_test, y_test)],
          verbose=100,
          )


# 4 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 : ', results)

from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import roc_auc_score, f1_score
y_pred = model. predict(x_test)
print('f1 : ', f1_score(y_test, y_pred))


# f1 :  0.9650349650349651

#########################################################################
# import pickle 
# path = 'C:\_data\_save\_pickle_test\\'
# pickle.dump(model, open(path + 'm39_pickle1-save.dat', 'wb'))

# 최종점수 :  0.9666666666666667
