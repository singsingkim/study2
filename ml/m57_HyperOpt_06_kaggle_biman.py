# 함수의 최대값 찾기
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from bayes_opt import BayesianOptimization
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import time 
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


# 2 모델
search_space = {
    'learning_rate' : hp.uniform('learning_rate', 0.001, 1),
    'max_depth' : hp.quniform('max_depth', 3, 10, 1),
    'num_leaves' : hp.quniform('num_leaves', 24, 40, 1),
    'min_child_samples' : hp.quniform('min_child_samples', 10, 200, 1),
    'min_child_weight' : hp.quniform('min_child_weight', 1, 50, 1),
    'subsample' : hp.uniform('subsample', 0.5, 1),
    'colsample_bytree' : hp.uniform('colsample_bytree', 0.5, 1),
    'max_bin' : hp.quniform('max_bin', 9, 500, 1),
    'reg_lambda' : hp.uniform('reg_lambda', -0.001, 10),
    'reg_alpha' : hp.uniform('reg_alpha', 0.01, 50),
}
# 위에 정의한것이 아래 함수에 매칭  ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
def xgb_hamsu(search_space):
    params = {
        'n_estimators' : 100,
        'learning_rate' : search_space['learning_rate'],    # 디폴트 0.0001 ~ 0.1
        'max_depth' : int(round(search_space['max_depth'])),    # 무조건 정수형 / 레이어의 깊이
        'num_leaves' : int(round(search_space['num_leaves'])),  
        'min_child_samples' : int(round(search_space['min_child_samples'])),
        'min_child_weight' : int(round(search_space['min_child_weight'])),
        'subsample' : max(min(search_space['subsample'], 1), 0),    # 0~1 사이의 값 / 1 이상이면 1 / 0 이하면 0
        'colsample_bytree' : search_space['colsample_bytree'],
        'max_bin' : max(int(round(search_space['max_bin'])), 10),   # 무조건 10 이상
        'reg_lambda' : max(search_space['reg_lambda'], 0),          # 무조건 양수만
        'reg_alpha' : search_space['reg_alpha'],
    }
    
    model = XGBClassifier(**params, n_jobs = 1)
    model.fit(x_train, y_train,
                eval_set = [(x_train, y_train),(x_test, y_test)],
                eval_metric = 'logloss',
                verbose = 0,
                early_stopping_rounds=50,
                )

    y_pred = model.predict(x_test)
    results = accuracy_score(y_test, y_pred)
    return -results

trial_val = Trials()

start_time = time.time()

best = fmin(
    fn = xgb_hamsu,
    space=search_space,
    algo=tpe.suggest,
    max_evals=50,
    trials=trial_val,
    rstate = np.random.default_rng(seed=10),
)


n_iter = 100

end_time = time.time()

print('best : ', best)
print(n_iter, '번 걸린시간 : ', round(end_time-start_time, 2), '초')

# best :  {'colsample_bytree': 0.7140870481707279, 'learning_rate': 0.6088639454291519,
#          'max_bin': 385.0, 'max_depth': 4.0, 'min_child_samples': 171.0, 'min_child_weight': 2.0, 
#          'num_leaves': 33.0, 'reg_alpha': 31.513153448816254, 'reg_lambda': 1.6132569521484321, 
#          'subsample': 0.7528114984407226}