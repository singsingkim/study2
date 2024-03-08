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
bayesian_params = {
    'learning_rate' : (0.001, 1),
    'max_depth' : (3, 10),
    'num_leaves' : (24, 40),
    'min_child_samples' : (10, 200),
    'min_child_weight' : (1, 50),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'max_bin' : (9, 500),
    'reg_lambda' : (-0.001, 10),
    'reg_alpha' : (0.01, 50),
}
# 위에 정의한것이 아래 함수에 매칭  ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
def xgb_hamsu(learning_rate, max_depth, num_leaves, min_child_samples, min_child_weight, 
              subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha):
    params = {
        'n_estimators' : 100,
        'learning_rate' : learning_rate,    # 디폴트 0.0001 ~ 0.1
        'max_depth' : int(round(max_depth)),    # 무조건 정수형 / 레이어의 깊이
        'num_leaves' : int(round(num_leaves)),  
        
        'min_child_samples' : int(round(min_child_samples)),
        'min_child_weight' : int(round(min_child_weight)),
        'subsample' : max(min(subsample, 1), 0),    # 0~1 사이의 값 / 1 이상이면 1 / 0 이하면 0
        'colsample_bytree' : colsample_bytree,
        'max_bin' : max(int(round(max_bin)), 10),   # 무조건 10 이상
        'reg_lambda' : max(reg_lambda, 0),          # 무조건 양수만
        'reg_alpha' : reg_alpha,
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
    return results

bay = BayesianOptimization(
    f = xgb_hamsu,
    pbounds = bayesian_params,
    random_state = 123,        
)

start_time = time.time()
n_iter = 100

bay.maximize(init_points=5, n_iter=n_iter)
end_time = time.time()

print(bay.max)
print(n_iter, '번 걸린시간 : ', round(end_time-start_time, 2), '초')

# {'target': 0.9912280701754386, 'params': {'colsample_bytree': 0.920794067252901, 'learning_rate': 0.5634566219126623, 
# 'max_bin': 49.49509534734083, 'max_depth': 5.155053048374102, 'min_child_samples': 52.49328402344391, 'min_child_weight': 4.397463098162169, 
# 'num_leaves': 34.86427982145582, 'reg_alpha': 0.5330304142630012, 'reg_lambda': 5.2203319810925315, 'subsample': 0.8730966057768486}}
# 100 번 걸린시간 :  13.31 초