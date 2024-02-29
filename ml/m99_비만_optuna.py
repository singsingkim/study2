import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
import optuna

# 데이터 불러오기
path = "c:/_data/kaggle/비만/"
train = pd.read_csv(path + "train.csv", index_col=0)
test = pd.read_csv(path + "test.csv", index_col=0)
sample = pd.read_csv(path + "sample_submission.csv")

x = train.drop(['NObeyesdad'], axis=1)
y = train['NObeyesdad']

TRAINSIZE = 0.85
# RS = 61
NUM = 96
SAVENAME = f'biman{NUM}'

lb = LabelEncoder()

# 라벨 인코딩할 열 선택
columns_to_encode = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

# 라벨 인코딩 수행
for column in columns_to_encode:
    lb.fit(x[column])
    x[column] = lb.transform(x[column])

for column in columns_to_encode:
    lb.fit(test[column])
    test[column] = lb.transform(test[column])

# 훈련 데이터와 테스트 데이터로 분할
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=TRAINSIZE, random_state=123, stratify=y)

import random
r = random.randint(1, 100)

# 목적 함수 정의 및 최적화
def objective(trial):
    lgbm_params = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": r,
        "num_class": 7,
        "learning_rate": trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'feature_pre_filter': False,
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 1.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 1.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
    }

    # LightGBM 모델 생성 및 훈련
    model = LGBMClassifier(**lgbm_params)
    model.fit(x_train, y_train)

    # 검증 세트에 대한 예측 및 정확도 계산
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

# 하이퍼파라미터 최적화를 위한 Optuna를 사용한 study 생성
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# 최적의 하이퍼파라미터 확인
best_params = study.best_params
print("Best Hyperparameters:", best_params)

# 최적의 하이퍼파라미터로 모델 재훈련
best_model = LGBMClassifier(**best_params)
best_model.fit(x_train, y_train)

# 최적 모델 저장 및 테스트 세트에 대한 예측 및 제출 파일 저장
y_submit = best_model.predict(test)
sample['NObeyesdad'] = y_submit
sample.to_csv(path + F"{SAVENAME}.csv", index=False)

# 랜덤 r 값 출력
print("optuna Random seed (r):", r)

# 최고의 trial 정보 출력
print("Best trial:")
trial = study.best_trial
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

