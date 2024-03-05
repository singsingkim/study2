import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# 데이터 불러오기
x, y = load_breast_cancer(return_X_y=True)

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123
)

# 데이터 스케일링
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# XGBoost 모델 및 초기 파라미터 설정
parameters = {
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'max_depth': 3,
    'gamma': 0,
    'min_child_weight': 0.2,
    'subsample': 0.4,
    'colsample_bytree': 0.1,
    'colsample_bylevel': 0.1,
    'colsample_bynode': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': 0,
}

model = XGBClassifier(**parameters)

# 모델 훈련
model.fit(x_train_scaled, y_train,
          eval_set=[(x_train_scaled, y_train), (x_test_scaled, y_test)],
          verbose=1,
          eval_metric='logloss'
          )

# 초기 모델 평가
initial_accuracy = model.score(x_test_scaled, y_test)
print('초기 모델 성능:', initial_accuracy)

# 피처 중요도 기반으로 피처 제거 및 평가
num_features = x_train.shape[1]
best_accuracy = initial_accuracy
best_num_features = num_features

for i in range(num_features, 0, -1):
    feature_indices = np.argsort(model.feature_importances_)[::-1][:i]
    x_train_subset = x_train_scaled[:, feature_indices]
    x_test_subset = x_test_scaled[:, feature_indices]

    # 모델 다시 훈련
    model.fit(x_train_subset, y_train,
              eval_set=[(x_train_subset, y_train), (x_test_subset, y_test)],
              verbose=0,
              eval_metric='logloss'
              )

    # 모델 평가
    accuracy = model.score(x_test_subset, y_test)
    print(f'{i}개의 피처를 사용한 모델의 성능:', accuracy)

    # 최고 성능 갱신 여부 확인
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_num_features = i

    # 피처 중요도 출력
    print('남은 피처 중요도:', model.feature_importances_)

print(f'최적의 성능을 얻는 피처 수: {best_num_features}, 최적의 성능: {best_accuracy}')
