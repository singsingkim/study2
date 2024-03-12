# restore_best_weights
# save_best_only
# 에 대한 고찰

import warnings
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score, accuracy_score, f1_score
import time

# 1
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x, x.shape)   # (569, 30)
print(y, y.shape)   # (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, test_size=0.3,
    shuffle=True, random_state=123,
    # stratify=y
    )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train))  # 
print(np.min(x_test))   # 
print(np.max(x_train))  # 
print(np.max(x_test))   # 


# # 2
model = Sequential()
model.add(Dense(64, input_dim = 30))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# model.summary()


# # 3 컴파일 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

learning_rate = 0.0001    # 1.0 / 0.1 / 0.01 / 0.001 / 0.0001 # 100에포 고정

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate))

model.fit(x_train, y_train,
          epochs=100, batch_size=10, validation_split=0.2)


# 4 평가, 예측
print("==================== 1. 기본 출력 ======================")

loss = model.evaluate(x_test, y_test, verbose=0)

y_predict = model.predict(x_test)
y_predict = np.around(model.predict(x_test))

print("lr : {0}, 로스 : {1:.4f}".format(learning_rate, loss))

acc = accuracy_score(y_predict, y_test)
print("lr : {0}, acc : {1:.4f}".format(learning_rate, acc))

f1 = f1_score(y_predict, y_test)
print("lr : {0}, f1 : {1:.4f}".format(learning_rate, f1))

# lr : 1.0, 로스 : 0.6983
# lr : 1.0, acc : 0.6023
# lr : 1.0, f1 : 0.7518

# lr : 0.1, 로스 : 0.6767
# lr : 0.1, acc : 0.6023
# lr : 0.1, f1 : 0.7518

# lr : 0.01, 로스 : 0.3027
# lr : 0.01, acc : 0.9766
# lr : 0.01, f1 : 0.9810

# lr : 0.001, 로스 : 0.1153
# lr : 0.001, acc : 0.9766
# lr : 0.001, f1 : 0.9806

# lr : 0.0001, 로스 : 0.0716
# lr : 0.0001, acc : 0.9883
# lr : 0.0001, f1 : 0.9904

#########################실습####################################
# 4개의 파일을 lr 5개씩 줘서 성능비교
# 5. dacon_dechul
# 6. kaggle_biman

# 9. dacon_ddarung
# 10. kaggle_bike


