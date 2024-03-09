# restore_best_weights
# save_best_only
# 에 대한 고찰

import warnings
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score
import time

# 1
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, test_size=0.3,
    shuffle=True, random_state=4,
    stratify=y)

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


# # 3
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


es = EarlyStopping(
    monitor='val_loss', 
    mode='min', 
    patience=20, 
    verbose=0, 
    restore_best_weights=True
    )

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    patience=10,    # es 페이션스 보다 적게 해야한다
    mode='auto',
    verbose=1,
    factor=0.5,     # 반으로 줄인다
    )

from keras.optimizers import Adam

learning_rate = 0.01       # adam 일 경우 디폴트 0.001

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate = learning_rate))

hist = model.fit(x_train, y_train,
          callbacks=[es, rlr],
          epochs=1000, batch_size=10, validation_split=0.2)
   
# model = load_model('..\_data\_save\MCP\keras25_MCP1.hdf5')     
# 체크포인트로 저장한것도 모델과 가중치가 같이 저장된다.

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


############################실습#################################
# 4개의 파일을 lr 5개씩 줘서 성능비교
# 5. dacon_dechul
# 6. kaggle_biman

# 9. dacon_ddarung
# 10. kaggle_bike


