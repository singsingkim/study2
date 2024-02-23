import warnings


import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import time

# 1
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, test_size=0.3,
    shuffle=True, random_state=4)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler


# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.min(x_train))  # 0.0
# print(np.min(x_test))   # -0.010370370370370367
# print(np.max(x_train))  # 1.0
# print(np.max(x_test))   # 1.0789745710150918

# scaler = MaxAbsScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.min(x_train))  # 
# print(np.min(x_test))   # 
# print(np.max(x_train))  # 
# print(np.max(x_test))   # 

# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.min(x_train))  # 
# print(np.min(x_test))   # 
# print(np.max(x_train))  # 
# print(np.max(x_test))   # 

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train))  # 
print(np.min(x_test))   # 
print(np.max(x_train))  # 
print(np.max(x_test))   # 


# # 2
model = Sequential()
model.add(Dense(64, input_dim = 13))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

# model.summary()


# # 3
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=0, restore_best_weights=True)

mcp = ModelCheckpoint(
    monitor='val_loss', mode = 'auto', verbose=1,save_best_only=True,
    filepath='..\_data\_sava\MCP\keras25_MCP3.hdf5'
)

model.compile(loss='mse', optimizer='adam')

hist = model.fit(x_train, y_train,
          callbacks=[es, mcp],
          epochs=10, batch_size=10, validation_split=0.2)
model.save("..\_data\_save\keras25_3_save_model.h5")

# model = load_model('..\_data\_save\MCP\keras25_MCP1.hdf5')
# 체크포인트로 저장한것도 모델과 가중치가 같이 저장된다.

# 4
print("==================== 1. 기본 출력 ======================")

loss = model.evaluate(x_test, y_test, verbose=0)
print("로스 : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_predict, y_test)
print("R2 스코어 : ", r2)

print("==================== 2. load_model 출력 =================")

model2 = load_model('..\_data\_save\keras25_3_save_model.h5')
loss2 = model2.evaluate(x_test, y_test, verbose=0)
print("로스 : ", loss2)

y_predict2 = model2.predict(x_test)
r2 = r2_score(y_predict, y_test)
print("R2 스코어 : ", r2)

print("==================== 3. MCP 출력 =================")

model3 = load_model('..\_data\_sava\MCP\keras25_MCP3.hdf5')
loss3 = model3.evaluate(x_test, y_test, verbose=0)
print("로스 : ", loss3)

y_predict3 = model3.predict(x_test)
r2 = r2_score(y_predict, y_test)
print("R2 스코어 : ", r2)