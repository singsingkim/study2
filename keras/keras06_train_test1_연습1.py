import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])
x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 1000, batch_size = 2)

loss = model.evaluate(x_test, y_test)
result = model.predict([11000,7])

print("로스 : ", loss)
print("[10]의 예측값 : ", result)