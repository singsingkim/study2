import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense

# 1 데이터
x_data = np.array([[0,0],[0,1],[1,0],[1,1]])
y_data = np.array([0,1,1,0])
print(x_data.shape, y_data.shape)   # (4, 2) (4,)

# 2 모델
# model = LinearSVC()
# model = Perceptron()
model = Sequential()
model.add(Dense(1, input_dim = 2, activation='sigmoid'))  # 4행 2열 # 0 또는 1 이니까 시그모이드


# 3 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)

# 4 평가, 예측
# acc = model.score(x_data, y_data)
# print('model.score : ', acc)
results = model.evaluate(x_data, y_data)
print('acc : ', results[1])

y_predict = model.predict(x_data)
print(y_predict)
y_predict = np.round(y_predict)#.reshape(-1,).astype(int)
acc2 = accuracy_score(y_predict, y_data)    # y_predict 는 현재 sigmoid 통과한 상태
print('accuracy_score : ', acc2)
print('='*70)
print(y_data)
print(y_predict)
