import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN 
from keras.callbacks import EarlyStopping
# 1 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9]])
y = np.array([4,5,6,7,8,9,10])  
print(x.shape, y.shape) # (7, 3) (7,)
    # 3차원 모양 맞춰주기 위해 리쉐이프 필요

x = x.reshape(7,3,1)
print(x.shape, y.shape) # (7, 3, 1) (7,)

# 2 모델
model = Sequential()
# model.add(SimpleRNN(units=512, input_shape=(3,1))) # (timesteps, features)
# 3-D tensor with shape (batch_size, timesteps, features).
model.add(SimpleRNN(units=10, input_length=3, input_dim = 1))
# 위와 같다. 인풋랭스=타임스탭스 ,,, 인풋쉐이프=인풋딤
# model.add(SimpleRNN(units=10, input_dim = 1, input_length=3))
# 위와 같다. 가독성 편하게 하기 위에꺼 사용한다. 

model.add(Dense(32))
model.add(Dense(512))
model.add(Dense(32)) 
model.add(Dense(128))
model.add(Dense(32))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

# 3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss',
                mode='min',
                patience=500,
                verbose=1,
                restore_best_weights=True
                )
model.fit(x, y, epochs=10000, callbacks=[es])

# 4 평가, 예측
results  = model.evaluate(x,y)
print('로스는 : ',results)
y_pred = np.array([8,9,10]).reshape(1,3,1)
y_pred = model.predict(y_pred)
# (3,) -> (1, 3, 1)
print('[8,9,10] 의 결과 : ', y_pred)
