import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import cifar10
from keras.callbacks import EarlyStopping
import tensorflow as tf

tf.random.set_seed(777)
np.random.seed(777)
print(tf.__version__)   # 2.9.0

from keras.applications import VGG16

# 1 데이터 
# CIFAR-10 데이터 로드
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 데이터 전처리
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


# 2 모델
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

vgg16.trainable = False     # 가중치를 동결한다

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

# 3 컴파일, 훈련
es = EarlyStopping(monitor='val_loss', mode='min', patience=6000, restore_best_weights=True)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=1024, verbose=1, validation_split=0.2, callbacks=[es])

# 4 평가, 예측
results = model.evaluate(x_test, y_test)
predict = np.argmax( model.predict(x_test),axis=1)
print('loss = ', results[0])
print('acc = ', results[1])

