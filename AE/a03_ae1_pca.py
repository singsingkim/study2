# 비지도학습에서 y는X다.
import numpy as np
from keras.datasets import mnist
import tensorflow as tf
tf.random.set_seed(888)
np.random.seed(888)


#1. 데이터
(X_train, _), (X_test, _) = mnist.load_data()

X_train = X_train.reshape(60000, 28*28).astype('float32')/255.
X_test = X_test.reshape(10000, 28*28).astype('float32')/255.

X_train_noised = X_train + np.random.normal(0, 0.1, size=X_train.shape) # 평균 0, 표준편차 0.1
X_test_noised = X_test + np.random.normal(0, 0.1, size=X_test.shape)
print(np.max(X_train_noised), np.min(X_train_noised))

X_train_noised = np.clip(X_train_noised, 0, 1)
X_test_noised = np.clip(X_test_noised, 0, 1)

print(np.max(X_train_noised), np.min(X_train_noised))


# 2.모델
from keras.models import Sequential, Model
from keras.layers import Dense, Input

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(28*28,)))
    model.add(Dense(784, activation='sigmoid'))
    return model


hidden_size = 713   # PCA 1.0 일 떄 성능
# hidden_size = 486   # PCA 0.999 일 떄 성능
# hidden_size = 331   # PCA 0.99 일 떄 성능
# hidden_size = 154   # PCA 0.95 일 떄 성능


model = autoencoder(hidden_layer_size=hidden_size)    


'''


input_img = Input(shape=(784,))

###인코더
encoded = Dense(64, activation='relu')(input_img)   # 연산량 784X64 + 64X785
# encoded = Dense(32, activation='relu')(input_img)   # 연산량 784X33 + 32X785
# encoded = Dense(1, activation='relu')(input_img)   # 연산량 784X2 + 1X785
# encoded = Dense(1024, activation='relu')(input_img)   # 연산량 784X1025 + 1024X785


###디코더
decoded = Dense(784, activation='linear')(encoded)        # activation을 통과했을때 모양을 생각해보고 사용하기
# decoded = Dense(784, activation='sigmoid')(encoded)
# decoded = Dense(784, activation='relu')(encoded)
# decoded = Dense(784, activation='tanh')(encoded)


autoencoder = Model(input_img, decoded)

autoencoder.summary()
# Total params: 101,200
'''

# 3. 컴파일, 훈련
model.compile(optimizer='adam', loss='mse', )
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy', )

model.fit(X_train_noised, X_train, epochs=30, batch_size=128, validation_split=0.2)


# 4.평가, 예측
decoded_imgs = model.predict(X_test_noised)
# evaluate는 지표를 신뢰하기 힘듬

import matplotlib.pyplot as plt

n = 10
plt.figure(figsize=(20, 4))

for i in range(n):
    ax = plt.subplot(3, n, i+1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(X_test_noised[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, n, i + 1 + n + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

