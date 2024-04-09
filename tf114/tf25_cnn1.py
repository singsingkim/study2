import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(777)

#1 데이터
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.


#2 모델
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 28, 28, 1])  # 4차원
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])  


# layer1
w1 = tf.compat.v1.get_variable('w1', shape=[2, 2, 1, 64])
                                            # 커널사이즈, 컬러(채널), 필터(아웃풋)

l1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='VALID') # 4차원 맞추기 위한 형식 / 세칸 뛰우고 싶으면 : [1,3,3,1]
# model.add(Conv2d(64, kernel_size=(2,2), input_shape=(28,28,1), stride=(1,1)))

print(w1)   # <tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
print(l1)   # Tensor("Conv2D:0", shape=(?, 27, 27, 64), dtype=float32)


# layer2
w2 = tf.compat.v1.get_variable('w2', shape=[3, 3, 64, 32])  # 레이어1의 필터가 컬러로 들어온다

l2 = tf.nn.conv2d(l1, w2, strides=[1,2,2,1], padding='SAME')

print(w2)   # <tf.Variable 'w2:0' shape=(3, 3, 64, 32) dtype=float32_ref>
print(l2)   # Tensor("Conv2D_1:0", shape=(?, 14, 14, 32), dtype=float32)








print(np.__version__)

