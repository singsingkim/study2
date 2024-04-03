import tensorflow as tf
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_testn = to_categorical(y_test)

print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape)  # (60000, 10) (10000,)

x_train = x_train.reshape(60000, 28 * 28).astype('float32')/255
x_test = x_test.reshape(60000, 28 * 28).astype('float32')/255

# [실습] 맹그러바


