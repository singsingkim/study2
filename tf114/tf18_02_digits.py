import tensorflow as tf
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
tf.set_random_seed(777)

data, target = load_digits(return_X_y=True)

x_data = data
y_data = target
print(x_data.shape, y_data.shape)   # (1797, 64) (1797,)
print(x_data[:1])
   