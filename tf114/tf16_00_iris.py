from sklearn.datasets import load_iris
import tensorflow as tf
tf.compat.v1.set_random_seed(777)
from sklearn.preprocessing import MinMaxScaler

# 1 데이터
data, target = load_iris(return_X_y=True)
x, y = data, target

x_data = x[y != 2]  # y =2 인값을 삭제하고 나머지만 가져온다 
y_data = y[y != 2]  # y =2 인값을 삭제하고 나머지만 가져온다 

scaler = MinMaxScaler()
x_data = scaler.fit_transform(x_data)

print(x_data.shape, y_data.shape)   # (100, 4) (100,)

x = tf.compat.v1.placeholder(tf.float32, shape=[None,4])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([4,1]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias')

hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)  # 행렬 곱계산할 때 행렬연산함수 사용

# ↓↓↓↓↓↓↓↓↓↓↓ binary_crossentropy  ↓↓↓↓↓↓↓↓↓↓↓↓
loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss)

# 3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 10001

for step in range(epoch):
    cost_val, _, w_v, b_v = sess.run([loss, train, w, b], feed_dict={x:x_data, y:y_data})
    
    if step % 20 == 0 :
        print(step, 'loss : ', cost_val)


print(type(w_v))    # <class 'numpy.ndarray'>
                    # 텐서플로 에서는 넘파이로 생성
                    

# 4 평가, 예측
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None,4])
y_pred = tf.sigmoid(tf.matmul(x_test, w_v) + b_v)
y_predict = sess.run(tf.cast(y_pred > 0.5, dtype=tf.float32), feed_dict={x_test:x_data})
# y_predict = sess.run(tf.cast(y_pred, dtype=tf.float32), feed_dict={x_test:x_data})
print(y_predict)

from sklearn.metrics import r2_score, mean_squared_error, f1_score, accuracy_score
import numpy as np

# 위의 결과는 회귀

# 시그모이드 하려면 y의 값이 0~1 사이의 값으로 감싸야한다


acc = accuracy_score(y_data, y_predict)
# acc = accuracy_score(y_data, np.round(y_predict))

print('acc : ', acc)    # acc :  0.56
    

sess.close()