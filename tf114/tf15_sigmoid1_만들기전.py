import tensorflow as tf
tf.compat.v1.set_random_seed(777)

# 1 데이터
x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2],] # (6,2)
y_data = [[0],[0],[0],[1],[1],[1],] # (6,1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None,2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,1]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias')

#   x         w         y
# (N, 2) * (2, 1) = (N, 1)

# 2  모델
# hypothesis = x * w * b
# 엑스는 (6, 2) * 가중치는 (2,1) 이렇게 곱해져야 (6, 1) 쉐잎이 나온다

hypothesis = tf.compat.v1.matmul(x, w) + b  # 행렬 곱계산할 때 행렬연산함수 사용

# 3-1 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000001)
train = optimizer.minimize(loss)


# 3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 1001

for step in range(epoch):
    cost_val, _, w_v, b_v = sess.run([loss, train, w, b], feed_dict={x:x_data, y:y_data})
    
    if step % 20 == 0 :
        print(step, 'loss : ', cost_val)


print(type(w_v))    # <class 'numpy.ndarray'>
                    # 텐서플로 에서는 넘파이로 생성
                    

x_test = tf.compat.v1.placeholder(tf.float32, shape=[None,2])

# y_pred = x_test * w_v + b_v
y_pred = tf.matmul(x_test, w_v) + b_v
y_predict = sess.run(y_pred, feed_dict={x_test:x_data})
print(y_predict)

from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_data, y_predict)
mse = mean_squared_error(y_data, y_predict)

print('r2 : ', r2)      # r2 :  -58.95523263996157
print('mse : ', mse)    # mse :  14.988808159990393

sess.close()

# 위의 결과는 회귀