import tensorflow as tf
tf.compat.v1.set_random_seed(777)

# 1 데이터
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [[0],[1],[1],[0]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2, 1]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias')


# [실습] 맹그러바

# 2 모델
hypothesis = tf.compat.v1.nn.sigmoid(tf.compat.v1.matmul(x, w) + b)

# 3-1 컴파일
loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis))

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)
# train = optimizer.minimize(loss)
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)


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
                    
# x_tset = tf.compat.v1.placeholder(tf.float32, shape=([None, 4]))

y_pred = tf.matmul(x, w_v) + b_v

y_predict = sess.run(y_pred, feed_dict={x:x_data}).round()
print(y_predict)

from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
acc = accuracy_score(y_data, y_predict)
# mse = mean_squared_error(y_data, y_predict)

print('acc : ', acc)      
# print('mse : ', mse)    

sess.close()

# 위의 결과는 회귀