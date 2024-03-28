# [실습] 맹그러

#1 데이터
x1_data = [73.,93.,89.,96.,73.]
x2_data = [80.,88.,91.,98.,66.]
x3_data = [75.,93.,90.,100.,70.]
y_data = [152.,185.,180.,196.,142.]
# y = w1x1 + w2x2 + w3x3 + b : 밸류어블4개(w3, b1), 플레이스홀더 4개
# 그라디언트디센트옵티마이저 사용
#################################################
import tensorflow as tf
tf.set_random_seed(777)

x1 = tf.compat.v1.placeholder(tf.float32)
x2 = tf.compat.v1.placeholder(tf.float32, shape=[None])
x3 = tf.compat.v1.placeholder(tf.float32, shape=[None])
y = tf.compat.v1.placeholder(tf.float32)


w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32, name='weight')
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32, name='weight')
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32, name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32, name='weight')


# 2 모델
hypothesis = x3 * w3 + x2 * w2 + x1 * w1 + b


# 3-1 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)   # GD = 경사하강법(loss 최소화) / lr = 보폭을 얼마나 잡는지
train = optimizer.minimize(loss)    # loss 를 최소화 시키겠다.


# 3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())


epoch = 1001

for step in range(epoch):
    cost_val, _ = sess.run([loss, train], feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
    
    if step % 20 == 0:
        print(step, 'loss : ', cost_val)
sess.close()        
    
