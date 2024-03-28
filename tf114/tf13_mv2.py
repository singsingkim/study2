############## 맹그러바 ##############
import tensorflow as tf
tf.set_random_seed(777)

#1 데이터
x_data = [[73,51,65],
          [92,98,11],
          [89,31,33],
          [99,33,100],
          [17,66,79]]   # 5, 3

y_data = [[152],[185],[180],[205],[142]]    # 5, 1


# 모델 = 하이퍼시스 = x * w + b

# x = tf.compat.v1.placeholder(tf.float32, shape=(5,3))
# y = tf.compat.v1.placeholder(tf.float32, shape=(5,1))

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
# input_shape = (3,)
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# w = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32, name='weight')
w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3, 1]), dtype=tf.float32, name='weight')
# 인풋 x 의 열의 개수와 동일해야한다
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32, name='bias')

#   x         w         y
# (N, 13) * (13, 1) = (N, 1)

# 모델
# hypothesis = x * w * b
# 엑스는 (5, 3) * 가중치는 (3,1) 이렇게 곱해져야 (5, 1) 쉐잎이 나온다

hypothesis = tf.compat.v1.matmul(x, w) + b  # 행렬 계산할 때 행렬연산함수 사용

# 3-1 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000001)
train = optimizer.minimize(loss)


# 3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 1001

for step in range(epoch):
    cost_val, _ = sess.run([loss, train], feed_dict={x:x_data, y:y_data})
    
    if step % 20 == 0 :
        print(step, 'loss : ', cost_val)
sess.close()