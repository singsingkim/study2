import tensorflow as tf
tf.set_random_seed(123)

# 1. 데이터
x = [1,2,3,4,5]
y = [1,2,3,4,5]

# y = wx + b
w = tf.Variable(111, dtype=tf.float32)  # weight
b = tf.Variable(0, dtype=tf.float32)    # bias
   
###[실습] 맹그러!!
# 2. 모델
hypothesis = x * w + b
       
# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# 3-2, 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

# 4. 예측
# 그래프 모양의 역순으로 생각하기
epochs = 101
for step in range(epochs):
    sess.run(train)
    print(step, sess.run(loss), sess.run(w), sess.run(b))
sess.close()    
