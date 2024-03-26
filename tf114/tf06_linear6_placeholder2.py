import tensorflow as tf
tf.set_random_seed(123)

# 1. 데이터
x_data = [1,2,3,4,5]
y_data = [3,5,7,9,11]        # 웨이트2, 바이어스1 (x*2+1)
x = tf.compat.v1.placeholder(tf.float32, shape=[None])
y = tf.compat.v1.placeholder(tf.float32, shape=[None])



# y = wx + b
# w = tf.Variable(111, dtype=tf.float32)  # weight
# b = tf.Variable(0, dtype=tf.float32)    # bias
w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)  # 정규분포
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)  # 정규분포
print(w)    # <tf.Variable 'Variable:0' shape=(1,) dtype=float32_ref>

# sess = tf.compat.v1.Session()
# sess.run(tf.global_variables_initializer()) # 변수 초기화
# print(sess.run(w))  # [-1.5080816] - 시드고정했을때 고정됌


# 2. 모델 
# y = wx + b
# hypothesis = w * x + b   # 이거 아니다. 이제는 말할 수 있다.
hypothesis = x * w + b      # 실질적인 predict

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse(차이의 제곱의 평균)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)   # GD = 경사하강법(loss 최소화) / lr = 보폭을 얼마나 잡는지
train = optimizer.minimize(loss)    # loss 를 최소화 시키겠다.
# model.compile(loss='mse', optimizer='sgd') 여기까지 진행한것

# 3-2. 훈련
# sess = tf.compat.v1.Session()
# 2번 파일과 같이 close 를 잡거나, with 로 범위 지정하여 자동 close
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer()) # 변수를 초기화 시키겠다.

    # model.fit
    epochs = 101
    for step in range(epochs):
        # sess.run(train)         # 여기까지는 단순히 1 에포
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                             feed_dict={x:x_data, y:y_data})
        if step % 20 == 0:
            print(step, loss_val, w_val, b_val)
            # print(step, sess.run(loss), sess.run(w), sess.run(b))   # verbose 와 model.weight 에서 확인했던 놈들.
            # 이거 아님 print(step, sess.run(hypothesis, feed_dict={x:range(5), w:range(3,len(5),2), b:1}))   # verbose 와 model.weight 에서 확인했던 놈들.
    # sess.close()        
    
        

