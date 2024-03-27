import tensorflow as tf
tf.compat.v1.set_random_seed(777)

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
hypothesis = x * w + b      # 실질적인 predict  옵티마이저 사용하기 위해

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse(차이의 제곱의 평균)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0823)   # GD = 경사하강법(loss 최소화) / lr = 보폭을 얼마나 잡는지
train = optimizer.minimize(loss)    # loss 를 최소화 시키겠다.
# model.compile(loss='mse', optimizer='sgd') 여기까지 진행한것


# [실습]
# 07_2 를 카피해서 아래를 맹글어바

################### 1. Session() // sess.run(변수) ##########################
sess = tf.compat.v1.Session()
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

# 4 예측
x_pred = [6,7,8]
# 예측값을 뽑아봐
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

# y_predict = xw + b

# 1 파이썬방식 해결
y_pred = x_pred * w_val + b_val
print('[6,7,8]의 예측 : ', y_pred)

# 2 placeholder 해결
y_pred2 = x_test * w_val + b_val
print('[6,7,8]의 예측 : ', sess.run(y_pred2, feed_dict={x_test:x_pred}), w_val, b_val)

sess.close()


################### 2. Session() // 변수.eval(session=sess) ##########################
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer()) # 변수를 초기화 시키겠다.

# model.fit
epochs = 101
for step in range(epochs):
    # sess.run(train)         # 여기까지는 단순히 1 에포
    # _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
    #                                         feed_dict={x:x_data, y:y_data})

    # w와 b만 빼서 w.eval()과 b.eval() 형태로 변환
    _, loss_val = sess.run([train, loss], feed_dict={x:x_data, y:y_data})
    w_val = w.eval(session=sess)
    b_val = b.eval(session=sess)

    if step % 20 == 0:
        print(step, loss_val, w_val, b_val)
        # print(step, sess.run(loss), sess.run(w), sess.run(b))   # verbose 와 model.weight 에서 확인했던 놈들.

# 4 예측
x_pred = [6,7,8]
# 예측값을 뽑아봐
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

# y_predict = xw + b

# 1 파이썬방식 해결
y_pred = x_pred * w_val + b_val
print('[6,7,8]의 예측 : ', y_pred)

# 2 placeholder 해결
y_pred2 = x_test * w_val + b_val
print('[6,7,8]의 예측 : ', sess.run(y_pred2, feed_dict={x_test:x_pred}), w_val, b_val)

sess.close()





################### 3. interactiveSession() // 변수.eval() ##########################
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.global_variables_initializer()) # 변수를 초기화 시키겠다.

# model.fit
epochs = 101
for step in range(epochs):
    # sess.run(train)         # 여기까지는 단순히 1 에포
    # _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
    #                                         feed_dict={x:x_data, y:y_data})

    # w와 b만 빼서 w.eval()과 b.eval() 형태로 변환
    _, loss_val = sess.run([train, loss], feed_dict={x:x_data, y:y_data})
    w_val = w.eval()
    b_val = b.eval()

    if step % 20 == 0:
        print(step, loss_val, w_val, b_val)
        # print(step, sess.run(loss), sess.run(w), sess.run(b))   # verbose 와 model.weight 에서 확인했던 놈들.

# 4 예측
x_pred = [6,7,8]
# 예측값을 뽑아봐
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

# y_predict = xw + b

# 1 파이썬방식 해결
y_pred = x_pred * w_val + b_val
print('[6,7,8]의 예측 : ', y_pred)

# 2 placeholder 해결
y_pred2 = x_test * w_val + b_val
print('[6,7,8]의 예측 : ', sess.run(y_pred2, feed_dict={x_test:x_pred}), w_val, b_val)

sess.close()
