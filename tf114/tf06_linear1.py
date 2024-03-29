import tensorflow as tf
tf.set_random_seed(123)

# 1. 데이터
x = [1,2,3]
y = [1,2,3]

# y = wx + b
w = tf.Variable(111, dtype=tf.float32)  # weight
b = tf.Variable(0, dtype=tf.float32)    # bias

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
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer()) # 변수를 초기화 시키겠다.

# model.fit
epochs = 101
for step in range(epochs):
    sess.run(train)         # 여기까지는 단순히 1 에포
    if step % 20 == 0:
        print(step, sess.run(loss), sess.run(w), sess.run(b))   # verbose 와 model.weight 에서 확인했던 놈들.
sess.close()        
        

# 0 44682.156 100.73333 -4.4
# 20 616.13007 25.758467 -35.11292
# 40 197.22882 17.902163 -36.406902
# 60 175.841 16.470543 -34.97625
# 80 159.67189 15.682805 -33.35919
# 100 145.01625 14.98699 -31.793972        