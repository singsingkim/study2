import tensorflow as tf
tf.compat.v1.set_random_seed(777)

변수 = tf.compat.v1.Variable(tf.random_normal([2]), name='weight')
print(변수) # <tf.Variable 'weight:0' shape=(2,) dtype=float32_ref>

# 초기화 첫 번째
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())   # 변수초기화 안하면 에러 뜸
aaa = sess.run(변수)
print('aaa : ', aaa)    # aaa :  [ 2.2086694  -0.73225045]  # 시드 777 일 때
sess.close()


# 초기화 두 번째
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = 변수.eval(session = sess) # 텐서플로 데이터형인 '변수'를 파이썬에서 볼 수 있게 바꿔준다.
print('bbb : ', bbb)    # bbb :  [ 2.2086694  -0.73225045]
sess.close()


# 초기화 세 번째
sess = tf.compat.v1.InteractiveSession()    # 이후 부터는 각 세션을 통과시키지 않고 eval 시키면 된다.
sess.run(tf.compat.v1.global_variables_initializer())
ccc = 변수.eval()
print('ccc : ', ccc)    # ccc :  [ 2.2086694  -0.73225045]
sess.close()    # 메모리를 수동으로 닫아주는 행위

# 초기화 네 번째
