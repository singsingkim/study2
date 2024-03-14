import tensorflow as tf
print(tf.__version__)           # 1.14.0
print(tf.executing_eagerly())   # False     # 즉시실행모드

# 즉시실행모드 -> 텐서1의 그래프형태의 구성없이 자연스러운 파이썬 문법으로 실행시킨다.

# tf.compat.v1.disable_eager_execution()  # 즉시실행모드 off  // 텐서플로 1.0 문법 // 디폴트
# tf.compat.v1.enable_eager_execution()   # 즉시실행모드 on   // 텐서플로 2.0 사용가능

print(tf.executing_eagerly())   # True

hello = tf.constant('hello world')

sess = tf.compat.v1.Session()

print(sess.run(hello))


#     가상환경     즉시실행모드         사용가능
# 1.14.0          disable(디폴트)       가능
# 1.14.0          enable                불가능
# 2.9.0           disable               가능
# 2.9.0           enable(디폴트)        불가능  

# 텐서2 에서 텐서1 을 사용할려면 eagermode를 disable 로 잡아야 사용 가능