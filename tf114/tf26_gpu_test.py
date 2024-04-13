import tensorflow as tf

##########################################################################
# tf.compat.v1.enable_eager_execution()       # 즉시실행 모드 1.0
# 텐서플로 버전 :  1.14.0
# 즉시실행모드 :  True

# tf.compat.v1.disable_eager_execution()    # 즉시모드 모드 1.0
# 텐서플로 버전 :  1.14.0
# 즉시실행모드 :  False
##########################################################################
# tf.compat.v1.enable_eager_execution()       # 즉시실행 해 2.0
# 텐서플로 버전 :  2.9.0
# 즉시실행모드 :  True

tf.compat.v1.disable_eager_execution()    # 즉시모드 해 2.0
# 텐서플로 버전 :  2.9.0
# 즉시실행모드 :  False
##########################################################################



print('텐서플로 버전 : ', tf.__version__)   
print('즉시실행모드 : ', tf.executing_eagerly())


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print(gpus[0])
    except RuntimeError as e:
        print(e)
else :
    print('GPU 없다')


##########################################################################
# 텐서플로 버전 :  1.14.0
# 즉시실행모드 :  True
# GPU 없다

# 텐서플로 버전 :  1.14.0
# 즉시실행모드 :  False
# GPU 없다
##########################################################################
# 텐서플로 버전 :  2.9.0
# 즉시실행모드 :  True
# PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')

# 텐서플로 버전 :  2.9.0
# 즉시실행모드 :  False
# PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')
##########################################################################
