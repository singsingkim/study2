import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, Input

# 1 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

# 2 모델
def build_model(drop=0.5, optimizer='adam', activation='relu',
                node1=128, node2=64, node3=32, lr=0.001) : # 인풋값

    inputs = Input(shape=(28*28,), name='inputs')
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=optimizer, metrics=['acc'],
                  loss = 'sparse_categorical_crossentropy')
    return model

def create_hyperparameter():
    batchs = [100, 200, 300, 400, 500]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    node1 = [128, 64, 32, 16]
    node2 = [128, 64, 32, 16]
    node3 = [128, 64, 32, 16]
    return {'batch_size' : batchs,
            'optimizer' : optimizers,
            'drop' : dropouts,
            'activation' : activations,
            'node1' : node1,
            'node2' : node2,
            'node3' : node3
            }
    
hyperparameters = create_hyperparameter()
print(create_hyperparameter)

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor     # keras -> 사이킷런
keras_model = KerasClassifier(build_fn=build_model, verbose=1)

# model = RandomizedSearchCV(build_model, 
model = RandomizedSearchCV(keras_model, 
                           hyperparameters, cv=2, n_iter=1, n_jobs=1, verbose=1)    # 모델, 하이퍼파라미터, 크로스발리데이션

import time
start_time = time.time()
model.fit(x_train, y_train, epochs=3)   # 오류 : 케라스모델, 케라스파라미터 -> 랜덤서치파라미터로 변경해야 오류 안뜸(사이킷런머신러닝모델로 모델 변경)
end_time = time.time()

print('걸린시간 : ', round(end_time - start_time, 2))
print('model.best_params_', model.best_params_)
print('model.best_estimator_', model.best_estimator_)
print('model.best_score', model.best_score_)
print('model.score : ', model.score(x_test, y_test))

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print('acc_score : ', accuracy_score(y_test, y_predict))

# 걸린시간 :  9.8
# model.best_params_ {'optimizer': 'adam', 'node3': 128, 'node2': 64, 'node1': 32, 'drop': 0.4, 'batch_size': 300, 'activation': 'elu'}
# model.best_estimator_ <keras.wrappers.scikit_learn.KerasClassifier object at 0x000002704347EC10>
# model.best_score 0.9064500033855438
# 34/34 [==============================] - 0s 1ms/step - loss: 0.2618 - acc: 0.9216
# model.score :  0.9215999841690063
# 313/313 [==============================] - 0s 495us/step
# acc_score :  0.9216