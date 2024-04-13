# cnn 으로 만든다.
# early_stopping 적용
# mcp 적용(모델체크포이트)

#[실습] 맹그러

from sklearn.datasets import load_breast_cancer
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint

# 1 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
#  1 0 0 1 0 0 0 0 1 0 0 0 1 0 1 0 1 1 0 1 
print(x.shape)  # (569, 30)
print(y.shape)  # (569,)

x = x.reshape(-1, 5, 3, 2)
print(x.shape)  # (569, 5, 3, 2)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123, shuffle=True,
    stratify=y
)

print(x_train.shape, x_test.shape)  # (455, 5, 3, 2) (114, 5, 3, 2)
print(y_train.shape, y_test.shape)  # (455,) (114,)

# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.min(x_train))  # 
# print(np.min(x_test))   # 
# print(np.max(x_train))  # 
# print(np.max(x_test))   # 



# 2 모델
def build_model(drop=0.5, optimizer='adam', activation='relu',
                node1=128, node2=64, node3=32, lr=0.001) : # 인풋값

    inputs = Input(shape=(5,3,2), name='inputs')
    x = Conv2D(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Conv2D(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Conv2D(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Flatten(x)
    outputs = Dense(2, activation='sigmoid', name='outputs')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=optimizer, metrics=['acc'],
                  loss = 'binary_crossentropy')
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

