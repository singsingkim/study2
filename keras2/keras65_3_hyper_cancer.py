import numpy as np
from sklearn.datasets import load_breast_cancer
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

#1. Data
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)

#2. Model
def build_model(drop=0.5, optimizer='adam', activation='relu', node1=128, node2=64, node3=32, callbacks=None):
    inputs = Input(shape=(30,))  # Adjusted input shape for breast cancer dataset
    x = Dense(node1, activation=activation)(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation)(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation)(x)
    x = Dropout(drop)(x)
    outputs = Dense(1, activation='sigmoid')(x)  # Binary classification output
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    if callbacks:
        model.fit(x_train, y_train, callbacks=callbacks)
    return model

# Hyperparameters
def create_hyperparameter():
    batchs = [100, 200, 300, 400, 500]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    node1 = [128, 64, 32, 16]
    node2 = [128, 64, 32, 16]
    node3 = [128, 64, 32, 16]
    return {'batch_size': batchs,
            'optimizer': optimizers,
            'drop': dropouts,
            'activation': activations,
            'node1': node1,
            'node2': node2,
            'node3': node3}

hyperparameters = create_hyperparameter()

keras_model = KerasClassifier(build_fn=build_model, verbose=1)

early_stopping = EarlyStopping(monitor='loss', mode='min', patience=40, restore_best_weights=True)
mcp = ModelCheckpoint(filepath='C:/_data/_save/keras65_save_model3.h5', monitor='loss', mode='min', save_best_only=True)

model = RandomizedSearchCV(estimator=keras_model, param_distributions=hyperparameters, cv=3, n_iter=10, verbose=1, random_state=42)

fit_params = {'callbacks': [early_stopping, mcp], 'batch_size' : 32}
import time
start_time = time.time()
model.fit(x_train, y_train, **fit_params, epochs = 100)
end_time = time.time()
print("걸린시간 : ", round(end_time - start_time,2))
print('model.best_params_ :', model.best_params_ )
print('model.best_estimator_ ', model.best_estimator_)
print('model.best_score_ :', model.best_score_)
print('model.score : ', model.score(x_test, y_test))
'''
걸린시간 :  62.98
model.best_params_ : {'optimizer': 'rmsprop', 'node3': 16, 'node2': 64, 'node1': 16, 'drop': 0.3, 'batch_size': 500, 'activation': 'elu'}     
model.best_estimator_  <keras.wrappers.scikit_learn.KerasClassifier object at 0x0000016F45E0FD00>
model.best_score_ : 0.9120528697967529
model.score :  0.9532163739204407
'''