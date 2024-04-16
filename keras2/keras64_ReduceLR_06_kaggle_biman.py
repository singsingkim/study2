import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, LSTM, Conv1D

# get data
path = "C:/_data/kaggle/obesity/"
train_csv = pd.read_csv(path + "train.csv")
test_csv = pd.read_csv(path + "test.csv")

# train_csv = colume_preprocessing(train_csv)

train_csv['BMI'] =  train_csv['Weight'] / (train_csv['Height'] ** 2)
test_csv['BMI'] =  test_csv['Weight'] / (test_csv['Height'] ** 2)

lbe = LabelEncoder()
cat_features = train_csv.select_dtypes(include='object').columns.values
for feature in cat_features :
    train_csv[feature] = lbe.fit_transform(train_csv[feature])
    if feature == "CALC" and "Always" not in lbe.classes_ :
        lbe.classes_ = np.append(lbe.classes_, "Always")
    if feature == "NObeyesdad":
        continue
    test_csv[feature] = lbe.transform(test_csv[feature]) 
                
x, y = train_csv.drop(["NObeyesdad"], axis=1), train_csv.NObeyesdad

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=777, train_size=0.8,stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = Sequential()
model.add(Dense(16, input_shape = (18,)))
model.add(Dense(32,activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(7, activation='softmax'))

from keras.optimizers import Adam
learning_rates = [1.0, 0.1, 0.01, 0.001, 0.0001]
for learning_rate in learning_rates : 

    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])
    history = model.fit(x_train, y_train, epochs=100, batch_size=1000, verbose= 0, validation_split=0.2)
    model.fit(x_train, y_train)
    loss = model.evaluate(x_test, y_test)
    pred = model.predict(x_test)
    acc = accuracy_score(pred, y_test)
    print("lr : {0}, 로스 : {1}".format(learning_rate, loss))
    print("lr : {0}, ACC : {1}".format(learning_rate, acc))
'''
XGBClassifier ACC : 0.8916
RandomForestClassifier ACC : 0.8943
LogisticRegression ACC : 0.8324
스태킹 결과 : 0.8960
'''