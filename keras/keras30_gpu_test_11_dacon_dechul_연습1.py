from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import RobustScaler,StandardScaler
import datetime
import time

path = "c://_data//dacon//dechul//"

train_csv=pd.read_csv(path+"train.csv",index_col=0)

test_csv=pd.read_csv(path+"test.csv",index_col=0)

sub_csv=pd.read_csv(path+"sample_submission.csv")

# train_csv=train_csv[train_csv['근로기간'] != 'Unknown']

# unique,count = np.unique(train_csv['근로기간'], return_counts=True)

train_csv.iloc[28730,3]='OWN'  
test_csv.iloc[34486,7]='기타'


# df=pd.DataFrame(train_csv)
# df1=pd.DataFrame(test_csv)

le=LabelEncoder()


# train_csv['대출기간']=train_le.fit_transform(train_csv['대출기간'])
# test_csv['대출기간']=test_le.fit_transform(test_csv['대출기간'])
# train_csv['대출기간'] = train_csv['대출기간'].str.split().str[0].astype(int)
# test_csv['대출기간'] = test_csv['대출기간'].str.split().str[0].astype(int)
le.fit(train_csv['주택소유상태'])
le.fit(test_csv['주택소유상태'])
train_csv['주택소유상태']=le.transform(train_csv['주택소유상태'])
test_csv['주택소유상태']=le.transform(test_csv['주택소유상태'])

le.fit(train_csv['대출목적'])
le.fit(test_csv['대출목적'])
train_csv['대출목적']=le.transform(train_csv['대출목적'])
test_csv['대출목적']=le.transform(test_csv['대출목적'])

le.fit(train_csv['대출기간'])
le.fit(test_csv['대출기간'])
train_csv['대출기간']=train_csv['대출기간'].replace({' 36 months':36,' 60 months':60}).astype(int)
test_csv['대출기간']=test_csv['대출기간'].replace({' 36 months':36,' 60 months':60}).astype(int)


le.fit(train_csv['근로기간'])
le.fit(test_csv['근로기간'])
train_csv['근로기간']=le.transform(train_csv['근로기간'])
test_csv['근로기간']=le.transform(test_csv['근로기간'])

le.fit(train_csv['대출등급'])

# train_csv['근로기간']=le.fit_transform(train_csv['근로기간'])
# train_csv['주택소유상태']=le.fit_transform(train_csv['주택소유상태'])
# train_csv['대출목적']=le.fit_transform(train_csv['대출목적'])

# test_csv['근로기간']=le.fit_transform(test_csv['근로기간'])
# test_csv['주택소유상태']=le.fit_transform(test_csv['주택소유상태'])
# test_csv['대출목적']=le.fit_transform(test_csv['대출목적'])


# train_csv['대출등급']=le.fit_transform(train_csv['대출등급'])


# x=train_csv.drop('대출등급',axis=1)
# y=train_csv['대출등급']

x=train_csv.drop(['대출등급'],axis=1)
y=train_csv['대출등급']

y=y.values.reshape(-1,1)
ohe=OneHotEncoder(sparse=False)
ohe.fit(y)
y=ohe.transform(y)
# yo = to_categorical(y)

# train_csv.dropna

# print(np.unique(y)) #['A' 'B' 'C' 'D' 'E' 'F' 'G']
# print(pd.value_counts(y))
# B    28817
# C    27623
# A    16772
# D    13354
# E     7354
# F     1954
# G      420

# print(train_csv.head(8))

# print(x.shape,y.shape) #(96294, 13) (96294, 7)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,
                                               random_state=1967,stratify=y)

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)


model=Sequential()
model.add(Dense(131,input_shape=(13,),activation='relu'))
model.add(Dense(73,activation='relu'))
model.add(Dense(131,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(131,activation='relu'))
model.add(Dense(131,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(131,activation='relu'))
model.add(Dense(131,activation='relu'))
model.add(Dense(73,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(131,activation='relu'))
model.add(Dense(131,activation='relu'))
model.add(Dense(7,activation='softmax'))

# 3
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
es=EarlyStopping(monitor='val_acc',mode='auto',patience=200,verbose=1,
                 restore_best_weights=True)
mcp=ModelCheckpoint(monitor='val_acc',mode='auto',verbose=1,save_best_only=True,
                    filepath=('c:/_data/_save/MCP/_test_MCP.hdf5'))
start_time=time.time()
hist=model.fit(x_train,y_train,epochs=10000,batch_size=100,validation_split=0.1,callbacks=[es,mcp])
end_time=time.time()

# 4
result=model.evaluate(x_test,y_test)
print("loss",result[0])
print("acc",result[1])

y_predict=model.predict(x_test)
y_test=np.argmax(y_test,axis=1)
y_predict=np.argmax(y_predict,axis=1)

# y_test=le.inverse_transform(y_test)
# y_predict=le.inverse_transform(y_predict)

# y_submit=model.predict(test_csv)
y_submit=model.predict(test_csv) 
y_submit=np.argmax(y_submit,axis=1)
y_submit=le.inverse_transform(y_submit)


f1score=f1_score(y_test,y_predict,average='macro')
print("f1_score:",f1score)
# # y_submit=np.argmax(model.predict(test_csv),axis=1)

sub_csv['대출등급']=y_submit
sub_csv.to_csv(path+"dechul_sub_0118_4.csv",index=False)
print("걸린 시간 : ", round(end_time - start_time,2),"초")

# scaler = MinMaxScaler()   /   f1_score: 0.3077110807495997

# scaler = StandardScaler() /   f1_score: 0.31699199191443983

# scaler = MaxAbsScaler()   /   f1_score: 0.3183024740291938

# scaler = RobustScaler()   /   f1_score: 0.3159634653253046

# loss 0.47353947162628174
# acc 0.8445482850074768
# f1_score: 0.7970325748260676


# print("f1:",f1score)
# print("loss:",result[0])
# print("r2:",result[1])
# plt.figure(figsize=(1,1))
# plt.plot(hist.history['loss'],c='orange',label='loss',marker='.')
# plt.plot(hist.history['val_loss'],c='red',label='val_loss',marker='.')
# plt.plot(hist.history['acc'],c='pink',label='acc',marker='.')
# plt.legend(loc='upper right')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid()
# plt.show()

# loss 0.4111034572124481
# acc 0.862438976764679
# f1_score: 0.8328453208376894

