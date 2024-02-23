
import pandas as pd
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import time


np_path='c:/_data/_save_npy//'
x=np.load(np_path+'keras39_3_x_train.npy')
y=np.load(np_path+'keras39_3_y_train.npy')
test=np.load(np_path+'keras39_3_test.npy')


print(x.shape,y.shape)  #(3000, 100, 100, 3) (3000,)
print(test.shape)   #(3000, 100, 100, 3)


x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=123,train_size=0.8,
                                                 shuffle=True,
                                                 stratify=y)

print(x_train.shape)    #(2400, 100, 100, 3)

model=Sequential()
model.add(Conv2D(12,(2,2),input_shape=(100,100,3),
                 strides=2,padding='valid'))
model.add(MaxPooling2D())
model.add(Conv2D(4,(2,2),activation='relu'))
model.add(Flatten())
model.add(Dense(4))
model.add(Dropout(0.25))
model.add(Dense(3))
model.add(Dense(1,activation='sigmoid'))


filepath = 'c:/_data/_save/MCP/cat_and_dog//'

es=EarlyStopping(monitor='val_loss',mode='auto',patience=100,
                 restore_best_weights=True)
mcp=ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,save_best_only=True,
                    filepath=filepath)

# 3 컴파일 훈련
model.compile(loss='binary_crossentropy',optimizer='adam',
              metrics=['acc'])
hist=model.fit(x_train,y_train,epochs=10,batch_size=20,
          verbose=1,validation_split=0.1,callbacks=[es,mcp])

result=model.evaluate(x_test,y_test)
y_predict=model.predict(test)
y_predict=np.round(y_predict.flatten()) 
# y_predict=np.around(y_predict.reshape(-1))
print(y_predict)

print("loss:",result)

import os
filename=os.listdir('c:/_data/image/cat_and_dog/test/test//')
print(filename)
filename[0]=filename[0].replace(".jpg","")  # 사진파일 이름 jpg 없애서 숫자만 제목으로 남게 한다. / 0번째꺼 일단 적용해보려

len(filename)
print(len(filename))    #파일갯수 확인

for i in range(len(filename)):
    filename[i] = filename[i].replace(".jpg","")
    
print(len(filename),len(y_predict)) #둘 갯수 같나 확인

# sub_df = pd.DataFrame({"Id":filename, "Target":y_predict})
# sub_df.to_csv("c:/_data/kaggle/"+"subfile_0124.csv",index=False)







# 