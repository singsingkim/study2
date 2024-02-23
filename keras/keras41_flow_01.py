import sys
import tensorflow as tf
print('텐서플로: ', tf.__version__) # 텐서플로:  2.9.0
print('파이썬버전: ', sys.version)  # 파이썬버전:  3.9.18



from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img # 이미지 가져옴
from tensorflow.keras.preprocessing.image import img_to_array # 이미지 수치화,/주로 한장짜리 수치화
import matplotlib.pyplot as plt
import numpy as np


path = ('c:/_data/image//cat_and_dog//train/Cat//1.jpg')
img = load_img(
    path,
    target_size = (150, 150)
)
print(img)
# <PIL.Image.Image image mode=RGB size=150x150 at 0x25E1CECFFA0>
print(type(img))    # <class 'PIL.Image.Image'>
# plt.imshow(img)
# plt.show()

arr = img_to_array(img)
print(arr)
print(arr.shape)    # (281, 300, 3) -> (150, 150, 3)
print(type(arr))    # <class 'numpy.ndarray'>

# 차원증가
img = np.expand_dims(arr, axis=0)   # 익스펜디드 딤 : 차원을 늘려라
print(img.shape)    # (1, 150, 150, 3)      # 액시스 1 일때 (150, 1, 150, 3)

############################# 여기부터 증폭 ############################
datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.5,
    zoom_range=0.5,
    shear_range=30,    # 정사각형 이미지를 찌그러트린다
    fill_mode='nearest'
)

it = datagen.flow(img,              # img 는 현재 넘파이 형태 
                  batch_size=1) 
print(it)
fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(10,10))

for row in range(3):
    for col in range(5):
        batch = it.next() # 이미지 변환을 한번 X 곱하기 5 번
        print(batch)
        print(batch.shape)  # (1, 150, 150, 3)    
        image=batch[0].astype('uint8')  # 위에서 리스케일 안해주어서 사용    # uint8 은 픽셀 값이 0에서 255 사이의 값으로 제한 두는것 
        # print(image)  
        print(image.shape)  # (150, 150, 3)
        ax[row][col].imshow(image)
        ax[row][col].axis('off') # 액시스 off : xy 축 안보여줌 / 액시스 on 축보여줌

print(np.min(batch), np.max(batch)) # 0.0    232.0
plt.show()

