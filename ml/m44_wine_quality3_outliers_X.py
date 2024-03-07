# [실습]
# 1. 아웃라이어 확인
# 2. 아웃라이어 처리
# 3. 44_1 이든 44_2 이든 수정해서 맹그러


import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])  # 퍼센트 지점
    print('1사분위 : ', quartile_1)
    print('q2 : ', q2)
    print('3사분위 : ', quartile_3)
    iqr = quartile_3 - quartile_1   # 이상치 찾는 인스턴스 정의
    # 최대값이 이상치라면 최대값최소값으로 구하는 이상치는 이상치를 구한다고 할수없다
    print('iqr : ', iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    # -10 의 1.5 범위만큼 과 50의 1.5 범위만큼을 이상치로 생각을 하고 배제
    # 4~10 까지는 안전빵이라고 정의

    # 조건문(인덱스 반환) 
    return lower_bound, upper_bound, np.where((data_out > upper_bound) | (data_out < lower_bound))


# 1 데이터
path = 'c:/_data/dacon/wine/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(train_csv)

submission_csv=pd.read_csv(path + "sample_submission.csv")

train_csv['type']=train_csv['type'].map({'white':1,'red':0}).astype(int)
test_csv['type']=test_csv['type'].map({'white':1,'red':0}).astype(int)

x=train_csv.drop(['quality'],axis=1)
y=train_csv['quality']

label=LabelEncoder()
label.fit(y)
y=label.transform(y)


#1. 데이터
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=456, train_size=0.8,
    # stratify=y
)

# 데이터에 대한 이상치를 찾습니다.
# outliers_indices = outliers(x_train)
lower_bound, upper_bound, outliers_indices = outliers(x_train)
print("이상치 인덱스:", outliers_indices)


# 이상치 제거
# upper_bound = quartile_3 + (iqr * 1.5)
# lower_bound = quartile_1 - (iqr * 1.5)
x_train = x_train[(x_train < upper_bound) & (x_train > lower_bound)]
y_train = y_train[x_train.index]  # x_train의 인덱스에 맞게 y_train을 조정합니다.





# 2 모델
model = RandomForestClassifier()
# model.set_params(early_stopping_rounds=200, **xgb_params)


# 3 훈련

model.fit(x_train, y_train,
        #   eval_set=[(x_train,y_train), (x_test,y_test)],
        #   verbose=1,
        #   eval_metric='auc'
)

# 4 평가, 예측
results = model.score(x_test,y_test)
print('최종점수', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('add_score', acc)



# random_state = 456
# 최종점수 0.6945454545454546
# add_score 0.6945454545454546






# 박스 플롯 그리기
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# 한글 폰트 경로 설정
font_path = "C:/Windows/Fonts/malgun.ttf"  # 사용하고자 하는 한글 폰트 경로로 변경해주세요
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# 마이너스 부호 깨짐 방지 설정
plt.rcParams['axes.unicode_minus'] = False

# 그림의 크기 설정
plt.figure(figsize=(8, 8))
plt.boxplot(x_train)
plt.ylabel('LoL', rotation=0)
# plt.xlabel('123')
plt.title('박스플롯')
plt.show()











