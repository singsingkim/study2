import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
# 1. Data
path = "c://_data//dacon//wine//"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv",index_col=0)
# print(test_csv)
submission_csv=pd.read_csv(path + "sample_submission.csv")

train_csv['type']=train_csv['type'].map({'white':1,'red':0}).astype(int)
test_csv['type']=test_csv['type'].map({'white':1,'red':0}).astype(int)

x=train_csv.drop(['quality'],axis=1)
y=train_csv['quality']

label=LabelEncoder()
label.fit(y)
y=label.transform(y)


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==========================================================================================
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
    # 4~10 까지는 안정빵이라고 정의
    
    # 조건문(인덱스 반환) 
    return np.where((data_out>upper_bound) |    # 19보다 크거나
                    (data_out>lower_bound))
# ==========================================================================================

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

best_accuracy = 0
best_n_components = 0

for n in range(x.shape[1], 1, -1):
    pca = PCA(n_components=n)
    x_pca = pca.fit_transform(x_scaled)

    x_train, x_test, y_train, y_test = train_test_split(
        x_pca, y, train_size=0.8, random_state=123, shuffle=True, # stratify=y
    )

    # 모델
    model = RandomForestClassifier(random_state=123)

    # 훈련
    model.fit(x_train, y_train)

    # 평가, 예측
    accuracy = model.score(x_test, y_test)
    print(f"Number of Components: {n}, Accuracy: {accuracy}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_n_components = n

print(f"\nBest Number of Components: {best_n_components}, Best Accuracy: {best_accuracy}")



evr = pca.explained_variance_ratio_     # 1.0 에 가까워야 좋다
print(evr)
print(sum(evr))

evr_cumsum = np.cumsum(evr)
print(evr_cumsum)


outliers_loc = outliers(x_pca)
print('이상치의 위치 : ', outliers_loc)

import matplotlib.pyplot as plt
plt.boxplot(x_pca)
plt.show()






'''
import matplotlib.pyplot as plt
plt.plot(evr_cumsum)
plt.grid()
plt.show()
'''

