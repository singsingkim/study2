import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

# 1. Data
path = "c://_data//kaggle//bike//"
train_csv=pd.read_csv(path+"train.csv",index_col=0)
test_csv=pd.read_csv(path+"test.csv",index_col=0)
submission_csv=pd.read_csv(path+"sampleSubmission.csv")

train_csv=train_csv.dropna()
test_csv=test_csv.fillna(test_csv.mean())

x=train_csv.drop(['count','casual','registered'],axis=1)
y=train_csv['count']

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
    model = RandomForestRegressor(random_state=123)

    # 훈련
    model.fit(x_train, y_train)

    # 평가, 예측
    accuracy = model.score(x_test, y_test)
    print(f"Number of Components: {n}, Accuracy: {accuracy}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_n_components = n

print(f"\nBest Number of Components: {best_n_components}, Best Accuracy: {best_accuracy}")


# Number of Components: 8, Accuracy: 0.27577108985070087
# Number of Components: 7, Accuracy: 0.2739088631449439
# Number of Components: 6, Accuracy: 0.246764563629029
# Number of Components: 5, Accuracy: 0.2520962330766885
# Number of Components: 4, Accuracy: 0.25310336997588456
# Number of Components: 3, Accuracy: 0.24150387731825707
# Number of Components: 2, Accuracy: 0.1674907131474147

# Best Number of Components: 8, Best Accuracy: 0.27577108985070087
# PS C:\Study> 


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

