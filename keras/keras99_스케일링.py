스케일링은 머신 러닝에서 주어진 특성(feature)들의 값을 조정하는 과정입니다. 이 과정은 특성 간의 크기 차이나 단위 차이로 인해 발생하는 문제를 해결하고, 모델의 성능을 향상시키는 데 도움이 됩니다. 스케일링을 적용할 때, 보통 train set과 test set을 따로 다룹니다.

Train set에서 fit과 transform 사용하는 이유:

먼저, fit 메서드는 데이터의 분포를 학습하여 스케일링에 필요한 파라미터를 계산합니다. 예를 들어, 평균과 표준편차를 계산할 수 있습니다.
그 다음, transform 메서드는 계산된 파라미터를 사용하여 데이터를 변환합니다. 이는 각 특성의 값을 조정하여 스케일링하는 과정입니다.
Train set에서는 fit과 transform 메서드를 함께 사용하여 데이터의 분포를 학습하고, 이에 맞게 스케일링합니다. 따라서 모델이 학습하는 데 사용되는 데이터의 특성들이 일관된 방식으로 변환됩니다.
Test set에서는 transform만 사용하는 이유:

Test set은 모델이 처음 보는 데이터이기 때문에, 학습할 때 사용한 것과 동일한 변환을 적용해야 합니다. 그러나 test set에서는 새로운 데이터에 대해 스케일링을 학습할 필요가 없습니다. 대신, train set에서 이미 학습한 파라미터를 사용하여 변환을 수행합니다.
따라서 test set에서는 fit 메서드를 사용하지 않고, transform 메서드만을 사용하여 train set에서 학습한 스케일링을 동일하게 적용합니다.
이렇게 함으로써, 모델이 새로운 데이터에 대해 일관된 방식으로 예측을 수행할 수 있게 됩니다. 또한, train set과 test set 간에 일관된 스케일링이 유지되므로 모델의 성능을 신뢰할 수 있습니다.



from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler


# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.min(x_train))  # 
# print(np.min(x_test))   # 
# print(np.max(x_train))  # 
# print(np.max(x_test))   # 

# scaler = MaxAbsScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.min(x_train))  # 
# print(np.min(x_test))   # 
# print(np.max(x_train))  # 
# print(np.max(x_test))   # 

# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.min(x_train))  # 
# print(np.min(x_test))   # 
# print(np.max(x_train))  # 
# print(np.max(x_test))   # 

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


