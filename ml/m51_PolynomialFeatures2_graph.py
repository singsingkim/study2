import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

plt.rcParams['font.family'] = 'Malgun Gothic'

# 1 데이터 
np.random.seed(777)
x = 2 * np.random.rand(100, 1) - 1  # 100개를 0~1 까지 출력, -> 0~2 까지 출력 -> -1~1 까지 출력
y = 3 * x**2 + 2 * x + 1 + np.random.randn(100, 1)  # 100개의 0~1 노이즈 추가   # y = 3x^ + 2x + 1 + 노이즈
print(x)

pf = PolynomialFeatures(degree=2, include_bias=False)
x_pf = pf.fit_transform(x)
print(x_pf)

# 2  모델
# model = LinearRegression()
# model2 = LinearRegression()   # 그림으로 가독성을 보여주기 위함
model = RandomForestRegressor()
model2 = RandomForestRegressor()    # 그림으로 모양은 비슷하게 나오지만 성능향상은 있다. 여러 선택지를 고르는 실력을 키우기위해

# 3 훈련
model.fit(x, y)
model2.fit(x_pf, y)

# 그림 그리기
plt.scatter(x, y, color='blue', label='원데이터')
plt.xlabel('x')
plt.xlabel('y')
plt.title('Polynomial Regression Example')

# 다항식 회귀 그래프 그리기
x_plot = np.linspace(-1, 1, 100).reshape(-1,1)
x_plot_pf = pf.transform(x_plot)
y_plot = model.predict(x_plot)
y_plot2 = model2.predict(x_plot_pf)
plt.plot(x_plot, y_plot, color='red', label='Ploynomial Regression')
plt.plot(x_plot, y_plot2, color='green', label ='기냥')

plt.legend()
plt.show()
