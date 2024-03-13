# 함수의 최소값 찾기
import numpy as np
import hyperopt as hp
import pandas as pd
print(hp.__version__)   # 0.2.7     # 개발자가 판매할때 1.0.0 정의한다. 그 이하는 베타 버전으로 인식해도 무방하다

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK


search_space = {'x1' : hp.quniform('x1', -10, 10, 1),       # 범위
                'x2' : hp.quniform('x2', -15, 15, 1)}
                    # hp.quniform(label., low, high, q)

# hp.quniform(label., low, high, q) : label로 지정된 입력 값 변수 검색 공간을 최소값 low에서 최대값 high까지 q의 간격을 가지고 설정
# hp.quniform(label, low, high) : 최소값 low에서 최대값 highg 까지 정규분포 형태의 검색 공간 설정
# hp.randint(label, upper) : 0부터 최대값 upper 가지 random한 정수값으로 검색 공간 설정.
# hp.loguniform(label.low, high) : exp(uniform(low, high))값을 밙환하며, 반환값의 log 변환 된 값은 정규분호 형태를 가지는 검색 공간 설정

def objective_func(search_space):   # 로스로 넣어주면 최소의 로스를 찾아준다
    x1 = search_space['x1']
    x2 = search_space['x2']
    
    return_value = x1**2 -20*x2
    
    return return_value

trial_val = Trials()

best  = fmin(
    fn = objective_func,
    space = search_space,
    algo = tpe.suggest,     # 알고리즘, 디폴트
    max_evals = 20,
    trials=trial_val,
    rstate = np.random.default_rng(seed=10)
    # rstate = 123,
)

print(best)     # {'x1': 0.0, 'x2': 15.0}       # 알고리즘의 최소값을 구해줌

print(trial_val.results)        # 20번을 계산했음
# [{'loss': -216.0, 'status': 'ok'}, {'loss': -175.0, 'status': 'ok'}, {'loss': 129.0, 'status': 'ok'}, {'loss': 200.0, 'status': 'ok'}, 
#  {'loss': 240.0, 'status': 'ok'}, {'loss': -55.0, 'status': 'ok'}, {'loss': 209.0, 'status': 'ok'}, {'loss': -176.0, 'status': 'ok'}, 
#  {'loss': -11.0, 'status': 'ok'}, {'loss': -51.0, 'status': 'ok'}, {'loss': 136.0, 'status': 'ok'}, {'loss': -51.0, 'status': 'ok'}, 
#  {'loss': 164.0, 'status': 'ok'}, {'loss': 321.0, 'status': 'ok'}, {'loss': 49.0, 'status': 'ok'}, {'loss': -300.0, 'status': 'ok'}, 
#  {'loss': 160.0, 'status': 'ok'}, {'loss': -124.0, 'status': 'ok'}, {'loss': -11.0, 'status': 'ok'}, {'loss': 0.0, 'status': 'ok'}]

print(trial_val.vals)           # x1, x2 의 키값들 출력
# {'x1': [-2.0, -5.0, 7.0, 10.0, 10.0, 5.0, 7.0, -2.0, -7.0, 7.0, 4.0, -7.0, -8.0, 9.0, -7.0, 0.0, -0.0, 4.0, 3.0, -0.0], 
#  'x2': [11.0, 10.0, -4.0, -5.0, -7.0, 4.0, -8.0, 9.0, 3.0, 5.0, -6.0, 5.0, -5.0, -12.0, 0.0, 15.0, -8.0, 7.0, 1.0, 0.0]}

# [실습] 요렇게 이쁘게 나오게 맹그러
# 판다스 데이터프레임 사용
# |   iter    |  target   |    x1    |    x2     |
# -------------------------------------------------------------------------------------------------------------------------------------------------
# | 1         | 0.9474    | 0.8482    | 0.2869    |
# | 2         | 0.8772    | 0.6716    | 0.7293    |
# | 3         | 0.9561    | 0.8172    | 0.8496    |

target = [aaa['loss'] for aaa in trial_val.results]
print(target)
df = pd.DataFrame({
    'target' : target,
    'x1' : trial_val.vals['x1'],
    'x2' : trial_val.vals['x2'],
    })
print(df)

