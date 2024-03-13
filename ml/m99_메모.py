# 함수의 최소값 찾기
import numpy as np
import hyperopt as hp
import pandas as pd
print(hp.__version__)   # 0.2.7     # 개발자가 판매할때 1.0.0 정의한다. 그 이하는 베타 버전으로 인식해도 무방하다

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK


search_space = {'x1' : hp.quniform('x1', -10, 10, 1),       # 범위  # 딕셔너리 형태
                'x2' : hp.quniform('x2', -15, 15, 1)}
                    # hp.quniform(label., low, high, q) # q : 분할 단위

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