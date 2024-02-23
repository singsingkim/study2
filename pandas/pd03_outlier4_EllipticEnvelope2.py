import numpy as np
aaa = np.array([[-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50],
                       #   25%      50%       75%
    # [100, 200, -30, 400, 500, 600, -70000, 800, 900, 1000, 210, 420, 350]]
    [100, 200, -30, 400, 500, 600, 110, 800, 900, 1000, 210, 420, 350]]
    ).T #(13, 2)


from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.1)

outliers.fit(aaa)
results = outliers.predict(aaa)
print(results)  # [-1  1  1  1  1  1  1  1  1  1  1  1 -1] 
                # 두개를 하나로 잡아서 인식
                # 컬럼별로 인식하는것도 방법
                
                

