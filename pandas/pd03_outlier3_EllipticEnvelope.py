import numpy as np
aaa = np.array([-10, 2, 3, 4, 5, 6, 700, 8, 9, 10, 11, 12, 50])
print(aaa.shape)        # (13,)

aaa = aaa.reshape(-1,1) # (13, 1)

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.3)

outliers.fit(aaa)
results = outliers.predict(aaa)
print(results)  # [-1  1  1  1  1  1  1  1  1  1  1  1 -1]