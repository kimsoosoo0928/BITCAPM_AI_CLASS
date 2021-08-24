import numpy as np

A = np.array(   [[1,  2,10000,3,4,  6,   7,  8, 90,  100,  5000],
                [1000,2000,3,4000,5000,6000,7000,8,9000,10000, 1001]])

A = A.transpose()
print(A.shape)

from sklearn.covariance import EllipticEnvelope

outliers = EllipticEnvelope(contamination=.2)
outliers.fit(A)

results = outliers.predict(A)

print(results)