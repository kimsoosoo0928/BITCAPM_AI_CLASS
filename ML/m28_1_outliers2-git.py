# 실습 : 다차원의 outlier가 출력되도록 함수 수정

import numpy as np

A = np.array(   [[1,  2,10000,3,4,  6,   7,  8, 90,  100,  5000],
                [1000,2000,3,4000,5000,6000,7000,8,9000,10000, 1001]])



A = A.transpose()
print(A.shape) # (10, 2)

#################################재사용##################################
def outliers(data_out):
    quantile_1, q2, quantile_3 = np.percentile(data_out, [25, 50, 75])
    print("1사분위 : ", quantile_1)
    print("q2 : ", q2)
    print("3사분위 : ", quantile_3)
    iqr = quantile_3 - quantile_3
    lower_bound = quantile_1 - (iqr * 1.5)
    upper_bound = quantile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) | (data_out>lower_bound))
outliers_loc = outliers(A)
print("이상치의 위치 : ", outliers_loc)
########################################################################

import matplotlib.pyplot as plt

plt.boxplot(A)
plt.show()