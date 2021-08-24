# 이상치 처리
#1. 삭제
#2. Nan처리후 -> 보간 / linear
#3. 결측치 처리 방법과 유사
#4. scaler -> Rubsterscalert, QuantileTransformer ... 등등등 ...
#5. 모델링 : tree 계열... DT, RF, XG, LGBM

import numpy as np
from numpy.lib.function_base import quantile
A = np.array([1,2,-1000,3,4,5,6,7,8,90,100,500])

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


# 시각화
# 실습
# 위 데이터를 bocplot으로 그리시오!
import matplotlib.pyplot as plt

plt.boxplot(A)
plt.show()


