  
# 실습
#TODO 다차원의 outlier가 출력되도록 함수 수정

import numpy as np


aaa = np.array([[1, 2, 10000, 4, 5, 6, 7, 8, 90, 100, 5000],
                [1000, 2000, 3, 4000, 5000, 6000, 7000, 8, 9000, 10000, 1001]])
# (2, 10)   ->  (10, 2)

aaa = aaa.transpose()
print(aaa.shape)

def outliers(data_out):
    n_list = []
    for i in range(data_out.shape[1]):
        quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])
        print('1사분위 : ', quartile_1)
        print('q2 : ', q2)
        print('3사분위 : ', quartile_3)
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 - (iqr * 1.5)
        # print(iqr)
        # print(lower_bound)
        # print(upper_bound)
        
        m = np.where((data_out[:, i]>upper_bound) | (data_out[:, i]<lower_bound))
        n = np.count_nonzero((data_out[:, i]>upper_bound) | (data_out[:, i]<lower_bound))
        #! count_nonzero 요소 개수세기(정수 값 또는 정수 값 배열을 반환)
        n_list.append([i+1,'columns', m, 'outlier_num :', n])
    return np.array(n_list)


outliers_loc = outliers(aaa)

print('이상치의 위치 : ', outliers_loc)

