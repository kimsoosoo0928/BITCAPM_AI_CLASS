# 실습
# 아웃라이어 확인


import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_wine
from xgboost import XGBClassifier

#DATA
datasets = pd.read_csv('./_data/winequality-white.csv',
                        index_col=None, header=0, sep=';')

print(datasets.head())
print(datasets.shape) # (4898, 12)
print(datasets.describe())

# datasets = datasets.values
# print(type(datasets)) # <class 'numpy.ndarray'>

# x = datasets[:, :11]
# y = datasets[:, 11]



# from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
# from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)

import matplotlib.pyplot as plt

count_data = datasets.groupby('quality')['quality'].count()
print(count_data)

plt.bar(count_data.index, count_data)
plt.show()

# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# #MODEL
# model = XGBClassifier(n_jobs=-1)

# model.fit(x_train, y_train)

# score = model.score(x_test, y_test)

# print("acc : ", score) # acc :  0.6816326530612244