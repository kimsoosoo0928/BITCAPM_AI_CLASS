# coefficient 계수

#1.data
x = [-3, 31, -11, 4, 0, 22, -2, -5, -25, -14]
y = [-3, 65, -19, 11, 3, 47, -1, -7, -47, -25]

import pandas as pd
df = pd.DataFrame({'X': x, 'Y': y})
print(df)
print(df.shape) # (10, 2)

x_train = df.loc[:, 'X']
y_train = df.loc[:, 'Y']
print(x_train.shape, y_train.shape) # (10,) (10,)
x_train = x_train.values.reshape(len(x_train), 1) #(10, ) -> (10, 1)
print(x_train.shape, y_train.shape) # (10, 1) (10,)
# *values : 넘파이로 바꾼다.

#2.model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

#3.fit
model.fit(x_train, y_train)

#4.eval, pred
score = model.score(x_train, y_train)
print('score : ', score)
print('기울기 : ', model.coef_)
print('절편 : ', model.intercept_)

# score :  1.0
# 기울기 :  [2.]
# 절편 :  3.0
