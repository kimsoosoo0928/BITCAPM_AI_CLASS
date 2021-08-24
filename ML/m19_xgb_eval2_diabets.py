from sklearn import datasets
from xgboost import XGBRegressor
from sklearn.datasets import load_boston, load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. data

datasets = load_diabetes()
x = datasets['data']
y = datasets['target']

print(x.shape, y.shape) # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8,test_size=0.2, shuffle=True, random_state=66)

scaler = MinMaxScaler() 
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

# 2. model 
model = XGBRegressor(n_estimators = 100, learning_rate=0.1, n_jobs=1)

# 3. fit
model.fit(x_train, y_train, verbose=1, eval_metric='rmse', # ['mae','logloss']
                eval_set=[(x_train, y_train), (x_test, y_test)]) # fit set & val set

# 4. eval
results = model.score(x_test, y_test)
print("result : ", results)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("r2 score : ", r2)

'''
result :  0.2043357307070307
r2 score :  0.2043357307070307
'''