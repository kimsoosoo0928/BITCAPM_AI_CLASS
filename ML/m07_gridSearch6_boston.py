import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression # 분류모델 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
import warnings
import time
warnings.filterwarnings('ignore')

datasets = load_boston()


x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.7,test_size=0.3, shuffle=True, random_state=8)

n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle=True, random_state=66)
parameters = [
    {'n_estimators':[100, 200], 'max_depth':[6, 8, 10, 12], 'min_samples_leaf':[3, 5, 7, 10]},
    {'max_depth':[6, 8, 10, 12]},
    {'min_samples_leaf':[3, 5, 7, 10], 'min_samples_split':[2, 3, 5, 10]},
    {'min_samples_split':[2, 3, 5, 10]},
    {'n_jobs':[-1, 2, 4], 'max_depth':[6, 8, 10, 12]}
]

#2. 모델 구성
model = GridSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1)

#2. 모델 구성
# model = GridSearchCV(SVC(), parameter, cv=kfold)
#3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
print('Best estimator : ', model.best_estimator_)
print('Best score  :', model.best_score_)

'''
Best estimator :  RandomForestRegressor(max_depth=8)
Best score  : 0.8451992879926076
'''