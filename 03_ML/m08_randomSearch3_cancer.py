import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression # 분류모델 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')

datasets = load_breast_cancer()


x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.7,test_size=0.3, shuffle=True, random_state=8)

n_splits = 5

kfold = KFold(n_splits = n_splits, shuffle=True, random_state=66)
parameter = [
    {'n_jobs' : [-1],'n_estimators' : [100,200],'max_depth' : [6, 8, 10], 'min_samples_lead' : [5, 7, 10]},
    {'n_jobs' : [-1],'max_depth' : [6, 8, 10, 12], 'min_samples_lead' : [3, 5, 7, 10], 'min_samples_split' : [2, 3, 5, 10]},
    {'n_jobs' : [-1],'min_samples_lead' : [3, 5, 7, 10], 'min_samples_split' : [2, 3, 5, 10]},
    {'n_jobs' : [-1],'min_samples_split' : [2, 3, 5, 10]},
    {'n_jobs' : [-1]}
]

#2. 모델 구성
# model = GridSearchCV(SVC(), parameter, cv=kfold, verbose=1)
model = RandomizedSearchCV(SVC(), parameter, cv=kfold, verbose=1)
# model = SVC()

#3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)
print("best_params_ : ", model.best_params_)
print("best_score_ : ", model.best_score_)

print('model.score : ', model.score(x_test, y_test))

y_pred = model.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, y_pred))

'''
최적의 매개변수 :  SVC(C=1000, kernel='linear')
best_params_ :  {'kernel': 'linear', 'C': 1000}
best_score_ :  0.954778481012658
model.score :  0.9707602339181286
accuracy_score :  0.9707602339181286
'''