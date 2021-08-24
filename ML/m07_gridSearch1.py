import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression # 분류모델 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

datasets = load_iris()


x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.7,test_size=0.3, shuffle=True, random_state=8)

n_splits = 5

kfold = KFold(n_splits = n_splits, shuffle=True, random_state=66)
parameter = [
    {"C":[1, 10, 100, 1000], "kernel":["linear"]}, # 4x5 = 20번
    {"C":[1, 10, 100], "kernel":["rbf"], "gamma":[0.001, 0.0001]}, # 3x1x2x5 = 30번
    {"C":[1, 10, 100, 1000], "kernel":["sigmoid"], "gamma":[0.001, 0.0001]} # 4x1x2x5 = 40번
    # total : 90번
]

#2. 모델 구성
model = GridSearchCV(SVC(), parameter, cv=kfold)
# model = SVC()

#3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)
print("best_score_ : ", model.best_score_)

print('model.score : ', model.score(x_test, y_test))

y_pred = model.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, y_pred))