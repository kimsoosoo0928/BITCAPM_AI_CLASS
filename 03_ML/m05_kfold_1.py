import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression # 분류모델 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

datasets = load_iris()


x = datasets.data
y = datasets.target

n_splits = 5

kfold = KFold(n_splits = n_splits, shuffle=True, random_state=66)

#2. 모델 구성

# model = LinearSVC()
# acc :  [0.96666667 0.96666667 1.         0.9        1.        ]
# avg acc :  0.9667
# model = SVC()
# acc :  [0.96666667 0.96666667 1.         0.93333333 0.96666667]
# avg acc :  0.9667
# model = KNeighborsClassifier()
# acc :  [0.96666667 0.96666667 1.         0.9        0.96666667]
# avg acc :  0.96
# model = LogisticRegression()
# acc :  [1.         0.96666667 1.         0.9        0.96666667]
# avg acc :  0.9667
# model = DecisionTreeClassifier()
# acc :  [0.96666667 0.96666667 1.         0.9        0.93333333]
# avg acc :  0.9533
model = RandomForestClassifier()
# acc :  [0.93333333 0.96666667 1.         0.9        0.96666667]
# avg acc :  0.9533

print("======================평가예측======================")
#3. 컴파일 및 훈련, 평가 및 예측

scores = cross_val_score(model, x, y, cv=kfold)

print("acc : ", scores)
print("avg acc : ", round(np.mean(scores),4)) 



