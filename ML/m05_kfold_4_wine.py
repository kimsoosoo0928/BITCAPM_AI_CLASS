import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
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

datasets = load_wine()


x = datasets.data
y = datasets.target

n_splits = 5

kfold = KFold(n_splits = n_splits, shuffle=True, random_state=66)

#2. 모델 구성

# model = LinearSVC()
# acc :  [0.61111111 0.94444444 0.88888889 0.82857143 0.85714286]
# avg acc :  0.826
# model = SVC()
# acc :  [0.69444444 0.69444444 0.61111111 0.62857143 0.6       ]
# avg acc :  0.6457
# model = KNeighborsClassifier()
# acc :  [0.69444444 0.77777778 0.61111111 0.62857143 0.74285714]
# avg acc :  0.691
# model = LogisticRegression()
# acc :  [0.97222222 0.94444444 0.94444444 0.94285714 1.        ]
# avg acc :  0.9608
# model = DecisionTreeClassifier()
# acc :  [0.91666667 0.97222222 0.91666667 0.88571429 0.91428571]
# avg acc :  0.9211
model = RandomForestClassifier()
# acc :  [1.         0.97222222 1.         0.97142857 1.        ]
# avg acc :  0.9887

print("======================평가예측======================")
#3. 컴파일 및 훈련, 평가 및 예측

scores = cross_val_score(model, x, y, cv=kfold)

print("acc : ", scores)
print("avg acc : ", round(np.mean(scores),4)) 