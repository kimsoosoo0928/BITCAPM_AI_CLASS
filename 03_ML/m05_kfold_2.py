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

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=8)

n_splits = 5

kfold = KFold(n_splits = n_splits, shuffle=True, random_state=66)

#2. 모델 구성

# model = LinearSVC()
# acc :  [0.96666667 0.96666667 1.         0.9        1.        ]
# avg acc :  0.9667
# acc :  [1.         0.91666667 1.         0.95833333 0.91666667]
# avg acc :  0.9583
# model = SVC()
# acc :  [0.95833333 0.95833333 1.         0.91666667 1.        ]
# avg acc :  0.9667
# acc :  [0.96666667 0.96666667 1.         0.93333333 0.96666667]
# avg acc :  0.9667
# model = KNeighborsClassifier()
# acc :  [0.96666667 0.96666667 1.         0.9        0.96666667]
# avg acc :  0.96
# acc :  [1.         0.95833333 1.         0.95833333 0.95833333]
# avg acc :  0.975
# model = LogisticRegression()
# acc :  [1.         0.96666667 1.         0.9        0.96666667]
# avg acc :  0.9667
# acc :  [1.         0.95833333 1.         0.95833333 0.95833333]
# avg acc :  0.975
# model = DecisionTreeClassifier()
# acc :  [0.96666667 0.96666667 1.         0.9        0.93333333]
# avg acc :  0.9533
# acc :  [1.         0.95833333 1.         0.95833333 0.95833333]
# avg acc :  0.975
# model = RandomForestClassifier()
# acc :  [0.93333333 0.96666667 1.         0.9        0.96666667]
# avg acc :  0.95333
# acc :  [1.         0.95833333 1.         0.95833333 0.95833333]
# avg acc :  0.975
print("======================평가예측======================")
#3. 컴파일 및 훈련, 평가 및 예측

scores = cross_val_score(model, x_train, y_train, cv=kfold)

print("acc : ", scores)
print("avg acc : ", round(np.mean(scores),4)) 



