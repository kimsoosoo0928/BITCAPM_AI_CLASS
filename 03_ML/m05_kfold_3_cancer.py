import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer 
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

datasets = load_breast_cancer()


x = datasets.data
y = datasets.target

n_splits = 5

kfold = KFold(n_splits = n_splits, shuffle=True, random_state=66)

#2. 모델 구성

# model = LinearSVC()
# acc :  [0.87719298 0.92105263 0.89473684 0.87719298 0.94690265]
# avg acc :  0.9034
# model = SVC()
# acc :  [0.89473684 0.92982456 0.89473684 0.92105263 0.96460177]
# avg acc :  0.921
# model = KNeighborsClassifier()
# acc :  [0.92105263 0.92105263 0.92105263 0.92105263 0.95575221]
# avg acc :  0.928
# model = LogisticRegression()
# acc :  [0.94736842 0.95614035 0.88596491 0.95614035 0.96460177]
# avg acc :  0.942
# model = DecisionTreeClassifier()
# acc :  [0.95614035 0.92982456 0.93859649 0.89473684 0.9380531 ]
# avg acc :  0.9315
# model = RandomForestClassifier()
# acc :  [0.97368421 0.96491228 0.96491228 0.95614035 0.98230088]
# avg acc :  0.9684

print("======================평가예측======================")
#3. 컴파일 및 훈련, 평가 및 예측

scores = cross_val_score(model, x, y, cv=kfold)

print("acc : ", scores)
print("avg acc : ", round(np.mean(scores),4)) 



