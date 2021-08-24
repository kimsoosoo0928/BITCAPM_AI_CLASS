# 회귀 데이터를 classifier로 만들었을 경우 에러 확인 !!!

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
 

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.7,test_size=0.3, shuffle=True, random_state=66) 

from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
# scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

#2. 모델 구성
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = LogisticRegression()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()

#3. 컴파일 및 훈련 + EarlyStopping
model.fit(x_train, y_train)

# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy']) # 이진 분류에 사용되는 binary_crossentropy, metrics는 결과에 반영은 안되고 보여주기만 한다.

# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)

# hist = model.fit(x_train, y_train, epochs=1000, batch_size=8, 
#             validation_split=0.2 ,callbacks=[es]) # es 적용

print("======================평가예측======================")
#4. 평가 및 예측

results = model.score(x_test, y_test) # acc 출력
print(results)

