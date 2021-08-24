from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression # 분류모델 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
datasets = load_breast_cancer()

print(datasets.DESCR)
print(datasets.feature_names) # (569, 30)

x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(569, 30) (569,) input=30 output=1
print(y[:20])
print(np.unique) # y라는 데이터는 0,1로만 이루어져있다datetime A combination of a date and a time. Attributes: ()

# 실습 : 모델 시작 !

#1. 데이터
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.7,test_size=0.3, shuffle=True, random_state=8)

#1-1. 데이터 전처리
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler() 
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



# loss = model.evaluate(x_test, y_test) # binary_crossentropy
# print('loss : ', loss[0])
# print('accuracy : ', loss[1])

from sklearn.metrics import r2_score, accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)

print("===============예측==================")
print(y_test[:5])
y_predict2 = model.predict(x_test[:5])
print(y_predict2)
