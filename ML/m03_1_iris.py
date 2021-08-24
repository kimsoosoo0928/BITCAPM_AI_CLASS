import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

'''
- Iris-Setosa : 0
- Iris-Versicolour : 1
- Iris-Virginica : 2
'''

x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (150, 4) (150,)
print(y)


# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y) # 원핫인코딩끝!
# print(y[:5])
# print(y.shape) # (150,3)


#1. 데이터
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.7,test_size=0.3, shuffle=True, random_state=8)


#1-1. 데이터 전처리
from sklearn.preprocessing import MinMaxScaler, StandardScaler,QuantileTransformer
scaler = QuantileTransformer() 
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

#2. 모델 구성
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression # 분류모델 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier














# model = LinearSVC()
# accuracy_score :  0.8888888888888888
# model = SVC()
# accuracy_score :  0.9333333333333333
# model = KNeighborsClassifier()
# accuracy_score :  0.9111111111111111
model = KNeighborsRegressor()

# model = LogisticRegression()
# accuracy_score :  0.8222222222222222
# model = DecisionTreeClassifier()
# accuracy_score :  0.9111111111111111
# model = RandomForestClassifier()
# accuracy_score :  0.9111111111111111

# model = Sequential()
# model.add(Dense(128, activation='relu', input_shape=(4,)))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(3, activation='softmax')) # 다중분류위해서 softmax 사용, 원핫인코딩의 결과가 (150,3) 이기때문에 3이어야한다.


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

