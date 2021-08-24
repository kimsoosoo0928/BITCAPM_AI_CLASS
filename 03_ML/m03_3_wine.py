import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression # 분류모델 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

datasets = load_wine()
print(datasets.DESCR)
print(datasets.feature_names) 

x = datasets.data
y = datasets.target

print(x.shape, y.shape) 

# 완성하시오 !!!
# acc 0.8 이상 만들것 !!!

# 1-1 데이터
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.7,test_size=0.3, shuffle=True, random_state=8)

# 1-2 데이터 전처리
from sklearn.preprocessing import QuantileTransformer
scaler = QuantileTransformer() 
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

'''
모델의 갯수 :  41
AdaBoostClassifier 의 정답률 :  0.9259259259259259
BaggingClassifier 의 정답률 :  0.9629629629629629
BernoulliNB 의 정답률 :  0.3333333333333333
CalibratedClassifierCV 의 정답률 :  0.9814814814814815
CategoricalNB 의 정답률 :  0.24074074074074073
ClassifierChain 은 없는 모델
ComplementNB 의 정답률 :  0.8148148148148148
DecisionTreeClassifier 의 정답률 :  0.8333333333333334
DummyClassifier 의 정답률 :  0.42592592592592593
ExtraTreeClassifier 의 정답률 :  0.8703703703703703
ExtraTreesClassifier 의 정답률 :  0.9814814814814815
GaussianNB 의 정답률 :  0.9629629629629629
GaussianProcessClassifier 의 정답률 :  0.9629629629629629
GradientBoostingClassifier 의 정답률 :  0.9444444444444444
HistGradientBoostingClassifier 의 정답률 :  0.9629629629629629
KNeighborsClassifier 의 정답률 :  0.9629629629629629
LabelPropagation 의 정답률 :  0.9444444444444444
LabelSpreading 의 정답률 :  0.9444444444444444
LinearDiscriminantAnalysis 의 정답률 :  0.9629629629629629
LinearSVC 의 정답률 :  0.9814814814814815
LogisticRegression 의 정답률 :  0.9814814814814815
LogisticRegressionCV 의 정답률 :  0.9629629629629629
MLPClassifier 의 정답률 :  0.9629629629629629
MultiOutputClassifier 은 없는 모델
MultinomialNB 의 정답률 :  0.9074074074074074
NearestCentroid 의 정답률 :  0.9814814814814815
NuSVC 의 정답률 :  0.9629629629629629
OneVsOneClassifier 은 없는 모델
OneVsRestClassifier 은 없는 모델
OutputCodeClassifier 은 없는 모델
PassiveAggressiveClassifier 의 정답률 :  0.9814814814814815
Perceptron 의 정답률 :  0.9629629629629629
QuadraticDiscriminantAnalysis 의 정답률 :  0.9629629629629629
RadiusNeighborsClassifier 의 정답률 :  0.9629629629629629
RandomForestClassifier 의 정답률 :  0.9814814814814815
RidgeClassifier 의 정답률 :  0.9814814814814815
RidgeClassifierCV 의 정답률 :  0.9629629629629629
SGDClassifier 의 정답률 :  0.9814814814814815
SVC 의 정답률 :  0.9629629629629629
StackingClassifier 은 없는 모델
VotingClassifier 은 없는 모델
'''