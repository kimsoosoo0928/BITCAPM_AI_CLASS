from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression # 분류모델 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') # warning 무시 
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

allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')
print('모델의 갯수 : ', len(allAlgorithms))


for (name, algorithm) in allAlgorithms:
    try :
        model = algorithm()
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률 : ', acc)
    except :
        # continue
        print(name, '은 없는 모델')


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
AdaBoostClassifier 의 정답률 :  0.9532163742690059
BaggingClassifier 의 정답률 :  0.9590643274853801
BernoulliNB 의 정답률 :  0.9122807017543859
CalibratedClassifierCV 의 정답률 :  0.9649122807017544
CategoricalNB 은 없는 모델
ClassifierChain 은 없는 모델
ComplementNB 은 없는 모델
DecisionTreeClassifier 의 정답률 :  0.935672514619883
DummyClassifier 의 정답률 :  0.6140350877192983
ExtraTreeClassifier 의 정답률 :  0.8771929824561403
ExtraTreesClassifier 의 정답률 :  0.9707602339181286
GaussianNB 의 정답률 :  0.9298245614035088
GaussianProcessClassifier 의 정답률 :  0.9473684210526315
GradientBoostingClassifier 의 정답률 :  0.9824561403508771
HistGradientBoostingClassifier 의 정답률 :  0.9649122807017544
KNeighborsClassifier 의 정답률 :  0.9590643274853801
LabelPropagation 의 정답률 :  0.9181286549707602
LabelSpreading 의 정답률 :  0.9181286549707602
LinearDiscriminantAnalysis 의 정답률 :  0.9532163742690059
LinearSVC 의 정답률 :  0.9707602339181286
LogisticRegression 의 정답률 :  0.9766081871345029
MLPClassifier 의 정답률 :  0.9766081871345029
MultiOutputClassifier 은 없는 모델
MultinomialNB 은 없는 모델
NearestCentroid 의 정답률 :  0.9239766081871345
NuSVC 의 정답률 :  0.9415204678362573
OneVsOneClassifier 은 없는 모델
OneVsRestClassifier 은 없는 모델
OutputCodeClassifier 은 없는 모델
PassiveAggressiveClassifier 의 정답률 :  0.9532163742690059
Perceptron 의 정답률 :  0.9473684210526315
QuadraticDiscriminantAnalysis 의 정답률 :  0.9532163742690059
RadiusNeighborsClassifier 은 없는 모델
RandomForestClassifier 의 정답률 :  0.9707602339181286
RidgeClassifier 의 정답률 :  0.9590643274853801
RidgeClassifierCV 의 정답률 :  0.9590643274853801
SGDClassifier 의 정답률 :  0.9649122807017544
SVC 의 정답률 :  0.9766081871345029
StackingClassifier 은 없는 모델
VotingClassifier 은 없는 모델
======================평가예측======================
0.9766081871345029
accuracy_score :  0.9766081871345029
'''