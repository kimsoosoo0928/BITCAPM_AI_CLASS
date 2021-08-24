import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') # warning 무시 

datasets = load_iris()


x = datasets.data
y = datasets.target



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
41
AdaBoostClassifier 의 정답률 :  0.8888888888888888
BaggingClassifier 의 정답률 :  0.8888888888888888
BernoulliNB 의 정답률 :  0.35555555555555557
CalibratedClassifierCV 의 정답률 :  0.8444444444444444
CategoricalNB 의 정답률 :  0.3333333333333333
ClassifierChain 은 없는 모델
ComplementNB 의 정답률 :  0.6
DecisionTreeClassifier 의 정답률 :  0.9111111111111111
DummyClassifier 의 정답률 :  0.3111111111111111
ExtraTreeClassifier 의 정답률 :  0.8888888888888888
ExtraTreesClassifier 의 정답률 :  0.9111111111111111
GaussianNB 의 정답률 :  0.8888888888888888
GaussianProcessClassifier 의 정답률 :  0.8
GradientBoostingClassifier 의 정답률 :  0.8888888888888888
HistGradientBoostingClassifier 의 정답률 :  0.9111111111111111
KNeighborsClassifier 의 정답률 :  0.9111111111111111
LabelPropagation 의 정답률 :  0.9333333333333333
LabelSpreading 의 정답률 :  0.9333333333333333
LinearDiscriminantAnalysis 의 정답률 :  0.9777777777777777
LinearSVC 의 정답률 :  0.8888888888888888
LogisticRegression 의 정답률 :  0.8222222222222222
LogisticRegressionCV 의 정답률 :  0.9111111111111111
MLPClassifier 의 정답률 :  0.8444444444444444
MultiOutputClassifier 은 없는 모델
MultinomialNB 의 정답률 :  0.6222222222222222
NearestCentroid 의 정답률 :  0.8
NuSVC 의 정답률 :  0.9333333333333333
OneVsOneClassifier 은 없는 모델
OneVsRestClassifier 은 없는 모델
OutputCodeClassifier 은 없는 모델
PassiveAggressiveClassifier 의 정답률 :  0.8444444444444444
Perceptron 의 정답률 :  0.6222222222222222
QuadraticDiscriminantAnalysis 의 정답률 :  0.9555555555555556
RadiusNeighborsClassifier 의 정답률 :  0.6666666666666666
RandomForestClassifier 의 정답률 :  0.9111111111111111
RidgeClassifier 의 정답률 :  0.7555555555555555
RidgeClassifierCV 의 정답률 :  0.8222222222222222
SGDClassifier 의 정답률 :  0.8888888888888888
SVC 의 정답률 :  0.9333333333333333
StackingClassifier 은 없는 모델
VotingClassifier 은 없는 모델
'''