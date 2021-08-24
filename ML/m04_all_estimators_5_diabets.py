import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.utils import all_estimators
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore') # warning 무시 

#1. 데이터 
datasets = load_diabetes()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.4)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, random_state=66, test_size=0.5)

print(x.shape, y.shape) # (442, 10) (442,)

print(datasets.feature_names)
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print(datasets.DESCR)

print(y[:30])
print(np.min(y), np.max(y))

#2. 모델 구성

# allAlgorithms = all_estimators(type_filter='classifier')
allAlgorithms = all_estimators(type_filter='regressor')
print('모델의 갯수 : ', len(allAlgorithms))


for (name, algorithm) in allAlgorithms:
    try :
        model = algorithm()
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        r2 = r2_score(y_test, y_predict)
        print(name, '의 r2_score : ', r2)
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


'''
모델의 갯수 :  54
ARDRegression 의 r2_score :  0.4824529107842852
AdaBoostRegressor 의 r2_score :  0.5226979549603046
BaggingRegressor 의 r2_score :  0.4060146381048776
BayesianRidge 의 r2_score :  0.4821267535518958
CCA 의 r2_score :  0.34446286222395817
DecisionTreeRegressor 의 r2_score :  0.111905462652374
DummyRegressor 의 r2_score :  -0.0009005784670110817
ElasticNet 의 r2_score :  0.007935835515496592
ElasticNetCV 의 r2_score :  0.45774782148241866
ExtraTreeRegressor 의 r2_score :  -0.10482910816705981
ExtraTreesRegressor 의 r2_score :  0.5007826770062831
GammaRegressor 의 r2_score :  0.005963789360376004
GaussianProcessRegressor 의 r2_score :  -17.470563825441456
GradientBoostingRegressor 의 r2_score :  0.4790815753782165
HistGradientBoostingRegressor 의 r2_score :  0.5168159763272684
HuberRegressor 의 r2_score :  0.4701773727813686
IsotonicRegression 은 없는 모델
KNeighborsRegressor 의 r2_score :  0.38591567032320184
KernelRidge 의 r2_score :  -3.2081396006792735
Lars 의 r2_score :  0.4416481775018317
LarsCV 의 r2_score :  0.4848525810645119
Lasso 의 r2_score :  0.3593786713755054
LassoCV 의 r2_score :  0.4837961347586611
LassoLars 의 r2_score :  0.4070021600107506
LassoLarsCV 의 r2_score :  0.4840260587350461
LassoLarsIC 의 r2_score :  0.48101013408216886
LinearRegression 의 r2_score :  0.4865486228911733
LinearSVR 의 r2_score :  -0.4286883208713208
MLPRegressor 의 r2_score :  -2.8248286292198133
MultiOutputRegressor 은 없는 모델
MultiTaskElasticNet 은 없는 모델
MultiTaskElasticNetCV 은 없는 모델
MultiTaskLasso 은 없는 모델
MultiTaskLassoCV 은 없는 모델
NuSVR 의 r2_score :  0.12753034662884155
OrthogonalMatchingPursuit 의 r2_score :  0.306212680326449
OrthogonalMatchingPursuitCV 의 r2_score :  0.46626623705099157
PLSCanonical 의 r2_score :  -1.1826969046919542
PLSRegression 의 r2_score :  0.4611705528212522
PassiveAggressiveRegressor 의 r2_score :  0.4786221830523689
PoissonRegressor 의 r2_score :  0.3749806030785017
RANSACRegressor 의 r2_score :  -0.09781079729215114
RadiusNeighborsRegressor 의 r2_score :  -0.0009005784670110817
RandomForestRegressor 의 r2_score :  0.4920848209350571
RegressorChain 은 없는 모델
Ridge 의 r2_score :  0.4212717539644605
RidgeCV 의 r2_score :  0.48532360374112016
SGDRegressor 의 r2_score :  0.4164809229095068
SVR 의 r2_score :  0.1421855413494606
StackingRegressor 은 없는 모델
TheilSenRegressor 의 r2_score :  0.45920857694510797
TransformedTargetRegressor 의 r2_score :  0.4865486228911733
TweedieRegressor 의 r2_score :  0.005875999954029876
VotingRegressor 은 없는 모델
'''


