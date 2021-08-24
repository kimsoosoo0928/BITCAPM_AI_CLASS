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
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') # warning 무시 

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
ARDRegression 의 r2_score :  0.8037449551797082
AdaBoostRegressor 의 r2_score :  0.8656964339223286
BaggingRegressor 의 r2_score :  0.86427236614972
BayesianRidge 의 r2_score :  0.8037638373928007
CCA 의 r2_score :  0.775727268564683
DecisionTreeRegressor 의 r2_score :  0.737317829911649
DummyRegressor 의 r2_score :  -0.005227869326375867
ElasticNet 의 r2_score :  0.11111593122649766
ElasticNetCV 의 r2_score :  0.7971602685398055
ExtraTreeRegressor 의 r2_score :  0.5658658159030137
ExtraTreesRegressor 의 r2_score :  0.8966074864053218
GammaRegressor 의 r2_score :  0.1552927455615204
GaussianProcessRegressor 의 r2_score :  -2.922037281680814
GradientBoostingRegressor 의 r2_score :  0.9146550941517312
HistGradientBoostingRegressor 의 r2_score :  0.8992462166487565
HuberRegressor 의 r2_score :  0.7673847338137735
IsotonicRegression 은 없는 모델
KNeighborsRegressor 의 r2_score :  0.7843808845761827
KernelRidge 의 r2_score :  0.7730664031930277
Lars 의 r2_score :  0.8044888426543626
LarsCV 의 r2_score :  0.8032830033921295
Lasso 의 r2_score :  0.2066075044438942
LassoCV 의 r2_score :  0.8046645285582139
LassoLars 의 r2_score :  -0.005227869326375867
LassoLarsCV 의 r2_score :  0.8044516427844496
LassoLarsIC 의 r2_score :  0.7983441148086403
LinearRegression 의 r2_score :  0.8044888426543627
LinearSVR 의 r2_score :  0.5923017181240726
MLPRegressor 의 r2_score :  0.20775695419501472
MultiOutputRegressor 은 없는 모델
MultiTaskElasticNet 은 없는 모델
MultiTaskElasticNetCV 은 없는 모델
MultiTaskLasso 은 없는 모델
MultiTaskLassoCV 은 없는 모델
NuSVR 의 r2_score :  0.5641595335839983
OrthogonalMatchingPursuit 의 r2_score :  0.5651272222459415
OrthogonalMatchingPursuitCV 의 r2_score :  0.7415292549226281
PLSCanonical 의 r2_score :  -2.271724502623781
PLSRegression 의 r2_score :  0.7738717095948147
PassiveAggressiveRegressor 의 r2_score :  0.7129989564649479
PoissonRegressor 의 r2_score :  0.5823083831246141
RANSACRegressor 의 r2_score :  0.2338519430778212
RadiusNeighborsRegressor 의 r2_score :  0.3637807499142598
RandomForestRegressor 의 r2_score :  0.8863183768405702
RegressorChain 은 없는 모델
Ridge 의 r2_score :  0.7840559169142114
RidgeCV 의 r2_score :  0.8040739643153296
SGDRegressor 의 r2_score :  0.775346045217807
SVR 의 r2_score :  0.5591605061048468
StackingRegressor 은 없는 모델
TheilSenRegressor 의 r2_score :  0.7582277647740654
TransformedTargetRegressor 의 r2_score :  0.8044888426543627
TweedieRegressor 의 r2_score :  0.1504397995060236
VotingRegressor 은 없는 모델
'''


