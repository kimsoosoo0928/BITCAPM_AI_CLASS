import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') # warning 무시 
from sklearn.model_selection import KFold, cross_val_score

datasets = load_boston()


x = datasets.data
y = datasets.target



#1. 데이터

#1-1. 데이터 전처리

#2. 모델 구성

allAlgorithms = all_estimators(type_filter='regressor')
# allAlgorithms = all_estimators(type_filter='regressor')
print('모델의 갯수 : ', len(allAlgorithms))

kfold = KFold(n_splits=5, shuffle=True, random_state=66)
for (name, algorithm) in allAlgorithms:
    try :
        model = algorithm()

        scores = cross_val_score(model, x, y, cv=kfold)
    
        print(name, '의 score : ', scores, '평균 : ', round(np.mean(scores), 4))
    except :
        # continue
        print(name, '은 없는 모델')

'''
모델의 갯수 :  54
ARDRegression 의 score :  [0.80125693 0.76317071 0.56809285 0.6400258  0.71991866] 평
균 :  0.6985
AdaBoostRegressor 의 score :  [0.90606531 0.80850999 0.7718163  0.82049206 0.86210554] 평균 :  0.8338
BaggingRegressor 의 score :  [0.90999991 0.82053187 0.82693227 0.86338288 0.88100684] 
평균 :  0.8604
BayesianRidge 의 score :  [0.79379186 0.81123808 0.57943979 0.62721388 0.70719051] 평
균 :  0.7038
CCA 의 score :  [0.79134772 0.73828469 0.39419624 0.5795108  0.73224276] 평균 :  0.6471
DecisionTreeRegressor 의 score :  [0.80861208 0.70422402 0.79863737 0.72167606 0.8165147 ] 평균 :  0.7699
DummyRegressor 의 score :  [-0.00053702 -0.03356375 -0.00476023 -0.02593069 -0.00275911] 평균 :  -0.0135
ElasticNet 의 score :  [0.73383355 0.76745241 0.59979782 0.60616114 0.64658354] 평균 :  0.6708
ElasticNetCV 의 score :  [0.71677604 0.75276545 0.59116613 0.59289916 0.62888608] 평균
 :  0.6565
ExtraTreeRegressor 의 score :  [0.74611457 0.80813482 0.46982697 0.66910302 0.84890948] 평균 :  0.7084
ExtraTreesRegressor 의 score :  [0.93084721 0.852783   0.78877397 0.87820813 0.9245535 ] 평균 :  0.875
GammaRegressor 의 score :  [-0.00058757 -0.03146716 -0.00463664 -0.02807276 -0.00298635] 평균 :  -0.0136
GaussianProcessRegressor 의 score :  [-6.07310526 -5.51957093 -6.33482574 -6.36383476 
-5.35160828] 평균 :  -5.9286
GradientBoostingRegressor 의 score :  [0.94530834 0.83349462 0.8266905  0.88718125 0.92957248] 평균 :  0.8844
HistGradientBoostingRegressor 의 score :  [0.93235978 0.82415907 0.78740524 0.88879806 0.85766226] 평균 :  0.8581
HuberRegressor 의 score :  [0.74400323 0.64244715 0.52848946 0.37100122 0.63403398] 평
균 :  0.584
IsotonicRegression 의 score :  [nan nan nan nan nan] 평균 :  nan
KNeighborsRegressor 의 score :  [0.59008727 0.68112533 0.55680192 0.4032667  0.41180856] 평균 :  0.5286
KernelRidge 의 score :  [0.83333255 0.76712443 0.5304997  0.5836223  0.71226555] 평균 
:  0.6854
Lars 의 score :  [0.77467361 0.79839316 0.5903683  0.64083802 0.68439384] 평균 :  0.6977
LarsCV 의 score :  [0.80141197 0.77573678 0.57807429 0.60068407 0.70833854] 평균 :  0.6928
Lasso 의 score :  [0.7240751  0.76027388 0.60141929 0.60458689 0.63793473] 평균 :  0.6657
LassoCV 의 score :  [0.71314939 0.79141061 0.60734295 0.61617714 0.66137127] 평균 :  0.6779
LassoLars 의 score :  [-0.00053702 -0.03356375 -0.00476023 -0.02593069 -0.00275911] 평
균 :  -0.0135
LassoLarsCV 의 score :  [0.80301044 0.77573678 0.57807429 0.60068407 0.72486787] 평균 
:  0.6965
LassoLarsIC 의 score :  [0.81314239 0.79765276 0.59012698 0.63974189 0.72415009] 평균 
:  0.713
LinearRegression 의 score :  [0.81112887 0.79839316 0.59033016 0.64083802 0.72332215] 
평균 :  0.7128
LinearSVR 의 score :  [ 0.76150377  0.68802653 -0.66463081  0.53450776  0.15950321] 평
균 :  0.2958
MLPRegressor 의 score :  [0.52888682 0.50706703 0.47455327 0.38721677 0.50959204] 평균
 :  0.4815
MultiOutputRegressor 은 없는 모델
MultiTaskElasticNet 의 score :  [nan nan nan nan nan] 평균 :  nan
MultiTaskElasticNetCV 의 score :  [nan nan nan nan nan] 평균 :  nan
MultiTaskLasso 의 score :  [nan nan nan nan nan] 평균 :  nan
MultiTaskLassoCV 의 score :  [nan nan nan nan nan] 평균 :  nan
NuSVR 의 score :  [0.2594254  0.33427351 0.263857   0.11914968 0.170599  ] 평균 :  0.2295
OrthogonalMatchingPursuit 의 score :  [0.58276176 0.565867   0.48689774 0.51545117 0.52049576] 평균 :  0.5343
OrthogonalMatchingPursuitCV 의 score :  [0.75264599 0.75091171 0.52333619 0.59442374 0.66783377] 평균 :  0.6578
PLSCanonical 의 score :  [-2.23170797 -2.33245351 -2.89155602 -2.14746527 -1.44488868] 평균 :  -2.2096
PLSRegression 의 score :  [0.80273131 0.76619347 0.52249555 0.59721829 0.73503313] 평
균 :  0.6847
PassiveAggressiveRegressor 의 score :  [ 0.24551922 -3.3720598  -2.89250118 -2.19591202  0.39265386] 평균 :  -1.5645
PoissonRegressor 의 score :  [0.85659255 0.8189989  0.66691488 0.67998192 0.75195656] 
평균 :  0.7549
RANSACRegressor 의 score :  [0.66307741 0.72402633 0.58402449 0.58613965 0.50871702] 
평균 :  0.6132
RadiusNeighborsRegressor 의 score :  [nan nan nan nan nan] 평균 :  nan
RandomForestRegressor 의 score :  [0.92749516 0.84796606 0.80827125 0.8871352  0.90638901] 평균 :  0.8755
RegressorChain 은 없는 모델
Ridge 의 score :  [0.80984876 0.80618063 0.58111378 0.63459427 0.72264776] 평균 :  0.7109
RidgeCV 의 score :  [0.81125292 0.80010535 0.58888304 0.64008984 0.72362912] 평균 :  0.7128
SGDRegressor 의 score :  [-7.53045282e+26 -5.08881179e+26 -7.54185085e+25 -7.60189712e+25
 -4.49498346e+26] 평균 :  -3.725724574209013e+26
SVR 의 score :  [0.23475113 0.31583258 0.24121157 0.04946335 0.14020554] 평균 :  0.1963
StackingRegressor 은 없는 모델
TheilSenRegressor 의 score :  [0.79545143 0.72214175 0.59412698 0.54393893 0.7203848 ] 평균 :  0.6752
TransformedTargetRegressor 의 score :  [0.81112887 0.79839316 0.59033016 0.64083802 0.72332215] 평균 :  0.7128
TweedieRegressor 의 score :  [0.7492543  0.75457294 0.56286929 0.57989884 0.63242475] 
평균 :  0.6558
VotingRegressor 은 없는 모델
'''