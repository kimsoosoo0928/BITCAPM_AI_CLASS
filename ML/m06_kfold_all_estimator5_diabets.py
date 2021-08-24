import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') # warning 무시 
from sklearn.model_selection import KFold, cross_val_score

datasets = load_diabetes()


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
ARDRegression 의 score :  [0.49874835 0.48765748 0.56284846 0.37728801 0.53474369] 평
균 :  0.4923
AdaBoostRegressor 의 score :  [0.39788477 0.45530288 0.50808134 0.33538165 0.41284606] 평균 :  0.4219
BaggingRegressor 의 score :  [0.32849261 0.42335133 0.3671617  0.32117691 0.36158969] 
평균 :  0.3604
BayesianRidge 의 score :  [0.50082189 0.48431051 0.55459312 0.37600508 0.5307344 ] 평
균 :  0.4893
CCA 의 score :  [0.48696409 0.42605855 0.55244322 0.21708682 0.50764701] 평균 :  0.438DecisionTreeRegressor 의 score :  [-0.23206414 -0.02948661 -0.05322411 -0.00828791  0.12560346] 평균 :  -0.0395
DummyRegressor 의 score :  [-1.54258856e-04 -2.98519672e-03 -1.53442062e-05 -3.80334913e-03
 -9.58335111e-03] 평균 :  -0.0033
ElasticNet 의 score :  [ 0.00810127  0.00637294  0.00924848  0.0040621  -0.00081988] 
평균 :  0.0054
ElasticNetCV 의 score :  [0.43071558 0.461506   0.49133954 0.35674829 0.4567084 ] 평균
 :  0.4394
ExtraTreeRegressor 의 score :  [-0.06909707  0.07382836 -0.02143741 -0.19265067  0.03549258] 평균 :  -0.0348
ExtraTreesRegressor 의 score :  [0.38868403 0.4808745  0.52224076 0.38106702 0.46249624] 평균 :  0.4471
GammaRegressor 의 score :  [ 0.00523561  0.00367973  0.0060814   0.00174734 -0.00306898] 평균 :  0.0027
GaussianProcessRegressor 의 score :  [ -5.63607385 -15.27402151  -9.94987358 -12.46886944 -12.04807958] 평균 :  -11.0754
GradientBoostingRegressor 의 score :  [0.39041077 0.48310705 0.48047899 0.39605508 0.44589411] 평균 :  0.4392
HistGradientBoostingRegressor 의 score :  [0.28899498 0.43812684 0.51713242 0.37267554 0.35643755] 평균 :  0.3947
HuberRegressor 의 score :  [0.50334705 0.47508237 0.54650576 0.36883712 0.5173073 ] 평
균 :  0.4822
IsotonicRegression 의 score :  [nan nan nan nan nan] 평균 :  nan
KNeighborsRegressor 의 score :  [0.39683913 0.32569788 0.43311217 0.32635899 0.35466969] 평균 :  0.3673
KernelRidge 의 score :  [-3.38476443 -3.49366182 -4.0996205  -3.39039111 -3.60041537] 
평균 :  -3.5938
Lars 의 score :  [ 0.49198665 -0.66475442 -1.04410299 -0.04236657  0.51190679] 평균 : 
 -0.1495
LarsCV 의 score :  [0.4931481  0.48774421 0.55427158 0.38001456 0.52413596] 평균 :  0.4879
Lasso 의 score :  [0.34315574 0.35348212 0.38594431 0.31614536 0.3604865 ] 평균 :  0.3518
LassoCV 의 score :  [0.49799859 0.48389346 0.55926851 0.37740074 0.51636393] 평균 :  0.487
LassoLars 의 score :  [0.36543887 0.37812653 0.40638095 0.33639271 0.38444891] 평균 : 
 0.3742
LassoLarsCV 의 score :  [0.49719648 0.48426377 0.55975856 0.37984022 0.51190679] 평균 
:  0.4866
LassoLarsIC 의 score :  [0.49940515 0.49108789 0.56130589 0.37942384 0.5247894 ] 평균 
:  0.4912
LinearRegression 의 score :  [0.50638911 0.48684632 0.55366898 0.3794262  0.51190679] 
평균 :  0.4876
LinearSVR 의 score :  [-0.33470258 -0.31629592 -0.41890962 -0.30194229 -0.47369946] 평
균 :  -0.3691
MLPRegressor 의 score :  [-2.84548327 -3.00474482 -3.13832481 -2.69390457 -3.22217623] 평균 :  -2.9809
MultiOutputRegressor 은 없는 모델
MultiTaskElasticNet 의 score :  [nan nan nan nan nan] 평균 :  nan
MultiTaskElasticNetCV 의 score :  [nan nan nan nan nan] 평균 :  nan
MultiTaskLasso 의 score :  [nan nan nan nan nan] 평균 :  nan
MultiTaskLassoCV 의 score :  [nan nan nan nan nan] 평균 :  nan
NuSVR 의 score :  [0.14471275 0.17351835 0.18539957 0.13894135 0.1663745 ] 평균 :  0.1618
OrthogonalMatchingPursuit 의 score :  [0.32934491 0.285747   0.38943221 0.19671679 0.35916077] 평균 :  0.3121
OrthogonalMatchingPursuitCV 의 score :  [0.47845357 0.48661326 0.55695148 0.37039612 0.53615516] 평균 :  0.4857
PLSCanonical 의 score :  [-0.97507923 -1.68534502 -0.8821301  -1.33987816 -1.16041996] 평균 :  -1.2086
PLSRegression 의 score :  [0.47661395 0.4762657  0.5388494  0.38191443 0.54717873] 평
균 :  0.4842
PassiveAggressiveRegressor 의 score :  [0.4600622  0.48793612 0.52169682 0.36042662 0.50034128] 평균 :  0.4661
PoissonRegressor 의 score :  [0.32061441 0.35803358 0.3666005  0.28203414 0.34340626] 
평균 :  0.3341
RANSACRegressor 의 score :  [ 0.14572255  0.13836054  0.22371182  0.23093295 -0.50953162] 평균 :  0.0458
RadiusNeighborsRegressor 의 score :  [-1.54258856e-04 -2.98519672e-03 -1.53442062e-05 
-3.80334913e-03
 -9.58335111e-03] 평균 :  -0.0033
RandomForestRegressor 의 score :  [0.37805512 0.47721037 0.49005468 0.40396592 0.42156007] 평균 :  0.4342
RegressorChain 은 없는 모델
Ridge 의 score :  [0.40936669 0.44788406 0.47057299 0.34467674 0.43339091] 평균 :  0.4212
RidgeCV 의 score :  [0.49525464 0.48761091 0.55171354 0.3801769  0.52749194] 평균 :  0.4884
SGDRegressor 의 score :  [0.3932582  0.44171765 0.46456931 0.32956563 0.415089  ] 평균
 :  0.4088
SVR 의 score :  [0.14331635 0.18438697 0.17864042 0.1424597  0.1468719 ] 평균 :  0.1591
StackingRegressor 은 없는 모델
TheilSenRegressor 의 score :  [0.50741672 0.46596797 0.54602636 0.31813892 0.51966186] 평균 :  0.4714
TransformedTargetRegressor 의 score :  [0.50638911 0.48684632 0.55366898 0.3794262  0.51190679] 평균 :  0.4876
TweedieRegressor 의 score :  [ 0.00585525  0.00425899  0.00702558  0.00183408 -0.00315042] 평균 :  0.0032
VotingRegressor 은 없는 모델
'''