import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') # warning 무시 
from sklearn.model_selection import KFold, cross_val_score

datasets = load_iris()


x = datasets.data
y = datasets.target



#1. 데이터

#1-1. 데이터 전처리

#2. 모델 구성

allAlgorithms = all_estimators(type_filter='classifier')
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
모델의 갯수 :  41
AdaBoostClassifier 의 score :  [0.63333333 0.93333333 1.         0.9        0.96666667] 평균
 :  0.8867
BaggingClassifier 의 score :  [0.93333333 0.96666667 1.         0.9        0.96666667] 평균 
:  0.9533
BernoulliNB 의 score :  [0.3        0.33333333 0.3        0.23333333 0.3       ] 평균 :  0.2933
CalibratedClassifierCV 의 score :  [0.9        0.83333333 1.         0.86666667 0.96666667] 
평균 :  0.9133
CategoricalNB 의 score :  [0.9        0.93333333 0.93333333 0.9        1.        ] 평균 :  0.9333
ClassifierChain 은 없는 모델
ComplementNB 의 score :  [0.66666667 0.66666667 0.7        0.6        0.7       ] 평균 :  0.6667
DecisionTreeClassifier 의 score :  [0.96666667 0.96666667 1.         0.9        0.93333333] 
평균 :  0.9533
DummyClassifier 의 score :  [0.3        0.33333333 0.3        0.23333333 0.3       ] 평균 : 
 0.2933
ExtraTreeClassifier 의 score :  [0.96666667 0.93333333 0.96666667 0.83333333 1.        ] 평 
균 :  0.94
ExtraTreesClassifier 의 score :  [0.96666667 0.96666667 1.         0.86666667 0.96666667] 평
균 :  0.9533
GaussianNB 의 score :  [0.96666667 0.9        1.         0.9        0.96666667] 평균 :  0.9467
GaussianProcessClassifier 의 score :  [0.96666667 0.96666667 1.         0.9        0.96666667] 평균 :  0.96
GradientBoostingClassifier 의 score :  [0.96666667 0.96666667 1.         0.93333333 0.96666667] 평균 :  0.9667
HistGradientBoostingClassifier 의 score :  [0.86666667 0.96666667 1.         0.9        0.96666667] 평균 :  0.94
KNeighborsClassifier 의 score :  [0.96666667 0.96666667 1.         0.9        0.96666667] 평
균 :  0.96
LabelPropagation 의 score :  [0.93333333 1.         1.         0.9        0.96666667] 평균 :  0.96
LabelSpreading 의 score :  [0.93333333 1.         1.         0.9        0.96666667] 평균 :  
0.96
LinearDiscriminantAnalysis 의 score :  [1.  1.  1.  0.9 1. ] 평균 :  0.98
LinearSVC 의 score :  [0.96666667 0.96666667 1.         0.9        1.        ] 평균 :  0.9667
LogisticRegression 의 score :  [1.         0.96666667 1.         0.9        0.96666667] 평균
 :  0.9667
LogisticRegressionCV 의 score :  [1.         0.96666667 1.         0.9        1.        ] 평
균 :  0.9733
MLPClassifier 의 score :  [0.96666667 0.96666667 1.         0.93333333 1.        ] 평균 :  0.9733
MultiOutputClassifier 은 없는 모델
MultinomialNB 의 score :  [0.96666667 0.93333333 1.         0.93333333 1.        ] 평균 :  0.9667
NearestCentroid 의 score :  [0.93333333 0.9        0.96666667 0.9        0.96666667] 평균 : 
 0.9333
NuSVC 의 score :  [0.96666667 0.96666667 1.         0.93333333 1.        ] 평균 :  0.9733
OneVsOneClassifier 은 없는 모델
OneVsRestClassifier 은 없는 모델
OutputCodeClassifier 은 없는 모델
PassiveAggressiveClassifier 의 score :  [0.86666667 0.9        0.9        0.86666667 0.76666667] 평균 :  0.86
Perceptron 의 score :  [0.66666667 0.66666667 0.93333333 0.73333333 0.9       ] 평균 :  0.78QuadraticDiscriminantAnalysis 의 score :  [1.         0.96666667 1.         0.93333333 1.   
     ] 평균 :  0.98
RadiusNeighborsClassifier 의 score :  [0.96666667 0.9        0.96666667 0.93333333 1.       
 ] 평균 :  0.9533
RandomForestClassifier 의 score :  [0.93333333 0.96666667 1.         0.9        0.96666667] 
평균 :  0.9533
RidgeClassifier 의 score :  [0.86666667 0.8        0.93333333 0.7        0.9       ] 평균 : 
 0.84
RidgeClassifierCV 의 score :  [0.86666667 0.8        0.93333333 0.7        0.9       ] 평균 
:  0.84
SGDClassifier 의 score :  [0.86666667 0.86666667 0.93333333 0.6        0.9       ] 평균 :  0.8333
SVC 의 score :  [0.96666667 0.96666667 1.         0.93333333 0.96666667] 평균 :  0.9667
StackingClassifier 은 없는 모델
VotingClassifier 은 없는 모델
'''