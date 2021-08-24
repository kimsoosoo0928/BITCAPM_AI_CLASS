import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') # warning 무시 
from sklearn.model_selection import KFold, cross_val_score

datasets = load_breast_cancer()


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
AdaBoostClassifier 의 score :  [0.94736842 0.99122807 0.94736842 0.96491228 0.97345133] 평균
 :  0.9649
BaggingClassifier 의 score :  [0.95614035 0.92105263 0.9122807  0.93859649 0.95575221] 평균 
:  0.9368
BernoulliNB 의 score :  [0.64035088 0.65789474 0.62280702 0.5877193  0.62831858] 평균 :  0.6274
CalibratedClassifierCV 의 score :  [0.89473684 0.93859649 0.89473684 0.92982456 0.97345133] 
평균 :  0.9263
CategoricalNB 의 score :  [nan nan nan nan nan] 평균 :  nan
ClassifierChain 은 없는 모델
ComplementNB 의 score :  [0.86842105 0.92982456 0.87719298 0.9122807  0.89380531] 평균 :  0.8963
DecisionTreeClassifier 의 score :  [0.9122807  0.92982456 0.92105263 0.87719298 0.95575221] 
평균 :  0.9192
DummyClassifier 의 score :  [0.64035088 0.65789474 0.62280702 0.5877193  0.62831858] 평균 : 
 0.6274
ExtraTreeClassifier 의 score :  [0.9122807  0.92982456 0.87719298 0.92982456 0.92920354] 평
균 :  0.9157
ExtraTreesClassifier 의 score :  [0.96491228 0.98245614 0.97368421 0.94736842 1.        ] 평
균 :  0.9737
GaussianNB 의 score :  [0.93859649 0.96491228 0.9122807  0.93859649 0.95575221] 평균 :  0.942
GaussianProcessClassifier 의 score :  [0.87719298 0.89473684 0.89473684 0.94736842 0.94690265] 평균 :  0.9122
GradientBoostingClassifier 의 score :  [0.95614035 0.97368421 0.95614035 0.93859649 0.98230088] 평균 :  0.9614
HistGradientBoostingClassifier 의 score :  [0.97368421 0.98245614 0.96491228 0.96491228 0.98230088] 평균 :  0.9737
KNeighborsClassifier 의 score :  [0.92105263 0.92105263 0.92105263 0.92105263 0.95575221] 평
균 :  0.928
LabelPropagation 의 score :  [0.36842105 0.35964912 0.4122807  0.42105263 0.38938053] 평균 :  0.3902
LabelSpreading 의 score :  [0.36842105 0.35964912 0.4122807  0.42105263 0.38938053] 평균 :  
0.3902
LinearDiscriminantAnalysis 의 score :  [0.94736842 0.98245614 0.94736842 0.95614035 0.97345133] 평균 :  0.9614
LinearSVC 의 score :  [0.64035088 0.94736842 0.90350877 0.94736842 0.63716814] 평균 :  0.8152
LogisticRegression 의 score :  [0.94736842 0.95614035 0.88596491 0.95614035 0.96460177] 평균
 :  0.942
LogisticRegressionCV 의 score :  [0.95614035 0.97368421 0.90350877 0.96491228 0.96460177] 평
균 :  0.9526
MLPClassifier 의 score :  [0.90350877 0.89473684 0.86842105 0.94736842 0.9380531 ] 평균 :  0.9104
MultiOutputClassifier 은 없는 모델
MultinomialNB 의 score :  [0.85964912 0.92105263 0.87719298 0.9122807  0.89380531] 평균 :  0.8928
NearestCentroid 의 score :  [0.86842105 0.89473684 0.85964912 0.9122807  0.91150442] 평균 : 
 0.8893
NuSVC 의 score :  [0.85964912 0.9122807  0.83333333 0.87719298 0.88495575] 평균 :  0.8735
OneVsOneClassifier 은 없는 모델
OneVsRestClassifier 은 없는 모델
OutputCodeClassifier 은 없는 모델
PassiveAggressiveClassifier 의 score :  [0.90350877 0.93859649 0.86842105 0.92982456 0.59292035] 평균 :  0.8467
Perceptron 의 score :  [0.40350877 0.80701754 0.85964912 0.86842105 0.94690265] 평균 :  0.7771
QuadraticDiscriminantAnalysis 의 score :  [0.93859649 0.95614035 0.93859649 0.98245614 0.94690265] 평균 :  0.9525
RadiusNeighborsClassifier 의 score :  [nan nan nan nan nan] 평균 :  nan
RandomForestClassifier 의 score :  [0.96491228 0.96491228 0.97368421 0.95614035 0.97345133] 
평균 :  0.9666
RidgeClassifier 의 score :  [0.95614035 0.98245614 0.92105263 0.95614035 0.95575221] 평균 : 
 0.9543
RidgeClassifierCV 의 score :  [0.94736842 0.97368421 0.93859649 0.95614035 0.96460177] 평균 
:  0.9561
SGDClassifier 의 score :  [0.88596491 0.93859649 0.86842105 0.86842105 0.92920354] 평균 :  0.8981
SVC 의 score :  [0.89473684 0.92982456 0.89473684 0.92105263 0.96460177] 평균 :  0.921
StackingClassifier 은 없는 모델
VotingClassifier 은 없는 모델
'''