import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') # warning 무시 
from sklearn.model_selection import KFold, cross_val_score

datasets = load_wine()


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
AdaBoostClassifier 의 score :  [0.88888889 0.86111111 0.88888889 0.94285714 0.97142857] 평균
 :  0.9106
BaggingClassifier 의 score :  [1.         0.91666667 0.91666667 0.97142857 1.        ] 평균 
:  0.961
BernoulliNB 의 score :  [0.41666667 0.47222222 0.27777778 0.48571429 0.34285714] 평균 :  0.399
CalibratedClassifierCV 의 score :  [0.94444444 0.94444444 0.88888889 0.88571429 0.91428571] 
평균 :  0.9156
CategoricalNB 의 score :  [       nan        nan        nan 0.94285714        nan] 평균 :  nan
ClassifierChain 은 없는 모델
ComplementNB 의 score :  [0.69444444 0.80555556 0.55555556 0.6        0.6       ] 평균 :  0.6511
DecisionTreeClassifier 의 score :  [0.94444444 0.97222222 0.91666667 0.88571429 0.88571429] 
평균 :  0.921
DummyClassifier 의 score :  [0.41666667 0.47222222 0.27777778 0.48571429 0.34285714] 평균 : 
 0.399
ExtraTreeClassifier 의 score :  [0.94444444 0.77777778 0.77777778 0.88571429 0.88571429] 평 
균 :  0.8543
ExtraTreesClassifier 의 score :  [1.         0.97222222 1.         1.         1.        ] 평
균 :  0.9944
GaussianNB 의 score :  [1.         0.91666667 0.97222222 0.97142857 1.        ] 평균 :  0.9721
GaussianProcessClassifier 의 score :  [0.44444444 0.30555556 0.55555556 0.62857143 0.45714286] 평균 :  0.4783
GradientBoostingClassifier 의 score :  [0.97222222 0.91666667 0.88888889 0.97142857 0.97142857] 평균 :  0.9441
HistGradientBoostingClassifier 의 score :  [0.97222222 0.94444444 1.         0.97142857 1.  
      ] 평균 :  0.9776
KNeighborsClassifier 의 score :  [0.69444444 0.77777778 0.61111111 0.62857143 0.74285714] 평
균 :  0.691
LabelPropagation 의 score :  [0.52777778 0.47222222 0.5        0.4        0.54285714] 평균 :  0.4886
LabelSpreading 의 score :  [0.52777778 0.47222222 0.5        0.4        0.54285714] 평균 :  
0.4886
LinearDiscriminantAnalysis 의 score :  [1.         0.97222222 1.         0.97142857 1.      
  ] 평균 :  0.9887
LinearSVC 의 score :  [0.83333333 0.86111111 0.77777778 0.85714286 0.74285714] 평균 :  0.8144
LogisticRegression 의 score :  [0.97222222 0.94444444 0.94444444 0.94285714 1.        ] 평균
 :  0.9608
LogisticRegressionCV 의 score :  [1.         0.94444444 0.97222222 0.94285714 0.97142857] 평
균 :  0.9662
MLPClassifier 의 score :  [0.19444444 0.08333333 0.36111111 0.17142857 0.85714286] 평균 :  0.3335
MultiOutputClassifier 은 없는 모델
MultinomialNB 의 score :  [0.77777778 0.91666667 0.86111111 0.82857143 0.82857143] 평균 :  0.8425
NearestCentroid 의 score :  [0.69444444 0.72222222 0.69444444 0.77142857 0.74285714] 평균 : 
 0.7251
NuSVC 의 score :  [0.91666667 0.86111111 0.91666667 0.85714286 0.8       ] 평균 :  0.8703
OneVsOneClassifier 은 없는 모델
OneVsRestClassifier 은 없는 모델
OutputCodeClassifier 은 없는 모델
PassiveAggressiveClassifier 의 score :  [0.72222222 0.69444444 0.52777778 0.6        0.57142857] 평균 :  0.6232
Perceptron 의 score :  [0.61111111 0.80555556 0.47222222 0.48571429 0.62857143] 평균 :  0.6006
QuadraticDiscriminantAnalysis 의 score :  [0.97222222 1.         1.         1.         1.   
     ] 평균 :  0.9944
RadiusNeighborsClassifier 의 score :  [nan nan nan nan nan] 평균 :  nan
RandomForestClassifier 의 score :  [1.         0.94444444 1.         0.97142857 1.        ] 
평균 :  0.9832
RidgeClassifier 의 score :  [1.         1.         1.         0.97142857 1.        ] 평균 : 
 0.9943
RidgeClassifierCV 의 score :  [1.         1.         1.         0.97142857 1.        ] 평균 
:  0.9943
SGDClassifier 의 score :  [0.52777778 0.72222222 0.55555556 0.6        0.6       ] 평균 :  0.6011
SVC 의 score :  [0.69444444 0.69444444 0.61111111 0.62857143 0.6       ] 평균 :  0.6457
StackingClassifier 은 없는 모델
VotingClassifier 은 없는 모델
'''