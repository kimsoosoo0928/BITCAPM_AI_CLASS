from sklearn.datasets import load_diabetes, load_iris, load_breast_cancer, load_wine, load_boston
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split, RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, r2_score
import time
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

# parmeters = [
#     {'n_jobs' : [-1], 'n_estimators' : [100, 200], 'max_depth' : [6, 8, 10], 'min_samples_leaf' : [5, 7, 10]},
#     {'n_jobs' : [-1], 'max_depth' : [6, 8, 10], 'min_samples_leaf' : [3, 6, 9, 11], 'min_samples_split' : [3, 4, 5]},
#     {'n_jobs' : [-1], 'min_samples_leaf' : [3, 5, 7], 'min_samples_split' : [3, 4, 5]},
#     {'n_jobs' : [-1], 'min_samples_split' : [2, 3, 5, 10]}
# ]

# parmeters = [
#     {'randomforestclassifier__n_jobs' : [-1], 'randomforestclassifier__n_estimators' : [100, 200], 'randomforestclassifier__max_depth' : [6, 8, 10], 'randomforestclassifier__min_samples_leaf' : [5, 7, 10]},
#     {'randomforestclassifier__n_jobs' : [-1], 'randomforestclassifier__max_depth' : [6, 8, 10], 'randomforestclassifier__min_samples_leaf' : [3, 6, 9, 11], 'randomforestclassifier__min_samples_split' : [3, 4, 5]},
#     {'randomforestclassifier__n_jobs' : [-1], 'randomforestclassifier__min_samples_leaf' : [3, 5, 7], 'randomforestclassifier__min_samples_split' : [3, 4, 5]},
#     {'randomforestclassifier__n_jobs' : [-1], 'randomforestclassifier__min_samples_split' : [2, 3, 5, 10]}
# ]

parmeters = [
    {'rf__n_jobs' : [-1], 'rf__n_estimators' : [100, 200], 'rf__max_depth' : [6, 8, 10], 'rf__min_samples_leaf' : [5, 7, 10]},
    {'rf__n_jobs' : [-1], 'rf__max_depth' : [6, 8, 10], 'rf__min_samples_leaf' : [3, 6, 9, 11], 'rf__min_samples_split' : [3, 4, 5]},
    {'rf__n_jobs' : [-1], 'rf__min_samples_leaf' : [3, 5, 7], 'rf__min_samples_split' : [3, 4, 5]},
    {'rf__n_jobs' : [-1], 'rf__min_samples_split' : [2, 3, 5, 10]}
]

# 2. 모델 구성
# pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())
pipe = Pipeline([('scaler', MinMaxScaler()), ('rf', RandomForestRegressor())])
#! Pipeline은 전체를 list로 감싸야 한다.
#! 약어를 정할 수 있다. ('rf', RandomForestClassifier()) -> RandomForestClassifier를 rf라고 한다.


model = RandomizedSearchCV(pipe, parmeters, cv=kfold, verbose=1)
#! pipe라는 모델에는 parmeters를 가지고 있지 않아서 이렇게 사용 불가(parmeters는 RandomForestClassifier 파라미터이다.)
#^ 파라미터에 어떤 모델의 파라미터인지 모델명 명시 ex){모델명(소문자)__파라미터 : 0 } -> {randomforestclassifier__n_jobs' : [-1]}
#! pipe는 랩핑한 모델



# 3. 컴파일, 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time() - start_time


# 4. 평가, 예측

print('최적의 매개변수 : ', model.best_estimator_)

print('best_params_ : ', model.best_params_)

print('best_score_ : ', model.best_score_)

print('model.score : ', model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('R2 : ', r2_score(y_test, y_predict))

print('걸린 시간 : ', end_time)

'''
최적의 매개변수 :  Pipeline(steps=[('scaler', MinMaxScaler()),
                ('rf',
                 RandomForestRegressor(max_depth=8, min_samples_leaf=6,
                                       min_samples_split=5, n_jobs=-1))])
best_params_ :  {'rf__n_jobs': -1, 'rf__min_samples_split': 5, 'rf__min_samples_leaf': 6, 'rf__max_depth': 8}
best_score_ :  0.49386621743816334
model.score :  0.39019541468661234
R2 :  0.39019541468661223
걸린 시간 :  11.84804630279541
'''