# 실습
# data = diabets 

# 1. 상단모델에 그리드서치 or 랜덤서치로 튜닝한 모델 구성
# 최적의 R2값과 피처임포턴스 구할것

# 2. 위 스레드값으로 SelectFromModel 돌려서
# 최적의 피처 갯수 구할것

# 3. 위 피처 갯수로 피처 갯수 조정한뒤
# 그걸로 다시 랜덤 서치, 그리드 서치해서
# 최적의 R2 구할 것

# 1번값과 3번값 비교




from xgboost import XGBRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV


#1.data
x, y = load_diabetes(return_X_y=True)
print(x.shape, y.shape) # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

n_splits = 5

kfold = KFold(n_splits = n_splits, shuffle=True, random_state=66)
parameters = [
    {'n_estimators':[100, 200], 'max_depth':[6, 8, 10, 12], 'min_samples_split':[2, 3, 5, 10]},
    {'max_depth':[6, 8, 10, 12], 'min_samples_split':[2, 3, 5, 10]},
    {'min_samples_leaf':[3, 5, 7, 10], 'min_samples_split':[2, 3, 5, 10]},
    {'min_samples_split':[2, 3, 5, 10]},
    {'n_jobs':[-1, 2, 4], 'min_samples_split':[2, 3, 5, 10]}
]

#2.model
model = RandomizedSearchCV(XGBRegressor(), parameters, cv=kfold, verbose=1)

#3.fit
model.fit(x_train, y_train)

#4.eval, pred
score = model.score(x_test, y_test)
print("model.score : ", score) # model.score :  0.9221188601856797

thresholds = np.sort(model.feature_importances_)
print(thresholds)

# [0.00134153 0.00363372 0.01203115 0.01220458 0.01447935 0.01479119
#  0.0175432  0.03041655 0.04246345 0.0518254  0.06949984 0.30128643
#  0.42848358]

print("================================================")
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    
    select_x_trian = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_trian.shape, select_x_test.shape)

    selection_model = RandomizedSearchCV(n_jobs=-1)
    selection_model.fit(select_x_trian, y_train)
    
    y_predict = selection_model.predict(select_x_test)
    
    score = r2_score(y_test, y_predict)

    print("Thresh=%3.f, n=%d, R2: %.2f%%" %(thresh,select_x_trian.shape[1], score*100))