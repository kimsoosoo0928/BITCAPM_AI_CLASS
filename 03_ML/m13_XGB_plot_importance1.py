from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRFRegressor
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from xgboost import plot_importance

# 1. data
datasets = load_boston()
x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, train_size=0.8, random_state=66
)

# 2. model
model = DecisionTreeRegressor(max_depth=4)
# model = GradientBoostingClassifier()
# model = RandomForestRegressor()
model = XGBRFRegressor()

# 3. fit
model.fit(x_train, y_train)

# 4. eval, pred
acc = model.score(x_test, y_test)
print('acc : ', acc)



print(model.feature_importances_) # 강력함
# [0.         0.0125026  0.03213177 0.95536562]
# iris의 컬럼은 4개, 전부 더하면 1
# 첫번째 칼럼은 acc를 만드는데 도움을 주지 않았다.
# 

import matplotlib.pyplot as plt
import numpy as np

# def plot_feature_importances_dataset(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_,
#             align='center')
#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel("Feature Importances")
#     plt.ylabel("Features")
#     plt.ylim(-1, n_features)

# plot_feature_importances_dataset(model)
# plt.show()

plot_importance(model)
plt.show