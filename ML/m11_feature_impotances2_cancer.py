from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRFRegressor
from sklearn.datasets import load_iris,load_breast_cancer
from sklearn.model_selection import train_test_split

# 1. data
datasets = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, train_size=0.8, random_state=66
)

# 2. model
# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
model = XGBClassifier()

# 3. fit
model.fit(x_train, y_train)

# 4. eval, pred
acc = model.score(x_test, y_test)
print('acc : ', acc)


print(model.feature_importances_) 

'''
model = DecisionTreeClassifier
acc :  0.9298245614035088
[0.         0.0624678  0.         0.         0.         0.
 0.         0.00492589 0.         0.         0.         0.
 0.         0.01297421 0.         0.         0.02364429 0.
 0.         0.         0.         0.01695087 0.         0.75156772
 0.         0.         0.00485651 0.11522388 0.         0.00738884]

model = RandomForestClassifier
[0.0197549  0.01571862 0.0169272  0.06743669 0.00486097 0.01031253
 0.03621617 0.07927706 0.00383975 0.00286575 0.01574603 0.00536754
 0.00630736 0.04155111 0.00267391 0.00350267 0.00683882 0.00700122
 0.00481098 0.00413465 0.13318952 0.02112425 0.16294886 0.11702865
 0.01410417 0.01237413 0.01964873 0.15124066 0.00605613 0.00714096]
'''

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
            align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()