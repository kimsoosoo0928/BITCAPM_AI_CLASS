from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRFRegressor
import pandas as pd
import numpy as np

# 1. data
datasets = load_iris()

x_data = datasets.data
y_data = datasets.target

df = pd.DataFrame(x_data, columns=datasets.feature_names)

x_data = df[['petal length (cm)', 
                'petal width (cm)']]
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, train_size=0.8, random_state=66
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

'''
acc :  0.9333333333333333
'''

print(model.feature_importances_) # 강력함
# [0.         0.0125026  0.03213177 0.95536562]
# iris의 컬럼은 4개, 전부 더하면 1
# 첫번째 칼럼은 acc를 만드는데 도움을 주지 않았다.


# 그림 그리기

# import matplotlib.pyplot as plt
# import numpy as np

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

'''
DecisionTreeClassifier original
acc :  0.9333333333333333
[0.0125026  0.         0.53835801 0.44913938]

edit
acc :  0.9333333333333333
[0.03253859 0.96746141]

RandomForestClassifier original
acc :  0.9333333333333333
[0.06342126 0.02072778 0.42673455 0.48911641]

edit
acc :  0.9666666666666667
[0.44124022 0.55875978]

GradientBoostingClassifier original
acc :  0.9666666666666667
[0.00501924 0.0122915  0.26064689 0.72204237]

edit
acc :  0.9666666666666667
[0.29368172 0.70631828]

XGBClassifier original
acc :  0.9
[0.01835513 0.0256969  0.62045246 0.3354955 ]

edit
acc :  0.9666666666666667
[0.510896   0.48910394]
'''