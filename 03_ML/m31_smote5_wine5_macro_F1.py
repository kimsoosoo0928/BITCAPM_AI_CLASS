# range change 
# 3,4,5 -> 0
# 6 -> 1
# 7,8,9 -> 2

from numpy.lib.function_base import average
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import time
import warnings
warnings=warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, f1_score

# 1. data
datasets = pd.read_csv('./_data/winequality-white.csv', sep=';',
                       index_col=None, header=0 ) # (4898, 12)

datasets = datasets.values

x = datasets[:,0:11] # (4898, 11)
y = datasets[:,11] # (4898,)
y = np.array(y)
# print(pd.Series(y).value_counts())
# 6.0    2198
# 5.0    1457
# 7.0     880
# 8.0     175
# 4.0     163
# 3.0      20
# 9.0       5 <- too small, merge with 8

##############################################
#                label merge
##############################################

# Mine
# for i in range(y.shape[0]):
#     if y[i] == 9.0:
#         y[i] = 8.0
#     elif y[i] == 7.0:
#         y[i] = 8.0
#     elif y[i] == 4.0:
#         y[i] = 5.0

# Class
for index, value in enumerate(y):
    if value == 3.0:
        y[index] = 0
    elif value ==4.0:
        y[index] = 0
    elif value == 5.0:
        y[index] = 0
    elif value == 6.0:
        y[index] = 1
    elif value == 7.0:
        y[index] = 2
    elif value == 8.0:
        y[index] = 2
    elif value == 9.0:
        y[index] = 2        

# print(pd.Series(y).value_counts())
# 6.0    2198
# 5.0    1457
# 7.0     880
# 4.0     183
# 8.0     180

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.2, shuffle=True, random_state=777, stratify=None)

# from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
# scaler = RobustScaler()
# scaler.fit(x_train) 
# x_train = scaler.transform(x_train) 
# x_test = scaler.transform(x_test) 

# print(pd.Series(y_train).value_counts())

# 2. model
model = XGBClassifier(n_jobs=-1)

# 3. train
model.fit(x_train, y_train, eval_metric='mlogloss')

# 4. eval
score = model.score(x_test, y_test)

y_pred = model.predict(x_test)
f1 = f1_score(y_test, y_pred, average='macro')

print("==================SMOTE==================")

st = time.time()
smote = SMOTE(random_state=77, k_neighbors=60)
et = time.time() - st
x_smote, y_smote = smote.fit_resample(x_train, y_train)

# print(pd.Series(y_smote).value_counts())

# 2. model
model2 = XGBClassifier(n_jobs=-1)

# 3. train
model2.fit(x_smote, y_smote, eval_metric='mlogloss')

# 4. eval
score2 = model2.score(x_test, y_test)

y_pred2 = model2.predict(x_test)
f12 = f1_score(y_test, y_pred2, average='macro')

print("before smote :", x_train.shape, y_train.shape)
print("after smote  :", x_smote.shape, y_smote.shape)
print("before somote labels :\n",pd.Series(y_train).value_counts())
print("after somote labels  :\n",pd.Series(y_smote).value_counts())

print("model_best_score_default :", score)
print("model_best_score_smote   :", score2)
print("f1_score_default:", f1)
print("f1_score_smote  :", f12)


# model_best_score_default : 0.7091836734693877
# model_best_score_smote   : 0.6673469387755102
# f1_score_default: 0.7016259228564506
# f1_score_smote  : 0.6609887478722781