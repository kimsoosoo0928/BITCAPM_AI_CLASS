import numpy as np
from sklearn.datasets import load_boston, load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (442, 10) (442,) 

pca = PCA(n_components=8)
x = pca.fit_transform(x)
print(x)
print(x.shape) # (442, 7)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)
print(sum(pca_EVR))

cumsum = np.cumsum(pca_EVR)
print(cumsum)

print(np.argmax(cumsum >= 0.94)+1) # argmax

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()

'''
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66, shuffle=True)

#2. 모델 
from xgboost import XGBRegressor
model = XGBRegressor()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print("result : ", results) 
'''
