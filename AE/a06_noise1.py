### 앞뒤가 똑같은 오~토인코더~(중요하지 않은 특성들은 도태됨)    / (특징이 강한 것을 더 강하게 해주는 것은 아님)

import numpy as np
from tensorflow.keras.datasets import mnist


# 1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train1 = x_train.reshape(60000, 784).astype('float')/255
x_train2 = x_train.reshape(60000, 28*28).astype('float')/255
x_test = x_test.reshape(10000,784).astype('float')/255

# add noise 
x_train_noised = x_train + np.random.normal(0, 0.1, size = x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size = x_test.shape)
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)


# 2. 모델
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten

def autoEncoder1(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,), activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

def autoEncoder2(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size, kernel_size=(2, 2), 
                input_shape=(28, 28, 1),
                activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (2, 2), activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (2, 2), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(784, activation='sigmoid'))
    return model


model1 = autoEncoder1(hidden_layer_size=104)      # pca 95%
model2 = autoEncoder2(hidden_layer_size=104)

model1.compile(optimizer='adam', loss='mse')
model2.compile(optimizer='adam', loss='mse')

model1.fit(x_train_noised, x_train1, epochs=10, batch_size=32)
model2.fit(x_train_noised, x_train2, epochs=10, batch_size=32)


output1 = model1.predict(x_test)
output2 = model2.predict(x_test)


from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = \
         plt.subplots(3, 5, figsize=(20, 7))



# 이미지 다섯 개 무작위
random_images = random.sample(range(output1.shape[0]), 5)
random_images = random.sample(range(output2.shape[0]), 5)


# 원본 이미지
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i == 0:
        ax.set_ylabel('INPUT', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# basic 오토인코더가 출력한 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output1[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i == 0:
        ax.set_ylabel('basic', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# cnn 오토인코더가 출력한 이미지 
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output2[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i == 0:
        ax.set_ylabel('deep', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()