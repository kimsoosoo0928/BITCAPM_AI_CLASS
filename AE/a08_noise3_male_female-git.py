# 실습, 과제
# keras61_5 남자 여자 데이터에 노이즈를 넣어서 
# 기미, 주근깨, 여드름 제거하시오

### 앞뒤가 똑같은 오~토인코더~(중요하지 않은 특성들은 도태됨)    / (특징이 강한 것을 더 강하게 해주는 것은 아님)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# 1. data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

x_train = np.load('./_save/_npy/k59_mw_x_train.npy')
y_train = np.load('./_save/_npy/k59_mw_y_train.npy')
x_test = np.load('./_save/_npy/k59_mw_x_test.npy')
y_test = np.load('./_save/_npy/k59_mw_y_test.npy')

augment_size = 400 

randidx = np.random.randint(x_train.shape[0], size=augment_size)
print(x_train.shape[0]) # 60000
print(randidx)          # [20693 47880 21722 ... 50370 50531 26723]
print(randidx.shape)    # (40000,)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
print(x_augmented.shape) # (40000, 28, 28)

x_augmented = x_augmented.reshape(x_augmented.shape[0], 150, 150, 3)
x_train = x_train.reshape(x_train.shape[0], 150, 150, 3)
x_test = x_test.reshape(x_test.shape[0], 150, 150, 3)

x_augmented = train_datagen.flow(x_augmented, np.zeros(augment_size),
                                 batch_size=augment_size, shuffle=False).next()[0]

print(x_augmented.shape) # (40000, 28, 28)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

# add noise 
x_train_noised = x_train + np.random.normal(0, 0.2, size = x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.2, size = x_test.shape)
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)


# 2. model
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, Dropout, UpSampling2D

def autoEncoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(hidden_layer_size, kernel_size=(2, 2), 
                input_shape=(150, 150, 3),
                activation='relu', padding='same'))
    model.add(MaxPooling2D(3,3))
    model.add(Conv2D(100, (2, 2), activation='relu', padding='same'))
    
    model.add(UpSampling2D(size=(1,1)))

    model.add(Conv2D(3, (2, 2), activation='sigmoid', padding='same'))
    return model


model = autoEncoder(hidden_layer_size=154)      # pca 95%

# 3. compile train
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_acc', patience=10, mode='auto', verbose=1)

import time 

start_time = time.time()
hist = model.fit(x_train_noised, x_train, epochs=100, verbose=2,
    validation_split=0.05, callbacks=[es])
end_time = time.time() - start_time                          

output = model.predict(x_test_noised)

# 5. visualize
from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10),
    (ax11, ax12, ax13, ax14, ax15)) = \
    plt.subplots(3, 5, figsize = (20, 7))

# 이미지 다섯 개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]), 5)

# original image
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(150, 150,3), cmap = 'gray')
    if i == 0:
        ax.set_ylabel('INPUT', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# noised image
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(150, 150,3), cmap = 'gray')
    if i == 0:
        ax.set_ylabel('OUTPUT', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# original image
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(x_test_noised[random_images[i]].reshape(150, 150,3), cmap = 'gray')
    if i == 0:
        ax.set_ylabel('noise', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()




