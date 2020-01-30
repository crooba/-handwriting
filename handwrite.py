#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

# 일정한 결과를 도출하기 위해 랜덤시드 고정시킵니다.
np.random.seed(3)

# 데이터를 변형 확대합니다.
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        '/Users/canu/Desktop/handwriting-master 2/test', # train 데이터 경로
        target_size=(24, 24),
        batch_size=3,
        class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        '/Users/canu/Desktop/handwriting-master 2/test', # test 데이터 경로
        target_size=(24, 24),    
        batch_size=3,
        class_mode='categorical')

# 2. CNN인공신경망을 설계합니다.
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(24,24,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 신경망에 필요한 주요함수들을 설정 해 줍니다.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 인공신경망에 학습명령을 내립니다.
model.fit_generator(
        train_generator,
        steps_per_epoch=15,
        epochs=50,
        validation_data=test_generator,
        validation_steps=5)

# 인공신경망의 학습결과를 평가합니다.
print("-- Evaluate --")
scores = model.evaluate_generator(test_generator, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

# 실제 테스트셋으로 인공신경망 예측을 해봅니다.
print("-- Predict --")
output = model.predict_generator(test_generator, steps=5)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(test_generator.class_indices)
print(output)










