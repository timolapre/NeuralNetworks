import tensorflow as tf
#from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random

DATADIR = "E:/Downloads/EnglishHnd/English/Hnd/Img"
CATEGORIES = ["0","1","2","3","4","5","6","7","8","9"]

TrainingData = []

def CreateTrainingData():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        ClassNum = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                ImgArray = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                NewArray = cv2.resize(ImgArray, (300,225)) #300 225
                TrainingData.append([NewArray, ClassNum])
            except Exception as e:
                pass

CreateTrainingData()
random.shuffle(TrainingData)

TrainImages = []
TrainLabels = []

for features, labels in TrainingData:
    TrainImages.append(features)
    TrainLabels.append(labels)

TrainImages = np.array(TrainImages).reshape(-1, 300, 225, 1)
TrainLabels = np.array(TrainLabels)

"""import pickle 
PickleOut = open("TrainImages.pickle", "wb")
pickle.dump(TrainImages, PickleOut)
PickleOut.close()

PickleOut = open("TrainLabels.pickle", "wb")
pickle.dump(TrainLabels, PickleOut)
PickleOut.close()"""

TrainImages = TrainImages/255.0

'''model = keras.Sequential()[
    keras.layers.Flatten(input_shape=(300, 225)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.relu)
])'''

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = TrainImages.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(10))
model.add(Activation('sigmoid'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

model.fit(TrainImages, TrainLabels, batch_size=20, epochs=10)