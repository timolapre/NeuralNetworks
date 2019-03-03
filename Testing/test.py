from __future__ import print_function
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt


#Google QuickDraw data -----------------------------------------------------------------------------------------------------------------
# All data from google

img_rows = 28
img_cols = 28

def GetData(x,ClassNum):
    array = []
    temparray = []
    temparray2 = []
    for n in range(200):
        count = 0
        for p in x[n]:#range(28*28):
            count += 1
            temparray.append([1-(p/255)])
            if(count == 28):
                array.append(temparray)
                temparray = []
                count = 0            
        #print(array)
        FullData.append((array,ClassNum))
        array = []


FullData = []

DATADIR = "./Data"
CATEGORIES = ["Circle","Square","Triangle"]

for category in CATEGORIES:
    path = os.path.join(DATADIR, category + ".npy")
    ClassNum = [0,0,0]
    ClassNum[CATEGORIES.index(category)] = 1
    load = np.load(path)
    GetData(load,ClassNum)

print(len(FullData))
print(len(FullData[0][0]))
print(len(FullData[0][0][0]))
random.shuffle(FullData)

x_Data = []
y_Data = []

for features, labels in FullData:
    x_Data.append(features)
    y_Data.append(labels)

print(len(x_Data))
print(len(x_Data[0]))
print(len(x_Data[0][0]))





#print(x_data[999])
#print(len(x[0]))
#plt.imshow(x_data[0])
#plt.show()


#TRAINING CODE  -----------------------------------------------------------------------------------------------------------------

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#Change the first 2 parameter to change trainingdata
model.fit([x_Data], [y_Data],
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)
          #validation_data=(x_SelfMade, y_SelfMade)) #not sure what this does


#Mnist Data -----------------------------------------------------------------------------------------------------------------
# Mnist Data set from the keras API (mnist.load_data())
'''import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 20
num_classes = 10
epochs = 20

img_rows, img_cols = 28, 28

(x_mnistTrain, y_mnistTrain), (x_mnistTest, y_mnistTest) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_mnistTrain = x_mnistTrain.reshape(x_mnistTrain.shape[0], 1, img_rows, img_cols)
    x_mnistTest = x_mnistTest.reshape(x_mnistTest.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_mnistTrain = x_mnistTrain.reshape(x_mnistTrain.shape[0], img_rows, img_cols, 1)
    x_mnistTest = x_mnistTest.reshape(x_mnistTest.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_mnistTrain = x_mnistTrain.astype('float32')[:10000] #taking the first x of data
x_mnistTest = x_mnistTest.astype('float32')

x_mnistTrain /= 255
x_mnistTest /= 255
x_mnistTrain = 1-x_mnistTrain
x_mnistTest = 1-x_mnistTest

y_mnistTrain = keras.utils.to_categorical(y_mnistTrain, num_classes)[:10000] #taking the first x of data (should be same as x_mnistTrain)
y_mnistTest = keras.utils.to_categorical(y_mnistTest, num_classes)

print(x_mnistTrain[0])'''