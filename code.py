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


#DRAWING CODE -----------------------------------------------------------------------------------------------------------------
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse, Line
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.base import runTouchApp
from kivy.lang import Builder
from kivy.graphics import (Canvas, Translate, Fbo, ClearColor, ClearBuffers, Scale)
from kivy.core.image import Image, Texture
from kivy.graphics import Fbo, Color, Rectangle
from kivy.config import Config
from kivy.clock import Clock
Config.set('graphics', 'width', '400')
Config.set('graphics', 'height', '400')


def CheckDrawing():
    DATADIR = "E:/OneDrive - Universiteit Utrecht/Documents/GitHubResporities/NN-HandwrittenCharacters/test.png"

    SelfMadeData = []

    ImgArray = cv2.imread(DATADIR, cv2.IMREAD_GRAYSCALE)
    NewArray = cv2.resize(ImgArray, (img_rows,img_cols))
    x_LiveDrawn = []
    x_LiveDrawn.append(NewArray)

    x_LiveDrawn = np.array(x_LiveDrawn).reshape(-1, img_rows, img_cols, 1)

    x_LiveDrawn = x_LiveDrawn/255.0

    testpredict = model.predict(x_LiveDrawn)
    return testpredict[0].argmax()



class MyPaintWidget(Widget):
    '''def __init__(self):
        super(MyPaintWidget,self).__init__()
        with self.canvas:
            Rectangle(position=(400,400), size=(800,800))'''

    def on_touch_down(self, touch):
        color = (0, 0, 0)
        with self.canvas:
            Color(*color)
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=5)

    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]

    def on_touch_up(self, touch):
        self.export_to_png("test.png")
        
class MyPaintApp(App):

    def build(self):
        self.answer = 0
        parent = Widget(size=(400,400))
        self.painter = MyPaintWidget(size=[400,400])
        clearbtn = Button(text='Clear')
        clearbtn.bind(on_release=self.clear_canvas)
        self.Answerbtn = Button(text=str(self.answer), pos=(0,100), size=(50,50))
        with self.painter.canvas:
            Rectangle(position=(200,200), size=(400,400))
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(self.Answerbtn)   

        Clock.schedule_interval(self.update,0.1)

        return parent

    def clear_canvas(self, obj):
        self.painter.canvas.clear()
        with self.painter.canvas:
            Rectangle(position=(200,200), size=(400,400))

    def update(self, *args):
        self.answer = CheckDrawing()
        self.Answerbtn.text = str(self.answer)



#Mnist Data -----------------------------------------------------------------------------------------------------------------
# Mnist Data set from the keras API (mnist.load_data())
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



#SelfMadeData -----------------------------------------------------------------------------------------------------------------
# All images drawn by hand in paint
DATADIR = "./TestImages"
CATEGORIES = ["0","1","2","3","4","5","6","7","8","9"]

SelfMadeData = []

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    ClassNum = [0,0,0,0,0,0,0,0,0,0]
    ClassNum[CATEGORIES.index(category)] = 1
    for img in os.listdir(path):
        try:
            ImgArray = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            NewArray = cv2.resize(ImgArray, (img_rows,img_cols))
            SelfMadeData.append([NewArray, ClassNum])
        except Exception as e:
            pass

random.shuffle(SelfMadeData)

x_SelfMade = []
y_SelfMade = []

for features, labels in SelfMadeData:
    x_SelfMade.append(features)
    y_SelfMade.append(labels)

x_SelfMade = np.array(x_SelfMade).reshape(-1, img_rows, img_cols, 1)
y_SelfMade = np.array(y_SelfMade)

x_SelfMade = x_SelfMade/255.0



#Chars74K Data -----------------------------------------------------------------------------------------------------------------
# From this website => http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/#download

#TIMO DATADIR
#DATADIR = "E:/OneDrive - Universiteit Utrecht/Downloads/EnglishHnd/English/Hnd/Img"
#TIM DATADIR (jow vul hier jouw path in en maak van mijne een comment (#))
DATADIR = ""

CATEGORIES = ["0","1","2","3","4","5","6","7","8","9"]

TrainingData = []

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    ClassNum = [0,0,0,0,0,0,0,0,0,0]
    ClassNum[CATEGORIES.index(category)] = 1
    for img in os.listdir(path):
        try:
            ImgArray = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            NewArray = cv2.resize(ImgArray, (img_rows,img_cols))
            TrainingData.append([NewArray, ClassNum])
        except Exception as e:
            pass

random.shuffle(TrainingData)

x_74KData = []
y_74KData = []

for features, labels in TrainingData:
    x_74KData.append(features)
    y_74KData.append(labels)

x_74KData = np.array(x_74KData).reshape(-1, img_rows, img_cols, 1)
y_74KData = np.array(y_74KData)

x_74KData = x_74KData/255.0



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
model.fit(x_74KData, y_74KData,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)
          #validation_data=(x_SelfMade, y_SelfMade)) #not sure what this does



#OUTPUT CODE -----------------------------------------------------------------------------------------------------------------

#Chars74K data
score = model.evaluate(x_74KData, y_74KData, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


#SelfMade data
score2 = model.evaluate(x_SelfMade, y_SelfMade, verbose=0)
print('Test loss:', score2[0])
print('Test accuracy:', score2[1])

testpredict2 = model.predict(x_SelfMade)
for x in range(len(testpredict2)):
    print(testpredict2[x].argmax(), " == ", y_SelfMade[x].argmax())


'''#MnistTrain data
score3 = model.evaluate(x_mnistTrain, y_mnistTrain, verbose=0)
print('Test loss:', score3[0])
print('Test accuracy:', score3[1])


#MnistTest data
score4 = model.evaluate(x_mnistTest, y_mnistTest, verbose=0)
print('Test loss:', score4[0])
print('Test accuracy:', score4[1])'''



#Live Drawn Data -----------------------------------------------------------------------------------------------------------------
MyPaintApp().run()