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

img_rows, img_cols = 28, 28

def CheckDrawing():
    DATADIR = "E:/OneDrive - Universiteit Utrecht/Documents/GitHubResporities/NN-HandwrittenCharacters/drawing.png"

    try:
        ImgArray = cv2.imread(DATADIR, cv2.IMREAD_GRAYSCALE)
        NewArray = cv2.resize(ImgArray, (img_rows,img_cols))
        x_LiveDrawn = []
        x_LiveDrawn.append(NewArray)

        x_LiveDrawn = np.array(x_LiveDrawn).reshape(-1, img_rows, img_cols, 1)

        x_LiveDrawn = x_LiveDrawn/255.0

        testpredict = model.predict(x_LiveDrawn)
        return CATEGORIES[testpredict[0].argmax()]
    except Exception as e:
        pass

def CheckDrawing2():
    DATADIR = "E:/OneDrive - Universiteit Utrecht/Documents/GitHubResporities/NN-HandwrittenCharacters/drawing.png"

    try:
        ImgArray = cv2.imread(DATADIR, cv2.IMREAD_GRAYSCALE)
        NewArray = cv2.resize(ImgArray, (img_rows,img_cols))
        x_LiveDrawn = []
        x_LiveDrawn.append(NewArray)

        x_LiveDrawn = np.array(x_LiveDrawn).reshape(-1, img_rows, img_cols, 1)

        x_LiveDrawn = x_LiveDrawn/255.0

        testpredict = model.predict(x_LiveDrawn)
        return testpredict
    except Exception as e:
        pass



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
        self.export_to_png("drawing.png")
        answer = CheckDrawing2()
        total = sum(answer[0])
        for x in range(len(CATEGORIES)):
            print(CATEGORIES[x] + " = " + str(round(answer[0][x]/total*100,2))  + "%")
        print()
        
class MyPaintApp(App):

    def build(self):
        self.answer = 0
        parent = Widget(size=(400,400))
        self.painter = MyPaintWidget(size=[400,400])
        clearbtn = Button(text='Clear')
        clearbtn.bind(on_release=self.clear_canvas)
        self.Answerbtn = Button(text=str(self.answer), pos=(0,100), size=(100,50))
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

#Google QuickDraw data -----------------------------------------------------------------------------------------------------------------
# All data from google

def GetData(x,ClassNum):
    array = []
    temparray = []
    for n in range(600):
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
CATEGORIES = ["Circle","Square","Triangle","Hexagon", "Star", "Diamond", "Smiley", "Zigzag", "Line", "Octagon", "Sun"]

for category in CATEGORIES:
    path = os.path.join(DATADIR, category + ".npy")
    ClassNum = [0,0,0,0,0,0,0,0,0,0,0]
    ClassNum[CATEGORIES.index(category)] = 1
    load = np.load(path)
    GetData(load,ClassNum)

random.shuffle(FullData)

x_TrainingData = []
y_TrainingData = []

for features, labels in FullData[:(500*11)]:
    x_TrainingData.append(features)
    y_TrainingData.append(labels)

x_TrainingData = np.array(x_TrainingData).reshape(-1, img_rows, img_cols, 1)
y_TrainingData = np.array(y_TrainingData)


x_TestData = []
y_TestData = []

for features, labels in FullData[(500*11):]:
    x_TestData.append(features)
    y_TestData.append(labels)

x_TestData = np.array(x_TestData).reshape(-1, img_rows, img_cols, 1)
y_TestData = np.array(y_TestData)

#TRAINING CODE  -----------------------------------------------------------------------------------------------------------------
batch_size = 50
num_classes = len(CATEGORIES)
epochs = 5
input_shape = (img_rows, img_cols,1)

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
model.fit(x_TrainingData, y_TrainingData,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_TestData, y_TestData)) #not sure what this does

#OUTPUT CODE -----------------------------------------------------------------------------------------------------------------
TrainingScore = model.evaluate(x_TrainingData, y_TrainingData, verbose=0)
print('Training loss:', TrainingScore[0])
print('Training accuracy:', TrainingScore[1])

TestScore = model.evaluate(x_TestData, y_TestData, verbose=0)
print('Test loss:', TestScore[0])
print('Test accuracy:', TestScore[1])

#Live Drawn Data -----------------------------------------------------------------------------------------------------------------
MyPaintApp().run()