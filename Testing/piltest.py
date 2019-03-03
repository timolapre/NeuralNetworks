from random import random
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse, Line
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button

from kivy.uix.widget import Widget
from kivy.base import runTouchApp
from kivy.lang import Builder
from kivy.graphics import (Canvas, Translate, Fbo, ClearColor,
                           ClearBuffers, Scale)
from kivy.core.image import Image, Texture
from kivy.graphics import Fbo, Color, Rectangle
from kivy.config import Config
from kivy.core.window import Window
from kivy.uix.stencilview import StencilView
Config.set('graphics', 'width', '400')
Config.set('graphics', 'height', '400')

class MyPaintWidget(Widget):
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
        parent = Widget(size=(400,400))
        self.painter = MyPaintWidget(size=[400,400])
        clearbtn = Button(text='Clear')
        clearbtn.bind(on_release=self.clear_canvas)
        with self.painter.canvas:
            Rectangle(position=(200,200), size=(400,400))
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        return parent

    def clear_canvas(self, obj):
        self.painter.canvas.clear()
        with self.painter.canvas:
            Rectangle(position=(200,200), size=(400,400))

MyPaintApp().run()




'''from kivy.uix.widget import Widget
from kivy.base import runTouchApp
from kivy.lang import Builder
from kivy.graphics import (Canvas, Translate, Fbo, ClearColor, ClearBuffers, Scale)


kv=
BoxLayout:
    orientation: 'vertical'
    MyWidget:
        id: wgt
    BoxLayout:
        size_hint_y: .1
        orientation: 'horizontal'
        Label:
            text: 'enter scale:'
        TextInput:
            id: txtinpt
            text: '2.5'
        Button:
            text: 'Export with new scale'
            on_release: wgt.export_scaled_png('kvwgt.png', image_scale=float(txtinpt.text))

<MyWidget>:
    canvas:
        PushMatrix:
        Color:
            rgba: (0, 0, 1, .75)
        Ellipse:
            pos: (self.x + self.width // 5, self.y + self.height // 5)
            size: (self.width * 3 // 5, self.height * 3 // 5)
        Color:
            rgba: (1, 0, 0, .5)
        Rectangle:
            pos: (self.x + self.width // 4, self.y + self.height // 4)
            size: (self.width // 2, self.height // 2)
        Rotate:
            origin:
                self.center
            angle:
                45
    Button:
        text: 'useless child widget\\njust for demonstration'
        center: root.center
        size: (root.width // 2, root.height // 8)
        canvas:
            PopMatrix:



class MyWidget(Widget):

    def export_scaled_png(self, filename, image_scale=1):
        re_size = (self.width * image_scale, 
                self.height * image_scale)

        if self.parent is not None:
            canvas_parent_index = self.parent.canvas.indexof(self.canvas)
            if canvas_parent_index > -1:
                self.parent.canvas.remove(self.canvas)

        fbo = Fbo(size=re_size, with_stencilbuffer=True)

        with fbo:
            ClearColor(0, 0, 0, 0)
            ClearBuffers()
            Scale(image_scale, -image_scale, image_scale)
            Translate(-self.x, -self.y - self.height, 0)

        fbo.add(self.canvas)
        fbo.draw()
        fbo.texture.save(filename, flipped=False)
        fbo.remove(self.canvas)

        if self.parent is not None and canvas_parent_index > -1:
            self.parent.canvas.insert(canvas_parent_index, self.canvas)



runTouchApp(Builder.load_string(kv))'''