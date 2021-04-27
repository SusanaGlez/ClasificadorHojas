import threading

import PySimpleGUI as sg
import io
import random
from PIL import Image

class ClassifierGUI:
    def __init__(self, images_dir, images, labels_reader, transfer_learner):
        self.images_dir = images_dir
        self.images = images
        self.transfer_learner = transfer_learner
        self.labels_reader = labels_reader
        self.transfer_learner.set_printer(self)
        self.window = None
        self.img1 = None
        self.img2 = None
        self.img3 = None

    def showWindow(self):

        self.img1, self.img2, self.img3 = self.get_random_images()

        model_column = [
            [sg.Button('Iniciar entrenamiento', key = 'start'), sg.Button('Cargar modelo', key = 'load-model')],
            [sg.Multiline('', size=(50, 30), font='courier 10', background_color='white', text_color='black',
                  key='output', disabled=True, autoscroll=True,)]
        ]

        validation_column = [
            [sg.Button('Cargar imagenes', key='load-img'), sg.Button('Clasificar', key='validate')],
            [sg.Image(key='img1', data=self.get_image_data(self.images_dir+self.img1))],
            [sg.Text('Clasificación: '+self.labels_reader.get_img_printing_label(self.img1), key='class1', size=(30, 1))],
            [sg.Text('Predicción: -', key='pred1', size=(30, 1))],
            [sg.Image(key='img2', data=self.get_image_data(self.images_dir+self.img2))],
            [sg.Text('Clasificación: '+self.labels_reader.get_img_printing_label(self.img2), key='class2', size=(30, 1))],
            [sg.Text('Predicción: -', key='pred2', size=(30, 1))],
            [sg.Image(key='img3', data=self.get_image_data(self.images_dir+self.img3))],
            [sg.Text('Clasificación: '+self.labels_reader.get_img_printing_label(self.img3), key='class3', size=(30, 1))],
            [sg.Text('Predicción: -', key='pred3', size=(30, 1))],
        ]

        layout = [
            [
                sg.Column(model_column),
                sg.VSeparator(),
                sg.Column(validation_column)
            ]
        ]

        self.window = sg.Window("Clasificador de salud de hojas de manzano", layout)

        while True:
                event, values = self.window.read()
                if event in (sg.WIN_CLOSED, 'Exit'):
                    break
                if event == 'load-img':
                    self.load_images()
                if event == 'start':
                    thread = threading.Thread(target=self.train, daemon=True)
                    thread.start()
                if event == 'validate':
                    thread = threading.Thread(target=self.validate, daemon=True)
                    thread.start()
                if event == 'load-model':
                    thread = threading.Thread(target=self.load, daemon=True)
                    thread.start()
        self.window.close()

    def get_random_images(self):
        i1 = random.randint(0, len(self.images) - 1)
        i2 = random.randint(0, len(self.images) - 1)
        i3 = random.randint(0, len(self.images) - 1)

        return self.images[i1], self.images[i2], self.images[i3]

    def get_image_data(self, filename):
        image = Image.open(filename)
        image.thumbnail((200, 200))
        bio = io.BytesIO()
        image.save(bio, format="PNG")
        return bio.getvalue()

    def load_images(self):
        self.img1, self.img2, self.img3 = self.get_random_images()
        self.window['img1'].Update(data=self.get_image_data(self.images_dir+self.img1))
        self.window['class1'].Update(value='Clasificación: ' + self.labels_reader.get_img_printing_label(self.img1))
        self.window['img2'].Update(data=self.get_image_data(self.images_dir+self.img2))
        self.window['class2'].Update(value='Clasificación: ' + self.labels_reader.get_img_printing_label(self.img2))
        self.window['img3'].Update(data=self.get_image_data(self.images_dir+self.img3))
        self.window['class3'].Update(value='Clasificación: ' + self.labels_reader.get_img_printing_label(self.img3))
        self.window.FindElement('pred1').Update(value='Predicción: -')
        self.window.FindElement('pred2').Update(value='Predicción: -')
        self.window.FindElement('pred3').Update(value='Predicción: -')

    def train(self):
        self.transfer_learner.train_model(15)
        self.transfer_learner.save_trained_model()

    def load(self):
        self.transfer_learner.load_model()

    def validate(self):
        pred1 = self.transfer_learner.predict(self.images_dir+self.img1)
        self.window.FindElement('pred1').Update(value='Predicción: '+self.labels_reader.get_printing_label(pred1))
        pred2 =self.transfer_learner.predict(self.images_dir+self.img2)
        self.window.FindElement('pred2').Update(value='Predicción: ' + self.labels_reader.get_printing_label(pred2))
        pred3 = self.transfer_learner.predict(self.images_dir+self.img3)
        self.window.FindElement('pred3').Update(value='Predicción: ' + self.labels_reader.get_printing_label(pred3))

    def update_train_output(self, output):
        self.window.FindElement('output').Update(output+'\n', append=True)
