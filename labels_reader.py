import pandas as pd


class LabelsReader:
    def __init__(self, index_file):
        self.classes = None
        self.printing_classes = None
        self.data_index = None
        self.load_csv(index_file)

    def load_csv(self, index_file):
        new_names = ['image', 'healthy', 'multiple_diseases', 'rust', 'scab']
        self.data_index = pd.read_csv(index_file, names=new_names, skiprows=1, delimiter=',')
        self.classes = ['healthy', 'multiple_diseases', 'rust', 'scab']
        self.printing_classes = ['Saludable', 'Múltiples enfermedades', 'Enferma - Óxido', 'Enferma - Sarna']

    def get_label(self, image_name):
        image_name = image_name.split( ".", 1 )[ 0 ]
        row = self.data_index.loc[(self.data_index['image'] == image_name)]
        for i in range(len(self.classes)):
            class_value = row[self.classes[i]].item()
            if class_value == 1:
                break
        return i, self.classes[i]

    def get_img_printing_label(self, image_name):
        i, _ = self.get_label(image_name)
        return self.printing_classes[i]

    def get_printing_label(self, i):
        return self.printing_classes[i]
