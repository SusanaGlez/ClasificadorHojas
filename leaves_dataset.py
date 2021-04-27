import os
import torch
import PIL
from torch.utils.data import Dataset


class LeavesDataset(Dataset):
    def __init__(self, images_dir, images, label_reader, transform=None):
        self.images = images
        self.label_reader = label_reader
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        label_id, _ = self.label_reader.get_label(image_name)
        label = torch.tensor(label_id)
        image_path = os.path.join(self.images_dir, image_name)
        image = PIL.Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label
