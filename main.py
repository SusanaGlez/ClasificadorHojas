import os
import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import transfer_learner as tl
import leaves_dataset as ld
import classifier_gui as gui
import labels_reader as lr
import configparser as cp


def main():
    config = cp.RawConfigParser()
    config.read('config.ini')

    index_file = config.get('general', 'index_file')
    images_dir = config.get('general', 'images_dir')
    path = config.get('general', 'model_path')

    labels_reader = lr.LabelsReader(index_file)

    images = os.listdir(images_dir)

    lists = []
    for i in range(len(labels_reader.classes)):
        lists.append([])

    for i in range(len(images)):
        if images[i].startswith('Train'):
            labelId, _ = labels_reader.get_label(images[i])
            lists[labelId].append(images[i])

    images_train = []
    images_val = []

    for i in range(len(lists)):
        images_train.extend(lists[i][:int((len(lists[i]) / 4) * 3)])
        images_val.extend(lists[i][int((len(lists[i]) / 4) * 3):])

    transfer_learner = create_transfer_learner(path, images_dir, images_train, images_val, labels_reader)

    set_training_parameters(transfer_learner)

    interface = gui.ClassifierGUI(images_dir, images_val, labels_reader, transfer_learner)

    interface.showWindow()


def set_training_parameters(transfer_learner):
    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(transfer_learner.model.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    transfer_learner.set_training_properties(criterion, optimizer_ft, exp_lr_scheduler)


def create_transfer_learner(path, images_dir, images_train, images_val, labels_reader):
    dataloaders, dataset_sizes = create_dataloaders(images_dir, images_train, images_val, labels_reader)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = models.resnet18(pretrained=True)

    #for param in model_ft.parameters():
    #    param.requires_grad = False

    num_ftrs = model_ft.fc.in_features

    model_ft.fc = nn.Linear(num_ftrs, len(labels_reader.classes))

    model_ft = model_ft.to(device)

    transfer_learner = tl.TransferLearner(path, model_ft, device, dataloaders, labels_reader.classes, dataset_sizes)

    return transfer_learner


def create_dataloaders(images_dir, images_train, images_val, labels_reader):
    image_datasets = {
        'train': ld.LeavesDataset(images_dir, images_train, labels_reader, tl.TransferLearner.data_transforms['train']),
        'val': ld.LeavesDataset(images_dir, images_val, labels_reader, tl.TransferLearner.data_transforms['val'])
    }

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    return dataloaders, dataset_sizes


if __name__ == '__main__':
    main()
