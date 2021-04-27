import torch
import time
import copy
from torchvision import transforms
from PIL import Image


class TransferLearner:
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    def __init__(self, path, model, device, dataloaders, class_names, dataset_sizes):
        self.path = path
        self.model = model
        self.device = device
        self.dataloaders = dataloaders
        self.class_names = class_names
        self.dataset_sizes = dataset_sizes
        self.specialized_model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.printer = None

    def set_training_properties(self, criterion, optimizer, scheduler):
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def set_printer(self, printer):
        self.printer = printer

    def print(self, text):
        if self.printer:
            self.printer.update_train_output(text)
        else:
            print(text)

    def train_model(self, num_epochs=25):

        if (self.criterion is None) | (self.optimizer is None) | (self.scheduler is None):
            return

        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        self.specialized_model = None

        self.print('Entrenando modelo...')

        for epoch in range(num_epochs):
            self.print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            self.print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.print('Comenzando fase de entrenamiento...')
                    self.model.train()
                else:
                    self.print('Comenzando fase de validación...')
                    self.model.eval()

                epoch_loss = 0.0
                epoch_corrects = 0

                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    run_loss = loss.item() * inputs.size(0)
                    run_corrects = torch.sum(preds == labels.data)

                    epoch_loss += run_loss
                    epoch_corrects += run_corrects

                if phase == 'train':
                    self.scheduler.step()

                epoch_loss = epoch_loss / self.dataset_sizes[phase]
                epoch_acc = epoch_corrects.double() / self.dataset_sizes[phase]

                self.print('Epoch {} Error: {:.4f} Exactitud: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            self.print('')

        time_elapsed = time.time() - since
        self.print('Entrenamiento completado en {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        self.print('Mejor valor de exactitud: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        self.specialized_model = self.model

    def save_trained_model(self):
        self.print("Guardando modelo...")
        torch.save(self.specialized_model, self.path)
        self.print("Modelo guardado en: " + self.path)

    def load_model(self):
        self.print("Cargando modelo...")
        self.specialized_model = self.model
        self.specialized_model = torch.load(self.path)
        self.print("Modelo cargado de: " + self.path)

    def predict(self, image_file):
        img = Image.open(image_file)
        img_t = TransferLearner.data_transforms['val'](img)
        batch_t = torch.unsqueeze(img_t, 0)
        batch_t = batch_t.to(self.device)

        self.specialized_model.eval()

        out = self.specialized_model(batch_t)
        out = out.cpu()
        prediction = int(torch.max(out.data, 1)[1].numpy())
        return prediction
