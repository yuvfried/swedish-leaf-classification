from time import time
import copy
import numpy as np
import torch.nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2 as cv
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter

# import sys
# sys.setrecursionlimit(3000)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for PyTorch')
DATAPATH = r"C:\Users\yuvfr\proj_university\swedish-leaf-dataset"


def load_image(filename, height=224, width=224):
    img = cv.imread(filename, cv.IMREAD_COLOR)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (height, width))
    return img


class LeafDataset(Dataset):
    def __init__(self, main_dir, loader, transform, label):
        self.main_dir = main_dir
        self.img_list = os.listdir(main_dir)
        self.loader = loader
        self.transform = transform
        self.label = torch.zeros(size=(2,))
        self.label[label] = 1

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.img_list[idx])
        image = self.loader(img_loc)
        tensor_image = self.transform(image)
        return tensor_image, self.label


def split_leaf_datasets(datasets, test_perc):
    splits = [torch.utils.data.random_split(
        data, lengths=(
            int(len(data) * (1 - test_perc)), int(len(data) * test_perc) + 1))
        for data in datasets]

    train = torch.utils.data.ConcatDataset([split[0] for split in splits])
    valid = torch.utils.data.ConcatDataset([split[1] for split in splits])
    return train, valid


class AutoEncoder(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(3,32, (3,3), stride=(2,2))
        self.conv2 = torch.nn.Conv2d(32,64, (3,3), stride=(2,2))
        # self.conv3 = torch.nn.Conv2d(64, 128, (3, 3), stride=(2, 2))

        h_in, w_in = 224, 224
        for conv in [self.conv1,self.conv2]:
            h_in, w_in = self.conv2d_out_shape(
                h_in, w_in, conv.padding, conv.dilation, conv.kernel_size, conv.stride)

        # print(h_in, w_in)
        self.fc = torch.nn.Linear(h_in * w_in * self.conv2.out_channels, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # convert to flat
        print(x.shape)
        x = x.view(batch_size, -1)
        print(x.shape)
        return torch.sigmoid(self.fc(x))

    @staticmethod
    def conv2d_out_shape(h_in, w_in, padding, dilation, kernel_size, stride):
        out_dim = lambda dim_in, axis: \
            np.floor((dim_in + 2*padding[axis] - dilation[axis]*(kernel_size[axis]-1) - 1) / stride[0] + 1).astype(int)
        h_out = out_dim(h_in, 0)
        w_out = out_dim(w_in, 1)
        return h_out, w_out


class TransferredModel(torch.nn.Module):

    def __init__(self, base_model, task_model, fine_tune):
        super().__init__()
        self.base_model = base_model
        self.task_model = task_model
        if not fine_tune:
            self.freeze_model(self.base_model)
        self.fine_tune = fine_tune

    @staticmethod
    def freeze_model(model):
        for param in model.parameters():
            param.requires_grad = False
        model.eval()  # to be not optimized and use all neurons w/o dropout/batchnorm etc..

    def forward(self, x):
        with torch.set_grad_enabled(self.fine_tune):
            features = self.base_model(x)
        out = self.task_model(features)
        return out


class TransferredResNet(TransferredModel):

    def __init__(self, fine_tune=True):
        base_model = models.resnet18(pretrained=True)
        last_fc_in_features = model.fc.in_features
        self.__remove_last_fc_layer(base_model)
        task_model = torch.nn.Linear(last_fc_in_features, 2)
        super().__init__(base_model, task_model, fine_tune)

    @staticmethod
    def __remove_last_fc_layer(model):
        class IdentityLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x

        model.fc = IdentityLayer()


class ResNetTransfer(torch.nn.Module):

    def __init__(self, fine_tune=True):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True).to(device)
        resnet_fc_in_features = self.resnet.fc.in_features
        self.remove_resnet_fc_layer()
        self.fc = torch.nn.Linear(resnet_fc_in_features, 2).to(device)
        if not fine_tune:
            self.freeze_base_model()
        self.fine_tune = fine_tune

    def remove_resnet_fc_layer(self):
        class IdentityLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x

        self.resnet.fc = IdentityLayer()

    def freeze_base_model(self):
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.eval()  # to be not optimized and use all neurons w/o dropout/batchnorm etc..

    def forward(self, x):
        with torch.set_grad_enabled(self.fine_tune):
            features = self.resnet(x)
        out = self.fc(features)
        # out = torch.argmax(fc_out, 1).double().view(-1, 1)
        return out


class ZeroOneScale:

    def __init__(self):
        pass

    def __call__(self, sample, *args, **kwargs):
        return sample/255


class Trainer:

    def __init__(self, model, criterion, optimizer, lr_scheduler=None, num_epochs=25):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.lr_scheduler = lr_scheduler

    def make_epoch(self, dataloader, train=True):
        running_num_examples = 0
        running_loss = 0.0
        running_corrects = 0
        with torch.set_grad_enabled(train):
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.lr_scheduler.step()

                # statistics
                running_num_examples += inputs.shape[0]
                running_loss += loss.item() * inputs.shape[0]
                running_corrects += torch.sum(
                    torch.argmax(outputs, dim=1) == torch.argmax(labels, dim=1))
                # running_corrects += torch.sum(preds. == labels.data)

        epoch_loss = running_loss / running_num_examples
        epoch_acc = float(running_corrects) / running_num_examples
        self.print_epoch_stats(epoch_loss, epoch_acc, train=train)

        return epoch_loss, epoch_acc

    @staticmethod
    def print_epoch_stats(epoch_loss, epoch_acc, train=True):
        print(f'{"train" if train else "valid"} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    def train(self, train_loader, val_loader):
        writer = SummaryWriter()
        best_model_weights = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        start = time()
        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)

            self.model.train()
            train_loss, train_acc = self.make_epoch(train_loader, train=True)
            self.model.eval()
            val_loss, val_acc = self.make_epoch(val_loader, train=False)
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Loss/valid', val_loss, epoch)
            writer.add_scalar('Accuracy/valid', val_acc, epoch)

            # deep copy the model
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                best_model_weights = copy.deepcopy(self.model.state_dict())

        time_elapsed = time() - start
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best Epoch: {best_epoch}')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        self.model.load_state_dict(best_model_weights)


if __name__ == "__main__":


    preprocess = transforms.Compose([
        transforms.ToTensor(),
        #     ZeroOneScale(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    leaf9_ds = LeafDataset(os.path.join(DATAPATH, 'leaf9'), load_image, preprocess, label=0)
    leaf10_ds = LeafDataset(os.path.join(DATAPATH, 'leaf10'), load_image, preprocess, label=1)
    train, valid = split_leaf_datasets([leaf9_ds, leaf10_ds], 0.1)
    batch_size = 4
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=True)

    # resnet = models.resnet18(pretrained=True).to(device)
    # model = TransferredResNet(resnet, fine_tune=False)
    model = ResNetTransfer(fine_tune=False)


    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    trainer = Trainer(model, criterion, optimizer, lr_scheduler=exp_lr_scheduler)
    trainer.train(train_loader, valid_loader)
    torch.save(model.state_dict(), os.path.join(DATAPATH, "model.pt"))
