from time import time
import copy
import numpy as np
import torch.nn
from torch.utils.data import Dataset, DataLoader
import os
import cv2 as cv
import torch
import torch.nn.functional as F
from torchvision import transforms, models
import wandb
# import warnings
# warnings.filterwarnings("error")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for PyTorch')
DATA_PATH = r"C:\Users\yuvfr\proj_university\swedish-leaf-classification\swedish-leaf-dataset"
MODEL_PATH = r"C:\Users\yuvfr\proj_university\swedish-leaf-classification\models"


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


def split_torch_datasets(datasets, test_perc):
    splits = [torch.utils.data.random_split(
        data, lengths=(
            int(len(data) * (1 - test_perc)), int(len(data) * test_perc) + 1))
        for data in datasets]

    train = torch.utils.data.ConcatDataset([split[0] for split in splits])
    test = torch.utils.data.ConcatDataset([split[1] for split in splits])
    return train, test


class AutoEncoder(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(3, 32, (3, 3), stride=(2, 2))
        self.conv2 = torch.nn.Conv2d(32, 64, (3, 3), stride=(2, 2))
        # self.conv3 = torch.nn.Conv2d(64, 128, (3, 3), stride=(2, 2))

        h_in, w_in = 224, 224
        for conv in [self.conv1, self.conv2]:
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
            np.floor(
                (dim_in + 2 * padding[axis] - dilation[axis] * (kernel_size[axis] - 1) - 1) / stride[0] + 1).astype(int)
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
        last_fc_in_features = base_model.fc.in_features
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
        self.resnet.eval()  # use all neurons w/o dropout/batchnorm etc..

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
        return sample / 255


class Trainer:

    def __init__(self, model, criterion, optimizer,
                 lr_scheduler=None, epochs=25, logger=wandb):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = epochs
        self.lr_scheduler = lr_scheduler
        self.logger = logger

    def step(self, inputs, labels):
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        return outputs, loss

    def train_epoch(self, dataloader, epoch_num):
        running_num_examples = 0
        running_loss = 0.0
        running_corrects = 0
        self.model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs, loss = self.step(inputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            # calculate statistics
            running_num_examples += inputs.shape[0]
            running_loss += loss.item() * inputs.shape[0]
            running_corrects += torch.sum(
                torch.argmax(outputs, dim=1) == torch.argmax(labels, dim=1)).item()

            # log to wandb
            self.logger.log({
                f"batch_Loss_train": running_loss / running_num_examples,
                f"batch_accuracy_train": running_corrects / running_num_examples
            }, step=epoch_num * len(dataloader.dataset) + running_num_examples)

        train_loss = running_loss / running_num_examples
        train_acc = running_corrects / running_num_examples
        return train_loss, train_acc

    def valid_epoch(self, dataloader):
        running_num_examples = 0
        running_loss = 0.0
        running_corrects = 0
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs, loss = self.step(inputs, labels)

                # calculate statistics
                running_num_examples += inputs.shape[0]
                running_loss += loss.item() * inputs.shape[0]
                running_corrects += torch.sum(
                    torch.argmax(outputs, dim=1) == torch.argmax(labels, dim=1)).item()

        val_loss = running_loss / running_num_examples
        val_acc = running_corrects / running_num_examples
        return val_loss, val_acc

    # def make_epoch(self, dataloader, epoch_num, is_train=True):
    #     running_num_examples = 0
    #     running_loss = 0.0
    #     running_corrects = 0
    #     phase = "Train" if is_train else "Valid"
    #     with torch.set_grad_enabled(is_train):
    #         for inputs, labels in dataloader:
    #             inputs = inputs.to(device)
    #             labels = labels.to(device)
    #             outputs = self.model(inputs)
    #             loss = self.criterion(outputs, labels)
    #             if is_train:
    #                 self.optimizer.zero_grad()
    #                 loss.backward()
    #                 self.optimizer.step()
    #                 self.lr_scheduler.step()
    #
    #             # calculate statistics
    #             running_num_examples += inputs.shape[0]
    #             running_loss += loss.item() * inputs.shape[0]
    #             running_corrects += torch.sum(
    #                 torch.argmax(outputs, dim=1) == torch.argmax(labels, dim=1)).item()
    #
    #             # log to wandb
    #             self.logger.log({
    #                 f"Loss/{phase}": running_loss / running_num_examples,
    #                 f"Accuracy/{phase}": running_corrects / running_num_examples
    #             }, step=epoch_num * len(dataloader.dataset) + running_num_examples)
    #
    #     epoch_loss = running_loss / running_num_examples
    #     epoch_acc = running_corrects/running_num_examples
    #
    #     return epoch_loss, epoch_acc

    # @staticmethod
    # def print_epoch_stats(epoch_loss, epoch_acc, train=True):
    #     print(f'{"train" if train else "valid"} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    def train(self, train_loader, val_loader):
        best_model_weights = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        best_epoch = -1

        self.logger.watch(self.model, self.criterion, log="all", log_freq=30)
        start = time()
        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)

            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc = self.valid_epoch(val_loader)
            metrics_dict = {"train_loss": train_loss, "train_acc": train_acc,
                            "val_loss": val_loss, "val_acc": val_acc}
            self.logger.log(metrics_dict)
            for k in metrics_dict:
                print(f'\t{k}:\t{metrics_dict[k]:.4f}')
            #
            # print(f'\tTrain Loss: {train_loss:.4f} Train Acc: {train_acc:.4f}\n'
            #       f'\tValid Loss: {val_loss:.4f} Valid Acc: {val_acc:.4f}')

            # deep copy the model
            if val_acc >= best_acc:
                best_epoch = epoch
                best_acc = val_acc
                best_model_weights = copy.deepcopy(self.model.state_dict())

        time_elapsed = time() - start
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best Epoch: {best_epoch}')
        print(f'Best val Acc: {best_acc:4f}')
        self.logger.log({"best_epoch": best_epoch, "best_val_acc": best_acc})

        # load best model weights
        self.model.load_state_dict(best_model_weights)


if __name__ == "__main__":

    config = {
        "leaf_classes": (9, 10),
        "epochs": 10,
        "learning_rate": 0.001,
        "momentum": 0.9,
        "criterion": torch.nn.CrossEntropyLoss(),
        "optimizer": "SGD",
        "batch_size": 4,
        "valid_test_perc": (0.15, 0.15),
        "fine_tune": True,   # fine_tune, freeze
        "model_name": "resnet_fine_tuning",
    }

    wandb.init(project="swedish-leaf-classification", group="transfer-learning",
               entity="yuvfried", job_type="train", name=config['model_name'],
               resume="allow", mode="online", config=config)

    wandb.config = config

    # data
    leaf1, leaf2 = wandb.config['leaf_classes']
    valid_perc, test_perc = wandb.config['valid_test_perc']
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    leaf1_ds = LeafDataset(os.path.join(DATA_PATH, f'leaf{leaf1}'), load_image, preprocess, label=0)
    leaf2_ds = LeafDataset(os.path.join(DATA_PATH, f'leaf{leaf2}'), load_image, preprocess, label=1)
    train, valid = split_torch_datasets([leaf1_ds, leaf2_ds], test_perc=valid_perc)
    wandb.log({"train_size": len(train), "valid_size":len(valid)})
    batch_size = wandb.config['batch_size']
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=len(valid), shuffle=True)

    # model
    fine_tune = wandb.config['fine_tune']
    model = TransferredResNet(fine_tune=fine_tune).to(device)
    if fine_tune:
        params_to_optimize = model.parameters()
    else:
        params_to_optimize = model.task_model.parameters()

    optimizer = torch.optim.SGD(
        params_to_optimize, lr=wandb.config["learning_rate"], momentum=wandb.config["momentum"])

    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    trainer = Trainer(model, wandb.config['criterion'], optimizer,
                      lr_scheduler=exp_lr_scheduler, epochs=config['epochs'])
    trainer.train(train_loader, valid_loader)

    # save
    model_abs_name = os.path.join(MODEL_PATH, f"{wandb.config['model_name']}.pt")
    torch.save(model.state_dict(), model_abs_name)
    artifact = wandb.Artifact(wandb.config['model_name'], type='model')
    artifact.add_file(model_abs_name)
    wandb.run.log_artifact(artifact)

    wandb.finish()
