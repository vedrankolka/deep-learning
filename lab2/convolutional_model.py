import torch
import nn
import math
from torch.utils import data
from torchvision.datasets import MNIST
from typing import Dict
import argparse
import numpy as np
from pathlib import Path
import skimage as ski
import skimage.io
import os
from sklearn import metrics
import matplotlib.pyplot as plt


class ConvollutionalModel(torch.nn.Module):

    def __init__(self):
        super(ConvollutionalModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5,
                               stride=1, padding=2, padding_mode='replicate')
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # self.relu1 = nn.ReLU
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5,
                               stride=1, padding=2, padding_mode='replicate')
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # self.relu2 = nn.ReLU
        self.flatten = torch.nn.Flatten(start_dim=1, end_dim=-1)
        self.fc1 = torch.nn.Linear(in_features=1568, out_features=512, bias=True)
        # self.relu3 = nn.ReLu
        self.fc2 = torch.nn.Linear(in_features=self.fc1.out_features, out_features=10, bias=True)

    def forward(self, x):
        s1 = self.conv1(x)
        a1 = self.pool1(s1)
        h1 = torch.relu(a1)
        s2 = self.conv2(h1)
        a2 = self.pool2(s2)
        h2 = torch.relu(a2)
        f1 = self.flatten(h2)
        s3 = self.fc1(f1)
        h3 = torch.relu(s3)
        s4 = self.fc2(h3)
        return s4


class MyDataset(data.Dataset):
    def __init__(self, x, y):
        super(MyDataset, self).__init__()
        assert len(x) == len(y), "Lengths of x and y must match."
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def train(train_x, train_y, valid_x, valid_y, model: torch.nn.Module,
          loss, optimizer, scheduler=None, config=dict(), callbacks=[]):

    batch_size = config.get('batch_size', 64)
    max_epochs = config.get('max_epochs', 5)
    verbose = config.get('verbose', False)
    print_frequency = config.get('print_frequency', 100)

    train_dataset = MyDataset(train_x, train_y)
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(max_epochs):
        print(f"Epoch {epoch}")
        train_loss = 0.0
        batch_count = 0
        for batch, batch_data in enumerate(train_dataloader):
            batch_count += 1
            train_x_batch, train_y_batch = batch_data
            logits = model.forward(train_x_batch)
            batch_loss = loss(logits, train_y_batch)
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += batch_loss

            if verbose and batch % print_frequency == 0:
                print(f"epoch: {epoch} batch: {batch} loss: {batch_loss}")

        if scheduler is not None:
            scheduler.step()

        train_loss /= batch_count
        if len(callbacks) > 0:
            with torch.no_grad():
                logits = model.forward(valid_x)
                validation_loss = loss(logits, valid_y)
                stop_training = [cb(epoch, -1, train_loss, validation_loss, model) for cb in callbacks]
                if any(stop_training):
                    break


def evaluate(name, x, yt, model, loss):
    print(f"\nRunning evaluation: {name}")
    with torch.no_grad():
        logits = model.forward(x)
        avg_loss = loss(logits, yt)
        yp = np.argmax(logits, axis=1)
        accuracy = metrics.accuracy_score(y_true=yt, y_pred=yp)
        print(f"{name} accuracy: {accuracy}")
        print(f"{name} avg loss: {avg_loss}")


def draw_conv_filters(epoch, step, weights, save_dir, layer_name):
    w = weights
    C = w.shape[1]
    num_filters = w.shape[0]
    k = w.shape[2]
    w = w.reshape(num_filters, C, k, k)
    w -= w.min()
    w /= w.max()
    border = 1
    cols = 8
    rows = math.ceil(num_filters / cols)
    width = cols * k + (cols-1) * border
    height = rows * k + (rows-1) * border
    #for i in range(C):
    for i in range(1):
        img = np.zeros([height, width])
        for j in range(num_filters):
            r = int(j / cols) * (k + border)
            c = int(j % cols) * (k + border)
            img[r:r+k, c:c+k] = w[j, i]
    filename = '%s_epoch_%02d_step_%06d_input_%03d.png' % (layer_name, epoch, step, i)
    # img = (img*255).astype(np.uint8)
    ski.io.imsave(os.path.join(save_dir, filename), img)


class EarlyStoppingCallback:

    def __init__(self, patience):
        self.best_validation_loss = None
        self.patience = patience
        self.count = 0

    def __call__(self, epoch, batch, train_loss, validation_loss, model=None):
        if validation_loss is None:
            return True

        validation_loss = validation_loss.detach().numpy()

        if self.best_validation_loss is None or validation_loss < self.best_validation_loss:
            self.best_validation_loss = validation_loss
            self.count = 0
            return False
        elif validation_loss > self.best_validation_loss:
            self.count += 1
            print(f"patience = {self.patience} count = {self.count} {validation_loss} > {self.best_validation_loss}"
                  f" returning {self.count >= self.patience}")
            return self.count >= self.patience


class SaveFiltersImageCallback:

    def __init__(self, save_dir):
        self.save_dir = save_dir

    def __call__(self, epoch, batch, train_loss, validation_loss, model=None):
        if batch >= 0:
            draw_conv_filters(epoch, batch, model.conv1.weight.detach().numpy(), self.save_dir, "conv1")


class LossTracerCallback:

    def __init__(self):
        self.train_loss_trace = []
        self.validation_loss_trace = []

    def __call__(self, epoch, batch, train_loss, validation_loss, model=None):
        self.validation_loss_trace.append(validation_loss.detach().numpy())
        self.train_loss_trace.append(train_loss.detach().numpy())


def dense_to_one_hot(y, class_count):
    return np.eye(class_count)[y]


def parse_arguments():
    default_data_dir = Path(__file__).parent / 'data'
    default_save_dir = Path(__file__).parent / 'out'
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch_size", help="batch size", type=int, default=64)
    parser.add_argument("-me", "--max_epochs", help="max epochs", type=int, default=10)
    parser.add_argument("-lr", "--learning_rate", help="learning rate", type=float, default=0.01)
    parser.add_argument("-g", "--gamma", help="gamma idk really", type=float, default=1 - 1e-4)
    parser.add_argument("-wd", "--weight_decay", help="L2 regularization factor", type=float, default=1e-2)
    parser.add_argument("-sd", "--save_dir", help="dir where save filters", type=str, default=default_save_dir)
    parser.add_argument("-dd", "--data_dir", help="mnist data directory", type=str, default=default_data_dir)
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.add_argument("-pf", "--print_frequency", help="not really frequency", type=int, default=100)
    parser.add_argument("-es", "--early_stopping", action="store_true", default=False)
    parser.add_argument("--patience", help="patience for early stopping", type=int, default=1)
    parser.add_argument("-nt", "--no_trace", help="don't trace loss?", action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_arguments()
    config = vars(args)
    # get the data
    ds_train, ds_test = MNIST(args.data_dir, train=True, download=False), MNIST(args.data_dir, train=False)
    train_x = ds_train.data.reshape([-1, 1, 28, 28]) / 255
    train_y = ds_train.targets
    train_x, valid_x = train_x[:55000], train_x[55000:]
    train_y, valid_y = train_y[:55000], train_y[55000:]
    test_x = ds_test.data.reshape([-1, 1, 28, 28]) / 255
    test_y = ds_test.targets
    train_mean = train_x.mean()
    train_x, valid_x, test_x = (x - train_mean for x in (train_x, valid_x, test_x))
    train_y_oh, valid_y_oh, test_y_oh = (dense_to_one_hot(y, 10) for y in (train_y, valid_y, test_y))
    # build the model and other stuff
    model = ConvollutionalModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    loss = torch.nn.CrossEntropyLoss()
    # construct callbacks
    callbacks = []
    if args.early_stopping is True:
        callbacks.append(EarlyStoppingCallback(args.patience))
    if args.save_dir is not None:
        callbacks.append(SaveFiltersImageCallback(args.save_dir))
    loss_tracer_callback = None
    if args.no_trace is not True:
        loss_tracer_callback = LossTracerCallback()
        callbacks.append(loss_tracer_callback)
    # train the model
    for k, v in config.items():
        print(f"{k}: {v}")

    train(train_x, train_y, valid_x, valid_y, model, loss, optimizer, scheduler, config, callbacks)
    evaluate("Test", test_x, test_y, model, loss)
    if loss_tracer_callback is not None:
        train_loss_trace = np.array(loss_tracer_callback.train_loss_trace)
        validation_loss_trace = np.array(loss_tracer_callback.validation_loss_trace)
        epochs = np.arange(0, len(train_loss_trace))
        plt.plot(epochs, train_loss_trace, label="train")
        plt.plot(epochs, validation_loss_trace, label="validation")
        plt.title = "Loss"
        plt.show()
