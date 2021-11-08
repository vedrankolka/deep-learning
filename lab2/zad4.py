import torch
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from pathlib import Path
import argparse
import convolutional_model as cm
import skimage as ski
import math


def shuffle_data(data_x, data_y):
  indices = np.arange(data_x.shape[0])
  np.random.shuffle(indices)
  shuffled_data_x = np.ascontiguousarray(data_x[indices])
  shuffled_data_y = np.ascontiguousarray(data_y[indices])
  return shuffled_data_x, shuffled_data_y

def unpickle(file):
  fo = open(file, 'rb')
  dict = pickle.load(fo, encoding='latin1')
  fo.close()
  return dict


def evaluate(Y, Y_):
    pr = []
    n = max(Y_)+1
    M = np.bincount(n * Y_ + Y, minlength=n*n).reshape(n, n)
    for i in range(n):
        tp_i = M[i, i]
        fn_i = np.sum(M[i, :]) - tp_i
        fp_i = np.sum(M[:, i]) - tp_i
        tn_i = np.sum(M) - fp_i - fn_i - tp_i
        recall_i = tp_i / (tp_i + fn_i)
        precision_i = tp_i / (tp_i + fp_i)
        pr.append( (recall_i, precision_i) )

    accuracy = np.trace(M) / np.sum(M)
    return accuracy, pr, M


def draw_conv_filters(epoch, step, weights, save_dir):
    w = weights.copy()
    num_filters = w.shape[0]
    num_channels = w.shape[1]
    k = w.shape[2]
    assert w.shape[3] == w.shape[2]
    w = w.transpose(2, 3, 1, 0)
    w -= w.min()
    w /= w.max()
    border = 1
    cols = 8
    rows = math.ceil(num_filters / cols)
    width = cols * k + (cols-1) * border
    height = rows * k + (rows-1) * border
    img = np.zeros([height, width, num_channels])
    for i in range(num_filters):
        r = int(i / cols) * (k + border)
        c = int(i % cols) * (k + border)
        img[r:r+k, c:c+k, :] = w[:, :, :, i]

    img = (img * 255).astype(np.uint8)
    filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
    ski.io.imsave(os.path.join(save_dir, filename), img)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(train_x, train_y, valid_x, valid_y, model: torch.nn.Module,
          loss, optimizer, scheduler=None, config=dict()):

    batch_size = config.get('batch_size', 64)
    max_epochs = config.get('max_epochs', 5)
    verbose = config.get('verbose', False)
    print_frequency = config.get('print_frequency', 100)

    train_dataset = cm.MyDataset(train_x, train_y)
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # stvari koje treba pratit
    lrs = []
    train_losses = []
    valid_losses = []
    avg_train_accuracies = []
    avg_valid_accuracies = []

    for epoch in range(max_epochs):
        print(f"Epoch {epoch}")
        for batch, batch_data in enumerate(train_dataloader):
            train_x_batch, train_y_batch = batch_data
            logits = model.forward(train_x_batch)
            batch_loss = loss(logits, train_y_batch)
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % print_frequency == 0:
                print(f"epoch: {epoch} batch: {batch} loss: {batch_loss}")
                weights = model[0].weight.detach().numpy()
                draw_conv_filters(epoch, batch_size*batch, weights, config['save_dir'])

        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            learning_rate = get_lr(optimizer)
            lrs.append(learning_rate)

            train_loss = 0.0
            batch_count = 0
            train_y_pred = []
            for batch_x, batch_y in train_dataloader:
                batch_count += 1
                batch_logits = model.forward(batch_x)
                batch_loss = loss(batch_logits, batch_y)
                train_loss += batch_loss
                train_y_pred.append(torch.argmax(batch_logits, axis=1))

            train_loss /= batch_count
            train_y_pred = torch.hstack(train_y_pred)
            train_accuracy, pr, train_conf_matrix = evaluate(train_y, train_y_pred)
            train_losses.append(train_loss)
            avg_train_accuracies.append(train_accuracy)

            valid_logits = model.forward(valid_x)
            valid_loss = loss(valid_logits, valid_y)
            valid_y_pred = torch.argmax(valid_logits, axis=1)
            valid_accuracy, pr, valid_conf_matrix = evaluate(valid_y, valid_y_pred)
            valid_losses.append(valid_loss)
            avg_valid_accuracies.append(valid_accuracy)

    return lrs, train_losses, avg_train_accuracies, valid_losses, avg_valid_accuracies


DATA_DIR = default_data_dir = Path(__file__).parent / 'data' / 'cifar-10-batches-py'

img_height = 32
img_width = 32
num_channels = 3
num_classes = 10

train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
train_y = []
for i in range(1, 6):
    subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
    train_x = np.vstack((train_x, subset['data']))
    train_y += subset['labels']
train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1)
train_y = np.array(train_y, dtype=np.long)

subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1).astype(np.float32)
test_y = np.array(subset['labels'], dtype=np.long)

valid_size = 5000
train_x, train_y = shuffle_data(train_x, train_y)
valid_x = train_x[:valid_size, ...]
valid_y = train_y[:valid_size, ...]
train_x = train_x[valid_size:, ...]
train_y = train_y[valid_size:, ...]
data_mean = train_x.mean((0, 1, 2))
data_std = train_x.std((0, 1, 2))

train_x = (train_x - data_mean) / data_std
valid_x = (valid_x - data_mean) / data_std
test_x = (test_x - data_mean) / data_std

train_x = torch.from_numpy(train_x.transpose(0, 3, 1, 2))
valid_x = torch.from_numpy(valid_x.transpose(0, 3, 1, 2))
test_x = torch.from_numpy(test_x.transpose(0, 3, 1, 2))

train_y = torch.from_numpy(train_y)
valid_y = torch.from_numpy(valid_y)
test_y = torch.from_numpy(test_y)
# =================== ARGS  stuff =================
args = cm.parse_arguments()
config = vars(args)
# =================== MODEL stuff =================
model = torch.nn.Sequential(
    torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2, padding_mode='replicate'),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=3, stride=2),
    torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2, padding_mode='replicate'),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=3, stride=2),
    torch.nn.Flatten(start_dim=1, end_dim=-1),
    torch.nn.Linear(in_features=1568, out_features=256, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=256, out_features=128, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=128, out_features=10, bias=True)
)

optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
loss = torch.nn.CrossEntropyLoss()

results = train(train_x, train_y, valid_x, valid_y, model, loss, optimizer, scheduler, config)
lrs, train_losses, avg_train_accuracies, valid_losses, avg_valid_accuracies = results
epochs = np.arange(0, len(lrs))

fig, (ax1, ax2, ax3) = plt.subplots(3)

ax1.plot(epochs, lrs, label='learning rate')
ax1.set_title('Learning rate')
ax1.legend()

ax2.plot(epochs, train_losses, label='train')
ax2.plot(epochs, valid_losses, label='validation')
ax2.set_title('Cross-entropy loss')
ax2.legend()

ax3.plot(epochs, avg_train_accuracies, label='train')
ax3.plot(epochs, avg_valid_accuracies, label='validation')
ax3.set_title('Average class accuracy')
ax3.legend()

plt.show()
