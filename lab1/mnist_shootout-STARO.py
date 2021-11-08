import torch
from torch import optim
import torchvision
import matplotlib.pyplot as plt
import pt_deep
import data
import numpy as np

dataset_root = "./mnist"
mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=False)
mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=False)

x_train, y_train = mnist_train.data, mnist_train.targets
x_test, y_test = mnist_test.data, mnist_test.targets
x_train, x_test = x_train.double().div_(255.0), x_test.double().div_(255.0)

N = x_train.shape[0]
D = x_train.shape[1] * x_train.shape[2]
C = y_train.max().add_(1).item()

x_train_pt = torch.flatten(x_train, 1, 2)
y_train_pt_oh = torch.from_numpy(data.class_to_onehot(y_train))
x_test_pt = torch.flatten(x_test, 1, 2)
y_test_pt = y_test

x_train_np = x_train_pt.detach().numpy()
y_train_np = y_train.detach().numpy()
x_test_np = x_test_pt.detach().numpy()
y_test_np = y_test.detach().numpy()

def eval_and_print(model, x_train, y_train, x_test, y_test):
    """Doesn't save."""
    probs_train = model.eval(x_train)
    probs_test = model.eval(x_test)
    y_predicted_train = probs_train.argmax(axis=1)
    y_predicted_test = probs_test.argmax(axis=1)

    train_accuracy, train_pr, train_conf_m = (
        data.eval_perf_multi(y_predicted_train, y_train)
    )
    test_accuracy, test_pr, test_conf_m = (
        data.eval_perf_multi(y_predicted_test, y_test)
    )

    print("train")
    print("Accuracy: ", train_accuracy)
    print("Precision / Recall: ", train_pr)
    print("Confussion Matrix:\n", train_conf_m)
    print("test")
    print("Accuracy: ", test_accuracy)
    print("Precision / Recall: ", test_pr)
    print("Confussion Matrix:\n", test_conf_m)


def load_from_file(path):
    model_dict = torch.load(path)
    model = pt_deep.PTDeep([784, 10], activations=model_dict['activations'])
    model.weights = model_dict['weights']
    model.biases = model_dict['biases']
    model.loss_trace = model_dict['loss_trace']
    return model

def save_to_file(model, path):
    torch.save({
        'activations': model.activations,
        'weights': model.weights,
        'biases': model.biases,
        'loss_trace': model.loss_trace
    }, path)

def zad1_train():
    # [784, 10]
    model1 = pt_deep.PTDeep([784, 10], [pt_deep.my_softmax], param_lambda=0.05)
    model1.train_mb(x_train_pt, y_train_pt_oh, param_niter=2_000, param_delta=1e-4, print_frequency=100, early_stopping=True)
    torch.save({
        'activations': model1.activations,
        'weights': model1.weights,
        'biases': model1.biases,
        'loss_trace': model1.loss_trace
    }, './models/mnist_784_10_early_stopping_mb_Adam.pt')
    # [784, 100, 10]
    model2 = pt_deep.PTDeep([784, 100, 10], [torch.sigmoid, pt_deep.my_softmax])
    model2.train_mb(x_train_pt, y_train_pt_oh, param_niter=10_000, param_delta=1e-4, print_frequency=100, early_stopping=True)
    torch.save({
        'activations': model2.activations,
        'weights': model2.weights,
        'biases': model2.biases,
        'loss_trace': model2.loss_trace
    }, './models/mnist_784_100_10_early_stopping_mb_Adam.pt')


def zad1_eval(model1, model2):
    if isinstance(model1, str):
        model1_dict = torch.load('./models/mnist_784_10_early_stopping_mb_Adam.pt')
        model1 = pt_deep.PTDeep([784, 10], activations=model1_dict['activations'])
        model1.weights = model1_dict['weights']
        model1.biases = model1_dict['biases']
        model1.loss_trace = model1_dict['loss_trace']

    eval_and_print(model1, x_train_np, y_train_np, x_test_np, y_test_np)

    model2_dict = torch.load('./models/mnist_784_100_10_early_stopping_mb_Adam.pt')
    model2 = pt_deep.PTDeep([784, 10], activations=model2_dict['activations'])
    model2.weights = model2_dict['weights']
    model2.biases = model2_dict['biases']
    model2.loss_trace = model2_dict['loss_trace']

    eval_and_print(model2, x_train_np, y_train_np, x_test_np, y_test_np)

    # dobiti one koji najvise pridonose gubitku
    y_test_np_oh = data.class_to_onehot(y_test_np)
    probs = model2.eval(x_test_np)
    correct_class_probs = np.sum(probs * y_test_np_oh, axis=1)
    min_prob_indexes = np.argsort(correct_class_probs)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.imshow(x_test[min_prob_indexes[0]], cmap='gray')
    ax1.set_title(y_test_np[min_prob_indexes[0]])
    ax2.imshow(x_test[min_prob_indexes[1]], cmap='gray')
    ax2.set_title(y_test_np[min_prob_indexes[1]])
    ax3.imshow(x_test[min_prob_indexes[2]], cmap='gray')
    ax3.set_title(y_test_np[min_prob_indexes[2]])
    ax4.imshow(x_test[min_prob_indexes[3]], cmap='gray')
    ax4.set_title(y_test_np[min_prob_indexes[3]])
    plt.show()

def zad3():
    model1 = pt_deep.PTDeep([784, 10], [pt_deep.my_softmax], param_lambda=0.05)
    optimizer = optim.Adam(model1.parameters(), lr=1e-4, weight_decay=0.05)
    model1.train_mb(x_train_pt, y_train_pt_oh, param_niter=2_000, param_delta=1e-4, print_frequency=100,
                    early_stopping=True, optimizer=optimizer)

