import numpy as np
import matplotlib.pyplot as plt
import data


param_niter = 100_000
param_delta = 0.0005
param_print_frequency = 500


def sigmoid(s):
    return 1 / (1 + np.exp(-s))


def binlogreg_train(X: np.ndarray, Y_: np.ndarray):
    """Train a binary logistic regression model on the given data.

    Arguments:
        X (np.ndarray): design matrix (NxD)
        Y_ (np.ndarray): labels (Nx1)

    Returns:
        w, b: weights and biases of the model
    """

    w = np.random.randn(2)
    b = 0

    for i in range(param_niter):
        # posterior probabilities c_1 (N x 1)
        probs = sigmoid(X @ w + b)
        # print
        if i % param_print_frequency == 0:
            # loss function
            loss = np.sum(-Y_ * np.log(probs) - (1 - Y_) * np.log(1 - probs))  # scalar
            print(f"iteration {i}: loss {loss}")
        # gradients
        diff = probs - Y_
        grad_w = np.sum(diff @ X) # D x 1
        grad_b = np.sum(diff)  # 1 x 1
        # new parameters
        w += -param_delta * grad_w
        b += -param_delta * grad_b

    return w, b


def binlogreg_classify(X, w, b):
    """Classify datapoints from X with a linear regresison model with parameters w and b.

    Arguments:
        X (np.ndarray): design matrix (NxD)
        w (np.ndarray): weights of the model (Dx1)
        b (float)     : bias of the model (1x1)

    Returns:
        probs (np.ndarray): P(Y == 1 | X)
    """
    return sigmoid(X @ w + b)


def binlogreg_decfun(w, b):
    return lambda X: binlogreg_classify(X, w, b)


if __name__ == "__main__":
    np.random.seed(100)
    # get the training dataset
    X, Y_ = data.sample_gauss_2d(2, 100)
    # train the model
    w, b = binlogreg_train(X, Y_)
    # evaluate the model on the training dataset
    probs = binlogreg_classify(X, w, b)
    Y = probs >= 0.5
    # report performance
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    AP = data.eval_AP(Y_[probs.argsort()])
    print(accuracy, recall, precision, AP)
    # now lets plot stuff
    fun = binlogreg_decfun(w, b)
    bounding_box = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(fun, bounding_box, offset=0.5)
    data.graph_data(X, Y_, Y)
    plt.show()
