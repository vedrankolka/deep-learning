import numpy as np
import matplotlib.pyplot as plt
import data


def sigmoid(s):
    return 1 / (1 + np.exp(-s))


# stabilni softmax
def stable_softmax(x):
    max_by_row = np.max(x, axis=1).reshape((-1, 1))
    exp_x_shifted = np.exp(x - max_by_row)
    exp_sums = np.sum(exp_x_shifted, axis=1).reshape((-1, 1))
    probs = exp_x_shifted / exp_sums
    return probs


def one_hot_encode(Y, C):
    Y_encoded = []
    for y in Y:
        y_encoded = np.zeros(C)
        y_encoded[y] = 1
        Y_encoded.append(y_encoded)

    return Y_encoded


def logreg_train(X, Y_, param_niter=100_000, param_delta=0.1, print_frequency=500):
    C = int(np.max(Y_)) + 1
    N = len(X)
    n = len(X[0])
    W = np.random.randn(n, C)
    b = np.zeros((1, C)) # 1 x C
    Y_one_hot_encoded = one_hot_encode(Y_, C)

    for i in range(param_niter):
        scores = (X @ W) + b # N x C
        probs = stable_softmax(scores)
        # dijagnostički ispis
        if i % print_frequency == 0:
            logprobs = np.log(probs)  # N x C
            loss = - np.sum(logprobs * Y_one_hot_encoded)  # scalar
            print(f"iteration {i}: loss {loss}")

        # derivacije komponenata gubitka po mjerama
        dL_ds = probs - Y_one_hot_encoded  # N x C
        # gradijenti parametara
        grad_W = np.transpose((1 / N) * np.transpose(dL_ds) @ X) # C x D (ili D x C)
        grad_b = (1 / N) * np.sum(dL_ds, axis=0) # C x 1 (ili 1 x C)
        # poboljšani parametri
        W -= param_delta * grad_W
        b -= param_delta * grad_b

    return W, b


def logreg_classify(X, W, b):
    return stable_softmax(X @ W + b)


def logreg_decfun(X, W, b):
    """
    Decorator for  method logreg_classify
    """
    def classify(X):
        return logreg_classify(X, W, b).argmax(axis=1)

    return classify


def sample_gauss_2d(C, N):
    """
    Method for instantiating data

    Arguments
    C - number of classes
    N - number of samples in each class

    Return Values
    x - array of features NxC
    Y - array of labels   Nx1
    """
    x = []
    Y = []
    for i in range(0, C):
        G = data.Random2DGaussian()
        samples = G.get_sample(N)
        x.append(samples.tolist())
        Y.append([i for j in range(N)])
    x = np.reshape(x, (N * C, 2))
    Y = np.reshape(Y, (N * C, 1))

    return x, Y


if __name__ == "__main__":
    np.random.seed(100)
    # get the training dataset
    X, Y_ = sample_gauss_2d(3, 100)
    W, b = logreg_train(X, Y_)
    # evaluate the model on the training dataset
    probs = logreg_classify(X, W, b)
    print(probs.shape)
    # predicted classes
    Y = np.hstack([np.argmax(probs[i][:]) for i in range(probs.shape[0])])
    # reshaping for other methods purposes
    Y_ = np.hstack(Y_)
    accuracy, pr, M = data.eval_perf_multi(Y, Y_)
    print("Accuracy: ", accuracy)
    print("Precision / Recall: ", pr)
    print("Confussion Matrix: ", M)
    # graph the decision surface
    decfun = logreg_decfun(X, W, b)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
    # graph the data points
    data.graph_data(X, Y_, Y, special=[])
    # show the plot
    plt.show()
