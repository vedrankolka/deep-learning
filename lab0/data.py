import numpy as np
import matplotlib.pyplot as plt


class Random2DGaussian:

    min = np.array([0, 0])
    max = np.array([10, 10])
    scalecov = 5

    def __init__(self):
        delta = self.max - self.min
        mean = self.min + delta * np.random.random_sample(2)
        eigval = (np.random.random_sample(2) * delta / self.scalecov) ** 2
        theta = np.random.random_sample() * 2 * np.pi
        R = np.array([[np.cos(theta), np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        cov = np.transpose(R) @ np.diag(eigval) @ R
        # definition of the method
        self.get_sample = lambda n: np.random.multivariate_normal(mean, cov, n)


def sample_gauss_2d(C, N):
    """Sample a C classification problem with N samples.

    Arguments:
        C (int): number of classes
        N (int): number of samples

    Returns:
        X, Y_: a classification problem
    """
    distributions = []
    Ys = []
    for i in range(C):
        distributions.append(Random2DGaussian())
        Ys.append(i)

    X = np.vstack([d.get_sample(N) for d in distributions])
    Y_ = np.hstack([[Y] * N for Y in Ys])

    return X, Y_


def eval_perf_binary(Y, Y_):
    """Evaluate the binary classification based on the true and predicted labels.

    Arguments:
        Y: predicted labels
        Y_: true labels

    Returns:
        accuracy, recall, precision
    """
    tp = sum(np.logical_and(Y_ == True, Y == True))
    fn = sum(np.logical_and(Y_ == True, Y == False))
    tn = sum(np.logical_and(Y_ == False, Y == False))
    fp = sum(np.logical_and(Y_ == False, Y == True))

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + fn + tn + fp)

    return accuracy, recall, precision


def eval_AP(ranked_labels):
    """I do not know what this should do.

    Arguments:
        ranked_labels (np.ndarray): sorted array of probabilities?

    Returns:
        average_precision (float): average precision? wtf?
    """
    n = len(ranked_labels)
    pos = np.sum(ranked_labels)
    neg = n - pos
    # first all are "declared positive"
    tp = pos
    tn = 0
    fn = 0
    fp = neg
    # summed precision
    sumprec = 0
    for x in ranked_labels:
        precision = tp / (tp + fp)
        if x:
            sumprec += precision

        tp -= x
        fn += x
        fp -= not x
        tn += not x

    return sumprec / pos


def graph_data(X, Y_, Y, special=[]):
    """Creates a scatter plot (visualize with plt.show)
    Arguments:
        X:       datapoints
        Y_:      groundtruth classification indices
        Y:       predicted class indices
        special: use this to emphasize some points
    Returns:
        None
    """
    # Y_ = np.reshape(Y_,(1,20))
    # Y = np.reshape(Y,(1,20))
    # colors of the datapoint markers
    palette = ([0.1, 0.2, 0.5], [0.9, 0.2, 0.5])
    colors = np.tile([0.0, 0.0, 0.0], (Y_.shape[0], 1))
    for i in range(len(palette)):
        colors[Y_ == i] = palette[i]
    # sizes of the datapoint markers
    sizes = np.repeat(20, len(Y_))
    sizes[special] = 40

    # draw the correctly classified datapoints
    good = (Y_ == Y)
    plt.scatter(X[good, 0], X[good, 1], c=colors[good], s=sizes[good], marker='o')

    # draw the incorrectly classified datapoints
    bad = (Y_ != Y)
    plt.scatter(X[bad, 0], X[bad, 1], c=colors[bad], s=sizes[bad], marker='s')


def my_dummy_decision(X):
    scores = X[:, 0] + X[:, 1] - 5
    return scores


def graph_surface(function, rect, offset=0.5, width=256, height=256):
    """Creates a surface plot (visualize with plt.show)
    Arguments:
      function: surface to be plotted
      rect:     function domain provided as:
                ([x_min,y_min], [x_max,y_max])
      offset:   the level plotted as a contour plot
    Returns:
      None
    """

    lsw = np.linspace(rect[0][1], rect[1][1], width)
    lsh = np.linspace(rect[0][0], rect[1][0], height)
    xx0, xx1 = np.meshgrid(lsh, lsw)
    grid = np.stack((xx0.flatten(), xx1.flatten()), axis=1)

    # get the values and reshape them
    values = function(grid).reshape((width, height))

    # fix the range and offset
    delta = offset if offset else 0
    maxval = max(np.max(values) - delta, - (np.min(values) - delta))

    # draw the surface and the offset
    plt.pcolormesh(xx0, xx1, values,
                   vmin=delta - maxval, vmax=delta + maxval)

    if offset != None:
        plt.contour(xx0, xx1, values, colors='black', levels=[offset])


def eval_perf_multi(Y, Y_):
    pr = []
    n = max(Y_) + 1
    M = np.bincount(n * Y_ + Y, minlength=n * n).reshape(n, n)
    for i in range(n):
        tp_i = M[i, i]
        fn_i = np.sum(M[i, :]) - tp_i
        fp_i = np.sum(M[:, i]) - tp_i
        tn_i = np.sum(M) - fp_i - fn_i - tp_i
        recall_i = tp_i / (tp_i + fn_i)
        precision_i = tp_i / (tp_i + fp_i)
        pr.append((recall_i, precision_i))

    accuracy = np.trace(M) / np.sum(M)

    return accuracy, pr, M


def sample_gmm_2d(ncomponents, nclasses, nsamples):
    # create the distributions and groundtruth labels
    Gs = []
    Ys = []
    for i in range(ncomponents):
        Gs.append(Random2DGaussian())
        Ys.append(np.random.randint(nclasses))

    # sample the dataset
    X = np.vstack([G.get_sample(nsamples) for G in Gs])
    Y_ = np.hstack([[Y] * nsamples for Y in Ys])

    return X, Y_


if __name__ == "__main__":
    np.random.seed(100)
    # get the training dataset
    X, Y_ = sample_gauss_2d(2, 100)
    # get the class predictions
    Y = my_dummy_decision(X) > 0.5
    # graph the data points
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(my_dummy_decision, bbox, offset=0)
    graph_data(X, Y_, Y)
    # show the results
    plt.show()
