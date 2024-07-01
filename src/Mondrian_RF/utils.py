import heapq
import numpy as np
import scipy.sparse
from copy import deepcopy
from numpy.linalg import norm
from sklearn.metrics import mean_squared_error

def train(X, y, M, lifetime_max, rng, delta=0):
    """
    Trains a Mondrian kernel with M trees and lifetime lifetime_max.
    :param X:               training inputs
    :param y:               training regression targets
    :param M:               number of Mondrian trees
    :param lifetime_max:    terminal lifetime
    :param rng:           random number generator
    :return: history, w_trees
    :history: list of tuples (birth_time, tree, feature, dim, loc), where feature is the index of feature being split
    :w_trees: list of weights for each tree
    """
    
    X = np.array(X)
    y = np.array(y)
    N, _ = np.shape(X)

    # subtract target means
    y_mean = np.mean(y)
    y_train = y - y_mean

    history = []

    # initialize sparse feature matrix
    indptr = range(0, M * N + 1, M)
    indices = list(range(M)) * N
    data = np.ones(N * M) / np.sqrt(M)
    Z_all = scipy.sparse.csr_matrix((data, indices, indptr), shape=(N, M))
    C = M

    # bounding box for all datapoints used to sample first cut in each tree
    feature_data = [np.array(range(N)) for _ in range(M)]
    lX = np.min(X, 0)
    uX = np.max(X, 0)

    # event = tuple (time, tree, feature, dim, loc), where feature is the index of feature being split
    events = []
    active_features = []
    active_features_in_tree = [[] for _ in range(M)]
    for m in range(M):
        cut_time, dim, loc = sample_cut(lX, uX, 0.0, rng)
        if cut_time < lifetime_max:
            heapq.heappush(events, (cut_time, m, m, dim, loc))
        active_features.append(m)
        active_features_in_tree[m].append(m)

    while len(events) > 0:
        (birth_time, m, c, dim, loc) = heapq.heappop(events)

        # construct new feature
        Xd = X[feature_data[c], dim]
        feature_l = (feature_data[c])[Xd <= loc]
        feature_r = (feature_data[c])[Xd  > loc]
        feature_data.append(feature_l)
        feature_data.append(feature_r)

        active_features.remove(c)
        active_features_in_tree[m].remove(c)
        active_features.append(C + 0)
        active_features.append(C + 1)
        active_features_in_tree[m].append(C + 0)
        active_features_in_tree[m].append(C + 1)

        # move datapoints from split feature to child features
        Z_all.indices[feature_l * M + m] = C + 0
        Z_all.indices[feature_r * M + m] = C + 1
        Z_all = scipy.sparse.csr_matrix((Z_all.data, Z_all.indices, Z_all.indptr), shape=(N, C + 2), copy=False)

        # sample the cut for each child
        feature_l_train = feature_l[feature_l < N]
        feature_r_train = feature_r[feature_r < N]
        lX_l = np.min(X[feature_l_train, :], axis=0)
        uX_l = np.max(X[feature_l_train, :], axis=0)
        cut_time_l, dim_l, loc_l = sample_cut(lX_l, uX_l, birth_time, rng)
        lX_r = np.min(X[feature_r_train, :], axis=0)
        uX_r = np.max(X[feature_r_train, :], axis=0)
        cut_time_r, dim_r, loc_r = sample_cut(lX_r, uX_r, birth_time, rng)

        # add new cuts to heap
        if cut_time_l < lifetime_max:
            heapq.heappush(events, (cut_time_l, m, C + 0, dim_l, loc_l))
        if cut_time_r < lifetime_max:
            heapq.heappush(events, (cut_time_r, m, C + 1, dim_r, loc_r))

        C += 2
        history.append((birth_time, m, c, dim, loc))

    w_trees = []
    for m in range(M):
        Z_train = Z_all[:, active_features_in_tree[m]]
        w_tree = np.linalg.solve(np.transpose(Z_train).dot(Z_train) + delta / M * np.identity(len(active_features_in_tree[m])),
                                np.transpose(Z_train).dot(y_train))
        w_trees.append(w_tree)

    return history, w_trees

def evaluate(y, X_test, M, history, w_trees):
    N = np.shape(X_test)[0]
    X_all = np.array(np.r_[X_test])
    history = deepcopy(history)

    # subtract target means
    y_mean = np.mean(y)

    # initialize sparse feature matrix
    indptr = range(0, M * N + 1, M)
    indices = list(range(M)) * N
    data = np.ones(N * M) / np.sqrt(M)
    Z_all = scipy.sparse.csr_matrix((data, indices, indptr), shape=(N, M))
    C = M

    trees_y_hat_test = np.zeros((N, M))
    
    feature_data = [np.array(range(N)) for _ in range(M)]
    active_features = []
    active_features_in_tree = [[] for _ in range(M)]
    for m in range(M):
        active_features.append(m)
        active_features_in_tree[m].append(m)

    while len(history) > 0:
        (birth_time, m, c, dim, loc) = history.pop(0)

        # construct new feature
        Xd = X_all[feature_data[c], dim]
        feature_l = (feature_data[c])[Xd <= loc]
        feature_r = (feature_data[c])[Xd  > loc]
        feature_data.append(feature_l)
        feature_data.append(feature_r)

        active_features.remove(c)
        active_features_in_tree[m].remove(c)
        active_features.append(C + 0)
        active_features.append(C + 1)
        active_features_in_tree[m].append(C + 0)
        active_features_in_tree[m].append(C + 1)

        # move datapoints from split feature to child features
        Z_all.indices[feature_l * M + m] = C + 0
        Z_all.indices[feature_r * M + m] = C + 1
        Z_all = scipy.sparse.csr_matrix((Z_all.data, Z_all.indices, Z_all.indptr), shape=(N, C + 2), copy=False)
        C += 2

    for m in range(M):
        Z_test = Z_all[:, active_features_in_tree[m]]
        w_tree = w_trees[m]
        trees_y_hat_test[:, m] = np.squeeze(Z_test.dot(w_tree))

    y_hat_test = y_mean + np.mean(trees_y_hat_test, 1)

    return y_hat_test


# SAMPLING
def sample_discrete(weights, rng):
    cumsums = np.cumsum(weights)
    cut = cumsums[-1] * rng.rand()
    return np.searchsorted(cumsums, cut)


def sample_cut(lX, uX, birth_time, rng):
    rate = np.sum(uX - lX)
    if rate > 0:
        E = rng.exponential(scale=1.0/rate)
        cut_time = birth_time + E
        dim = sample_discrete(uX - lX, rng=rng)
        loc = lX[dim] + (uX[dim] - lX[dim]) * rng.rand()
        return cut_time, dim, loc
    else:
        return np.Infinity, None, None


def two_one_norm(H):
    return np.sum(np.apply_along_axis(norm, 0, H)) / H.shape[1]

def evaluate_all_lifetimes(X, y, X_test, y_test, M, lifetime_max, delta=0, rng = np.random):
    """
    Sweeps through Mondrian kernels with all lifetime in [0, lifetime_max]. This can be used to (1) construct a Mondrian
    feature map with lifetime lifetime_max, to (2) find a suitable lifetime (inverse kernel width), or to (3) compare
    Mondrian kernel to Mondrian forest across lifetimes.
    :param X:                       training inputs
    :param y:                       training regression targets
    :param X_test:                  test inputs
    :param y_test:                  test regression targets
    :param M:                       number of Mondrian trees
    :param lifetime_max:            terminal lifetime
    :param delta:                   ridge regression regularization hyperparameter
    :param validation:              flag indicating whether a validation set should be created by halving the test set
    :param mondrian_kernel:         flag indicating whether mondrian kernel should be evaluated
    :param mondrian_forest:         flag indicating whether mondrian forest should be evaluated
    :param weights_from_lifetime:   lifetime at which forest and kernel learned weights should be saved
    :return: dictionary res containing all results
    """

    N, D = np.shape(X)
    N_test = np.shape(X_test)[0]
    X_all = np.array(np.r_[X, X_test])
    N_all = N + N_test

    # subtract target means
    y_mean = np.mean(y)
    y_train = y - y_mean

    # initialize sparse feature matrix
    indptr = range(0, M * N_all + 1, M)
    indices = list(range(M)) * N_all
    data = np.ones(N_all * M) / np.sqrt(M)
    Z_all = scipy.sparse.csr_matrix((data, indices, indptr), shape=(N_all, M))
    C = M

    # bounding box for all datapoints used to sample first cut in each tree
    feature_data = [np.array(range(N_all)) for _ in range(M)]
    lX = np.min(X, 0)
    uX = np.max(X, 0)

    # event = tuple (time, tree, feature, dim, loc), where feature is the index of feature being split
    events = []
    active_features = []
    active_features_in_tree = [[] for _ in range(M)]
    for m in range(M):
        cut_time, dim, loc = sample_cut(lX, uX, 0.0, rng)
        if cut_time < lifetime_max:
            heapq.heappush(events, (cut_time, m, m, dim, loc))
        active_features.append(m)
        active_features_in_tree[m].append(m)

    # iterate through birth times in increasing order
    list_times = []
    trees_y_hat_test = np.zeros((N_test, M))
    list_forest_error_test = []

    counter = 0

    while len(events) > 0:
        (birth_time, m, c, dim, loc) = heapq.heappop(events)

        # construct new feature
        Xd = X_all[feature_data[c], dim]
        feature_l = (feature_data[c])[Xd <= loc]
        feature_r = (feature_data[c])[Xd  > loc]
        feature_data.append(feature_l)
        feature_data.append(feature_r)

        active_features.remove(c)
        active_features_in_tree[m].remove(c)
        active_features.append(C + 0)
        active_features.append(C + 1)
        active_features_in_tree[m].append(C + 0)
        active_features_in_tree[m].append(C + 1)

        # move datapoints from split feature to child features
        Z_all.indices[feature_l * M + m] = C + 0
        Z_all.indices[feature_r * M + m] = C + 1
        Z_all = scipy.sparse.csr_matrix((Z_all.data, Z_all.indices, Z_all.indptr), shape=(N_all, C + 2), copy=False)

        # sample the cut for each child
        feature_l = feature_l[feature_l < N]
        feature_r = feature_r[feature_r < N]
        lX_l = np.min(X_all[feature_l, :], axis=0)
        uX_l = np.max(X_all[feature_l, :], axis=0)
        cut_time_l, dim_l, loc_l = sample_cut(lX_l, uX_l, birth_time, rng)
        lX_r = np.min(X_all[feature_r, :], axis=0)
        uX_r = np.max(X_all[feature_r, :], axis=0)
        cut_time_r, dim_r, loc_r = sample_cut(lX_r, uX_r, birth_time, rng)

        # add new cuts to heap
        if cut_time_l < lifetime_max:
            heapq.heappush(events, (cut_time_l, m, C + 0, dim_l, loc_l))
        if cut_time_r < lifetime_max:
            heapq.heappush(events, (cut_time_r, m, C + 1, dim_r, loc_r))

        C += 2
        Z_train = Z_all[:N, active_features_in_tree[m]]
        Z_test = Z_all[N:, active_features_in_tree[m]]
        if counter % 50 == 0:
            w_tree = np.linalg.solve(np.transpose(Z_train).dot(Z_train) + delta / M * np.identity(len(active_features_in_tree[m])),
                                np.transpose(Z_train).dot(y_train))
            trees_y_hat_test[:, m] = np.squeeze(Z_test.dot(w_tree))

            list_times.append(birth_time)
            y_hat_test = y_mean + np.mean(trees_y_hat_test, 1)
            list_forest_error_test.append(mean_squared_error(y_test, y_hat_test))
        
        counter += 1

    # this function returns a dictionary with all values of interest stored in it
    results = {'times': list_times, 'y_hat_test': y_hat_test, 'mse': list_forest_error_test}
    
    return results, y_hat_test