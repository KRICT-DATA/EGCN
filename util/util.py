import pandas
from numpy import *


def zscore(X):
    if len(X.shape) == 1:
        means = mean(X)
        stds = std(X)

        for i in range(0, X.shape[0]):
            if stds == 0:
                X[i] = 0
            else:
                X[i] = (X[i] - means) / stds
    else:
        means = mean(X, axis=0)
        stds = std(X, axis=0)

        for i in range(0, X.shape[0]):
            for j in range(0, X.shape[1]):
                if stds[j] == 0:
                    X[i, j] = 0
                else:
                    X[i, j] = (X[i, j] - means[j]) / stds[j]

    return X


def replace_nan(X, replace_val):
    X[isnan(X)] = replace_val

    return X


def adj_mat_to_edges(adj_mat):
    edges = []

    for i in range(0, adj_mat.shape[0]):
        for j in range(0, adj_mat.shape[1]):
            if adj_mat[i, j] == 1:
                edges.append((i, j))

    return edges


def get_one_hot_vector(label, num_classes):
    one_hot_vector = zeros(num_classes)
    one_hot_vector[label-1] = 1

    return one_hot_vector
