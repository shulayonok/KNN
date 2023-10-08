import numpy as np


# Необучаемый алгоритм, каждый раз пересчёт рассояния, поэтому самый медленный
class Classification_KNN:
    def __init__(self, k=5, pow=2):
        self.k = k
        self.pow = pow
        self.data = None
        self.target = None

    def fit(self, X, target):
        self.data = X
        self.target = target

    def prediction(self, X):
        return np.array([self.one_prediction(x) for x in X])

    def one_prediction(self, x_test):
        distances = np.array([self.distance(x_test, x_train) for x_train in self.data])
        k_nearests = np.argsort(distances)[:self.k]
        labels = np.array([self.target[i] for i in k_nearests])
        return self.argmaxT(labels)

    def distance(self, x_test, x_train):
        return np.sqrt(np.sum((x_test - x_train)**self.pow))

    def argmaxT(self, target):
        unique, count = np.unique(target, return_counts=True)
        return unique[np.argmax(count)]


class Regression_KNN:
    def __init__(self, k=5, pow=2):
        self.k = k
        self.pow = pow
        self.data = None
        self.target = None

    def fit(self, X, target):
        self.data = X
        self.target = target

    def prediction(self, X):
        return np.array([self.one_prediction(x) for x in X])

    def one_prediction(self, x_test):
        distances = np.array([self.distance(x_test, x_train) for x_train in self.data])
        k_nearests = np.argsort(distances)[:self.k]
        labels = np.array([self.target[i] for i in k_nearests])
        return self.argmaxT(labels)

    def distance(self, x_test, x_train):
        return np.sqrt(np.sum((x_test - x_train)**self.pow))

    def argmaxT(self, target):
        return np.sum(target) / len(target)
