import numpy as np
from numpy.linalg import LinAlgError


class MyLDA:
    def __init__(self, num_class=2, labels=None):
        if labels is None:
            self.labels = ['class0', 'class1']
        self.num_class = num_class
        self.w = None
        self.mu1_w_project = None
        self.mu0_w_project = None

    def fit(self, X, y):
        data=
        for class_i in range(self.num_class):


        X0 = X[y == 0]
        X1 = X[y == 1]

        mu0 = np.average(X0, axis=0).reshape(1, -1)
        mu1 = np.average(X1, axis=0).reshape(1, -1)

        sigma1 = np.dot((X1 - mu1).T, (X1 - mu1))
        sigma0 = np.dot((X0 - mu0).T, (X0 - mu0))
        Sw = sigma0 + sigma1

        try:
            Sw_inv = np.linalg.inv(Sw)
        except LinAlgError:
            print('Sw does not have an invertible matrix and has been replaced by a pseudo-invertible matrix.')
            Sw_inv = np.linalg.pinv(Sw)

        self.w = Sw_inv.dot((mu1 - mu0).reshape(-1, 1))
        self.mu1_w_project = mu1.dot(self.w)
        self.mu0_w_project = mu0.dot(self.w)

    def predict(self, X):
        return
