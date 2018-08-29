import time
import numpy as np


class SocialCorrelation(object):
    def __init__(self):
        self.beta = None
        self.X = None

    def load_result(self, path):
        ctime = time.time()
        print("Loading result...",)
        self.X = np.load(path + "X.npy")
        self.beta = 1.0 + 1.0 / np.mean(np.log(1.0 + self.X[self.X > 0]))
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def save_result(self, path):
        ctime = time.time()
        print("Saving result...",)
        np.save(path + "X", self.X)
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def compute_beta(self, check_in_matrix, social_matrix):
        ctime = time.time()
        print("Precomputing social correlation parameter beta...", )

        S = social_matrix
        R = check_in_matrix

        X = S.dot(R)
        beta = 1.0 + 1.0 / np.mean(np.log(1.0 + X[X > 0]))

        print("Done. Elapsed time:", time.time() - ctime, "s")

        self.beta = beta
        self.X = X

    def predict(self, u, l):
        return 1.0 - (1.0 + self.X[u, l]) ** (1 - self.beta)
