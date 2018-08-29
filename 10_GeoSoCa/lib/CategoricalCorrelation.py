import time
import numpy as np


class CategoricalCorrelation(object):
    def __init__(self):
        self.Y = None
        self.gamma = None

    def load_result(self, path):
        ctime = time.time()
        print("Loading result...",)
        self.Y = np.load(path + "Y.npy")
        self.gamma = 1.0 + 1.0 / np.mean(np.log(1.0 + self.Y[self.Y > 0]))
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def save_result(self, path):
        ctime = time.time()
        print("Saving result...",)
        np.save(path + "Y", self.Y)
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def compute_gamma(self, check_in_matrix, poi_cate_matrix):
        ctime = time.time()
        print("Precomputing categorical correlation parameter beta...", )

        B = check_in_matrix.dot(poi_cate_matrix)
        P = poi_cate_matrix.T

        Y = B.dot(P)
        gamma = 1.0 + 1.0 / np.mean(np.log(1.0 + Y[Y > 0]))

        print("Done. Elapsed time:", time.time() - ctime, "s")

        self.gamma = gamma
        self.Y = Y

    def predict(self, u, l):
        return 1.0 - (1.0 + self.Y[u, l]) ** (1 - self.gamma)
