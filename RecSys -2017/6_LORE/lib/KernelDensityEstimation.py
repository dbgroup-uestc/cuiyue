import time
import math
import numpy as np

from collections import defaultdict


class KernelDensityEstimation(object):
    def __init__(self):
        self.poi_coos = None
        self.L = None
        self.bw = None

    def precompute_kernel_parameters(self, sparse_check_in_matrix, poi_coos):
        self.poi_coos = poi_coos

        ctime = time.time()
        print("Precomputing kernel parameters...", )

        training_locations = defaultdict(list)
        for uid in range(sparse_check_in_matrix.shape[0]):
            training_locations[uid] = [poi_coos[lid]
                                       for lid in sparse_check_in_matrix[uid].nonzero()[1].tolist()]

        L = training_locations

        bw = {}
        for u in L:
            if len(L[u]) > 1:
                std = np.std([coo for coo in L[u]], axis=0)
                bw[u] = 1.0 / (len(L[u])**(1.0/6)) * np.sqrt(0.5 * std.dot(std))

        print("Done. Elapsed time:", time.time() - ctime, "s")
        self.L = L
        self.bw = bw

    def K(self, x):
        return np.exp(-0.5 * np.sum(x * x, axis=1)) / (2 * math.pi)

    def predict(self, u, lj):
        if u in self.L and u in self.bw:
            lat_j, lng_j = self.poi_coos[lj]
            x = [np.array([lat_i - lat_j, lng_i - lng_j]) / self.bw[u] for lat_i, lng_i in self.L[u]]
            return sum(self.K(np.array(x))) / len(self.L[u]) / (self.bw[u] ** 2)
        return 1.0
