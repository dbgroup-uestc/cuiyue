import cKDE
import time
import math
import numpy as np
from collections import defaultdict


def dist(loc1, loc2):# obsulute distance
    lat1, long1 = loc1[0], loc1[1]
    lat2, long2 = loc2[0], loc2[1]
    if abs(lat1 - lat2) < 1e-6 and abs(long1 - long2) < 1e-6:
        return 0.0
    degrees_to_radians = math.pi/180.0
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians
    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) +
           math.cos(phi1)*math.cos(phi2))
    arc = math.acos( cos )
    earth_radius = 6371
    return arc * earth_radius


class KernelDensityEstimation(object):
    def __init__(self):
        self.L = None
        self.poi_coos = None
        self.d, self.d_cpp = None, None
        self.h = None

    def precompute_kernel_parameters(self, sparse_check_in_matrix, poi_coos):# in sparse_check_in_matrix, data only stored in id form which of location can be mapped to poi_coos
        self.poi_coos = poi_coos

        ctime = time.time()
        print("Precomputing kernel parameters...", )

        training_locations = defaultdict(list)
        for uid in range(sparse_check_in_matrix.shape[0]):
            training_locations[uid] = [poi_coos[lid] for lid in sparse_check_in_matrix[uid].nonzero()[1].tolist()]
            # uid can be used to indicate order is because fortunatelty users had been enumerated form 0 to their total number in dataset

        L = training_locations

        d_cpp = {}
        d = defaultdict(list)
        for u in L:
            if len(L[u]) > 1:
                for i in range(len(L[u])):
                    for j in range(i+1, len(L[u])):# for the sake of no repeat
                        d[u].append(dist(L[u][i], L[u][j]))# d stands for distance
                d_cpp[u] = cKDE.new_doubleArray(len(d[u]))
                for i, v in enumerate(d[u]):
                    cKDE.set_doubleItem(d_cpp[u], i, v)
        h = {}
        for u in d:
            if not d[u]:
                h[u] = 1.06 * np.std(self.d[u]) * (1.0 / len(self.d[u])**0.2)
        print("Done. Elapsed time:", time.time() - ctime, "s")
        self.L = L
        self.h = h
        self.d = d
        self.d_cpp = d_cpp

    def K(self, x):
        return 1.0 / np.sqrt(2 * math.pi) * np.exp(-(x**2)/2)

    def f(self, dij, u):
        return 1.0 * np.sum([self.K((dij - d) / self.h[u])for d in self.d[u]]) / len(self.d[u]) / self.h[u]
        # a very smart way of computing

    def predict(self, u, lj):
        if u in self.h and not self.h[u] == 0:
            lat_j, lng_j = self.poi_coos[lj]
            return np.sum([cKDE.prob(lat_i, lng_i, lat_j, lng_j, self.d_cpp[u], len(self.d[u]), self.h[u])for lat_i, lng_i in self.L[u]]) / len(self.L[u])# actually can use function f in the current class directly
        return 1.0

