import time
import math
import numpy as np

from collections import defaultdict


class AdaptiveKernelDensityEstimation(object):
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.poi_coos = None
        self.check_in_matrix = None

        self.R = None
        self.N = None
        self.H1, self.H2 = None, None
        self.h = None

    def precompute_kernel_parameters(self, check_in_matrix, poi_coos):
        self.poi_coos = poi_coos

        ctime = time.time()
        print("Precomputing kernel parameters...", )

        training_locations = defaultdict(list)
        for uid in range(check_in_matrix.shape[0]):
            training_locations[uid] = [[lid, np.array(poi_coos[lid])]
                                       for lid in check_in_matrix[uid].nonzero()[0].tolist()]

        N = {uid: np.sum(check_in_matrix[uid]) for uid in range(check_in_matrix.shape[0])}

        self.check_in_matrix = check_in_matrix
        self.N = N

        R = training_locations
        self.R = R

        H1, H2 = {}, {}
        for uid in R:
            mean_coo = np.sum([check_in_matrix[uid, lid] * coo
                               for lid, coo in R[uid]], axis=0) / N[uid]

            # The equation (5) in the paper is not correct.
            mean_coo_sq_diff = np.sum([check_in_matrix[uid, lid] * (coo - mean_coo) ** 2
                                       for lid, coo in R[uid]], axis=0) / N[uid]
            H1[uid], H2[uid] = 1.06 / (len(R[uid])**0.2) * np.sqrt(mean_coo_sq_diff)

        self.H1, self.H2 = H1, H2

        h = defaultdict(lambda: defaultdict(int))
        for uid in R:
            if not H1[uid] == 0 and not H2[uid] == 0:
                f_geo_vals = {li[0]: self.f_geo_with_fixed_bandwidth(uid, li, R) for li in R[uid]}
                g = np.prod(list(f_geo_vals.values())) ** (1.0 / len(R[uid]))
                for lid, f_geo_val in f_geo_vals.items():
                    h[uid][lid] = (g / f_geo_val) ** self.alpha
        self.h = h
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def f_geo_with_fixed_bandwidth(self, u, l, R):
        l, (lat, lng) = l
        return np.sum([self.check_in_matrix[u, li] * self.K_H(u, lat, lng, lat_i, lng_i)
                       for li, (lat_i, lng_i) in R[u]]) / self.N[u]

    def f_geo_with_local_bandwidth(self, u, l, R):
        l, (lat, lng) = l
        return np.sum([self.check_in_matrix[u, li] * self.K_Hh(u, lat, lng, lat_i, lng_i, li)
                       for li, (lat_i, lng_i) in R[u]]) / self.N[u]

    def K_H(self, u, lat, lng, lat_i, lng_i):
        return (1.0 / (2 * math.pi * self.H1[u] * self.H2[u]) *
                np.exp(-((lat - lat_i)**2 / (2 * self.H1[u]**2)) -
                       ((lng - lng_i)**2 / (2 * self.H2[u]**2))))

    def K_Hh(self, u, lat, lng, lat_i, lng_i, li):
        return (1.0 / (2 * math.pi * self.H1[u] * self.H2[u] * self.h[u][li]**2) *
                np.exp(-((lat - lat_i)**2 / (2 * self.H1[u]**2 * self.h[u][li]**2)) -
                       ((lng - lng_i)**2 / (2 * self.H2[u]**2 * self.h[u][li]**2))))

    def predict(self, u, l):
        if not self.H1[u] == 0 and not self.H2[u] == 0 and not sum(self.h[u].values()) == 0:
            l = [l, self.poi_coos[l]]
            return self.f_geo_with_local_bandwidth(u, l, self.R)
        return 1.0

