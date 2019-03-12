import math
import time
import numpy as np
from collections import defaultdict


def dist(loc1, loc2):
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


class PowerLaw(object):
    def __init__(self, a=None, b=None):
        self.a = a
        self.b = b
        self.check_in_matrix = None
        self.visited_lids = {}
        self.poi_coos = None

    @staticmethod
    def compute_distance_distribution(check_in_matrix, poi_coos):
        distribution = defaultdict(int)
        for uid in range(check_in_matrix.shape[0]):
            lids = check_in_matrix[uid, :].nonzero()[0]
            for i in range(len(lids)):
                for j in range(i+1, len(lids)):
                    lid1, lid2 = lids[i], lids[j]
                    coo1, coo2 = poi_coos[lid1], poi_coos[lid2]
                    distance = int(dist(coo1, coo2))
                    distribution[distance] += 1
        total = 1.0 * sum(distribution.values())
        for distance in distribution:
            distribution[distance] /= total
        distribution = sorted(distribution.items(), key=lambda k: k[0])
        return zip(*distribution[1:])

    def fit_distance_distribution(self, check_in_matrix, poi_coos):
        self.check_in_matrix = check_in_matrix
        for uid in range(check_in_matrix.shape[0]):
            self.visited_lids[uid] = check_in_matrix[uid, :].nonzero()[0]

        ctime = time.time()
        print("Fitting distance distribution...", )
        self.poi_coos = poi_coos
        x, t = self.compute_distance_distribution(check_in_matrix, poi_coos)
        x = np.log10(x)
        t = np.log10(t)
        w0, w1 = np.random.random(), np.random.random()
        max_iterations = 2000
        lambda_w = 0.1
        alpha = 1e-5
        for iteration in range(max_iterations):
            Ew = 0.0
            d_w0, d_w1 = 0.0, 0.0
            for n in range(len(x)):
                d_w0 += (w0 + w1 * x[n] - t[n])
                d_w1 += (w0 + w1 * x[n] - t[n]) * x[n]
            w0 -= alpha * (d_w0 + lambda_w * w0)
            w1 -= alpha * (d_w1 + lambda_w * w1)
            for n in range(len(x)):
                Ew += 0.5 * (w0 + w1 * x[n] - t[n])**2
            Ew += 0.5 * lambda_w * (w0**2 + w1**2)
        print("Done. Elapsed time:", time.time() - ctime, "s")
        self.a, self.b = 10**w0, w1

    def pr_d(self, d):
        d = max(0.01, d)
        return self.a * (d ** self.b)

    def predict(self, uid, lj):
        lj = self.poi_coos[lj]
        return np.prod([self.pr_d(dist(self.poi_coos[li], lj)) for li in self.visited_lids[uid]])
